"""Explicit, isolated CUDA acceptance workers; never starts a formal experiment."""
import argparse
import json
from pathlib import Path
import sys


def compare_tree(left, right, *, tolerance, prefix="root"):
    import numpy as np
    import torch
    maximum = 0.0
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), prefix
        right = right.to(left.device)
        if left.is_floating_point():
            torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance, msg=lambda message: f"{prefix}: {message}")
            maximum = float((left - right).abs().max()) if left.numel() else 0.0
        else:
            assert torch.equal(left, right), prefix
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), prefix
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), prefix
        maximum = max((compare_tree(left[k], right[k], tolerance=tolerance, prefix=f"{prefix}.{k}")
                       for k in left), default=0.0)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), prefix
        maximum = max((compare_tree(a, b, tolerance=tolerance, prefix=f"{prefix}.{i}")
                       for i, (a, b) in enumerate(zip(left, right))), default=0.0)
    else:
        assert left == right, prefix
    return maximum


def probe(args):
    import torch
    from training.disparity_refiner import train as trainer
    from training.disparity_refiner.frozen_base import base_sha256, verify_frozen_base, set_training_mode
    from training.disparity_refiner.losses import supervised_iteration_loss
    from training.disparity_refiner.export_assets import predict

    config = json.loads(args.config.read_text())
    trainer._seed_everything(config["seed"])
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    model = trainer.InfiniDepth(model_path=config["model"]["checkpoint"]).to(device)
    model.attach_disparity_refiner(backend="official_flex", voxel_resolution=config["model"]["voxel_resolution"],
                                  max_disparity_span=config["model"].get("max_disparity_span"))
    historical = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert historical["refiner_config"]["backend"] == "official_flex"
    model.load_state_dict(historical["model"], strict=True)
    del historical
    model.refiner_only = True
    groups = trainer._parameter_groups(model, refiner_only=True)
    model.frozen_base_sha256 = base_sha256(model)
    optimizer = trainer._build_optimizer(groups, config["training"]["weight_decay"])
    trainer._configure_stage_learning_rates(optimizer, groups["dino"], config["training"]["stages"]["stage1"], 1)
    samples = trainer._load_samples(config, config["runs"][0], args.source,
                                   split="val", sample_ids=config["evaluation"]["sample_ids"][:1])
    sample = samples[0]
    hw = (config["model"]["height"], config["model"]["width"])
    chunk = config["model"]["query_chunk_size"]
    predictions = predict(model, sample, device, hw, chunk)
    assert all(torch.isfinite(value).all() for value in predictions.values())
    set_training_mode(model)
    output = model.forward_dense_refined(sample.image[None].to(device), query_hw=hw,
                                        num_refinement_steps=3, detach_base_from_refiner=True, chunk_size=chunk)
    loss, _ = supervised_iteration_loss(output.disparity_sequence, sample.target_disparity[None].to(device),
        sample.valid_mask[None].to(device), gradient_weight=config["training"]["gradient_weight"],
        gradient_scales=config["training"]["gradient_scales"])
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {name: parameter.grad.detach().cpu() for name, parameter in model.disparity_refiner.named_parameters()
                 if parameter.grad is not None}
    assert gradients and any(bool(value.abs().sum()) for value in gradients.values())
    assert all(torch.isfinite(value).all() for value in gradients.values())
    maximum_residual = max(float(value.detach().abs().max()) for value in output.bounded_residuals)
    assert maximum_residual <= 0.100001
    verify_frozen_base(model)
    optimizer.step()
    verify_frozen_base(model)
    record = dict(sample_id=sample.sample_id, predictions=predictions, gradients=gradients,
                  loss=loss.detach().cpu(), frozen_base_sha256=model.frozen_base_sha256,
                  updated_ssr={k: v.detach().cpu() for k, v in model.disparity_refiner.state_dict().items()})
    torch.save(record, args.output / "probe.pt")
    return dict(status="passed", source=str(args.source), sample_id=sample.sample_id,
                checkpoint=str(args.checkpoint), checkpoint_sha256=trainer._sha256(args.checkpoint),
                loss=float(loss.detach()), bounded_residual_abs_max=maximum_residual,
                base_frozen=True, gradient_tensors=len(gradients),
                peak_memory_bytes=torch.cuda.max_memory_allocated())


def compare_probes(args):
    import torch
    left, right = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    return dict(status="passed", tolerance=1e-6,
                max_absolute_difference=compare_tree(left, right, tolerance=1e-6))


def compare_resumes(args):
    import torch
    from training.disparity_refiner import train as trainer
    left, right = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    checks = {}
    for key in ("model", "optimizer", "config_sha256", "stage_step", "total_step", "frozen_base_sha256"):
        checks[key] = compare_tree(left[key], right[key], tolerance=1e-6, prefix=key)
    a, b = left["resume_state"], right["resume_state"]
    for key in ("rng", "rank_rng"):
        checks[key] = compare_tree(a.get(key), b.get(key), tolerance=0, prefix=key)
    assert b["stage"]["resume_sample_sequence_verified"]
    for key in ("best_step", "best_score", "best_evaluation", "expected_next_global_indices"):
        checks[key] = compare_tree(a["stage"][key], b["stage"][key], tolerance=1e-6, prefix=key)
    generator = trainer.ShuffledCycleSampler(0)
    assert trainer._restore_rng_state(right, generator)
    restored = trainer._capture_rng_state(generator)
    checks["restored_rng"] = compare_tree(b["rng"], restored, tolerance=0)
    expected = b["stage"]["expected_next_global_indices"]
    assert generator.sample_indices(generator.count, len(expected)) == expected
    return dict(status="passed", tolerance=1e-6, max_absolute_differences=checks,
                stage_step=right["stage_step"], next_sample_sequence_verified=True)


def roundtrip(args):
    import torch
    from training.disparity_refiner import train as trainer
    from training.disparity_refiner.frozen_base import base_sha256, verify_frozen_base
    config = json.loads(args.config.read_text())
    model = trainer.InfiniDepth(model_path=config["model"]["checkpoint"]).cuda()
    model.attach_disparity_refiner(backend="official_flex", voxel_resolution=config["model"]["voxel_resolution"],
                                  max_disparity_span=config["model"].get("max_disparity_span"))
    model.refiner_only = True
    groups = trainer._parameter_groups(model, refiner_only=True)
    model.frozen_base_sha256 = base_sha256(model)
    optimizer = trainer._build_optimizer(groups, config["training"]["weight_decay"])
    state = trainer._restore_checkpoint(args.checkpoint, model, optimizer, trainer._sha256(args.config))
    compare_tree(state["model"], model.state_dict(), tolerance=0)
    compare_tree(state["optimizer"], optimizer.state_dict(), tolerance=0)
    verify_frozen_base(model)
    sampler = trainer.ShuffledCycleSampler(0)
    assert trainer._restore_rng_state(state, sampler)
    compare_tree(state["resume_state"]["rng"], trainer._capture_rng_state(sampler), tolerance=0)
    expected = state["resume_state"]["stage"]["expected_next_global_indices"]
    assert sampler.sample_indices(sampler.count, len(expected)) == expected
    return dict(status="passed", checkpoint=str(args.checkpoint), stage_step=state["stage_step"],
                model_exact=True, optimizer_exact=True, rng_exact=True, sampler_exact=True,
                frozen_base_verified=True, next_global_sample_indices=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "compare-probes", "compare-resumes", "roundtrip"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, never an existing run")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inputs", nargs=2, type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    if args.output.is_symlink() or args.source in args.output.resolve().parents:
        raise ValueError("Acceptance output must not overlap source")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    result = {"probe": probe, "compare-probes": compare_probes, "compare-resumes": compare_resumes,
              "roundtrip": roundtrip}[args.mode](args)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

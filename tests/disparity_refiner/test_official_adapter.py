from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from InfiniDepth.model.disparity_refiner import voxelize_disparity, DisparitySparseRefiner
from InfiniDepth.model.official_disparity_adapter import official_sparse_inputs
from training.disparity_refiner import train as trainer
from training.disparity_refiner.frozen_base import base_sha256, set_training_mode, preserve_training_mode, verify_frozen_base


def test_official_coordinate_adapter_preserves_pixel_order_and_offsets():
    disparity = torch.linspace(-1, 1, 2 * 32 * 48).reshape(2, 32, 48)
    shell = voxelize_disparity(disparity)
    feats, coords, shape = official_sparse_inputs(shell, torch.zeros(2, 8, 2, 3))
    assert torch.equal(coords[:, (0, 3, 1, 2)], shell.coordinates)
    assert torch.equal(feats[:, 2], disparity.flatten())
    assert list(shape) == [2, 32, 48, shell.spatial_shape[0], 3]
    assert torch.equal(coords[:, 3].reshape_as(disparity) + shell.disparity_offsets[:, None, None], shell.logical_disparity_bins)
    with pytest.raises(ValueError, match="visual"):
        official_sparse_inputs(shell, torch.zeros(2, 8, 1, 3))


class TinyFrozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = torch.nn.Linear(1, 1)
        self.basic_encoder = torch.nn.BatchNorm1d(1)
        self.depth_implicit_head = torch.nn.Linear(1, 1)
        self.disparity_refiner = torch.nn.Sequential(torch.nn.BatchNorm1d(1), torch.nn.Linear(1, 1))
        self.refiner_only = True
        self.disparity_refiner_config = {"backend": "reference"}

    def forward_dense_refined(self, image, **kwargs):
        with torch.no_grad():
            base = self.depth_implicit_head(self.basic_encoder(self.pretrained(image.flatten().reshape(-1, 1))))
        delta = self.disparity_refiner(base).reshape_as(image[:, 0])
        base = base.reshape_as(delta)
        sequence = [base] + [base + k * delta for k in range(1, int(kwargs.get("num_refinement_steps", 3)) + 1)]
        return SimpleNamespace(disparity_sequence=sequence, voxel_statistics=[], raw_residuals=[], bounded_residuals=[])


def test_frozen_groups_buffers_and_eval_restoration_on_exception():
    model = TinyFrozen()
    groups = trainer._parameter_groups(model, refiner_only=True)
    optimizer = trainer._build_optimizer(groups, .01)
    assert len(optimizer.param_groups) == 1
    assert not groups["head"] and not groups["dino"]
    model.frozen_base_sha256 = base_sha256(model)
    set_training_mode(model)
    assert not model.training and model.disparity_refiner.training
    model.forward_dense_refined(torch.randn(1, 1, 4, 4)).disparity_sequence[3].sum().backward()
    optimizer.step()
    verify_frozen_base(model)

    @preserve_training_mode
    def broken_evaluation(model):
        model.eval()
        raise ValueError("evaluation failed")

    with pytest.raises(ValueError):
        broken_evaluation(model)
    assert not model.basic_encoder.training and model.disparity_refiner.training
    model.basic_encoder.running_mean.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        verify_frozen_base(model)


def test_checkpoint_between_full_evals_and_sampler_resume(tmp_path, monkeypatch):
    model = TinyFrozen()
    groups = trainer._parameter_groups(model, refiner_only=True)
    model.frozen_base_sha256 = base_sha256(model)
    optimizer = trainer._build_optimizer(groups, .01)
    generator = trainer.ShuffledCycleSampler(174)
    sample = SimpleNamespace(sample_id="x", image=torch.ones(1, 4, 4), target_disparity=torch.ones(4, 4), valid_mask=torch.ones(4, 4, dtype=torch.bool), structure_mask=None)
    stage = dict(min_steps=3, max_steps=3, eval_every=2, full_eval_every=3, checkpoint_every=1, learning_rates=dict(ssr=1e-5, head=0, dino=0))
    config = dict(training=dict(microbatch_size=1, global_batch_size=8, gradient_weight=.5, gradient_scales=2, gradient_clip_norm=1), model=dict(height=4, width=4, query_chunk_size=16), evaluation=dict(sample_ids=["x"]))
    monkeypatch.setattr(trainer, "_distributed_metadata", lambda *a: {"world_size": 1})
    monkeypatch.setattr(trainer, "_peak_cuda_memory", lambda *a: 0)
    saved_steps = []
    save = trainer._save_checkpoint
    def record_save(path, **kwargs):
        if path.name == "last.pt":
            saved_steps.append(kwargs["stage_step"])
        return save(path, **kwargs)
    monkeypatch.setattr(trainer, "_save_checkpoint", record_save)
    monkeypatch.setattr(trainer, "_pause_requested", lambda *a, **k: k["stage_step"] == 1)
    kwargs = dict(stage="stage1", stage_config=stage, model=model, optimizer=optimizer, parameter_groups=groups, samples=[sample], evaluation_samples=[sample], output=tmp_path, device=torch.device("cpu"), config=config, config_sha256="test", generator=generator, total_step=0)
    with pytest.raises(trainer.TrainingPaused):
        trainer.train_stage(**kwargs)
    checkpoint = tmp_path / "checkpoints/last.pt"
    state = trainer._restore_checkpoint(checkpoint, model, optimizer, "test")
    assert state["stage_step"] == 1 and state["evaluation"] == {}
    assert trainer._restore_rng_state(state, generator)
    monkeypatch.setattr(trainer, "_pause_requested", lambda *a, **k: False)
    report, step = trainer.train_stage(**dict(kwargs, total_step=1, initial_stage_step=1, initial_stage_state=trainer._checkpoint_stage_state(state)))
    assert step == 3 and report["resume_sample_sequence_verified"]
    assert saved_steps == [1, 2, 3]
    assert model.disparity_refiner.training and not model.basic_encoder.training
    verify_frozen_base(model)
    model.disparity_refiner_config = {"backend": "official_flex"}
    with pytest.raises(ValueError, match="backend"):
        trainer._restore_checkpoint(checkpoint, model, optimizer, "test")


def test_automation_immutable_configs(tmp_path):
    from experiment.exp6_4_official_ssr_disparity_adapter.automate import immutable_json
    path = tmp_path / "config.json"
    immutable_json(path, {"backend": "spconv"})
    immutable_json(path, {"backend": "spconv"})
    with pytest.raises(RuntimeError, match="revision"):
        immutable_json(path, {"backend": "official_flex"})
    assert json.loads(path.read_text()) == {"backend": "spconv"}


def test_final_step_pause_recomputes_full_evaluation_without_training():
    model = TinyFrozen()
    trainer._parameter_groups(model, refiner_only=True)
    model.frozen_base_sha256 = base_sha256(model)
    set_training_mode(model)
    sample = SimpleNamespace(sample_id="x", image=torch.ones(1, 4, 4), target_disparity=torch.ones(4, 4), valid_mask=torch.ones(4, 4, dtype=torch.bool), structure_mask=None)
    state, improved = trainer._refresh_final_evaluation(model, [sample], torch.device("cpu"), dict(height=4, width=4, query_chunk_size=16), {"best_score": float("inf"), "final_evaluation": None})
    assert improved and state["final_evaluation"]["sample_count"] == 1
    assert state["best_evaluation"] == state["final_evaluation"]
    assert model.disparity_refiner.training and not model.training
    verify_frozen_base(model)


def test_automation_test_gate_rejects_skips_and_failures(tmp_path):
    from experiment.exp6_4_official_ssr_disparity_adapter.automate import check_test_report
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite><testcase/></testsuite></testsuites>')
    check_test_report(path, 1)
    for tag in ("skipped", "failure", "error"):
        path.write_text(f'<testsuites><testsuite><testcase><{tag}/></testcase></testsuite></testsuites>')
        with pytest.raises(RuntimeError, match="gate"):
            check_test_report(path, 1)


def test_provenance_allows_immutable_non_git_source_snapshot(tmp_path):
    assert trainer._git_value(tmp_path, "rev-parse", "HEAD") == "unavailable"


def test_shared_gpu_gate_is_fixed_to_gpu3_and_ignores_external_pids():
    from experiment.exp6_4_official_ssr_disparity_adapter.automate import gpu_ready
    from experiment.schedule_exp3 import GpuState
    options = dict(gpu_index=3, minimum_free_memory_mib=22528,
                   maximum_utilization_percent=10, allow_shared_gpu=True)
    shared = GpuState(3, "gpu3", "4090", 46000, 0, (10, 11))
    assert gpu_ready(shared, options)
    assert not gpu_ready(GpuState(2, "gpu2", "4090", 46000, 0, ()), options)
    assert not gpu_ready(GpuState(3, "gpu3", "4090", 20000, 0, ()), options)
    assert not gpu_ready(GpuState(3, "gpu3", "4090", 46000, 11, ()), options)
    options["allow_shared_gpu"] = False
    assert not gpu_ready(shared, options)


def test_monitor_accepts_single_stage_and_explicit_run_paths(tmp_path, monkeypatch):
    from experiment import monitor_exp2
    experiment = tmp_path / "experiment/exp6_4"
    config = experiment / "arms/spconv/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"training": {"stages": {"stage1": {"max_steps": 20000}}}}))
    run = experiment / "runs/spconv"
    monkeypatch.setattr(monitor_exp2, "process_is_alive", lambda pid: True)
    monkeypatch.setattr(monitor_exp2, "process_identity_error", lambda *args, **kwargs: None)
    result = monitor_exp2.check_once(experiment, 123, 7200, config_path=config, run_path=run)
    assert result["health"] == "starting"
    assert (run / "monitor/status.json").is_file()


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["spconv", "official_flex"])
def test_cuda_zero_identity_backward_and_official_pixel_order(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(173)
    refiner = DisparitySparseRefiner(visual_dim=8, backend=backend).cuda().train()
    disparity = torch.rand(1, 32, 48, device="cuda")
    visual = torch.rand(1, 8, 2, 3, device="cuda")
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-5)
    first, _ = refiner(disparity, visual)
    assert torch.equal(first, torch.zeros_like(first))
    for _ in range(3):
        optimizer.zero_grad()
        raw, _ = refiner(disparity, visual)
        ((raw - .01) ** 2).mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in refiner.parameters())
        optimizer.step()
    if backend == "official_flex":
        network = refiner.unet.network
        shell = voxelize_disparity(disparity)
        features, coords, shape = official_sparse_inputs(shell, visual)
        order = torch.randperm(len(coords), device="cuda")
        with torch.no_grad():
            expected = network(features, coords, shape, visual)
            actual = network(features[order], coords[order], shape, visual)
        torch.testing.assert_close(actual, expected[order], atol=1e-5, rtol=1e-4)

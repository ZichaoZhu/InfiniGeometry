"""Reuse the frozen Exp6-3 geometry/Local protocol; no metric reimplementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.disparity_refiner.runtime_paths import evaluation_paths, protected_output


def load_protocol(config=None):
    source, masks, moge = evaluation_paths(config)
    sys.path.append(str(moge))
    spec = importlib.util.spec_from_file_location("exp6_3_protocol", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STEPS = (0, 1, 3, 5)
    return module, masks


def compare(experiment):
    import numpy as np
    from experiment.schedule_exp3 import atomic_json
    records = {}
    sequences = {}
    for backend in ("spconv", "official_flex"):
        run = experiment / "runs" / backend
        records[backend] = {r["sample_id"]: r for r in [json.loads(line) for line in (run / "evaluation/last/per_image.jsonl").read_text().splitlines()]}
        steps = [json.loads(line) for line in (run / "metrics/steps.jsonl").read_text().splitlines()]
        sequences[backend] = {r["total_step"]: r["global_sample_indices"] for r in steps}
    if sequences["spconv"] != sequences["official_flex"] or len(sequences["spconv"]) != 20000:
        raise RuntimeError("A/B global sampling sequences differ or are incomplete")
    a, b = records["spconv"], records["official_flex"]
    if set(a) != set(b) or len(a) != 100:
        raise RuntimeError("Paired evaluation requires the same Val100")
    result = {"format": "exp6-4-paired-analysis-v1", "sample_count": len(a), "sampling_sequence_verified": True, "metrics": {}}
    for metric in ("native_disparity_mae", "affine_depth_rel", "affine_point_rel", "local_point_rel", "local_point_delta_0.01", "local_depth_rel"):
        scenes = {}
        for sample_id in sorted(a):
            va, vb = a[sample_id]["metrics"]["k3"].get(metric), b[sample_id]["metrics"]["k3"].get(metric)
            if va is not None and vb is not None:
                scenes.setdefault(sample_id.split("_cam_")[0], []).append(float(vb) - float(va))
            ka, kb = a[sample_id]["metrics"]["k0"].get(metric), b[sample_id]["metrics"]["k0"].get(metric)
            if (ka is None) != (kb is None) or (ka is not None and not np.isclose(ka, kb, rtol=1e-6, atol=1e-7)):
                raise RuntimeError(f"A/B K0 differs for {sample_id}: {metric}")
        groups = list(scenes.values())
        rng = np.random.default_rng(173)
        boot = [np.mean([value for index in rng.integers(0, len(groups), len(groups)) for value in groups[index]]) for _ in range(2000)]
        differences = [value for group in groups for value in group]
        result["metrics"][metric] = {"mean_b_minus_a": float(np.mean(differences)), "scene_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(), "valid_images": len(differences)}
    primary = result["metrics"]
    result["candidate_improvement"] = primary["local_point_rel"]["mean_b_minus_a"] < 0 and all(primary[m]["mean_b_minus_a"] <= 0 for m in ("native_disparity_mae", "affine_point_rel"))
    result["claim_boundary"] = "Single-seed exploratory comparison; consult paired CI and common K0. No automatic default replacement."
    atomic_json(experiment / "metrics/paired_report.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return
    import torch
    from InfiniDepth.model import InfiniDepth
    from training.disparity_refiner.train import _load_samples, _refinement_monitor
    from training.disparity_refiner.export_assets import load_refiner_checkpoint
    from experiment.schedule_exp3 import atomic_json

    config = json.loads(args.config.read_text())
    if "readonly_source_root" in config["data"]:
        args.output = protected_output(config, args.output)
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Evaluation requires a new/empty output directory")
    protocol, masks = load_protocol(config)
    samples = _load_samples(config, config["runs"][0], ROOT, split="val", sample_ids=config["evaluation"]["full_sample_ids"])
    samples = [samples[i] for i in range(min(len(samples), args.limit or len(samples)))]
    assets = protocol.load_local_assets(masks, samples)
    torch.cuda.set_device(0)
    model = InfiniDepth(model_path=config["model"]["checkpoint"]).cuda()
    model.attach_disparity_refiner(backend=config["model"]["refiner_backend"], voxel_resolution=config["model"]["voxel_resolution"])
    checkpoint_sha = load_refiner_checkpoint(model, args.checkpoint)
    model.eval()
    records_path = args.output / "per_image.jsonl"
    provenance = dict(checkpoint_sha256=checkpoint_sha, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), limit=args.limit)
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise RuntimeError("Evaluation resume uses different inputs")
    atomic_json(provenance_path, provenance)
    records = protocol.load_existing(records_path)
    for sample in samples:
        if sample.sample_id in records:
            continue
        started = time.time()
        with torch.no_grad():
            output = model.forward_dense_refined(sample.image[None].cuda(), query_hw=(config["model"]["height"], config["model"]["width"]), num_refinement_steps=5, chunk_size=config["model"]["query_chunk_size"])
            _refinement_monitor(output)
            gt, rays = protocol.gt_for_sample(sample, torch.device("cuda:0"))
            low, high = sample.disparity_quantiles
            predictions, native = {}, {}
            for k in protocol.STEPS:
                normalized = output.disparity_sequence[k][0].float()
                native[k] = normalized
                raw = normalized * (high - low) + low
                depth = torch.where(torch.isfinite(raw) & (raw > 1e-6), raw.reciprocal(), torch.full_like(raw, float("nan")))
                predictions[k] = (depth, rays * depth[..., None])
            record = protocol.prediction_record(sample, predictions=predictions, gt=gt, method=config["model"]["refiner_backend"], source_root=Path(config["data"]["source_root"]), sharp_boundary=True, elapsed_seconds=time.time() - started, native_disparities=native, local_asset=assets[sample.sample_id])
        from experiment.monitor_exp2 import all_finite
        if not all_finite(protocol.jsonable(record["metrics"])):
            raise FloatingPointError(f"Non-finite evaluation metrics: {sample.sample_id}")
        records[sample.sample_id] = protocol.jsonable(record)
        temporary = records_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records.values()))
        temporary.replace(records_path)
        print(sample.sample_id, flush=True)
    summary = protocol.aggregate(list(records.values()), config["model"]["refiner_backend"])
    summary.update(status="completed", checkpoint_sha256=checkpoint_sha, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), refiner_parameters=sum(p.numel() for p in model.disparity_refiner.parameters()))
    atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()

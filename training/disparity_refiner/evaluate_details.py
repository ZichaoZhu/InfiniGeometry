from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from InfiniDepth.model import InfiniDepth
from training.disparity_refiner.data import (
    HypersimDisparityDataset,
    HypersimDisparitySample,
    ensure_within,
)
from training.disparity_refiner.evaluate_damping import _load_checkpoint
from training.disparity_refiner.losses import disparity_detail_metrics
from training.disparity_refiner.train import (
    EVALUATION_STEPS,
    _atomic_json,
    _load_samples,
    _selected_run,
    _sha256,
)


LOWER_IS_BETTER = {
    "full_mae": True,
    "multiscale_gradient_error": True,
    "boundary_f1": False,
    "edge_band_mae": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SSR detail metrics")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _relative_improvement(baseline: float, value: float, lower_is_better: bool) -> float:
    numerator = baseline - value if lower_is_better else value - baseline
    return numerator / max(abs(baseline), 1e-12)


def _metric_values(
    per_image: Mapping[str, object], key: str, metric: str
) -> list[float]:
    return [
        float(image[key][metric])
        for image in per_image.values()
        if image[key][metric] is not None
    ]


@torch.no_grad()
def evaluate_details(
    model: InfiniDepth,
    samples: Sequence[HypersimDisparitySample],
    *,
    device: torch.device,
    query_hw: tuple[int, int],
    chunk_size: int,
    metric_config: Mapping[str, object],
) -> dict[str, object]:
    model.eval()
    per_image: dict[str, object] = {}
    for sample in samples:
        output = model.forward_dense_refined(
            sample.image[None].to(device),
            query_hw=query_hw,
            num_refinement_steps=max(EVALUATION_STEPS),
            chunk_size=chunk_size,
        )
        target = sample.target_disparity.to(device)
        radial = sample.radial_depth.to(device)
        valid = sample.valid_mask.to(device)
        per_image[sample.sample_id] = {
            f"k{iteration}": disparity_detail_metrics(
                output.disparity_sequence[iteration][0],
                target,
                radial,
                valid,
                sample.disparity_quantiles,
                gradient_scales=int(metric_config["gradient_scales"]),
                boundary_threshold=float(metric_config["boundary_threshold"]),
                edge_band_radius=int(metric_config["edge_band_radius"]),
            )
            for iteration in EVALUATION_STEPS
        }

    aggregate = {
        f"k{iteration}": {
            metric: float(np.mean(_metric_values(per_image, f"k{iteration}", metric)))
            for metric in (*LOWER_IS_BETTER, "edge_pixels")
        }
        for iteration in EVALUATION_STEPS
    }
    metric_sample_counts = {
        metric: len(_metric_values(per_image, "k0", metric))
        for metric in LOWER_IS_BETTER
    }
    comparisons = {}
    for metric, lower_is_better in LOWER_IS_BETTER.items():
        baseline = float(aggregate["k0"][metric])
        comparisons[metric] = {}
        for iteration in EVALUATION_STEPS[1:]:
            key = f"k{iteration}"
            value = float(aggregate[key][metric])
            eligible = [
                image
                for image in per_image.values()
                if image["k0"][metric] is not None and image[key][metric] is not None
            ]
            comparisons[metric][key] = {
                "relative_improvement_over_k0": _relative_improvement(
                    baseline, value, lower_is_better
                ),
                "better_than_k0_count": sum(
                    image[key][metric] < image["k0"][metric]
                    if lower_is_better
                    else image[key][metric] > image["k0"][metric]
                    for image in eligible
                ),
                "evaluated_count": len(eligible),
            }
    return {
        "sample_count": len(samples),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "metric_sample_counts": metric_sample_counts,
        "per_image": per_image,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    experiment_config_path = experiment / "config.json"
    experiment_config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    parent_config_path = project_root / str(experiment_config["parent_config"])
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    safe_root = Path(str(parent_config["server"]["safe_root"]))
    ensure_within(experiment, safe_root, name="experiment")
    parent_config_path = ensure_within(
        parent_config_path, project_root, name="parent config"
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Detail evaluation requires CUDA")
    torch.cuda.set_device(device)

    sample_ids = [str(value) for value in parent_config["evaluation"]["full_sample_ids"]]
    samples = _load_samples(
        parent_config,
        _selected_run(parent_config, "main"),
        project_root,
        split=str(parent_config["evaluation"]["split"]),
        sample_ids=sample_ids,
    )
    if not isinstance(samples, HypersimDisparityDataset):
        raise TypeError("Exp3-2 expects the lazy Hypersim validation dataset")

    base_checkpoint = ensure_within(
        Path(str(parent_config["model"]["checkpoint"])),
        safe_root,
        name="base checkpoint",
    )
    model = InfiniDepth(model_path=str(base_checkpoint)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(parent_config["model"]["voxel_resolution"]),
        max_disparity_span=parent_config["model"].get("max_disparity_span"),
    )
    query_hw = (
        int(parent_config["model"]["height"]),
        int(parent_config["model"]["width"]),
    )
    chunk_size = int(parent_config["model"]["query_chunk_size"])
    report_path = experiment / "metrics" / "report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {
            "experiment_id": experiment_config["experiment_id"],
            "format": "infinidepth-disparity-detail-eval-v1",
            "status": "running",
            "config_sha256": _sha256(experiment_config_path),
            "parent_config": str(parent_config_path.relative_to(project_root)),
            "parent_config_sha256": _sha256(parent_config_path),
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "metric_config": experiment_config["metrics"],
            "checkpoints": {},
        }
    )

    for spec in experiment_config["checkpoints"]:
        checkpoint_id = str(spec["id"])
        if checkpoint_id in report["checkpoints"]:
            continue
        checkpoint_path = ensure_within(
            project_root / str(spec["path"]), safe_root, name=checkpoint_id
        )
        checkpoint = _load_checkpoint(model, checkpoint_path)
        started = time.time()
        print(f"Evaluating detail metrics for {checkpoint_id}", flush=True)
        value = evaluate_details(
            model,
            samples,
            device=device,
            query_hw=query_hw,
            chunk_size=chunk_size,
            metric_config=experiment_config["metrics"],
        )
        report["checkpoints"][checkpoint_id] = {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "stage": str(checkpoint["stage"]),
            "stage_step": int(checkpoint["stage_step"]),
            "total_step": int(checkpoint["total_step"]),
            "elapsed_seconds": time.time() - started,
            "evaluation": value,
        }
        del checkpoint
        _atomic_json(report_path, report)

    report["status"] = "completed"
    _atomic_json(report_path, report)


if __name__ == "__main__":
    main()

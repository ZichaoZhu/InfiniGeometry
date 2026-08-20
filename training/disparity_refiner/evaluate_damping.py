from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct

from InfiniDepth.model import InfiniDepth
from training.disparity_refiner.data import HypersimDisparityDataset, ensure_within
from training.disparity_refiner.train import (
    _atomic_json,
    _load_samples,
    _selected_run,
    _sha256,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference-time SSR damping")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _diagnostic(evaluation: Mapping[str, object]) -> dict[str, object]:
    aggregate = evaluation["aggregate"]
    per_image = evaluation["per_image"]
    k0 = float(aggregate["k0"]["full_mae"])
    k1 = float(aggregate["k1"]["full_mae"])
    k3 = float(aggregate["k3"]["full_mae"])
    k5 = float(aggregate["k5"]["full_mae"])
    return {
        "k3_relative_improvement_over_k0": (k0 - k3) / k0,
        "k3_relative_improvement_over_k1": (k1 - k3) / k1,
        "k5_relative_improvement_over_k3": (k3 - k5) / k3,
        "k3_better_than_k0_count": sum(
            value["k3"]["full_mae"] < value["k0"]["full_mae"]
            for value in per_image.values()
        ),
        "k3_better_than_k1_count": sum(
            value["k3"]["full_mae"] < value["k1"]["full_mae"]
            for value in per_image.values()
        ),
        "k5_better_than_k3_count": sum(
            value["k5"]["full_mae"] < value["k3"]["full_mae"]
            for value in per_image.values()
        ),
    }


def _load_checkpoint(model: InfiniDepth, path: Path) -> Mapping[str, object]:
    safe_numpy_types = [
        _reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_numpy_types):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    if checkpoint.get("format") != "infinidepth-disparity-refiner-v1":
        raise ValueError(f"Unsupported checkpoint: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    experiment_config = json.loads(
        (experiment / "config.json").read_text(encoding="utf-8")
    )
    parent_config_path = project_root / str(experiment_config["parent_config"])
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    safe_root = Path(str(parent_config["server"]["safe_root"]))
    ensure_within(experiment, safe_root, name="experiment")
    parent_config_path = ensure_within(
        parent_config_path, project_root, name="parent config"
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Damping evaluation requires CUDA")
    torch.cuda.set_device(device)

    run = _selected_run(parent_config, "main")
    sample_ids = [str(value) for value in parent_config["evaluation"]["full_sample_ids"]]
    samples = _load_samples(
        parent_config,
        run,
        project_root,
        split=str(parent_config["evaluation"]["split"]),
        sample_ids=sample_ids,
    )
    if not isinstance(samples, HypersimDisparityDataset):
        raise TypeError("Exp3-1 expects the lazy Hypersim validation dataset")

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
    alphas = [float(value) for value in experiment_config["residual_scales"]]
    report_path = experiment / "metrics" / "report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {
            "experiment_id": experiment_config["experiment_id"],
            "format": "infinidepth-disparity-damping-eval-v1",
            "status": "running",
            "parent_config": str(parent_config_path.relative_to(project_root)),
            "parent_config_sha256": _sha256(parent_config_path),
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "residual_scales": alphas,
            "checkpoints": {},
        }
    )

    for spec in experiment_config["checkpoints"]:
        checkpoint_id = str(spec["id"])
        checkpoint_path = ensure_within(
            project_root / str(spec["path"]), safe_root, name=checkpoint_id
        )
        results = report["checkpoints"].setdefault(
            checkpoint_id,
            {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "evaluations": {},
            },
        )
        missing = [alpha for alpha in alphas if str(alpha) not in results["evaluations"]]
        if not missing:
            continue
        checkpoint = _load_checkpoint(model, checkpoint_path)
        results.update(
            stage=str(checkpoint["stage"]),
            stage_step=int(checkpoint["stage_step"]),
            total_step=int(checkpoint["total_step"]),
        )
        del checkpoint
        for alpha in missing:
            started = time.time()
            print(f"Evaluating {checkpoint_id} residual_scale={alpha}", flush=True)
            value = evaluate(
                model,
                samples,
                range(len(samples)),
                device=device,
                query_hw=query_hw,
                chunk_size=chunk_size,
                residual_scale=alpha,
            )
            results["evaluations"][str(alpha)] = {
                **value,
                "diagnostic": _diagnostic(value),
                "elapsed_seconds": time.time() - started,
            }
            _atomic_json(report_path, report)

    report["status"] = "completed"
    _atomic_json(report_path, report)


if __name__ == "__main__":
    main()

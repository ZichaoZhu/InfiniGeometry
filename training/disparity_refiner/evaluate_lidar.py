from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.backup import verify_checkpoint_directory
from training.disparity_refiner.data import ensure_within
from training.disparity_refiner.train import _load_samples, _selected_run
from training.disparity_refiner.train_lidar import (
    CHECKPOINT_FORMAT,
    _evaluation_sample_ids,
    _local_mask_index,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an immutable Exp4 LiDAR refiner checkpoint on Val100"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="main_retry2_20260821")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_file(path: Path) -> Path:
    return path / "checkpoint.pt" if path.is_dir() else path


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Formal evaluation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _comparison(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    aggregate = evaluation["aggregate"]
    k0 = aggregate["k0"]
    k3 = aggregate["k3"]
    mae0 = float(k0["metric_disparity_mae_1_per_m"])
    mae3 = float(k3["metric_disparity_mae_1_per_m"])
    rel0 = k0.get("local_point_rel")
    rel3 = k3.get("local_point_rel")
    delta0 = k0.get("local_point_delta_0_01")
    delta3 = k3.get("local_point_delta_0_01")
    return {
        "k3_vs_k0_metric_disparity_mae_fraction": (mae3 - mae0) / max(abs(mae0), 1e-12),
        "k3_vs_k0_local_point_rel_fraction": (
            None
            if rel0 is None or rel3 is None
            else (float(rel3) - float(rel0)) / max(abs(float(rel0)), 1e-12)
        ),
        "k3_vs_k0_local_point_delta_0_01_difference": (
            None if delta0 is None or delta3 is None else float(delta3) - float(delta0)
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Formal Exp4 evaluation requires a CUDA device")
    project_root = Path(__file__).resolve().parents[2]
    config_path = ensure_within(args.config.resolve(), project_root, name="config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    safe_root = Path(str(config["server"]["safe_root"]))
    output_root = ensure_within(
        Path(str(config["server"]["output_root"])), safe_root, name="primary output root"
    )
    checkpoint = ensure_within(args.checkpoint, output_root, name="checkpoint")
    checkpoint_file = _checkpoint_file(checkpoint)
    checkpoint_dir = checkpoint_file.parent
    verify_checkpoint_directory(checkpoint_dir)
    output = ensure_within(args.output, output_root, name="evaluation output")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    base_checkpoint = ensure_within(
        Path(str(config["model"]["checkpoint"])), safe_root, name="base checkpoint"
    )
    base_sha = _sha256(base_checkpoint)
    if base_sha != str(config["model"]["checkpoint_sha256"]):
        raise ValueError("DepthSensor base checkpoint SHA-256 mismatch")
    checkpoint_value = torch.load(checkpoint_file, map_location=device, weights_only=True)
    if checkpoint_value.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported LiDAR refiner checkpoint format")
    if checkpoint_value.get("base_checkpoint_sha256") != base_sha:
        raise ValueError("LiDAR refiner checkpoint expects another DepthSensor base")
    if checkpoint_value.get("config_sha256") != _sha256(config_path):
        raise ValueError("LiDAR refiner checkpoint was produced by another config")

    run_config = _selected_run(config, args.run_id)
    sample_ids = _evaluation_sample_ids(config, project_root)
    samples = _load_samples(
        config,
        run_config,
        project_root,
        split=str(config["evaluation"]["split"]),
        sample_ids=sample_ids,
    )
    local_mask_root = ensure_within(
        Path(str(config["evaluation"]["local_points"]["mask_root"])),
        output_root,
        name="local mask root",
    )
    _, local_mask_entries = _local_mask_index(
        local_mask_root, [sample.sample_id for sample in samples], require_complete=True
    )
    model = InfiniDepth_DepthSensor(model_path=str(base_checkpoint)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(config["model"]["voxel_resolution"]),
        max_disparity_span=config["model"].get("max_disparity_span"),
    )
    assert model.disparity_refiner is not None
    model.disparity_refiner.load_state_dict(checkpoint_value["refiner"], strict=True)
    evaluation = evaluate(
        model,
        samples,
        device=device,
        config=config,
        local_mask_root=local_mask_root,
        local_mask_entries=local_mask_entries,
    )
    _write_json(
        output,
        {
            "checkpoint": {
                "path": str(checkpoint_dir),
                "sha256": _sha256(checkpoint_file),
                "step": int(checkpoint_value["step"]),
            },
            "comparison": _comparison(evaluation),
            "evaluation": evaluation,
            "format": "infinidepth-lidar-formal-evaluation-v1",
            "run_id": args.run_id,
        },
    )


if __name__ == "__main__":
    main()

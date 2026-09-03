from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.backup import verify_checkpoint_directory
from training.disparity_refiner.data import ensure_within
from training.disparity_refiner.train_lidar import CHECKPOINT_FORMAT
from training.disparity_refiner.waymo import (
    aggregate_sparse_metrics,
    list_tfrecords,
    load_waymo_sample,
    sparse_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot Exp4 evaluation on Waymo v1 TFRecords")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _comparison(
    per_image: Mapping[str, Mapping[str, Mapping[str, float]]],
    aggregate: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, higher_is_better in (
        ("metric_disparity_mae_1_per_m", False),
        ("radial_depth_abs_rel", False),
        ("point_delta_0_01", True),
    ):
        k0 = float(aggregate["k0"][name])
        k3 = float(aggregate["k3"][name])
        differences = np.asarray(
            [float(value["k3"][name]) - float(value["k0"][name]) for value in per_image.values()]
        )
        generator = np.random.default_rng(seed)
        bootstrap = differences[
            generator.integers(0, differences.size, size=(10000, differences.size))
        ].mean(axis=1)
        result[name] = {
            "k3_minus_k0": k3 - k0,
            "k3_vs_k0_fraction": (k3 - k0) / max(abs(k0), 1e-12),
            "paired_difference_95_percent_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "k3_better_count": int(
                sum(
                    (float(value["k3"][name]) > float(value["k0"][name]))
                    if higher_is_better
                    else (float(value["k3"][name]) < float(value["k0"][name]))
                    for value in per_image.values()
                )
            ),
        }
    return result


def _validate_config(config: Mapping[str, object]) -> None:
    if config.get("experiment_id") != "exp5_waymo_generalization":
        raise ValueError("Unexpected Exp5 config")
    if str(config["data"]["split"]) != "validation":
        raise ValueError("Exp5 formal evaluation is fixed to the Waymo validation split")
    if str(config["evaluation"]["camera"]) != "FRONT":
        raise ValueError("Initial Exp5 evaluation is fixed to the FRONT camera")
    if str(config["evaluation"]["lidar"]) != "TOP":
        raise ValueError("Initial Exp5 evaluation is fixed to the TOP LiDAR")
    if int(config["evaluation"]["prompt_stride"]) < 2:
        raise ValueError("Exp5 must hold out LiDAR points from the prompt")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Exp5 Waymo evaluation requires a CUDA device")
    project_root = Path(__file__).resolve().parents[2]
    config_path = ensure_within(args.config.resolve(), project_root, name="config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    config_sha = _sha256(config_path)
    safe_root = Path(str(config["server"]["safe_root"]))
    output_root = ensure_within(
        Path(str(config["server"]["output_root"])), safe_root, name="output root"
    )
    output = ensure_within(args.output, output_root, name="evaluation output")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    data_root = Path(str(config["data"]["root"])).resolve()
    allowed_data_root = Path(str(config["data"]["allowed_root"])).resolve()
    if data_root != allowed_data_root / str(config["data"]["split"]):
        raise PermissionError(f"Unexpected Waymo source: {data_root}")
    reader_root = ensure_within(
        Path(str(config["data"]["reader_root"])), safe_root, name="Waymo reader"
    )
    reader_commit = _git_commit(reader_root)
    if reader_commit != str(config["data"]["reader_commit"]):
        raise ValueError(f"Waymo reader commit mismatch: {reader_commit}")
    tfrecords = list_tfrecords(data_root, expected_count=int(config["data"]["expected_count"]))
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        tfrecords = tfrecords[: args.max_files]

    checkpoint = ensure_within(
        Path(str(config["checkpoint"]["path"])), safe_root, name="Exp4 checkpoint"
    )
    checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    verify_checkpoint_directory(checkpoint_dir)
    checkpoint_sha = _sha256(checkpoint_file)
    if checkpoint_sha != str(config["checkpoint"]["sha256"]):
        raise ValueError(f"Exp4 checkpoint SHA-256 mismatch: {checkpoint_sha}")
    base_checkpoint = ensure_within(
        Path(str(config["model"]["checkpoint"])), safe_root, name="DepthSensor checkpoint"
    )
    base_sha = _sha256(base_checkpoint)
    if base_sha != str(config["model"]["checkpoint_sha256"]):
        raise ValueError(f"DepthSensor checkpoint SHA-256 mismatch: {base_sha}")
    checkpoint_value = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if checkpoint_value.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported Exp4 checkpoint format")
    if checkpoint_value.get("base_checkpoint_sha256") != base_sha:
        raise ValueError("Exp4 checkpoint expects another DepthSensor base")

    progress_path = output / "progress.json"
    progress: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "format": "infinidepth-exp5-waymo-progress-v1",
        "per_image": {},
        "reader_commit": reader_commit,
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = (config_sha, checkpoint_sha, reader_commit)
        actual = (
            progress.get("config_sha256"),
            progress.get("checkpoint_sha256"),
            progress.get("reader_commit"),
        )
        if actual != expected:
            raise ValueError("Existing progress belongs to another evaluation")
    per_image: dict[str, dict[str, object]] = dict(progress["per_image"])
    unknown_sources = sorted(set(per_image) - {path.name for path in tfrecords})
    if unknown_sources:
        raise ValueError(f"Progress contains unexpected sources: {unknown_sources[:5]}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model_cfg = config["model"]
    evaluation_cfg = config["evaluation"]
    iterations = tuple(int(value) for value in evaluation_cfg["iterations"])
    model = InfiniDepth_DepthSensor(model_path=str(base_checkpoint)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(model_cfg["voxel_resolution"]),
        max_disparity_span=model_cfg.get("max_disparity_span"),
    )
    assert model.disparity_refiner is not None
    model.disparity_refiner.load_state_dict(checkpoint_value["refiner"], strict=True)
    model.eval()
    started = time.time()
    with torch.no_grad():
        for index, path in enumerate(tfrecords, start=1):
            source_key = path.name
            if source_key in per_image:
                continue
            sample = load_waymo_sample(
                path,
                reader_root=reader_root,
                frame_index=int(evaluation_cfg["frame_index"]),
                frame_search=int(evaluation_cfg["frame_search"]),
                output_hw=(int(model_cfg["height"]), int(model_cfg["width"])),
                prompt_stride=int(evaluation_cfg["prompt_stride"]),
                camera_name=str(evaluation_cfg["camera"]),
                lidar_name=str(evaluation_cfg["lidar"]),
                min_prompt_points=int(evaluation_cfg["min_prompt_points"]),
                min_evaluation_points=int(evaluation_cfg["min_evaluation_points"]),
            )
            output_value = model.forward_dense_refined(
                sample.image[None].to(device),
                prompt_disparity=sample.prompt_disparity[None].to(device),
                prompt_mask=sample.prompt_mask[None].to(device),
                query_hw=(int(model_cfg["height"]), int(model_cfg["width"])),
                num_refinement_steps=max(iterations),
                detach_base_from_refiner=True,
                chunk_size=int(model_cfg["query_chunk_size"]),
            )
            output_scale = float(output_value.reference_scale.reshape(-1)[0].item())
            if not np.isclose(output_scale, float(sample.reference_scale), rtol=1e-5):
                raise ValueError("DepthSensor prompt normalization scale mismatch")
            per_k = {
                f"k{iteration}": sparse_metrics(
                    output_value.disparity_sequence[iteration][0], sample
                )
                for iteration in iterations
            }
            per_image[source_key] = {
                "metadata": sample.metadata,
                **per_k,
            }
            progress["per_image"] = per_image
            progress["updated_at_unix"] = time.time()
            _atomic_json(progress_path, progress)
            print(
                f"[{index}/{len(tfrecords)}] {source_key} "
                f"K0={per_k['k0']['radial_depth_abs_rel']:.6f} "
                f"K3={per_k['k3']['radial_depth_abs_rel']:.6f}",
                flush=True,
            )

    metric_values = {
        key: {name: value for name, value in item.items() if name.startswith("k")}
        for key, item in per_image.items()
    }
    if len(metric_values) != len(tfrecords):
        raise RuntimeError(
            f"Evaluation is incomplete: {len(metric_values)}/{len(tfrecords)} sources"
        )
    aggregate = aggregate_sparse_metrics(metric_values, iterations)
    report = {
        "aggregate": aggregate,
        "checkpoint": {
            "base_sha256": base_sha,
            "path": str(checkpoint_dir),
            "sha256": checkpoint_sha,
            "step": int(checkpoint_value["step"]),
        },
        "comparison": _comparison(
            metric_values, aggregate, seed=int(config["seed"])
        ),
        "data": {
            "camera": str(evaluation_cfg["camera"]),
            "frame_index": int(evaluation_cfg["frame_index"]),
            "held_out_rule": f"TOP first-return range-image column modulo {evaluation_cfg['prompt_stride']} != 0",
            "lidar": str(evaluation_cfg["lidar"]),
            "prompt_rule": f"TOP first-return range-image column modulo {evaluation_cfg['prompt_stride']} == 0",
            "root": str(data_root),
            "sample_count": len(per_image),
            "source_file_count": len(tfrecords),
            "split": str(config["data"]["split"]),
        },
        "duration_seconds_this_invocation": time.time() - started,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "reader_commit": reader_commit,
        },
        "format": "infinidepth-exp5-waymo-evaluation-v1",
        "metric_semantics": {
            "aggregation": "macro average over one FRONT image per Waymo validation segment",
            "evaluation_support": "held-out TOP LiDAR first-return pixels not supplied as prompt",
            "metric_disparity_mae_1_per_m": "MAE of reciprocal camera-centric radial range",
            "point_delta_0_01": "fraction with radial 3D error below 0.01 times min predicted/GT radius",
            "radial_depth_abs_rel": "absolute relative error of camera-centric radial range",
            "radial_depth_rmse_m": "RMSE of camera-centric radial range in meters",
        },
        "per_image": per_image,
    }
    _atomic_json(report_path, report)
    print(json.dumps({"aggregate": aggregate, "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()

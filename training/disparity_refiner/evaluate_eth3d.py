from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.backup import verify_checkpoint_directory
from training.disparity_refiner.data import ensure_within
from training.disparity_refiner.eth3d import (
    load_eth3d_sample,
    make_eth3d_lidar_prompt,
    metric_values,
    read_input_manifest,
    sha256,
    stable_prompt_seed,
)
from training.disparity_refiner.lidar import load_segment_masks
from training.disparity_refiner.train_lidar import CHECKPOINT_FORMAT


PROGRESS_FORMAT = "infinidepth-exp5-eth3d-progress-v1"
REPORT_FORMAT = "infinidepth-exp5-eth3d-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot Exp4 LiDAR Refiner evaluation on ETH3D")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--no-local-masks", action="store_true")
    parser.add_argument("--allow-partial-local-masks", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path, project_root: Path) -> tuple[Path, dict[str, Any]]:
    path = ensure_within(path.expanduser().resolve(), project_root, name="config")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "exp5_ETH3D":
        raise ValueError("Unexpected ETH3D Exp5 config")
    if value["model"].get("variant") != "InfiniDepth_DepthSensor":
        raise ValueError("ETH3D evaluation requires InfiniDepth_DepthSensor")
    if value["data"].get("depth_representation") != "camera_optical_axis_z_depth":
        raise ValueError("ETH3D z-depth to radial-range conversion must be explicit")
    if list(value["evaluation"]["iterations"]) != [0, 1, 3, 5]:
        raise ValueError("ETH3D evaluation is fixed to K0/K1/K3/K5")
    return path, value


def _load_mask_index(
    root: Path, sample_ids: Sequence[str], *, require_complete: bool
) -> dict[str, Mapping[str, object]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "infinidepth-moge3-local-mask-manifest-v1":
        raise ValueError(f"Unsupported ETH3D local-mask manifest: {manifest_path}")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError(f"ETH3D local masks are not complete: {manifest_path}")
    entries = manifest.get("samples", {})
    missing = sorted(set(sample_ids) - set(entries))
    if missing:
        raise ValueError(f"ETH3D local masks lack samples: {missing[:5]}")
    result: dict[str, Mapping[str, object]] = {}
    for sample_id in sample_ids:
        entry = entries[sample_id]
        path = ensure_within(root / str(entry["path"]), root, name=f"{sample_id} mask")
        if sha256(path) != str(entry["sha256"]):
            raise ValueError(f"ETH3D local-mask SHA-256 mismatch: {sample_id}")
        result[sample_id] = entry
    return result


def _aggregate(records: Sequence[Mapping[str, Mapping[str, object]]], iterations: Sequence[int]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("No ETH3D metrics to aggregate")
    output: dict[str, dict[str, float]] = {}
    for iteration in iterations:
        key = f"k{iteration}"
        names = set().union(*(record[key] for record in records))
        values: dict[str, float] = {}
        for name in names:
            if name.startswith("_local_") or name in {
                "local_point_rel",
                "local_point_delta_0_01",
                "local_segment_count",
                "local_global_scale",
            }:
                continue
            finite = [
                float(record[key][name])
                for record in records
                if record[key].get(name) is not None and math.isfinite(float(record[key][name]))
            ]
            if finite:
                values[name] = float(np.mean(finite))
        segments = sum(float(record[key].get("local_segment_count", 0.0)) for record in records)
        if segments:
            values["local_segment_count"] = segments
            values["local_point_rel"] = sum(
                float(record[key].get("_local_point_rel_sum", 0.0)) for record in records
            ) / segments
            values["local_point_delta_0_01"] = sum(
                float(record[key].get("_local_point_delta_0_01_sum", 0.0)) for record in records
            ) / segments
        output[key] = values
    return output


def _aggregate_by_scene(
    per_image: Mapping[str, Mapping[str, object]], iterations: Sequence[int]
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    by_scene: dict[str, list[Mapping[str, Mapping[str, object]]]] = {}
    for record in per_image.values():
        by_scene.setdefault(str(record["scene"]), []).append(record["held_out"])
    scene_values = {scene: _aggregate(records, iterations) for scene, records in sorted(by_scene.items())}
    return scene_values, _aggregate(list(scene_values.values()), iterations)


def _comparison(
    per_image: Mapping[str, Mapping[str, object]], iterations: Sequence[int], seed: int
) -> dict[str, object]:
    if 0 not in iterations or 3 not in iterations:
        return {}
    names = (
        ("metric_disparity_mae_1_per_m", False),
        ("radial_depth_abs_rel", False),
        ("point_delta_0_01", True),
        ("local_point_rel", False),
        ("local_point_delta_0_01", True),
    )
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in per_image.values():
        grouped.setdefault(str(record["scene"]), []).append(record["held_out"])
    result: dict[str, object] = {}
    for name, higher_is_better in names:
        scene_differences = []
        image_better = 0
        applicable = 0
        for records in grouped.values():
            differences = []
            for record in records:
                k0, k3 = record["k0"].get(name), record["k3"].get(name)
                if k0 is None or k3 is None:
                    continue
                differences.append(float(k3) - float(k0))
                applicable += 1
                image_better += int(float(k3) > float(k0) if higher_is_better else float(k3) < float(k0))
            if differences:
                scene_differences.append(float(np.mean(differences)))
        if not scene_differences:
            continue
        values = np.asarray(scene_differences, dtype=np.float64)
        generator = np.random.default_rng(seed)
        bootstrap = values[generator.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        result[name] = {
            "paired_scene_macro_k3_minus_k0": float(values.mean()),
            "paired_scene_block_bootstrap_95_percent_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "applicable_image_count": applicable,
            "k3_better_image_count": image_better,
        }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("ETH3D evaluation requires a CUDA device")
    project_root = Path(__file__).resolve().parents[2]
    config_path, config = _load_config(args.config, project_root)
    safe_root = Path(str(config["server"]["safe_root"])).resolve()
    output_root = ensure_within(Path(str(config["server"]["output_root"])), safe_root, name="output root")
    output = ensure_within(args.output.expanduser().resolve(), output_root, name="evaluation output")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    input_path = ensure_within(
        (args.input_manifest or Path(str(config["data"]["input_manifest"]))).expanduser().resolve(),
        output_root,
        name="ETH3D input manifest",
    )
    inputs = read_input_manifest(input_path)
    samples = list(inputs["samples"])
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    elif len(samples) != int(config["data"]["expected_sample_count"]):
        raise ValueError(f"ETH3D manifest has {len(samples)} samples, expected full evaluation")
    sample_ids = [str(sample["id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("ETH3D manifest has duplicate sample IDs")

    checkpoint_dir = ensure_within(Path(str(config["checkpoint"]["path"])), safe_root, name="Exp4 checkpoint")
    verify_checkpoint_directory(checkpoint_dir)
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    checkpoint_sha = sha256(checkpoint_file)
    if checkpoint_sha != str(config["checkpoint"]["sha256"]):
        raise ValueError("Exp4 checkpoint SHA-256 mismatch")
    base_checkpoint = ensure_within(Path(str(config["model"]["checkpoint"])), safe_root, name="DepthSensor checkpoint")
    base_sha = sha256(base_checkpoint)
    if base_sha != str(config["model"]["checkpoint_sha256"]):
        raise ValueError("DepthSensor checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("base_checkpoint_sha256") != base_sha:
        raise ValueError("Incompatible Exp4 checkpoint")

    local_root: Path | None = None
    local_entries: dict[str, Mapping[str, object]] = {}
    if not args.no_local_masks:
        if args.allow_partial_local_masks and args.max_samples is None:
            raise ValueError("Partial ETH3D local masks are only allowed for --max-samples smoke tests")
        local_root = ensure_within(
            Path(str(config["evaluation"]["local_points"]["mask_root"])),
            output_root,
            name="ETH3D local-mask root",
        )
        local_entries = _load_mask_index(
            local_root, sample_ids, require_complete=not args.allow_partial_local_masks
        )

    config_sha = sha256(config_path)
    input_sha = sha256(input_path)
    progress_path = output / "progress.json"
    progress: dict[str, Any] = {
        "format": PROGRESS_FORMAT,
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "input_manifest_sha256": input_sha,
        "local_masks": None if local_root is None else str(local_root),
        "per_image": {},
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = (config_sha, checkpoint_sha, input_sha, None if local_root is None else str(local_root))
        actual = (
            progress.get("config_sha256"),
            progress.get("checkpoint_sha256"),
            progress.get("input_manifest_sha256"),
            progress.get("local_masks"),
        )
        if actual != expected or progress.get("format") != PROGRESS_FORMAT:
            raise ValueError("Existing ETH3D progress belongs to another evaluation")
    per_image: dict[str, Any] = dict(progress["per_image"])
    unexpected = sorted(set(per_image) - set(sample_ids))
    if unexpected:
        raise ValueError(f"ETH3D progress contains unexpected samples: {unexpected[:3]}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = InfiniDepth_DepthSensor(model_path=str(base_checkpoint)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(config["model"]["voxel_resolution"]),
        max_disparity_span=config["model"].get("max_disparity_span"),
    )
    assert model.disparity_refiner is not None
    model.disparity_refiner.load_state_dict(checkpoint["refiner"], strict=True)
    model.eval()
    iterations = tuple(int(value) for value in config["evaluation"]["iterations"])
    local_cfg = config["evaluation"]["local_points"]
    started = time.time()
    with torch.inference_mode():
        for index, entry in enumerate(samples, start=1):
            sample_id = str(entry["id"])
            if sample_id in per_image:
                continue
            sample = load_eth3d_sample(
                entry,
                output_hw=(int(config["model"]["height"]), int(config["model"]["width"])),
            )
            prompt, prompt_mask, _, reference_scale = make_eth3d_lidar_prompt(
                sample, config["lidar"], seed=stable_prompt_seed(int(config["seed"]), sample.sample_id)
            )
            output_value = model.forward_dense_refined(
                sample.image[None].to(device),
                prompt_disparity=prompt[None, None].to(device),
                prompt_mask=prompt_mask[None, None].to(device),
                query_hw=(int(config["model"]["height"]), int(config["model"]["width"])),
                num_refinement_steps=max(iterations),
                detach_base_from_refiner=True,
                chunk_size=int(config["model"]["query_chunk_size"]),
            )
            segment_masks = (
                None
                if local_root is None
                else torch.from_numpy(load_segment_masks(local_root / str(local_entries[sample_id]["path"])))
            )
            full_mask = sample.valid_mask
            held_out_mask = sample.valid_mask & ~prompt_mask
            values: dict[str, dict[str, object]] = {"all_valid": {}, "held_out": {}}
            for iteration in iterations:
                prediction = output_value.disparity_sequence[iteration][0]
                values["all_valid"][f"k{iteration}"] = metric_values(
                    prediction,
                    sample,
                    reference_scale=reference_scale,
                    evaluation_mask=full_mask,
                    segment_masks=None,
                    min_segment_pixels=int(local_cfg["min_segment_pixels"]),
                    delta_threshold=float(local_cfg["delta_threshold"]),
                )
                values["held_out"][f"k{iteration}"] = metric_values(
                    prediction,
                    sample,
                    reference_scale=reference_scale,
                    evaluation_mask=held_out_mask,
                    segment_masks=segment_masks,
                    min_segment_pixels=int(local_cfg["min_segment_pixels"]),
                    delta_threshold=float(local_cfg["delta_threshold"]),
                )
            per_image[sample_id] = {
                "scene": sample.scene,
                "prompt_pixel_count": int(prompt_mask.sum().item()),
                "valid_pixel_count": int(full_mask.sum().item()),
                "held_out_pixel_count": int(held_out_mask.sum().item()),
                **values,
            }
            progress["per_image"] = per_image
            progress["updated_at_unix"] = time.time()
            _atomic_json(progress_path, progress)
            print(f"{index}/{len(samples)} {sample_id}", flush=True)

    if len(per_image) != len(samples):
        raise RuntimeError(f"ETH3D evaluation incomplete: {len(per_image)}/{len(samples)}")
    ordered = {sample_id: per_image[sample_id] for sample_id in sample_ids}
    held_records = [value["held_out"] for value in ordered.values()]
    all_records = [value["all_valid"] for value in ordered.values()]
    by_scene, scene_macro = _aggregate_by_scene(ordered, iterations)
    report = {
        "format": REPORT_FORMAT,
        "checkpoint": {
            "path": str(checkpoint_dir),
            "sha256": checkpoint_sha,
            "step": int(checkpoint["step"]),
            "base_checkpoint_sha256": base_sha,
        },
        "data": {
            "dataset": inputs["dataset"],
            "split": str(config["data"]["split"]),
            "sample_count": len(samples),
            "scene_count": len(by_scene),
            "input_manifest_sha256": input_sha,
            "depth_representation": inputs["depth_representation"],
            "depth_to_radial_range": inputs["depth_to_radial_range"],
            "prompt_rule": "deterministic Exp4 virtual 64-line LiDAR, stride 4",
            "primary_evaluation_mask": "valid ETH3D depth pixels excluding virtual LiDAR prompt pixels",
        },
        "metric_semantics": {
            "metric_disparity_mae_1_per_m": "MAE of reciprocal camera-centric radial range in 1/m",
            "radial_depth_abs_rel": "absolute relative error of camera-centric radial range",
            "point_delta_0_01": "radial point error below 0.01 times min(predicted, GT) range",
            "local_point_rel": "MoGe-3 local point Rel after one global scale and a 3D translation per retained segment",
            "local_point_delta_0_01": "MoGe-3 local point delta at threshold 0.01",
            "aggregation": "standard metrics are image macro means; local metrics are retained-segment macro means",
        },
        "aggregate": {"held_out": _aggregate(held_records, iterations), "all_valid": _aggregate(all_records, iterations)},
        "scene_aggregate": {"per_scene": by_scene, "scene_macro": scene_macro},
        "comparison": _comparison(ordered, iterations, int(config["seed"])),
        "per_image": ordered,
        "duration_seconds_this_invocation": time.time() - started,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    _atomic_json(report_path, report)


if __name__ == "__main__":
    main()

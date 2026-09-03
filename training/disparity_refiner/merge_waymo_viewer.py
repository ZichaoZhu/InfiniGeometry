from __future__ import annotations

"""Create the unified Exp5 Waymo viewer manifest without copying point-cloud assets."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


FRONT_TAG = "exp5_waymo_val202_seed173"
SIDE_TAG = "exp5_waymo_side_selected_20260903"
TARGET_TAG = "exp5_waymo"
TARGET_EXPERIMENT = "exp5_waymo"
REPLACED_EXPERIMENTS = {
    "exp5_waymo_generalization",
    "exp5_waymo_side_visualization",
    TARGET_EXPERIMENT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge existing Exp5 Waymo FRONT and SIDE assets")
    parser.add_argument(
        "--viewer-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiment" / "viewer",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, must_not_exist: bool) -> None:
    if must_not_exist and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _asset_values(sample: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    ground_truth = sample.get("groundTruth")
    if isinstance(ground_truth, Mapping) and "url" in ground_truth:
        yield ground_truth
    stages = sample.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError(f"Sample {sample.get('id')} has no stages")
    for stage in stages.values():
        if not isinstance(stage, Mapping):
            raise ValueError(f"Sample {sample.get('id')} has an invalid stage")
        for asset in stage.values():
            if isinstance(asset, Mapping) and "url" in asset:
                yield asset


def _verify_assets(public_root: Path, samples: Iterable[Mapping[str, Any]]) -> None:
    for sample in samples:
        rgb_url = sample.get("rgbUrl")
        if not isinstance(rgb_url, str) or not rgb_url.startswith("/data/"):
            raise ValueError(f"Sample {sample.get('id')} has an invalid RGB URL")
        rgb_path = (public_root / rgb_url.lstrip("/")).resolve()
        if public_root.resolve() not in rgb_path.parents or not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        for asset in _asset_values(sample):
            url = asset.get("url")
            expected_sha = asset.get("sha256")
            if not isinstance(url, str) or not url.startswith("/data/"):
                raise ValueError(f"Sample {sample.get('id')} has an invalid asset URL")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError(f"Sample {sample.get('id')} has no SHA-256")
            path = (public_root / url.lstrip("/")).resolve()
            if public_root.resolve() not in path.parents or not path.is_file():
                raise FileNotFoundError(path)
            actual_sha = _sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {path}: {actual_sha}")


def _sample_copy(sample: Mapping[str, Any], *, label: str, order: int, camera_prefix: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(sample))
    value.pop("split", None)
    value["label"] = label
    value["order"] = order
    description = str(value.get("description", ""))
    if not description.startswith(camera_prefix):
        value["description"] = f"{camera_prefix} · {description}" if description else camera_prefix
    return value


def _validate_source(manifest: Mapping[str, Any], *, expected_experiment: str) -> list[dict[str, Any]]:
    if manifest.get("experiment") != expected_experiment:
        raise ValueError(f"Unexpected source experiment: {manifest.get('experiment')}")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 5:
        raise ValueError(f"{expected_experiment} must have exactly five samples")
    if manifest.get("resolution") != {"height": 384, "width": 512}:
        raise ValueError(f"{expected_experiment} resolution does not match Exp5")
    return [dict(sample) for sample in samples]


def merge(viewer_root: Path) -> Path:
    viewer_root = viewer_root.expanduser().resolve()
    public_root = viewer_root / "public"
    data_root = public_root / "data"
    front_path = data_root / FRONT_TAG / "manifest.json"
    side_path = data_root / SIDE_TAG / "manifest.json"
    front = _read_json(front_path)
    side = _read_json(side_path)
    front_samples = _validate_source(front, expected_experiment="exp5_waymo_generalization")
    side_samples = _validate_source(side, expected_experiment="exp5_waymo_side_visualization")
    if front.get("stages") != side.get("stages"):
        raise ValueError("FRONT and SIDE stage definitions differ")
    _verify_assets(public_root, [*front_samples, *side_samples])

    samples = [
        *[
            _sample_copy(sample, label=f"Waymo FRONT {index:02d}", order=index, camera_prefix="FRONT")
            for index, sample in enumerate(front_samples, start=1)
        ],
        *[
            _sample_copy(
                sample,
                label=f"Waymo SIDE {index:02d}",
                order=index + len(front_samples),
                camera_prefix=str(sample["description"]).split(" · ", 1)[0],
            )
            for index, sample in enumerate(side_samples, start=1)
        ],
    ]
    sample_ids = [str(sample["id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("FRONT and SIDE selections overlap")

    manifest = {
        "coordinateSpace": front["coordinateSpace"],
        "defaultStages": front["defaultStages"],
        "displayNote": "Waymo FRONT 与手选 SIDE 图像；每张图使用对应相机标定及 64 线虚拟 LiDAR prompt，预测点云仅显示 0-100 m。",
        "experiment": TARGET_EXPERIMENT,
        "groundTruthDetail": "未作为 prompt 输入的 TOP LiDAR 第一回波投影到对应相机；稀疏点云，不是稠密 GT。",
        "groundTruthTitle": "Waymo held-out TOP LiDAR",
        "metricsNote": "十张样例仅用于定性可视化；FRONT 五张来自固定随机选择，SIDE 五张由用户选定，均未用于选择 checkpoint。",
        "resolution": front["resolution"],
        "samples": samples,
        "stages": front["stages"],
        "steps": front["steps"],
        "version": 1,
        "voxelization": front["voxelization"],
        "websiteOrder": sample_ids,
    }
    output_dir = data_root / TARGET_TAG
    output = output_dir / "manifest.json"
    provenance = {
        "format": "infinidepth-exp5-waymo-combined-selection-v1",
        "front": {
            "manifest": str(front_path.relative_to(public_root)),
            "manifest_sha256": _sha256(front_path),
            "sample_ids": [str(sample["id"]) for sample in front_samples],
        },
        "side": {
            "manifest": str(side_path.relative_to(public_root)),
            "manifest_sha256": _sha256(side_path),
            "sample_ids": [str(sample["id"]) for sample in side_samples],
        },
        "target_experiment": TARGET_EXPERIMENT,
    }
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _atomic_json(output, manifest, must_not_exist=True)
    _atomic_json(output_dir / "selection.json", provenance, must_not_exist=True)

    catalog_path = data_root / "experiments.json"
    catalog = _read_json(catalog_path)
    experiments = catalog.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Viewer catalog has no experiments list")
    catalog["experiments"] = [
        item for item in experiments if isinstance(item, Mapping) and item.get("id") not in REPLACED_EXPERIMENTS
    ] + [
        {
            "id": TARGET_EXPERIMENT,
            "label": "Exp5 Waymo",
            "manifestUrl": f"/data/{TARGET_TAG}/manifest.json",
            "shortLabel": "Exp5 Waymo",
            "stageDetails": {
                "best_step_22500": "Exp4 step 22,500 在 Waymo FRONT 与手选 SIDE 图像上的 zero-shot 推理",
                "initial": "冻结 LiDAR-conditioned DepthSensor 的 K=0 输出",
            },
            "stageLabels": {
                "best_step_22500": "Waymo Zero-shot",
                "initial": "LiDAR 初始",
            },
            "summary": "Exp4 LiDAR Refiner 在 Waymo 的 zero-shot 点云可视化：5 张 FRONT 加 5 张手选 SIDE。",
        }
    ]
    _atomic_json(catalog_path, catalog, must_not_exist=False)
    return output


def main() -> None:
    output = merge(parse_args().viewer_root)
    print(output)


if __name__ == "__main__":
    main()

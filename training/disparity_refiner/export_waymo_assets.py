from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.backup import verify_checkpoint_directory
from training.disparity_refiner.data import ensure_within
from training.disparity_refiner.evaluate_waymo import _validate_config
from training.disparity_refiner.export_assets import public_url, sha256, write_json, write_point_cloud
from training.disparity_refiner.train_lidar import CHECKPOINT_FORMAT
from training.disparity_refiner.waymo import (
    WaymoSparseSample,
    camera_rays_from_calibration,
    list_tfrecords,
    load_waymo_sample,
)


STEPS = (0, 1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export five Waymo point-cloud viewer samples")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--viewer-config",
        type=Path,
        help="Optional viewer configuration relative to the experiment directory",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_sources(viewer: Mapping[str, object]) -> list[tuple[str, str]]:
    """Read either legacy FRONT strings or explicit {source, camera} selections."""
    values = viewer.get("sources")
    if not isinstance(values, list):
        raise ValueError("Waymo viewer sources must be a list")
    selections: list[tuple[str, str]] = []
    for value in values:
        if isinstance(value, str):
            source, camera = value, "FRONT"
        elif isinstance(value, Mapping):
            source = str(value.get("source", ""))
            camera = str(value.get("camera", ""))
        else:
            raise ValueError("Waymo viewer source must be a string or an object")
        if not source.endswith(".tfrecord") or not camera:
            raise ValueError(f"Invalid Waymo viewer selection: {value}")
        selections.append((source, camera))
    if len(selections) != 5 or len(set(selections)) != 5:
        raise ValueError("Waymo viewer requires exactly five unique source/camera selections")
    return selections


def _catalog_sort_key(entry: Mapping[str, object]) -> tuple[int, str]:
    label = str(entry["label"])
    match = re.search(r"\d+", label)
    return (int(match.group()) if match else 10**9, label)


def _asset(path: Path, public_root: Path, metadata: Mapping[str, object], checkpoint: str) -> dict[str, object]:
    return {
        **metadata,
        "alignment": {"scale": 1.0, "zShift": 0.0},
        "checkpointSha256": checkpoint,
        "url": public_url(path, public_root),
    }


def _camera_rays(sample: WaymoSparseSample, output_hw: tuple[int, int]) -> np.ndarray:
    return camera_rays_from_calibration(
        sample.metadata["camera_intrinsic"],
        input_hw=(
            int(sample.metadata["camera_image_height"]),
            int(sample.metadata["camera_image_width"]),
        ),
        output_hw=output_hw,
    )


def _prediction_points(
    normalized_disparity: torch.Tensor,
    reference_scale: float,
    rays: np.ndarray,
    max_depth: float,
) -> np.ndarray:
    metric_disparity = normalized_disparity.float().numpy() * reference_scale
    valid = np.isfinite(metric_disparity) & (metric_disparity > 1e-6)
    radial_depth = np.full_like(metric_disparity, np.nan, dtype=np.float32)
    radial_depth[valid] = 1.0 / metric_disparity[valid]
    radial_depth[radial_depth > max_depth] = np.nan
    return rays * radial_depth[..., None]


def _ground_truth_points(
    sample: WaymoSparseSample,
    rays: np.ndarray,
    max_depth: float,
) -> np.ndarray:
    radial_depth = sample.target_radial_depth.numpy()
    valid = sample.evaluation_mask.numpy() & np.isfinite(radial_depth) & (radial_depth <= max_depth)
    points = np.full((*radial_depth.shape, 3), np.nan, dtype=np.float32)
    points[valid] = rays[valid] * radial_depth[valid, None]
    return points


@torch.no_grad()
def _predict(
    model: InfiniDepth_DepthSensor,
    sample: WaymoSparseSample,
    *,
    device: torch.device,
    output_hw: tuple[int, int],
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    output = model.forward_dense_refined(
        sample.image[None].to(device),
        prompt_disparity=sample.prompt_disparity[None].to(device),
        prompt_mask=sample.prompt_mask[None].to(device),
        query_hw=output_hw,
        num_refinement_steps=max(STEPS),
        detach_base_from_refiner=True,
        chunk_size=chunk_size,
    )
    output_scale = float(output.reference_scale.reshape(-1)[0].item())
    if not np.isclose(output_scale, float(sample.reference_scale), rtol=1e-5):
        raise ValueError("DepthSensor prompt normalization scale mismatch")
    return {step: output.disparity_sequence[step][0].detach().cpu() for step in STEPS}


def _export_sample(
    sample: WaymoSparseSample,
    predictions: Mapping[int, torch.Tensor],
    *,
    output: Path,
    public_root: Path,
    output_hw: tuple[int, int],
    max_depth: float,
    base_sha: str,
    checkpoint_sha: str,
    checkpoint_step: int,
    label_prefix: str,
    order: int,
) -> dict[str, object]:
    colors = np.round(sample.image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    rays = _camera_rays(sample, output_hw)
    sample_dir = output / sample.sample_id
    rgb_path = sample_dir / "source_rgb.jpg"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colors).save(rgb_path, quality=92)

    gt_path = sample_dir / "held_out_top_lidar.ply"
    ground_truth = _asset(
        gt_path,
        public_root,
        write_point_cloud(gt_path, _ground_truth_points(sample, rays, max_depth), colors),
        "waymo-held-out-top-lidar",
    )
    k0_path = sample_dir / "initial_k0.ply"
    k0 = _asset(
        k0_path,
        public_root,
        write_point_cloud(
            k0_path,
            _prediction_points(predictions[0], float(sample.reference_scale), rays, max_depth),
            colors,
        ),
        base_sha,
    )
    initial = {"k0": k0, **{f"k{step}": {"alias": "initial.k0"} for step in STEPS[1:]}}
    best: dict[str, object] = {"k0": {"alias": "initial.k0"}}
    for step in STEPS[1:]:
        path = sample_dir / f"best_step_{checkpoint_step}_k{step}.ply"
        best[f"k{step}"] = _asset(
            path,
            public_root,
            write_point_cloud(
                path,
                _prediction_points(
                    predictions[step], float(sample.reference_scale), rays, max_depth
                ),
                colors,
            ),
            checkpoint_sha,
        )
    description = " · ".join(
        value
        for value in (
            str(sample.metadata.get("camera", "")),
            str(sample.metadata.get("location", "")),
            str(sample.metadata.get("time_of_day", "")),
            str(sample.metadata.get("weather", "")),
        )
        if value
    )
    return {
        "cropXYXY": [0, 0, output_hw[1], output_hw[0]],
        "description": description or "Waymo validation 随机样本",
        "groundTruth": ground_truth,
        "id": sample.sample_id,
        "label": f"{label_prefix} {order:02d}",
        "order": order,
        "rgbUrl": public_url(rgb_path, public_root),
        "split": "val",
        "stages": {"initial": initial, f"best_step_{checkpoint_step}": best},
        "websiteEnabled": True,
    }


def _update_catalog(
    catalog_path: Path,
    *,
    experiment_id: str,
    asset_tag: str,
    checkpoint_stage: str,
    summary: str,
    label: str,
    short_label: str,
    stage_labels: Mapping[str, str],
    stage_details: Mapping[str, str],
) -> None:
    catalog = dict(_load_json(catalog_path))
    entries = [entry for entry in catalog["experiments"] if entry["id"] != experiment_id]
    entries.append(
        {
            "id": experiment_id,
            "label": label,
            "shortLabel": short_label,
            "manifestUrl": f"/data/{asset_tag}/manifest.json",
            "summary": summary,
            "stageLabels": dict(stage_labels),
            "stageDetails": dict(stage_details),
        }
    )
    catalog["experiments"] = sorted(entries, key=_catalog_sort_key)
    write_json(catalog_path, catalog)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config = _load_json(experiment / "config.json")
    viewer_config = args.viewer_config or Path("viewer.json")
    if viewer_config.is_absolute() or ".." in viewer_config.parts:
        raise ValueError("viewer-config must be relative to the experiment directory")
    viewer = _load_json(experiment / viewer_config)
    _validate_config(config)
    if viewer.get("format") != "infinidepth-exp5-waymo-viewer-v1":
        raise ValueError("Unexpected Waymo viewer config")
    selections = _selected_sources(viewer)
    sources = [source for source, _ in selections]
    sample_policy = viewer.get("sample_policy")
    if not isinstance(sample_policy, Mapping):
        raise ValueError("Waymo viewer requires sample_policy metadata")
    display = viewer.get("display", {})
    if not isinstance(display, Mapping):
        raise ValueError("Waymo viewer display metadata must be an object")
    catalog_config = viewer.get("catalog", {})
    if not isinstance(catalog_config, Mapping):
        raise ValueError("Waymo viewer catalog metadata must be an object")

    safe_root = Path(str(config["server"]["safe_root"]))
    report_path = ensure_within(Path(str(viewer["report"])), safe_root, name="Waymo report")
    report = _load_json(report_path)
    if report.get("format") != "infinidepth-exp5-waymo-evaluation-v1":
        raise ValueError("Unexpected Waymo evaluation report")
    if int(report["data"]["sample_count"]) != int(config["data"]["expected_count"]):
        raise ValueError("Waymo report is incomplete")
    missing = sorted(set(sources) - set(report["per_image"]))
    if missing:
        raise ValueError(f"Selected Waymo sources are absent from the report: {missing}")

    data_root = Path(str(config["data"]["root"])).resolve()
    records = {path.name: path for path in list_tfrecords(data_root, expected_count=int(config["data"]["expected_count"]))}
    reader_root = ensure_within(Path(str(config["data"]["reader_root"])), safe_root, name="Waymo reader")
    checkpoint_dir = ensure_within(Path(str(config["checkpoint"]["path"])), safe_root, name="Exp4 checkpoint")
    verify_checkpoint_directory(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != str(config["checkpoint"]["sha256"]) or checkpoint_sha != str(report["checkpoint"]["sha256"]):
        raise ValueError("Exp4 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported Exp4 checkpoint format")
    checkpoint_step = int(checkpoint["step"])

    base_checkpoint = ensure_within(Path(str(config["model"]["checkpoint"])), safe_root, name="DepthSensor checkpoint")
    base_sha = sha256(base_checkpoint)
    if base_sha != str(config["model"]["checkpoint_sha256"]):
        raise ValueError("DepthSensor checkpoint SHA-256 mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Waymo viewer export requires CUDA")
    torch.cuda.set_device(device)
    model_cfg = config["model"]
    evaluation_cfg = config["evaluation"]
    output_hw = (int(model_cfg["height"]), int(model_cfg["width"]))
    model = InfiniDepth_DepthSensor(model_path=str(base_checkpoint)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(model_cfg["voxel_resolution"]),
        max_disparity_span=model_cfg.get("max_disparity_span"),
    )
    assert model.disparity_refiner is not None
    model.disparity_refiner.load_state_dict(checkpoint["refiner"], strict=True)
    model.eval()

    public_root = project_root / "experiment" / "viewer" / "public"
    asset_tag = str(viewer["asset_tag"])
    output = public_root / "data" / asset_tag
    if output.exists():
        raise FileExistsError(output)
    max_depth = float(viewer["visual_max_radial_depth_m"])
    samples = []
    for order, (source, camera_name) in enumerate(selections, start=1):
        sample = load_waymo_sample(
            records[source],
            reader_root=reader_root,
            frame_index=int(evaluation_cfg["frame_index"]),
            frame_search=int(evaluation_cfg["frame_search"]),
            output_hw=output_hw,
            prompt_stride=int(evaluation_cfg["prompt_stride"]),
            camera_name=camera_name,
            lidar_name=str(evaluation_cfg["lidar"]),
            min_prompt_points=int(evaluation_cfg["min_prompt_points"]),
            min_evaluation_points=int(evaluation_cfg["min_evaluation_points"]),
        )
        predictions = _predict(
            model,
            sample,
            device=device,
            output_hw=output_hw,
            chunk_size=int(model_cfg["query_chunk_size"]),
        )
        samples.append(
            _export_sample(
                sample,
                predictions,
                output=output,
                public_root=public_root,
                output_hw=output_hw,
                max_depth=max_depth,
                base_sha=base_sha,
                checkpoint_sha=checkpoint_sha,
                checkpoint_step=checkpoint_step,
                label_prefix=str(display.get("label_prefix", "Waymo Val")),
                order=order,
            )
        )
        print(f"[{order}/{len(sources)}] exported {source}", flush=True)

    checkpoint_stage = str(viewer["checkpoint_stage"])
    experiment_id = str(catalog_config.get("id", config["experiment_id"]))
    catalog_label = str(catalog_config.get("label", "Exp5"))
    catalog_short_label = str(catalog_config.get("short_label", catalog_label))
    summary = str(
        catalog_config.get("summary", "Exp4 LiDAR Refiner 在 Waymo 上的 zero-shot 点云可视化")
    )
    stage_labels = {
        "initial": str(catalog_config.get("initial_stage_label", "LiDAR 初始")),
        checkpoint_stage: str(catalog_config.get("checkpoint_stage_label", "Waymo Zero-shot")),
    }
    stage_details = {
        "initial": str(catalog_config.get("initial_stage_detail", "冻结 DepthSensor 的 K=0 输出")),
        checkpoint_stage: str(
            catalog_config.get(
                "checkpoint_stage_detail", "Exp4 step 22,500 在 Waymo 上 zero-shot 推理"
            )
        ),
    }
    write_json(
        output / "manifest.json",
        {
            "coordinateSpace": {
                "aligned": "metric OpenCV camera XYZ",
                "stored": "metric OpenCV camera XYZ",
                "threeDisplay": "[x, -y, -z]",
            },
            "defaultStages": {"left": "initial", "right": checkpoint_stage},
            "displayNote": str(
                display.get(
                    "note",
                    f"Waymo 图像；预测使用 prompt disparity 的中位数恢复米制，点云仅显示 0–{max_depth:g} m。",
                )
            ),
            "experiment": experiment_id,
            "groundTruthDetail": str(
                display.get(
                    "ground_truth_detail",
                    "未作为 prompt 输入的 TOP LiDAR 第一回波投影；稀疏点云，不是稠密 GT",
                )
            ),
            "groundTruthTitle": str(display.get("ground_truth_title", "Waymo held-out TOP LiDAR")),
            "metricsNote": str(
                display.get("metrics_note", "本批 Waymo 资产只用于可视化，未新增计算 Local Point 等评测指标。")
            ),
            "resolution": {"height": output_hw[0], "width": output_hw[1]},
            "samplePolicy": {
                "algorithm": str(sample_policy["algorithm"]),
                "countPerSplit": int(sample_policy["count"]),
                "seed": int(sample_policy["seed"]),
                "splitOrder": ["val"],
                "testUsage": "只用于定性可视化，不据此选择 checkpoint。",
            },
            "samples": samples,
            "stages": ["initial", checkpoint_stage],
            "steps": list(STEPS),
            "version": 1,
            "voxelization": {
                "depthCoordinate": "round(200 * normalized prompt disparity)",
                "depthScale": float(model_cfg["voxel_resolution"]),
                "spconvOrder": ["batch", "disparity_bin", "row", "column"],
            },
            "websiteSampleOrder": [sample["id"] for sample in samples],
        },
    )
    write_json(
        output / "selection.json",
        {
            "format": "infinidepth-exp5-waymo-viewer-selection-v1",
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "seed": int(sample_policy["seed"]),
            "sources": [
                {"camera": camera_name, "source": source}
                for source, camera_name in selections
            ],
        },
    )
    _update_catalog(
        public_root / "data" / "experiments.json",
        experiment_id=experiment_id,
        asset_tag=asset_tag,
        checkpoint_stage=checkpoint_stage,
        summary=summary,
        label=catalog_label,
        short_label=catalog_short_label,
        stage_labels=stage_labels,
        stage_details=stage_details,
    )


if __name__ == "__main__":
    main()

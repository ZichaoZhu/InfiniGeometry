from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.backup import verify_checkpoint_directory
from training.disparity_refiner.data import ensure_within
from training.disparity_refiner.eth3d import (
    load_eth3d_sample,
    make_eth3d_lidar_prompt,
    read_input_manifest,
    sha256,
    stable_prompt_seed,
)
from training.disparity_refiner.export_assets import public_url, write_point_cloud
from training.disparity_refiner.train_lidar import CHECKPOINT_FORMAT


STEPS = (0, 1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected ETH3D point-cloud viewer samples")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--viewer-config", type=Path, default=Path("viewer_selected.json"))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_public_urls(value: object, old_prefix: str, new_prefix: str) -> object:
    if isinstance(value, dict):
        return {key: _rewrite_public_urls(item, old_prefix, new_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_public_urls(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix)
    return value


def _asset(path: Path, public_root: Path, metadata: Mapping[str, object], checkpoint: str) -> dict[str, object]:
    return {
        **metadata,
        "alignment": {"scale": 1.0, "zShift": 0.0},
        "checkpointSha256": checkpoint,
        "url": public_url(path, public_root),
    }


def _prediction_points(disparity: torch.Tensor, reference_scale: torch.Tensor, rays: torch.Tensor, max_depth: float) -> np.ndarray:
    metric_disparity = (disparity.float() * reference_scale.float()).cpu().numpy()
    radial = np.full_like(metric_disparity, np.nan, dtype=np.float32)
    valid = np.isfinite(metric_disparity) & (metric_disparity > 1e-6)
    radial[valid] = 1.0 / metric_disparity[valid]
    radial[radial > max_depth] = np.nan
    return rays.cpu().numpy() * radial[..., None]


def _ground_truth_points(sample, max_depth: float) -> np.ndarray:
    radial = sample.radial_depth.cpu().numpy().copy()
    valid = sample.valid_mask.cpu().numpy() & np.isfinite(radial) & (radial <= max_depth)
    radial[~valid] = np.nan
    return sample.rays.cpu().numpy() * radial[..., None]


@torch.no_grad()
def _predict(model, sample, *, device: torch.device, output_hw: tuple[int, int], chunk_size: int) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    prompt, prompt_mask, _, reference_scale = make_eth3d_lidar_prompt(
        sample, model._exp5_lidar_settings, seed=model._exp5_lidar_seed
    )
    output = model.forward_dense_refined(
        sample.image[None].to(device),
        prompt_disparity=prompt[None, None].to(device),
        prompt_mask=prompt_mask[None, None].to(device),
        query_hw=output_hw,
        num_refinement_steps=max(STEPS),
        detach_base_from_refiner=True,
        chunk_size=chunk_size,
    )
    return ({step: output.disparity_sequence[step][0].detach().cpu() for step in STEPS}, reference_scale)


def _update_catalog(path: Path, *, asset_tag: str) -> None:
    catalog = dict(load_json(path))
    entries = [entry for entry in catalog["experiments"] if entry["id"] != "exp5_eth3d"]
    entries.append(
        {
            "id": "exp5_eth3d",
            "label": "Exp5 ETH3D",
            "shortLabel": "Exp5 ETH3D",
            "manifestUrl": f"/data/{asset_tag}/manifest.json",
            "summary": "Exp5 ETH3D 选定十张图像的 LiDAR Refiner 点云可视化。",
            "stageLabels": {"initial": "LiDAR 初始", "best_step_22500": "ETH3D Zero-shot"},
            "stageDetails": {
                "initial": "冻结 DepthSensor 的 K=0 输出",
                "best_step_22500": "Exp4 step 22,500 在 ETH3D 上 zero-shot 推理",
            },
        }
    )
    entries.sort(key=lambda value: (int(value["id"][3:value["id"].find("_")]), value["label"]))
    catalog["experiments"] = entries
    write_json(path, catalog)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config = load_json(experiment / "config.json")
    viewer_path = args.viewer_config if args.viewer_config.is_absolute() else experiment / args.viewer_config
    viewer = load_json(viewer_path)
    if config.get("experiment_id") != "exp5_ETH3D" or viewer.get("format") != "infinidepth-exp5-eth3d-viewer-v1":
        raise ValueError("Unexpected ETH3D Exp5 configuration")
    sources = [str(value) for value in viewer.get("sources", [])]
    if len(sources) != 10 or len(set(sources)) != 10:
        raise ValueError("ETH3D viewer requires exactly ten unique source IDs")
    safe_root = Path(str(config["server"]["safe_root"]))
    input_path = ensure_within(Path(str(config["data"]["input_manifest"])), safe_root, name="ETH3D input manifest")
    report_path = ensure_within(Path(str(viewer["report"])), safe_root, name="ETH3D report")
    inputs = read_input_manifest(input_path)
    report = load_json(report_path)
    entries = {str(item["id"]): item for item in inputs["samples"]}
    per_image = report.get("per_image")
    if not isinstance(per_image, Mapping) or set(sources) - set(entries) or set(sources) - set(per_image):
        raise ValueError("Selected ETH3D sources are absent from the completed report")
    checkpoint_dir = ensure_within(Path(str(config["checkpoint"]["path"])), safe_root, name="Exp4 checkpoint")
    verify_checkpoint_directory(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != str(config["checkpoint"]["sha256"]):
        raise ValueError("Exp4 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported Exp4 checkpoint format")
    base_path = ensure_within(Path(str(config["model"]["checkpoint"])), safe_root, name="DepthSensor checkpoint")
    base_sha = sha256(base_path)
    if base_sha != str(config["model"]["checkpoint_sha256"]):
        raise ValueError("DepthSensor checkpoint SHA-256 mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ETH3D viewer export requires CUDA")
    torch.cuda.set_device(device)
    model_cfg = config["model"]
    output_hw = (int(model_cfg["height"]), int(model_cfg["width"]))
    model = InfiniDepth_DepthSensor(model_path=str(base_path)).to(device)
    model.attach_disparity_refiner(backend="spconv", voxel_resolution=float(model_cfg["voxel_resolution"]), max_disparity_span=model_cfg.get("max_disparity_span"))
    assert model.disparity_refiner is not None
    model.disparity_refiner.load_state_dict(checkpoint["refiner"], strict=True)
    model.eval()
    # Keep prompt construction explicit without expanding the model API for this visual-only exporter.
    model._exp5_lidar_settings = config["lidar"]
    public_root = project_root / "experiment" / "viewer" / "public"
    asset_tag = str(viewer["asset_tag"])
    output = public_root / "data" / asset_tag
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    max_depth = float(viewer["visual_max_radial_depth_m"])
    samples = []
    try:
        for order, sample_id in enumerate(sources, start=1):
            entry = entries[sample_id]
            sample = load_eth3d_sample(entry, output_hw=output_hw)
            prompt, prompt_mask, _, reference_scale = make_eth3d_lidar_prompt(
                sample, config["lidar"], seed=stable_prompt_seed(int(config["seed"]), sample_id)
            )
            model_output = model.forward_dense_refined(
                sample.image[None].to(device),
                prompt_disparity=prompt[None, None].to(device),
                prompt_mask=prompt_mask[None, None].to(device),
                query_hw=output_hw,
                num_refinement_steps=max(STEPS),
                detach_base_from_refiner=True,
                chunk_size=int(model_cfg["query_chunk_size"]),
            )
            predictions = {step: model_output.disparity_sequence[step][0].detach().cpu() for step in STEPS}
            colors = np.round(sample.image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            sample_dir = temporary / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = sample_dir / "source_rgb.jpg"
            Image.fromarray(colors).save(rgb_path, quality=92)
            gt_path = sample_dir / "ground_truth.ply"
            gt = _asset(gt_path, public_root, write_point_cloud(gt_path, _ground_truth_points(sample, max_depth), colors), "eth3d-ground-truth")
            initial_path = sample_dir / "initial_k0.ply"
            initial_k0 = _asset(initial_path, public_root, write_point_cloud(initial_path, _prediction_points(predictions[0], reference_scale, sample.rays, max_depth), colors), base_sha)
            initial = {"k0": initial_k0, **{f"k{step}": {"alias": "initial.k0"} for step in STEPS[1:]}}
            best = {"k0": {"alias": "initial.k0"}}
            for step in STEPS[1:]:
                path = sample_dir / f"best_step_22500_k{step}.ply"
                best[f"k{step}"] = _asset(path, public_root, write_point_cloud(path, _prediction_points(predictions[step], reference_scale, sample.rays, max_depth), colors), checkpoint_sha)
            record = per_image[sample_id]
            samples.append({
                "cropXYXY": list(sample.metadata["crop_xyxy"]),
                "description": f"{sample.scene} · {record['valid_pixel_count']} valid pixels · {record['prompt_pixel_count']} virtual LiDAR pixels",
                "groundTruth": gt,
                "id": sample_id,
                "label": f"ETH3D {order:02d}",
                "order": order,
                "rgbUrl": public_url(rgb_path, public_root),
                "split": "val",
                "stages": {"initial": initial, "best_step_22500": best},
                "websiteEnabled": True,
            })
            print(f"[{order}/{len(sources)}] exported {sample_id}", flush=True)
        samples = _rewrite_public_urls(
            samples,
            public_url(temporary, public_root),
            public_url(output, public_root),
        )
        write_json(temporary / "manifest.json", {
            "coordinateSpace": {"aligned": "metric OpenCV camera XYZ", "stored": "metric OpenCV camera XYZ", "threeDisplay": "[x, -y, -z]"},
            "defaultStages": {"left": "initial", "right": "best_step_22500"},
            "displayNote": str(viewer["display"]["note"]),
            "experiment": "exp5_eth3d",
            "groundTruthDetail": str(viewer["display"]["ground_truth_detail"]),
            "groundTruthTitle": str(viewer["display"]["ground_truth_title"]),
            "metricsNote": str(viewer["display"]["metrics_note"]),
            "resolution": {"height": output_hw[0], "width": output_hw[1]},
            "samples": samples,
            "stages": ["initial", "best_step_22500"],
            "steps": list(STEPS),
            "version": 1,
            "voxelization": {"depthCoordinate": "round(200 * normalized prompt disparity)", "depthScale": float(model_cfg["voxel_resolution"]), "spconvOrder": ["batch", "disparity_bin", "row", "column"]},
            "websiteSampleOrder": sources,
        })
        write_json(temporary / "selection.json", {"format": "infinidepth-exp5-eth3d-viewer-selection-v1", "report": str(report_path), "report_sha256": sha256(report_path), "sources": sources})
        _update_catalog(public_root / "data" / "experiments.json", asset_tag=asset_tag)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

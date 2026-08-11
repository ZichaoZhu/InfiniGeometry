from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch

from InfiniDepth.model import InfiniDepth
from training.disparity_refiner.data import (
    HypersimDisparitySample,
    ensure_within,
    load_manifest,
    preload_samples,
    select_training_entries,
)
from training.disparity_refiner.losses import point_cloud_metrics


STEPS = (0, 1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 disparity Refiner 最终图和查看器资产")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_all_samples(
    config: Mapping[str, object], project_root: Path
) -> Sequence[HypersimDisparitySample]:
    manifest_path = project_root / str(config["data"]["manifest"])
    manifest = load_manifest(manifest_path, str(config["data"]["manifest_sha256"]))
    sample_ids = [str(value) for value in config["evaluation"]["sample_ids"]]
    entries = select_training_entries(manifest, sample_ids)
    selections = {
        str(entry["sample_id"]): entry
        for entry in config["structure_selections"]
    }
    model_cfg = config["model"]
    return preload_samples(
        entries,
        source_root=Path(str(config["data"]["source_root"])),
        height=int(model_cfg["height"]),
        width=int(model_cfg["width"]),
        structure_selections=selections,
    )


@torch.no_grad()
def predict(
    model: InfiniDepth,
    sample: HypersimDisparitySample,
    device: torch.device,
    query_hw: Tuple[int, int],
    chunk_size: int,
) -> Dict[int, torch.Tensor]:
    model.eval()
    output = model.forward_dense_refined(
        sample.image[None].to(device),
        query_hw=query_hw,
        num_refinement_steps=max(STEPS),
        chunk_size=chunk_size,
    )
    return {
        iteration: output.disparity_sequence[iteration][0].detach().cpu()
        for iteration in STEPS
    }


def camera_rays(sample: HypersimDisparitySample) -> np.ndarray:
    height, width = sample.radial_depth.shape
    matrix = np.asarray(sample.metadata["M_cam_from_uv"], dtype=np.float32)
    u = np.linspace(-1 + 1 / width, 1 - 1 / width, width, dtype=np.float32)
    v = np.linspace(-1 + 1 / height, 1 - 1 / height, height, dtype=np.float32)[::-1]
    grid_u, grid_v = np.meshgrid(u, v)
    uv1 = np.stack((grid_u, grid_v, np.ones_like(grid_u)), axis=-1)
    rays = uv1 @ matrix.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
    return rays * np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def disparity_to_radial(
    disparity: torch.Tensor,
    quantiles: Tuple[float, float],
) -> np.ndarray:
    low, high = quantiles
    raw = disparity.float().numpy() * (high - low) + low
    valid = np.isfinite(raw) & (raw > 1e-6)
    radial = np.full_like(raw, np.nan, dtype=np.float32)
    radial[valid] = 1.0 / raw[valid]
    return radial


def write_point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_points = points.reshape(-1, 3).astype(np.float32)
    flat_colors = colors.reshape(-1, 3).astype(np.uint8)
    vertices = np.empty(
        flat_points.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = flat_points.T
    vertices["red"], vertices["green"], vertices["blue"] = flat_colors.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    finite = np.isfinite(flat_points).all(axis=-1)
    bounds = {
        "min": flat_points[finite].min(axis=0).tolist(),
        "max": flat_points[finite].max(axis=0).tolist(),
    }
    return {
        "bytes": path.stat().st_size,
        "pointCount": int(flat_points.shape[0]),
        "validPointCount": int(finite.sum()),
        "sha256": sha256(path),
        "bounds": bounds,
    }


def public_url(path: Path, public_root: Path) -> str:
    return "/" + path.relative_to(public_root).as_posix()


def asset_record(
    path: Path,
    public_root: Path,
    metadata: Mapping[str, object],
    metrics: Mapping[str, object],
    checkpoint_sha256: str,
) -> Dict[str, object]:
    return {
        **metadata,
        "alignment": {"scale": 1.0, "zShift": 0.0},
        "checkpointSha256": checkpoint_sha256,
        "metrics": dict(metrics),
        "url": public_url(path, public_root),
    }


def export_sample_assets(
    *,
    sample: HypersimDisparitySample,
    predictions: Mapping[str, Mapping[int, torch.Tensor]],
    checkpoint_shas: Mapping[str, str],
    output: Path,
    public_root: Path,
    order: int,
    description: str,
    crop_xyxy: Sequence[int],
) -> Mapping[str, object]:
    colors = np.round(sample.image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    rays = camera_rays(sample)
    sample_dir = output / sample.sample_id
    rgb_path = sample_dir / "source_rgb.jpg"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colors).save(rgb_path, quality=92)
    gt_points = rays * sample.radial_depth.numpy()[..., None]
    gt_points[~sample.valid_mask.numpy()] = np.nan
    gt_tensor = torch.from_numpy(gt_points)
    gt_path = sample_dir / "ground_truth.ply"
    gt_meta = write_point_cloud(gt_path, gt_points, colors)
    ground_truth = asset_record(
        gt_path,
        public_root,
        gt_meta,
        point_cloud_metrics(
            gt_tensor, gt_tensor, sample.valid_mask, sample.structure_mask
        ),
        "ground-truth",
    )
    stages = {}
    for stage, maps in predictions.items():
        stage_assets = {}
        for iteration in STEPS:
            if stage == "initial" and iteration:
                stage_assets[f"k{iteration}"] = {"alias": "initial.k0"}
                continue
            disparity = maps[iteration]
            radial = disparity_to_radial(disparity, sample.disparity_quantiles)
            points = rays * radial[..., None]
            path = sample_dir / f"{stage}_k{iteration}.ply"
            metadata = write_point_cloud(path, points, colors)
            metrics = point_cloud_metrics(
                torch.from_numpy(points),
                gt_tensor,
                sample.valid_mask,
                sample.structure_mask,
            )
            stage_assets[f"k{iteration}"] = asset_record(
                path,
                public_root,
                metadata,
                metrics,
                checkpoint_shas[stage],
            )
        stages[stage] = stage_assets
    initial = stages["initial"]["k0"]
    initial_point_rel = float(initial["metrics"]["full"]["point_rel"])
    for stage_assets in stages.values():
        for asset in stage_assets.values():
            if "alias" not in asset:
                point_rel = float(asset["metrics"]["full"]["point_rel"])
                asset["pointRelReductionFromK0"] = (
                    0.0 if initial_point_rel == 0 else 1 - point_rel / initial_point_rel
                )
    return {
        "cropXYXY": list(crop_xyxy),
        "description": description,
        "disparityQuantiles": list(sample.disparity_quantiles),
        "groundTruth": ground_truth,
        "id": sample.sample_id,
        "label": f"样本 {order:02d}",
        "order": order,
        "rgbUrl": public_url(rgb_path, public_root),
        "stages": stages,
        "websiteEnabled": True,
    }


def load_refiner_checkpoint(
    model: InfiniDepth,
    path: Path,
) -> str:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if checkpoint.get("format") != "infinidepth-disparity-refiner-v1":
        raise ValueError(f"Unsupported checkpoint: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return sha256(path)


def export_training_curve(experiment: Path, config: Mapping[str, object]) -> None:
    run_count = len(config["runs"])
    columns = 2 if run_count > 1 else 1
    rows = (run_count + columns - 1) // columns
    figure, axes_value = plt.subplots(
        rows,
        columns,
        figsize=(13 if columns == 2 else 8, 3.8 * rows),
        sharex=False,
        squeeze=False,
    )
    axes = axes_value.ravel()
    for axis, run in zip(axes, config["runs"]):
        history_path = experiment / str(run["directory"]) / "metrics" / "history.jsonl"
        records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        steps = [int(record["total_step"]) for record in records]
        losses = [float(record["training"]["loss"]) for record in records]
        k0 = [
            float(record["evaluation"]["aggregate"]["k0"].get(
                "composite_score", record["evaluation"]["aggregate"]["k0"]["full_mae"]
            ))
            for record in records
        ]
        k3 = [
            float(record["evaluation"]["aggregate"]["k3"].get(
                "composite_score", record["evaluation"]["aggregate"]["k3"]["full_mae"]
            ))
            for record in records
        ]
        axis.plot(steps, losses, label="training loss", color="#2f6b4f", alpha=0.75)
        axis.plot(steps, k0, label="K0 score", color="#65717e", linewidth=1.5)
        axis.plot(steps, k3, label="K3 score", color="#c4473a", linewidth=1.5)
        joint_records = [record for record in records if record["stage"] == "joint"]
        if joint_records:
            stage_boundary = int(joint_records[0]["total_step"]) - int(joint_records[0]["stage_step"])
            axis.axvline(stage_boundary, color="#888", linestyle="--", linewidth=0.8)
        sample_ids = run.get("sample_ids")
        run_label = sample_ids[0] if sample_ids else "100 train images"
        axis.set_title(f"Run {run['id']}: {run_label}", fontsize=9)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("normalized disparity objective")
        axis.grid(alpha=0.2)
    for axis in axes[run_count:]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right")
    figure.tight_layout()
    figure.savefig(experiment / "artifacts" / "training_curve.png", dpi=180)
    plt.close(figure)


def export_comparison_figure(
    experiment: Path,
    samples: Sequence[HypersimDisparitySample],
    all_predictions: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
) -> None:
    figure, axes = plt.subplots(len(samples), 5, figsize=(15, 3 * len(samples)))
    columns = ("RGB", "GT disparity", "Initial K0", "Stage1 K3", "Joint K3")
    for row, sample in enumerate(samples):
        values = (
            sample.image.permute(1, 2, 0).numpy(),
            sample.target_disparity.numpy(),
            all_predictions[sample.sample_id]["initial"][0].numpy(),
            all_predictions[sample.sample_id]["stage1_best"][3].numpy(),
            all_predictions[sample.sample_id]["joint_best"][3].numpy(),
        )
        for column, value in enumerate(values):
            axis = axes[row, column]
            if column == 0:
                axis.imshow(np.clip(value, 0, 1))
            else:
                axis.imshow(value, cmap="magma", vmin=0, vmax=1)
            if row == 0:
                axis.set_title(columns[column])
            if column == 0:
                axis.set_ylabel(sample.sample_id, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(experiment / "artifacts" / "disparity_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config = load_json(experiment / "config.json")
    experiment_number = int(str(config["experiment_id"]).split("_", 1)[0][3:])
    experiment_tag = f"exp{experiment_number}"
    safe_root = Path(str(config["server"]["safe_root"]))
    ensure_within(experiment, safe_root, name="experiment")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Asset export requires the CUDA server")
    torch.cuda.set_device(device)
    samples = load_all_samples(config, project_root)
    checkpoint_path = ensure_within(
        Path(str(config["model"]["checkpoint"])), safe_root, name="base checkpoint"
    )
    model = InfiniDepth(model_path=str(checkpoint_path)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(config["model"]["voxel_resolution"]),
        max_disparity_span=config["model"].get("max_disparity_span"),
    )
    query_hw = (int(config["model"]["height"]), int(config["model"]["width"]))
    chunk_size = int(config["model"]["query_chunk_size"])
    initial_predictions = {
        sample.sample_id: predict(model, sample, device, query_hw, chunk_size)
        for sample in samples
    }
    initial_sha = sha256(checkpoint_path)
    all_predictions: Dict[str, Dict[str, Mapping[int, torch.Tensor]]] = {}
    viewer_samples = []
    public_root = project_root / "experiment" / "viewer" / "public"
    viewer_output = public_root / "data" / experiment_tag
    selections = {str(item["sample_id"]): item for item in config["structure_selections"]}
    if len(config["runs"]) == 1:
        run_by_sample = {sample.sample_id: config["runs"][0] for sample in samples}
    else:
        run_by_sample = {str(run["sample_ids"][0]): run for run in config["runs"]}
    for order, sample in enumerate(samples, start=1):
        run = run_by_sample[sample.sample_id]
        run_root = experiment / str(run["directory"])
        predictions: Dict[str, Mapping[int, torch.Tensor]] = {
            "initial": initial_predictions[sample.sample_id]
        }
        shas = {"initial": initial_sha}
        for stage, filename in (
            ("stage1_best", "stage1_best.pt"),
            ("joint_best", "joint_best.pt"),
        ):
            path = run_root / "checkpoints" / filename
            shas[stage] = load_refiner_checkpoint(model, path)
            predictions[stage] = predict(model, sample, device, query_hw, chunk_size)
        all_predictions[sample.sample_id] = predictions
        selection = selections[sample.sample_id]
        viewer_samples.append(
            export_sample_assets(
                sample=sample,
                predictions=predictions,
                checkpoint_shas=shas,
                output=viewer_output,
                public_root=public_root,
                order=order,
                description=str(selection["description"]),
                crop_xyxy=selection["crop_xyxy"],
            )
        )
    write_json(
        viewer_output / "manifest.json",
        {
            "availableStages": ["initial", "stage1_best", "joint_best"],
            "coordinateSpace": {
                "aligned": "metric camera XYZ",
                "stored": "metric camera XYZ",
                "threeDisplay": "[x, -y, -z]",
            },
            "defaultStages": {"left": "initial", "right": "joint_best"},
            "displayNote": "预测点云使用每张 GT 的 2%/98% disparity 统计反归一化，不是模型原生米制输出。",
            "experiment": config["experiment_id"],
            "resolution": {"height": query_hw[0], "width": query_hw[1]},
            "samples": viewer_samples,
            "stages": ["initial", "stage1_best", "joint_best"],
            "steps": list(STEPS),
            "websiteSampleOrder": [sample.sample_id for sample in samples],
            "voxelization": {
                "depthCoordinate": "round(200 * normalized disparity)",
                "depthScale": 200.0,
                "spconvOrder": ["batch", "disparity_bin", "row", "column"]
            },
            "version": 1,
        },
    )
    catalog_path = public_root / "data" / "experiments.json"
    if catalog_path.is_file():
        catalog = dict(load_json(catalog_path))
    else:
        catalog = {"defaultExperiment": str(config["experiment_id"]), "experiments": [], "version": 1}
    entries = [
        entry for entry in catalog["experiments"]
        if entry["id"] != config["experiment_id"]
    ]
    entries.append(
        {
            "id": config["experiment_id"],
            "label": f"Exp{experiment_number}",
            "manifestUrl": f"/data/{experiment_tag}/manifest.json",
            "shortLabel": f"Exp{experiment_number}",
            "stageDetails": {
                "initial": "官方 InfiniDepth 初始预测",
                "stage1_best": "仅训练 Detach Refiner 的最佳检查点",
                "joint_best": "联合训练的最佳检查点",
            },
            "stageLabels": {
                "initial": "官方初始",
                "stage1_best": "Detach 最佳",
                "joint_best": "联合最佳",
            },
            "summary": config["research_question"],
        }
    )
    catalog["experiments"] = sorted(entries, key=lambda entry: int(str(entry["label"])[3:]))
    write_json(catalog_path, catalog)
    export_training_curve(experiment, config)
    export_comparison_figure(experiment, samples, all_predictions)


if __name__ == "__main__":
    main()

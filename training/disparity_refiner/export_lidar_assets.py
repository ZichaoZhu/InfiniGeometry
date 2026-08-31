from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.data import (
    HypersimDisparitySample,
    ensure_within,
    load_manifest,
    preload_samples,
    select_manifest_entries,
)
from training.disparity_refiner.export_assets import (
    asset_record,
    public_url,
    sha256,
    write_json,
    write_point_cloud,
)
from training.disparity_refiner.lidar import (
    camera_rays,
    normalized_disparity_to_geometry,
    stack_lidar_batch,
)
from training.disparity_refiner.losses import point_cloud_metrics
from training.disparity_refiner.train_lidar import CHECKPOINT_FORMAT


STEPS = (0, 1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Exp4 LiDAR point-cloud viewer assets")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_samples(
    config: Mapping[str, object],
    viewer_config: Mapping[str, object],
    project_root: Path,
) -> Sequence[HypersimDisparitySample]:
    manifest_path = project_root / str(config["data"]["manifest"])
    manifest = load_manifest(manifest_path, str(config["data"]["manifest_sha256"]))
    source_root = Path(str(config["data"]["source_root"]))
    sample_specs = [
        {"sample_id": str(item["sample_id"]), "split": str(item["split"])}
        for item in viewer_config["samples"]
    ]
    if len({(item["split"], item["sample_id"]) for item in sample_specs}) != len(sample_specs):
        raise ValueError("Viewer sample split/ID pairs must be unique")
    entries_by_key = {}
    for split in dict.fromkeys(item["split"] for item in sample_specs):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported viewer split: {split}")
        sample_ids = [item["sample_id"] for item in sample_specs if item["split"] == split]
        for entry in select_manifest_entries(
            manifest, source_root=source_root, split=split, sample_ids=sample_ids
        ):
            entries_by_key[(split, str(entry["id"]))] = entry
    entries = [
        entries_by_key[(item["split"], item["sample_id"])] for item in sample_specs
    ]
    selections = {
        str(entry["sample_id"]): entry
        for entry in viewer_config.get("structure_selections", [])
    }
    cache_root = (
        None
        if config["data"].get("local_cache") is None
        else Path(str(config["data"]["local_cache"]))
    )
    model_cfg = config["model"]
    return preload_samples(
        entries,
        source_root=source_root,
        height=int(model_cfg["height"]),
        width=int(model_cfg["width"]),
        structure_selections=selections,
        cache_root=cache_root,
    )


def load_refiner_checkpoint(model: InfiniDepth_DepthSensor, path: Path) -> str:
    checkpoint_file = path / "checkpoint.pt" if path.is_dir() else path
    checkpoint = torch.load(
        checkpoint_file, map_location=next(model.parameters()).device, weights_only=True
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported LiDAR refiner checkpoint: {path}")
    assert model.disparity_refiner is not None
    model.disparity_refiner.load_state_dict(checkpoint["refiner"], strict=True)
    return sha256(checkpoint_file)


@torch.no_grad()
def predict(
    model: InfiniDepth_DepthSensor,
    sample: HypersimDisparitySample,
    config: Mapping[str, object],
    device: torch.device,
) -> tuple[Mapping[int, torch.Tensor], Mapping[int, torch.Tensor]]:
    model.eval()
    batch = stack_lidar_batch(
        [sample], config["lidar"], seed=int(config["seed"]), step=0, device=device
    )
    model_cfg = config["model"]
    output = model.forward_dense_refined(
        batch.image,
        prompt_disparity=batch.prompt_disparity,
        prompt_mask=batch.prompt_mask,
        query_hw=(int(model_cfg["height"]), int(model_cfg["width"])),
        num_refinement_steps=max(STEPS),
        detach_base_from_refiner=True,
        chunk_size=int(model_cfg["query_chunk_size"]),
    )
    rays = camera_rays(
        sample.metadata,
        sample.radial_depth.shape[0],
        sample.radial_depth.shape[1],
        device=device,
    )
    disparities = {}
    points = {}
    for iteration in STEPS:
        disparity = output.disparity_sequence[iteration][0]
        _, geometry, _ = normalized_disparity_to_geometry(
            disparity, batch.reference_scale[0], rays
        )
        disparities[iteration] = disparity.detach().cpu()
        points[iteration] = geometry.detach().cpu()
    return disparities, points


def export_sample_assets(
    *,
    sample: HypersimDisparitySample,
    predictions: Mapping[str, Mapping[int, torch.Tensor]],
    point_maps: Mapping[str, Mapping[int, torch.Tensor]],
    checkpoint_shas: Mapping[str, str],
    output: Path,
    public_root: Path,
    order: int,
    split: str,
    split_order: int,
    description: str,
    crop_xyxy: Sequence[int],
) -> Mapping[str, object]:
    colors = np.round(
        sample.image.permute(1, 2, 0).numpy().clip(0, 1) * 255
    ).astype(np.uint8)
    sample_dir = output / sample.sample_id
    rgb_path = sample_dir / "source_rgb.jpg"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colors).save(rgb_path, quality=92)
    rays = camera_rays(
        sample.metadata, sample.radial_depth.shape[0], sample.radial_depth.shape[1]
    ).numpy()
    gt_points = rays * sample.radial_depth.numpy()[..., None]
    gt_points[~sample.valid_mask.numpy()] = np.nan
    gt_tensor = torch.from_numpy(gt_points)
    structure_mask = (
        sample.structure_mask
        if sample.structure_mask is not None
        else sample.valid_mask
    )
    gt_path = sample_dir / "ground_truth.ply"
    gt_meta = write_point_cloud(gt_path, gt_points, colors)
    ground_truth = asset_record(
        gt_path,
        public_root,
        gt_meta,
        point_cloud_metrics(gt_tensor, gt_tensor, sample.valid_mask, structure_mask),
        "ground-truth",
    )
    stages = {}
    for stage, maps in point_maps.items():
        stage_assets = {}
        for iteration in STEPS:
            if stage == "initial" and iteration:
                stage_assets[f"k{iteration}"] = {"alias": "initial.k0"}
                continue
            points = maps[iteration].numpy()
            path = sample_dir / f"{stage}_k{iteration}.ply"
            metadata = write_point_cloud(path, points, colors)
            stage_assets[f"k{iteration}"] = asset_record(
                path,
                public_root,
                metadata,
                point_cloud_metrics(
                    maps[iteration], gt_tensor, sample.valid_mask, structure_mask
                ),
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
        "label": f"{split.title()} {split_order:02d}",
        "order": order,
        "rgbUrl": public_url(rgb_path, public_root),
        "split": split,
        "stages": stages,
        "websiteEnabled": True,
    }


def export_training_curve(experiment: Path, run_root: Path) -> None:
    history_path = run_root / "metrics" / "history.jsonl"
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    full = [record for record in records if record.get("scope") == "full"]
    if not full:
        raise ValueError("Exp4 history has no Val100 full evaluation records")
    figure, (evaluation_axis, loss_axis) = plt.subplots(1, 2, figsize=(13, 3.8))
    colors = {0: "#65717e", 1: "#2f6b4f", 3: "#c4473a", 5: "#9b6a35"}
    steps = [int(record["step"]) for record in full]
    for iteration in STEPS:
        values = [
            float(record["evaluation"]["aggregate"][f"k{iteration}"]["metric_disparity_mae_1_per_m"])
            for record in full
        ]
        evaluation_axis.plot(
            steps,
            values,
            color=colors[iteration],
            label=f"K{iteration}",
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
    evaluation_axis.set_title("Val100 full evaluation (every 2,500 steps)", fontsize=9)
    evaluation_axis.set_ylabel("metric disparity MAE (1/m)")
    evaluation_axis.set_xlabel("optimizer step")
    evaluation_axis.legend()
    evaluation_axis.grid(alpha=0.2)

    all_steps = [int(record["step"]) for record in records]
    losses = [float(record["training"]["loss"]) for record in records]
    loss_axis.plot(all_steps, losses, label="instantaneous loss", color="#2f6b4f", linewidth=1.2)
    loss_axis.set_title("batch-8 instantaneous training loss (every 500 steps)", fontsize=9)
    loss_axis.set_ylabel("training loss")
    loss_axis.set_xlabel("optimizer step")
    loss_axis.legend()
    loss_axis.grid(alpha=0.2)
    figure.tight_layout()
    artifact = experiment / "artifacts" / "training_curve_full_eval.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(artifact, dpi=180)
    plt.close(figure)


def export_comparison_figure(
    experiment: Path,
    samples: Sequence[HypersimDisparitySample],
    predictions: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
) -> None:
    figure, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))
    columns = ("RGB", "GT disparity", "LiDAR K0", "SSR K3 (best step 22,500)")
    for row, sample in enumerate(samples):
        values = (
            sample.image.permute(1, 2, 0).numpy(),
            sample.target_disparity.numpy(),
            predictions[sample.sample_id]["initial"][0].numpy(),
            predictions[sample.sample_id]["best_step_22500"][3].numpy(),
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
    artifact = experiment / "artifacts" / "disparity_comparison_viewer.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(artifact, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config = load_json(experiment / "config.json")
    viewer_config = load_json(experiment / "viewer.json")
    if str(viewer_config.get("format")) != "infinidepth-lidar-refiner-viewer-v1":
        raise ValueError("Unsupported Exp4 viewer configuration")
    asset_tag = str(viewer_config["asset_tag"])
    if not asset_tag.replace("_", "").isalnum() or not asset_tag.startswith("exp4"):
        raise ValueError(f"Invalid viewer asset tag: {asset_tag}")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Point-cloud export requires a CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    safe_root = Path(str(config["server"]["safe_root"]))
    output_root = ensure_within(
        Path(str(config["server"]["output_root"])), safe_root, name="primary output root"
    )
    base_checkpoint = ensure_within(
        Path(str(config["model"]["checkpoint"])), safe_root, name="base checkpoint"
    )
    checkpoint_spec = viewer_config["checkpoint"]
    checkpoint = ensure_within(
        output_root / str(checkpoint_spec["path"]), output_root, name="viewer checkpoint"
    )
    checkpoint_file = checkpoint / "checkpoint.pt" if checkpoint.is_dir() else checkpoint
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    public_root = project_root / "experiment" / "viewer" / "public"
    viewer_output = public_root / "data" / asset_tag
    if viewer_output.exists():
        raise FileExistsError(f"Viewer asset tag already exists: {viewer_output}")
    samples = load_all_samples(config, viewer_config, project_root)
    model = InfiniDepth_DepthSensor(model_path=str(base_checkpoint)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(config["model"]["voxel_resolution"]),
        max_disparity_span=config["model"].get("max_disparity_span"),
    )
    initial_sha = sha256(base_checkpoint)
    all_predictions: Dict[str, Dict[str, Mapping[int, torch.Tensor]]] = {}
    all_points: Dict[str, Dict[str, Mapping[int, torch.Tensor]]] = {}
    viewer_samples = []
    split_counts: Dict[str, int] = {}
    selections = {
        str(item["sample_id"]): item
        for item in viewer_config.get("structure_selections", [])
    }
    checkpoint_sha = load_refiner_checkpoint(model, checkpoint_file)
    for order, sample in enumerate(samples, start=1):
        best_disparities, best_points = predict(model, sample, config, device)
        # K=0 is emitted before SSR, so it is identical for the frozen base and best SSR.
        initial_disparities = {0: best_disparities[0]}
        initial_points = {0: best_points[0]}
        all_predictions[sample.sample_id] = {
            "initial": initial_disparities,
            "best_step_22500": best_disparities,
        }
        all_points[sample.sample_id] = {
            "initial": initial_points,
            "best_step_22500": best_points,
        }
        split = str(sample.metadata["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        selection = selections.get(sample.sample_id)
        viewer_samples.append(
            export_sample_assets(
                sample=sample,
                predictions=all_predictions[sample.sample_id],
                point_maps=all_points[sample.sample_id],
                checkpoint_shas={"initial": initial_sha, "best_step_22500": checkpoint_sha},
                output=viewer_output,
                public_root=public_root,
                order=order,
                split=split,
                split_order=split_counts[split],
                description=(
                    str(selection["description"])
                    if selection is not None
                    else f"{split.upper()} 随机样本（seed {viewer_config['sample_policy']['seed']}）"
                ),
                crop_xyxy=(
                    selection["crop_xyxy"]
                    if selection is not None
                    else [0, 0, int(config["model"]["width"]), int(config["model"]["height"])]
                ),
            )
        )
    write_json(
        viewer_output / "manifest.json",
        {
            "availableStages": ["initial", "best_step_22500"],
            "coordinateSpace": {
                "aligned": "metric camera XYZ",
                "stored": "metric camera XYZ",
                "threeDisplay": "[x, -y, -z]",
            },
            "defaultStages": {"left": "initial", "right": "best_step_22500"},
            "displayNote": "预测点云由同一张图的稀疏 LiDAR prompt 中位 disparity 还原为米制径向深度；K0 是冻结的 LiDAR-conditioned DepthSensor 输出。",
            "experiment": config["experiment_id"],
            "resolution": {
                "height": int(config["model"]["height"]),
                "width": int(config["model"]["width"]),
            },
            "samplePolicy": {
                "algorithm": viewer_config["sample_policy"]["algorithm"],
                "countPerSplit": viewer_config["sample_policy"]["count_per_split"],
                "seed": viewer_config["sample_policy"]["seed"],
                "splitOrder": viewer_config["sample_policy"]["split_order"],
                "testUsage": viewer_config["sample_policy"]["test_usage"],
            },
            "samples": viewer_samples,
            "stages": ["initial", "best_step_22500"],
            "steps": list(STEPS),
            "websiteSampleOrder": [sample.sample_id for sample in samples],
            "voxelization": {
                "depthCoordinate": "round(200 * LiDAR-normalized disparity)",
                "depthScale": 200.0,
                "spconvOrder": ["batch", "disparity_bin", "row", "column"],
            },
            "version": 1,
        },
    )
    catalog_path = public_root / "data" / "experiments.json"
    catalog = dict(load_json(catalog_path)) if catalog_path.is_file() else {
        "defaultExperiment": str(config["experiment_id"]),
        "experiments": [],
        "version": 1,
    }
    entries = [
        entry for entry in catalog["experiments"] if entry["id"] != config["experiment_id"]
    ]
    entries.append(
        {
            "id": config["experiment_id"],
            "label": "Exp4",
            "manifestUrl": f"/data/{asset_tag}/manifest.json",
            "shortLabel": "Exp4",
            "stageDetails": {
                "initial": "冻结 LiDAR-conditioned DepthSensor 的原始输出",
                "best_step_22500": "Val100 K3 metric disparity MAE 最优的 SSR checkpoint（step 22,500）",
            },
            "stageLabels": {"initial": "LiDAR 初始", "best_step_22500": "SSR 最佳 22.5k"},
            "summary": config["research_question"],
        }
    )
    catalog["experiments"] = sorted(entries, key=lambda entry: int(str(entry["label"])[3:]))
    write_json(catalog_path, catalog)
    run_root = output_root / str(checkpoint_spec["run_path"])
    export_training_curve(experiment, run_root)
    export_comparison_figure(experiment, samples, all_predictions)


if __name__ == "__main__":
    main()

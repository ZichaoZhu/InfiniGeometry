from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from InfiniDepth.model import InfiniDepth_DepthSensor
from training.disparity_refiner.backup import (
    BackupError,
    RunBackup,
    sha256,
    verify_checkpoint_directory,
)
from training.disparity_refiner.data import (
    HypersimDisparityDataset,
    HypersimDisparitySample,
    ensure_within,
)
from training.disparity_refiner.lidar import (
    camera_rays,
    load_segment_masks,
    local_point_metrics,
    normalized_disparity_to_geometry,
    stack_lidar_batch,
)
from training.disparity_refiner.losses import (
    disparity_metrics,
    refiner_iteration_loss,
)
from training.disparity_refiner.train import (
    EVALUATION_STEPS,
    ShuffledCycleSampler,
    _append_jsonl,
    _atomic_json,
    _load_samples,
    _refinement_monitor,
    _selected_run,
    _sha256,
)


CHECKPOINT_FORMAT = "infinidepth-lidar-refiner-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Exp4 LiDAR-conditioned refiner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="main")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-max-steps", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_config(config: Mapping[str, object], *, smoke: bool) -> None:
    model = config["model"]
    training = config["training"]
    data = config["data"]
    if model.get("variant") != "InfiniDepth_DepthSensor":
        raise ValueError("Exp4 requires model.variant=InfiniDepth_DepthSensor")
    if data.get("sample_ids") is not None:
        raise ValueError("Exp4 must train on the full configured train split")
    for run in config.get("runs", []):
        if run.get("sample_ids") is not None:
            raise ValueError("Exp4 runs may not restrict the train split")
    if int(data["expected_train_count"]) != 59542:
        raise ValueError("Exp4 expects all 59,542 valid Hypersim train samples")
    if not smoke and int(training["checkpoint_every"]) != 500:
        raise ValueError("Exp4 formal checkpoints must be saved every 500 steps")
    if int(training["global_batch_size"]) % int(training["microbatch_size"]):
        raise ValueError("global_batch_size must be divisible by microbatch_size")
    if int(training["max_steps"]) <= 0:
        raise ValueError("training.max_steps must be positive")
    local = config["evaluation"]["local_points"]
    if list(local["residual_scales"]) != [8, 16, 32]:
        raise ValueError("MoGe-3 residual scales must be [8,16,32]")
    if list(local["morphology_sizes"]) != [3, 5, 9, 17]:
        raise ValueError("MoGe-3 morphology sizes must be [3,5,9,17]")
    if float(local["delta_threshold"]) != 0.01 or int(local["min_segment_pixels"]) != 10:
        raise ValueError("MoGe-3 local evaluation requires delta=0.01 and >=10 pixels")
    server = config["server"]
    if Path(str(server["output_root"])) == Path(str(server["backup_root"])):
        raise ValueError("Primary and backup roots must differ")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng(sampler: ShuffledCycleSampler) -> Dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "sampling": sampler.getstate(),
    }


def _restore_rng(value: Mapping[str, object], sampler: ShuffledCycleSampler) -> None:
    random.setstate(value["python"])
    numpy_state = value["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            numpy_state[1].cpu().numpy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(value["torch_cpu"].cpu())
    cuda_state = value.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state.cpu())
    sampler.setstate(value["sampling"])


def _freeze_base(model: InfiniDepth_DepthSensor) -> Sequence[torch.nn.Parameter]:
    if model.disparity_refiner is None:
        raise RuntimeError("Disparity refiner is not attached")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.disparity_refiner.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _parameter_checksum(parameters: Sequence[torch.nn.Parameter]) -> tuple[float, float]:
    with torch.no_grad():
        total = sum(float(parameter.detach().double().sum().item()) for parameter in parameters)
        squares = sum(
            float(parameter.detach().double().square().sum().item()) for parameter in parameters
        )
    return total, squares


def _base_parameters(model: InfiniDepth_DepthSensor) -> Sequence[torch.nn.Parameter]:
    if model.disparity_refiner is None:
        return list(model.parameters())
    refiner_ids = {id(parameter) for parameter in model.disparity_refiner.parameters()}
    return [parameter for parameter in model.parameters() if id(parameter) not in refiner_ids]


def _checkpoint_file(path: Path) -> Path:
    return path / "checkpoint.pt" if path.is_dir() else path


def save_checkpoint(
    checkpoint_root: Path,
    *,
    model: InfiniDepth_DepthSensor,
    optimizer: torch.optim.Optimizer,
    sampler: ShuffledCycleSampler,
    step: int,
    config_sha256: str,
    base_checkpoint_sha256: str,
    evaluation: Mapping[str, object],
    elapsed_seconds: float,
) -> Path:
    if model.disparity_refiner is None:
        raise RuntimeError("Disparity refiner is not attached")
    final = checkpoint_root / f"step_{int(step):09d}"
    if final.exists():
        raise FileExistsError(f"Immutable checkpoint already exists: {final}")
    temporary = checkpoint_root / f".{final.name}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_scope": "disparity_refiner",
        "refiner": model.disparity_refiner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": _capture_rng(sampler),
        "step": int(step),
        "config_sha256": config_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "evaluation": dict(evaluation),
        "elapsed_seconds": float(elapsed_seconds),
    }
    checkpoint_path = temporary / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    digest = _sha256(checkpoint_path)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "checkpoint": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": digest,
        "model_scope": "disparity_refiner",
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "config_sha256": config_sha256,
        "created_at_unix": time.time(),
    }
    metadata_path = temporary / "metadata.json"
    _atomic_json(metadata_path, metadata)
    (temporary / "SHA256SUMS").write_text(
        f"{digest}  checkpoint.pt\n{_sha256(metadata_path)}  metadata.json\n",
        encoding="ascii",
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    _atomic_json(
        checkpoint_root.parent / "latest.json",
        {
            "format": CHECKPOINT_FORMAT,
            "step": int(step),
            "path": f"checkpoints/{final.name}",
        },
    )
    return final


def restore_checkpoint(
    path: Path,
    *,
    model: InfiniDepth_DepthSensor,
    optimizer: torch.optim.Optimizer,
    sampler: ShuffledCycleSampler,
    config_sha256: str,
    base_checkpoint_sha256: str,
) -> Mapping[str, object]:
    if model.disparity_refiner is None:
        raise RuntimeError("Disparity refiner is not attached")
    checkpoint_path = _checkpoint_file(path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported LiDAR refiner checkpoint format")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("Resume checkpoint was produced by another config")
    if checkpoint.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("Resume checkpoint expects another DepthSensor base")
    model.disparity_refiner.load_state_dict(checkpoint["refiner"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng(checkpoint["rng"], sampler)
    return checkpoint


def _copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise FileExistsError(f"Archived input differs from source: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _archive_inputs(
    output: Path,
    config_path: Path,
    config: Mapping[str, object],
    project_root: Path,
    base_checkpoint: Path,
    local_mask_manifest: Path,
    *,
    smoke: bool,
    resume_step: Optional[int] = None,
) -> Path:
    inputs = output / "inputs"
    _copy_once(config_path, inputs / "config.json")
    manifest = ensure_within(
        project_root / str(config["data"]["manifest"]), project_root, name="manifest"
    )
    _copy_once(manifest, inputs / manifest.name)
    sample_ids_config = config["evaluation"].get("sample_ids_config")
    if sample_ids_config is not None:
        source = ensure_within(
            project_root / str(sample_ids_config),
            project_root,
            name="evaluation sample config",
        )
        _copy_once(source, inputs / "evaluation_sample_ids_config.json")
    _copy_once(local_mask_manifest, inputs / "local_mask_manifest.json")
    if not smoke:
        _copy_once(base_checkpoint, inputs / "base_checkpoint.pt")
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("backup.py"),
        Path(__file__).resolve().with_name("lidar.py"),
        Path(__file__).resolve().with_name("prepare_local_masks.py"),
        Path(__file__).resolve().with_name("losses.py"),
        Path(__file__).resolve().with_name("data.py"),
        Path(__file__).resolve().with_name("train.py"),
        project_root / "InfiniDepth/model/model.py",
        project_root / "InfiniDepth/model/disparity_refiner.py",
        project_root / "InfiniDepth/utils/warp_utils.py",
    ]
    source_archive = (
        inputs
        if resume_step is None
        else inputs / f"resume_step_{int(resume_step):09d}"
    )
    for path in source_paths:
        _copy_once(path, source_archive / "source" / path.relative_to(project_root))
    source_manifest = source_archive / "source_manifest.json"
    _atomic_json(
        source_manifest,
        {
            "format": "infinidepth-source-manifest-v1",
            "files": {
                str(path.relative_to(project_root)): _sha256(path) for path in source_paths
            },
        },
    )
    return source_manifest


def _latest_local_checkpoint(output: Path) -> Optional[tuple[Path, int]]:
    latest_path = output / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = ensure_within(
        output / str(latest["path"]), output, name="latest local checkpoint"
    )
    verify_checkpoint_directory(checkpoint)
    return checkpoint, int(latest["step"])


def _backup_on_exit(
    backup: RunBackup,
    output: Path,
    report_path: Path,
    report: Mapping[str, object],
    *,
    step: int,
    status: str,
) -> None:
    _atomic_json(report_path, report)
    try:
        latest = _latest_local_checkpoint(output)
        if latest is not None:
            checkpoint, checkpoint_step = latest
            backup.backup_checkpoint(checkpoint, step=checkpoint_step)
        backup.finish(report_path, report, step=step, status=status)
    except Exception as exc:
        _atomic_json(
            report_path,
            {**report, "backup_status": "failed", "backup_error": str(exc)},
        )
        if isinstance(exc, BackupError):
            raise
        raise BackupError(f"Exit backup failed: {exc}") from exc


def _require_complete_local_cache(samples: Sequence[HypersimDisparitySample]) -> None:
    if not isinstance(samples, HypersimDisparityDataset) or samples.cache_root is None:
        raise ValueError("Formal Exp4 requires a lazy dataset with a local cache")
    missing = []
    for entry in samples.entries:
        for kind in ("rgb", "depth"):
            source = Path(str(entry[kind]["source"]))
            try:
                relative = source.relative_to(samples.source_root)
            except ValueError as exc:
                raise PermissionError(f"{kind} source is outside the Hypersim root") from exc
            cached = samples.cache_root / relative
            if not cached.is_file() or cached.is_symlink():
                missing.append(str(cached))
                if len(missing) == 5:
                    break
        if len(missing) == 5:
            break
    if missing:
        raise FileNotFoundError(
            "Exp4 refuses NAS fallback during training; local cache is incomplete: "
            + ", ".join(missing)
        )


def _local_mask_index(
    root: Path,
    sample_ids: Sequence[str],
    *,
    require_complete: bool,
) -> tuple[Path, Mapping[str, Mapping[str, object]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "infinidepth-moge3-local-mask-manifest-v1":
        raise ValueError(f"Unsupported local-mask manifest: {manifest_path}")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError(f"Local-mask cache is not complete: {manifest_path}")
    entries = manifest.get("samples", {})
    missing = sorted(set(sample_ids) - set(entries))
    if missing:
        raise ValueError(f"Local-mask cache lacks evaluation samples: {missing[:5]}")
    selected = {sample_id: entries[sample_id] for sample_id in sample_ids}
    for sample_id, value in selected.items():
        path = ensure_within(root / str(value["path"]), root, name=f"{sample_id} mask")
        if sha256(path) != str(value["sha256"]):
            raise ValueError(f"Local-mask SHA-256 mismatch: {sample_id}")
    return manifest_path, selected


def _mean_metrics(
    per_image: Mapping[str, Mapping[str, object]], iterations: Sequence[int]
) -> Dict[str, object]:
    aggregate: Dict[str, object] = {}
    for iteration in iterations:
        key = f"k{iteration}"
        names = set().union(*(value[key].keys() for value in per_image.values()))
        summary: Dict[str, float] = {}
        for name in names:
            if name.startswith("_local_") or name in {
                "local_point_rel",
                "local_point_delta_0_01",
                "local_segment_count",
            }:
                continue
            values = [value[key][name] for value in per_image.values()]
            finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
            if finite:
                summary[name] = float(np.mean(finite))
        segment_count = sum(float(value[key]["local_segment_count"]) for value in per_image.values())
        summary["local_segment_count"] = segment_count
        if segment_count:
            summary["local_point_rel"] = sum(
                float(value[key]["_local_point_rel_sum"]) for value in per_image.values()
            ) / segment_count
            summary["local_point_delta_0_01"] = sum(
                float(value[key]["_local_point_delta_0_01_sum"])
                for value in per_image.values()
            ) / segment_count
        aggregate[key] = summary
    return aggregate


@torch.no_grad()
def evaluate(
    model: InfiniDepth_DepthSensor,
    samples: Sequence[HypersimDisparitySample],
    *,
    device: torch.device,
    config: Mapping[str, object],
    local_mask_root: Path,
    local_mask_entries: Mapping[str, Mapping[str, object]],
    iterations: Sequence[int] = EVALUATION_STEPS,
) -> Dict[str, object]:
    model.eval()
    model_cfg = config["model"]
    metric_cfg = config["evaluation"]["local_points"]
    per_image: Dict[str, Mapping[str, object]] = {}
    for sample in samples:
        mask_entry = local_mask_entries[sample.sample_id]
        segment_masks = torch.from_numpy(
            load_segment_masks(local_mask_root / str(mask_entry["path"]))
        ).to(device)
        batch = stack_lidar_batch(
            [sample], config["lidar"], seed=int(config["seed"]), step=0, device=device
        )
        output = model.forward_dense_refined(
            batch.image,
            prompt_disparity=batch.prompt_disparity,
            prompt_mask=batch.prompt_mask,
            query_hw=(int(model_cfg["height"]), int(model_cfg["width"])),
            num_refinement_steps=max(iterations),
            detach_base_from_refiner=True,
            chunk_size=int(model_cfg["query_chunk_size"]),
        )
        scale = batch.reference_scale[0]
        rays = camera_rays(
            sample.metadata,
            sample.radial_depth.shape[0],
            sample.radial_depth.shape[1],
            device=device,
        )
        target_depth = sample.radial_depth.to(device)
        if tuple(segment_masks.shape[1:]) != tuple(target_depth.shape):
            segment_masks = F.interpolate(
                segment_masks[:, None].float(),
                size=target_depth.shape,
                mode="nearest",
            )[:, 0].bool()
        target_points = rays * target_depth[..., None]
        raw_target_disparity = torch.where(
            batch.valid_mask[0], target_depth.reciprocal(), torch.zeros_like(target_depth)
        )
        per_k: Dict[str, object] = {}
        for iteration in iterations:
            prediction = output.disparity_sequence[iteration][0]
            radial, points, prediction_valid = normalized_disparity_to_geometry(
                prediction, scale, rays
            )
            valid = batch.valid_mask[0] & prediction_valid
            normalized = disparity_metrics(
                prediction, batch.target_disparity[0], valid
            )
            metric_disparity_mae = float(
                ((prediction * scale - raw_target_disparity).abs()[valid]).mean().item()
            )
            depth_abs_rel = float(
                ((radial - target_depth).abs()[valid] / target_depth[valid]).mean().item()
            )
            geometry = local_point_metrics(
                points,
                target_points,
                valid,
                segment_masks,
                min_segment_pixels=int(metric_cfg["min_segment_pixels"]),
                delta_threshold=float(metric_cfg["delta_threshold"]),
            )
            per_k[f"k{iteration}"] = {
                **normalized,
                "metric_disparity_mae_1_per_m": metric_disparity_mae,
                "radial_depth_abs_rel": depth_abs_rel,
                **geometry,
            }
        per_image[sample.sample_id] = per_k
    aggregate = _mean_metrics(per_image, iterations)
    k3_better = sum(
        float(value["k3"]["metric_disparity_mae_1_per_m"])
        < float(value["k0"]["metric_disparity_mae_1_per_m"])
        for value in per_image.values()
    ) if 0 in iterations and 3 in iterations else 0
    local_rel_better = sum(
        value["k3"]["local_point_rel"] is not None
        and value["k0"]["local_point_rel"] is not None
        and float(value["k3"]["local_point_rel"])
        < float(value["k0"]["local_point_rel"])
        for value in per_image.values()
    ) if 0 in iterations and 3 in iterations else 0
    return {
        "format": "infinidepth-lidar-evaluation-v2",
        "metric_semantics": {
            "metric_disparity_mae_1_per_m": "MAE of reciprocal radial range in 1/m",
            "radial_depth_abs_rel": "absolute relative error of radial range",
            "local_point_rel": "MoGe-3 v2 point AbsRel after global scale and per-segment 3D translation",
            "local_point_delta_0_01": "fraction with 3D error below 0.01 times min aligned/GT point radius",
            "local_aggregation": "macro average over all retained segments",
        },
        "sample_count": len(samples),
        "aggregate": aggregate,
        "per_image": per_image,
        "k3_better_than_k0_count": int(k3_better),
        "k3_local_point_rel_better_than_k0_count": int(local_rel_better),
    }


def _selection_key(evaluation: Mapping[str, object]) -> float:
    k3 = evaluation["aggregate"]["k3"]
    return float(k3["metric_disparity_mae_1_per_m"])


def _evaluation_sample_ids(
    config: Mapping[str, object], project_root: Path
) -> Sequence[str]:
    evaluation = config["evaluation"]
    configured = evaluation.get("full_sample_ids")
    if configured is not None:
        return [str(value) for value in configured]
    source_path = ensure_within(
        project_root / str(evaluation["sample_ids_config"]),
        project_root,
        name="evaluation sample config",
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return [str(value) for value in source["evaluation"]["full_sample_ids"]]


def _sample_values(
    samples: Sequence[HypersimDisparitySample], indices: Sequence[int]
) -> Sequence[HypersimDisparitySample]:
    return [samples[index] for index in indices]


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Exp4 training and smoke tests require CUDA")
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, smoke=bool(args.smoke))
    project_root = Path(__file__).resolve().parents[2]
    config_path = ensure_within(config_path, project_root, name="config")
    safe_root = Path(str(config["server"]["safe_root"]))
    output_root = ensure_within(
        Path(str(config["server"]["output_root"])), safe_root, name="primary output root"
    )
    backup_safe_root = Path(str(config["server"]["backup_safe_root"]))
    backup_root = ensure_within(
        Path(str(config["server"]["backup_root"])), backup_safe_root, name="backup root"
    )
    output = ensure_within(args.output, output_root, name="experiment output")
    backup_output = ensure_within(
        backup_root / output.relative_to(output_root), backup_root, name="experiment backup"
    )
    if output.is_symlink():
        raise PermissionError("Experiment output may not be a symlink")
    if backup_output.is_symlink():
        raise PermissionError("Experiment backup may not be a symlink")
    if output.exists() and args.resume is None:
        existing = list(output.iterdir())
        log_only = len(existing) == 1 and existing[0] == output / "logs" and all(
            path.name == "train.log" for path in existing[0].iterdir()
        )
        if existing and not log_only:
            raise FileExistsError(f"A fresh run requires an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_config = _selected_run(config, args.run_id)
    seed = int(config["seed"]) + int(run_config.get("seed_offset", 0))
    _seed_everything(seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = ensure_within(
        Path(str(config["model"]["checkpoint"])), safe_root, name="base checkpoint"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    base_sha = _sha256(checkpoint_path)
    expected_base_sha = str(config["model"]["checkpoint_sha256"])
    if base_sha != expected_base_sha:
        raise ValueError(f"DepthSensor checkpoint SHA-256 mismatch: {base_sha}")
    config_sha = _sha256(config_path)

    if args.smoke:
        config["model"]["height"] = 96
        config["model"]["width"] = 128
        config["model"]["query_chunk_size"] = 2048
        smoke_ids = [str(config["evaluation"]["sample_ids"][0])]
        samples = _load_samples(
            config, run_config, project_root, split="val", sample_ids=smoke_ids
        )
        evaluation_samples = samples
        max_steps = int(args.smoke_max_steps)
        checkpoint_every = eval_every = full_eval_every = 1
    else:
        samples = _load_samples(config, run_config, project_root)
        evaluation_samples = _load_samples(
            config,
            run_config,
            project_root,
            split=str(config["evaluation"]["split"]),
            sample_ids=_evaluation_sample_ids(config, project_root),
        )
        max_steps = int(config["training"]["max_steps"])
        checkpoint_every = int(config["training"]["checkpoint_every"])
        eval_every = int(config["training"]["eval_every"])
        full_eval_every = int(config["training"]["full_eval_every"])
        if bool(config["data"].get("require_complete_local_cache", False)):
            _require_complete_local_cache(samples)
            _require_complete_local_cache(evaluation_samples)
    if max_steps <= 0 or checkpoint_every <= 0 or eval_every <= 0 or full_eval_every <= 0:
        raise ValueError("Step intervals must be positive")
    if checkpoint_every % eval_every or full_eval_every % eval_every:
        raise ValueError("checkpoint/full-eval intervals must be multiples of eval_every")
    evaluation_ids = [sample.sample_id for sample in evaluation_samples]
    local_mask_root = ensure_within(
        Path(str(config["evaluation"]["local_points"]["mask_root"])),
        output_root,
        name="local mask root",
    )
    local_mask_manifest, local_mask_entries = _local_mask_index(
        local_mask_root, evaluation_ids, require_complete=not bool(args.smoke)
    )

    model = InfiniDepth_DepthSensor(model_path=str(checkpoint_path)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(config["model"]["voxel_resolution"]),
        max_disparity_span=config["model"].get("max_disparity_span"),
    )
    refiner_parameters = _freeze_base(model)
    base_parameters = _base_parameters(model)
    base_checksum = _parameter_checksum(base_parameters)
    optimizer = torch.optim.AdamW(
        refiner_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    sampler = ShuffledCycleSampler(seed + 1)
    start_step = 0
    elapsed_before = 0.0
    last_evaluation: Mapping[str, object] = {}
    if args.resume is not None:
        try:
            resume = ensure_within(args.resume, output, name="local resume checkpoint")
            require_backup_marker = False
        except PermissionError:
            resume = ensure_within(args.resume, backup_output, name="NAS resume checkpoint")
            require_backup_marker = True
        verify_checkpoint_directory(resume, require_backup_marker=require_backup_marker)
        state = restore_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            config_sha256=config_sha,
            base_checkpoint_sha256=base_sha,
        )
        start_step = int(state["step"])
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        last_evaluation = state.get("evaluation", {})
    if start_step >= max_steps:
        raise ValueError(f"Resume step {start_step} is not below max step {max_steps}")

    archived_source_manifest = _archive_inputs(
        output,
        config_path,
        config,
        project_root,
        checkpoint_path,
        local_mask_manifest,
        smoke=bool(args.smoke),
        resume_step=start_step if args.resume is not None else None,
    )
    provenance_path = (
        output / "provenance.json"
        if start_step == 0
        else output / f"provenance_resume_step_{start_step:09d}.json"
    )
    _atomic_json(
        provenance_path,
        {
            "format": "infinidepth-lidar-refiner-provenance-v1",
            "experiment_id": config["experiment_id"],
            "run_id": args.run_id,
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_sha256": base_sha,
            "config_sha256": config_sha,
            "primary_run": str(output),
            "backup_run": str(backup_output),
            "backup_policy": "latest_local_checkpoint_on_exit",
            "archived_source_manifest": str(archived_source_manifest),
            "local_mask_manifest": str(local_mask_manifest),
            "local_mask_manifest_sha256": _sha256(local_mask_manifest),
            "train_sample_count": len(samples),
            "evaluation_sample_count": len(evaluation_samples),
            "smoke": bool(args.smoke),
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "started_at_unix": time.time(),
        },
    )
    report_path = output / "metrics" / "report.json"
    _atomic_json(
        report_path,
        {
            "format": "infinidepth-lidar-refiner-report-v2",
            "status": "smoke_running" if args.smoke else "running",
            "step": start_step,
            "resumed_from": str(args.resume) if args.resume else None,
        },
    )

    training = config["training"]
    microbatch = int(training["microbatch_size"])
    accumulation = int(training["global_batch_size"]) // microbatch
    sampled_ids = {str(value) for value in config["evaluation"]["sample_ids"]}
    sampled_evaluation = [
        sample for sample in evaluation_samples if sample.sample_id in sampled_ids
    ]
    if not sampled_evaluation:
        sampled_evaluation = [
            evaluation_samples[index] for index in range(min(5, len(evaluation_samples)))
        ]
    history_path = output / "metrics" / "history.jsonl"
    best_key: Optional[float] = None
    best_path = output / "best.json"
    if best_path.is_file():
        best_value = json.loads(best_path.read_text(encoding="utf-8"))
        best_key = float(best_value["selection_key"])
    started = time.time()
    backup = RunBackup(output, backup_output)
    last_completed_step = start_step
    try:
        for step in range(start_step + 1, max_steps + 1):
            model.eval()
            assert model.disparity_refiner is not None
            model.disparity_refiner.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated: MutableMapping[str, float] = {}
            sample_ids = []
            for _ in range(accumulation):
                indices = sampler.sample_indices(len(samples), microbatch)
                selected = _sample_values(samples, indices)
                sample_ids.extend(sample.sample_id for sample in selected)
                batch = stack_lidar_batch(
                    selected, config["lidar"], seed=seed, step=step, device=device
                )
                output_value = model.forward_dense_refined(
                    batch.image,
                    prompt_disparity=batch.prompt_disparity,
                    prompt_mask=batch.prompt_mask,
                    query_hw=(
                        int(config["model"]["height"]),
                        int(config["model"]["width"]),
                    ),
                    num_refinement_steps=3,
                    detach_base_from_refiner=True,
                    chunk_size=int(config["model"]["query_chunk_size"]),
                )
                if output_value.reference_scale is None or not torch.equal(
                    output_value.reference_scale.reshape(-1), batch.reference_scale
                ):
                    raise RuntimeError("DepthSensor and target disparity scales differ")
                loss, metrics = refiner_iteration_loss(
                    output_value.disparity_sequence,
                    batch.target_disparity,
                    batch.valid_mask,
                    gradient_weight=float(training["gradient_weight"]),
                    gradient_scales=int(training["gradient_scales"]),
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Training loss is non-finite")
                (loss / accumulation).backward()
                monitored = {
                    **{key: float(value.item()) for key, value in metrics.items()},
                    **_refinement_monitor(output_value),
                }
                for key, value in monitored.items():
                    accumulated[key] = accumulated.get(key, 0.0) + value / accumulation
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    refiner_parameters, float(training["gradient_clip_norm"])
                ).item()
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("Refiner gradient norm is non-finite")
            optimizer.step()
            last_completed_step = step
            if any(parameter.grad is not None for parameter in base_parameters):
                raise RuntimeError("Frozen DepthSensor base received a gradient")

            should_eval = step % eval_every == 0 or step == max_steps
            full_eval = step % full_eval_every == 0 or step == max_steps
            if should_eval:
                selected_evaluation = evaluation_samples if full_eval else sampled_evaluation
                iterations = (0, 1) if args.smoke else EVALUATION_STEPS
                last_evaluation = evaluate(
                    model,
                    selected_evaluation,
                    device=device,
                    config=config,
                    local_mask_root=local_mask_root,
                    local_mask_entries=local_mask_entries,
                    iterations=iterations,
                )
                record = {
                    "step": step,
                    "scope": "full" if full_eval else "sampled",
                    "training": dict(accumulated),
                    "gradient_norm": gradient_norm,
                    "evaluation": last_evaluation,
                    "global_sample_ids": sample_ids,
                    "elapsed_seconds": elapsed_before + time.time() - started,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
                _append_jsonl(history_path, record)
                is_best = False
                if full_eval and not args.smoke:
                    key = _selection_key(last_evaluation)
                    if best_key is None or key < best_key:
                        best_key = key
                        is_best = True
            else:
                is_best = False
            if step % checkpoint_every == 0 or step == max_steps:
                if not last_evaluation:
                    raise RuntimeError(
                        "Checkpoint requires an evaluation at the same or earlier step"
                    )
                checkpoint = save_checkpoint(
                    output / "checkpoints",
                    model=model,
                    optimizer=optimizer,
                    sampler=sampler,
                    step=step,
                    config_sha256=config_sha,
                    base_checkpoint_sha256=base_sha,
                    evaluation=last_evaluation,
                    elapsed_seconds=elapsed_before + time.time() - started,
                )
                if is_best:
                    _atomic_json(
                        best_path,
                        {
                            "format": CHECKPOINT_FORMAT,
                            "step": step,
                            "path": f"checkpoints/{checkpoint.name}",
                            "selection_key": best_key,
                        },
                    )
                _append_jsonl(
                    output / "logs" / "events.jsonl",
                    {
                        "event": "checkpoint_saved",
                        "step": step,
                        "path": str(checkpoint),
                        "created_at_unix": time.time(),
                    },
                )

        final_report = {
            "format": "infinidepth-lidar-refiner-report-v2",
            "status": "smoke_completed" if args.smoke else "completed",
            "steps": max_steps,
            "base_parameters_unchanged": _parameter_checksum(base_parameters)
            == base_checksum,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "backup_run": str(backup_output),
            "completed_at_unix": time.time(),
        }
    except BaseException as exc:
        interrupted_report = {
            "format": "infinidepth-lidar-refiner-report-v2",
            "status": "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "failed",
            "last_completed_step": last_completed_step,
            "error": f"{type(exc).__name__}: {exc}",
            "updated_at_unix": time.time(),
        }
        try:
            _backup_on_exit(
                backup,
                output,
                report_path,
                interrupted_report,
                step=last_completed_step,
                status="exit_backup_completed",
            )
        except BackupError:
            pass
        raise
    else:
        _backup_on_exit(
            backup,
            output,
            report_path,
            final_report,
            step=max_steps,
            status="completed",
        )


def _interrupt_on_sigterm(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def main() -> None:
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        run(parse_args())
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()

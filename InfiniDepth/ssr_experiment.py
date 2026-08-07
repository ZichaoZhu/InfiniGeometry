"""Server-only runner for the fixed-grid InfiniDepth SSR MVP."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import gc
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Dict, Iterable, Mapping, TextIO

import cv2
import h5py
import numpy as np
from PIL import Image
import torch

from .model.model import InfiniDepth
from .model.ssr import InfiniDepthSSR
from .model.ssr_geometry import InfiniDepthSSRInputs, build_ssr_inputs
from .model.ssr_losses import aligned_point_metrics, geometry_loss_sequence
from .utils.moge_utils import estimate_metric_depth_and_intrinsics_with_moge2
from .utils.moge_utils import _MOGE2_MODEL_CACHE


EXPECTED_ORIGIN = "https://github.com/ZichaoZhu/InfiniGeometry.git"
EXPECTED_UPSTREAM = "https://github.com/zju3dv/InfiniDepth.git"
EXPECTED_BASE = "36c6e0c31887fafc210184ee43ca475230704095"


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def safe_path(path: Path, safe_root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    root = safe_root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Write path is outside the personal safe root: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def load_hypersim_points(
    depth_path: Path,
    matrix: Iterable[Iterable[float]],
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = size
    with h5py.File(depth_path, "r") as handle:
        radial_depth = handle["dataset"][:].astype(np.float32)
    radial_depth = cv2.resize(radial_depth, (width, height), interpolation=cv2.INTER_NEAREST)
    u = (np.arange(width, dtype=np.float32) + 0.5) * (2.0 / width) - 1.0
    v = 1.0 - (np.arange(height, dtype=np.float32) + 0.5) * (2.0 / height)
    grid_u, grid_v = np.meshgrid(u, v)
    uv1 = np.stack((grid_u, grid_v, np.ones_like(grid_u)), axis=-1)
    rays = uv1 @ np.asarray(matrix, dtype=np.float32).T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
    rays *= np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
    valid = np.isfinite(radial_depth) & (radial_depth > 0)
    points = radial_depth[..., None] * rays
    points[~valid] = np.nan
    return torch.from_numpy(points).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def intrinsics_matrix(values: tuple[float, float, float, float], device: torch.device) -> torch.Tensor:
    fx, fy, cx, cy = values
    return torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )


def ssr_from_config(config: Mapping[str, object], device: torch.device) -> InfiniDepthSSR:
    ssr_config = config["model"]["ssr"]
    return InfiniDepthSSR(
        backend=ssr_config["backend"],
        voxel_resolution=ssr_config["voxel_resolution"],
        channels=ssr_config["channels"],
        visual_channels=ssr_config["visual_channels"],
        blocks_per_level=ssr_config["blocks_per_level"],
        normalization=ssr_config["normalization"],
        configured_depth_span=ssr_config["configured_depth_span"],
    ).to(device)


@torch.no_grad()
def evaluate(
    ssr: InfiniDepthSSR,
    inputs: InfiniDepthSSRInputs,
    gt_points: torch.Tensor,
    training_mask: torch.Tensor,
    k_values: Iterable[int],
    residual_bound: float,
) -> tuple[Dict[str, object], Dict[int, object]]:
    ssr.eval()
    metrics: Dict[str, object] = {}
    outputs = {}
    for k in k_values:
        output = ssr(inputs, K=int(k), residual_bound=residual_bound)
        outputs[int(k)] = output
        record: Dict[str, object] = aligned_point_metrics(
            output.points, gt_points, training_mask
        )
        if output.raw_residuals:
            raw = torch.cat([value.flatten() for value in output.raw_residuals])
            applied = torch.cat([value.flatten() for value in output.applied_residuals])
            record["raw_residual_percentiles"] = {
                str(percentile): float(torch.quantile(raw, percentile / 100.0))
                for percentile in (0, 50, 95, 99, 100)
            }
            record["applied_residual_percentiles"] = {
                str(percentile): float(torch.quantile(applied, percentile / 100.0))
                for percentile in (0, 50, 95, 99, 100)
            }
            record["voxel_statistics"] = [serializable_stats(item) for item in output.voxel_statistics]
        metrics[f"k{k}"] = record
    return metrics, outputs


def serializable_stats(stats: Mapping[str, object]) -> Dict[str, object]:
    result = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def write_ply(path: Path, points: torch.Tensor, colors: torch.Tensor, valid: torch.Tensor) -> None:
    xyz = points[valid].detach().cpu().numpy().astype(np.float32)
    rgb = (colors[valid].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    vertices = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def colorize_depth(depth: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    image = cv2.applyColorMap(
        np.round(255 * (1 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    image[~valid] = 0
    return image


def save_figures(
    experiment_dir: Path,
    history: list[Dict[str, object]],
    outputs: Mapping[int, object],
    gt_points: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures = experiment_dir / "artifacts/figures"
    figures.mkdir(parents=True, exist_ok=True)
    steps = [record["step"] for record in history]
    losses = [record["geometry_loss"] for record in history]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, losses, color="#2563eb", linewidth=2)
    axis.set_xlabel("Step")
    axis.set_ylabel("Geometry loss")
    axis.set_title("InfiniDepth SSR fixed-grid training")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "training_curve.png", dpi=180)
    plt.close(figure)

    valid_np = valid[0].detach().cpu().numpy()
    gt_depth = gt_points[0, ..., 2].detach().cpu().numpy()
    low, high = np.percentile(gt_depth[valid_np], (2, 98))
    panels = [("GT", gt_depth)] + [
        (f"K={k}", outputs[k].depth[0].detach().cpu().numpy()) for k in (0, 1, 3)
    ]
    rendered = []
    for label, depth in panels:
        panel = colorize_depth(depth, valid_np, float(low), float(high))
        cv2.putText(panel, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        rendered.append(panel)
    cv2.imwrite(str(figures / "geometry_comparison.png"), np.concatenate(rendered, axis=1))


def checkpoint_payload(
    ssr: InfiniDepthSSR,
    optimizer: torch.optim.Optimizer,
    step: int,
    score: float,
    config: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "ssr": ssr.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "selection_score": score,
        "config": copy.deepcopy(config["model"]["ssr"]),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }


def build_manifest(experiment_dir: Path, command: str, log_path: Path) -> Dict[str, object]:
    roles = {
        "artifacts/figures/training_curve.png": ("training curve", True),
        "artifacts/figures/geometry_comparison.png": ("GT/K0/K1/K3 comparison", True),
        "artifacts/checkpoints/best.pt": ("best K1 Point Rel checkpoint", False),
        "artifacts/checkpoints/last.pt": ("last resumable checkpoint", False),
        "artifacts/pointclouds/gt.ply": ("ground-truth point cloud", False),
        "artifacts/pointclouds/k0.ply": ("frozen base point cloud", False),
        "artifacts/pointclouds/k1.ply": ("selected one-step point cloud", False),
        "artifacts/pointclouds/k3.ply": ("diagnostic three-step point cloud", False),
        "artifacts/logs/run.log": ("raw server run log", False),
    }
    assets = []
    hashes: Dict[str, str] = {}
    for relative, (role, tracked) in roles.items():
        path = experiment_dir / relative
        if not path.exists():
            continue
        path.flush() if hasattr(path, "flush") else None
        digest = sha256_file(path)
        entry: Dict[str, object] = {
            "path": relative,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "generated_by": command,
            "tracked_by_git": tracked,
        }
        if digest in hashes:
            entry["alias_of"] = hashes[digest]
        else:
            hashes[digest] = relative
        assets.append(entry)
    return {"experiment_id": "exp1", "assets": assets}


def update_failure(experiment_dir: Path, reason: str, started: str) -> None:
    report_path = experiment_dir / "metrics/report.json"
    provenance_path = experiment_dir / "provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.setdefault("attempts", []).append({
        "status": "failed",
        "started_at": provenance.get("started_at") or started,
        "finished_at": now(),
        "run_commit": provenance.get("run_commit"),
        "failure_reason": reason,
    })
    report.update(status="failed", failure_reason=reason)
    provenance.update(status="failed", finished_at=provenance["attempts"][-1]["finished_at"])
    atomic_json(report_path, report)
    atomic_json(provenance_path, provenance)


def run(config_path: Path, safe_root: Path, log_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiment_dir = safe_path(config_path.resolve().parent, safe_root, must_exist=True)
    log_path = safe_path(log_path, safe_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = now()
    provenance_path = experiment_dir / "provenance.json"
    report_path = experiment_dir / "metrics/report.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    origin = git(project_root, "remote", "get-url", "origin")
    upstream = git(project_root, "remote", "get-url", "upstream")
    commit = git(project_root, "rev-parse", "HEAD")
    if origin != EXPECTED_ORIGIN or upstream != EXPECTED_UPSTREAM:
        raise RuntimeError(f"Unexpected remotes: origin={origin}, upstream={upstream}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE, commit],
        cwd=project_root,
        check=True,
    )
    dirty_before_run = bool(git(project_root, "status", "--porcelain"))
    if dirty_before_run:
        raise RuntimeError("Formal experiment requires a clean committed worktree")

    data = config["data"]
    rgb_path = Path(data["rgb"])
    depth_path = Path(data["radial_depth"])
    for path, expected in (
        (rgb_path, data["rgb_sha256"]),
        (depth_path, data["radial_depth_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Source checksum mismatch for {path}: {actual}")

    checkpoint_paths = {
        name: safe_path(project_root / relative, safe_root, must_exist=True)
        for name, relative in (
            ("infinidepth", config["model"]["infinidepth_checkpoint"]),
            ("moge2", config["model"]["moge2_checkpoint"]),
        )
    }
    try:
        import spconv

        spconv_version = getattr(spconv, "__version__", "unknown")
    except ImportError:
        spconv_version = "missing"
    provenance.update(
        status="running",
        run_commit=commit,
        git_dirty=False,
        started_at=started,
        finished_at=None,
        environment={
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "spconv": spconv_version,
        },
        sources=[
            {"role": "rgb", "path": str(rgb_path), "sha256": data["rgb_sha256"]},
            {"role": "radial_depth", "path": str(depth_path), "sha256": data["radial_depth_sha256"]},
            *[
                {"role": name, "path": str(path), "sha256": sha256_file(path)}
                for name, path in checkpoint_paths.items()
            ],
        ],
    )
    report.update(status="running", failure_reason=None)
    atomic_json(provenance_path, provenance)
    atomic_json(report_path, report)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Formal InfiniDepth SSR execution requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    height, width = (int(value) for value in config["model"]["input_size"])
    image = load_image(rgb_path, (height, width)).to(device)
    gt_points, gt_valid = load_hypersim_points(
        depth_path, data["m_cam_from_uv"], (height, width)
    )
    gt_points, gt_valid = gt_points.to(device), gt_valid.to(device)

    base = InfiniDepth(model_path=str(checkpoint_paths["infinidepth"]))
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base_versions = {
        name: value._version for name, value in base.state_dict(keep_vars=True).items()
    }
    reference_depth, reference_mask, moge_intrinsics = estimate_metric_depth_and_intrinsics_with_moge2(
        image, pretrained_model_name_or_path=str(checkpoint_paths["moge2"])
    )
    if moge_intrinsics is None:
        raise RuntimeError("MoGe2 did not return camera intrinsics")
    intrinsics = intrinsics_matrix(moge_intrinsics, device)
    inputs = build_ssr_inputs(
        base,
        image,
        reference_depth,
        reference_mask,
        intrinsics,
        source_tags={"metric_reference": "moge2", "supervision": "hypersim"},
    )
    if not all(item.success for item in inputs.alignment):
        raise RuntimeError(f"Formal reference alignment fell back: {inputs.alignment}")
    training_mask = inputs.valid_mask & gt_valid
    if not bool(training_mask.any()):
        raise RuntimeError("No pixels are jointly valid for reference and supervision")
    k0_hash = sha256_tensor(inputs.points0)

    reference_disparity = torch.where(
        reference_mask > 0,
        reference_depth.clamp_min(1e-8).reciprocal(),
        torch.zeros_like(reference_depth),
    )
    with torch.inference_mode():
        legacy_depth, _ = base.inference(
            image,
            inputs.query_coord.reshape(1, -1, 2),
            use_batch_infer=True,
            reference_disparity=reference_disparity,
            reference_mask=reference_mask,
        )
    legacy_depth = legacy_depth.reshape(1, height, width)
    regression_error = float((legacy_depth[training_mask] - inputs.depth0[training_mask]).abs().max())
    if regression_error > 1e-5:
        raise RuntimeError(f"K0 adapter regression error is too large: {regression_error}")

    base_state_unchanged = all(
        value._version == base_versions[name]
        for name, value in base.state_dict(keep_vars=True).items()
    )
    base_gradients_none = all(parameter.grad is None for parameter in base.parameters())
    moge2_gradients_none = all(
        parameter.grad is None
        for cached_model in _MOGE2_MODEL_CACHE.values()
        for parameter in cached_model.parameters()
    )
    if not (base_state_unchanged and base_gradients_none and moge2_gradients_none):
        raise RuntimeError("Frozen Base/MoGe2 state or gradient audit failed")

    del base
    _MOGE2_MODEL_CACHE.clear()
    del reference_depth, reference_mask, legacy_depth
    gc.collect()
    torch.cuda.empty_cache()
    ssr = ssr_from_config(config, device)
    training = config["training"]
    residual_bound = float(config["model"]["ssr"]["residual_bound"])
    optimizer = torch.optim.AdamW(
        ssr.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    initial_metrics, initial_outputs = evaluate(
        ssr, inputs, gt_points, training_mask, training["evaluation_k"], residual_bound
    )
    initial_loss, _ = geometry_loss_sequence(
        initial_outputs[1].points_sequence[1:],
        gt_points,
        training_mask,
        global_weight=training["loss_weights"]["global"],
        local_weight=training["loss_weights"]["local"],
        edge_weight=training["loss_weights"]["edge"],
        local_scales=training["local_scales"],
        generator=generator,
    )
    identity_error = float((initial_outputs[1].points - inputs.points0).abs().max())
    if identity_error != 0.0:
        raise RuntimeError(f"Zero-initialized SSR is not identity: {identity_error}")

    history = [{"step": 0, "geometry_loss": float(initial_loss), **initial_metrics}]
    history_path = experiment_dir / "metrics/history.jsonl"
    history_path.write_text(json.dumps(history[0], sort_keys=True) + "\n", encoding="utf-8")
    checkpoints = experiment_dir / "artifacts/checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best_score = float(initial_metrics["k1"]["point_rel"])
    best_step = 0
    torch.save(checkpoint_payload(ssr, optimizer, 0, best_score, config), checkpoints / "best.pt")
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in ssr.named_parameters()
        if "output_layer" in name
    }
    first_step_audit = None
    last_step = 0
    stop_reason = "maximum_steps"
    start_time = time.perf_counter()
    ssr.train()
    for step in range(1, int(training["steps"]) + 1):
        last_step = step
        optimizer.zero_grad(set_to_none=True)
        output = ssr(inputs, K=int(training["train_k"]), residual_bound=residual_bound)
        loss, terms = geometry_loss_sequence(
            output.points_sequence[1:],
            gt_points,
            training_mask,
            global_weight=training["loss_weights"]["global"],
            local_weight=training["loss_weights"]["local"],
            edge_weight=training["loss_weights"]["edge"],
            local_scales=training["local_scales"],
            generator=generator,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite geometry loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            ssr.parameters(), float(training["gradient_clip"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        optimizer.step()
        if step == 1:
            parameter_changed = any(
                not torch.equal(parameter.detach(), initial_parameters[name])
                for name, parameter in ssr.named_parameters()
                if name in initial_parameters
            )
            ssr.eval()
            with torch.no_grad():
                updated_output = ssr(inputs, K=1, residual_bound=residual_bound)
            k1_changed = not torch.equal(updated_output.points, inputs.points0)
            ssr.train()
            first_step_audit = {
                "ssr_parameter_changed": parameter_changed,
                "k1_differs_from_k0": k1_changed,
                "gradient_norm": float(gradient_norm),
                "loss": float(loss),
                "base_state_detached": all(not tensor.requires_grad for tensor in (
                    inputs.points0, inputs.dino_feature, inputs.basic_feature
                )),
                "base_gradients_none": base_gradients_none,
                "moge2_gradients_none": moge2_gradients_none,
                "base_state_unchanged": base_state_unchanged,
            }
            if not all(
                value for key, value in first_step_audit.items()
                if key not in {"gradient_norm", "loss"}
            ):
                raise RuntimeError(f"First-step audit failed: {first_step_audit}")
            del initial_parameters
        if step % int(training["evaluation_interval"]) == 0 or step == int(training["steps"]):
            metrics, evaluation_outputs = evaluate(
                ssr, inputs, gt_points, training_mask, training["evaluation_k"], residual_bound
            )
            evaluation_loss, _ = geometry_loss_sequence(
                evaluation_outputs[1].points_sequence[1:],
                gt_points,
                training_mask,
                global_weight=training["loss_weights"]["global"],
                local_weight=training["loss_weights"]["local"],
                edge_weight=training["loss_weights"]["edge"],
                local_scales=training["local_scales"],
                generator=generator,
            )
            record = {
                "step": step,
                "geometry_loss": float(evaluation_loss),
                "train_geometry_loss": float(loss),
                **metrics,
            }
            history.append(record)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            score = float(metrics["k1"]["point_rel"])
            if score < best_score:
                best_score, best_step = score, step
                torch.save(
                    checkpoint_payload(ssr, optimizer, step, score, config), checkpoints / "best.pt"
                )
            print(json.dumps({
                "step": step,
                "loss": float(loss),
                "gradient_norm": float(gradient_norm),
                "k0_point_rel": metrics["k0"]["point_rel"],
                "k1_point_rel": metrics["k1"]["point_rel"],
                "k3_point_rel": metrics["k3"]["point_rel"],
            }), flush=True)
            current_point_improvement = 1.0 - score / max(
                float(metrics["k0"]["point_rel"]), 1e-12
            )
            current_loss_reduction = 1.0 - float(evaluation_loss) / max(
                float(initial_loss), 1e-12
            )
            if (
                bool(training.get("stop_when_accepted", False))
                and current_point_improvement
                >= float(config["acceptance"]["minimum_k1_point_rel_relative_improvement"])
                and current_loss_reduction
                >= float(config["acceptance"]["minimum_geometry_loss_relative_reduction"])
            ):
                stop_reason = "acceptance_reached"
                break
            ssr.train()

    elapsed = time.perf_counter() - start_time
    torch.save(
        checkpoint_payload(ssr, optimizer, last_step, float(history[-1]["k1"]["point_rel"]), config),
        checkpoints / "last.pt",
    )
    best_checkpoint = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
    ssr.load_state_dict(best_checkpoint["ssr"])
    best_metrics, best_outputs = evaluate(
        ssr, inputs, gt_points, training_mask, training["evaluation_k"], residual_bound
    )
    best_loss, _ = geometry_loss_sequence(
        best_outputs[1].points_sequence[1:], gt_points, training_mask,
        global_weight=training["loss_weights"]["global"],
        local_weight=training["loss_weights"]["local"],
        edge_weight=training["loss_weights"]["edge"],
        local_scales=training["local_scales"], generator=generator,
    )
    expected_reload = best_outputs[1].points.detach().clone()
    del ssr
    torch.cuda.empty_cache()
    reloaded = ssr_from_config(config, device)
    reloaded.load_state_dict(best_checkpoint["ssr"])
    reloaded_metrics, reloaded_outputs = evaluate(
        reloaded, inputs, gt_points, training_mask, training["evaluation_k"], residual_bound
    )
    acceptance = config["acceptance"]
    if not torch.allclose(
        expected_reload,
        reloaded_outputs[1].points,
        rtol=float(acceptance["reload_rtol"]),
        atol=float(acceptance["reload_atol"]),
    ):
        raise RuntimeError("SSR checkpoint reload output mismatch")
    if sha256_tensor(inputs.points0) != k0_hash:
        raise RuntimeError("Frozen K0 tensor changed during training")

    k0_score = float(reloaded_metrics["k0"]["point_rel"])
    k1_score = float(reloaded_metrics["k1"]["point_rel"])
    point_improvement = 1.0 - k1_score / max(k0_score, 1e-12)
    loss_reduction = 1.0 - float(best_loss) / max(float(initial_loss), 1e-12)
    accepted = (
        point_improvement >= float(acceptance["minimum_k1_point_rel_relative_improvement"])
        and loss_reduction >= float(acceptance["minimum_geometry_loss_relative_reduction"])
    )

    pointclouds = experiment_dir / "artifacts/pointclouds"
    pointclouds.mkdir(parents=True, exist_ok=True)
    colors = image[0].permute(1, 2, 0)
    write_ply(pointclouds / "gt.ply", gt_points[0], colors, training_mask[0])
    for k in (0, 1, 3):
        write_ply(
            pointclouds / f"k{k}.ply", reloaded_outputs[k].points[0], colors, training_mask[0]
        )
    save_figures(experiment_dir, history, reloaded_outputs, gt_points, training_mask)

    best_hash = sha256_file(checkpoints / "best.pt")
    last_hash = sha256_file(checkpoints / "last.pt")
    if best_hash == last_hash:
        (checkpoints / "last.pt").unlink()
    report = {
        "experiment_id": "exp1",
        "status": "completed",
        "selection_metric": "k1.point_rel",
        "baseline": {**initial_metrics, "geometry_loss": float(initial_loss)},
        "best": {
            **reloaded_metrics,
            "step": best_step,
            "geometry_loss": float(best_loss),
        },
        "final": history[-1],
        "runtime_seconds": elapsed,
        "stop_reason": stop_reason,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "adapter_regression_max_abs": regression_error,
        "initial_identity_max_abs": identity_error,
        "k0_sha256": k0_hash,
        "alignment": [asdict(item) for item in inputs.alignment],
        "first_step_audit": first_step_audit,
        "acceptance": {
            "passed": accepted,
            "k1_point_rel_relative_improvement": point_improvement,
            "geometry_loss_relative_reduction": loss_reduction,
            "reload_passed": True,
        },
        "failure_reason": None,
    }
    finished = now()
    provenance.setdefault("attempts", []).append({
        "status": "completed",
        "started_at": started,
        "finished_at": finished,
        "run_commit": commit,
        "failure_reason": None,
    })
    provenance.update(status="completed", finished_at=finished)
    atomic_json(report_path, report)
    atomic_json(provenance_path, provenance)
    sys.stdout.flush()
    manifest = build_manifest(
        experiment_dir,
        provenance["command"],
        log_path,
    )
    atomic_json(experiment_dir / "artifacts/manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    experiment_dir = config_path.parent
    log_path = args.log or experiment_dir / "artifacts/logs/run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        tee_out, tee_err = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                run(config_path, args.safe_root, log_path)
            return 0
        except Exception as exc:
            update_failure(experiment_dir, f"{type(exc).__name__}: {exc}", started)
            raise


if __name__ == "__main__":
    raise SystemExit(main())

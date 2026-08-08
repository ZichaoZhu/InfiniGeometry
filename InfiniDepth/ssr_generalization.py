"""Server-only held-out evaluation for a frozen InfiniDepth SSR checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import gc
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Dict, Iterable, Mapping

import numpy as np
import torch

from .model.model import InfiniDepth
from .model.ssr_losses import aligned_point_metrics
from .model.ssr_geometry import build_ssr_inputs
from .ssr_experiment import (
    EXPECTED_BASE,
    EXPECTED_ORIGIN,
    EXPECTED_UPSTREAM,
    Tee,
    atomic_json,
    git,
    intrinsics_matrix,
    load_hypersim_points,
    load_image,
    now,
    safe_path,
    sha256_file,
    sha256_tensor,
    ssr_from_config,
    update_failure,
)
from .utils.moge_utils import (
    _MOGE2_MODEL_CACHE,
    estimate_metric_depth_and_intrinsics_with_moge2,
)


DATA_ROOT = Path("/nas1/datasets/hypersim/raw")


def validate_split(config: Mapping[str, object]) -> None:
    data = config["data"]
    samples = data["samples"]
    expected_samples = int(data["expected_sample_count"])
    expected_scenes = int(data["expected_scene_count"])
    if len(samples) != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {len(samples)}")
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Held-out sample IDs must be unique")
    excluded = set(data.get("excluded_sample_ids", []))
    overlap = sorted(excluded.intersection(sample_ids))
    if overlap:
        raise ValueError(f"Held-out split contains excluded samples: {overlap}")
    scenes = {sample["scene_id"] for sample in samples}
    if len(scenes) != expected_scenes:
        raise ValueError(f"Expected {expected_scenes} scenes, got {len(scenes)}")
    if [int(value) for value in config["evaluation"]["k_values"]] != [0, 1, 3]:
        raise ValueError("Held-out evaluation requires K values [0, 1, 3]")
    if int(config["evaluation"]["repeat_k"]) != 1:
        raise ValueError("Determinism audit requires repeat_k=1")
    for sample in samples:
        if not sample["sample_id"].startswith(f"{sample['scene_id']}_cam_00_frame."):
            raise ValueError(f"Sample ID does not match scene: {sample['sample_id']}")
        for path_key, hash_key in (
            ("rgb", "rgb_sha256"),
            ("radial_depth", "radial_depth_sha256"),
        ):
            path = Path(sample[path_key])
            if not path.is_absolute() or DATA_ROOT not in path.parents:
                raise ValueError(f"Input path is outside the read-only dataset root: {path}")
            digest = sample[hash_key]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Invalid SHA-256 for {sample['sample_id']} {path_key}")


def compact_k_metrics(metrics: Mapping[str, object], k_values: Iterable[int]) -> Dict[str, object]:
    fields = ("point_rel", "depth_rel", "alignment_scale", "alignment_z_shift")
    return {
        f"k{k}": {field: metrics[f"k{k}"][field] for field in fields}
        for k in k_values
    }


def aggregate_evaluation(
    records: list[Mapping[str, object]],
    acceptance: Mapping[str, float],
) -> Dict[str, object]:
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation")

    def metric_mean(items: list[Mapping[str, object]], k: int, metric: str) -> float:
        return float(np.mean([item["metrics"][f"k{k}"][metric] for item in items]))

    aggregate_metrics = {
        f"k{k}": {
            "point_rel": metric_mean(records, k, "point_rel"),
            "depth_rel": metric_mean(records, k, "depth_rel"),
        }
        for k in (0, 1, 3)
    }
    improvements = np.asarray(
        [float(item["k1_point_rel_relative_improvement"]) for item in records],
        dtype=np.float64,
    )
    by_scene: Dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in records:
        by_scene[str(item["scene_id"])].append(item)
    scene_records = {}
    for scene_id, items in sorted(by_scene.items()):
        k0 = metric_mean(items, 0, "point_rel")
        k1 = metric_mean(items, 1, "point_rel")
        scene_records[scene_id] = {
            "sample_count": len(items),
            "k0_point_rel": k0,
            "k1_point_rel": k1,
            "k1_point_rel_relative_improvement": 1.0 - k1 / max(k0, 1e-12),
        }
    scene_improvements = [
        value["k1_point_rel_relative_improvement"] for value in scene_records.values()
    ]
    mean_improvement = 1.0 - (
        aggregate_metrics["k1"]["point_rel"]
        / max(aggregate_metrics["k0"]["point_rel"], 1e-12)
    )
    win_rate = float(np.mean(improvements > 0.0))
    maximum_scene_regression = max(0.0, -min(scene_improvements))
    maximum_determinism_error = max(
        float(item["repeat_k1_max_abs"]) for item in records
    )
    deterministic_passed = maximum_determinism_error <= float(
        acceptance["maximum_repeat_k1_max_abs"]
    )
    k0_stable = all(bool(item["k0_stable"]) for item in records)
    passed = (
        mean_improvement
        >= float(acceptance["minimum_mean_k1_point_rel_relative_improvement"])
        and win_rate >= float(acceptance["minimum_sample_win_rate"])
        and maximum_scene_regression
        <= float(acceptance["maximum_scene_point_rel_relative_regression"])
        and deterministic_passed
        and k0_stable
    )
    return {
        "sample_count": len(records),
        "scene_count": len(scene_records),
        "metrics": aggregate_metrics,
        "mean_k1_point_rel_relative_improvement": mean_improvement,
        "mean_per_sample_k1_point_rel_relative_improvement": float(improvements.mean()),
        "median_per_sample_k1_point_rel_relative_improvement": float(np.median(improvements)),
        "sample_win_rate": win_rate,
        "maximum_sample_point_rel_relative_regression": max(0.0, -float(improvements.min())),
        "maximum_scene_point_rel_relative_regression": maximum_scene_regression,
        "per_scene": scene_records,
        "acceptance": {
            "passed": passed,
            "deterministic_passed": deterministic_passed,
            "k0_stable": k0_stable,
            "maximum_repeat_k1_max_abs": maximum_determinism_error,
            "thresholds": dict(acceptance),
        },
    }


def save_summary_figures(experiment_dir: Path, records: list[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = experiment_dir / "artifacts/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{item['scene_id'][3:]}:{item['frame_id']}" for item in records]
    x = np.arange(len(records))
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    for k, color, marker in ((0, "#444444", "o"), (1, "#1976d2", "s"), (3, "#d1495b", "^")):
        values = [item["metrics"][f"k{k}"]["point_rel"] for item in records]
        axis.plot(x, values, marker=marker, markersize=3, linewidth=1.2, color=color, label=f"K={k}")
    axis.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("Point Rel")
    axis.set_title("Held-out Point Rel by sample")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figure_dir / "point_rel_by_sample.png", dpi=160)
    plt.close(figure)

    improvements = np.asarray(
        [item["k1_point_rel_relative_improvement"] for item in records]
    ) * 100.0
    colors = np.where(improvements >= 0.0, "#2a9d8f", "#d1495b")
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.bar(x, improvements, color=colors, width=0.8)
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("K1 relative improvement over K0 (%)")
    axis.set_title("Held-out K1 improvement by sample")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figure_dir / "k1_relative_improvement.png", dpi=160)
    plt.close(figure)


def build_manifest(experiment_dir: Path, command: str, log_path: Path) -> Dict[str, object]:
    roles = {
        experiment_dir / "artifacts/figures/point_rel_by_sample.png": (
            "artifacts/figures/point_rel_by_sample.png",
            "K0/K1/K3 Point Rel comparison across held-out samples",
            True,
        ),
        experiment_dir / "artifacts/figures/k1_relative_improvement.png": (
            "artifacts/figures/k1_relative_improvement.png",
            "per-sample K1 relative improvement",
            True,
        ),
        log_path: ("artifacts/logs/run.log", "raw server evaluation log", False),
    }
    assets = []
    for path, (relative, role, tracked) in roles.items():
        if path.exists():
            assets.append({
                "path": relative,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "generated_by": command,
                "tracked_by_git": tracked,
            })
    return {"experiment_id": "exp2", "assets": assets}


def run(config_path: Path, safe_root: Path, log_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiment_dir = safe_path(config_path.resolve().parent, safe_root, must_exist=True)
    log_path = safe_path(log_path, safe_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_split(config)
    started = now()
    provenance_path = experiment_dir / "provenance.json"
    report_path = experiment_dir / "metrics/report.json"
    history_path = experiment_dir / "metrics/history.jsonl"
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
    if git(project_root, "status", "--porcelain"):
        raise RuntimeError("Formal experiment requires a clean committed worktree")

    data_sources = []
    for sample in config["data"]["samples"]:
        for role, path_key, hash_key in (
            ("rgb", "rgb", "rgb_sha256"),
            ("radial_depth", "radial_depth", "radial_depth_sha256"),
        ):
            path = Path(sample[path_key])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != sample[hash_key]:
                raise RuntimeError(f"Source checksum mismatch for {path}: {actual}")
            data_sources.append({
                "sample_id": sample["sample_id"],
                "role": role,
                "path": str(path),
                "sha256": actual,
            })

    model_config = config["model"]
    checkpoint_paths = {
        name: safe_path(project_root / relative, safe_root, must_exist=True)
        for name, relative in (
            ("infinidepth", model_config["infinidepth_checkpoint"]),
            ("moge2", model_config["moge2_checkpoint"]),
            ("ssr", model_config["ssr_checkpoint"]),
        )
    }
    for name, path in checkpoint_paths.items():
        expected = model_config[f"{name}_checkpoint_sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Checkpoint checksum mismatch for {path}: {actual}")

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
            *data_sources,
            *[
                {"role": name, "path": str(path), "sha256": model_config[f"{name}_checkpoint_sha256"]}
                for name, path in checkpoint_paths.items()
            ],
        ],
    )
    report.update(status="running", failure_reason=None)
    history_path.write_text("", encoding="utf-8")
    atomic_json(provenance_path, provenance)
    atomic_json(report_path, report)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Formal InfiniDepth SSR evaluation requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    height, width = (int(value) for value in model_config["input_size"])

    base = InfiniDepth(model_path=str(checkpoint_paths["infinidepth"]))
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base_versions = {
        name: value._version for name, value in base.state_dict(keep_vars=True).items()
    }
    ssr = ssr_from_config(config, device)
    checkpoint = torch.load(checkpoint_paths["ssr"], map_location="cpu", weights_only=False)
    if checkpoint.get("config") != model_config["ssr"]:
        raise RuntimeError("SSR checkpoint architecture does not match exp2 config")
    ssr.load_state_dict(checkpoint["ssr"], strict=True)
    source_step = int(checkpoint["step"])
    source_score = float(checkpoint["selection_score"])
    del checkpoint
    ssr.eval()
    for parameter in ssr.parameters():
        parameter.requires_grad_(False)

    records = []
    residual_bound = float(model_config["ssr"]["residual_bound"])
    k_values = [int(value) for value in config["evaluation"]["k_values"]]
    with history_path.open("a", encoding="utf-8", buffering=1) as history:
        for index, sample in enumerate(config["data"]["samples"], start=1):
            image = load_image(Path(sample["rgb"]), (height, width)).to(device)
            gt_points, gt_valid = load_hypersim_points(
                Path(sample["radial_depth"]),
                config["data"]["m_cam_from_uv"],
                (height, width),
            )
            gt_points, gt_valid = gt_points.to(device), gt_valid.to(device)
            reference_depth, reference_mask, moge_intrinsics = (
                estimate_metric_depth_and_intrinsics_with_moge2(
                    image,
                    pretrained_model_name_or_path=str(checkpoint_paths["moge2"]),
                )
            )
            for cached_model in _MOGE2_MODEL_CACHE.values():
                for parameter in cached_model.parameters():
                    parameter.requires_grad_(False)
            if moge_intrinsics is None:
                raise RuntimeError(f"MoGe2 did not return intrinsics for {sample['sample_id']}")
            inputs = build_ssr_inputs(
                base,
                image,
                reference_depth,
                reference_mask,
                intrinsics_matrix(moge_intrinsics, device),
                source_tags={"metric_reference": "moge2", "supervision": "hypersim"},
            )
            if not all(item.success for item in inputs.alignment):
                raise RuntimeError(
                    f"Reference alignment fell back for {sample['sample_id']}: {inputs.alignment}"
                )
            evaluation_mask = inputs.valid_mask & gt_valid
            if not bool(evaluation_mask.any()):
                raise RuntimeError(f"No jointly valid pixels for {sample['sample_id']}")
            k0_hash = sha256_tensor(inputs.points0)
            outputs = {}
            raw_metrics = {}
            with torch.no_grad():
                for k in k_values:
                    output = ssr(inputs, K=k, residual_bound=residual_bound)
                    outputs[k] = output
                    raw_metrics[f"k{k}"] = aligned_point_metrics(
                        output.points, gt_points, evaluation_mask
                    )
                repeat_k1 = ssr(inputs, K=1, residual_bound=residual_bound)
            repeat_error = float((outputs[1].points - repeat_k1.points).abs().max())
            k0_stable = sha256_tensor(inputs.points0) == k0_hash
            metrics = compact_k_metrics(raw_metrics, k_values)
            k0_score = float(metrics["k0"]["point_rel"])
            k1_score = float(metrics["k1"]["point_rel"])
            record = {
                "index": index,
                "sample_id": sample["sample_id"],
                "scene_id": sample["scene_id"],
                "frame_id": sample["frame_id"],
                "valid_pixel_count": int(evaluation_mask.sum()),
                "metrics": metrics,
                "k1_point_rel_relative_improvement": 1.0 - k1_score / max(k0_score, 1e-12),
                "repeat_k1_max_abs": repeat_error,
                "k0_sha256": k0_hash,
                "k0_stable": k0_stable,
                "alignment": [asdict(item) for item in inputs.alignment],
            }
            records.append(record)
            history.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({
                "sample": f"{index}/{len(config['data']['samples'])}",
                "sample_id": sample["sample_id"],
                "k0_point_rel": k0_score,
                "k1_point_rel": k1_score,
                "k3_point_rel": metrics["k3"]["point_rel"],
                "k1_improvement": record["k1_point_rel_relative_improvement"],
            }))
            del (
                image,
                gt_points,
                gt_valid,
                reference_depth,
                reference_mask,
                inputs,
                outputs,
                repeat_k1,
                evaluation_mask,
            )
            gc.collect()
            torch.cuda.empty_cache()

    base_state_unchanged = all(
        value._version == base_versions[name]
        for name, value in base.state_dict(keep_vars=True).items()
    )
    gradients_none = {
        "infinidepth": all(parameter.grad is None for parameter in base.parameters()),
        "ssr": all(parameter.grad is None for parameter in ssr.parameters()),
        "moge2": all(
            parameter.grad is None
            for model in _MOGE2_MODEL_CACHE.values()
            for parameter in model.parameters()
        ),
    }
    parameters_frozen = {
        "infinidepth": all(not parameter.requires_grad for parameter in base.parameters()),
        "ssr": all(not parameter.requires_grad for parameter in ssr.parameters()),
        "moge2": all(
            not parameter.requires_grad
            for model in _MOGE2_MODEL_CACHE.values()
            for parameter in model.parameters()
        ),
    }
    aggregate = aggregate_evaluation(records, config["acceptance"])
    frozen_audit_passed = (
        base_state_unchanged
        and all(gradients_none.values())
        and all(parameters_frozen.values())
    )
    aggregate["acceptance"]["frozen_audit_passed"] = frozen_audit_passed
    aggregate["acceptance"]["passed"] = (
        bool(aggregate["acceptance"]["passed"]) and frozen_audit_passed
    )
    elapsed = time.perf_counter() - start_time
    save_summary_figures(experiment_dir, records)
    report = {
        "experiment_id": "exp2",
        "status": "completed",
        "evaluation_mode": "frozen_checkpoint_heldout",
        "source_checkpoint": {
            "path": model_config["ssr_checkpoint"],
            "sha256": model_config["ssr_checkpoint_sha256"],
            "step": source_step,
            "selection_score": source_score,
        },
        "aggregate": {key: value for key, value in aggregate.items() if key != "acceptance"},
        "acceptance": aggregate["acceptance"],
        "audits": {
            "base_state_unchanged": base_state_unchanged,
            "gradients_none": gradients_none,
            "parameters_frozen": parameters_frozen,
            "trainable_parameter_count": sum(
                parameter.numel()
                for model in (base, ssr)
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "runtime_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
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
    atomic_json(
        experiment_dir / "artifacts/manifest.json",
        build_manifest(experiment_dir, provenance["command"], log_path),
    )
    _MOGE2_MODEL_CACHE.clear()


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

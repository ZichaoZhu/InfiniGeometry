from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from InfiniDepth.model import DisparityRefinementOutput, InfiniDepth
from training.disparity_refiner.data import (
    HypersimDisparitySample,
    ensure_within,
    load_manifest,
    preload_samples,
    select_training_entries,
)
from training.disparity_refiner.losses import disparity_metrics, supervised_iteration_loss


EVALUATION_STEPS = (0, 1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed-grid InfiniDepth native-disparity refiner"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _git_value(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _selected_run(
    config: Mapping[str, object],
    run_id: Optional[str],
) -> Mapping[str, object]:
    configured = config.get("runs")
    if configured is None:
        if run_id not in (None, "main"):
            raise ValueError("This experiment has a single run; omit --run-id")
        return {"id": "main"}
    if not run_id:
        raise ValueError("--run-id is required for a multi-run experiment")
    matches = [run for run in configured if str(run.get("id")) == run_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate run ID: {run_id}")
    return matches[0]


def _structure_selections(config: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    return {
        str(entry["sample_id"]): entry
        for entry in config.get("structure_selections", [])
    }


def _load_samples(
    config: Mapping[str, object],
    run: Mapping[str, object],
    project_root: Path,
) -> Sequence[HypersimDisparitySample]:
    data = config["data"]
    manifest_path = ensure_within(
        project_root / str(data["manifest"]), project_root, name="manifest"
    )
    manifest = load_manifest(manifest_path, str(data["manifest_sha256"]))
    requested = run.get("sample_ids", data.get("sample_ids"))
    entries = select_training_entries(manifest, requested)
    expected = int(data["expected_train_count"])
    if len(entries) != expected:
        raise ValueError(f"Expected {expected} training samples, got {len(entries)}")
    source_root = Path(str(data["source_root"]))
    if source_root.resolve() != Path("/nas1/datasets/hypersim/raw"):
        raise PermissionError("Hypersim source root must remain /nas1/datasets/hypersim/raw")
    model_cfg = config["model"]
    return preload_samples(
        entries,
        source_root=source_root,
        height=int(model_cfg["height"]),
        width=int(model_cfg["width"]),
        structure_selections=_structure_selections(config),
    )


def _parameter_groups(model: InfiniDepth) -> Dict[str, List[torch.nn.Parameter]]:
    if model.disparity_refiner is None:
        raise RuntimeError("Disparity refiner is not attached")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = {
        "ssr": list(model.disparity_refiner.parameters()),
        "head": list(model.basic_encoder.parameters())
        + list(model.depth_implicit_head.parameters()),
        "dino": list(model.pretrained.parameters()),
    }
    for parameters in groups.values():
        for parameter in parameters:
            parameter.requires_grad_(True)
    identities = [id(parameter) for values in groups.values() for parameter in values]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Optimizer parameter groups overlap")
    return groups


def _build_optimizer(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"name": name, "params": list(parameters), "lr": 0.0}
            for name, parameters in groups.items()
        ],
        weight_decay=float(weight_decay),
    )


def _configure_stage_learning_rates(
    optimizer: torch.optim.Optimizer,
    dino_parameters: Sequence[torch.nn.Parameter],
    stage_config: Mapping[str, object],
    stage_step: int,
) -> Dict[str, float]:
    peak = {name: float(value) for name, value in stage_config["learning_rates"].items()}
    freeze = int(stage_config.get("dino_freeze_steps", 0))
    warmup_end = int(stage_config.get("dino_warmup_end", freeze))
    if stage_step <= freeze:
        peak["dino"] = 0.0
        train_dino = False
    else:
        train_dino = True
        if warmup_end > freeze and stage_step < warmup_end:
            peak["dino"] *= (stage_step - freeze) / (warmup_end - freeze)
    for parameter in dino_parameters:
        parameter.requires_grad_(train_dino)
    for group in optimizer.param_groups:
        group["lr"] = peak[str(group["name"])]
    return peak


def _stack_batch(
    samples: Sequence[HypersimDisparitySample],
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.stack([samples[index].image for index in indices]).to(device)
    target = torch.stack([samples[index].target_disparity for index in indices]).to(device)
    mask = torch.stack([samples[index].valid_mask for index in indices]).to(device)
    return image, target, mask


def _sample_indices(
    generator: random.Random,
    count: int,
    batch_size: int,
) -> List[int]:
    if count == 1:
        return [0] * batch_size
    return [generator.randrange(count) for _ in range(batch_size)]


def _refinement_monitor(
    output_value: DisparityRefinementOutput,
) -> Dict[str, float]:
    monitor: Dict[str, float] = {}
    for iteration, (statistics, raw, bounded) in enumerate(
        zip(
            output_value.voxel_statistics,
            output_value.raw_residuals,
            output_value.bounded_residuals,
        ),
        start=1,
    ):
        if not bool(statistics["raw_residual_finite"]):
            raise FloatingPointError(f"K{iteration} raw residual contains non-finite values")
        if not torch.isfinite(bounded).all():
            raise FloatingPointError(f"K{iteration} bounded residual contains non-finite values")
        if float(bounded.detach().abs().amax().item()) > 0.1000001:
            raise RuntimeError(f"K{iteration} additive disparity residual exceeded 0.1")
        spans = statistics["disparity_span"]
        monitor[f"k{iteration}_active_voxels"] = float(statistics["active_voxels"])
        monitor[f"k{iteration}_disparity_span_max"] = float(spans.detach().amax().item())
        monitor[f"k{iteration}_raw_residual_min"] = float(raw.detach().amin().item())
        monitor[f"k{iteration}_raw_residual_max"] = float(raw.detach().amax().item())
        monitor[f"k{iteration}_raw_residual_mean"] = float(raw.detach().mean().item())
        monitor[f"k{iteration}_bounded_residual_min"] = float(bounded.detach().amin().item())
        monitor[f"k{iteration}_bounded_residual_max"] = float(bounded.detach().amax().item())
        monitor[f"k{iteration}_bounded_residual_mean"] = float(bounded.detach().mean().item())
        for level, count in enumerate(statistics["active_voxels_per_level"]):
            monitor[f"k{iteration}_level{level}_active_voxels"] = float(count)
    return monitor


@torch.no_grad()
def evaluate(
    model: InfiniDepth,
    samples: Sequence[HypersimDisparitySample],
    indices: Sequence[int],
    *,
    device: torch.device,
    query_hw: Tuple[int, int],
    chunk_size: int,
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    per_image: Dict[str, object] = {}
    for index in indices:
        sample = samples[index]
        output = model.forward_dense_refined(
            sample.image[None].to(device),
            query_hw=query_hw,
            num_refinement_steps=max(EVALUATION_STEPS),
            detach_base_from_refiner=False,
            chunk_size=chunk_size,
        )
        target = sample.target_disparity.to(device)
        valid = sample.valid_mask.to(device)
        structure = (
            None if sample.structure_mask is None else sample.structure_mask.to(device)
        )
        per_k = {}
        for iteration in EVALUATION_STEPS:
            per_k[f"k{iteration}"] = disparity_metrics(
                output.disparity_sequence[iteration][0], target, valid, structure
            )
        per_image[sample.sample_id] = per_k
    aggregate = {}
    for iteration in EVALUATION_STEPS:
        key = f"k{iteration}"
        full = [float(value[key]["full_mae"]) for value in per_image.values()]
        structures = [
            float(value[key]["structure_mae"])
            for value in per_image.values()
            if "structure_mae" in value[key]
        ]
        summary = {"full_mae": float(np.mean(full))}
        if structures:
            summary["structure_mae"] = float(np.mean(structures))
            summary["composite_score"] = summary["full_mae"] + summary["structure_mae"]
        aggregate[key] = summary
    improved = sum(
        float(value["k3"]["full_mae"]) < float(value["k0"]["full_mae"])
        for value in per_image.values()
    )
    if was_training:
        model.train()
    return {
        "sample_count": len(indices),
        "aggregate": aggregate,
        "per_image": per_image,
        "k3_better_than_k0_count": int(improved),
    }


def _selection_score(evaluation: Mapping[str, object]) -> float:
    k3 = evaluation["aggregate"]["k3"]
    return float(k3.get("composite_score", k3["full_mae"]))


def _save_checkpoint(
    path: Path,
    *,
    model: InfiniDepth,
    optimizer: torch.optim.Optimizer,
    stage: str,
    stage_step: int,
    total_step: int,
    config_sha256: str,
    evaluation: Mapping[str, object],
    include_optimizer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "format": "infinidepth-disparity-refiner-v1",
        "model": model.state_dict(),
        "optimizer_included": bool(include_optimizer),
        "stage": stage,
        "stage_step": int(stage_step),
        "total_step": int(total_step),
        "config_sha256": config_sha256,
        "evaluation": evaluation,
    }
    if include_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _restore_checkpoint(
    path: Path,
    model: InfiniDepth,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if checkpoint.get("format") != "infinidepth-disparity-refiner-v1":
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("Resume checkpoint was produced by another config")
    if not checkpoint.get("optimizer_included") or "optimizer" not in checkpoint:
        raise ValueError("Resume requires last.pt with optimizer state")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _checkpoint_manifest(checkpoint_dir: Path) -> Dict[str, object]:
    assets = []
    seen: Dict[str, Tuple[str, Path]] = {}
    for name in ("stage1_best.pt", "joint_best.pt", "last.pt"):
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        digest = _sha256(path)
        entry: MutableMapping[str, object] = {
            "path": str(path.name),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "tracked_by_git": False,
        }
        if digest in seen:
            alias_name, canonical_path = seen[digest]
            path.unlink()
            os.link(canonical_path, path)
            entry["alias_of"] = alias_name
        else:
            seen[digest] = (name, path)
        assets.append(entry)
    return {"version": 1, "assets": assets}


def _provenance(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, object],
    run: Mapping[str, object],
    command: Sequence[str],
) -> Dict[str, object]:
    checkpoint = Path(str(config["model"]["checkpoint"]))
    return {
        "format": "infinidepth-disparity-refiner-provenance-v1",
        "experiment_id": config["experiment_id"],
        "run_id": run["id"],
        "origin_url": _git_value(project_root, "remote", "get-url", "origin"),
        "upstream_url": _git_value(project_root, "remote", "get-url", "upstream"),
        "code_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "code_dirty": bool(_git_value(project_root, "status", "--porcelain")),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": config["data"]["manifest_sha256"],
        "source_root": config["data"]["source_root"],
        "command": list(command),
        "hostname": os.uname().nodename,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "started_at_unix": time.time(),
    }


def train_stage(
    *,
    stage: str,
    stage_config: Mapping[str, object],
    model: InfiniDepth,
    optimizer: torch.optim.Optimizer,
    parameter_groups: Mapping[str, Sequence[torch.nn.Parameter]],
    samples: Sequence[HypersimDisparitySample],
    output: Path,
    device: torch.device,
    config: Mapping[str, object],
    config_sha256: str,
    generator: random.Random,
    total_step: int,
    initial_stage_step: int = 0,
) -> Tuple[Dict[str, object], int]:
    training = config["training"]
    model_cfg = config["model"]
    microbatch = int(training["microbatch_size"])
    global_batch = int(training["global_batch_size"])
    if global_batch % microbatch:
        raise ValueError("global_batch_size must be divisible by microbatch_size")
    accumulation = global_batch // microbatch
    minimum_steps = int(stage_config["min_steps"])
    maximum_steps = int(stage_config["max_steps"])
    eval_every = int(stage_config["eval_every"])
    full_eval_every = int(stage_config["full_eval_every"])
    checkpoint_every = int(stage_config["checkpoint_every"])
    if checkpoint_every <= 0 or checkpoint_every % full_eval_every:
        raise ValueError("checkpoint_every must be a positive multiple of full_eval_every")
    plateau_patience = int(stage_config.get("plateau_patience_evals", 0))
    plateau_threshold = float(stage_config.get("plateau_relative_improvement", 0.0))
    detach = stage == "stage1"
    query_hw = (int(model_cfg["height"]), int(model_cfg["width"]))
    chunk_size = int(model_cfg["query_chunk_size"])
    evaluation_ids = set(str(value) for value in config["evaluation"]["sample_ids"])
    sampled_eval_indices = [
        index for index, sample in enumerate(samples) if sample.sample_id in evaluation_ids
    ]
    if not sampled_eval_indices:
        sampled_eval_indices = list(range(min(8, len(samples))))
    best_score = float("inf")
    best_step = 0
    best_evaluation: Optional[Dict[str, object]] = None
    previous_full_score: Optional[float] = None
    stale_evaluations = 0
    final_evaluation: Optional[Dict[str, object]] = None
    checkpoint_dir = output / "checkpoints"
    history_path = output / "metrics" / "history.jsonl"
    start_time = time.time()
    model.train()
    for stage_step in range(initial_stage_step + 1, maximum_steps + 1):
        rates = _configure_stage_learning_rates(
            optimizer, parameter_groups["dino"], stage_config, stage_step
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated_metrics: Dict[str, float] = {}
        for _ in range(accumulation):
            indices = _sample_indices(generator, len(samples), microbatch)
            image, target, mask = _stack_batch(samples, indices, device)
            output_value = model.forward_dense_refined(
                image,
                query_hw=query_hw,
                num_refinement_steps=3,
                detach_base_from_refiner=detach,
                chunk_size=chunk_size,
            )
            loss, metrics = supervised_iteration_loss(
                output_value.disparity_sequence,
                target,
                mask,
                gradient_weight=float(training["gradient_weight"]),
                gradient_scales=int(training["gradient_scales"]),
            )
            (loss / accumulation).backward()
            monitored = {
                **{key: float(value.item()) for key, value in metrics.items()},
                **_refinement_monitor(output_value),
            }
            for key, value in monitored.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value / accumulation
        gradient_norms = {}
        for name, parameters in parameter_groups.items():
            active = [parameter for parameter in parameters if parameter.grad is not None]
            if active:
                norm = torch.nn.utils.clip_grad_norm_(
                    active, float(training["gradient_clip_norm"])
                )
                gradient_norms[name] = float(norm.item())
            else:
                gradient_norms[name] = None
        optimizer.step()
        total_step += 1

        should_sample_eval = stage_step % eval_every == 0
        should_full_eval = stage_step % full_eval_every == 0 or stage_step == maximum_steps
        if should_sample_eval or should_full_eval:
            indices = list(range(len(samples))) if should_full_eval else sampled_eval_indices
            evaluation = evaluate(
                model,
                samples,
                indices,
                device=device,
                query_hw=query_hw,
                chunk_size=chunk_size,
            )
            record = {
                "stage": stage,
                "stage_step": stage_step,
                "total_step": total_step,
                "scope": "full" if should_full_eval else "sampled",
                "training": accumulated_metrics,
                "learning_rates": rates,
                "gradient_norms": gradient_norms,
                "evaluation": evaluation,
                "elapsed_seconds": time.time() - start_time,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            _append_jsonl(history_path, record)
            if should_full_eval:
                final_evaluation = evaluation
                score = _selection_score(evaluation)
                if previous_full_score is None:
                    stale_evaluations = 0
                else:
                    relative = (previous_full_score - score) / max(abs(previous_full_score), 1e-12)
                    if relative >= plateau_threshold:
                        stale_evaluations = 0
                    else:
                        stale_evaluations += 1
                previous_full_score = score
                if score < best_score:
                    best_score = score
                    best_step = stage_step
                    best_evaluation = evaluation
                    _save_checkpoint(
                        checkpoint_dir / f"{stage}_best.pt",
                        model=model,
                        optimizer=optimizer,
                        stage=stage,
                        stage_step=stage_step,
                        total_step=total_step,
                        config_sha256=config_sha256,
                        evaluation=evaluation,
                        include_optimizer=False,
                    )
                should_stop = (
                    plateau_patience > 0
                    and stage_step >= minimum_steps
                    and stale_evaluations >= plateau_patience
                )
                if (
                    stage_step % checkpoint_every == 0
                    or stage_step == maximum_steps
                    or should_stop
                ):
                    _save_checkpoint(
                        checkpoint_dir / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        stage=stage,
                        stage_step=stage_step,
                        total_step=total_step,
                        config_sha256=config_sha256,
                        evaluation=evaluation,
                        include_optimizer=True,
                    )
                if should_stop:
                    break
    if final_evaluation is None:
        raise RuntimeError("Stage completed without a full evaluation")
    if best_evaluation is None:
        raise RuntimeError("Stage completed without selecting a best checkpoint")
    return {
        "stage": stage,
        "completed_steps": stage_step,
        "best_step": best_step,
        "best_score": best_score,
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "elapsed_seconds": time.time() - start_time,
    }, total_step


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[2]
    config_path = ensure_within(config_path, project_root, name="config")
    safe_root = Path(str(config["server"]["safe_root"]))
    output = ensure_within(args.output, safe_root, name="experiment output")
    if output.exists() and output.is_symlink():
        raise PermissionError("Experiment output may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    run = _selected_run(config, args.run_id)
    config_sha256 = _sha256(config_path)
    seed = int(config["seed"]) + int(run.get("seed_offset", 0))
    _seed_everything(seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training and evaluation require a CUDA server")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    samples = _load_samples(config, run, project_root)
    checkpoint_path = ensure_within(
        Path(str(config["model"]["checkpoint"])), safe_root, name="base checkpoint"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = InfiniDepth(model_path=str(checkpoint_path)).to(device)
    model.attach_disparity_refiner(
        backend="spconv",
        voxel_resolution=float(config["model"]["voxel_resolution"]),
        max_disparity_span=config["model"].get("max_disparity_span"),
    )
    groups = _parameter_groups(model)
    optimizer = _build_optimizer(groups, float(config["training"]["weight_decay"]))
    total_step = 0
    initial_stage = None
    initial_stage_step = 0
    if args.resume is not None:
        resume = ensure_within(args.resume, safe_root, name="resume checkpoint")
        state = _restore_checkpoint(resume, model, optimizer, config_sha256)
        total_step = int(state["total_step"])
        initial_stage = str(state["stage"])
        initial_stage_step = int(state["stage_step"])

    provenance = _provenance(
        project_root,
        config_path,
        config,
        run,
        ["python", "-m", "training.disparity_refiner.train", *os.sys.argv[1:]],
    )
    provenance["seed"] = seed
    provenance["sample_ids"] = [sample.sample_id for sample in samples]
    provenance["disparity_quantiles"] = {
        sample.sample_id: list(sample.disparity_quantiles) for sample in samples
    }
    _atomic_json(output / "provenance.json", provenance)

    reports = []
    stages = config["training"]["stages"]
    for stage in ("stage1", "joint"):
        if initial_stage is not None and stage != initial_stage and not reports:
            if initial_stage == "joint" and stage == "stage1":
                continue
        stage_start = initial_stage_step if initial_stage == stage else 0
        report, total_step = train_stage(
            stage=stage,
            stage_config=stages[stage],
            model=model,
            optimizer=optimizer,
            parameter_groups=groups,
            samples=samples,
            output=output,
            device=device,
            config=config,
            config_sha256=config_sha256,
            generator=random.Random(seed + total_step + 1),
            total_step=total_step,
            initial_stage_step=stage_start,
        )
        reports.append(report)
        initial_stage = None
        initial_stage_step = 0

    manifest = _checkpoint_manifest(output / "checkpoints")
    _atomic_json(output / "artifacts" / "checkpoint_manifest.json", manifest)
    final_report = {
        "format": "infinidepth-disparity-refiner-report-v1",
        "experiment_id": config["experiment_id"],
        "run_id": run["id"],
        "status": "completed",
        "stages": reports,
        "total_steps": total_step,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "completed_at_unix": time.time(),
    }
    _atomic_json(output / "metrics" / "report.json", final_report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in os.sys.argv:
            output_index = os.sys.argv.index("--output") + 1
            if output_index < len(os.sys.argv):
                candidate = Path(os.sys.argv[output_index]).expanduser().resolve()
                safe_root = Path("/mnt/data/home/zhuzichao")
                if candidate == safe_root or safe_root in candidate.parents:
                    _atomic_json(
                        candidate / "metrics" / "report.json",
                        {
                            "format": "infinidepth-disparity-refiner-report-v1",
                            "status": "failed",
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "failed_at_unix": time.time(),
                        },
                    )
        raise

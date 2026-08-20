from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from InfiniDepth.model import DisparityRefinementOutput, InfiniDepth
from training.disparity_refiner.data import (
    HypersimDisparityDataset,
    HypersimDisparitySample,
    ensure_within,
    load_manifest,
    preload_samples,
    select_manifest_entries,
)
from training.disparity_refiner.losses import disparity_metrics, supervised_iteration_loss


EVALUATION_STEPS = (0, 1, 3, 5)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class DenseRefinementTrainer(torch.nn.Module):
    """Expose the custom dense forward through DDP's regular forward path."""

    def __init__(self, model: InfiniDepth) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, **kwargs: object) -> DisparityRefinementOutput:
        return self.model.forward_dense_refined(image, **kwargs)


class TrainingPaused(RuntimeError):
    def __init__(self, stage: str, stage_step: int, total_step: int) -> None:
        super().__init__(f"{stage} paused at step {stage_step}")
        self.stage = stage
        self.stage_step = stage_step
        self.total_step = total_step


class ShuffledCycleSampler:
    def __init__(self, seed: int) -> None:
        self.generator = random.Random(seed)
        self.count = 0
        self.order: List[int] = []
        self.position = 0

    def sample_indices(self, count: int, batch_size: int) -> List[int]:
        if count <= 0:
            raise ValueError("Cannot sample an empty dataset")
        if self.count not in (0, count):
            raise ValueError("Dataset size changed while resuming sampling")
        self.count = count
        selected = []
        while len(selected) < batch_size:
            if self.position >= len(self.order):
                self.order = list(range(count))
                self.generator.shuffle(self.order)
                self.position = 0
            take = min(batch_size - len(selected), len(self.order) - self.position)
            selected.extend(self.order[self.position : self.position + take])
            self.position += take
        return selected

    def getstate(self) -> Mapping[str, object]:
        return {
            "count": self.count,
            "generator": self.generator.getstate(),
            "order": self.order,
            "position": self.position,
        }

    def setstate(self, state: object) -> None:
        if not isinstance(state, Mapping):
            self.generator.setstate(state)
            self.count, self.order, self.position = 0, [], 0
            return
        self.generator.setstate(state["generator"])
        self.count = int(state["count"])
        self.order = [int(value) for value in state["order"]]
        self.position = int(state["position"])

    def seed(self, seed: int) -> None:
        self.generator.seed(seed)
        self.count, self.order, self.position = 0, [], 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed-grid InfiniDepth native-disparity refiner"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _gradient_accumulation(global_batch: int, microbatch: int, world_size: int) -> int:
    denominator = microbatch * world_size
    if min(global_batch, microbatch, world_size) <= 0 or global_batch % denominator:
        raise ValueError(
            "global_batch_size must be divisible by microbatch_size * world_size"
        )
    return global_batch // denominator


def _rank_sample_indices(
    global_indices: Sequence[int], rank: int, microbatch: int, world_size: int
) -> List[int]:
    expected = microbatch * world_size
    if len(global_indices) != expected:
        raise ValueError(f"Expected {expected} global indices, got {len(global_indices)}")
    start = rank * microbatch
    return list(global_indices[start : start + microbatch])


def _init_distributed(device_argument: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device = torch.device(device_argument)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Training and evaluation require a CUDA server")
        torch.cuda.set_device(device)
        return DistributedContext(0, int(device.index or 0), 1, device)
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        backend=str(dist.get_backend()),
    )


def _barrier(distributed: DistributedContext) -> None:
    if distributed.enabled:
        dist.barrier()


def _broadcast_object(value: object, distributed: DistributedContext) -> object:
    if not distributed.enabled:
        return value
    values = [value if distributed.is_main else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _raise_if_any_rank_failed(
    error: Optional[BaseException], distributed: DistributedContext
) -> None:
    if not distributed.enabled:
        if error is not None:
            raise error
        return
    failed = torch.tensor(
        int(error is not None), device=distributed.device, dtype=torch.int32
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if not int(failed.item()):
        return
    messages: List[Optional[str]] = [None] * distributed.world_size
    dist.all_gather_object(
        messages,
        None if error is None else f"rank {distributed.rank}: {type(error).__name__}: {error}",
    )
    raise FloatingPointError(
        "Synchronized DDP failure before optimizer.step(): "
        + "; ".join(message for message in messages if message)
    )


def _reduce_metrics(
    values: Mapping[str, float], distributed: DistributedContext
) -> Dict[str, float]:
    if not distributed.enabled:
        return dict(values)
    reduced = {}
    for key, value in values.items():
        tensor = torch.tensor(value, device=distributed.device, dtype=torch.float64)
        if key.endswith("_min"):
            operation = dist.ReduceOp.MIN
        elif key.endswith("_max"):
            operation = dist.ReduceOp.MAX
        else:
            operation = dist.ReduceOp.SUM
        dist.all_reduce(tensor, op=operation)
        result = float(tensor.item())
        if operation == dist.ReduceOp.SUM:
            result /= distributed.world_size
        reduced[key] = result
    return reduced


def _wrap_for_training(
    model: InfiniDepth, distributed: DistributedContext
) -> torch.nn.Module:
    trainer = DenseRefinementTrainer(model)
    if not distributed.enabled:
        return trainer
    return DistributedDataParallel(
        trainer,
        device_ids=[distributed.local_rank],
        output_device=distributed.local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
    )


def _accumulation_context(
    training_model: torch.nn.Module,
    distributed: DistributedContext,
    accumulation_step: int,
    accumulation: int,
) -> object:
    if distributed.enabled and accumulation_step < accumulation - 1:
        return training_model.no_sync()  # type: ignore[attr-defined]
    return nullcontext()


def _ddp_rewrap_required(trainable_before: bool, trainable_after: bool) -> bool:
    return trainable_before != trainable_after


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


def _capture_process_rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _capture_rng_state(generator: object) -> Dict[str, object]:
    return {
        **_capture_process_rng_state(),
        "sampling": generator.getstate(),  # type: ignore[attr-defined]
    }


def _gather_rng_states(
    generator: object, distributed: DistributedContext
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    process_rng = _capture_process_rng_state()
    sampling_state = generator.getstate()  # type: ignore[attr-defined]
    if distributed.enabled:
        gathered: List[Optional[Dict[str, object]]] = [None] * distributed.world_size
        dist.all_gather_object(
            gathered, {"process": process_rng, "sampling": sampling_state}
        )
        values = [value for value in gathered if value is not None]
        if any(value["sampling"] != values[0]["sampling"] for value in values[1:]):
            raise RuntimeError("DDP sampler states diverged across ranks")
        ranks = [value["process"] for value in values]
    else:
        ranks = [process_rng]
    legacy = {**ranks[0], "sampling": sampling_state}
    return legacy, ranks


def _restore_rng_state(
    state: Mapping[str, object],
    generator: object,
    rank: int = 0,
) -> bool:
    resume = state.get("resume_state", {})
    if not isinstance(resume, Mapping):
        return False
    rng = resume.get("rng")
    if not isinstance(rng, Mapping):
        return False
    rank_rng = resume.get("rank_rng")
    process_rng = rng
    if isinstance(rank_rng, Sequence) and rank < len(rank_rng):
        candidate = rank_rng[rank]
        if isinstance(candidate, Mapping):
            process_rng = candidate
    random.setstate(process_rng["python"])
    np.random.set_state(process_rng["numpy"])
    torch.set_rng_state(process_rng["torch_cpu"].cpu())
    if process_rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(process_rng["torch_cuda"].cpu())
    generator.setstate(rng["sampling"])  # type: ignore[attr-defined]
    return True


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
    *,
    split: str = "train",
    sample_ids: Optional[Sequence[str]] = None,
) -> Sequence[HypersimDisparitySample]:
    data = config["data"]
    manifest_path = ensure_within(
        project_root / str(data["manifest"]), project_root, name="manifest"
    )
    manifest = load_manifest(manifest_path, str(data["manifest_sha256"]))
    requested = (
        run.get("sample_ids", data.get("sample_ids"))
        if split == "train" and sample_ids is None
        else sample_ids
    )
    source_root = Path(str(data["source_root"]))
    if source_root.resolve() != Path("/nas1/datasets/hypersim/raw"):
        raise PermissionError("Hypersim source root must remain /nas1/datasets/hypersim/raw")
    entries = select_manifest_entries(
        manifest,
        source_root=source_root,
        split=split,
        sample_ids=requested,
    )
    if split == "train":
        excluded = {str(value) for value in data.get("excluded_sample_ids", [])}
        present = {str(entry["id"]) for entry in entries}
        missing = excluded - present if requested is None else set()
        if missing:
            raise ValueError(f"Excluded train samples are absent: {sorted(missing)}")
        entries = [entry for entry in entries if str(entry["id"]) not in excluded]
    expected = int(
        data["expected_train_count"]
        if split == "train"
        else len(requested or entries)
    )
    if len(entries) != expected:
        raise ValueError(f"Expected {expected} {split} samples, got {len(entries)}")
    model_cfg = config["model"]
    cache_root = None
    if data.get("local_cache") is not None:
        cache_root = ensure_within(
            Path(str(data["local_cache"])),
            Path(str(config["server"]["cache"])),
            name="Hypersim local cache",
        )
    if bool(data.get("lazy_loading", False)):
        return HypersimDisparityDataset(
            entries,
            source_root=source_root,
            height=int(model_cfg["height"]),
            width=int(model_cfg["width"]),
            structure_selections=_structure_selections(config),
            cache_root=cache_root,
        )
    return preload_samples(
        entries,
        source_root=source_root,
        height=int(model_cfg["height"]),
        width=int(model_cfg["width"]),
        structure_selections=_structure_selections(config),
        cache_root=cache_root,
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
    generator: object,
    count: int,
    batch_size: int,
) -> List[int]:
    shuffled_cycle = getattr(generator, "sample_indices", None)
    if shuffled_cycle is not None:
        return list(shuffled_cycle(count, batch_size))
    if count == 1:
        return [0] * batch_size
    return [generator.randrange(count) for _ in range(batch_size)]  # type: ignore[attr-defined]


def _peek_global_sample_indices(
    generator: object, count: int, batch_size: int, accumulation: int
) -> List[int]:
    clone = copy.deepcopy(generator)
    values = []
    for _ in range(accumulation):
        values.extend(_sample_indices(clone, count, batch_size))
    return values


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
    residual_scale: float = 1.0,
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
            residual_scale=residual_scale,
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
    generator: Optional[object] = None,
    stage_state: Optional[Mapping[str, object]] = None,
    completed_stage_reports: Sequence[Mapping[str, object]] = (),
    distributed_state: Optional[Mapping[str, object]] = None,
    gathered_rng: Optional[Tuple[Mapping[str, object], Sequence[Mapping[str, object]]]] = None,
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
    if distributed_state is not None:
        checkpoint["distributed"] = dict(distributed_state)
    if include_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
        if generator is None or stage_state is None:
            raise ValueError("Resumable checkpoints require RNG and stage state")
        legacy_rng, rank_rng = (
            gathered_rng if gathered_rng is not None else (_capture_rng_state(generator), [])
        )
        checkpoint["resume_state"] = {
            "completed_stage_reports": list(completed_stage_reports),
            "rng": dict(legacy_rng),
            "stage": dict(stage_state),
        }
        if rank_rng:
            checkpoint["resume_state"]["rank_rng"] = list(rank_rng)
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _restore_checkpoint(
    path: Path,
    model: InfiniDepth,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    world_size: int = 1,
) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if checkpoint.get("format") != "infinidepth-disparity-refiner-v1":
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("Resume checkpoint was produced by another config")
    distributed_state = checkpoint.get("distributed")
    if isinstance(distributed_state, Mapping):
        saved_world_size = int(distributed_state.get("world_size", 1))
        if saved_world_size != world_size:
            raise ValueError(
                f"Checkpoint world_size {saved_world_size} cannot resume with {world_size}"
            )
    elif world_size != 1:
        raise ValueError("Legacy single-GPU checkpoints cannot resume a DDP run")
    if not checkpoint.get("optimizer_included") or "optimizer" not in checkpoint:
        raise ValueError("Resume requires last.pt with optimizer state")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _checkpoint_stage_state(checkpoint: Mapping[str, object]) -> Dict[str, object]:
    resume = checkpoint.get("resume_state", {})
    saved = resume.get("stage") if isinstance(resume, Mapping) else None
    if isinstance(saved, Mapping):
        return dict(saved)
    evaluation = checkpoint["evaluation"]
    return {
        "best_evaluation": evaluation,
        "best_score": _selection_score(evaluation),
        "best_step": int(checkpoint["stage_step"]),
        "elapsed_seconds": 0.0,
        "final_evaluation": evaluation,
        "previous_full_score": _selection_score(evaluation),
        "stale_evaluations": 0,
    }


def _stage_report(stage: str, stage_step: int, state: Mapping[str, object]) -> Dict[str, object]:
    best_evaluation = state.get("best_evaluation")
    final_evaluation = state.get("final_evaluation")
    if best_evaluation is None or final_evaluation is None:
        raise RuntimeError(f"Cannot complete {stage} without full evaluation state")
    report = {
        "stage": stage,
        "completed_steps": int(stage_step),
        "best_step": int(state["best_step"]),
        "best_score": float(state["best_score"]),
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "elapsed_seconds": float(state.get("elapsed_seconds", 0.0)),
    }
    if "resume_sample_sequence_verified" in state:
        report["resume_sample_sequence_verified"] = bool(
            state["resume_sample_sequence_verified"]
        )
    return report


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


def _distributed_metadata(
    config: Mapping[str, object], distributed: DistributedContext
) -> Dict[str, object]:
    training = config["training"]
    microbatch = int(training["microbatch_size"])
    global_batch = int(training["global_batch_size"])
    nccl_version = torch.cuda.nccl.version() if torch.cuda.is_available() else None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_mapping = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        selected = None if visible is None else set(visible.split(","))
        for line in completed.stdout.splitlines():
            index, uuid, name = (value.strip() for value in line.split(",", 2))
            if selected is None or index in selected:
                gpu_mapping.append({"physical_index": index, "uuid": uuid, "name": name})
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu_mapping = []
    return {
        "world_size": distributed.world_size,
        "microbatch_size_per_rank": microbatch,
        "gradient_accumulation": _gradient_accumulation(
            global_batch, microbatch, distributed.world_size
        ),
        "global_batch_size": global_batch,
        "backend": distributed.backend,
        "nccl_version": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
        "cuda_visible_devices": visible,
        "gpu_mapping": gpu_mapping,
    }


def _peak_cuda_memory(distributed: DistributedContext) -> int:
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(distributed.device),
        device=distributed.device,
        dtype=torch.int64,
    )
    if distributed.enabled:
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return int(peak.item())


def _parameter_checksums(
    model: InfiniDepth, distributed: DistributedContext
) -> Tuple[List[List[float]], bool]:
    checksum = torch.zeros(2, device=distributed.device, dtype=torch.float64)
    with torch.no_grad():
        for parameter in model.parameters():
            value = parameter.detach().double()
            checksum[0] += value.sum()
            checksum[1] += value.square().sum()
    local = [float(value) for value in checksum.cpu().tolist()]
    if distributed.enabled:
        gathered: List[Optional[List[float]]] = [None] * distributed.world_size
        dist.all_gather_object(gathered, local)
        values = [value for value in gathered if value is not None]
    else:
        values = [local]
    consistent = all(
        math.isclose(value[0], values[0][0], rel_tol=1e-10, abs_tol=1e-8)
        and math.isclose(value[1], values[0][1], rel_tol=1e-10, abs_tol=1e-8)
        for value in values[1:]
    )
    return values, consistent


def _pause_requested(
    output: Path,
    distributed: DistributedContext,
    *,
    stage: str,
    stage_step: int,
) -> bool:
    requested = False
    path = output / "control" / "pause.request"
    if distributed.is_main and path.is_file():
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            request = {}
        requested_stage = request.get("stage")
        requested_step = int(request.get("after_stage_step", 0))
        requested = (
            requested_stage in (None, stage) and stage_step >= requested_step
        )
    return bool(_broadcast_object(requested, distributed))


def _provenance(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, object],
    run: Mapping[str, object],
    command: Sequence[str],
    distributed: Optional[DistributedContext] = None,
) -> Dict[str, object]:
    checkpoint = Path(str(config["model"]["checkpoint"]))
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("data.py"),
        project_root / "InfiniDepth/model/disparity_refiner.py",
        project_root / "experiment/schedule_exp3.py",
    ]
    launcher_command = os.environ.get("INFINIDEPTH_LAUNCH_COMMAND")
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
        "launcher_command": json.loads(launcher_command) if launcher_command else None,
        "source_sha256": {
            str(path.relative_to(project_root)): _sha256(path) for path in source_paths
        },
        "hostname": os.uname().nodename,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "distributed": None
        if distributed is None
        else _distributed_metadata(config, distributed),
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
    evaluation_samples: Sequence[HypersimDisparitySample],
    output: Path,
    device: torch.device,
    config: Mapping[str, object],
    config_sha256: str,
    generator: object,
    total_step: int,
    initial_stage_step: int = 0,
    initial_stage_state: Optional[Mapping[str, object]] = None,
    completed_stage_reports: Sequence[Mapping[str, object]] = (),
    distributed: Optional[DistributedContext] = None,
) -> Tuple[Dict[str, object], int]:
    distributed = distributed or DistributedContext(0, 0, 1, device)
    training = config["training"]
    model_cfg = config["model"]
    microbatch = int(training["microbatch_size"])
    global_batch = int(training["global_batch_size"])
    accumulation = _gradient_accumulation(
        global_batch, microbatch, distributed.world_size
    )
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
    evaluation_sample_ids = (
        evaluation_samples.sample_ids
        if isinstance(evaluation_samples, HypersimDisparityDataset)
        else [sample.sample_id for sample in evaluation_samples]
    )
    sampled_eval_indices = [
        index
        for index, sample_id in enumerate(evaluation_sample_ids)
        if sample_id in evaluation_ids
    ]
    if not sampled_eval_indices:
        sampled_eval_indices = list(range(min(8, len(evaluation_samples))))
    restored = dict(initial_stage_state or {})
    best_score = float(restored.get("best_score", float("inf")))
    best_step = int(restored.get("best_step", 0))
    best_evaluation = restored.get("best_evaluation")
    previous_full_score = restored.get("previous_full_score")
    stale_evaluations = int(restored.get("stale_evaluations", 0))
    final_evaluation = restored.get("final_evaluation")
    last_evaluation = restored.get("last_evaluation", final_evaluation)
    expected_next_indices = restored.get("expected_next_global_indices")
    resume_sample_sequence_verified = bool(
        restored.get("resume_sample_sequence_verified", False)
    )
    elapsed_before = float(restored.get("elapsed_seconds", 0.0))
    checkpoint_dir = output / "checkpoints"
    history_path = output / "metrics" / "history.jsonl"
    stage_start_time = time.time()
    model.train()
    next_step = initial_stage_step + 1
    _configure_stage_learning_rates(
        optimizer, parameter_groups["dino"], stage_config, next_step
    )
    training_model = _wrap_for_training(model, distributed)
    for stage_step in range(initial_stage_step + 1, maximum_steps + 1):
        optimizer_step_started = time.time()
        train_dino_before = any(
            parameter.requires_grad for parameter in parameter_groups["dino"]
        )
        rates = _configure_stage_learning_rates(
            optimizer, parameter_groups["dino"], stage_config, stage_step
        )
        train_dino_after = any(
            parameter.requires_grad for parameter in parameter_groups["dino"]
        )
        if _ddp_rewrap_required(train_dino_before, train_dino_after):
            _barrier(distributed)
            del training_model
            training_model = _wrap_for_training(model, distributed)
            _barrier(distributed)
        optimizer.zero_grad(set_to_none=True)
        accumulated_metrics: Dict[str, float] = {}
        global_step_indices: List[int] = []
        for accumulation_step in range(accumulation):
            global_indices = _sample_indices(
                generator, len(samples), microbatch * distributed.world_size
            )
            global_step_indices.extend(global_indices)
            indices = _rank_sample_indices(
                global_indices,
                distributed.rank,
                microbatch,
                distributed.world_size,
            )
            image, target, mask = _stack_batch(samples, indices, device)
            local_error: Optional[BaseException] = None
            loss: Optional[torch.Tensor] = None
            monitored: Dict[str, float] = {}
            sync_context = _accumulation_context(
                training_model, distributed, accumulation_step, accumulation
            )
            with sync_context:
                try:
                    output_value = training_model(
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
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Training loss is non-finite")
                    monitored = {
                        **{key: float(value.item()) for key, value in metrics.items()},
                        **_refinement_monitor(output_value),
                    }
                    for key, value in monitored.items():
                        if not math.isfinite(value):
                            raise FloatingPointError(
                                f"Training monitor {key} is non-finite"
                            )
                except BaseException as exc:
                    local_error = exc
                _raise_if_any_rank_failed(local_error, distributed)
                assert loss is not None
                (loss / accumulation).backward()
            for key, value in monitored.items():
                if key.endswith("_min"):
                    accumulated_metrics[key] = min(
                        accumulated_metrics.get(key, value), value
                    )
                elif key.endswith("_max"):
                    accumulated_metrics[key] = max(
                        accumulated_metrics.get(key, value), value
                    )
                else:
                    accumulated_metrics[key] = (
                        accumulated_metrics.get(key, 0.0) + value / accumulation
                    )
        if expected_next_indices is not None:
            if global_step_indices != [int(value) for value in expected_next_indices]:
                raise RuntimeError("Global sampler sequence changed after checkpoint resume")
            expected_next_indices = None
            resume_sample_sequence_verified = True
        gradient_norms = {}
        gradient_error: Optional[BaseException] = None
        for name, parameters in parameter_groups.items():
            active = [parameter for parameter in parameters if parameter.grad is not None]
            if active:
                norm = torch.nn.utils.clip_grad_norm_(
                    active,
                    float(training["gradient_clip_norm"]),
                    error_if_nonfinite=False,
                )
                gradient_norms[name] = float(norm.item())
                if not bool(torch.isfinite(norm)):
                    gradient_error = FloatingPointError(
                        f"Gradient norm for {name} is non-finite"
                    )
            else:
                gradient_norms[name] = None
        _raise_if_any_rank_failed(gradient_error, distributed)
        optimizer.step()
        total_step += 1
        optimizer_step_seconds = time.time() - optimizer_step_started

        pause_after_step = _pause_requested(
            output, distributed, stage=stage, stage_step=stage_step
        )
        should_sample_eval = not pause_after_step and stage_step % eval_every == 0
        should_full_eval = not pause_after_step and (
            stage_step % full_eval_every == 0 or stage_step == maximum_steps
        )
        if should_sample_eval or should_full_eval:
            accumulated_metrics = _reduce_metrics(accumulated_metrics, distributed)
            full_eval_ids = set(
                str(value)
                for value in config["evaluation"].get(
                    "full_sample_ids", evaluation_sample_ids
                )
            )
            full_eval_indices = [
                index for index, sample_id in enumerate(evaluation_sample_ids)
                if sample_id in full_eval_ids
            ]
            if not full_eval_indices:
                raise ValueError("No configured full evaluation samples are in the dataset")
            indices = full_eval_indices if should_full_eval else sampled_eval_indices
            _barrier(distributed)
            evaluation_payload: object = None
            if distributed.is_main:
                try:
                    value = evaluate(
                        model,
                        evaluation_samples,
                        indices,
                        device=device,
                        query_hw=query_hw,
                        chunk_size=chunk_size,
                    )
                    score = _selection_score(value)
                    if not math.isfinite(score):
                        raise FloatingPointError("Evaluation score is non-finite")
                    evaluation_payload = {"evaluation": value, "error": None}
                except BaseException as exc:
                    evaluation_payload = {
                        "evaluation": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            evaluation_payload = _broadcast_object(evaluation_payload, distributed)
            if not isinstance(evaluation_payload, Mapping):
                raise RuntimeError("Invalid distributed evaluation payload")
            if evaluation_payload.get("error"):
                raise FloatingPointError(str(evaluation_payload["error"]))
            evaluation = evaluation_payload["evaluation"]
            if not isinstance(evaluation, Mapping):
                raise RuntimeError("Evaluation did not return a mapping")
            last_evaluation = evaluation
            peak_memory = _peak_cuda_memory(distributed)
            record = {
                "stage": stage,
                "stage_step": stage_step,
                "total_step": total_step,
                "scope": "full" if should_full_eval else "sampled",
                "training": accumulated_metrics,
                "learning_rates": rates,
                "gradient_norms": gradient_norms,
                "evaluation": evaluation,
                "elapsed_seconds": elapsed_before + time.time() - stage_start_time,
                "peak_cuda_memory_bytes": peak_memory,
                "world_size": distributed.world_size,
                "microbatch_size_per_rank": microbatch,
                "gradient_accumulation": accumulation,
                "optimizer_step_seconds": optimizer_step_seconds,
                "global_sample_indices": global_step_indices,
            }
            if distributed.is_main:
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
                    if distributed.is_main:
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
                            distributed_state=_distributed_metadata(config, distributed),
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
                    stage_state = {
                        "best_evaluation": best_evaluation,
                        "best_score": best_score,
                        "best_step": best_step,
                        "elapsed_seconds": elapsed_before + time.time() - stage_start_time,
                        "final_evaluation": final_evaluation,
                        "previous_full_score": previous_full_score,
                        "stale_evaluations": stale_evaluations,
                        "last_evaluation": last_evaluation,
                        "expected_next_global_indices": _peek_global_sample_indices(
                            generator,
                            len(samples),
                            microbatch * distributed.world_size,
                            accumulation,
                        ),
                        "resume_sample_sequence_verified": resume_sample_sequence_verified,
                    }
                    gathered_rng = _gather_rng_states(generator, distributed)
                    if distributed.is_main:
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
                            generator=generator,
                            stage_state=stage_state,
                            completed_stage_reports=completed_stage_reports,
                            distributed_state=_distributed_metadata(config, distributed),
                            gathered_rng=gathered_rng,
                        )
                if should_stop:
                    break
        if pause_after_step:
            stage_state = {
                "best_evaluation": best_evaluation,
                "best_score": best_score,
                "best_step": best_step,
                "elapsed_seconds": elapsed_before + time.time() - stage_start_time,
                "final_evaluation": final_evaluation,
                "previous_full_score": previous_full_score,
                "stale_evaluations": stale_evaluations,
                "last_evaluation": last_evaluation,
                "expected_next_global_indices": _peek_global_sample_indices(
                    generator,
                    len(samples),
                    microbatch * distributed.world_size,
                    accumulation,
                ),
                "resume_sample_sequence_verified": resume_sample_sequence_verified,
            }
            gathered_rng = _gather_rng_states(generator, distributed)
            if distributed.is_main:
                _save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    stage=stage,
                    stage_step=stage_step,
                    total_step=total_step,
                    config_sha256=config_sha256,
                    evaluation=last_evaluation or {},
                    include_optimizer=True,
                    generator=generator,
                    stage_state=stage_state,
                    completed_stage_reports=completed_stage_reports,
                    distributed_state=_distributed_metadata(config, distributed),
                    gathered_rng=gathered_rng,
                )
                (output / "control" / "pause.request").unlink(missing_ok=True)
            _barrier(distributed)
            raise TrainingPaused(stage, stage_step, total_step)
    if final_evaluation is None:
        raise RuntimeError("Stage completed without a full evaluation")
    if best_evaluation is None:
        raise RuntimeError("Stage completed without selecting a best checkpoint")
    state = {
        "best_evaluation": best_evaluation,
        "best_score": best_score,
        "best_step": best_step,
        "elapsed_seconds": elapsed_before + time.time() - stage_start_time,
        "final_evaluation": final_evaluation,
        "resume_sample_sequence_verified": resume_sample_sequence_verified,
    }
    return _stage_report(stage, stage_step, state), total_step


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[2]
    safe_root = Path(str(config["server"]["safe_root"]))
    config_root = safe_root if args.smoke else project_root
    config_path = ensure_within(config_path, config_root, name="config")
    output = ensure_within(args.output, safe_root, name="experiment output")
    if output.exists() and output.is_symlink():
        raise PermissionError("Experiment output may not be a symlink")
    run = _selected_run(config, args.run_id)
    config_sha256 = _sha256(config_path)
    seed = int(config["seed"]) + int(run.get("seed_offset", 0))
    distributed = _init_distributed(args.device)
    device = distributed.device
    _seed_everything(seed + distributed.rank)
    if distributed.is_main:
        output.mkdir(parents=True, exist_ok=True)
    _barrier(distributed)
    torch.cuda.reset_peak_memory_stats(device)

    samples = _load_samples(config, run, project_root)
    evaluation_split = config["evaluation"].get("split")
    if evaluation_split is None:
        evaluation_samples = samples
    else:
        evaluation_samples = _load_samples(
            config,
            run,
            project_root,
            split=str(evaluation_split),
            sample_ids=[
                str(value)
                for value in config["evaluation"].get(
                    "full_sample_ids", config["evaluation"]["sample_ids"]
                )
            ],
        )
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
    initial_stage_state: Optional[Mapping[str, object]] = None
    reports: List[Dict[str, object]] = []
    sampling = str(config["training"].get("sampling", "with_replacement"))
    if sampling == "with_replacement":
        generator: object = random.Random(seed + 1)
    elif sampling == "shuffled_cycle":
        generator = ShuffledCycleSampler(seed + 1)
    else:
        raise ValueError(f"Unsupported training sampling mode: {sampling}")
    if args.resume is not None:
        resume = ensure_within(args.resume, safe_root, name="resume checkpoint")
        state = _restore_checkpoint(
            resume,
            model,
            optimizer,
            config_sha256,
            distributed.world_size,
        )
        total_step = int(state["total_step"])
        initial_stage = str(state["stage"])
        initial_stage_step = int(state["stage_step"])
        initial_stage_state = _checkpoint_stage_state(state)
        resume_state = state.get("resume_state", {})
        if isinstance(resume_state, Mapping):
            reports = [dict(value) for value in resume_state.get("completed_stage_reports", [])]
        if not _restore_rng_state(state, generator, distributed.rank):
            generator.seed(seed + total_step + 1)  # type: ignore[attr-defined]

    if distributed.is_main:
        provenance = _provenance(
            project_root,
            config_path,
            config,
            run,
            ["python", "-m", "training.disparity_refiner.train", *os.sys.argv[1:]],
            distributed,
        )
        provenance["seed"] = seed
        provenance["sample_ids"] = (
            list(samples.sample_ids)
            if isinstance(samples, HypersimDisparityDataset)
            else [sample.sample_id for sample in samples]
        )
        if not isinstance(samples, HypersimDisparityDataset):
            provenance["disparity_quantiles"] = {
                sample.sample_id: list(sample.disparity_quantiles) for sample in samples
            }
        provenance["evaluation_sample_ids"] = (
            list(evaluation_samples.sample_ids)
            if isinstance(evaluation_samples, HypersimDisparityDataset)
            else [sample.sample_id for sample in evaluation_samples]
        )
        _atomic_json(output / "provenance.json", provenance)
        _atomic_json(
            output / "metrics" / "report.json",
            {
                "experiment_id": config["experiment_id"],
                "format": "infinidepth-disparity-refiner-report-v1",
                "resumed_from": str(args.resume) if args.resume is not None else None,
                "run_id": run["id"],
                "started_at_unix": time.time(),
                "status": "running",
                "world_size": distributed.world_size,
            },
        )

    stages = config["training"]["stages"]
    for stage in ("stage1", "joint"):
        if initial_stage == "joint" and stage == "stage1":
            if not any(report.get("stage") == "stage1" for report in reports):
                stage1_path = output / "metrics" / "stage1_report.json"
                if not stage1_path.is_file():
                    raise RuntimeError("Joint resume requires the persisted Stage1 report")
                reports.append(json.loads(stage1_path.read_text(encoding="utf-8")))
            continue
        stage_start = initial_stage_step if initial_stage == stage else 0
        if initial_stage == stage and stage_start >= int(stages[stage]["max_steps"]):
            report = _stage_report(stage, stage_start, initial_stage_state or {})
        else:
            report, total_step = train_stage(
                stage=stage,
                stage_config=stages[stage],
                model=model,
                optimizer=optimizer,
                parameter_groups=groups,
                samples=samples,
                evaluation_samples=evaluation_samples,
                output=output,
                device=device,
                config=config,
                config_sha256=config_sha256,
                generator=generator,
                total_step=total_step,
                initial_stage_step=stage_start,
                initial_stage_state=initial_stage_state if initial_stage == stage else None,
                completed_stage_reports=reports,
                distributed=distributed,
            )
        reports = [value for value in reports if value.get("stage") != stage]
        reports.append(report)
        if distributed.is_main:
            _atomic_json(output / "metrics" / f"{stage}_report.json", report)
        _barrier(distributed)
        initial_stage = None
        initial_stage_step = 0
        initial_stage_state = None

    peak_memory = _peak_cuda_memory(distributed)
    checksums, parameters_consistent = _parameter_checksums(model, distributed)
    if not parameters_consistent:
        raise RuntimeError("Model parameters diverged across DDP ranks")
    if not distributed.is_main:
        return
    manifest = _checkpoint_manifest(output / "checkpoints")
    _atomic_json(output / "artifacts" / "checkpoint_manifest.json", manifest)
    final_report = {
        "format": "infinidepth-disparity-refiner-report-v1",
        "experiment_id": config["experiment_id"],
        "run_id": run["id"],
        "status": "completed",
        "stages": reports,
        "total_steps": total_step,
        "peak_cuda_memory_bytes": peak_memory,
        "world_size": distributed.world_size,
        "parameter_checksums": checksums,
        "parameters_consistent": parameters_consistent,
        "completed_at_unix": time.time(),
    }
    _atomic_json(output / "metrics" / "report.json", final_report)


if __name__ == "__main__":
    distributed_context: Optional[DistributedContext] = None
    try:
        main()
    except TrainingPaused as exc:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0 and "--output" in os.sys.argv:
            output_index = os.sys.argv.index("--output") + 1
            if output_index < len(os.sys.argv):
                candidate = Path(os.sys.argv[output_index]).expanduser().resolve()
                safe_root = Path("/mnt/data/home/zhuzichao")
                if candidate != safe_root and safe_root in candidate.parents:
                    _atomic_json(
                        candidate / "metrics" / "report.json",
                        {
                            "format": "infinidepth-disparity-refiner-report-v1",
                            "status": "paused",
                            "stage": exc.stage,
                            "stage_step": exc.stage_step,
                            "total_step": exc.total_step,
                            "paused_at_unix": time.time(),
                        },
                    )
    except Exception as exc:
        if int(os.environ.get("RANK", "0")) == 0 and "--output" in os.sys.argv:
            output_index = os.sys.argv.index("--output") + 1
            if output_index < len(os.sys.argv):
                candidate = Path(os.sys.argv[output_index]).expanduser().resolve()
                safe_root = Path("/mnt/data/home/zhuzichao")
                if candidate != safe_root and safe_root in candidate.parents:
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
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

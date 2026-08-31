from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from training.disparity_refiner.data import HypersimDisparitySample


@dataclass(frozen=True)
class LidarBatch:
    image: torch.Tensor
    target_disparity: torch.Tensor
    valid_mask: torch.Tensor
    prompt_disparity: torch.Tensor
    prompt_mask: torch.Tensor
    reference_scale: torch.Tensor


def stable_prompt_seed(seed: int, step: int, sample_id: str) -> int:
    payload = f"{int(seed)}:{int(step)}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def camera_rays(
    metadata: Mapping[str, object],
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    matrix = torch.as_tensor(metadata["M_cam_from_uv"], dtype=torch.float32, device=device)
    u = torch.linspace(
        -1.0 + 1.0 / width,
        1.0 - 1.0 / width,
        width,
        dtype=torch.float32,
        device=device,
    )
    v = torch.linspace(
        1.0 - 1.0 / height,
        -1.0 + 1.0 / height,
        height,
        dtype=torch.float32,
        device=device,
    )
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    uv1 = torch.stack((grid_u, grid_v, torch.ones_like(grid_u)), dim=-1)
    rays = uv1 @ matrix.T
    rays = F.normalize(rays, dim=-1, eps=1e-8)
    return rays * rays.new_tensor([1.0, -1.0, -1.0])


def make_lidar_prompt(
    sample: HypersimDisparitySample,
    settings: Mapping[str, object],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a deterministic multi-beam LiDAR pattern into the camera image."""
    radial = sample.radial_depth.float()
    valid = sample.valid_mask.bool()
    height, width = radial.shape
    beams = int(settings.get("vertical_beams", 64))
    stride = int(settings.get("horizontal_stride", 4))
    dropout = float(settings.get("dropout", 0.0))
    if beams < 2 or stride < 1 or not 0.0 <= dropout < 1.0:
        raise ValueError("Invalid LiDAR beam, stride, or dropout setting")

    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    phase = int(torch.randint(stride, (), generator=generator).item()) if stride > 1 else 0
    columns = torch.arange(phase, width, stride)
    rays = camera_rays(sample.metadata, height, width)
    elevation = torch.atan2(
        rays[..., 1],
        torch.linalg.vector_norm(rays[..., (0, 2)], dim=-1).clamp_min(1e-8),
    )
    margin = float(settings.get("vertical_margin_fraction", 0.02))
    if not 0.0 <= margin < 0.5:
        raise ValueError("vertical_margin_fraction must be in [0, 0.5)")
    low = elevation.amin()
    high = elevation.amax()
    span = high - low
    levels = torch.linspace(low + margin * span, high - margin * span, beams)
    distances = (elevation[:, columns][None] - levels[:, None, None]).abs()
    distances = distances.masked_fill(~valid[:, columns][None], float("inf"))
    rows = distances.argmin(dim=1)

    prompt_mask = torch.zeros_like(valid)
    beam_ids = torch.arange(beams)[:, None].expand_as(rows)
    column_ids = columns[None].expand_as(rows)
    selected = torch.isfinite(distances[beam_ids, rows, torch.arange(columns.numel())[None]])
    prompt_mask[rows[selected], column_ids[selected]] = True
    if dropout > 0.0:
        positions = prompt_mask.nonzero(as_tuple=False)
        keep = torch.rand(positions.shape[0], generator=generator) >= dropout
        prompt_mask.zero_()
        kept = positions[keep]
        prompt_mask[kept[:, 0], kept[:, 1]] = True
    prompt_mask &= valid
    if int(prompt_mask.sum()) <= 5:
        raise ValueError(f"LiDAR prompt for {sample.sample_id} has fewer than six valid points")

    raw_disparity = torch.zeros_like(radial)
    raw_disparity[valid] = radial[valid].reciprocal()
    prompt_disparity = torch.where(prompt_mask, raw_disparity, torch.zeros_like(raw_disparity))
    reference_scale = torch.quantile(prompt_disparity[prompt_mask], 0.5)
    if not bool(torch.isfinite(reference_scale)) or float(reference_scale) <= 0.0:
        raise ValueError(f"LiDAR prompt for {sample.sample_id} has an invalid disparity median")
    target = torch.where(valid, raw_disparity / reference_scale, torch.zeros_like(raw_disparity))
    return prompt_disparity, prompt_mask, target, reference_scale


def stack_lidar_batch(
    samples: Sequence[HypersimDisparitySample],
    settings: Mapping[str, object],
    *,
    seed: int,
    step: int,
    device: torch.device,
) -> LidarBatch:
    prepared = [
        make_lidar_prompt(
            sample,
            settings,
            seed=stable_prompt_seed(seed, step, sample.sample_id),
        )
        for sample in samples
    ]
    return LidarBatch(
        image=torch.stack([sample.image for sample in samples]).to(device),
        target_disparity=torch.stack([value[2] for value in prepared]).to(device),
        valid_mask=torch.stack([sample.valid_mask for sample in samples]).to(device),
        prompt_disparity=torch.stack([value[0] for value in prepared])[:, None].to(device),
        prompt_mask=torch.stack([value[1] for value in prepared])[:, None].to(device),
        reference_scale=torch.stack([value[3] for value in prepared]).to(device),
    )


def normalized_disparity_to_geometry(
    disparity: torch.Tensor,
    reference_scale: torch.Tensor,
    rays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_disparity = disparity.float() * reference_scale.float()
    valid = torch.isfinite(raw_disparity) & (raw_disparity > 1e-6)
    radial = torch.full_like(raw_disparity, float("nan"))
    radial[valid] = raw_disparity[valid].reciprocal()
    points = rays * radial[..., None]
    return radial, points, valid


def _robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0.0:
        return 1.4826 * mad
    tail = finite[finite > median]
    return 1.4826 * float(np.median(tail - median)) if tail.size else float("inf")


def coarse_fine_mask(
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    *,
    residual_scales: Sequence[int] = (8, 16, 32),
    morphology_sizes: Sequence[int] = (3, 5, 9, 17),
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """MoGe-3 Appendix B.1 coarse detector with pinned interpolation choices."""
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if disparity.shape != valid.shape or not valid.any():
        raise ValueError("Disparity and non-empty valid mask must have the same shape")
    safe = disparity.copy()
    safe[~valid] = float(np.median(safe[valid]))
    height, width = safe.shape
    selected = np.zeros_like(valid)
    for scale in residual_scales:
        low = cv2.resize(
            safe,
            (max(1, width // int(scale)), max(1, height // int(scale))),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR)
        residual = np.abs(safe - restored)
        selected |= residual > float(threshold_sigma) * _robust_sigma(residual[valid])
    for size in morphology_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
        opened = cv2.morphologyEx(safe, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(safe, cv2.MORPH_CLOSE, kernel)
        for residual in (np.maximum(safe - opened, 0.0), np.maximum(closed - safe, 0.0)):
            selected |= residual > float(threshold_sigma) * _robust_sigma(residual[valid])
    return selected & valid


def select_fine_segments(
    candidate_masks: Sequence[np.ndarray],
    coarse_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_density: float = 0.3,
    min_overlap_pixels: int = 5,
    max_area_fraction: float = 0.05,
) -> np.ndarray:
    coarse = np.asarray(coarse_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if coarse.shape != valid.shape:
        raise ValueError("Coarse and valid masks must have the same shape")
    maximum_area = float(max_area_fraction) * coarse.size
    selected = []
    for candidate in candidate_masks:
        segment = np.asarray(candidate, dtype=bool)
        if segment.shape != coarse.shape:
            raise ValueError("SAM2 segment shape differs from the coarse mask")
        area = int(segment.sum())
        overlap = int((segment & coarse).sum())
        density = overlap / area if area else 0.0
        if density >= float(min_density) and overlap >= int(min_overlap_pixels) and area <= maximum_area:
            selected.append(segment & valid)
    return (
        np.stack(selected).astype(bool, copy=False)
        if selected
        else np.zeros((0, *coarse.shape), dtype=bool)
    )


def save_segment_masks(path: Path, masks: np.ndarray) -> None:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("Segment masks must have shape [N,H,W]")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(
        masks.reshape(masks.shape[0], masks.shape[1] * masks.shape[2]), axis=1
    )
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        format=np.asarray("infinidepth-moge3-local-segments-v1"),
        packed=packed,
        shape=np.asarray(masks.shape, dtype=np.int64),
    )
    temporary.replace(path)


def load_segment_masks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as value:
        if str(value["format"].item()) != "infinidepth-moge3-local-segments-v1":
            raise ValueError(f"Unsupported local mask format: {path}")
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) != 3 or any(item < 0 for item in shape):
            raise ValueError(f"Invalid local segment shape: {shape}")
        unpacked = np.unpackbits(value["packed"], axis=1, count=shape[1] * shape[2])
    return unpacked.reshape(shape).astype(bool, copy=False)


@torch.no_grad()
def local_point_metrics(
    prediction_points: torch.Tensor,
    target_points: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_masks: torch.Tensor,
    *,
    min_segment_pixels: int = 10,
    delta_threshold: float = 0.01,
) -> Dict[str, float | None]:
    """MoGe-3 v2: one global scale, one translation per segment, segment macro mean."""
    if segment_masks.ndim != 3 or tuple(segment_masks.shape[1:]) != tuple(valid_mask.shape):
        raise ValueError("Segment masks must have shape [N,H,W] matching the point maps")
    valid = (
        valid_mask.bool()
        & torch.isfinite(prediction_points).all(dim=-1)
        & torch.isfinite(target_points).all(dim=-1)
    )
    target_norm = torch.linalg.vector_norm(target_points, dim=-1)
    valid &= target_norm > 1e-8
    if not bool(valid.any()):
        return {
            "local_point_rel": None,
            "local_point_delta_0_01": None,
            "local_segment_count": 0.0,
            "_local_point_rel_sum": 0.0,
            "_local_point_delta_0_01_sum": 0.0,
            "local_global_scale": None,
        }
    weights = target_norm[valid].reciprocal()
    prediction_valid = prediction_points[valid]
    target_valid = target_points[valid]
    denominator = (weights * prediction_valid.square().sum(dim=-1)).sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-12:
        return {
            "local_point_rel": None,
            "local_point_delta_0_01": None,
            "local_segment_count": 0.0,
            "_local_point_rel_sum": 0.0,
            "_local_point_delta_0_01_sum": 0.0,
            "local_global_scale": None,
        }
    scale = (weights * (prediction_valid * target_valid).sum(dim=-1)).sum() / denominator
    rel_values = []
    delta_values = []
    for segment in segment_masks.bool():
        selected = segment & valid
        if int(selected.sum()) < int(min_segment_pixels):
            continue
        segment_weights = target_norm[selected].reciprocal()
        scaled = prediction_points[selected] * scale
        target = target_points[selected]
        translation = (
            segment_weights[:, None] * (target - scaled)
        ).sum(dim=0) / segment_weights.sum()
        aligned = scaled + translation
        error = torch.linalg.vector_norm(aligned - target, dim=-1)
        target_distance = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-8)
        aligned_distance = torch.linalg.vector_norm(aligned, dim=-1)
        rel_values.append((error / target_distance).mean())
        delta_values.append(
            (
                error
                < float(delta_threshold) * torch.minimum(target_distance, aligned_distance)
            )
            .float()
            .mean()
        )
    if not rel_values:
        return {
            "local_point_rel": None,
            "local_point_delta_0_01": None,
            "local_segment_count": 0.0,
            "_local_point_rel_sum": 0.0,
            "_local_point_delta_0_01_sum": 0.0,
            "local_global_scale": float(scale.item()),
        }
    rel = torch.stack(rel_values)
    delta = torch.stack(delta_values)
    return {
        "local_point_rel": float(rel.mean().item()),
        "local_point_delta_0_01": float(delta.mean().item()),
        "local_segment_count": float(rel.numel()),
        "_local_point_rel_sum": float(rel.sum().item()),
        "_local_point_delta_0_01_sum": float(delta.sum().item()),
        "local_global_scale": float(scale.item()),
    }

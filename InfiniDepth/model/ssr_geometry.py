"""Geometry adapter between frozen InfiniDepth inference and SSR refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .model import (
    DisparityAlignment,
    InfiniDepthEncoding,
    align_reference_disparity,
)


@dataclass(frozen=True)
class InfiniDepthSSRInputs:
    image: torch.Tensor
    query_coord: torch.Tensor
    depth0: torch.Tensor
    valid_mask: torch.Tensor
    intrinsics: torch.Tensor
    points0: torch.Tensor
    dino_feature: torch.Tensor
    basic_feature: torch.Tensor
    raw_disparity: torch.Tensor
    alignment: Sequence[DisparityAlignment]
    source_tags: Mapping[str, str] = field(default_factory=dict)


def make_dense_query_coord(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return pixel-center normalized coordinates in ``(y, x)`` order."""
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("Dense query dimensions must be positive")
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)


def scale_pixel_center_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Scale pixel intrinsics without moving pixel centers."""
    source_height, source_width = source_size
    target_height, target_width = target_size
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("Image dimensions must be positive")
    scaled = intrinsics.clone().float()
    sx = target_width / float(source_width)
    sy = target_height / float(source_height)
    scaled[..., 0, 0] *= sx
    scaled[..., 1, 1] *= sy
    scaled[..., 0, 2] = (scaled[..., 0, 2] + 0.5) * sx - 0.5
    scaled[..., 1, 2] = (scaled[..., 1, 2] + 0.5) * sy - 0.5
    return scaled


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Backproject ``[B,H,W]`` metric z-depth to camera-space points."""
    if depth.ndim != 3:
        raise ValueError(f"Expected depth [B,H,W], got {tuple(depth.shape)}")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError(
            f"Expected intrinsics [B,3,3], got {tuple(intrinsics.shape)} for batch {depth.shape[0]}"
        )
    batch, height, width = depth.shape
    rows, cols = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack((cols, rows, torch.ones_like(cols)), dim=-1)
    pixels = pixels.reshape(1, height * width, 3).expand(batch, -1, -1)
    rays = pixels @ torch.linalg.inv(intrinsics.float()).transpose(-1, -2)
    return (rays * depth.float().reshape(batch, -1, 1)).reshape(batch, height, width, 3)


def fill_invalid_depth_with_median(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Regularize invalid shell positions while keeping a separate validity mask."""
    if depth.shape != valid_mask.shape:
        raise ValueError("Depth and validity mask shapes must match")
    filled = depth.float().clone()
    for batch_index in range(depth.shape[0]):
        valid = valid_mask[batch_index] & torch.isfinite(depth[batch_index]) & (depth[batch_index] > 0)
        if not bool(valid.any()):
            raise ValueError(f"Sample {batch_index} contains no valid reference depth")
        median = depth[batch_index][valid].float().median()
        filled[batch_index] = torch.where(valid, depth[batch_index].float(), median)
    return filled


def _as_bhw(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"Expected {name} [B,H,W] or [B,1,H,W], got {tuple(tensor.shape)}")


def build_ssr_inputs(
    model: torch.nn.Module,
    image: torch.Tensor,
    reference_depth: torch.Tensor,
    reference_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    chunk_size: int = 10000,
    source_tags: Mapping[str, str] | None = None,
) -> InfiniDepthSSRInputs:
    """Build detached SSR state; ground truth is intentionally not accepted."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
    batch, _, height, width = image.shape
    if (height, width) != (384, 512):
        raise ValueError(f"The fixed-grid MVP requires 384x512 input, got {height}x{width}")
    reference_depth_bhw = _as_bhw(reference_depth, "reference_depth").float()
    reference_mask_bhw = _as_bhw(reference_mask, "reference_mask") > 0
    if reference_depth_bhw.shape != (batch, height, width):
        reference_depth_bhw = F.interpolate(
            reference_depth_bhw[:, None], size=(height, width), mode="nearest"
        )[:, 0]
        reference_mask_bhw = F.interpolate(
            reference_mask_bhw[:, None].float(), size=(height, width), mode="nearest"
        )[:, 0] > 0
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.shape[0] == 1 and batch > 1:
        intrinsics = intrinsics.expand(batch, -1, -1)

    with torch.inference_mode():
        encoding: InfiniDepthEncoding = model.encode_image(image)
        query = make_dense_query_coord(batch, height, width, device=image.device)
        raw = model.decode_queries(
            encoding, query.reshape(batch, -1, 2), chunk_size=chunk_size
        ).reshape(batch, height, width).float()

    # Tensors created in inference mode cannot be saved for backward by SSR.
    # Cloning after the context produces ordinary detached tensors while still
    # guaranteeing that the frozen base graph is never retained.
    query = query.clone()
    raw = raw.clone()
    dino_feature = encoding.dino_feature.clone()
    basic_feature = encoding.basic_feature.clone()
    reference_valid = (
        reference_mask_bhw
        & torch.isfinite(reference_depth_bhw)
        & (reference_depth_bhw > 0)
    )
    reference_disparity = torch.where(
        reference_valid,
        reference_depth_bhw.clamp_min(1e-8).reciprocal(),
        torch.zeros_like(reference_depth_bhw),
    )
    aligned_maps = []
    alignment = []
    for batch_index in range(batch):
        aligned, metadata = align_reference_disparity(
            raw[batch_index],
            reference_disparity[batch_index],
            reference_valid[batch_index],
            random_state=0,
        )
        aligned_maps.append(aligned)
        alignment.append(metadata)
    aligned_disparity = torch.stack(aligned_maps)
    valid = reference_valid & torch.isfinite(aligned_disparity) & (aligned_disparity > 5e-3)
    depth0 = aligned_disparity.clamp_min(5e-3).reciprocal()
    regular_depth = fill_invalid_depth_with_median(depth0, valid)
    points0 = depth_to_points(regular_depth, intrinsics.to(image.device))
    return InfiniDepthSSRInputs(
        image=image.detach(),
        query_coord=query.detach(),
        depth0=regular_depth.detach(),
        valid_mask=valid.detach(),
        intrinsics=intrinsics.float().detach(),
        points0=points0.detach(),
        dino_feature=dino_feature,
        basic_feature=basic_feature,
        raw_disparity=raw.detach(),
        alignment=tuple(alignment),
        source_tags=dict(source_tags or {}),
    )

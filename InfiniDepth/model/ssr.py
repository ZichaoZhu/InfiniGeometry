"""Self-guided sparse refinement for InfiniDepth.

The factorized shell, sparse U-Net topology, and recursive log-depth update are
adapted from the MoGe-3 research implementation in this workspace. This module
is self-contained and has no runtime dependency on MoGe-3.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DisparityAlignment
from .ssr_geometry import InfiniDepthSSRInputs


def smooth_bound_log_depth_residual(
    residual: torch.Tensor,
    max_abs: Optional[float],
) -> torch.Tensor:
    if max_abs is None or float(max_abs) == 0.0:
        return residual
    limit = float(max_abs)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("Residual bound must be finite and non-negative")
    bounded = limit * torch.tanh(residual / limit)
    return torch.where(torch.isfinite(residual), bounded, residual)


def factorize_points(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points[...,3], got {tuple(points.shape)}")
    depth = points[..., 2:3].clamp_min(eps)
    return torch.cat((points[..., :2] / depth, depth.log()), dim=-1)


def unfactorize_points(factorized: torch.Tensor) -> torch.Tensor:
    if factorized.shape[-1] != 3:
        raise ValueError(f"Expected factorized[...,3], got {tuple(factorized.shape)}")
    depth = factorized[..., 2:3].exp()
    return torch.cat((factorized[..., :2] * depth, depth), dim=-1)


def _ceil_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


@dataclass(frozen=True)
class VoxelizedShell:
    coordinates: torch.Tensor
    features: torch.Tensor
    spatial_shape: Tuple[int, int, int]
    batch_size: int
    image_size: Tuple[int, int]
    logical_depth: torch.Tensor
    depth_offsets: torch.Tensor

    def statistics(self) -> Dict[str, object]:
        depth_span = (
            self.logical_depth.amax(dim=(-2, -1))
            - self.logical_depth.amin(dim=(-2, -1))
            + 1
        )
        return {
            "active_voxels": int(self.coordinates.shape[0]),
            "depth_span": depth_span,
            "depth_offsets": self.depth_offsets,
            "spatial_shape": self.spatial_shape,
        }


class VoxelDepthSpanError(RuntimeError):
    pass


def logical_voxel_depth_spans(
    factorized: torch.Tensor,
    *,
    voxel_resolution: float,
) -> torch.Tensor:
    logical = torch.round(float(voxel_resolution) * factorized[..., 2]).long()
    return logical.amax(dim=(-2, -1)) - logical.amin(dim=(-2, -1)) + 1


def effective_voxel_depth_limit(
    configured_limit: Optional[int],
    base_depth_spans: torch.Tensor,
    *,
    maximum_expansion: int = 0,
) -> Optional[int]:
    if configured_limit is None:
        return None
    if configured_limit <= 0 or maximum_expansion < 0:
        raise ValueError("Voxel depth limits must be positive")
    return max(
        int(configured_limit),
        int(base_depth_spans.detach().amax().item()) + int(maximum_expansion),
    )


def voxelize_factorized(
    factorized: torch.Tensor,
    *,
    voxel_resolution: float = 200.0,
    num_downsamples: int = 4,
    max_depth_span: Optional[int] = None,
) -> VoxelizedShell:
    """Voxelize logical ``(row,col,round(D*logZ))`` into spconv ``BZYX``."""
    if factorized.ndim != 4 or factorized.shape[-1] != 3:
        raise ValueError(f"Expected [B,H,W,3], got {tuple(factorized.shape)}")
    if not bool(torch.isfinite(factorized).all()):
        raise ValueError("SSR factorized coordinates must be finite")
    batch, height, width, _ = factorized.shape
    logical_depth = torch.round(voxel_resolution * factorized[..., 2]).long()
    offsets = logical_depth.amin(dim=(-2, -1))
    storage_depth = logical_depth - offsets[:, None, None]
    maximum_span = int((storage_depth.amax(dim=(-2, -1)) + 1).amax().item())
    if max_depth_span is not None and maximum_span > int(max_depth_span):
        raise VoxelDepthSpanError(
            f"Voxel depth span exceeded before allocation: {maximum_span} > {max_depth_span}"
        )

    rows, cols = torch.meshgrid(
        torch.arange(height, device=factorized.device, dtype=torch.long),
        torch.arange(width, device=factorized.device, dtype=torch.long),
        indexing="ij",
    )
    rows = rows[None].expand(batch, -1, -1)
    cols = cols[None].expand(batch, -1, -1)
    batches = torch.arange(batch, device=factorized.device)[:, None, None].expand(-1, height, width)
    coordinates = torch.stack((batches, storage_depth, rows, cols), dim=-1)
    stride = 2**num_downsamples
    spatial_shape = (
        _ceil_multiple(int(storage_depth.amax().item()) + 1, stride),
        _ceil_multiple(height, stride),
        _ceil_multiple(width, stride),
    )
    return VoxelizedShell(
        coordinates=coordinates.reshape(-1, 4).int().contiguous(),
        features=factorized.reshape(-1, 3).contiguous(),
        spatial_shape=spatial_shape,
        batch_size=batch,
        image_size=(height, width),
        logical_depth=logical_depth,
        depth_offsets=offsets,
    )


def _coordinate_hash(coordinates: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    depth_size, row_size, col_size = (int(value) for value in spatial_shape)
    batch, depth, row, col = coordinates.long().unbind(dim=-1)
    return ((batch * depth_size + depth) * row_size + row) * col_size + col


def gather_features_at_coordinates(
    features: torch.Tensor,
    source_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
    spatial_shape: Sequence[int],
) -> torch.Tensor:
    if torch.equal(source_coordinates, target_coordinates):
        return features
    source_hash = _coordinate_hash(source_coordinates, spatial_shape)
    target_hash = _coordinate_hash(target_coordinates, spatial_shape)
    sorted_hash, order = source_hash.sort()
    positions = torch.searchsorted(sorted_hash, target_hash)
    if positions.numel() and (
        bool((positions >= sorted_hash.numel()).any())
        or not torch.equal(sorted_hash[positions.clamp_max(sorted_hash.numel() - 1)], target_hash)
    ):
        raise RuntimeError("Sparse decoder did not recover every input voxel")
    return features[order[positions]]


def sample_visual_features_for_voxels(
    visual_features: torch.Tensor,
    coordinates: torch.Tensor,
    image_size: Tuple[int, int],
    level_scale: int,
) -> torch.Tensor:
    height, width = image_size
    visual = F.interpolate(
        visual_features.float(),
        size=(max(1, math.ceil(height / level_scale)), max(1, math.ceil(width / level_scale))),
        mode="bilinear",
        align_corners=False,
    )
    batch, _, row, col = coordinates.long().unbind(dim=-1)
    return visual[
        batch,
        :,
        row.clamp(0, visual.shape[-2] - 1),
        col.clamp(0, visual.shape[-1] - 1),
    ]


def _sparse_normalization(channels: int, normalization: str) -> nn.Module:
    if normalization == "batch_norm":
        return nn.BatchNorm1d(channels)
    if normalization == "layer_norm":
        return nn.LayerNorm(channels)
    if normalization == "group_norm":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported SSR normalization: {normalization}")


class _SparseResidualBlock(nn.Module):
    def __init__(self, spconv, channels: int, key: str, algorithm, normalization: str):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False, indice_key=f"{key}_1", algo=algorithm
        )
        self.conv2 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False, indice_key=f"{key}_2", algo=algorithm
        )
        self.norm1 = _sparse_normalization(channels, normalization)
        self.norm2 = _sparse_normalization(channels, normalization)

    def forward(self, sparse):
        identity = sparse.features
        sparse = sparse.replace_feature(F.relu(self.norm1(sparse.features), inplace=True))
        sparse = self.conv1(sparse)
        sparse = sparse.replace_feature(F.relu(self.norm2(sparse.features), inplace=True))
        sparse = self.conv2(sparse)
        return sparse.replace_feature(sparse.features + identity)


class SpconvSparseUNet(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
        normalization: str = "batch_norm",
    ):
        super().__init__()
        try:
            import spconv.pytorch as spconv
            from spconv.core import ConvAlgo
        except ImportError as exc:
            raise ImportError("SSR requires spconv 2.x in the CUDA environment") from exc
        self.spconv = spconv
        self.algorithm = ConvAlgo.MaskImplicitGemm
        self.channels = tuple(int(value) for value in channels)
        self.num_downsamples = len(self.channels) - 1
        if self.num_downsamples < 1:
            raise ValueError("SSR sparse U-Net needs at least two channel levels")
        self.input_projection = spconv.SubMConv3d(
            3, self.channels[0], 1, bias=False, indice_key="ssr_input", algo=self.algorithm
        )
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
                _SparseResidualBlock(spconv, width, f"enc_{level}_{block}", self.algorithm, normalization)
                for block in range(blocks_per_level)
            ])
            for level, width in enumerate(self.channels)
        ])
        self.downsample = nn.ModuleList([
            spconv.SparseConv3d(
                self.channels[level], self.channels[level + 1], 2, stride=2, bias=False,
                indice_key=f"ssr_down_{level}", algo=self.algorithm,
            )
            for level in range(self.num_downsamples)
        ])
        self.visual_projection = nn.Linear(visual_dim, visual_channels)
        self.bottleneck_fusion = spconv.SubMConv3d(
            self.channels[-1] + visual_channels,
            self.channels[-1],
            1,
            bias=False,
            indice_key="ssr_visual_fusion",
            algo=self.algorithm,
        )
        self.upsample = nn.ModuleList([
            spconv.SparseInverseConv3d(
                self.channels[level + 1], self.channels[level], 2, bias=False,
                indice_key=f"ssr_down_{level}", algo=self.algorithm,
            )
            for level in range(self.num_downsamples)
        ])
        self.decoder_fusion = nn.ModuleList([
            spconv.SubMConv3d(
                2 * self.channels[level], self.channels[level], 1, bias=False,
                indice_key=f"ssr_dec_fuse_{level}", algo=self.algorithm,
            )
            for level in range(self.num_downsamples)
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
                _SparseResidualBlock(
                    spconv, self.channels[level], f"dec_{level}_{block}", self.algorithm, normalization
                )
                for block in range(blocks_per_level)
            ])
            for level in range(self.num_downsamples)
        ])
        self.output_layer = spconv.SubMConv3d(
            self.channels[0], 1, 1, bias=True, indice_key="ssr_output", algo=self.algorithm
        )
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, shell: VoxelizedShell, visual_features: torch.Tensor):
        sparse = self.spconv.SparseConvTensor(
            shell.features, shell.coordinates, list(shell.spatial_shape), shell.batch_size
        )
        sparse = self.input_projection(sparse)
        skips = []
        active_counts = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                sparse = block(sparse)
            skips.append(sparse)
            active_counts.append(int(sparse.indices.shape[0]))
            if level < self.num_downsamples:
                sparse = self.downsample[level](sparse)
        visual = sample_visual_features_for_voxels(
            visual_features, sparse.indices, shell.image_size, 2**self.num_downsamples
        )
        sparse = sparse.replace_feature(
            torch.cat((sparse.features, self.visual_projection(visual)), dim=-1)
        )
        sparse = self.bottleneck_fusion(sparse)
        for level in reversed(range(self.num_downsamples)):
            sparse = self.upsample[level](sparse)
            skip = skips[level]
            skip_features = gather_features_at_coordinates(
                skip.features, skip.indices, sparse.indices, sparse.spatial_shape
            )
            sparse = sparse.replace_feature(torch.cat((sparse.features, skip_features), dim=-1))
            sparse = self.decoder_fusion[level](sparse)
            for block in self.decoder_blocks[level]:
                sparse = block(sparse)
        sparse = self.output_layer(sparse)
        residual = gather_features_at_coordinates(
            sparse.features, sparse.indices, shell.coordinates, shell.spatial_shape
        )
        return residual[:, 0].reshape(shell.batch_size, *shell.image_size), active_counts


class _DenseResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        update = self.conv1(F.relu(self.norm1(values), inplace=True)) * active
        update = self.conv2(F.relu(self.norm2(update), inplace=True)) * active
        return (values + update) * active


class ReferenceSparseUNet(nn.Module):
    """Dense masked reference backend for tiny CPU tests only."""

    def __init__(
        self,
        visual_dim: int,
        channels: Sequence[int] = (8, 16, 32),
        visual_channels: int = 16,
        blocks_per_level: int = 1,
        max_dense_voxels: int = 2_000_000,
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.num_downsamples = len(self.channels) - 1
        self.max_dense_voxels = int(max_dense_voxels)
        self.input_projection = nn.Conv3d(3, self.channels[0], 1, bias=False)
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([_DenseResidualBlock(width) for _ in range(blocks_per_level)])
            for width in self.channels
        ])
        self.downsample = nn.ModuleList([
            nn.Conv3d(self.channels[level], self.channels[level + 1], 2, stride=2, bias=False)
            for level in range(self.num_downsamples)
        ])
        self.visual_projection = nn.Conv2d(visual_dim, visual_channels, 1)
        self.bottleneck_fusion = nn.Conv3d(
            self.channels[-1] + visual_channels, self.channels[-1], 1, bias=False
        )
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(self.channels[level + 1], self.channels[level], 2, stride=2, bias=False)
            for level in range(self.num_downsamples)
        ])
        self.decoder_fusion = nn.ModuleList([
            nn.Conv3d(2 * self.channels[level], self.channels[level], 1, bias=False)
            for level in range(self.num_downsamples)
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([_DenseResidualBlock(self.channels[level]) for _ in range(blocks_per_level)])
            for level in range(self.num_downsamples)
        ])
        self.output_layer = nn.Conv3d(self.channels[0], 1, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, shell: VoxelizedShell, visual_features: torch.Tensor):
        depth_size, row_size, col_size = shell.spatial_shape
        count = shell.batch_size * depth_size * row_size * col_size
        if count > self.max_dense_voxels:
            raise RuntimeError(f"Reference backend voxel limit exceeded: {count:,}")
        values = shell.features.new_zeros((shell.batch_size, 3, depth_size, row_size, col_size))
        active = shell.features.new_zeros((shell.batch_size, 1, depth_size, row_size, col_size))
        batch, depth, row, col = shell.coordinates.long().unbind(dim=-1)
        values[batch, :, depth, row, col] = shell.features
        active[batch, :, depth, row, col] = 1
        values = self.input_projection(values) * active
        skips, active_skips, active_counts = [], [], []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                values = block(values, active)
            skips.append(values)
            active_skips.append(active)
            active_counts.append(int(active.sum().item()))
            if level < self.num_downsamples:
                values = self.downsample[level](values)
                active = F.max_pool3d(active, 2, stride=2)
                values *= active
        visual = self.visual_projection(F.interpolate(
            visual_features.float(), size=values.shape[-2:], mode="bilinear", align_corners=False
        ))[:, :, None].expand(-1, -1, values.shape[-3], -1, -1)
        values = self.bottleneck_fusion(torch.cat((values, visual), dim=1)) * active
        for level in reversed(range(self.num_downsamples)):
            values = self.upsample[level](values)
            target = skips[level].shape[-3:]
            values = values[..., : target[0], : target[1], : target[2]]
            active = active_skips[level]
            values = self.decoder_fusion[level](torch.cat((values, skips[level]), dim=1)) * active
            for block in self.decoder_blocks[level]:
                values = block(values, active)
        residual_grid = self.output_layer(values)
        return residual_grid[batch, 0, depth, row, col].reshape(
            shell.batch_size, *shell.image_size
        ), active_counts


class SelfGuidedSparseRefiner(nn.Module):
    def __init__(
        self,
        visual_dim: int = 1152,
        voxel_resolution: float = 200.0,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
        backend: str = "spconv",
        normalization: str = "batch_norm",
    ):
        super().__init__()
        self.voxel_resolution = float(voxel_resolution)
        if backend == "spconv":
            self.unet = SpconvSparseUNet(
                visual_dim, channels, visual_channels, blocks_per_level, normalization
            )
        elif backend == "reference":
            self.unet = ReferenceSparseUNet(
                visual_dim, channels, visual_channels, blocks_per_level
            )
        else:
            raise ValueError(f"Unsupported SSR backend: {backend}")

    @property
    def num_downsamples(self) -> int:
        return self.unet.num_downsamples

    def forward(
        self,
        factorized: torch.Tensor,
        visual_features: torch.Tensor,
        *,
        max_depth_span: Optional[int] = None,
    ):
        shell = voxelize_factorized(
            factorized.float(),
            voxel_resolution=self.voxel_resolution,
            num_downsamples=self.num_downsamples,
            max_depth_span=max_depth_span,
        )
        residual, active_counts = self.unet(shell, visual_features.float())
        stats = shell.statistics()
        stats["active_voxels_per_level"] = active_counts
        return residual, stats


@dataclass(frozen=True)
class InfiniDepthSSROutput:
    depth: torch.Tensor
    points: torch.Tensor
    depth_sequence: Tuple[torch.Tensor, ...]
    points_sequence: Tuple[torch.Tensor, ...]
    raw_residuals: Tuple[torch.Tensor, ...]
    applied_residuals: Tuple[torch.Tensor, ...]
    voxel_statistics: Tuple[Dict[str, object], ...]
    valid_mask: torch.Tensor
    alignment: Sequence[DisparityAlignment]


class InfiniDepthSSR(nn.Module):
    def __init__(
        self,
        *,
        backend: str = "spconv",
        voxel_resolution: float = 200.0,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
        normalization: str = "batch_norm",
        configured_depth_span: int = 512,
    ):
        super().__init__()
        self.configured_depth_span = int(configured_depth_span)
        self.refiner = SelfGuidedSparseRefiner(
            visual_dim=1024 + 128,
            voxel_resolution=voxel_resolution,
            channels=channels,
            visual_channels=visual_channels,
            blocks_per_level=blocks_per_level,
            backend=backend,
            normalization=normalization,
        )

    def forward(
        self,
        inputs: InfiniDepthSSRInputs,
        K: int,
        residual_bound: float = 0.1,
    ) -> InfiniDepthSSROutput:
        if K < 0:
            raise ValueError("K must be non-negative")
        points = inputs.points0.float()
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("SSR points0 must have shape [B,H,W,3]")
        visual = torch.cat((
            inputs.dino_feature.float(),
            F.interpolate(
                inputs.basic_feature.float(),
                size=inputs.dino_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        ), dim=1)
        base_spans = logical_voxel_depth_spans(
            factorize_points(points), voxel_resolution=self.refiner.voxel_resolution
        )
        expansion = int(math.ceil(2 * self.refiner.voxel_resolution * residual_bound * K)) + 2
        depth_limit = effective_voxel_depth_limit(
            self.configured_depth_span, base_spans, maximum_expansion=expansion
        )
        points_sequence: List[torch.Tensor] = [points]
        depth_sequence: List[torch.Tensor] = [points[..., 2]]
        raw_residuals: List[torch.Tensor] = []
        applied_residuals: List[torch.Tensor] = []
        statistics: List[Dict[str, object]] = []
        for _ in range(K):
            raw, stats = self.refiner(
                factorize_points(points), visual, max_depth_span=depth_limit
            )
            applied = smooth_bound_log_depth_residual(raw, residual_bound)
            points = points * applied.exp().unsqueeze(-1)
            raw_residuals.append(raw)
            applied_residuals.append(applied)
            statistics.append(stats)
            points_sequence.append(points)
            depth_sequence.append(points[..., 2])
        return InfiniDepthSSROutput(
            depth=depth_sequence[-1],
            points=points_sequence[-1],
            depth_sequence=tuple(depth_sequence),
            points_sequence=tuple(points_sequence),
            raw_residuals=tuple(raw_residuals),
            applied_residuals=tuple(applied_residuals),
            voxel_statistics=tuple(statistics),
            valid_mask=inputs.valid_mask,
            alignment=inputs.alignment,
        )

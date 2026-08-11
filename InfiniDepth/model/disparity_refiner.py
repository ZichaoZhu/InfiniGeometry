"""Sparse disparity refiner adapted from the MoGe-3 SSR dataflow.

The architectural reference is commit 5796a09d24de5515bf9ea7ea14b1727f4bb37526
from https://github.com/ZichaoZhu/MoGe.git. This implementation intentionally
replaces MoGe-3's log-depth shell and multiplicative update with InfiniDepth's
native normalized disparity shell and a bounded additive residual.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bound_disparity_residual(
    residual: torch.Tensor,
    max_abs: float = 0.1,
) -> torch.Tensor:
    """Smoothly bound one additive disparity update."""
    limit = float(max_abs)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("Disparity residual bound must be finite and positive")
    bounded = limit * torch.tanh(residual / limit)
    return torch.where(torch.isfinite(residual), bounded, residual)


def make_disparity_features(disparity: torch.Tensor) -> torch.Tensor:
    """Build per-pixel (x_norm, y_norm, disparity) features."""
    if disparity.ndim != 3:
        raise ValueError(f"Expected disparity [B,H,W], got {tuple(disparity.shape)}")
    if not torch.isfinite(disparity).all():
        raise ValueError("Disparity features must be finite")
    batch, height, width = disparity.shape
    ys = ((torch.arange(height, device=disparity.device, dtype=torch.float32) + 0.5) / height) * 2 - 1
    xs = ((torch.arange(width, device=disparity.device, dtype=torch.float32) + 0.5) / width) * 2 - 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)
    return torch.stack((grid_x, grid_y, disparity.float()), dim=-1)


def _ceil_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


@dataclass
class VoxelizedDisparityShell:
    coordinates: torch.Tensor
    features: torch.Tensor
    spatial_shape: Tuple[int, int, int]
    batch_size: int
    image_size: Tuple[int, int]
    logical_disparity_bins: torch.Tensor
    disparity_offsets: torch.Tensor

    def statistics(self) -> Dict[str, object]:
        spans = (
            self.logical_disparity_bins.amax(dim=(-2, -1))
            - self.logical_disparity_bins.amin(dim=(-2, -1))
            + 1
        )
        return {
            "active_voxels": int(self.coordinates.shape[0]),
            "disparity_span": spans.detach(),
            "disparity_offsets": self.disparity_offsets.detach(),
            "spatial_shape": self.spatial_shape,
        }


class VoxelDisparitySpanError(RuntimeError):
    pass


def logical_disparity_spans(
    disparity: torch.Tensor,
    *,
    voxel_resolution: float = 200.0,
) -> torch.Tensor:
    if disparity.ndim != 3:
        raise ValueError(f"Expected disparity [B,H,W], got {tuple(disparity.shape)}")
    bins = torch.round(float(voxel_resolution) * disparity).to(torch.long)
    return bins.amax(dim=(-2, -1)) - bins.amin(dim=(-2, -1)) + 1


def voxelize_disparity(
    disparity: torch.Tensor,
    *,
    voxel_resolution: float = 200.0,
    num_downsamples: int = 4,
    max_disparity_span: Optional[int] = None,
) -> VoxelizedDisparityShell:
    """Voxelize a disparity map using spconv order [batch, bin, row, col]."""
    features = make_disparity_features(disparity)
    batch_size, height, width = disparity.shape
    logical_bins = torch.round(float(voxel_resolution) * disparity).to(torch.long)
    offsets = logical_bins.amin(dim=(-2, -1))
    storage_bins = logical_bins - offsets[:, None, None]
    spans = logical_bins.amax(dim=(-2, -1)) - offsets + 1
    maximum_span = int(spans.amax().item())
    if max_disparity_span is not None:
        if int(max_disparity_span) <= 0:
            raise ValueError("Maximum disparity span must be positive")
        if maximum_span > int(max_disparity_span):
            raise VoxelDisparitySpanError(
                "Disparity voxel span exceeded the configured limit before sparse "
                f"allocation: {maximum_span} > {int(max_disparity_span)}"
            )

    rows, cols = torch.meshgrid(
        torch.arange(height, device=disparity.device, dtype=torch.long),
        torch.arange(width, device=disparity.device, dtype=torch.long),
        indexing="ij",
    )
    rows = rows.unsqueeze(0).expand(batch_size, -1, -1)
    cols = cols.unsqueeze(0).expand(batch_size, -1, -1)
    batches = torch.arange(batch_size, device=disparity.device, dtype=torch.long)
    batches = batches[:, None, None].expand(-1, height, width)
    coordinates = torch.stack((batches, storage_bins, rows, cols), dim=-1)

    stride = 2**int(num_downsamples)
    spatial_shape = (
        _ceil_multiple(int(storage_bins.amax().item()) + 1, stride),
        _ceil_multiple(height, stride),
        _ceil_multiple(width, stride),
    )
    return VoxelizedDisparityShell(
        coordinates=coordinates.reshape(-1, 4).to(torch.int32).contiguous(),
        features=features.reshape(-1, 3).float().contiguous(),
        spatial_shape=spatial_shape,
        batch_size=batch_size,
        image_size=(height, width),
        logical_disparity_bins=logical_bins,
        disparity_offsets=offsets,
    )


def _coordinate_hash(coordinates: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    z_size, y_size, x_size = (int(value) for value in spatial_shape)
    batch, depth, row, col = coordinates.to(torch.long).unbind(dim=-1)
    return ((batch * z_size + depth) * y_size + row) * x_size + col


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
    clamped = positions.clamp_max(max(0, sorted_hash.numel() - 1))
    if positions.numel() and (
        (positions >= sorted_hash.numel()).any()
        or not torch.equal(sorted_hash[clamped], target_hash)
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
    target_height = max(1, (height + level_scale - 1) // level_scale)
    target_width = max(1, (width + level_scale - 1) // level_scale)
    visual = F.interpolate(
        visual_features.float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    batch, _, row, col = coordinates.to(torch.long).unbind(dim=-1)
    return visual[
        batch,
        :,
        row.clamp_max(target_height - 1),
        col.clamp_max(target_width - 1),
    ]


class _SparseResidualBlock(nn.Module):
    def __init__(self, spconv, channels: int, indice_key: str, conv_algo):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False,
            indice_key=f"{indice_key}_1", algo=conv_algo,
        )
        self.conv2 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False,
            indice_key=f"{indice_key}_2", algo=conv_algo,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, sparse):
        identity = sparse.features
        sparse = sparse.replace_feature(F.relu(self.norm1(sparse.features), inplace=True))
        sparse = self.conv1(sparse)
        sparse = sparse.replace_feature(F.relu(self.norm2(sparse.features), inplace=True))
        sparse = self.conv2(sparse)
        return sparse.replace_feature(sparse.features + identity)


class SpconvDisparityUNet(nn.Module):
    """MoGe-3-style sparse U-Net adapted to a disparity shell."""

    def __init__(
        self,
        visual_dim: int,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
    ):
        super().__init__()
        try:
            import spconv.pytorch as spconv
            from spconv.core import ConvAlgo
        except ImportError as exc:
            raise ImportError(
                "The disparity refiner requires spconv 2.x matching the server CUDA runtime"
            ) from exc
        if tuple(int(value) for value in channels) != (32, 64, 128, 256, 512):
            raise ValueError("The production refiner uses channels [32,64,128,256,512]")
        if int(blocks_per_level) != 2:
            raise ValueError("The production refiner uses two residual blocks per level")
        self.spconv = spconv
        self.conv_algo = ConvAlgo.MaskImplicitGemm
        self.channels = tuple(int(value) for value in channels)
        self.num_downsamples = len(self.channels) - 1
        self.input_projection = spconv.SubMConv3d(
            3, self.channels[0], 1, bias=False, indice_key="disp_input", algo=self.conv_algo
        )
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
                _SparseResidualBlock(spconv, width, f"disp_enc_{level}_{block}", self.conv_algo)
                for block in range(blocks_per_level)
            ])
            for level, width in enumerate(self.channels)
        ])
        self.downsample = nn.ModuleList([
            spconv.SparseConv3d(
                self.channels[level], self.channels[level + 1], 2, stride=2,
                bias=False, indice_key=f"disp_down_{level}", algo=self.conv_algo,
            )
            for level in range(self.num_downsamples)
        ])
        self.visual_projection = nn.Linear(int(visual_dim), int(visual_channels))
        self.bottleneck_fusion = spconv.SubMConv3d(
            self.channels[-1] + int(visual_channels), self.channels[-1], 1,
            bias=False, indice_key="disp_visual_fusion", algo=self.conv_algo,
        )
        self.upsample = nn.ModuleList([
            spconv.SparseInverseConv3d(
                self.channels[level + 1], self.channels[level], 2, bias=False,
                indice_key=f"disp_down_{level}", algo=self.conv_algo,
            )
            for level in range(self.num_downsamples)
        ])
        self.decoder_fusion = nn.ModuleList([
            spconv.SubMConv3d(
                2 * self.channels[level], self.channels[level], 1, bias=False,
                indice_key=f"disp_dec_fuse_{level}", algo=self.conv_algo,
            )
            for level in range(self.num_downsamples)
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
                _SparseResidualBlock(spconv, self.channels[level], f"disp_dec_{level}_{block}", self.conv_algo)
                for block in range(blocks_per_level)
            ])
            for level in range(self.num_downsamples)
        ])
        self.output_layer = spconv.SubMConv3d(
            self.channels[0], 1, 1, bias=True,
            indice_key="disp_output", algo=self.conv_algo,
        )
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        shell: VoxelizedDisparityShell,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        sparse = self.spconv.SparseConvTensor(
            features=shell.features,
            indices=shell.coordinates,
            spatial_shape=list(shell.spatial_shape),
            batch_size=shell.batch_size,
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

        gathered = sample_visual_features_for_voxels(
            visual_features, sparse.indices, shell.image_size, 2**self.num_downsamples
        )
        gathered = self.visual_projection(gathered)
        sparse = sparse.replace_feature(torch.cat((sparse.features, gathered), dim=-1))
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
        height, width = shell.image_size
        return residual[:, 0].reshape(shell.batch_size, height, width), active_counts


class DenseReferenceDisparityRefiner(nn.Module):
    """Tiny dense backend for coordinate and zero-identity tests only."""

    def __init__(self, visual_dim: int, hidden_dim: int = 8, max_dense_voxels: int = 2_000_000):
        super().__init__()
        self.num_downsamples = 0
        self.max_dense_voxels = int(max_dense_voxels)
        self.input_projection = nn.Conv3d(3, hidden_dim, 1, bias=False)
        self.visual_projection = nn.Conv2d(visual_dim, hidden_dim, 1, bias=False)
        self.output_layer = nn.Conv3d(2 * hidden_dim, 1, 1, bias=True)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        shell: VoxelizedDisparityShell,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        z_size, y_size, x_size = shell.spatial_shape
        count = shell.batch_size * z_size * y_size * x_size
        if count > self.max_dense_voxels:
            raise RuntimeError(f"Dense reference allocation {count} exceeds {self.max_dense_voxels}")
        values = shell.features.new_zeros((shell.batch_size, 3, z_size, y_size, x_size))
        active = shell.features.new_zeros((shell.batch_size, 1, z_size, y_size, x_size))
        batch, depth, row, col = shell.coordinates.to(torch.long).unbind(dim=-1)
        values[batch, :, depth, row, col] = shell.features
        active[batch, :, depth, row, col] = 1
        values = self.input_projection(values) * active
        visual = self.visual_projection(F.interpolate(
            visual_features.float(), size=(y_size, x_size), mode="bilinear", align_corners=False
        ))
        visual = visual[:, :, None].expand(-1, -1, z_size, -1, -1)
        residual_grid = self.output_layer(torch.cat((values, visual), dim=1)) * active
        residual = residual_grid[batch, 0, depth, row, col]
        height, width = shell.image_size
        return residual.reshape(shell.batch_size, height, width), [int(active.sum().item())]


class DisparitySparseRefiner(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        voxel_resolution: float = 200.0,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
        backend: str = "spconv",
        max_disparity_span: Optional[int] = None,
    ):
        super().__init__()
        self.voxel_resolution = float(voxel_resolution)
        self.backend = str(backend)
        self.max_disparity_span = max_disparity_span
        if self.backend == "spconv":
            self.unet = SpconvDisparityUNet(
                visual_dim=visual_dim,
                channels=channels,
                visual_channels=visual_channels,
                blocks_per_level=blocks_per_level,
            )
        elif self.backend == "reference":
            self.unet = DenseReferenceDisparityRefiner(visual_dim=visual_dim)
        else:
            raise ValueError(f"Unsupported disparity refiner backend: {backend!r}")

    def forward(
        self,
        disparity: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        disparity = disparity.float()
        visual_features = visual_features.float()
        shell = voxelize_disparity(
            disparity,
            voxel_resolution=self.voxel_resolution,
            num_downsamples=self.unet.num_downsamples,
            max_disparity_span=self.max_disparity_span,
        )
        residual, active_counts = self.unet(shell, visual_features)
        stats = shell.statistics()
        stats["active_voxels_per_level"] = active_counts
        stats["raw_residual_min"] = residual.detach().amin()
        stats["raw_residual_max"] = residual.detach().amax()
        stats["raw_residual_mean"] = residual.detach().mean()
        stats["raw_residual_finite"] = bool(torch.isfinite(residual).all().item())
        return residual, stats

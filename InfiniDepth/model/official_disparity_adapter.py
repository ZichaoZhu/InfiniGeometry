"""Geometry-only adapter around the pinned, otherwise unchanged official U-Net."""
from __future__ import annotations

import torch
from torch import nn


def official_sparse_inputs(shell, visual_features):
    height, width = shell.image_size
    if height % 16 or width % 16:
        raise ValueError("Official SSR requires a query grid divisible by 16")
    if visual_features.shape[0] != shell.batch_size or visual_features.shape[-2:] != (height // 16, width // 16):
        raise ValueError("Official SSR visual features must match the query grid / 16")
    depth, padded_height, padded_width = shell.spatial_shape
    coords = shell.coordinates[:, (0, 2, 3, 1)].contiguous()
    shape = torch.Size([shell.batch_size, padded_height, padded_width, depth, 3])
    return shell.features, coords, shape


class OfficialDisparityUNet(nn.Module):
    num_downsamples = 4

    def __init__(self, visual_dim, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        from .official_moge_ssr.sparse_unet import Sparse3DUNet

        if tuple(channels) != (32, 64, 128, 256, 512):
            raise ValueError("Exp6-4 uses the pinned official channel configuration")
        self.network = Sparse3DUNet(
            in_channels=3, out_channels=1, encoder_channels=int(visual_dim),
            model_channels=list(channels), encoder_blocks_per_level=1,
            decoder_blocks_per_level=1, bottleneck_blocks=1,
            downsample_factors=[2, 2, 2, 2], encoder_downsample=16,
        )
        self.network.init_weights()

    def forward(self, shell, visual_features):
        features, coords, shape = official_sparse_inputs(shell, visual_features)
        residual = self.network(features, coords, shape, visual_features)
        if residual.shape != (shell.coordinates.shape[0], 1):
            raise RuntimeError("Official sparse decoder lost input voxels")
        # Official nearest upsampling explicitly restores each skip's coordinates.
        height, width = shell.image_size
        return residual[:, 0].reshape(shell.batch_size, height, width), []

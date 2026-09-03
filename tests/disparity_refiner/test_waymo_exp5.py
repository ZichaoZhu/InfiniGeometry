from __future__ import annotations

import numpy as np
import pytest
import torch

from training.disparity_refiner.waymo import (
    WaymoSparseSample,
    _rasterize_nearest,
    aggregate_sparse_metrics,
    camera_rays_from_calibration,
    range_image_to_vehicle_points,
    sparse_metrics,
)


def test_waymo_range_image_applies_per_pixel_motion_compensation() -> None:
    ranges = np.ones((1, 3), dtype=np.float32)
    pose = np.zeros((1, 3, 6), dtype=np.float32)
    pose[..., 3] = 3.0
    frame_pose = np.eye(4)
    frame_pose[0, 3] = 1.0
    points = range_image_to_vehicle_points(
        ranges,
        np.zeros(1),
        np.eye(4),
        range_image_pose=pose,
        frame_pose=frame_pose,
    )
    assert points[0, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert points[0, 1] == pytest.approx([3.0, 0.0, 0.0], abs=1e-6)


def test_waymo_rasterization_keeps_nearest_point() -> None:
    depth, points, mask = _rasterize_nearest(
        np.array([1, 1, 3]),
        np.array([1, 1, 3]),
        np.array([5.0, 2.0, 4.0]),
        np.array([[5.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]]),
        input_hw=(4, 4),
        output_hw=(4, 4),
    )
    assert mask.sum() == 2
    assert depth[1, 1] == pytest.approx(2.0)
    assert points[1, 1, 0] == pytest.approx(2.0)


def test_waymo_camera_rays_are_unit_length_and_forward_facing() -> None:
    rays = camera_rays_from_calibration(
        [4.0, 4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        input_hw=(4, 4),
        output_hw=(4, 4),
    )
    assert rays.shape == (4, 4, 3)
    assert np.linalg.norm(rays, axis=-1) == pytest.approx(np.ones((4, 4)))
    assert np.all(rays[..., 2] > 0)
    assert rays[1, 1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_waymo_sparse_metrics_use_only_held_out_points() -> None:
    prompt_mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    prompt_mask[0, 0, 0] = True
    evaluation_mask = torch.zeros(2, 2, dtype=torch.bool)
    evaluation_mask[1, 1] = True
    sample = WaymoSparseSample(
        sample_id="sample",
        image=torch.zeros(3, 2, 2),
        prompt_disparity=prompt_mask.float(),
        prompt_mask=prompt_mask,
        target_radial_depth=torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
        target_points=torch.zeros(2, 2, 3),
        evaluation_mask=evaluation_mask,
        reference_scale=torch.tensor(0.5),
        metadata={},
    )
    prediction = torch.tensor([[100.0, 100.0], [100.0, 1.0]])
    metrics = sparse_metrics(prediction, sample)
    assert metrics["metric_disparity_mae_1_per_m"] == pytest.approx(0.0)
    assert metrics["radial_depth_abs_rel"] == pytest.approx(0.0)
    assert metrics["point_delta_0_01"] == pytest.approx(1.0)
    aggregate = aggregate_sparse_metrics(
        {"sample": {"k0": metrics, "k3": metrics}}, (0, 3)
    )
    assert aggregate["k3"]["evaluation_point_count"] == 1.0

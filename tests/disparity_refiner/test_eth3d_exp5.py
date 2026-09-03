from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from training.disparity_refiner.eth3d import (
    ETH3DCamera,
    ETH3DSample,
    _camera_forward,
    camera_rays_from_colmap,
    center_crop_box,
    load_eth3d_sample,
    make_eth3d_lidar_prompt,
    metric_values,
    parse_colmap_images,
)
from training.disparity_refiner.prepare_eth3d_inputs import build_manifest


def _pinhole_camera(width: int, height: int) -> ETH3DCamera:
    return ETH3DCamera(
        camera_id=1,
        model="PINHOLE",
        width=width,
        height=height,
        params=(4.0, 4.0, (width - 1) / 2, (height - 1) / 2),
    )


def test_eth3d_pinhole_rays_are_unit_and_centered() -> None:
    rays = camera_rays_from_colmap(
        _pinhole_camera(4, 3),
        np.asarray([[1.5]], dtype=np.float64),
        np.asarray([[1.0]], dtype=np.float64),
    )
    assert rays.shape == (1, 1, 3)
    assert rays[0, 0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)


def test_eth3d_thin_prism_fisheye_forward_inverse_round_trip() -> None:
    camera = ETH3DCamera(
        camera_id=1,
        model="THIN_PRISM_FISHEYE",
        width=2048,
        height=1536,
        params=(1200.0, 1180.0, 1024.0, 768.0, 0.01, -0.002, 0.001, -0.001, 0.0002, -0.0001, 0.0003, -0.0002),
    )
    original_x = np.asarray([[0.17]], dtype=np.float64)
    original_y = np.asarray([[-0.11]], dtype=np.float64)
    distorted_x, distorted_y = _camera_forward(camera, original_x, original_y)
    rays = camera_rays_from_colmap(
        camera,
        distorted_x * camera.params[0] + camera.params[2],
        distorted_y * camera.params[1] + camera.params[3],
    )
    assert rays[0, 0, 0] / rays[0, 0, 2] == pytest.approx(original_x[0, 0], abs=2e-6)
    assert rays[0, 0, 1] / rays[0, 0, 2] == pytest.approx(original_y[0, 0], abs=2e-6)


def test_eth3d_loader_converts_optical_axis_depth_to_radial_range(tmp_path) -> None:
    rgb = np.full((3, 4, 3), 127, dtype=np.uint8)
    rgb_path = tmp_path / "frame.JPG"
    Image.fromarray(rgb).save(rgb_path)
    depth = np.full((3, 4), 2.0, dtype="<f4")
    depth[0, 0] = np.inf
    depth_path = tmp_path / "frame.raw"
    depth.tofile(depth_path)
    sample = load_eth3d_sample(
        {
            "id": "scene/frame",
            "scene": "scene",
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera": {
                "id": 1,
                "model": "PINHOLE",
                "width": 4,
                "height": 3,
                "params": [4.0, 4.0, 1.5, 1.0],
            },
        },
        output_hw=(3, 4),
    )
    assert sample.image.shape == (3, 3, 4)
    assert not sample.valid_mask[0, 0]
    assert sample.radial_depth[1, 1] == pytest.approx(2.0 / sample.rays[1, 1, 2].item())
    assert sample.radial_depth[0, 3] > 2.0
    assert sample.metadata["depth_representation"] == "camera_optical_axis_z_depth"


def test_eth3d_prompt_is_deterministic_and_metric_excludes_prompt_pixels() -> None:
    height, width = 4, 8
    rays = np.zeros((height, width, 3), dtype=np.float32)
    rays[..., 2] = 1.0
    rays[0, :, 1] = -0.25
    rays[-1, :, 1] = 0.25
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    sample = ETH3DSample(
        sample_id="scene/frame",
        scene="scene",
        image=torch.zeros(3, height, width),
        radial_depth=torch.full((height, width), 2.0),
        valid_mask=torch.ones(height, width, dtype=torch.bool),
        rays=torch.from_numpy(rays),
        metadata={},
    )
    settings = {
        "vertical_beams": 2,
        "horizontal_stride": 1,
        "vertical_margin_fraction": 0.02,
        "dropout": 0.0,
    }
    first = make_eth3d_lidar_prompt(sample, settings, seed=173)
    second = make_eth3d_lidar_prompt(sample, settings, seed=173)
    assert torch.equal(first[1], second[1])
    assert int(first[1].sum()) == 16
    prediction = torch.ones(height, width) / (2.0 * first[3])
    prediction[first[1]] = 50.0
    metrics = metric_values(
        prediction,
        sample,
        reference_scale=first[3],
        evaluation_mask=sample.valid_mask & ~first[1],
        segment_masks=None,
        min_segment_pixels=10,
        delta_threshold=0.01,
    )
    assert metrics["metric_disparity_mae_1_per_m"] == pytest.approx(0.0)
    assert metrics["evaluation_pixel_count"] == 16.0


def test_eth3d_center_crop_and_colmap_image_parser(tmp_path) -> None:
    assert center_crop_box((300, 600), (384, 512)) == (100, 0, 500, 300)
    images = tmp_path / "images.txt"
    images.write_text(
        "# comment\n1 1 0 0 0 0 0 0 7 dslr_images/frame.JPG\n\n",
        encoding="utf-8",
    )
    assert parse_colmap_images(images) == [
        {
            "image_id": 1,
            "camera_id": 7,
            "name": "dslr_images/frame.JPG",
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
        }
    ]


def test_eth3d_input_index_treats_ground_truth_jpg_as_raw_float_depth(tmp_path) -> None:
    scene = tmp_path / "pipes"
    calibration = scene / "dslr_calibration_jpg"
    image_dir = scene / "images" / "dslr_images"
    depth_dir = scene / "ground_truth_depth" / "dslr_images"
    calibration.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    (calibration / "cameras.txt").write_text("1 PINHOLE 4 3 4 4 1.5 1\n", encoding="utf-8")
    (calibration / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 dslr_images/DSC_0001.JPG\n\n", encoding="utf-8"
    )
    Image.fromarray(np.full((3, 4, 3), 127, dtype=np.uint8)).save(image_dir / "DSC_0001.JPG")
    np.ones((3, 4), dtype="<f4").tofile(depth_dir / "DSC_0001.JPG")
    manifest = build_manifest(tmp_path, expected_scenes=1, expected_samples=1)
    record = manifest["samples"][0]
    assert record["scene"] == "pipes"
    assert record["depth_path"].endswith("ground_truth_depth/dslr_images/DSC_0001.JPG")

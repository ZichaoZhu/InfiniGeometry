from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from training.disparity_refiner.lidar import local_point_metrics, normalized_disparity_to_geometry


@dataclass(frozen=True)
class ETH3DCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ETH3DSample:
    sample_id: str
    scene: str
    image: torch.Tensor
    radial_depth: torch.Tensor
    valid_mask: torch.Tensor
    rays: torch.Tensor
    metadata: dict[str, object]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_colmap_cameras(path: Path) -> dict[int, ETH3DCamera]:
    cameras: dict[int, ETH3DCamera] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed COLMAP camera line in {path}: {line}")
        camera = ETH3DCamera(
            camera_id=int(fields[0]),
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
        if camera.camera_id in cameras:
            raise ValueError(f"Duplicate COLMAP camera ID {camera.camera_id}")
        _validate_camera(camera)
        cameras[camera.camera_id] = camera
    if not cameras:
        raise ValueError(f"No COLMAP cameras found in {path}")
    return cameras


def parse_colmap_images(path: Path) -> list[dict[str, object]]:
    """Parse COLMAP's two-line image records while discarding 2D feature tracks."""
    records: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed COLMAP image line in {path}: {line}")
        record = {
            "image_id": int(fields[0]),
            "camera_id": int(fields[8]),
            "name": fields[9],
            "qvec": [float(value) for value in fields[1:5]],
            "tvec": [float(value) for value in fields[5:8]],
        }
        if index >= len(lines):
            raise ValueError(f"Missing feature-track line after image {record['image_id']}")
        index += 1
        records.append(record)
    if not records:
        raise ValueError(f"No COLMAP images found in {path}")
    return records


def _validate_camera(camera: ETH3DCamera) -> None:
    counts = {
        "SIMPLE_PINHOLE": 3,
        "PINHOLE": 4,
        "SIMPLE_RADIAL": 4,
        "RADIAL": 5,
        "OPENCV": 8,
        "FULL_OPENCV": 12,
        "OPENCV_FISHEYE": 8,
        "THIN_PRISM_FISHEYE": 12,
    }
    expected = counts.get(camera.model)
    if expected is None:
        raise ValueError(f"Unsupported ETH3D COLMAP camera model: {camera.model}")
    if len(camera.params) != expected:
        raise ValueError(
            f"ETH3D camera {camera.camera_id} {camera.model} needs {expected} params, "
            f"found {len(camera.params)}"
        )
    if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
        raise ValueError(f"Invalid ETH3D camera {camera.camera_id}")


def _intrinsics(camera: ETH3DCamera) -> tuple[float, float, float, float, tuple[float, ...]]:
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        focal, cx, cy, *extra = camera.params
        return focal, focal, cx, cy, tuple(extra)
    fx, fy, cx, cy, *extra = camera.params
    return fx, fy, cx, cy, tuple(extra)


def _camera_forward(camera: ETH3DCamera, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP normalized-coordinate forward distortion for the models ETH3D uses."""
    _, _, _, _, extra = _intrinsics(camera)
    radius2 = x * x + y * y
    if camera.model == "SIMPLE_PINHOLE" or camera.model == "PINHOLE":
        return x, y
    if camera.model == "SIMPLE_RADIAL":
        return x * (1.0 + extra[0] * radius2), y * (1.0 + extra[0] * radius2)
    if camera.model == "RADIAL":
        radial = 1.0 + extra[0] * radius2 + extra[1] * radius2 * radius2
        return x * radial, y * radial
    if camera.model in {"OPENCV", "FULL_OPENCV"}:
        k1, k2, p1, p2, *rest = extra
        k3 = rest[0] if rest else 0.0
        if camera.model == "FULL_OPENCV":
            k4, k5, k6 = rest[1:4]
            numerator = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
            denominator = 1.0 + k4 * radius2 + k5 * radius2**2 + k6 * radius2**3
            radial = numerator / denominator
        else:
            radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        return (
            x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x),
            y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    if camera.model in {"OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"}:
        if camera.model == "OPENCV_FISHEYE":
            k1, k2, k3, k4 = extra
            p1 = p2 = sx1 = sy1 = 0.0
        else:
            k1, k2, p1, p2, k3, k4, sx1, sy1 = extra
        radius = np.sqrt(radius2)
        theta = np.arctan(radius)
        theta2 = theta * theta
        radial = k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        scale = np.ones_like(radius)
        nonzero = radius > 1e-12
        scale[nonzero] = theta[nonzero] / radius[nonzero]
        fisheye_x = x * scale
        fisheye_y = y * scale
        fisheye_radius2 = fisheye_x * fisheye_x + fisheye_y * fisheye_y
        return (
            fisheye_x
            * (1.0 + radial)
            + 2.0 * p1 * fisheye_x * fisheye_y
            + p2 * (fisheye_radius2 + 2.0 * fisheye_x * fisheye_x)
            + sx1 * fisheye_radius2,
            fisheye_y
            * (1.0 + radial)
            + p1 * (fisheye_radius2 + 2.0 * fisheye_y * fisheye_y)
            + 2.0 * p2 * fisheye_x * fisheye_y
            + sy1 * fisheye_radius2,
        )
    raise AssertionError(camera.model)


def camera_rays_from_colmap(
    camera: ETH3DCamera,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
) -> np.ndarray:
    """Invert the original-image camera calibration and return unit OpenCV camera rays."""
    _validate_camera(camera)
    x_pixels = np.asarray(x_pixels, dtype=np.float64)
    y_pixels = np.asarray(y_pixels, dtype=np.float64)
    if x_pixels.shape != y_pixels.shape:
        raise ValueError("ETH3D pixel coordinate arrays must have the same shape")
    fx, fy, cx, cy, _ = _intrinsics(camera)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid ETH3D focal length for camera {camera.camera_id}")
    distorted_x = (x_pixels - cx) / fx
    distorted_y = (y_pixels - cy) / fy
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        undistorted_x, undistorted_y = distorted_x, distorted_y
    else:
        undistorted_x, undistorted_y = distorted_x.copy(), distorted_y.copy()
        # Vectorized Newton steps are deterministic and avoid a dependency on pycolmap.
        for _ in range(12):
            projected_x, projected_y = _camera_forward(camera, undistorted_x, undistorted_y)
            error_x = projected_x - distorted_x
            error_y = projected_y - distorted_y
            epsilon = 1e-6
            x_dx, y_dx = _camera_forward(camera, undistorted_x + epsilon, undistorted_y)
            x_dy, y_dy = _camera_forward(camera, undistorted_x, undistorted_y + epsilon)
            jacobian_xx = (x_dx - projected_x) / epsilon
            jacobian_yx = (y_dx - projected_y) / epsilon
            jacobian_xy = (x_dy - projected_x) / epsilon
            jacobian_yy = (y_dy - projected_y) / epsilon
            determinant = jacobian_xx * jacobian_yy - jacobian_xy * jacobian_yx
            stable = np.abs(determinant) > 1e-12
            delta_x = np.zeros_like(undistorted_x)
            delta_y = np.zeros_like(undistorted_y)
            delta_x[stable] = (
                jacobian_yy[stable] * error_x[stable]
                - jacobian_xy[stable] * error_y[stable]
            ) / determinant[stable]
            delta_y[stable] = (
                -jacobian_yx[stable] * error_x[stable]
                + jacobian_xx[stable] * error_y[stable]
            ) / determinant[stable]
            undistorted_x -= delta_x
            undistorted_y -= delta_y
        if not np.isfinite(undistorted_x).all() or not np.isfinite(undistorted_y).all():
            raise ValueError(f"ETH3D camera inversion diverged for camera {camera.camera_id}")
    rays = np.stack((undistorted_x, undistorted_y, np.ones_like(undistorted_x)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-12)
    return rays.astype(np.float32)


def center_crop_box(
    input_hw: tuple[int, int], output_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    input_height, input_width = input_hw
    output_height, output_width = output_hw
    if min(input_height, input_width, output_height, output_width) <= 0:
        raise ValueError("Image sizes must be positive")
    input_ratio = input_width / input_height
    output_ratio = output_width / output_height
    if input_ratio >= output_ratio:
        crop_width = int(round(input_height * output_ratio))
        crop_height = input_height
    else:
        crop_width = input_width
        crop_height = int(round(input_width / output_ratio))
    x0 = (input_width - crop_width) // 2
    y0 = (input_height - crop_height) // 2
    return x0, y0, x0 + crop_width, y0 + crop_height


def _output_pixel_centers(
    crop_box: tuple[int, int, int, int], output_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = crop_box
    output_height, output_width = output_hw
    x = (np.arange(output_width, dtype=np.float64) + 0.5) * (x1 - x0) / output_width + x0
    y = (np.arange(output_height, dtype=np.float64) + 0.5) * (y1 - y0) / output_height + y0
    return np.meshgrid(x, y)


def _read_depth_raw(path: Path, *, height: int, width: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(height) * int(width)
    if values.size != expected:
        raise ValueError(f"ETH3D depth size mismatch for {path}: {values.size} != {expected}")
    return values.reshape(height, width)


def _entry_camera(entry: Mapping[str, object]) -> ETH3DCamera:
    value = entry.get("camera")
    if not isinstance(value, Mapping):
        raise ValueError(f"ETH3D sample {entry.get('id')} has no camera")
    return ETH3DCamera(
        camera_id=int(value["id"]),
        model=str(value["model"]),
        width=int(value["width"]),
        height=int(value["height"]),
        params=tuple(float(item) for item in value["params"]),
    )


def load_eth3d_sample(
    entry: Mapping[str, object], *, output_hw: tuple[int, int]
) -> ETH3DSample:
    camera = _entry_camera(entry)
    rgb_path = Path(str(entry["rgb_path"])).resolve()
    depth_path = Path(str(entry["depth_path"])).resolve()
    if not rgb_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"Missing ETH3D input for {entry.get('id')}")
    with Image.open(rgb_path) as value:
        rgb_source = value.convert("RGB")
        input_width, input_height = rgb_source.size
        if (input_width, input_height) != (camera.width, camera.height):
            raise ValueError(
                f"ETH3D calibration/image size mismatch for {entry.get('id')}: "
                f"{camera.width}x{camera.height} != {input_width}x{input_height}"
            )
        crop_box = center_crop_box((input_height, input_width), output_hw)
        rgb = rgb_source.crop(crop_box).resize(
            (output_hw[1], output_hw[0]), Image.Resampling.LANCZOS
        )
    source_depth = _read_depth_raw(depth_path, height=input_height, width=input_width)
    grid_x, grid_y = _output_pixel_centers(crop_box, output_hw)
    nearest_x = np.clip(np.floor(grid_x).astype(np.int64), 0, input_width - 1)
    nearest_y = np.clip(np.floor(grid_y).astype(np.int64), 0, input_height - 1)
    z_depth = source_depth[nearest_y, nearest_x]
    rays = camera_rays_from_colmap(camera, grid_x, grid_y)
    valid = np.isfinite(z_depth) & (z_depth > 0.0) & np.isfinite(rays).all(axis=-1)
    radial = np.zeros_like(z_depth, dtype=np.float32)
    # ETH3D DSLR depth maps are camera optical-axis z-depth; convert to radial range.
    radial[valid] = z_depth[valid] / rays[..., 2][valid]
    image = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
    return ETH3DSample(
        sample_id=str(entry["id"]),
        scene=str(entry["scene"]),
        image=image,
        radial_depth=torch.from_numpy(radial),
        valid_mask=torch.from_numpy(valid),
        rays=torch.from_numpy(rays),
        metadata={
            "camera": {
                "id": camera.camera_id,
                "model": camera.model,
                "params": list(camera.params),
            },
            "crop_xyxy": list(crop_box),
            "depth_representation": "camera_optical_axis_z_depth",
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
        },
    )


def stable_prompt_seed(seed: int, sample_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{sample_id}".encode("utf-8")).digest()[:8], "little"
    )


def make_eth3d_lidar_prompt(
    sample: ETH3DSample,
    settings: Mapping[str, object],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the same deterministic virtual 64-line prompt construction as Exp4."""
    radial = sample.radial_depth.float()
    valid = sample.valid_mask.bool()
    height, width = radial.shape
    beams = int(settings["vertical_beams"])
    stride = int(settings["horizontal_stride"])
    margin = float(settings["vertical_margin_fraction"])
    dropout = float(settings.get("dropout", 0.0))
    if beams < 2 or stride < 1 or not 0.0 <= margin < 0.5 or not 0.0 <= dropout < 1.0:
        raise ValueError("Invalid ETH3D virtual LiDAR configuration")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    phase = int(torch.randint(stride, (), generator=generator).item()) if stride > 1 else 0
    columns = torch.arange(phase, width, stride)
    rays = sample.rays.float()
    elevation = torch.atan2(
        rays[..., 1], torch.linalg.vector_norm(rays[..., (0, 2)], dim=-1).clamp_min(1e-8)
    )
    low, high = elevation.amin(), elevation.amax()
    levels = torch.linspace(low + margin * (high - low), high - margin * (high - low), beams)
    distances = (elevation[:, columns][None] - levels[:, None, None]).abs()
    distances = distances.masked_fill(~valid[:, columns][None], float("inf"))
    rows = distances.argmin(dim=1)
    prompt_mask = torch.zeros_like(valid)
    beam_ids = torch.arange(beams)[:, None].expand_as(rows)
    column_ids = columns[None].expand_as(rows)
    selected = torch.isfinite(distances[beam_ids, rows, torch.arange(columns.numel())[None]])
    prompt_mask[rows[selected], column_ids[selected]] = True
    if dropout:
        positions = prompt_mask.nonzero(as_tuple=False)
        keep = torch.rand(positions.shape[0], generator=generator) >= dropout
        prompt_mask.zero_()
        kept = positions[keep]
        prompt_mask[kept[:, 0], kept[:, 1]] = True
    prompt_mask &= valid
    if int(prompt_mask.sum()) <= 5:
        raise ValueError(f"ETH3D prompt has fewer than six points: {sample.sample_id}")
    raw_disparity = torch.zeros_like(radial)
    raw_disparity[valid] = radial[valid].reciprocal()
    prompt_disparity = torch.where(prompt_mask, raw_disparity, torch.zeros_like(raw_disparity))
    reference_scale = torch.quantile(prompt_disparity[prompt_mask], 0.5)
    if not bool(torch.isfinite(reference_scale)) or float(reference_scale) <= 0.0:
        raise ValueError(f"ETH3D prompt has invalid disparity median: {sample.sample_id}")
    target_disparity = torch.where(valid, raw_disparity / reference_scale, torch.zeros_like(raw_disparity))
    return prompt_disparity, prompt_mask, target_disparity, reference_scale


@torch.no_grad()
def metric_values(
    normalized_disparity: torch.Tensor,
    sample: ETH3DSample,
    *,
    reference_scale: torch.Tensor,
    evaluation_mask: torch.Tensor,
    segment_masks: torch.Tensor | None,
    min_segment_pixels: int,
    delta_threshold: float,
) -> dict[str, float | None]:
    target_depth = sample.radial_depth.to(normalized_disparity.device)
    target_valid = sample.valid_mask.to(normalized_disparity.device)
    rays = sample.rays.to(normalized_disparity.device)
    radial, prediction_points, prediction_valid = normalized_disparity_to_geometry(
        normalized_disparity, reference_scale.to(normalized_disparity.device), rays
    )
    valid = evaluation_mask.bool().to(normalized_disparity.device) & target_valid & prediction_valid
    if not bool(valid.any()):
        raise ValueError(f"ETH3D prediction has no valid evaluation pixels: {sample.sample_id}")
    target_disparity = target_depth[valid].reciprocal()
    prediction_disparity = (normalized_disparity * reference_scale)[valid]
    error = (radial[valid] - target_depth[valid]).abs()
    result: dict[str, float | None] = {
        "evaluation_pixel_count": float(valid.sum().item()),
        "metric_disparity_mae_1_per_m": float((prediction_disparity - target_disparity).abs().mean().item()),
        "radial_depth_abs_rel": float((error / target_depth[valid]).mean().item()),
        "radial_depth_rmse_m": float(torch.sqrt(error.square().mean()).item()),
        "point_delta_0_01": float(
            (error < float(delta_threshold) * torch.minimum(radial[valid], target_depth[valid]))
            .float()
            .mean()
            .item()
        ),
    }
    if segment_masks is not None:
        target_points = rays * target_depth[..., None]
        result.update(
            local_point_metrics(
                prediction_points,
                target_points,
                valid,
                segment_masks.to(normalized_disparity.device),
                min_segment_pixels=min_segment_pixels,
                delta_threshold=delta_threshold,
            )
        )
    return result


def read_input_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "infinidepth-exp5-eth3d-input-manifest-v1":
        raise ValueError(f"Unsupported ETH3D input manifest: {path}")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"ETH3D input manifest has no samples: {path}")
    return value

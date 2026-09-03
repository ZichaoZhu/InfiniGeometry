from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
import sys
from typing import Any, Sequence
import zlib

# The lightweight reader ships legacy generated protos. Keep the compatibility
# backend local to Waymo evaluation instead of downgrading the training environment.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class WaymoSparseSample:
    sample_id: str
    image: torch.Tensor
    prompt_disparity: torch.Tensor
    prompt_mask: torch.Tensor
    target_radial_depth: torch.Tensor
    target_points: torch.Tensor
    evaluation_mask: torch.Tensor
    reference_scale: torch.Tensor
    metadata: dict[str, object]


def list_tfrecords(root: Path, *, expected_count: int | None = None) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".tfrecord")
    if expected_count is not None and len(paths) != int(expected_count):
        raise ValueError(f"Expected {expected_count} TFRecords under {root}, found {len(paths)}")
    return paths


def _reader_modules(reader_root: Path) -> tuple[Any, Any]:
    reader_root = reader_root.expanduser().resolve()
    package = reader_root / "simple_waymo_open_dataset_reader"
    if not package.is_dir():
        raise FileNotFoundError(package)
    root_text = str(reader_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2

    return WaymoDataFileReader, dataset_pb2


def _decode_matrix(blob: bytes, message_type: Any) -> np.ndarray:
    value = message_type()
    value.ParseFromString(zlib.decompress(blob))
    return np.asarray(value.data).reshape(value.shape.dims)


def camera_rays_from_calibration(
    intrinsic: Sequence[float],
    *,
    input_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Return unit camera rays for the resized, distorted Waymo image grid."""
    values = np.asarray(intrinsic, dtype=np.float64).reshape(-1)
    if values.size != 9:
        raise ValueError(f"Expected 9 Waymo camera intrinsic values, found {values.size}")
    input_height, input_width = input_hw
    output_height, output_width = output_hw
    if min(input_height, input_width, output_height, output_width) <= 0:
        raise ValueError("Camera image dimensions must be positive")
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = values
    if not np.isfinite(values).all() or fu <= 0 or fv <= 0:
        raise ValueError("Waymo camera intrinsic values are invalid")
    x = (np.arange(output_width, dtype=np.float64) + 0.5) * input_width / output_width - 0.5
    y = (np.arange(output_height, dtype=np.float64) + 0.5) * input_height / output_height - 0.5
    grid_x, grid_y = np.meshgrid(x, y)
    distorted_x = (grid_x - cu) / fu
    distorted_y = (grid_y - cv) / fv
    undistorted_x = distorted_x.copy()
    undistorted_y = distorted_y.copy()
    for _ in range(8):
        radius2 = undistorted_x**2 + undistorted_y**2
        radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        delta_x = 2 * p1 * undistorted_x * undistorted_y + p2 * (
            radius2 + 2 * undistorted_x**2
        )
        delta_y = p1 * (radius2 + 2 * undistorted_y**2) + 2 * p2 * (
            undistorted_x * undistorted_y
        )
        undistorted_x = (distorted_x - delta_x) / radial
        undistorted_y = (distorted_y - delta_y) / radial
    normalized = np.stack((undistorted_x, undistorted_y), axis=-1)
    rays = np.concatenate(
        (normalized, np.ones((output_height, output_width, 1), dtype=np.float64)), axis=-1
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-12)
    return rays.astype(np.float32)


def _rotation_zyx(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        axis=-1,
    ).reshape((*roll.shape, 3, 3))


def range_image_to_vehicle_points(
    ranges: np.ndarray,
    beam_inclinations: np.ndarray,
    lidar_extrinsic: np.ndarray,
    *,
    range_image_pose: np.ndarray | None,
    frame_pose: np.ndarray,
) -> np.ndarray:
    """Convert a Waymo range image to the vehicle frame at the frame timestamp."""
    ranges = np.asarray(ranges, dtype=np.float64)
    lidar_extrinsic = np.asarray(lidar_extrinsic, dtype=np.float64).reshape(4, 4)
    frame_pose = np.asarray(frame_pose, dtype=np.float64).reshape(4, 4)
    height, width = ranges.shape
    inclinations = np.asarray(beam_inclinations, dtype=np.float64).reshape(-1)
    if inclinations.size != height:
        raise ValueError(f"Expected {height} beam inclinations, found {inclinations.size}")
    inclinations = inclinations[::-1]
    azimuth_correction = np.arctan2(lidar_extrinsic[1, 0], lidar_extrinsic[0, 0])
    azimuth = np.linspace(np.pi, -np.pi, width) - azimuth_correction
    cos_inc = np.cos(inclinations)[:, None]
    lidar_points = np.stack(
        (
            np.cos(azimuth)[None] * cos_inc * ranges,
            np.sin(azimuth)[None] * cos_inc * ranges,
            np.sin(inclinations)[:, None] * np.ones((1, width)) * ranges,
            np.ones_like(ranges),
        ),
        axis=-1,
    )
    points = np.einsum("ij,hwj->hwi", lidar_extrinsic, lidar_points)
    if range_image_pose is not None:
        pose = np.asarray(range_image_pose, dtype=np.float64)
        if pose.shape != (height, width, 6):
            raise ValueError(f"Unexpected range-image pose shape: {pose.shape}")
        rotation = _rotation_zyx(pose[..., 0], pose[..., 1], pose[..., 2])
        global_xyz = np.einsum("hwij,hwj->hwi", rotation, points[..., :3]) + pose[..., 3:]
        global_points = np.concatenate((global_xyz, np.ones((*ranges.shape, 1))), axis=-1)
        points = np.einsum("ij,hwj->hwi", np.linalg.inv(frame_pose), global_points)
    return points[..., :3].astype(np.float32)


def _rasterize_nearest(
    x: np.ndarray,
    y: np.ndarray,
    radial: np.ndarray,
    points: np.ndarray,
    *,
    input_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_height, input_width = input_hw
    output_height, output_width = output_hw
    x = np.floor(np.asarray(x) * output_width / input_width).astype(np.int64)
    y = np.floor(np.asarray(y) * output_height / input_height).astype(np.int64)
    radial = np.asarray(radial, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    valid = (
        np.isfinite(radial)
        & (radial > 0)
        & (x >= 0)
        & (x < output_width)
        & (y >= 0)
        & (y < output_height)
        & np.isfinite(points).all(axis=-1)
    )
    x, y, radial, points = x[valid], y[valid], radial[valid], points[valid]
    flat = y * output_width + x
    order = np.lexsort((radial, flat))
    flat = flat[order]
    first = np.empty(flat.size, dtype=bool)
    if flat.size:
        first[0] = True
        first[1:] = flat[1:] != flat[:-1]
    chosen = order[first]
    mask = np.zeros(output_height * output_width, dtype=bool)
    depth = np.zeros(output_height * output_width, dtype=np.float32)
    point_map = np.zeros((output_height * output_width, 3), dtype=np.float32)
    selected_flat = y[chosen] * output_width + x[chosen]
    mask[selected_flat] = True
    depth[selected_flat] = radial[chosen]
    point_map[selected_flat] = points[chosen]
    return (
        depth.reshape(output_height, output_width),
        point_map.reshape(output_height, output_width, 3),
        mask.reshape(output_height, output_width),
    )


def _frame_to_sample(
    frame: Any,
    dataset_pb2: Any,
    *,
    source_name: str,
    frame_index: int,
    camera_id: int,
    lidar_id: int,
    output_hw: tuple[int, int],
    prompt_stride: int,
    min_prompt_points: int,
    min_evaluation_points: int,
) -> WaymoSparseSample:
    camera = next((value for value in frame.images if value.name == camera_id), None)
    camera_calibration = next(
        (value for value in frame.context.camera_calibrations if value.name == camera_id), None
    )
    laser = next((value for value in frame.lasers if value.name == lidar_id), None)
    laser_calibration = next(
        (value for value in frame.context.laser_calibrations if value.name == lidar_id), None
    )
    if any(value is None for value in (camera, camera_calibration, laser, laser_calibration)):
        raise ValueError("Frame is missing the requested camera, LiDAR, or calibration")
    if prompt_stride < 2:
        raise ValueError("prompt_stride must be at least two to leave held-out points")

    image = Image.open(io.BytesIO(camera.image)).convert("RGB")
    input_width, input_height = image.size
    height, width = output_hw
    resized = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)
    resized = resized.copy() / 255.0

    return1 = laser.ri_return1
    range_image = _decode_matrix(return1.range_image_compressed, dataset_pb2.MatrixFloat)
    camera_projection = _decode_matrix(
        return1.camera_projection_compressed, dataset_pb2.MatrixInt32
    )
    range_image_pose = _decode_matrix(
        return1.range_image_pose_compressed, dataset_pb2.MatrixFloat
    )
    ranges = range_image[..., 0]
    inclinations = np.asarray(laser_calibration.beam_inclinations, dtype=np.float64)
    if inclinations.size == 0:
        inclinations = np.linspace(
            laser_calibration.beam_inclination_min,
            laser_calibration.beam_inclination_max,
            ranges.shape[0],
        )
    frame_pose = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    vehicle_points = range_image_to_vehicle_points(
        ranges,
        inclinations,
        np.asarray(laser_calibration.extrinsic.transform).reshape(4, 4),
        range_image_pose=range_image_pose,
        frame_pose=frame_pose,
    )
    global_points = np.einsum(
        "ij,hwj->hwi",
        frame_pose,
        np.concatenate((vehicle_points, np.ones((*ranges.shape, 1), dtype=np.float32)), axis=-1),
    )
    camera_pose = np.asarray(camera.pose.transform, dtype=np.float64).reshape(4, 4)
    camera_extrinsic = np.asarray(camera_calibration.extrinsic.transform, dtype=np.float64).reshape(
        4, 4
    )
    camera_points = np.einsum(
        "ij,hwj->hwi",
        np.linalg.inv(camera_extrinsic) @ np.linalg.inv(camera_pose),
        global_points,
    )[..., :3].astype(np.float32)
    radial = np.linalg.norm(camera_points, axis=-1)

    first_projection = camera_projection[..., 0] == camera_id
    second_projection = camera_projection[..., 3] == camera_id
    projected = first_projection | second_projection
    x = np.where(first_projection, camera_projection[..., 1], camera_projection[..., 4])
    y = np.where(first_projection, camera_projection[..., 2], camera_projection[..., 5])
    columns = np.broadcast_to(np.arange(ranges.shape[1])[None], ranges.shape)
    valid = projected & np.isfinite(ranges) & (ranges > 0) & np.isfinite(radial) & (radial > 0)
    prompt_source = valid & (columns % prompt_stride == 0)
    evaluation_source = valid & ~prompt_source

    prompt_depth, _, prompt_mask = _rasterize_nearest(
        x[prompt_source],
        y[prompt_source],
        radial[prompt_source],
        camera_points[prompt_source],
        input_hw=(input_height, input_width),
        output_hw=output_hw,
    )
    target_depth, target_points, evaluation_mask = _rasterize_nearest(
        x[evaluation_source],
        y[evaluation_source],
        radial[evaluation_source],
        camera_points[evaluation_source],
        input_hw=(input_height, input_width),
        output_hw=output_hw,
    )
    evaluation_mask &= ~prompt_mask
    target_depth[~evaluation_mask] = 0
    target_points[~evaluation_mask] = 0
    prompt_count = int(prompt_mask.sum())
    evaluation_count = int(evaluation_mask.sum())
    if prompt_count < min_prompt_points or evaluation_count < min_evaluation_points:
        raise ValueError(
            f"Insufficient projected points: prompt={prompt_count}, evaluation={evaluation_count}"
        )
    prompt_disparity = np.zeros_like(prompt_depth)
    prompt_disparity[prompt_mask] = 1.0 / prompt_depth[prompt_mask]
    reference_scale = float(np.median(prompt_disparity[prompt_mask]))
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Prompt disparity has an invalid median")
    context = str(frame.context.name)
    timestamp = int(frame.timestamp_micros)
    camera_text = dataset_pb2.CameraName.Name.Name(camera_id)
    sample_id = f"{context}_{timestamp}_{camera_text}"
    return WaymoSparseSample(
        sample_id=sample_id,
        image=torch.from_numpy(resized).permute(2, 0, 1).contiguous(),
        prompt_disparity=torch.from_numpy(prompt_disparity)[None].contiguous(),
        prompt_mask=torch.from_numpy(prompt_mask)[None].contiguous(),
        target_radial_depth=torch.from_numpy(target_depth).contiguous(),
        target_points=torch.from_numpy(target_points).contiguous(),
        evaluation_mask=torch.from_numpy(evaluation_mask).contiguous(),
        reference_scale=torch.tensor(reference_scale, dtype=torch.float32),
        metadata={
            "camera": camera_text,
            "camera_image_height": input_height,
            "camera_image_width": input_width,
            "camera_intrinsic": [float(value) for value in camera_calibration.intrinsic],
            "evaluation_point_count": evaluation_count,
            "frame_index": int(frame_index),
            "location": str(frame.context.stats.location),
            "prompt_point_count": prompt_count,
            "source": source_name,
            "time_of_day": str(frame.context.stats.time_of_day),
            "timestamp_micros": timestamp,
            "weather": str(frame.context.stats.weather),
        },
    )


def load_waymo_sample(
    path: Path,
    *,
    reader_root: Path,
    frame_index: int,
    frame_search: int,
    output_hw: tuple[int, int],
    prompt_stride: int,
    camera_name: str = "FRONT",
    lidar_name: str = "TOP",
    min_prompt_points: int = 6,
    min_evaluation_points: int = 64,
) -> WaymoSparseSample:
    if frame_index < 0 or frame_search < 0:
        raise ValueError("frame_index and frame_search must be non-negative")
    WaymoDataFileReader, dataset_pb2 = _reader_modules(reader_root)
    try:
        camera_id = dataset_pb2.CameraName.Name.Value(camera_name)
        lidar_id = dataset_pb2.LaserName.Name.Value(lidar_name)
    except ValueError as error:
        raise ValueError(f"Unsupported Waymo camera or LiDAR: {camera_name}, {lidar_name}") from error
    reader = WaymoDataFileReader(str(path))
    last_error: Exception | None = None
    try:
        for _ in range(frame_index):
            reader.read_record(header_only=True)
        for offset in range(frame_search + 1):
            index = frame_index + offset
            try:
                frame = reader.read_record()
            except StopIteration:
                break
            try:
                return _frame_to_sample(
                    frame,
                    dataset_pb2,
                    source_name=path.name,
                    frame_index=index,
                    camera_id=camera_id,
                    lidar_id=lidar_id,
                    output_hw=output_hw,
                    prompt_stride=prompt_stride,
                    min_prompt_points=min_prompt_points,
                    min_evaluation_points=min_evaluation_points,
                )
            except ValueError as error:
                last_error = error
    finally:
        reader.file.close()
    detail = f": {last_error}" if last_error is not None else ""
    raise ValueError(f"No valid Waymo frame found in {path.name}{detail}")


def sparse_metrics(
    normalized_disparity: torch.Tensor,
    sample: WaymoSparseSample,
) -> dict[str, float]:
    scale = sample.reference_scale.to(normalized_disparity.device)
    target_depth = sample.target_radial_depth.to(normalized_disparity.device)
    evaluation_mask = sample.evaluation_mask.to(normalized_disparity.device)
    metric_disparity = normalized_disparity.float() * scale
    valid = evaluation_mask & torch.isfinite(metric_disparity) & (metric_disparity > 1e-6)
    if int(valid.sum()) < 1:
        raise ValueError("Prediction has no valid held-out LiDAR points")
    target_disparity = target_depth[valid].reciprocal()
    prediction_disparity = metric_disparity[valid]
    prediction_depth = prediction_disparity.reciprocal()
    target = target_depth[valid]
    point_error = (prediction_depth - target).abs()
    return {
        "evaluation_point_count": float(valid.sum().item()),
        "metric_disparity_mae_1_per_m": float(
            (prediction_disparity - target_disparity).abs().mean().item()
        ),
        "radial_depth_abs_rel": float((point_error / target).mean().item()),
        "radial_depth_rmse_m": float(torch.sqrt((point_error.square()).mean()).item()),
        "point_delta_0_01": float(
            (point_error < 0.01 * torch.minimum(prediction_depth, target)).float().mean().item()
        ),
    }


def aggregate_sparse_metrics(
    per_image: dict[str, dict[str, dict[str, float]]], iterations: Sequence[int]
) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for iteration in iterations:
        key = f"k{iteration}"
        names = set().union(*(value[key] for value in per_image.values()))
        aggregate[key] = {
            name: float(np.mean([float(value[key][name]) for value in per_image.values()]))
            for name in names
        }
    return aggregate

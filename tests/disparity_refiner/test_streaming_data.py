from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from training.disparity_refiner.data import (
    HypersimDisparityDataset,
    load_manifest,
    select_manifest_entries,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_indexed_manifest_builds_lazy_samples(tmp_path: Path) -> None:
    rgb = tmp_path / "scene" / "images" / "scene_cam_00_final_preview"
    depth = tmp_path / "scene" / "images" / "scene_cam_00_geometry_hdf5"
    rgb.mkdir(parents=True)
    depth.mkdir(parents=True)
    Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(
        rgb / "frame.0000.tonemap.jpg"
    )
    with h5py.File(depth / "frame.0000.depth_meters.hdf5", "w") as handle:
        handle["dataset"] = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
    image_index = tmp_path / "images.csv"
    image_index.write_text(
        "scene_name,camera_name,frame_id,split_partition_name\n"
        "scene,cam_00,0,train\n",
        encoding="utf-8",
    )
    camera_index = tmp_path / "cameras.csv"
    camera_index.write_text(
        "scene_name,settings_output_img_height,settings_output_img_width,"
        + ",".join(f"M_cam_from_uv_{i}{j}" for i in range(3) for j in range(3))
        + "\nscene,4,6,1,0,0,0,1,0,0,0,-1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "depth_semantics": "Euclidean distance in meters from camera optical center",
                "image_index": {"path": "images.csv", "sha256": digest(image_index)},
                "camera_index": {"path": "cameras.csv", "sha256": digest(camera_index)},
            }
        ),
        encoding="utf-8",
    )
    entries = select_manifest_entries(
        load_manifest(manifest_path),
        source_root=tmp_path,
        split="train",
    )
    dataset = HypersimDisparityDataset(
        entries,
        source_root=tmp_path,
        height=4,
        width=6,
        structure_selections={},
    )
    assert dataset.sample_ids == ["scene_cam_00_frame.0000"]
    assert "image" not in dataset.entries[0]
    sample = dataset[0]
    assert sample.image.shape == (3, 4, 6)
    assert sample.metadata["M_cam_from_uv"][2][2] == -1


def test_lazy_dataset_populates_and_reuses_local_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rgb = source / "scene/images/scene_cam_00_final_preview"
    depth = source / "scene/images/scene_cam_00_geometry_hdf5"
    rgb.mkdir(parents=True)
    depth.mkdir(parents=True)
    rgb_path = rgb / "frame.0000.tonemap.jpg"
    depth_path = depth / "frame.0000.depth_meters.hdf5"
    Image.fromarray(np.full((4, 6, 3), 63, dtype=np.uint8)).save(rgb_path)
    with h5py.File(depth_path, "w") as handle:
        handle["dataset"] = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
    entry = {
        "id": "scene_cam_00_frame.0000",
        "rgb": {"source": str(rgb_path)},
        "depth": {"source": str(depth_path)},
    }
    cache = tmp_path / "cache"
    dataset = HypersimDisparityDataset(
        [entry],
        source_root=source,
        height=4,
        width=6,
        structure_selections={},
        cache_root=cache,
    )
    first = dataset[0]
    rgb_path.unlink()
    depth_path.unlink()
    second = dataset[0]
    assert np.array_equal(first.image.numpy(), second.image.numpy())
    assert (cache / "scene/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg").is_file()
    assert (cache / "scene/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5").is_file()

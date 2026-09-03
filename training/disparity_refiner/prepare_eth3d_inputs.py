from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.disparity_refiner.eth3d import parse_colmap_cameras, parse_colmap_images, sha256


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index extracted ETH3D high-res DSLR training inputs")
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=13)
    parser.add_argument("--expected-samples", type=int, default=454)
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_parts(path: Path, root: Path) -> set[str]:
    return set(path.resolve().relative_to(root.resolve()).parts[:-1])


def _best_candidate(
    candidates: Iterable[Path], *, reference: Path, root: Path, expected_scene: str | None
) -> Path:
    options = list(candidates)
    if not options:
        raise FileNotFoundError(reference)
    reference_parts = _relative_parts(reference, root)
    def score(candidate: Path) -> tuple[int, int, str]:
        parts = _relative_parts(candidate, root)
        scene_score = int(expected_scene is not None and expected_scene in parts)
        return scene_score, len(parts & reference_parts), str(candidate)
    return max(options, key=score)


def _scene_name(path: Path, root: Path) -> str:
    parts = path.resolve().relative_to(root.resolve()).parts
    if not parts:
        raise ValueError(f"Cannot infer ETH3D scene from {path}")
    for part in parts:
        if part.endswith("_dslr_depth"):
            return part[: -len("_dslr_depth")]
    return parts[0]


def build_manifest(root: Path, *, expected_scenes: int, expected_samples: int) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not (path.parent.name == "dslr_images" and path.parent.parent.name == "ground_truth_depth")
    )
    raw_depths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.parent.name == "dslr_images" and path.parent.parent.name == "ground_truth_depth"
    )
    by_image_name: dict[str, list[Path]] = {}
    by_depth_name: dict[str, list[Path]] = {}
    for path in images:
        by_image_name.setdefault(path.name, []).append(path)
    for path in raw_depths:
        by_depth_name.setdefault(path.name, []).append(path)
    records: list[dict[str, object]] = []
    calibration_files = sorted(root.rglob("cameras.txt"))
    for cameras_path in calibration_files:
        images_path = cameras_path.with_name("images.txt")
        if not images_path.is_file():
            continue
        cameras = parse_colmap_cameras(cameras_path)
        for record in parse_colmap_images(images_path):
            image_name = Path(str(record["name"])).name
            # ETH3D stores raw float32 depth under the original DSLR ``.JPG`` name.
            depth_name = image_name
            rgb_path = _best_candidate(
                by_image_name.get(image_name, []),
                reference=images_path,
                root=root,
                expected_scene=None,
            )
            depth_path = _best_candidate(
                by_depth_name.get(depth_name, []),
                reference=rgb_path,
                root=root,
                expected_scene=None,
            )
            scene = _scene_name(depth_path, root)
            camera = cameras.get(int(record["camera_id"]))
            if camera is None:
                raise ValueError(f"Image {record['name']} references an unknown camera")
            records.append(
                {
                    "id": f"{scene}/{Path(image_name).stem}",
                    "scene": scene,
                    "image_name": str(record["name"]),
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "camera": {
                        "id": camera.camera_id,
                        "model": camera.model,
                        "width": camera.width,
                        "height": camera.height,
                        "params": list(camera.params),
                    },
                    "calibration": {
                        "cameras_path": str(cameras_path),
                        "images_path": str(images_path),
                        "cameras_sha256": sha256(cameras_path),
                        "images_sha256": sha256(images_path),
                    },
                }
            )
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        duplicate = next(value for value in ids if ids.count(value) > 1)
        raise ValueError(f"Duplicate ETH3D sample ID: {duplicate}")
    records.sort(key=lambda value: (str(value["scene"]), str(value["id"])))
    scenes = sorted({str(record["scene"]) for record in records})
    if len(scenes) != int(expected_scenes) or len(records) != int(expected_samples):
        raise ValueError(
            f"ETH3D layout mismatch: scenes={len(scenes)}/{expected_scenes}, "
            f"samples={len(records)}/{expected_samples}"
        )
    camera_models: dict[str, int] = {}
    for record in records:
        model = str(record["camera"]["model"])
        camera_models[model] = camera_models.get(model, 0) + 1
    return {
        "format": "infinidepth-exp5-eth3d-input-manifest-v1",
        "dataset": "ETH3D high-res multi-view training DSLR",
        "depth_representation": "camera_optical_axis_z_depth",
        "depth_to_radial_range": "range = z / unit_camera_ray_z",
        "scene_count": len(scenes),
        "sample_count": len(records),
        "scenes": scenes,
        "camera_models": camera_models,
        "source_root": str(root),
        "samples": records,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.extracted_root,
        expected_scenes=args.expected_scenes,
        expected_samples=args.expected_samples,
    )
    _atomic_json(args.output, manifest)
    print(args.output)


if __name__ == "__main__":
    main()

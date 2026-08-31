from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from training.disparity_refiner.backup import atomic_copy, atomic_json, sha256
from training.disparity_refiner.data import ensure_within
from training.disparity_refiner.lidar import (
    coarse_fine_mask,
    load_segment_masks,
    save_segment_masks,
    select_fine_segments,
)
from training.disparity_refiner.train import _load_samples, _selected_run
from training.disparity_refiner.train_lidar import _evaluation_sample_ids


MASK_FORMAT = "infinidepth-moge3-local-mask-manifest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed Exp4 MoGe-3 local masks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise FileExistsError(f"Existing backup differs: {destination}")
        return
    atomic_copy(source, destination)


def _preview(path: Path, image: np.ndarray, coarse: np.ndarray, segments: np.ndarray) -> None:
    overlay = image.copy()
    overlay[coarse] = (0.55 * overlay[coarse] + 0.45 * np.asarray([255, 80, 40])).astype(
        np.uint8
    )
    union = segments.any(axis=0) if len(segments) else np.zeros_like(coarse)
    overlay[union] = (0.45 * overlay[union] + 0.55 * np.asarray([30, 220, 90])).astype(
        np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    if not cv2.imwrite(str(temporary), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write mask preview: {temporary}")
    temporary.replace(path)


def _load_generator(settings: Mapping[str, object], device: str):
    repository = Path(str(settings["repository"])).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != str(settings["repository_commit"]):
        raise ValueError(f"SAM2 repository commit mismatch: {commit}")
    sys.path.insert(0, str(repository))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    checkpoint = Path(str(settings["checkpoint"])).resolve()
    if sha256(checkpoint) != str(settings["checkpoint_sha256"]):
        raise ValueError("SAM2 checkpoint SHA-256 mismatch")
    model = build_sam2(
        str(settings["model_config"]),
        str(checkpoint),
        device=device,
        apply_postprocessing=bool(settings.get("apply_postprocessing", True)),
    )
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=int(settings.get("points_per_side", 32)),
        points_per_batch=int(settings.get("points_per_batch", 64)),
        pred_iou_thresh=float(settings.get("pred_iou_thresh", 0.8)),
        stability_score_thresh=float(settings.get("stability_score_thresh", 0.95)),
        box_nms_thresh=float(settings.get("box_nms_thresh", 0.7)),
        crop_n_layers=int(settings.get("crop_n_layers", 0)),
        min_mask_region_area=int(settings.get("min_mask_region_area", 0)),
        output_mode="binary_mask",
    )


def run(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for SAM2, but CUDA is unavailable")
    project_root = Path(__file__).resolve().parents[2]
    config_path = ensure_within(args.config, project_root, name="config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["evaluation"]["local_points"]
    output_root = Path(str(config["server"]["output_root"]))
    output = ensure_within(
        args.output or Path(str(settings["mask_root"])), output_root, name="local mask output"
    )
    backup_root = Path(str(config["server"]["backup_root"]))
    backup = ensure_within(
        backup_root / output.relative_to(output_root), backup_root, name="local mask backup"
    )
    output.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)

    expected_ids = list(_evaluation_sample_ids(config, project_root))
    requested = expected_ids if args.sample_id is None else [str(value) for value in args.sample_id]
    missing = sorted(set(requested) - set(expected_ids))
    if missing:
        raise ValueError(f"Requested samples are not in fixed Val100: {missing}")
    samples = _load_samples(
        config,
        _selected_run(config, "main"),
        project_root,
        split=str(config["evaluation"]["split"]),
        sample_ids=requested,
    )
    generator = _load_generator(settings["sam2"], args.device)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != MASK_FORMAT or manifest.get("settings") != settings:
            raise ValueError("Existing local-mask manifest uses other settings")
    else:
        manifest = {
            "format": MASK_FORMAT,
            "paper_protocol": "MoGe-3 arXiv:2607.17967v2 Appendix B",
            "implementation_note": "Paper formulas with pinned OpenCV/SAM2 choices; not claimed bit-exact to unreleased preprocessing code.",
            "settings": settings,
            "expected_sample_ids": expected_ids,
            "samples": {},
            "created_at_unix": time.time(),
        }

    for sample in samples:
        relative = Path("segments") / f"{sample.sample_id}.npz"
        mask_path = output / relative
        image = (
            sample.image.permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
        )
        valid = sample.valid_mask.cpu().numpy()
        disparity = np.zeros_like(sample.radial_depth.cpu().numpy(), dtype=np.float32)
        disparity[valid] = 1.0 / sample.radial_depth.cpu().numpy()[valid]
        coarse = coarse_fine_mask(
            disparity,
            valid,
            residual_scales=settings["residual_scales"],
            morphology_sizes=settings["morphology_sizes"],
            threshold_sigma=float(settings["threshold_sigma"]),
        )
        if mask_path.exists():
            segments = load_segment_masks(mask_path)
        else:
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if str(args.device).startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                candidates = [value["segmentation"] for value in generator.generate(image)]
            segments = select_fine_segments(
                candidates,
                coarse,
                valid,
                min_density=float(settings["min_density"]),
                min_overlap_pixels=int(settings["min_overlap_pixels"]),
                max_area_fraction=float(settings["max_area_fraction"]),
            )
            save_segment_masks(mask_path, segments)
        preview_path = output / "previews" / f"{sample.sample_id}.png"
        if not preview_path.exists():
            _preview(preview_path, image, coarse, segments)
        _copy_immutable(mask_path, backup / relative)
        _copy_immutable(preview_path, backup / preview_path.relative_to(output))
        manifest["samples"][sample.sample_id] = {
            "path": str(relative),
            "sha256": sha256(mask_path),
            "height": int(segments.shape[1]),
            "width": int(segments.shape[2]),
            "segment_count": int(segments.shape[0]),
            "coarse_pixel_count": int(coarse.sum()),
            "selected_union_pixel_count": int(segments.any(axis=0).sum()) if len(segments) else 0,
            "preview": str(preview_path.relative_to(output)),
            "preview_sha256": sha256(preview_path),
        }
        manifest["sample_count"] = len(manifest["samples"])
        manifest["status"] = (
            "complete" if set(manifest["samples"]) == set(expected_ids) else "partial"
        )
        manifest["updated_at_unix"] = time.time()
        atomic_json(manifest_path, manifest)
        atomic_copy(manifest_path, backup / "manifest.json")
        print(
            f"{sample.sample_id}: coarse={int(coarse.sum())}, segments={len(segments)}",
            flush=True,
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from typing import Mapping

import cv2
import numpy as np
import torch

from training.disparity_refiner.eth3d import load_eth3d_sample, read_input_manifest, sha256
from training.disparity_refiner.lidar import coarse_fine_mask, load_segment_masks, save_segment_masks, select_fine_segments
from training.disparity_refiner.prepare_local_masks import _load_generator


MASK_FORMAT = "infinidepth-moge3-local-mask-manifest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed MoGe-3 local masks for ETH3D Exp5")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _preview(path: Path, image: np.ndarray, coarse: np.ndarray, segments: np.ndarray) -> None:
    overlay = image.copy()
    overlay[coarse] = (0.55 * overlay[coarse] + 0.45 * np.asarray([255, 80, 40])).astype(np.uint8)
    union = segments.any(axis=0) if len(segments) else np.zeros_like(coarse)
    overlay[union] = (0.45 * overlay[union] + 0.55 * np.asarray([30, 220, 90])).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write ETH3D mask preview: {temporary}")
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "exp5_ETH3D":
        raise ValueError("Unexpected ETH3D Exp5 config")
    return value


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("ETH3D local-mask generation requires CUDA")
    config = _load_config(args.config.resolve())
    inputs = read_input_manifest(args.input_manifest.resolve())
    samples = list(inputs["samples"])
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    settings = config["evaluation"]["local_points"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != MASK_FORMAT or manifest.get("settings") != settings:
            raise ValueError("Existing ETH3D mask output uses another protocol")
    else:
        manifest = {
            "format": MASK_FORMAT,
            "paper_protocol": "MoGe-3 Appendix B local point evaluation",
            "implementation_note": "ETH3D GT disparity coarse detector plus SAM2 candidate segments; masks are evaluation-only.",
            "settings": settings,
            "input_manifest_sha256": sha256(args.input_manifest),
            "expected_sample_ids": [str(sample["id"]) for sample in inputs["samples"]],
            "samples": {},
            "status": "partial",
            "created_at_unix": time.time(),
        }
    generator = _load_generator(settings["sam2"], args.device)
    for index, entry in enumerate(samples, start=1):
        sample = load_eth3d_sample(
            entry,
            output_hw=(int(config["model"]["height"]), int(config["model"]["width"])),
        )
        relative = Path("segments") / f"{sample.sample_id}.npz"
        mask_path = output / relative
        if mask_path.exists():
            segments = load_segment_masks(mask_path)
        else:
            valid = sample.valid_mask.numpy()
            disparity = np.zeros_like(sample.radial_depth.numpy(), dtype=np.float32)
            disparity[valid] = 1.0 / sample.radial_depth.numpy()[valid]
            coarse = coarse_fine_mask(
                disparity,
                valid,
                residual_scales=settings["residual_scales"],
                morphology_sizes=settings["morphology_sizes"],
                threshold_sigma=float(settings["threshold_sigma"]),
            )
            image = sample.image.permute(1, 2, 0).mul(255).round().byte().numpy()
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
            with torch.inference_mode(), autocast:
                candidates = [item["segmentation"] for item in generator.generate(image)]
            segments = select_fine_segments(
                candidates,
                coarse,
                valid,
                min_density=float(settings["min_density"]),
                min_overlap_pixels=int(settings["min_overlap_pixels"]),
                max_area_fraction=float(settings["max_area_fraction"]),
            )
            save_segment_masks(mask_path, segments)
        valid = sample.valid_mask.numpy()
        disparity = np.zeros_like(sample.radial_depth.numpy(), dtype=np.float32)
        disparity[valid] = 1.0 / sample.radial_depth.numpy()[valid]
        coarse = coarse_fine_mask(
            disparity,
            valid,
            residual_scales=settings["residual_scales"],
            morphology_sizes=settings["morphology_sizes"],
            threshold_sigma=float(settings["threshold_sigma"]),
        )
        image = sample.image.permute(1, 2, 0).mul(255).round().byte().numpy()
        preview_relative = Path("previews") / f"{sample.sample_id}.png"
        preview_path = output / preview_relative
        if not preview_path.exists():
            _preview(preview_path, image, coarse, segments)
        manifest["samples"][sample.sample_id] = {
            "path": str(relative),
            "sha256": sha256(mask_path),
            "height": int(segments.shape[1]),
            "width": int(segments.shape[2]),
            "segment_count": int(segments.shape[0]),
            "coarse_pixel_count": int(coarse.sum()),
            "selected_union_pixel_count": int(segments.any(axis=0).sum()) if len(segments) else 0,
            "preview": str(preview_relative),
            "preview_sha256": sha256(preview_path),
        }
        manifest["sample_count"] = len(manifest["samples"])
        manifest["status"] = (
            "complete"
            if set(manifest["samples"]) == set(manifest["expected_sample_ids"])
            else "partial"
        )
        manifest["updated_at_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        print(f"{index}/{len(samples)} {sample.sample_id}: segments={len(segments)}", flush=True)


if __name__ == "__main__":
    main()

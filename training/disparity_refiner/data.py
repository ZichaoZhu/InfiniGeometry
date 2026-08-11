from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


@dataclass
class HypersimDisparitySample:
    sample_id: str
    image: torch.Tensor
    target_disparity: torch.Tensor
    valid_mask: torch.Tensor
    radial_depth: torch.Tensor
    disparity_quantiles: Tuple[float, float]
    structure_mask: Optional[torch.Tensor]
    metadata: Dict[str, object]


def ensure_within(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PermissionError(f"{name} must stay under {resolved_root}: {resolved}")
    return resolved


def load_manifest(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, object]:
    import hashlib

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Manifest SHA-256 mismatch: {digest} != {expected_sha256}")
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("depth_semantics") != "Euclidean distance in meters from camera optical center":
        raise ValueError("Manifest does not declare Hypersim radial-distance depth semantics")
    return manifest


def select_training_entries(
    manifest: Mapping[str, object],
    sample_ids: Optional[Sequence[str]] = None,
) -> Sequence[Dict[str, object]]:
    train_entries = [
        dict(entry)
        for entry in manifest.get("samples", [])
        if entry.get("split") == "train"
    ]
    by_id = {str(entry["id"]): entry for entry in train_entries}
    if len(by_id) != len(train_entries):
        raise ValueError("Training manifest contains duplicate sample IDs")
    if sample_ids is None:
        return train_entries
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Configured sample IDs must be unique")
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Samples absent from training manifest: {missing}")
    return [by_id[sample_id] for sample_id in sample_ids]


def normalize_radial_disparity(
    radial_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float = 0.02,
) -> Tuple[torch.Tensor, Tuple[float, float]]:
    valid = valid_mask & torch.isfinite(radial_depth) & (radial_depth > 0)
    if int(valid.sum()) < 2:
        raise ValueError("At least two valid radial-depth pixels are required")
    raw = torch.zeros_like(radial_depth, dtype=torch.float32)
    raw[valid] = radial_depth[valid].float().reciprocal()
    low = torch.quantile(raw[valid], float(quantile))
    high = torch.quantile(raw[valid], 1.0 - float(quantile))
    if not torch.isfinite(low) or not torch.isfinite(high) or float(high - low) <= 1e-6:
        raise ValueError("Invalid disparity quantiles")
    normalized = torch.zeros_like(raw)
    normalized[valid] = (raw[valid] - low) / (high - low)
    return normalized, (float(low.item()), float(high.item()))


def build_structure_mask(
    radial_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    crop_xyxy: Sequence[int],
    selection: Mapping[str, object],
) -> torch.Tensor:
    x0, y0, x1, y1 = (int(value) for value in crop_xyxy)
    height, width = radial_depth.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid structure crop: {crop_xyxy}")
    crop_valid = valid_mask[y0:y1, x0:x1]
    crop_depth = radial_depth[y0:y1, x0:x1]
    if not bool(crop_valid.any()):
        raise ValueError("Locked structure crop contains no valid depth pixels")
    mask_type = str(selection.get("type"))
    if mask_type != "near_quantile":
        raise ValueError(f"Unsupported locked structure mask: {mask_type}")
    quantile = float(selection.get("quantile", 0.5))
    threshold = torch.quantile(crop_depth[crop_valid], quantile)
    mask = torch.zeros_like(valid_mask)
    mask[y0:y1, x0:x1] = crop_valid & (crop_depth <= threshold)
    if int(mask.sum()) < 16:
        raise ValueError("Locked structure mask contains fewer than 16 pixels")
    return mask


def _read_source_path(
    entry: Mapping[str, object],
    kind: str,
    source_root: Path,
) -> Path:
    value = entry.get(kind)
    if not isinstance(value, Mapping) or "source" not in value:
        raise ValueError(f"Sample {entry.get('id')} has no {kind}.source")
    path = Path(str(value["source"]))
    if path.is_symlink():
        raise PermissionError(f"Refusing symlinked shared input: {path}")
    path = ensure_within(path, source_root, name=f"{kind} source")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(value["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Source byte count mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != str(value["sha256"]):
        raise ValueError(
            f"Source SHA-256 mismatch for {path}: {actual_sha256} != {value['sha256']}"
        )
    return path


def load_hypersim_disparity_sample(
    entry: Mapping[str, object],
    *,
    source_root: Path,
    height: int,
    width: int,
    structure_selection: Optional[Mapping[str, object]] = None,
) -> HypersimDisparitySample:
    rgb_path = _read_source_path(entry, "rgb", source_root)
    depth_path = _read_source_path(entry, "depth", source_root)
    with Image.open(rgb_path) as image_file:
        image_file = image_file.convert("RGB").resize(
            (int(width), int(height)), Image.Resampling.LANCZOS
        )
        image = np.asarray(image_file, dtype=np.float32).copy() / 255.0
    with h5py.File(depth_path, "r") as depth_file:
        radial_depth = depth_file["dataset"][:].astype(np.float32)
    radial_depth = cv2.resize(
        radial_depth,
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
    radial = torch.from_numpy(radial_depth).float().contiguous()
    valid = torch.isfinite(radial) & (radial > 0)
    target, quantiles = normalize_radial_disparity(radial, valid)
    structure_mask = None
    if structure_selection is not None:
        structure_mask = build_structure_mask(
            radial,
            valid,
            structure_selection["crop_xyxy"],
            structure_selection["display_mask"],
        )
    return HypersimDisparitySample(
        sample_id=str(entry["id"]),
        image=torch.from_numpy(image).permute(2, 0, 1).contiguous(),
        target_disparity=target,
        valid_mask=valid,
        radial_depth=radial,
        disparity_quantiles=quantiles,
        structure_mask=structure_mask,
        metadata=dict(entry),
    )


def preload_samples(
    entries: Iterable[Mapping[str, object]],
    *,
    source_root: Path,
    height: int,
    width: int,
    structure_selections: Mapping[str, Mapping[str, object]],
) -> Sequence[HypersimDisparitySample]:
    return [
        load_hypersim_disparity_sample(
            entry,
            source_root=source_root,
            height=height,
            width=width,
            structure_selection=structure_selections.get(str(entry["id"])),
        )
        for entry in entries
    ]

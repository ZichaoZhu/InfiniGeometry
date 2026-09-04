"""Attach existing Exp5 report values to versioned viewer manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def eth_metrics(values: dict) -> dict:
    return {
        "dataset": "ETH3D",
        "scopeLabel": "held-out · excluding virtual LiDAR prompt pixels",
        "evaluationPointCount": int(values["evaluation_pixel_count"]),
        "metricDisparityMae1PerM": float(values["metric_disparity_mae_1_per_m"]),
        "radialDepthAbsRel": float(values["radial_depth_abs_rel"]),
        "radialDepthRmseM": float(values["radial_depth_rmse_m"]),
        "pointDelta001": float(values["point_delta_0_01"]),
        "localPointRel": float(values["local_point_rel"]),
        "localPointDelta001": float(values["local_point_delta_0_01"]),
        "localSegmentCount": int(values["local_segment_count"]),
    }


def waymo_metrics(record: dict) -> dict:
    return {
        "dataset": "Waymo",
        "scopeLabel": "held-out TOP LiDAR · sparse evaluation points",
        "evaluationPointCount": int(record["evaluation_point_count"]),
        "metricDisparityMae1PerM": float(record["metric_disparity_mae_1_per_m"]),
        "radialDepthAbsRel": float(record["radial_depth_abs_rel"]),
        "radialDepthRmseM": float(record["radial_depth_rmse_m"]),
        "pointDelta001": float(record["point_delta_0_01"]),
    }


def attach_eth_per_k(manifest: dict, report: dict) -> None:
    for sample in manifest["samples"]:
        record = report["per_image"].get(sample["id"])
        if record is None:
            raise KeyError(f"ETH3D report missing {sample['id']}")
        initial = eth_metrics(record["held_out"]["k0"])
        sample["stages"]["initial"]["k0"]["displayMetrics"] = initial
        for key in ("k1", "k3", "k5"):
            values = record["held_out"][key]
            sample["stages"]["best_step_22500"][key]["displayMetrics"] = eth_metrics(values)


def waymo_source(sample_id: str, report: dict) -> dict | None:
    match = re.search(r"_(\d+)_(FRONT|SIDE_LEFT|SIDE_RIGHT)$", sample_id)
    if not match:
        raise ValueError(f"Unexpected Waymo sample id: {sample_id}")
    timestamp, camera = int(match.group(1)), match.group(2)
    for value in report["per_image"].values():
        metadata = value.get("metadata", {})
        if int(metadata.get("timestamp_micros", -1)) == timestamp and metadata.get("camera") == camera:
            return value
    return None


def attach_waymo(manifest: dict, report: dict) -> None:
    for sample in manifest["samples"]:
        record = waymo_source(sample["id"], report)
        if record is None:
            # SIDE assets were exported for qualitative viewing only; no SIDE formal report exists.
            continue
        values = record["k0"]
        initial = waymo_metrics(values)
        sample["stages"]["initial"]["k0"]["displayMetrics"] = initial
        for key in ("k1", "k3", "k5"):
            sample["stages"]["best_step_22500"][key]["displayMetrics"] = waymo_metrics(record[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth-report", type=Path, required=True)
    parser.add_argument("--waymo-report", type=Path, required=True)
    parser.add_argument("--viewer-root", type=Path, required=True)
    args = parser.parse_args()
    eth_report = load(args.eth_report)
    waymo_report = load(args.waymo_report)
    eth_manifest = load(args.viewer_root / "public/data/exp5_eth3d_selected_20260903_r3/manifest.json")
    waymo_manifest = load(args.viewer_root / "public/data/exp5_waymo/manifest.json")
    attach_eth_per_k(eth_manifest, eth_report)
    attach_waymo(waymo_manifest, waymo_report)
    eth_manifest["metricsNote"] = "展示 held-out 指标；ETH3D 同时显示 MoGe-3 Local Point 指标。指标不用于 checkpoint 选择。"
    waymo_manifest["metricsNote"] = "展示 FRONT 五张样例的 held-out TOP LiDAR 指标；SIDE 五张仅有定性可视化，暂无对应正式逐图报告。指标不用于 checkpoint 选择。"
    eth_dir = args.viewer_root / "public/data/exp5_eth3d_metrics_20260904_r1"
    waymo_dir = args.viewer_root / "public/data/exp5_waymo_metrics_20260904_r1"
    write(eth_dir, eth_manifest)
    write(waymo_dir, waymo_manifest)
    provenance = {
        "format": "infinidepth-exp5-display-metrics-v1",
        "source_reports": {
            "eth3d": {"path": str(args.eth_report), "sha256": sha256(args.eth_report)},
            "waymo": {"path": str(args.waymo_report), "sha256": sha256(args.waymo_report)},
        },
        "eth3d_manifest": str(eth_dir / "manifest.json"),
        "waymo_manifest": str(waymo_dir / "manifest.json"),
        "note": "Metadata-only manifests reference pre-existing PLY/RGB assets; SIDE Waymo has no formal report and remains unscored.",
    }
    for directory in (eth_dir, waymo_dir):
        (directory / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate compact, reproducible InfiniDepth experiment records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


EXPERIMENT_NAME = re.compile(r"^exp([1-9][0-9]*)_[a-z0-9]+(?:_[a-z0-9]+)*$")
ALLOWED_STATUS = {"planned", "running", "completed", "failed"}
REQUIRED_FILES = {
    "README.md",
    "run.sh",
    "config.json",
    "provenance.json",
    "metrics/report.json",
    "artifacts/manifest.json",
    ".gitignore",
}
LARGE_SUFFIXES = {".pt", ".pth", ".ckpt", ".ply", ".log"}
MAX_TRACKED_BYTES = 10 * 1024 * 1024


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files(root: Path) -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", str(root)],
        cwd=root.parent,
        check=True,
        capture_output=True,
    )
    return {
        (root.parent / item.decode()).resolve()
        for item in result.stdout.split(b"\0")
        if item
    }


def validate_experiment(path: Path, tracked: set[Path]) -> list[str]:
    errors = []
    match = EXPERIMENT_NAME.fullmatch(path.name)
    if match is None:
        errors.append(f"invalid experiment directory name: {path.name}")
        return errors
    missing = sorted(name for name in REQUIRED_FILES if not (path / name).is_file())
    if missing:
        errors.append(f"{path.name}: missing required files: {', '.join(missing)}")
        return errors

    config = load_json(path / "config.json")
    provenance = load_json(path / "provenance.json")
    report = load_json(path / "metrics/report.json")
    manifest = load_json(path / "artifacts/manifest.json")
    expected_id = f"exp{match.group(1)}"
    for label, document in (("config", config), ("provenance", provenance), ("report", report)):
        if document.get("experiment_id") != expected_id:
            errors.append(f"{path.name}: {label} experiment_id must be {expected_id}")
    statuses = [provenance.get("status"), report.get("status")]
    if any(status not in ALLOWED_STATUS for status in statuses):
        errors.append(f"{path.name}: invalid status {statuses}")
    if len(set(statuses)) != 1:
        errors.append(f"{path.name}: provenance/report status mismatch {statuses}")

    entries = manifest.get("assets")
    if not isinstance(entries, list):
        errors.append(f"{path.name}: manifest assets must be a list")
        entries = []
    hashes = {}
    registered = set()
    for entry in entries:
        relative = entry.get("path")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"{path.name}: invalid manifest path {relative!r}")
            continue
        registered.add(relative)
        asset = path / relative
        if not asset.exists():
            if report.get("status") == "completed":
                errors.append(f"{path.name}: completed asset is missing: {relative}")
            continue
        actual_size = asset.stat().st_size
        actual_hash = sha256(asset)
        if entry.get("bytes") != actual_size or entry.get("sha256") != actual_hash:
            errors.append(f"{path.name}: stale manifest metadata for {relative}")
        previous = hashes.get(actual_hash)
        if previous is not None and entry.get("alias_of") != previous:
            errors.append(f"{path.name}: duplicate asset content: {previous} and {relative}")
        else:
            hashes[actual_hash] = relative

    for asset in (path / "artifacts").rglob("*"):
        if asset.is_file() and asset.name != "manifest.json":
            relative = asset.relative_to(path).as_posix()
            if relative not in registered:
                errors.append(f"{path.name}: unregistered artifact: {relative}")
    for file_path in path.rglob("*"):
        if file_path.is_dir() and not any(file_path.iterdir()) and report.get("status") == "completed":
            errors.append(f"{path.name}: empty directory after completion: {file_path.relative_to(path)}")
        if file_path.is_file() and file_path.resolve() in tracked:
            if file_path.suffix in LARGE_SUFFIXES:
                errors.append(f"{path.name}: large runtime type tracked by Git: {file_path.name}")
            if file_path.stat().st_size > MAX_TRACKED_BYTES:
                errors.append(f"{path.name}: tracked file exceeds 10 MiB: {file_path.name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = args.paths or sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("exp")
    )
    tracked = tracked_files(root)
    errors = []
    numbers = []
    for path in paths:
        path = path.resolve()
        match = EXPERIMENT_NAME.fullmatch(path.name)
        if match:
            numbers.append(int(match.group(1)))
        errors.extend(validate_experiment(path, tracked))
    if numbers and sorted(numbers) != list(range(1, max(numbers) + 1)):
        errors.append(f"experiment numbering is not contiguous: {sorted(numbers)}")
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"validated {len(paths)} experiment(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

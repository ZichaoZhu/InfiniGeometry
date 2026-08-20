from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time


SOURCE_ROOT = Path("/nas1/datasets/hypersim/raw")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_list(config: dict[str, object]) -> list[str]:
    evaluation = config["evaluation"]
    data = config["data"]
    assert isinstance(evaluation, dict) and isinstance(data, dict)
    validation = set(str(value) for value in evaluation["full_sample_ids"])
    index = SOURCE_ROOT / "ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
    paths: list[str] = []
    train_count = validation_count = 0
    with index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene, camera, frame = (
                row["scene_name"],
                row["camera_name"],
                int(row["frame_id"]),
            )
            sample_id = f"{scene}_{camera}_frame.{frame:04d}"
            split = row["split_partition_name"]
            if split == "train":
                train_count += 1
            elif split == "val" and sample_id in validation:
                validation_count += 1
            else:
                continue
            paths.extend(
                (
                    f"{scene}/images/scene_{camera}_final_preview/frame.{frame:04d}.tonemap.jpg",
                    f"{scene}/images/scene_{camera}_geometry_hdf5/frame.{frame:04d}.depth_meters.hdf5",
                )
            )
    expected = int(data["expected_train_count"]) + len(data.get("excluded_sample_ids", []))
    if train_count != expected or validation_count != len(validation):
        raise ValueError(
            f"Expected {expected} train and {len(validation)} validation samples, "
            f"got {train_count} and {validation_count}"
        )
    if len(paths) != 2 * (train_count + validation_count) or len(set(paths)) != len(paths):
        raise ValueError("Exp3 cache file list is incomplete or contains duplicates")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm the Exp3 Hypersim cache")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    experiment = args.experiment.resolve()
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    data = config["data"]
    server = config["server"]
    source = Path(str(data["source_root"])).resolve()
    cache = Path(str(data["local_cache"])).resolve()
    cache_parent = Path(str(server["cache"])).resolve()
    if source != SOURCE_ROOT or cache_parent not in cache.parents:
        raise PermissionError("Exp3 cache paths are outside the configured safe roots")
    paths = file_list(config)
    metadata = cache / ".exp3_cache"
    status = metadata / "status.json"
    chunk_size = (len(paths) + args.workers - 1) // args.workers
    shards = [
        paths[index * chunk_size : (index + 1) * chunk_size]
        for index in range(args.workers)
    ]
    metadata.mkdir(parents=True, exist_ok=True)
    for index, values in enumerate(shards):
        (metadata / f"files-{index}.txt").write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )
    plan = {
        "state": "planned" if args.plan_only else "copying",
        "source": str(source),
        "cache": str(cache),
        "file_count": len(paths),
        "sample_count": len(paths) // 2,
        "workers": args.workers,
        "updated_at_unix": time.time(),
    }
    atomic_json(status, plan)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False))
        return
    processes = []
    for index in range(args.workers):
        log = (metadata / f"rsync-{index}.log").open("a", encoding="utf-8")
        command = [
            "rsync",
            "-aR",
            "--partial",
            f"--files-from={metadata / f'files-{index}.txt'}",
            f"{source}/",
            f"{cache}/",
        ]
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    returncodes = []
    for process, log in processes:
        returncodes.append(process.wait())
        log.close()
    if any(returncodes):
        atomic_json(status, {**plan, "state": "failed", "returncodes": returncodes})
        raise SystemExit(f"rsync failed: {returncodes}")
    missing = [path for path in paths if not (cache / path).is_file()]
    if missing:
        atomic_json(
            status,
            {**plan, "state": "failed", "missing_count": len(missing), "missing": missing[:20]},
        )
        raise FileNotFoundError(f"Cache is missing {len(missing)} files")
    total_bytes = sum((cache / path).stat().st_size for path in paths)
    completed = {
        **plan,
        "state": "completed",
        "total_bytes": total_bytes,
        "completed_at_unix": time.time(),
    }
    atomic_json(status, completed)
    print(json.dumps(completed, ensure_ascii=False))


if __name__ == "__main__":
    main()

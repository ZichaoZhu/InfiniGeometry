from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Mapping


ALLOWED_STATUS = {"planned", "running", "completed", "failed"}
AUDITED_BASE_COMMIT = "36c6e0c31887fafc210184ee43ca475230704095"
EXPERIMENT_NAME = re.compile(r"^exp([1-9][0-9]*)_[a-z0-9]+(?:_[a-z0-9]+)*$")
REQUIRED_FILES = (
    "README.md",
    "config.json",
    "provenance.json",
    "metrics/report.json",
    "artifacts/manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 InfiniDepth 实验目录")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-git", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_directories(root: Path) -> List[Path]:
    directories = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("exp")]
    invalid = [path.name for path in directories if EXPERIMENT_NAME.fullmatch(path.name) is None]
    if invalid:
        raise ValueError(f"实验目录命名无效: {invalid}")
    return sorted(directories, key=lambda path: int(EXPERIMENT_NAME.fullmatch(path.name).group(1)))


def index_statuses(root: Path) -> Dict[int, str]:
    result = {}
    for line in (root / "README.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\| exp([0-9]+) \| ([a-z]+) \|", line)
        if match:
            result[int(match.group(1))] = match.group(2)
    return result


def check_path_boundaries(config: Mapping[str, object]) -> None:
    server = config["server"]
    safe_root = Path(str(server["safe_root"]))
    if safe_root != Path("/mnt/data/home/zhuzichao"):
        raise ValueError("server.safe_root 必须是个人服务器根目录")
    for key in ("project_root", "environment", "cache", "temporary"):
        path = Path(str(server[key]))
        if path != safe_root and safe_root not in path.parents:
            raise ValueError(f"server.{key} 越过个人目录: {path}")
    checkpoint = Path(str(config["model"]["checkpoint"]))
    if checkpoint != safe_root and safe_root not in checkpoint.parents:
        raise ValueError(f"model.checkpoint 越过个人目录: {checkpoint}")
    if Path(str(config["data"]["source_root"])) != Path("/nas1/datasets/hypersim/raw"):
        raise ValueError("共享 Hypersim 源路径发生变化")


def check_assets(experiment: Path, manifest: Mapping[str, object], status: str) -> None:
    entries = list(manifest.get("assets", []))
    seen_paths = set()
    seen_sha: Dict[str, str] = {}
    for entry in entries:
        required = {
            "path",
            "bytes",
            "sha256",
            "purpose",
            "generated_by",
            "tracked_by_git",
        }
        missing = required - set(entry)
        if missing:
            raise ValueError(f"资产条目缺少字段 {missing}: {entry}")
        relative = str(entry["path"])
        if relative in seen_paths:
            raise ValueError(f"资产路径重复: {relative}")
        seen_paths.add(relative)
        candidate = experiment.parent / relative
        if candidate.is_symlink():
            raise ValueError(f"资产不得使用符号链接: {candidate}")
        path = candidate.resolve()
        experiment_number = EXPERIMENT_NAME.fullmatch(experiment.name).group(1)
        viewer_root = (
            experiment.parent / "viewer" / "public" / "data" / f"exp{experiment_number}"
        ).resolve()
        allowed_roots = (experiment.resolve(), viewer_root)
        if not any(path == root or root in path.parents for root in allowed_roots):
            raise ValueError(f"资产路径越界或属于其他实验: {path}")
        if not path.is_file():
            raise ValueError(f"登记资产不存在: {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"资产字节数不匹配: {path}")
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"资产 SHA-256 不匹配: {path}")
        if digest in seen_sha and entry.get("alias_of") != seen_sha[digest]:
            raise ValueError(f"重复 SHA 未登记别名: {relative}")
        seen_sha.setdefault(digest, relative)
    if status == "completed":
        required_figures = {
            f"{experiment.name}/artifacts/training_curve.png",
            f"{experiment.name}/artifacts/disparity_comparison.png",
        }
        if not required_figures.issubset(seen_paths):
            raise ValueError("结束状态必须登记两张最终汇总图")
    checkpoint_names = {
        path.name for path in experiment.glob("runs/*/checkpoints/*.pt")
    }
    if checkpoint_names - {"stage1_best.pt", "joint_best.pt", "last.pt"}:
        raise ValueError(f"存在冗余 checkpoint: {sorted(checkpoint_names)}")
    figure_names = {path.name for path in (experiment / "artifacts").glob("*.png")}
    if figure_names - {"training_curve.png", "disparity_comparison.png"}:
        raise ValueError(f"存在冗余最终图: {sorted(figure_names)}")
    asset_suffixes = {
        ".pt", ".pth", ".ckpt", ".ply", ".png", ".jpg", ".jpeg",
        ".log", ".npy", ".npz", ".gif", ".mp4", ".mov", ".pdf",
        ".csv", ".hdf5",
    }
    for path in experiment.rglob("*"):
        if path.is_file() and path.suffix.lower() in asset_suffixes:
            relative = str(path.relative_to(experiment.parent))
            if relative not in seen_paths:
                raise ValueError(f"实验资产未登记: {relative}")


def check_empty_directories(experiment: Path) -> None:
    for path in experiment.rglob("*"):
        if path.is_dir() and not any(path.iterdir()):
            raise ValueError(f"存在空目录: {path}")


def check_git(root: Path) -> None:
    project_root = root.parent
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", "-z", AUDITED_BASE_COMMIT],
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    changed_paths = set(changed.stdout.split(b"\0"))
    for raw in tracked.stdout.split(b"\0"):
        if not raw:
            continue
        if raw not in changed_paths:
            continue
        path = project_root / raw.decode("utf-8")
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Git 误收大文件: {path}")
        if path.suffix.lower() in {".pt", ".ckpt", ".ply", ".log"}:
            raise ValueError(f"Git 误收服务器资产: {path}")


def validate(root: Path, check_git_files: bool) -> None:
    root = root.resolve()
    experiments = experiment_directories(root)
    numbers = [int(EXPERIMENT_NAME.fullmatch(path.name).group(1)) for path in experiments]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"实验编号不连续: {numbers}")
    indexed = index_statuses(root)
    if set(indexed) != set(numbers):
        raise ValueError(f"全局索引与实验目录不一致: index={indexed}, dirs={numbers}")
    for experiment, number in zip(experiments, numbers):
        for relative in REQUIRED_FILES:
            if not (experiment / relative).is_file():
                raise ValueError(f"缺少必需文件: {experiment / relative}")
        config = load_json(experiment / "config.json")
        provenance = load_json(experiment / "provenance.json")
        report = load_json(experiment / "metrics" / "report.json")
        assets = load_json(experiment / "artifacts" / "manifest.json")
        expected_id = experiment.name
        for name, value in (
            ("config", config.get("experiment_id")),
            ("report", report.get("experiment_id")),
            ("assets", assets.get("experiment_id")),
        ):
            if value != expected_id:
                raise ValueError(f"{name} 的 experiment_id 不一致: {value}")
        statuses = {
            str(config.get("status")),
            str(provenance.get("status")),
            str(report.get("status")),
            str(assets.get("status")),
            str(indexed[number]),
        }
        if len(statuses) != 1 or next(iter(statuses)) not in ALLOWED_STATUS:
            raise ValueError(f"实验状态不一致: {statuses}")
        status = next(iter(statuses))
        required_config = {"research_question", "claim_boundary", "seed", "data", "model", "training", "evaluation", "server"}
        if required_config - set(config):
            raise ValueError(f"config 缺少字段: {required_config - set(config)}")
        check_path_boundaries(config)
        manifest_path = (root.parent / str(config["data"]["manifest"])).resolve()
        project_root = root.parent.resolve()
        if manifest_path != project_root and project_root not in manifest_path.parents:
            raise ValueError(f"数据 manifest 路径越界: {manifest_path}")
        if sha256(manifest_path) != config["data"]["manifest_sha256"]:
            raise ValueError("数据 manifest SHA-256 不匹配")
        runs = list(config.get("runs", []))
        if runs:
            run_ids = [str(run["id"]) for run in runs]
            if len(run_ids) != len(set(run_ids)):
                raise ValueError("运行 ID 重复")
            for run in runs:
                directory = experiment / str(run["directory"])
                if not directory.is_dir():
                    raise ValueError(f"缺少运行目录: {directory}")
        check_assets(experiment, assets, status)
        check_empty_directories(experiment)
    if check_git_files:
        check_git(root)


def main() -> None:
    args = parse_args()
    validate(args.root, args.check_git)
    print("实验目录验证通过")


if __name__ == "__main__":
    main()

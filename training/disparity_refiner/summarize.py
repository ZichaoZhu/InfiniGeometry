from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Mapping, MutableMapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 InfiniDepth disparity Refiner 实验")
    parser.add_argument("--experiment", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> MutableMapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def current_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def best_stage(run_report: Mapping[str, object]) -> Mapping[str, object]:
    stages = list(run_report["stages"])
    if {str(stage["stage"]) for stage in stages} != {"stage1", "joint"}:
        raise ValueError("Run report must contain completed stage1 and joint stages")
    return min(stages, key=lambda stage: float(stage["best_score"]))


def score_for(evaluation: Mapping[str, object], iteration: str) -> float:
    values = evaluation["aggregate"][iteration]
    return float(values.get("composite_score", values["full_mae"]))


def collect_assets(experiment: Path, experiment_tag: str) -> List[Dict[str, object]]:
    assets: List[Dict[str, object]] = []
    seen: Dict[str, str] = {}
    config = load_json(experiment / "config.json")
    run_roots = {
        (experiment / str(run["directory"])).resolve(): str(run["id"])
        for run in config["runs"]
    }
    patterns = (
        "runs/*/checkpoints/*.pt",
        "runs/**/*.log",
        "artifacts/*.png",
        f"../viewer/public/data/{experiment_tag}/**/*",
    )
    for pattern in patterns:
        for path in sorted(experiment.glob(pattern)):
            if not path.is_file():
                continue
            path = path.resolve()
            digest = sha256(path)
            relative = str(path.relative_to(experiment.parent.resolve()))
            run_id = next(
                (
                    identifier
                    for run_root, identifier in run_roots.items()
                    if path == run_root or run_root in path.parents
                ),
                None,
            )
            if path.suffix == ".pt" and run_id is not None:
                purpose = "保留的训练 checkpoint"
                run_argument = "" if run_id == "main" else f" {run_id}"
                generated_by = (
                    f"bash experiment/{experiment.name}/run.sh{run_argument}"
                )
            elif path.suffix == ".log" and run_id is not None:
                purpose = "训练日志"
                run_argument = "" if run_id == "main" else f" {run_id}"
                generated_by = (
                    f"bash experiment/{experiment.name}/run.sh{run_argument}"
                )
            elif path.parent == (experiment / "artifacts").resolve():
                purpose = "最终训练曲线或 disparity 定性对比图"
                generated_by = (
                    "python -m training.disparity_refiner.export_assets "
                    f"--experiment experiment/{experiment.name}"
                )
            else:
                purpose = "三窗口查看器的服务器端数据"
                generated_by = (
                    "python -m training.disparity_refiner.export_assets "
                    f"--experiment experiment/{experiment.name}"
                )
            entry: Dict[str, object] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "purpose": purpose,
                "generated_by": generated_by,
                "tracked_by_git": path.parent == experiment / "artifacts" and path.suffix == ".png",
            }
            if digest in seen:
                entry["alias_of"] = seen[digest]
            else:
                seen[digest] = relative
            assets.append(entry)
    return assets


def update_index(
    project_root: Path,
    experiment_number: int,
    experiment_name: str,
    question: str,
    status: str,
    conclusion: str,
    commit: str,
) -> None:
    index = project_root / "experiment" / "README.md"
    lines = index.read_text(encoding="utf-8").splitlines()
    replacement = (
        f"| exp{experiment_number} | {status} | {question} "
        f"| `{commit[:12]}` | K3 全图 MAE + 锁定细结构 MAE | {conclusion} "
        f"| [exp{experiment_number}]({experiment_name}/) |"
    )
    matches = [
        index for index, line in enumerate(lines)
        if line.startswith(f"| exp{experiment_number} |")
    ]
    if len(matches) != 1:
        raise ValueError(f"Global experiment index must contain exactly one exp{experiment_number} row")
    lines[matches[0]] = replacement
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    project_root = Path(__file__).resolve().parents[2]
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config_path = experiment / "config.json"
    config = load_json(config_path)
    experiment_number = int(str(config["experiment_id"]).split("_", 1)[0][3:])
    experiment_tag = f"exp{experiment_number}"
    threshold = float(config["evaluation"]["gate_improvement_fraction"])
    minimum_successes = int(config["evaluation"]["gate_minimum_successes"])
    run_results = []
    failed_runs = []
    run_provenance: Dict[str, Dict[str, object]] = {}
    for run in config["runs"]:
        run_root = experiment / str(run["directory"])
        provenance_path = run_root / "provenance.json"
        if provenance_path.is_file():
            source = load_json(provenance_path)
            run_provenance[str(run["id"])] = {
                "code_commit": source.get("code_commit"),
                "code_dirty": source.get("code_dirty"),
                "config_sha256": source.get("config_sha256"),
                "sample_ids": source.get("sample_ids"),
                "seed": source.get("seed"),
            }
        report_path = run_root / "metrics" / "report.json"
        if not report_path.is_file():
            continue
        report = load_json(report_path)
        if report.get("status") == "failed":
            failed_runs.append(
                {
                    "run_id": run["id"],
                    "sample_ids": run["sample_ids"],
                    "failure_type": report.get("failure_type"),
                    "failure_message": report.get("failure_message"),
                }
            )
            continue
        if report.get("status") != "completed":
            continue
        selected = best_stage(report)
        evaluation = selected["best_evaluation"]
        k0 = score_for(evaluation, "k0")
        k3 = score_for(evaluation, "k3")
        improvement = (k0 - k3) / max(abs(k0), 1e-12)
        improved_images = int(evaluation["k3_better_than_k0_count"])
        minimum_improved_images = int(config["evaluation"].get("minimum_improved_images", 0))
        sample_ids = run.get("sample_ids")
        run_results.append(
            {
                "run_id": run["id"],
                "sample_id": sample_ids[0] if sample_ids else None,
                "selected_stage": selected["stage"],
                "selected_step": selected["best_step"],
                "k0_composite_score": k0,
                "k3_composite_score": k3,
                "relative_improvement": improvement,
                "k3_better_than_k0_count": improved_images,
                "passed": (
                    improvement >= threshold
                    and improved_images >= minimum_improved_images
                ),
            }
        )

    expected = len(config["runs"])
    successes = sum(bool(result["passed"]) for result in run_results)
    all_runs_complete = len(run_results) == expected
    gate_passed = all_runs_complete and successes >= minimum_successes
    final_figures = [
        experiment / "artifacts" / "training_curve.png",
        experiment / "artifacts" / "disparity_comparison.png",
    ]
    figures_complete = all(path.is_file() for path in final_figures)
    if failed_runs:
        status = "failed"
        conclusion = f"{len(failed_runs)} 个运行异常终止，停止后续实验"
    elif not all_runs_complete or not figures_complete:
        status = "running"
        conclusion = f"已完成 {len(run_results)}/{expected} 个运行，等待完整训练或资产导出"
    elif gate_passed and expected == 1:
        status = "completed"
        selected = run_results[0]
        conclusion = (
            f"百图训练正常完成，平均综合分数改善 {100 * float(selected['relative_improvement']):.2f}%，"
            f"{selected['k3_better_than_k0_count']}/100 张改善"
        )
    elif gate_passed:
        status = "completed"
        conclusion = f"{successes}/{expected} 个单图运行通过 1% 门槛，可登记 Exp2"
    elif expected == 1:
        status = "failed"
        selected = run_results[0]
        conclusion = (
            f"百图训练正常完成，但验收未通过：平均综合分数改善 "
            f"{100 * float(selected['relative_improvement']):.2f}%，"
            f"{selected['k3_better_than_k0_count']}/100 张改善"
        )
    else:
        status = "failed"
        conclusion = f"仅 {successes}/{expected} 个单图运行通过 1% 门槛，停止 Exp2"

    report = {
        "acceptance": {
            "minimum_successful_runs": minimum_successes,
            "required_completed_runs": expected,
            "required_k3_relative_improvement": threshold,
        },
        "completed_run_count": len(run_results),
        "experiment_id": config["experiment_id"],
        "failed_runs": failed_runs,
        "gate_passed": gate_passed if all_runs_complete else None,
        "result": conclusion,
        "runs": run_results,
        "status": status,
        "successful_run_count": successes,
    }
    write_json(experiment / "metrics" / "report.json", report)
    config["status"] = status
    write_json(config_path, config)
    provenance = load_json(experiment / "provenance.json")
    commit = current_commit(project_root)
    provenance.pop("run_commit", None)
    provenance["run_commits"] = sorted({
        str(value["code_commit"])
        for value in run_provenance.values()
        if value.get("code_commit")
    })
    provenance["runs"] = run_provenance
    provenance["summary_commit"] = commit
    provenance["status"] = status
    write_json(experiment / "provenance.json", provenance)
    assets = collect_assets(experiment, experiment_tag)
    write_json(
        experiment / "artifacts" / "manifest.json",
        {
            "assets": assets,
            "experiment_id": config["experiment_id"],
            "status": status,
            "version": 1,
        },
    )
    update_index(
        project_root,
        experiment_number,
        experiment.name,
        str(config["research_question"]),
        status,
        conclusion,
        commit,
    )


if __name__ == "__main__":
    main()

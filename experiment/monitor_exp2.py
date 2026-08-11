from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Optional, Sequence


EXP2_NAME = "exp2_infinidepth_disparity_ssr_hypersim100_overfit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监控 Exp2 训练状态")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--stale-seconds", type=int, default=7200)
    parser.add_argument("--daemon", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_is_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        fields = stat.read_text(encoding="utf-8").split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    return True


def process_identity_error(pid: int, experiment: Path) -> Optional[str]:
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        cwd = (proc / "cwd").resolve(strict=True)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return f"无法验证训练进程身份: {type(exc).__name__}"
    project_root = experiment.parent.parent
    required = (
        "training.disparity_refiner.train",
        str(experiment / "config.json"),
        str(experiment / "runs" / "main"),
    )
    if cwd != project_root:
        return f"训练进程工作目录不匹配: {cwd}"
    missing = [value for value in required if value not in command]
    if missing:
        return f"训练进程命令不属于 Exp2: missing={missing}"
    return None


def read_history(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def all_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(all_finite(item) for item in value)
    return True


def score(record: Mapping[str, object], key: str) -> float:
    values = record["evaluation"]["aggregate"][key]
    return float(values.get("composite_score", values["full_mae"]))


def status_signature(status: Mapping[str, object]) -> tuple[object, ...]:
    return (
        status.get("health"),
        status.get("stage"),
        status.get("stage_step"),
        status.get("total_step"),
        status.get("report_status"),
    )


def check_once(
    experiment: Path,
    training_pid: int,
    stale_seconds: int,
    *,
    now: Optional[float] = None,
) -> Mapping[str, object]:
    checked_at = time.time() if now is None else now
    run = experiment / "runs" / "main"
    monitor = run / "monitor"
    history_path = run / "metrics" / "history.jsonl"
    report_path = run / "metrics" / "report.json"
    previous_path = monitor / "status.json"
    previous = load_json(previous_path) if previous_path.is_file() else {}
    config = load_json(experiment / "config.json")
    records = read_history(history_path)
    latest = records[-1] if records else None
    report = load_json(report_path) if report_path.is_file() else {}
    report_status = str(report.get("status", "running"))
    alive = process_is_alive(training_pid)
    identity_error = process_identity_error(training_pid, experiment) if alive else None
    alerts: list[tuple[str, str]] = []

    if report_status == "failed":
        alerts.append(("training_failed", str(report.get("failure_message", "训练失败"))))
    elif report_status != "completed":
        if not alive:
            alerts.append(("training_exited", "Exp2 训练进程已经退出"))
        elif identity_error:
            alerts.append(("process_identity_mismatch", identity_error))

    if latest is not None:
        checked_values = {
            "evaluation": latest.get("evaluation"),
            "gradient_norms": latest.get("gradient_norms"),
            "training": latest.get("training"),
        }
        if not all_finite(checked_values):
            alerts.append(("non_finite_metrics", "最近一次训练或评估记录包含 NaN/Inf"))
        for key, value in latest.get("training", {}).items():
            if "bounded_residual_" in key and key.endswith(("_min", "_max")):
                if abs(float(value)) > 0.1000001:
                    alerts.append(("residual_out_of_bounds", f"{key}={value}"))
                    break

    progress_anchor = history_path if history_path.is_file() else run / "provenance.json"
    last_eval_at = (
        progress_anchor.stat().st_mtime
        if progress_anchor.is_file()
        else float(previous.get("last_eval_at_unix", checked_at))
    )
    if alive and checked_at - last_eval_at > stale_seconds:
        alerts.append(("evaluation_stale", f"{int(checked_at - last_eval_at)} 秒没有新评估"))

    full = [record for record in records if record.get("scope") == "full"]
    if len(full) >= 2 and all(score(record, "k3") >= score(record, "k0") for record in full[-2:]):
        alerts.append(("ssr_no_gain_two_full_evals", "连续两次完整评估中 K3 均未优于 K0"))

    stages = config["training"]["stages"]
    total_max = sum(int(stages[name]["max_steps"]) for name in ("stage1", "joint"))
    stage = str(latest["stage"]) if latest else "stage1"
    stage_step = int(latest["stage_step"]) if latest else 0
    total_step = int(latest["total_step"]) if latest else 0
    eta_seconds = None
    if latest and stage_step > 0 and float(latest.get("elapsed_seconds", 0.0)) > 0:
        seconds_per_step = float(latest["elapsed_seconds"]) / stage_step
        eta_seconds = max(total_max - total_step, 0) * seconds_per_step

    if report_status == "completed":
        health = "completed"
    elif alerts:
        health = "alert"
    elif latest is None:
        health = "starting"
    else:
        health = "healthy"
    status = {
        "checked_at_unix": checked_at,
        "eta_seconds": eta_seconds,
        "health": health,
        "k0_score": score(latest, "k0") if latest else None,
        "k3_score": score(latest, "k3") if latest else None,
        "last_eval_at_unix": last_eval_at,
        "process_alive": alive,
        "progress_fraction": total_step / total_max,
        "report_status": report_status,
        "stage": stage,
        "stage_step": stage_step,
        "total_step": total_step,
        "training_pid": training_pid,
    }
    atomic_json(previous_path, status)
    if not previous or status_signature(previous) != status_signature(status):
        append_jsonl(monitor / "history.jsonl", status)

    event = None
    if report_status == "completed":
        event = ("training_completed", "Exp2 训练已经完成")
    elif alerts:
        event = alerts[0]
    if event is not None:
        kind, message = event
        raw_id = f"{kind}:{stage}:{total_step}:{report_status}"
        payload = {
            "created_at_unix": checked_at,
            "event_id": hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16],
            "kind": kind,
            "message": message,
            "stage": stage,
            "stage_step": stage_step,
            "total_step": total_step,
        }
        alert_path = monitor / "alert.json"
        old_alert = load_json(alert_path) if alert_path.is_file() else {}
        if old_alert.get("event_id") != payload["event_id"]:
            atomic_json(alert_path, payload)
    return status


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0 or args.stale_seconds <= 0:
        raise ValueError("监控间隔和停滞阈值必须为正数")
    project_root = Path(__file__).resolve().parents[1]
    expected = project_root / "experiment" / EXP2_NAME
    experiment = args.experiment.resolve(strict=True)
    if experiment != expected:
        raise PermissionError(f"监控器只允许操作 {expected}")
    monitor = experiment / "runs" / "main" / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    with (monitor / "monitor.pid").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Exp2 monitor 已在运行") from exc
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        if args.daemon:
            time.sleep(5)
        while True:
            status = check_once(experiment, args.training_pid, args.stale_seconds)
            if not args.daemon or status["health"] == "completed":
                break
            if status["health"] == "alert" and not status["process_alive"]:
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

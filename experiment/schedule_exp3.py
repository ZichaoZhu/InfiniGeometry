from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class GpuState:
    index: int
    uuid: str
    name: str
    memory_free_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()


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


def query_gpus() -> list[GpuState]:
    gpu_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    process_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    processes: dict[str, list[int]] = {}
    for line in process_output.splitlines():
        if not line.strip():
            continue
        uuid, pid = (part.strip() for part in line.split(",", 1))
        processes.setdefault(uuid, []).append(int(pid))
    states = []
    for line in gpu_output.splitlines():
        if not line.strip():
            continue
        index, uuid, name, free, utilization = (
            part.strip() for part in line.split(",", 4)
        )
        states.append(
            GpuState(
                index=int(index),
                uuid=uuid,
                name=name,
                memory_free_mib=int(free),
                utilization_percent=int(utilization),
                compute_pids=tuple(sorted(processes.get(uuid, []))),
            )
        )
    return sorted(states, key=lambda gpu: gpu.index)


def is_strictly_idle(
    gpu: GpuState, minimum_free_memory_mib: int, maximum_utilization_percent: int
) -> bool:
    return (
        not gpu.compute_pids
        and gpu.memory_free_mib >= minimum_free_memory_mib
        and gpu.utilization_percent <= maximum_utilization_percent
    )


def update_idle_counts(
    gpus: Sequence[GpuState],
    previous: Mapping[str, int],
    *,
    minimum_free_memory_mib: int,
    maximum_utilization_percent: int,
) -> dict[str, int]:
    return {
        gpu.uuid: previous.get(gpu.uuid, 0) + 1
        if is_strictly_idle(
            gpu, minimum_free_memory_mib, maximum_utilization_percent
        )
        else 0
        for gpu in gpus
    }


def choose_gpus(
    gpus: Sequence[GpuState],
    idle_counts: Mapping[str, int],
    *,
    required_idle_checks: int,
    minimum_gpus: int,
    maximum_gpus: int,
) -> list[GpuState]:
    ready = [gpu for gpu in gpus if idle_counts.get(gpu.uuid, 0) >= required_idle_checks]
    if len(ready) >= maximum_gpus:
        return ready[:maximum_gpus]
    if len(ready) >= minimum_gpus:
        return ready[:minimum_gpus]
    return []


def child_processes(pid: int) -> set[int]:
    descendants = {pid}
    pending = [pid]
    while pending:
        parent = pending.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in path.read_text().split()]
        except (FileNotFoundError, PermissionError, ValueError):
            children = []
        for child in children:
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def external_compute_pids(gpus: Sequence[GpuState], training_pid: int) -> list[int]:
    own = child_processes(training_pid)
    return sorted(
        {pid for gpu in gpus for pid in gpu.compute_pids if pid not in own}
    )


def build_train_command(
    *,
    world_size: int,
    config: Path,
    output: Path,
    resume: Optional[Path] = None,
    smoke: bool = False,
) -> list[str]:
    if world_size == 1:
        command = [sys.executable, "-m", "training.disparity_refiner.train"]
    else:
        torchrun = shutil.which("torchrun")
        if torchrun is None:
            raise FileNotFoundError("torchrun is not available in the active environment")
        command = [
            torchrun,
            "--standalone",
            f"--nproc_per_node={world_size}",
            "-m",
            "training.disparity_refiner.train",
        ]
    command += [
        "--config",
        str(config),
        "--run-id",
        "main",
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    if smoke:
        command.append("--smoke")
    return command


def make_smoke_config(
    config_path: Path, project_root: Path, destination: Path, sample_count: int = 8
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (project_root / "experiment/data/hypersim100_train_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    sample_ids = [str(value["id"]) for value in manifest["samples"][:sample_count]]
    validation_id = str(config["evaluation"]["full_sample_ids"][0])
    config["experiment_id"] += "_ddp_smoke"
    config["data"]["expected_train_count"] = len(sample_ids)
    config["data"]["local_cache"] = None
    config["runs"][0]["sample_ids"] = sample_ids
    config["evaluation"]["sample_ids"] = [validation_id]
    config["evaluation"]["full_sample_ids"] = [validation_id]
    for name, steps in (("stage1", 3), ("joint", 1)):
        stage = config["training"]["stages"][name]
        stage["min_steps"] = steps
        stage["max_steps"] = steps
        stage["eval_every"] = 1
        stage["full_eval_every"] = 1
        stage["checkpoint_every"] = 1
        stage["plateau_patience_evals"] = 0
    stage1 = config["training"]["stages"]["stage1"]
    stage1["dino_freeze_steps"] = 1
    stage1["dino_warmup_end"] = 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, config)
    return destination


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def optimizer_step_times(output: Path) -> list[float]:
    history = output / "metrics/history.jsonl"
    if not history.is_file():
        return []
    values = []
    for line in history.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "optimizer_step_seconds" in record:
                values.append(float(record["optimizer_step_seconds"]))
    return values


class Scheduler:
    def __init__(
        self,
        experiment: Path,
        poll_interval: Optional[int] = None,
        host: Optional[str] = None,
        world_size: Optional[int] = None,
    ) -> None:
        self.experiment = experiment.resolve()
        self.project_root = self.experiment.parents[1]
        self.config_path = self.experiment / "config.json"
        self.config = read_json(self.config_path)
        scheduler = self.config["scheduler"]
        self.poll_interval = int(poll_interval or scheduler["poll_interval_seconds"])
        self.required_idle_checks = int(scheduler["required_idle_checks"])
        self.minimum_gpus = int(scheduler["minimum_gpus"])
        self.maximum_gpus = int(scheduler["maximum_gpus"])
        if world_size is not None:
            self.minimum_gpus = self.maximum_gpus = int(world_size)
        self.minimum_free_memory_mib = int(scheduler["minimum_free_memory_mib"])
        self.maximum_utilization_percent = int(
            scheduler["maximum_utilization_percent"]
        )
        self.host = host or str(scheduler["host"])
        self.scheduler_dir = self.experiment / "scheduler"
        self.status_path = self.scheduler_dir / "status.json"
        self.history_path = self.scheduler_dir / "history.jsonl"
        self.alert_path = self.scheduler_dir / "alert.json"
        self.incident_path = self.scheduler_dir / "incident.jsonl"
        self.output = self.experiment / "runs/main"
        self.idle_counts: dict[str, int] = {}
        self.last_signature: Optional[tuple[object, ...]] = None

    def write_status(self, state: str, **values: object) -> None:
        status = {
            "state": state,
            "host": os.uname().nodename,
            "updated_at_unix": time.time(),
            **values,
        }
        atomic_json(self.status_path, status)
        signature = (
            state,
            values.get("world_size"),
            values.get("training_pid"),
            tuple(values.get("candidate_gpus", [])),
            tuple(sorted(self.idle_counts.items())),
        )
        if signature != self.last_signature:
            append_jsonl(self.history_path, status)
            self.last_signature = signature

    def alert(self, kind: str, message: str, **values: object) -> None:
        event = {
            "kind": kind,
            "message": message,
            "created_at_unix": time.time(),
            **values,
        }
        atomic_json(self.alert_path, event)
        append_jsonl(self.incident_path, event)

    def strict_now(self, selected: Sequence[GpuState]) -> bool:
        current = {gpu.uuid: gpu for gpu in query_gpus()}
        return all(
            gpu.uuid in current
            and is_strictly_idle(
                current[gpu.uuid],
                self.minimum_free_memory_mib,
                self.maximum_utilization_percent,
            )
            for gpu in selected
        )

    def wait_for_gpus(self, required_world_size: Optional[int] = None) -> list[GpuState]:
        while True:
            gpus = query_gpus()
            self.idle_counts = update_idle_counts(
                gpus,
                self.idle_counts,
                minimum_free_memory_mib=self.minimum_free_memory_mib,
                maximum_utilization_percent=self.maximum_utilization_percent,
            )
            selected = choose_gpus(
                gpus,
                self.idle_counts,
                required_idle_checks=self.required_idle_checks,
                minimum_gpus=required_world_size or self.minimum_gpus,
                maximum_gpus=required_world_size or self.maximum_gpus,
            )
            self.write_status(
                "waiting",
                candidate_gpus=[asdict(gpu) for gpu in selected],
                idle_counts=self.idle_counts,
                required_world_size=required_world_size,
            )
            if selected and self.strict_now(selected):
                return selected
            time.sleep(self.poll_interval)

    def run_process(
        self,
        command: Sequence[str],
        selected: Sequence[GpuState],
        *,
        output: Path,
        state: str,
        allow_pause: bool,
    ) -> str:
        if not self.strict_now(selected):
            self.write_status(
                "launch_race",
                candidate_gpus=[asdict(gpu) for gpu in selected],
                world_size=len(selected),
            )
            append_jsonl(
                self.incident_path,
                {
                    "kind": "launch_race_avoided",
                    "created_at_unix": time.time(),
                    "candidate_gpus": [asdict(gpu) for gpu in selected],
                },
            )
            return "resources_changed"
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "logs/torchrun.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu.index) for gpu in selected
        )
        environment["INFINIDEPTH_LAUNCH_COMMAND"] = json.dumps(list(command))
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=self.project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.write_status(
                state,
                candidate_gpus=[asdict(gpu) for gpu in selected],
                world_size=len(selected),
                training_pid=process.pid,
                command=list(command),
                started_at_unix=time.time(),
            )
            pause_written = False
            while process.poll() is None:
                time.sleep(min(self.poll_interval, 5))
                current = {gpu.uuid: gpu for gpu in query_gpus()}
                active = [current[gpu.uuid] for gpu in selected if gpu.uuid in current]
                external = external_compute_pids(active, process.pid)
                if external and allow_pause and not pause_written:
                    control = output / "control/pause.request"
                    control.parent.mkdir(parents=True, exist_ok=True)
                    control.write_text(
                        json.dumps(
                            {
                                "reason": "external_compute_process_detected",
                                "external_pids": external,
                                "requested_at_unix": time.time(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    pause_written = True
                    append_jsonl(
                        self.incident_path,
                        {
                            "kind": "resource_pause_requested",
                            "external_pids": external,
                            "training_pid": process.pid,
                            "created_at_unix": time.time(),
                        },
                    )
            returncode = process.wait()
        report = read_json(output / "metrics/report.json")
        if returncode == 0 and report.get("status") in {"paused", "completed"}:
            return str(report["status"])
        self.alert(
            "training_failed",
            f"Training process exited with code {returncode}",
            returncode=returncode,
            log=str(log_path),
        )
        return "failed"

    def run_smoke(self, selected: Sequence[GpuState]) -> Optional[bool]:
        marker = self.scheduler_dir / "smoke_passed.json"
        previous = read_json(marker)
        if int(previous.get("world_size", 0)) == len(selected):
            return True
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = self.experiment / "runs" / f"ddp_smoke_{stamp}"
        config = make_smoke_config(
            self.config_path, self.project_root, root / "config.json"
        )
        ddp = root / "ddp"
        pause = ddp / "control/pause.request"
        pause.parent.mkdir(parents=True, exist_ok=True)
        pause.write_text(
            '{"reason":"smoke_resume_test","stage":"stage1","after_stage_step":2}\n',
            encoding="utf-8",
        )
        command = build_train_command(
            world_size=len(selected), config=config, output=ddp, smoke=True
        )
        pause_status = self.run_process(
            command,
            selected,
            output=ddp,
            state="smoke_ddp_pause",
            allow_pause=True,
        )
        if pause_status == "resources_changed":
            return None
        if pause_status != "paused":
            self.alert("smoke_failed", "DDP smoke did not cooperatively pause")
            return False
        checkpoint = ddp / "checkpoints/last.pt"
        if not checkpoint.is_file():
            self.alert("smoke_failed", "DDP pause did not produce last.pt")
            return False
        resume_command = build_train_command(
            world_size=len(selected),
            config=config,
            output=ddp,
            resume=checkpoint,
            smoke=True,
        )
        resume_status = self.run_process(
            resume_command,
            selected,
            output=ddp,
            state="smoke_ddp_resume",
            allow_pause=True,
        )
        if resume_status == "resources_changed":
            return None
        if resume_status != "completed":
            return False
        ddp_times = optimizer_step_times(ddp)
        if not ddp_times:
            self.alert("smoke_failed", "Smoke did not record optimizer step timing")
            return False
        ddp_report = read_json(ddp / "metrics/report.json")
        resume_verified = any(
            bool(stage.get("resume_sample_sequence_verified"))
            for stage in ddp_report.get("stages", [])
        )
        if not resume_verified:
            self.alert("smoke_failed", "DDP resume did not verify the next sample sequence")
            return False
        result = {
            "ddp_median_step_seconds": statistics.median(ddp_times),
            "world_size": len(selected),
            "completed_at_unix": time.time(),
            "parameter_checksums": ddp_report.get("parameter_checksums"),
            "parameters_consistent": ddp_report.get("parameters_consistent"),
            "resume_sample_sequence_verified": resume_verified,
        }
        atomic_json(root / "smoke_result.json", result)
        append_jsonl(self.incident_path, {"kind": "smoke_passed", **result})
        atomic_json(marker, result)
        return True

    def run(self) -> None:
        if os.uname().nodename != self.host:
            raise RuntimeError(
                f"Scheduler is configured for {self.host}"
            )
        while True:
            selected = self.wait_for_gpus()
            smoke = self.run_smoke(selected)
            if smoke is None:
                continue
            if not smoke:
                return
            break
        required_world_size = len(selected)
        while True:
            selected = self.wait_for_gpus(required_world_size)
            if not self.strict_now(selected):
                continue
            resume = self.output / "checkpoints/last.pt"
            command = build_train_command(
                world_size=required_world_size,
                config=self.config_path,
                output=self.output,
                resume=resume if resume.is_file() else None,
            )
            status = self.run_process(
                command,
                selected,
                output=self.output,
                state="training",
                allow_pause=True,
            )
            if status == "resources_changed":
                continue
            if status == "completed":
                self.write_status(
                    "completed", world_size=required_world_size, candidate_gpus=[]
                )
                self.alert("training_completed", "Exp3 training completed")
                return
            if status != "paused":
                return
            self.write_status(
                "resource_paused",
                world_size=required_world_size,
                candidate_gpus=[],
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule Exp3 DDP on idle local GPUs")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--poll-interval-seconds", type=int)
    parser.add_argument("--host")
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler = Scheduler(
        args.experiment,
        args.poll_interval_seconds,
        args.host,
        args.world_size,
    )
    scheduler.scheduler_dir.mkdir(parents=True, exist_ok=True)
    lock_path = scheduler.scheduler_dir / "scheduler.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Exp3 scheduler is already running") from exc
        (scheduler.scheduler_dir / "scheduler.pid").write_text(
            f"{os.getpid()}\n", encoding="utf-8"
        )
        scheduler.run()


if __name__ == "__main__":
    main()

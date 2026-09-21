"""Sequential A/B orchestration using the existing GPU and monitor helpers."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.monitor_exp2 import check_once, process_identity_error, all_finite, process_is_alive
from experiment.schedule_exp3 import (atomic_json, append_jsonl, query_gpus,
                                      external_compute_pids, build_train_command)


def read_json(path):
    return json.loads(path.read_text())


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Existing configuration differs; create a revision: {path}")
    else:
        atomic_json(path, value)


def check_test_report(path, minimum_tests, *, allow_skips=False):
    suites = ET.parse(path).getroot()
    cases = suites.findall(".//testcase")
    if len(cases) < minimum_tests or suites.findall(".//failure") or suites.findall(".//error") or (not allow_skips and suites.findall(".//skipped")):
        raise RuntimeError(f"Test gate did not pass: {path}")


def gpu_ready(gpu, options):
    return (
        gpu.index == int(options["gpu_index"])
        and gpu.memory_free_mib >= int(options["minimum_free_memory_mib"])
        and gpu.utilization_percent <= int(options["maximum_utilization_percent"])
        and (bool(options.get("allow_shared_gpu", False)) or not gpu.compute_pids)
    )


def prepare_config(experiment, backend, *, smoke=False):
    config = copy.deepcopy(read_json(experiment / "config.json"))
    config["model"]["refiner_backend"] = backend
    config["server"]["project_root"] = str(ROOT)
    config["server"]["environment"] = str(Path(sys.executable).parent.parent)
    config["experiment_id"] += "_" + backend + ("_smoke" if smoke else "")
    if smoke:
        manifest = read_json(ROOT / "experiment/data/hypersim100_train_manifest.json")
        ids = [item["id"] for item in manifest["samples"][:8]]
        config["data"]["expected_train_count"] = len(ids)
        config["runs"][0]["sample_ids"] = ids
        config["evaluation"]["sample_ids"] = config["evaluation"]["full_sample_ids"][:1]
        config["evaluation"]["full_sample_ids"] = config["evaluation"]["sample_ids"]
        stage = config["training"]["stages"]["stage1"]
        steps = int(config["automation"]["smoke_steps"])
        stage.update(min_steps=steps, max_steps=steps, eval_every=5, full_eval_every=steps, checkpoint_every=5)
    destination = experiment / ("smoke" if smoke else "arms") / backend / "config.json"
    immutable_json(destination, config)
    return destination


class SequenceRunner:
    def __init__(self, experiment):
        self.experiment = experiment
        self.output = experiment / "automation"
        self.options = read_json(experiment / "config.json")["automation"]
        self.gpu = None
        self.idle_counts = {}

    def status(self, phase, **values):
        record = dict(phase=phase, updated_at_unix=time.time(), automation_pid=os.getpid(), **values)
        atomic_json(self.output / "status.json", record)
        append_jsonl(self.output / "history.jsonl", record)

    def idle_gpu(self):
        while True:
            gpus = query_gpus()
            self.idle_counts = {
                gpu.uuid: self.idle_counts.get(gpu.uuid, 0) + 1 if gpu_ready(gpu, self.options) else 0
                for gpu in gpus
            }
            ready = [gpu for gpu in gpus if gpu_ready(gpu, self.options) and self.idle_counts[gpu.uuid] >= int(self.options["required_idle_checks"])]
            if ready:
                fresh = next((gpu for gpu in query_gpus() if gpu.uuid == ready[0].uuid), None)
                if fresh and gpu_ready(fresh, self.options):
                    self.gpu = fresh
                    return fresh
                self.idle_counts = {}
            self.status("waiting_for_idle_gpu", candidates=[asdict(gpu) for gpu in gpus], idle_counts=self.idle_counts)
            time.sleep(self.options["poll_seconds"])

    def execute(self, command, log, phase, *, config=None, run=None):
        gpu = self.idle_gpu()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.uuid, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
                   INFINIDEPTH_CHECKPOINT=read_json(self.experiment / "config.json")["model"]["checkpoint"],
                   INFINIDEPTH_TEST_TMP_ROOT=str(self.output / "test_tmp"),
                   FLEX_GEMM_AUTOTUNE_MODE="always",
                   FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(self.output / "flex_gemm_autotune_cache.json"),
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            if run is not None:
                atomic_json(run / "training_process.json", {"pid": process.pid, "command": command, "time": time.time()})
            self.status(phase, child_pid=process.pid, command=command, gpu=asdict(gpu), run=str(run) if run else None)
            append_jsonl(self.output / "incident.jsonl", dict(event="launch", phase=phase, command=command, pid=process.pid, gpu=asdict(gpu), time=time.time()))
            next_monitor = 0.0
            while process.poll() is None:
                if time.time() >= next_monitor:
                    selected = [item for item in query_gpus() if item.uuid == gpu.uuid]
                    outside = external_compute_pids(selected, process.pid)
                    if run is not None:
                        identity = process_identity_error(process.pid, self.experiment, config_path=config, run_path=run)
                        if process.poll() is not None:
                            break
                        try:
                            owner = Path(f"/proc/{process.pid}").stat().st_uid
                        except FileNotFoundError:
                            process.wait()
                            break
                        if identity or owner != os.getuid():
                            raise RuntimeError(f"Child identity mismatch; no process touched: {identity}")
                        if (outside and not self.options.get("allow_shared_gpu", False)) or not selected:
                            atomic_json(run / "control/pause.request", {"reason": "GPU resource conflict", "external_pids": outside})
                            self.status("resource_pause_requested", child_pid=process.pid, gpu=asdict(gpu))
                        monitor = check_once(self.experiment, process.pid, 7200, config_path=config, run_path=run)
                        self.status(phase, child_pid=process.pid, gpu=asdict(gpu), run=str(run), monitor=monitor)
                    elif (outside and not self.options.get("allow_shared_gpu", False)) or not selected:
                        # Popen owns this exact child; never signal an external GPU PID.
                        process.terminate()
                        process.wait(timeout=60)
                        raise RuntimeError("Resource conflict during CUDA verification; rerun verification on an idle GPU")
                    next_monitor = time.time() + self.options["poll_seconds"]
                time.sleep(5)
        if process.returncode != 0:
            raise RuntimeError(f"{phase} exited {process.returncode}; inspect {log}")

    def train(self, config, run, phase, *, pause_at=None):
        report_path = run / "metrics/report.json"
        report = read_json(report_path) if report_path.exists() else {}
        if report.get("status") == "completed":
            return report
        process_record = run / "training_process.json"
        if process_record.exists() and process_is_alive(int(read_json(process_record)["pid"])):
            raise RuntimeError(f"Previous child may still be alive; refusing duplicate launch: {run}")
        if report.get("status") == "failed":
            raise RuntimeError(f"Previous failure requires diagnosis, not blind restart: {run}")
        last = run / "checkpoints/last.pt"
        if pause_at is not None and not last.exists():
            immutable_json(run / "control/pause.request", {"stage": "stage1", "after_stage_step": pause_at, "reason": "smoke checkpoint recovery"})
        if report and not last.exists():
            raise RuntimeError(f"Unfinished run has no checkpoint; preserve it and create a revision: {run}")
        while True:
            command = build_train_command(world_size=1, config=config, output=run, resume=last if last.exists() else None)
            self.execute(command, run / "training.log", phase, config=config, run=run)
            report = read_json(report_path)
            if report.get("status") == "completed":
                return report
            if report.get("status") != "paused" or not last.exists():
                raise RuntimeError(f"Training did not complete or save a pause checkpoint: {run}")
            append_jsonl(self.output / "incident.jsonl", dict(event="resume_saved_pause", run=str(run), checkpoint=str(last), report=report, time=time.time()))

    def run(self):
        sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for folder in (ROOT / "InfiniDepth", ROOT / "training", self.experiment)
                   for path in sorted(folder.rglob("*.py"))}
        sources.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / "experiment").glob("*.py"))})
        immutable_json(self.output / "source_hashes.json", sources)
        packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
        immutable_json(self.output / "environment.json", {"python": sys.version, "executable": sys.executable, "packages": packages})
        for backend in ("spconv", "official_flex"):
            prepare_config(self.experiment, backend)
            prepare_config(self.experiment, backend, smoke=True)
        cpu_xml = self.output / "cpu_tests.xml"
        with (self.output / "cpu_tests.log").open("a") as handle:
            subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/disparity_refiner", "-m", "not cuda", "--disable-warnings", f"--junitxml={cpu_xml}"], cwd=ROOT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"), stdout=handle, stderr=subprocess.STDOUT, check=True)
        check_test_report(cpu_xml, 78)
        marker = self.output / "cuda_tests_passed.json"
        if not marker.exists():
            cuda_xml = self.output / "cuda_tests.xml"
            self.execute([sys.executable, "-m", "pytest", "-q", "tests/disparity_refiner", "-m", "cuda", f"--junitxml={cuda_xml}"], self.output / "cuda_tests.log", "cuda_verification")
            check_test_report(cuda_xml, 10)
            atomic_json(marker, {"passed": True, "time": time.time()})
        smoke_results = {}
        for backend in ("spconv", "official_flex"):
            config = prepare_config(self.experiment, backend, smoke=True)
            run = config.parent / "runs/main"
            report = self.train(config, run, f"smoke_{backend}", pause_at=self.options["smoke_pause_step"])
            if not report["stages"][0].get("resume_sample_sequence_verified"):
                raise RuntimeError("Smoke did not verify checkpoint sampling continuity")
            records = [json.loads(line) for line in (run / "metrics/steps.jsonl").read_text().splitlines()]
            records = {item["total_step"]: item for item in records}
            measured = [item["optimizer_step_seconds"] for step, item in records.items() if step > self.options["smoke_warmup_steps"]]
            if len(measured) < 50 or not all_finite(measured):
                raise RuntimeError("Smoke throughput measurement is incomplete")
            smoke_results[backend] = dict(mean_optimizer_step_seconds=statistics.mean(measured), peak_cuda_memory_bytes=report["peak_cuda_memory_bytes"], frozen_base_sha256=report["frozen_base_sha256"])
            destination = run / "evaluation_smoke"
            if not (destination / "summary.json").exists():
                self.execute([sys.executable, str(self.experiment / "evaluate_pair.py"), "--config", str(config), "--checkpoint", str(run / "checkpoints/last.pt"), "--output", str(destination), "--limit", "1"], destination / "evaluation.log", f"metric_smoke_{backend}")
        if smoke_results["spconv"]["frozen_base_sha256"] != smoke_results["official_flex"]["frozen_base_sha256"]:
            raise RuntimeError("A/B use different frozen Base weights")
        atomic_json(self.output / "smoke_summary.json", smoke_results)
        for backend in ("spconv", "official_flex"):
            config = prepare_config(self.experiment, backend)
            run = self.experiment / "runs" / backend
            self.train(config, run, f"training_{backend}")
            for checkpoint_name in ("last", "stage1_best"):
                destination = run / "evaluation" / checkpoint_name
                if not (destination / "summary.json").exists():
                    self.execute([sys.executable, str(self.experiment / "evaluate_pair.py"), "--config", str(config), "--checkpoint", str(run / "checkpoints" / f"{checkpoint_name}.pt"), "--output", str(destination)], destination / "evaluation.log", f"evaluation_{backend}_{checkpoint_name}")
        subprocess.run([sys.executable, str(self.experiment / "evaluate_pair.py"), "--compare", str(self.experiment)], cwd=ROOT, check=True)
        self.status("completed")
        atomic_json(self.output / "alert.json", {"kind": "completed", "time": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    if experiment != Path(__file__).resolve().parent or experiment.stat().st_uid != os.getuid():
        raise PermissionError("Runner only controls its own Exp6-4 directory")
    runner = SequenceRunner(experiment)
    runner.output.mkdir(parents=True, exist_ok=True)
    with (runner.output / "automation.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        try:
            runner.run()
        except Exception as exc:
            runner.status("failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(runner.output / "alert.json", {"kind": "failed", "error": str(exc), "time": time.time()})
            raise


if __name__ == "__main__":
    main()

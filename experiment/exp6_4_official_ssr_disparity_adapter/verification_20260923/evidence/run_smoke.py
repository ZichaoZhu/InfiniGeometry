"""Independent recovery acceptance; retain the earlier numerical failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
source = root / "current"
config = source / "experiment/exp6_4_official_ssr_disparity_adapter/local/acceptance6.json"
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
           TMPDIR=str(root / "tmp"), FLEX_GEMM_AUTOTUNE_MODE="always", FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(root / "tmp/flex.json"))
common = [sys.executable, "-m", "training.disparity_refiner.train", "--config", str(config), "--run-id", "main", "--device", "cuda:0"]
for label in ("continuous", "interrupted", "resumed"):
    free = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    assert int(free.strip()) >= 22528, "GPU resource gate failed; no new task launched"
    output = root / ("continuous" if label == "continuous" else "resumed")
    command = common + ["--output", str(output)]
    if label == "resumed":
        command += ["--resume", str(output / "checkpoints/last.pt")]
    print(json.dumps(dict(label=label, event="start", time=time.time())), flush=True)
    with (root / (label + ".log")).open("x") as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        with (root / (label + ".launch.json")).open("x") as handle:
            json.dump(dict(command=command, pid=child.pid, cwd=str(source), time=time.time()), handle, indent=2)
        if label == "interrupted":
            deadline = time.monotonic() + 180
            while not (output / "provenance.json").exists() and child.poll() is None:
                if time.monotonic() > deadline:
                    child.terminate()
                    child.wait()
                    raise TimeoutError("Own smoke startup timed out")
                time.sleep(.2)
            assert child.poll() is None
            (output / "control").mkdir()
            with (output / "control/pause.request").open("x") as handle:
                json.dump(dict(stage="stage1", after_stage_step=3, reason="handoff acceptance"), handle)
        code = child.wait()
    report = json.loads((output / "metrics/report.json").read_text())
    expected = "paused" if label == "interrupted" else "completed"
    assert code == 0 and report["status"] == expected, (label, code, report)
    with (root / (label + ".report.json")).open("x") as handle:
        json.dump(report, handle, indent=2)
    if label == "interrupted":
        # Own acceptance checkpoint only: retain the exact pause point for the round-trip check.
        import shutil
        shutil.copyfile(output / "checkpoints/last.pt", root / "pause_step3.pt")
        shutil.copyfile(output / "checkpoints/stage1_best.pt", root / "pause_best3.pt")
    print(json.dumps(dict(label=label, event="exit", status=report["status"], time=time.time())), flush=True)

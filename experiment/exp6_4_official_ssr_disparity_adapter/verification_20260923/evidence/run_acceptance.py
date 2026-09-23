"""One-off acceptance orchestration. Never touches historical outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "current"
EXPERIMENT = SOURCE / "experiment/exp6_4_official_ssr_disparity_adapter"
HISTORICAL = Path("/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/exp6_4_20260911/source_v8")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
           TMPDIR=str(ROOT / "tmp"), FLEX_GEMM_AUTOTUNE_MODE="always", FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(ROOT / "tmp/flex.json"))


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def gpu_gate():
    raw = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if int(raw.strip()) < 22528:
        raise RuntimeError("GPU2 has insufficient free memory; no new child launched")


def run(label, args, *, pause_run=None, pause_step=3, expected=0):
    gpu_gate()
    print(json.dumps(dict(event="start", label=label, time=time.time())), flush=True)
    with (ROOT / (label + ".log")).open("x") as log:
        child = subprocess.Popen([sys.executable, *args], cwd=SOURCE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / (label + ".launch.json"), dict(command=child.args, pid=child.pid, cwd=str(SOURCE), time=time.time()))
        if pause_run:
            deadline = time.monotonic() + 180
            while not (pause_run / "provenance.json").exists() and child.poll() is None:
                if time.monotonic() > deadline:
                    child.terminate()
                    child.wait()
                    raise TimeoutError("Own smoke did not reach provenance; stopped only this child")
                time.sleep(.2)
            assert child.poll() is None, label
            write(pause_run / "control/pause.request", dict(stage="stage1", after_stage_step=pause_step, reason="handoff acceptance"))
        code = child.wait()
    print(json.dumps(dict(event="exit", label=label, returncode=code, time=time.time())), flush=True)
    if code != expected:
        raise RuntimeError(f"{label} returned {code}, expected {expected}; see its log")


machine = dict(safe_root="/mnt/data/home/zhuzichao", data_root="/nas1/datasets/hypersim/raw",
               cache_root="/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr/hypersim_exp3",
               checkpoint="/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/InfiniDepth/checkpoints/depth/infinidepth.ckpt")
write(ROOT / "paths.json", machine)
prepared = EXPERIMENT / "local/official_smoke56.json"
run("prepare", [str(EXPERIMENT / "prepare_official.py"), "--paths", str(ROOT / "paths.json"),
     "--output-config", str(prepared), "--output", str(ROOT / "unused_smoke56"), "--smoke", "--check-assets"])
config = json.loads(prepared.read_text())
stage = config["training"]["stages"]["stage1"]
stage.update(min_steps=6, max_steps=6, eval_every=3, full_eval_every=3, checkpoint_every=3)
config["experiment_id"] += "_handoff_acceptance6"
path = EXPERIMENT / "local/acceptance6.json"
write(path, config)

# Refuse NAS fallbacks: the selected 8 train + 1 validation samples must already be cached.
sys.path.insert(0, str(SOURCE))
from training.disparity_refiner.data import load_manifest, select_manifest_entries
manifest = load_manifest(SOURCE / config["data"]["manifest"], config["data"]["manifest_sha256"])
for split, ids in [("train", config["runs"][0]["sample_ids"]), ("val", config["evaluation"]["sample_ids"])]:
    for entry in select_manifest_entries(manifest, source_root=Path(machine["data_root"]), split=split, sample_ids=ids):
        for kind in ("rgb", "depth"):
            cached = Path(machine["cache_root"]) / Path(entry[kind]["source"]).relative_to(machine["data_root"])
            assert cached.is_file() and not cached.is_symlink(), cached

worker = str(ROOT / "verify_handoff_cuda.py")
checkpoint = HISTORICAL / "experiment/exp6_4_official_ssr_disparity_adapter/runs/official_flex/checkpoints/stage1_best.pt"
for name in ("before", "current"):
    run("probe_" + name, [worker, "probe", "--source", str(ROOT / name), "--config", str(path),
        "--checkpoint", str(checkpoint), "--output", str(ROOT / ("probe_" + name))])
run("compare_probes", [worker, "compare-probes", "--source", str(SOURCE), "--output", str(ROOT / "compare_probes"),
    "--inputs", str(ROOT / "probe_before/probe.pt"), str(ROOT / "probe_current/probe.pt")])
common = ["-m", "training.disparity_refiner.train", "--config", str(path), "--run-id", "main", "--device", "cuda:0"]
run("continuous", [*common, "--output", str(ROOT / "continuous")])
run("interrupted", [*common, "--output", str(ROOT / "resumed")], pause_run=ROOT / "resumed")
assert json.loads((ROOT / "resumed/metrics/report.json").read_text())["status"] == "paused"
run("resumed", [*common, "--output", str(ROOT / "resumed"), "--resume", str(ROOT / "resumed/checkpoints/last.pt")])
run("compare_resumes", [worker, "compare-resumes", "--source", str(SOURCE), "--output", str(ROOT / "compare_resumes"),
    "--inputs", str(ROOT / "continuous/checkpoints/last.pt"), str(ROOT / "resumed/checkpoints/last.pt")])
write(ROOT / "acceptance_complete.json", dict(status="passed", time=time.time(), scope="short-step acceptance, not a new scientific experiment"))

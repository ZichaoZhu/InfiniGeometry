"""Time-bounded diagnostic only; use immutable source/checkpoint and new outputs."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def delta(a, b):
    import torch
    assert a.shape == b.shape
    error = (a.float() - b.float()).abs()
    return dict(exact=torch.equal(a, b), close_1e6=torch.allclose(a, b, atol=1e-6, rtol=1e-6),
                max_abs=float(error.max()), mean_abs=float(error.mean()),
                p99_abs=float(torch.quantile(error.flatten(), .99)),
                different=int(torch.count_nonzero(error)), elements=error.numel())


def canonical(feats, coords):
    import numpy as np
    coords = coords.detach().cpu()
    feats = feats.detach().cpu()
    array = coords.numpy()
    order = np.lexsort(tuple(array[:, i] for i in reversed(range(array.shape[1]))))
    return dict(coords=coords[order], features=feats[order],
                order_sha=hashlib.sha256(array.tobytes()).hexdigest())


def worker(args):
    import torch
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    # ponytail: a diagnostic cap, not a reservation; stop on OOM instead of retrying.
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    if "deterministic" in args.profile:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(args.source))
    spec = importlib.util.spec_from_file_location("acceptance_worker", args.acceptance)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import flex_gemm
    from flex_gemm import config as flex_config
    from training.disparity_refiner import export_assets
    if args.profile == "ieee":
        flex_config.SPCONV_ALLOW_TF32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    original_predict = export_assets.predict
    originals = []
    traces = []
    hooks = []

    def sorted_pool(module, inputs, kwargs):
        assert module.kernel_size == module.stride == (2, 2, 2)
        assert module.padding == (0, 0, 0) and kwargs.get("neighbor_cache") is None
        coords = inputs[1].clone()
        coords[:, 1:] = torch.div(coords[:, 1:], 2, rounding_mode="floor")
        kwargs["output_coords"] = torch.unique(coords, dim=0, sorted=True).contiguous()
        return inputs, kwargs

    def observe(model, sample, device, hw, chunk):
        if "canonical" in args.profile:
            for module in model.disparity_refiner.modules():
                if module.__class__.__name__ == "SparsePool3d":
                    hooks.append(module.register_forward_pre_hook(sorted_pool, with_kwargs=True))
        for repeat in range(2):
            iteration = [0]
            trace = {}
            capture_hooks = []

            def enter(module, inputs):
                iteration[0] += 1
                trace[f"input_d_{iteration[0]}"] = inputs[0].detach().cpu()
                if iteration[0] == 1:
                    trace["visual"] = inputs[1].detach().cpu()

            def capture(name):
                def hook(module, inputs, output):
                    if iteration[0] == 1:
                        coords = output[1] if module.__class__.__name__ in ("SparsePool3d", "PoolDown") else inputs[1]
                        trace[name] = canonical(output[0], coords)
                return hook

            capture_hooks.append(model.disparity_refiner.register_forward_pre_hook(enter))
            for name, module in model.disparity_refiner.unet.network.named_modules():
                if module.__class__.__name__ in ("SparseResBlock3d", "SparsePool3d", "PoolDown"):
                    capture_hooks.append(module.register_forward_hook(capture(name)))
            try:
                with torch.no_grad():
                    prediction = original_predict(model, sample, device, hw, chunk)
            finally:
                for handle in capture_hooks:
                    handle.remove()
            originals.append(prediction)
            traces.append(trace)
        torch.save(dict(predictions=originals, traces=traces), args.output / "trace.pt")
        write(args.output / "same_process.json", compare(originals[0], originals[1], traces[0], traces[1]))
        return originals[0]

    export_assets.predict = observe
    try:
        result = helper.probe(args)
    finally:
        export_assets.predict = original_predict
        for hook in hooks:
            hook.remove()
    from flex_gemm.autotuner import save_autotune_cache
    save_autotune_cache(str(args.output / "selected_kernels.json"))
    result.update(profile=args.profile, elapsed_seconds=time.monotonic() - started,
                  torch_deterministic=torch.are_deterministic_algorithms_enabled(),
                  flex_cuda_extension=flex_config.USE_CUDA_EXTENSION,
                  flex_allow_tf32=flex_config.SPCONV_ALLOW_TF32,
                  autotune_mode=flex_config.AUTOTUNE_MODE,
                  cache_input_sha256=sha(args.cache),
                  cache_input_unchanged=sha(args.cache) == args.cache_sha,
                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    write(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


def compare(a, b, ta, tb):
    import torch
    result = dict(predictions={str(k): delta(a[k], b[k]) for k in a}, trace={})
    for name in ta:
        x, y = ta[name], tb[name]
        if isinstance(x, dict):
            same_coords = torch.equal(x["coords"], y["coords"])
            result["trace"][name] = dict(same_coordinate_set=same_coords,
                                         same_row_order=x["order_sha"] == y["order_sha"])
            if same_coords:
                result["trace"][name]["features"] = delta(x["features"], y["features"])
        else:
            result["trace"][name] = delta(x, y)
            if name.startswith("input_d_"):
                result["trace"][name]["different_voxel_bins"] = int(torch.count_nonzero(torch.round(x*200) != torch.round(y*200)))
    return result


def summarize(root):
    import torch
    result = {}
    for profile in ("baseline", "deterministic", "canonical", "canonical_deterministic", "ieee"):
        paths = [root / f"{profile}_{i}" for i in (1, 2)]
        if not all((path / "result.json").exists() for path in paths):
            continue
        x, y = [torch.load(p / "trace.pt", map_location="cpu", weights_only=False) for p in paths]
        value = compare(x["predictions"][0], y["predictions"][0], x["traces"][0], y["traces"][0])
        del x, y
        x, y = [torch.load(p / "probe.pt", map_location="cpu", weights_only=False) for p in paths]
        value["loss"] = delta(x["loss"], y["loss"])
        value["gradients"] = {key: delta(x["gradients"][key], y["gradients"][key]) for key in x["gradients"]}
        value["updated_ssr"] = {key: delta(x["updated_ssr"][key], y["updated_ssr"][key]) for key in x["updated_ssr"]}
        value["same_frozen_base"] = x["frozen_base_sha256"] == y["frozen_base_sha256"]
        result[profile] = value
        del x, y
        gc.collect()
    write(root / "comparisons.json", result)
    print(json.dumps({name: data["predictions"] for name, data in result.items()}, indent=2))


def run(args):
    started = time.monotonic()
    root = args.output.resolve(strict=True)
    assert root.parent == Path("/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry")
    runs = root / "runs"
    runs.mkdir()
    cache_sha = sha(args.cache)
    records = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=str(args.gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TMPDIR=str(root / "tmp"),
               TRITON_CACHE_DIR=str(root / "triton"), FLEX_GEMM_AUTOTUNE_MODE="always",
               FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0", FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(args.cache))
    for profile in args.profiles:
        for repeat in (1, 2):
            remaining = args.budget_seconds - (time.monotonic() - started)
            if remaining < 90:
                raise TimeoutError("Diagnostic time budget exhausted; no further GPU process started")
            gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            free, utilization = map(int, gpu.strip().split(","))
            if free < 14000 or utilization > 30:
                raise RuntimeError("GPU admission no longer met; stop instead of touching other tasks")
            name = f"{profile}_{repeat}"
            output = runs / name
            output.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--source", str(args.source),
                "--acceptance", str(args.acceptance), "--config", str(args.config), "--checkpoint", str(args.checkpoint),
                "--output", str(output), "--profile", profile, "--cache", str(args.cache), "--cache-sha", cache_sha]
            child_env = dict(env)
            if "deterministic" in profile:
                child_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            step_start = time.monotonic()
            with (root / f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=args.source, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                write(root / f"{name}.launch.json", dict(pid=child.pid, command=command, cwd=str(args.source),
                    gpu_before=gpu.strip(), start_time=time.time(), profile=profile))
                print(json.dumps(dict(event="start", name=name, pid=child.pid)), flush=True)
                try:
                    code = child.wait(timeout=min(420, remaining - 30))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise TimeoutError("Only the child launched by this diagnostic was stopped")
            record = dict(name=name, returncode=code, seconds=time.monotonic()-step_start)
            records.append(record)
            write(root / f"{name}.exit.json", record)
            print(json.dumps(dict(event="exit", **record)), flush=True)
            if code and repeat == 1:
                break
    write(root / "run_summary.json", dict(elapsed_seconds=time.monotonic()-started, records=records,
         source=str(args.source), acceptance_sha=sha(args.acceptance), config_sha=sha(args.config),
         checkpoint_sha=sha(args.checkpoint), input_cache_sha=cache_sha, cache_unchanged=cache_sha == sha(args.cache)))
    summarize(runs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "worker", "summarize"])
    for name in ("source", "acceptance", "config", "checkpoint", "output", "cache"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--cache-sha")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--budget-seconds", type=int, default=1800)
    p.add_argument("--profiles", nargs="+", default=["baseline", "deterministic", "canonical"])
    p.add_argument("--profile", default="baseline")
    args = p.parse_args()
    try:
        {"run": run, "worker": worker, "summarize": lambda a: summarize(a.output)}[args.mode](args)
    except Exception:
        if args.output is not None and args.output.exists():
            write(args.output / "error.json", dict(error=traceback.format_exc(), time=time.time()))
        raise


if __name__ == "__main__":
    main()

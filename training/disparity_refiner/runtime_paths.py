"""Explicit machine paths; legacy configurations retain their original defaults."""
from pathlib import Path
import hashlib


LEGACY_SOURCE_ROOT = Path("/nas1/datasets/hypersim/raw")
LEGACY_PROTOCOL_ROOT = Path("/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp6-3_moge3_vs_infinidepth_exp3")
LEGACY_MOGE_ROOT = Path("/mnt/data/home/zhuzichao/projects/MoGe/deployments/official_v3_74fbce0_20260902")
PROTOCOL_SHA256 = "1940967e0d1896f71be61eb4da3877f46d07dd695f3642e3980670f55b74c2cc"
MASK_SHA256 = "41da5ed4a431afe13fdb66d84f78191e86b684fbd83c9ccea82c2e9f4de11449"


def explicit_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path(path.anchor):
        raise ValueError(f"An explicit non-root absolute path is required: {value}")
    return path.resolve()


def source_root(config):
    data = config["data"]
    root = explicit_path(data["source_root"])
    approved = explicit_path(data.get("readonly_source_root", str(LEGACY_SOURCE_ROOT)))
    if root != approved:
        raise PermissionError("Hypersim source root differs from the approved read-only root")
    if "readonly_source_root" in data and data.get("local_cache"):
        cache = explicit_path(data["local_cache"])
        if cache == root or root in cache.parents or cache in root.parents:
            raise PermissionError("Writable cache and read-only data must not overlap")
    return root


def protected_output(config, output):
    """No writes into inputs, caches or source code; resolve symlinks before checks."""
    candidate = Path(output).expanduser()
    if candidate.is_symlink():
        raise PermissionError("Output may not be a symlink")
    destination = explicit_path(candidate)
    safe = explicit_path(config["server"]["safe_root"])
    if destination == safe or safe not in destination.parents:
        raise PermissionError("Output must be a child of safe_root")
    protected = [source_root(config), explicit_path(config["data"]["local_cache"]),
                 explicit_path(config["server"]["project_root"])]
    for key in ("protocol_root", "official_moge_root"):
        if config["evaluation"].get(key):
            protected.append(explicit_path(config["evaluation"][key]))
    for root in protected:
        if destination == root or root in destination.parents or destination in root.parents:
            raise PermissionError(f"Output overlaps a protected input directory: {root}")
    checkpoint = explicit_path(config["model"]["checkpoint"])
    if destination == checkpoint or destination in checkpoint.parents:
        raise PermissionError("Output contains the Base checkpoint")
    return destination


def apply_machine_paths(config, paths):
    required = {"safe_root", "data_root", "cache_root", "checkpoint"}
    allowed = required | {"protocol_root", "official_moge_root"}
    if set(paths) - allowed or required - set(paths):
        raise ValueError(f"Machine paths require {sorted(required)}; allowed keys: {sorted(allowed)}")
    values = {key: str(explicit_path(value)) for key, value in paths.items()}
    config["server"].update(safe_root=values["safe_root"], cache=values["cache_root"],
                            temporary=str(Path(values["safe_root"]) / "tmp"))
    config["data"].update(source_root=values["data_root"], readonly_source_root=values["data_root"],
                          local_cache=values["cache_root"])
    config["model"]["checkpoint"] = values["checkpoint"]
    for key in ("protocol_root", "official_moge_root"):
        if key in values:
            config["evaluation"][key] = values[key]
    if Path(values["safe_root"]) not in Path(values["checkpoint"]).parents:
        raise PermissionError("Base checkpoint must remain inside safe_root")
    if Path(values["data_root"]) == Path(values["cache_root"]) or Path(values["data_root"]) in Path(values["cache_root"]).parents:
        raise PermissionError("Writable cache may not be inside read-only data")
    protected_output(config, Path(values["safe_root"]) / "tmp")


def evaluation_paths(config=None):
    evaluation = (config or {}).get("evaluation", {})
    directory = explicit_path(evaluation.get("protocol_root", str(LEGACY_PROTOCOL_ROOT)))
    source = directory / "code/run_common_eval.py"
    masks = directory / "local_detail/masks/hypersim_val100/moge3_v2_sam2_1_small_v1"
    for path, expected in ((source, PROTOCOL_SHA256), (masks / "manifest.json", MASK_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Fixed evaluation asset SHA-256 mismatch: {path}")
    moge = explicit_path(evaluation.get("official_moge_root", str(LEGACY_MOGE_ROOT)))
    if not (moge / "moge/test/metrics.py").is_file():
        raise FileNotFoundError(moge / "moge/test/metrics.py")
    return source, masks, moge

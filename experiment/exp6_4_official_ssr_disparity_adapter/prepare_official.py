"""Prepare one official_flex run. Never launches training, evaluation or a scheduler."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.exp6_4_official_ssr_disparity_adapter.automate import immutable_json, prepare_config
from training.disparity_refiner.runtime_paths import evaluation_paths, protected_output, source_root


def check_assets(config):
    manifest_path = ROOT / config["data"]["manifest"]
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["data"]["manifest_sha256"]:
        raise ValueError("Training manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    root = source_root(config)
    for key in ("image_index", "camera_index"):
        item = manifest[key]
        path = (root / item["path"]).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Metadata path or SHA-256 mismatch: {key}")
    if not Path(config["model"]["checkpoint"]).is_file():
        raise FileNotFoundError(config["model"]["checkpoint"])
    if not Path(config["data"]["local_cache"]).is_dir():
        raise FileNotFoundError(config["data"]["local_cache"])
    evaluation_paths(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, empty run directory outside the source checkout")
    parser.add_argument("--smoke", action="store_true", help="Use the historical 56-step smoke configuration")
    parser.add_argument("--check-only", action="store_true", help="Validate and print commands without writing files")
    parser.add_argument("--check-assets", action="store_true", help="Also check weights presence, metadata and fixed evaluation hashes; no dataset scan")
    args = parser.parse_args()
    destination = args.output_config.expanduser().resolve()
    # Restrict generated configurations to a new local area, never historical arms/configs.
    local = Path(__file__).resolve().parent / "local"
    if local.resolve() not in destination.parents or args.output_config.is_symlink():
        raise PermissionError(f"Generated configuration must stay under {local}")
    paths = json.loads(args.paths.read_text())
    config = prepare_config(Path(__file__).resolve().parent, "official_flex", smoke=args.smoke,
                            paths=paths, write=False)
    output = protected_output(config, args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Preparation requires a new/empty run; resume directly with the unchanged existing config")
    if destination.exists() and json.loads(destination.read_text()) != config:
        raise FileExistsError("Existing configuration differs; use a new output-config")
    if args.check_assets:
        check_assets(config)
    if not args.check_only:
        immutable_json(destination, config)
    command = [sys.executable, "-m", "training.disparity_refiner.train", "--config", str(destination),
               "--output", str(output), "--run-id", "main", "--device", "cuda:0"]
    print(json.dumps({"mode": "check_only" if args.check_only else "prepared", "backend": "official_flex",
                      "assets_checked": args.check_assets, "gpu_validation": "not_run", "config": str(destination),
                      "train": shlex.join(command),
                      "resume": shlex.join(command + ["--resume", str(output / "checkpoints/last.pt")]),
                      "note": "Set CUDA_VISIBLE_DEVICES explicitly; this command did not start any process."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

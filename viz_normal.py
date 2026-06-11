"""
Visualize the normal-only model: RGB | predicted normal | GT normal.

Loads a trained checkpoint, runs forward_test on a few val frames, colors the
normals (n -> (n*0.5+0.5) RGB), masks invalid GT pixels black, and writes
side-by-side PNGs. Per-frame median angular error is printed.

Run on the server (GPU) from the repo root, e.g.:
  CKPT=experiments/outputs/infinigeometry_normal/vits_normal_run1/checkpoints/last.ckpt \
  OUT=/root/autodl-tmp/viz_normal  N=6  $PY viz_normal.py
"""
import os
import sys

CKPT = os.environ["CKPT"]
OUT = os.environ.get("OUT", "/root/autodl-tmp/viz_normal")
N = int(os.environ.get("N", "6"))

sys.argv = [
    "viz_normal.py",
    "--c", "training/exp_configs/exps/infinigeometry_normal_vits.yaml",
    "--i", "training/exp_configs/components/data/train/infinigeometry_train_hypersim_normal.yaml",
    "exp_name=viz_normal",
]

import cv2
import hydra
import numpy as np
import torch
from training.config.config import cfg

os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model ----
pipeline = hydra.utils.instantiate(cfg.model.pipeline, _recursive_=False)
sd = torch.load(CKPT, map_location="cpu")["state_dict"]
sd = {k[len("pipeline."):]: v for k, v in sd.items() if k.startswith("pipeline.")}
missing, unexpected = pipeline.load_state_dict(sd, strict=False)
print(f"[ckpt] {CKPT}\n[ckpt] loaded; missing={len(missing)} unexpected={len(unexpected)}")
pipeline.to(device).eval()

# ---- val dataset ----
ds = hydra.utils.instantiate(cfg.data.val_dataset.dataset_opts[0], _recursive_=False)
total = len(ds)
idxs = [int(i * total / N) for i in range(N)]
print(f"[data] val frames={total}, picking {idxs}")


def collate_one(sample):
    batch = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def colorize_normal(n_chw, valid_hw=None):
    """n_chw: [3,H,W] in [-1,1] -> uint8 RGB [H,W,3]; invalid -> black."""
    n = n_chw.transpose(1, 2, 0)
    rgb = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    if valid_hw is not None:
        rgb[~valid_hw] = 0
    return rgb


for i in idxs:
    sample = ds[i]
    batch = collate_one(sample)
    with torch.no_grad():
        out = pipeline.forward_test(batch)
    pred = out["normal"][0].float().cpu().numpy()        # [3,H,W]
    gt = batch["normal"][0].float().cpu().numpy()         # [3,H,W]
    H, W = pred.shape[-2:]

    valid = None
    if "normal_valid" in batch:
        valid = batch["normal_valid"][0, 0].cpu().numpy().astype(bool)
    else:
        valid = np.linalg.norm(gt, axis=0) > 0.5

    # per-frame median angular error
    dot = np.clip((pred * gt).sum(axis=0), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot[valid])) if valid.any() else np.array([np.nan])
    med = float(np.median(ang))

    img = sample["image"].transpose(1, 2, 0)              # [512,672,3] in [0,1]
    img = cv2.resize((img * 255).astype(np.uint8), (W, H))
    pred_rgb = colorize_normal(pred)                       # show pred everywhere
    gt_rgb = colorize_normal(gt, valid)                    # GT invalid -> black

    panel = np.concatenate([img, pred_rgb, gt_rgb], axis=1)  # [H, 3W, 3] RGB
    name = sample.get("image_name", f"frame{i}")
    fn = os.path.join(OUT, f"{i:03d}_med{med:04.1f}deg.png")
    cv2.imwrite(fn, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"[{i:3d}] median={med:5.1f}deg  ->  {fn}")

print(f"DONE  ({N} panels in {OUT}; layout: RGB | pred | GT)")

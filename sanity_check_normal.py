"""
CPU sanity check for the InfiniGeometry normal-only pipeline (no GPU needed).

Verifies, end to end on one real batch:
  1. config merging (exp + data yaml) works;
  2. dataset reads normal_cam.hdf5, masks NaN / non-unit pixels, and the
     transform chain emits sampled_normal_for_disparity / _mask_for_disparity;
  3. ViT-S backbone builds and loads DINOv3 weights (strict=True);
  4. forward_train produces a finite angular loss and backward reaches both
     the normal head and the backbone.

Run on the server from the repo root:
  SANITY_STAGE=data  $PY sanity_check_normal.py   # dataset/transform only
  SANITY_STAGE=model $PY sanity_check_normal.py   # tiny model fwd/bwd
  $PY sanity_check_normal.py                      # everything, full size
                                                  # (needs > 2GB RAM)

The staged + tiny variants exist because the AutoDL no-GPU mode container
is capped at 2GB RAM; full-size single-process sanity gets OOM-killed.
"""
import os
import sys

STAGE = os.environ.get("SANITY_STAGE", "all")  # data | model | all
TINY = STAGE == "model"  # shrink image/queries so torch+ViT-S fit in 2GB

sys.argv = [
    "sanity_check_normal.py",
    "--c", "training/exp_configs/exps/infinigeometry_normal_vits.yaml",
    "--i", "training/exp_configs/components/data/train/infinigeometry_train_hypersim_normal.yaml",
    "exp_name=sanity_normal",
    "data.train_loader_opts.batch_size=1",
    "data.train_loader_opts.num_workers=0",
]

import hydra
import torch
from training.config.config import cfg

torch.set_num_threads(1)
if TINY:
    t = cfg.data.train_dataset.dataset_opts[0].transforms
    t[0].height, t[0].width = 128, 160   # Crop_Resize
    t[1].sample_q = 200                  # RapidSampleQueryPairs

print(f"[cfg] stage={STAGE} entry={cfg.entry} encoder={cfg.model.pipeline.config.encoder} "
      f"predict_normal={cfg.model.pipeline.config.predict_normal}")

# ---- data ----
datamodule = hydra.utils.instantiate(cfg.data, wo_train=False, _recursive_=False)
loader = datamodule.train_dataloader()
batch = next(iter(loader))
print("[batch] tensor keys:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")

n_gt = batch["sampled_normal_for_disparity"]
n_mask = batch["sampled_normal_mask_for_disparity"]
assert n_gt.ndim == 3 and n_gt.shape[-1] == 3 and n_gt.shape[1] > 0, \
    f"normal GT missing/empty: {tuple(n_gt.shape)}"
assert n_mask.shape == n_gt.shape[:2], f"mask misaligned: {tuple(n_mask.shape)}"
lens = n_gt.norm(dim=-1)
valid_ratio = n_mask.float().mean().item()
print(f"[normal GT] shape={tuple(n_gt.shape)} valid_ratio={valid_ratio:.3f}")
if n_mask.any():
    print(f"[normal GT] |n| over valid: min={lens[n_mask].min():.4f} max={lens[n_mask].max():.4f}")
assert valid_ratio > 0.05, "almost no valid normals in this batch — check masks"

if STAGE == "data":
    print("SANITY OK (data stage)")
    sys.exit(0)

# ---- model (CPU) ----
pipeline = hydra.utils.instantiate(cfg.model.pipeline, _recursive_=False)
pipeline.train()
out = pipeline.forward_train(batch)
print(f"[forward] {({k: round(float(v), 5) for k, v in out.items()})}")
assert torch.isfinite(out["loss"]), "loss is not finite"

out["loss"].backward()
head_grad = pipeline.depth_implicit_head.out_layer.layers[-2].weight.grad
bb_grad = pipeline.pretrained.blocks[0].attn.qkv.weight.grad
assert head_grad is not None and head_grad.abs().sum() > 0, "no gradient in normal head"
assert bb_grad is not None, "no gradient in backbone (end-to-end expected)"
print(f"[backward] head grad abs-mean={head_grad.abs().mean():.3e} "
      f"backbone grad abs-mean={bb_grad.abs().mean():.3e}")

print("SANITY OK")

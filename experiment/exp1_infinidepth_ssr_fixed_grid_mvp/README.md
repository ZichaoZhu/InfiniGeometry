# exp1: InfiniDepth SSR fixed-grid MVP

## Hypothesis

A zero-initialized SSR attached to the frozen official InfiniDepth RGB model can
learn a useful one-step geometric correction on one fixed Hypersim sample
without changing K0 or introducing ground truth into the inference path.

## Single controlled change

The official InfiniDepth base and MoGe2 scale reference remain frozen. The only
trainable component is the newly initialized SSR, trained at K=1 on a fixed
384x512 grid.

## Acceptance criteria

- K1 Point Rel improves by at least 1% relative to K0 at the selected checkpoint.
- Deterministic geometry loss falls by at least 5% from initialization.
- K0 tensor SHA-256 remains unchanged throughout training.
- Base and MoGe2 parameters never receive gradients.
- Save/reload agrees at `rtol=1e-6`, `atol=1e-7`.

## Current status

Planned. No result or conclusion is claimed until the server run finishes and
`metrics/report.json` changes to `completed` or `failed`.

## Reproduction

Run on `ZJU3DV-S115` from the server checkout:

```bash
bash experiment/exp1_infinidepth_ssr_fixed_grid_mvp/run.sh
```

The launcher refuses output paths outside `/mnt/data/home/zhuzichao/`.

## Asset policy

Only `best.pt`, `last.pt`, `gt.ply`, `k0.ply`, `k1.ply`, `k3.ply`, one training
curve, one geometry comparison, and one raw run log may be retained. Large
assets stay on the server and are represented in `artifacts/manifest.json`.

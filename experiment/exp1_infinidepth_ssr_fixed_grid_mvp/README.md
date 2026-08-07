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

Completed and accepted on 2026-08-07. The selected checkpoint was reached at
step 200, where K1 Point Rel improved from `0.06967275` to `0.06896392`
(`1.017%`) and deterministic geometry loss fell from `0.43732962` to
`0.37090552` (`15.19%`). Diagnostic K3 Point Rel was `0.06822955`.

K0 remained bitwise stable, the adapter and zero-initialized SSR both had zero
maximum absolute regression error, frozen Base/MoGe2 gradients stayed `None`,
and checkpoint reload passed the required tolerance. The run stopped when both
primary thresholds were first satisfied. Full structured values are in
`metrics/report.json`; the two earlier operational failures and their fixes are
retained in `provenance.json` under the same experiment ID.

## Conclusion and limitation

The fixed-grid SSR MVP satisfies its scoped acceptance criteria without
changing the frozen InfiniDepth prediction path. This is a one-sample training
experiment and demonstrates optimization viability only; it does not establish
generalization to held-out Hypersim scenes or other datasets.

## Reproduction

Run on `ZJU3DV-S115` from the server checkout:

```bash
bash experiment/exp1_infinidepth_ssr_fixed_grid_mvp/run.sh
```

The launcher refuses output paths outside `/mnt/data/home/zhuzichao/`.
After the run, validate retained server assets with:

```bash
python experiment/validate_experiment.py --require-untracked-assets \
  experiment/exp1_infinidepth_ssr_fixed_grid_mvp
```

## Asset policy

Only `best.pt`, `last.pt`, `gt.ply`, `k0.ply`, `k1.ply`, `k3.ply`, one training
curve, one geometry comparison, and one raw run log may be retained. Large
assets stay on the server and are represented in `artifacts/manifest.json`.

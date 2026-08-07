# InfiniDepth experiment registry

Experiment identifiers are global, monotonic, and never reused. A new top-level
directory is created only when the research question, primary variable, data
scope, or training strategy changes. Operational retries stay in the same
experiment and are recorded in its provenance.

| ID | Research question | Status | Base commit | Run commit | Primary metric | Conclusion |
|---|---|---|---|---|---|---|
| [exp1](exp1_infinidepth_ssr_fixed_grid_mvp/README.md) | Can a frozen official InfiniDepth base be improved by a fixed-grid SSR on one Hypersim sample? | completed | `36c6e0c31887fafc210184ee43ca475230704095` | `ae887f8c7c9f3ac857a0aec33e38f4e7cb622a52` | K1 Point Rel | Accepted at step 200: K1 improved 1.017% and geometry loss fell 15.19%. |

Allowed status values are `planned`, `running`, `completed`, and `failed`.
Unexecuted work must never be described as a completed experiment.

Run `python experiment/validate_experiment.py` before committing experiment
records. Large checkpoints, point clouds, and logs remain in the matching
server directory and are represented by checksums in `artifacts/manifest.json`.
On the asset-owning server, add `--require-untracked-assets` to verify that all
server-only files are present and match the manifest.

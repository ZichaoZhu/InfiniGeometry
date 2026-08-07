# InfiniDepth experiment registry

Experiment identifiers are global, monotonic, and never reused. A new top-level
directory is created only when the research question, primary variable, data
scope, or training strategy changes. Operational retries stay in the same
experiment and are recorded in its provenance.

| ID | Research question | Status | Base commit | Primary metric | Conclusion |
|---|---|---|---|---|---|
| [exp1](exp1_infinidepth_ssr_fixed_grid_mvp/README.md) | Can a frozen official InfiniDepth base be improved by a fixed-grid SSR on one Hypersim sample? | planned | `36c6e0c31887fafc210184ee43ca475230704095` | K1 Point Rel | Pending server execution |

Allowed status values are `planned`, `running`, `completed`, and `failed`.
Unexecuted work must never be described as a completed experiment.

Run `python experiment/validate_experiment.py` before committing experiment
records. Large checkpoints, point clouds, and logs remain in the matching
server directory and are represented by checksums in `artifacts/manifest.json`.

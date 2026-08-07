# expN_short_ascii_description

## Hypothesis

State one falsifiable research question and its acceptance criterion.

## Single controlled change

Describe the primary variable. Do not mix unrelated implementation changes.

## Configuration

- Code commit and clean/dirty state:
- Base commit:
- Data and immutable source checksums:
- Model checkpoints and checksums:
- Trainable modules:
- Resolution, batch size, and seed:
- Optimizer and losses:
- Hardware and software environment:

The resolved machine-readable configuration belongs in `config.json`; do not
duplicate the complete JSON here.

## Results

Summarize baseline, best, final, runtime, peak memory, and numerical status.
The canonical values belong in `metrics/report.json` and evaluation history in
`metrics/history.jsonl`.

## Conclusion and limitations

State what the result demonstrates and what it cannot establish. Failed or
incomplete runs remain evidence and must be labelled accurately.

## Reproduction

Keep the launcher in `run.sh`. Record the exact invocation and environment in
`provenance.json`.

## Assets

Register every retained asset in `artifacts/manifest.json` with its role,
relative or external path, size, SHA-256, generating command, and Git status.
Do not copy datasets, pretrained weights, source snapshots, caches, or duplicate
metric formats into the experiment directory.

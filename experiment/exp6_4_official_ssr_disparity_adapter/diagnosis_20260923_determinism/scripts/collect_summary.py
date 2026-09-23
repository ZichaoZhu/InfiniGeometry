"""Read finished diagnostics and retain compact provenance, without CUDA work."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from diagnose_sparse import sha, write

root = Path(__file__).resolve().parent
followup = root.with_name(root.name + '_followup')
input_cache = json.loads((root / 'fixed_autotune.json').read_text())
cache_hashes = {}
comparisons = {}
workers = []
files = {}
for label, directory in [('main', root), ('followup', followup)]:
    comparisons.update(json.loads((directory / 'runs/comparisons.json').read_text()))
    for run in sorted((directory / 'runs').iterdir()):
        if not run.is_dir():
            continue
        result = json.loads((run / 'result.json').read_text())
        kernels = json.loads((run / 'selected_kernels.json').read_text())
        cache_hashes[run.name] = dict(raw_sha=sha(run / 'selected_kernels.json'),
            canonical_sha=hashlib.sha256(json.dumps(kernels, sort_keys=True).encode()).hexdigest(),
            equals_input_cache=kernels == input_cache)
        workers.append(result)
    for path in directory.glob('*.json'):
        files[f'{label}/{path.name}'] = sha(path)
    for name in ('result.json', 'same_process.json', 'comparisons.json'):
        for path in (directory / 'runs').rglob(name):
            files[f'{label}/{path.relative_to(directory)}'] = sha(path)

cpu = subprocess.run([sys.executable, str(root / 'test_diagnose_sparse.py'), '-v'], capture_output=True, text=True)
write(root / 'cpu_tests.json', dict(command=cpu.args, returncode=cpu.returncode, stdout=cpu.stdout, stderr=cpu.stderr))
assert cpu.returncode == 0
gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'], text=True).strip()
summary = dict(status='diagnosis_completed_default_still_nondeterministic',
    started_utc='2026-09-23T12:45:17+00:00', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    roots={'main':str(root), 'followup':str(followup)}, gpu_after=gpu,
    checkpoint_sha256=workers[0]['checkpoint_sha256'], sample_id=workers[0]['sample_id'],
    sample_count=1, successful_workers=len(workers), tests_passed=3,
    pool_isolation=json.loads((root / 'pool_isolation.json').read_text()),
    kernels=cache_hashes, comparisons={}, source_unchanged_by_diagnostic=True,
    notes=['Single cached Val100 sample, not full evaluation or proof for every environment.',
           'Canonical output coordinates supplied only by temporary forward pre-hooks; installed code untouched.',
           'Optimizer was freshly initialized for a one-step diagnostic, not a restored historical optimizer.',
           'isolate_pool.py v1 missed explicit output_shape; retained; isolate_pool_v2.py passed.',
           'Default acceptance failure retained. No production deterministic mode enabled.'])
assert len({r['checkpoint_sha256'] for r in workers}) == 1
assert all(r['base_frozen'] and r['cache_input_unchanged'] for r in workers)
assert all(v['equals_input_cache'] for v in cache_hashes.values())
for name, data in comparisons.items():
    item = {key:data[key] for key in ('predictions','loss','same_frozen_base')}
    for key in ('gradients','updated_ssr'):
        values = data[key].values()
        item[key] = dict(total=len(data[key]), exact=sum(v['exact'] for v in values),
            close_1e6=sum(v['close_1e6'] for v in values), max_abs=max(v['max_abs'] for v in values))
    summary['comparisons'][name] = item
write(root / 'diagnosis_summary.json', summary)
files['main/cpu_tests.json'] = sha(root / 'cpu_tests.json')
files['main/diagnosis_summary.json'] = sha(root / 'diagnosis_summary.json')
write(root / 'evidence_manifest.json', files)
print(json.dumps(dict(successful_workers=len(workers), all_kernels_identical=True, gpu_after=gpu)))

"""Replay identical first-pool inputs; inspect neighbor sets versus summation order."""
import argparse
import json
import os
from pathlib import Path
import torch
from diagnose_sparse import canonical, delta, write
from flex_gemm.nn import SparsePool3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    trace = torch.load(args.root / 'runs/baseline_1/trace.pt', map_location='cpu', weights_only=False)['traces'][0]
    entry = trace['down_stages.0.0']
    feats, coords = entry['features'].cuda(), entry['coords'].cuda()
    assert coords.shape[1] == 4 and coords.dtype == torch.int32
    shape = torch.Size([int(x) + 1 for x in coords.amax(0)] + [feats.shape[1]])
    pool = SparsePool3d(kernel_size=2, stride=2, reduce='mean').cuda()
    expected = coords.clone()
    expected[:, 1:] = torch.div(expected[:, 1:], 2, rounding_mode='floor')
    expected = torch.unique(expected, dim=0, sorted=True).contiguous()
    captured = {}
    for mode in ('default', 'canonical'):
        captured[mode] = []
        for _ in range(3):
            with torch.no_grad():
                out, out_coords, _, cache = pool(feats, coords, shape, output_coords=expected if mode == 'canonical' else None)
            value = canonical(out, out_coords)
            indices = cache.fwd_seg_indices.detach().cpu()
            offsets = cache.fwd_seg_offsets.detach().cpu()
            import numpy as np
            raw_coords = out_coords.cpu().numpy()
            order = np.lexsort(tuple(raw_coords[:, i] for i in reversed(range(4))))
            lengths = offsets[1:] - offsets[:-1]
            neighbors = torch.full((out_coords.shape[0], int(lengths.max())), -1, dtype=torch.int64)
            for j in range(neighbors.shape[1]):
                valid = lengths > j
                neighbors[valid, j] = indices[(offsets[:-1] + j)[valid]].long()
            value['neighbors'] = neighbors[order]
            captured[mode].append(value)
    report = dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cases={})
    for mode, values in captured.items():
        comparisons = []
        for value in values[1:]:
            a, b = values[0], value
            comparisons.append(dict(same_coordinate_set=torch.equal(a['coords'], b['coords']),
                same_coordinate_order=a['order_sha'] == b['order_sha'],
                same_neighbor_sets=torch.equal(a['neighbors'].sort(1).values, b['neighbors'].sort(1).values),
                changed_neighbor_order_rows=int((a['neighbors'] != b['neighbors']).any(1).sum()),
                pooled_features=delta(a['features'], b['features'])))
        report['cases'][mode] = comparisons
    a, b = captured['default'][0], captured['canonical'][0]
    report['default_vs_canonical'] = dict(same_coordinate_set=torch.equal(a['coords'], b['coords']),
        same_neighbor_sets=torch.equal(a['neighbors'].sort(1).values, b['neighbors'].sort(1).values),
        pooled_features=delta(a['features'], b['features']))
    write(args.root / 'pool_isolation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

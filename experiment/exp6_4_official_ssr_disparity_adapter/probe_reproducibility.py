"""Bounded repeated-forward check; never changes weights or formal metrics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument("--project-root",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.project_root.resolve()))
from InfiniDepth.model import InfiniDepth
from training.disparity_refiner.train import _load_samples
from training.disparity_refiner.export_assets import load_refiner_checkpoint

experiment=a.project_root/"experiment/exp6_4_official_ssr_disparity_adapter"
config=json.loads((experiment/"arms/official_flex/config.json").read_text())
reference={r["sample_id"]:r for r in [json.loads(s) for s in (a.output/"inputs/official_flex/per_image.jsonl").read_text().splitlines()]}
samples=_load_samples(config,config["runs"][0],a.project_root,split="val",sample_ids=config["evaluation"]["full_sample_ids"][:2])
torch.cuda.set_device(0)
model=InfiniDepth(model_path=config["model"]["checkpoint"]).cuda()
model.attach_disparity_refiner(backend="official_flex",voxel_resolution=200)
ckpt=load_refiner_checkpoint(model,experiment/"runs/official_flex/checkpoints/last.pt")
model.eval()
results=[]
for sample in samples:
    first=None
    for repeat in range(3):
        with torch.no_grad():
            output=model.forward_dense_refined(sample.image[None].cuda(),query_hw=(384,512),num_refinement_steps=5,chunk_size=10000)
        seq=np.stack([d[0].float().cpu().numpy() for d in output.disparity_sequence])
        target=sample.target_disparity.numpy()
        mask=sample.valid_mask.numpy()
        if first is None: first=seq.copy()
        result=dict(sample_id=sample.sample_id,repeat=repeat,
                    mae_vs_reference={str(k):float(np.mean(np.abs(seq[k]-target)[mask],dtype=np.float64))-reference[sample.sample_id]["metrics"][f"k{k}"]["native_disparity_mae"] for k in (0,1,3,5)},
                    map_mae_vs_first=[float(np.abs(s-f)[mask].mean()) for s,f in zip(seq,first)],
                    bins_changed_vs_first=[float((np.round(200*s)!=np.round(200*f))[mask].mean()) for s,f in zip(seq,first)])
        results.append(result)
        print(json.dumps(result),flush=True)
path=a.output/"repeated_forward_probe.json"
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps(dict(checkpoint_sha256=ckpt,results=results),indent=2)+"\n")

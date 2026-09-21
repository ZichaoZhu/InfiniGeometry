"""Exp6-4 paired diagnostics. Summary needs only the standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ARMS = ("spconv", "official_flex")
STEPS = (0, 1, 3, 5)
METRICS = ("native_disparity_mae", "affine_point_rel", "local_point_rel",
           "affine_depth_rel", "local_depth_rel", "local_point_delta_0.01",
           "boundary_f1_radius1")
EPS = 1e-6


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def indexed(records):
    result = {r["sample_id"]: r for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate sample IDs")
    return result


def direction(metric):
    return -1 if "delta" in metric or "f1" in metric else 1


def outcome(improvement):
    return "win" if improvement > EPS else "loss" if improvement < -EPS else "tie"


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    def q(p):
        pos = (len(values) - 1) * p
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return dict(n=len(values), mean=statistics.mean(values), median=q(.5), p10=q(.1), p90=q(.9))


def paired_inputs(output, config):
    records = {a: indexed(lines(output / "inputs" / a / "per_image.jsonl")) for a in ARMS}
    expected = set(config["evaluation"]["full_sample_ids"])
    if len(expected) != 100 or any(set(r) != expected for r in records.values()):
        raise ValueError("Expected identical, unique Val100 IDs")
    for sid in sorted(expected):
        a, b = (records[arm][sid] for arm in ARMS)
        for key in ("input_tensor_sha256", "rgb_source_sha256", "depth_source_sha256", "input_resolution", "local_asset"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Input mismatch {sid}: {key}")
        for metric in METRICS:
            va, vb = (r["metrics"]["k0"].get(metric) for r in (a, b))
            if (va is None) != (vb is None) or (va is not None and not math.isclose(va, vb, rel_tol=1e-6, abs_tol=1e-7)):
                raise ValueError(f"K0 mismatch {sid}: {metric}")
        for k in STEPS:
            ma, mb = (r["metrics"][f"k{k}"] for r in (a, b))
            for m in (ma, mb):
                if m["invalid_prediction_rate"] != 0 or any(v is not None and not math.isfinite(v) for v in m.values()):
                    raise ValueError(f"Invalid prediction/metric: {sid} K{k}")
            if ma["valid_pixel_count"] != mb["valid_pixel_count"]:
                raise ValueError(f"Mask mismatch: {sid} K{k}")
    for arm in ARMS:
        aggregate = read(output / "inputs" / arm / "summary.json")["aggregate"]
        for k in STEPS:
            for metric in METRICS:
                values = [r["metrics"][f"k{k}"].get(metric) for r in records[arm].values()]
                avg = statistics.mean(v for v in values if v is not None)
                if not math.isclose(avg, aggregate[f"k{k}"][metric], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Aggregate mismatch {arm} K{k}: {metric}")
    return records


def summarize(args):
    output, config = args.output, read(args.config)
    records = paired_inputs(output, config)
    all_rows, summary, scenes = [], {}, []
    for metric in METRICS:
        rows = []
        sign = direction(metric)
        for sid in sorted(records[ARMS[0]]):
            a, b = (records[arm][sid]["metrics"] for arm in ARMS)
            if any(m[f"k{k}"].get(metric) is None for m in (a, b) for k in STEPS):
                continue
            row = dict(sample_id=sid, scene=sid.split("_cam_")[0], metric=metric)
            for arm, values in zip(ARMS, (a, b)):
                row.update({f"{arm}_k{k}": values[f"k{k}"][metric] for k in STEPS})
                row[f"{arm}_gain"] = sign * (values["k0"][metric] - values["k3"][metric])
            row["official_minus_spconv"] = b["k3"][metric] - a["k3"][metric]
            row["official_gain_over_spconv"] = -sign * row["official_minus_spconv"]
            row["spconv_outcome"] = outcome(row["spconv_gain"])
            row["official_outcome"] = outcome(row["official_flex_gain"])
            rows.append(row)
        groups, transitions = {}, {}
        for row in rows:
            name = row["spconv_outcome"] + "/" + row["official_outcome"]
            groups[name] = groups.get(name, 0) + 1
        gains = [r["official_gain_over_spconv"] for r in rows]
        for arm in ARMS:
            def value(r, k):
                return sign * r[f"{arm}_k{k}"]
            transitions[arm] = {
                "k1_better_k0_k3_worse_k1": sum(value(r, 1) < value(r, 0)-EPS and value(r, 3) > value(r, 1)+EPS for r in rows),
                "k1_better_k0_k3_worse_k0": sum(value(r, 1) < value(r, 0)-EPS and value(r, 3) > value(r, 0)+EPS for r in rows),
                "k3_better_k0_k5_worse_k3": sum(value(r, 3) < value(r, 0)-EPS and value(r, 5) > value(r, 3)+EPS for r in rows),
                "k3_worse_k0_k5_worse_k3": sum(value(r, 3) > value(r, 0)+EPS and value(r, 5) > value(r, 3)+EPS for r in rows),
                "step_changes": {f"{p}_to_{q}": distribution([value(r, p)-value(r, q) for r in rows]) for p,q in ((0,1),(1,3),(3,5))},
                "oracle_best_k_diagnostic_only": {f"k{k}": sum(min(STEPS, key=lambda x: value(r, x)) == k for r in rows) for k in STEPS},
                "gain_over_k0": distribution([r[f"{arm}_gain"] for r in rows]),
                "worst_five": [{"sample_id": r["sample_id"], "gain": r[f"{arm}_gain"]} for r in sorted(rows, key=lambda r: r[f"{arm}_gain"])[:5]],
            }
        failures = {a: {r["sample_id"] for r in rows if r[f"{a}_gain"] < -EPS} for a in ARMS}
        union = failures[ARMS[0]] | failures[ARMS[1]]
        positive = sorted([g for g in gains if g > 0], reverse=True)
        summary[metric] = {
            "official_minus_spconv": distribution([r["official_minus_spconv"] for r in rows]),
            "official_gain_over_spconv": distribution(gains),
            "official_win_tie_loss": {o: sum(outcome(g)==o for g in gains) for o in ("win","tie","loss")},
            "relative_k0_groups_spconv_official": groups,
            "failure_overlap": dict(intersection=len(failures[ARMS[0]] & failures[ARMS[1]]), union=len(union), jaccard=len(failures[ARMS[0]] & failures[ARMS[1]])/len(union) if union else None),
            "top5_share_of_positive_official_gain": sum(positive[:5])/sum(positive) if positive else None,
            "transitions": transitions,
        }
        for scene in sorted({r["scene"] for r in rows}):
            sr = [r for r in rows if r["scene"]==scene]
            scenes.append(dict(metric=metric, scene=scene, images=len(sr), official_gain=statistics.mean(r["official_gain_over_spconv"] for r in sr)))
        all_rows.extend(rows)
    curves = {}
    for arm in ARMS:
        history = lines(output / "inputs" / arm / "history.jsonl")
        curves[arm] = [dict(step=r["stage_step"], k0=r["evaluation"]["aggregate"]["k0"]["full_mae"], k3=r["evaluation"]["aggregate"]["k3"]["full_mae"], k5=r["evaluation"]["aggregate"]["k5"]["full_mae"], loss=r["training"]["loss"]) for r in history if r["scope"]=="full"]
    local = [r for r in all_rows if r["metric"]=="local_point_rel"]
    pools = {
        "official_wins": sorted([r for r in local if r["official_gain_over_spconv"]>EPS], key=lambda r:-r["official_gain_over_spconv"]),
        "spconv_wins": sorted([r for r in local if r["official_gain_over_spconv"] < -EPS], key=lambda r:r["official_gain_over_spconv"]),
        "near_tie": sorted(local, key=lambda r:abs(r["official_gain_over_spconv"])),
        "both_worse": sorted([r for r in local if r["spconv_gain"] < -EPS and r["official_flex_gain"] < -EPS], key=lambda r:min(r["spconv_gain"],r["official_flex_gain"])),
    }
    selected, used, used_scenes = [], set(), set()
    for category, pool in pools.items():
        chosen = []
        for allow_scene in (False, True):
            for row in pool:
                if len(chosen)==3:
                    break
                if row["sample_id"] in used or (not allow_scene and row["scene"] in used_scenes):
                    continue
                chosen.append(dict(category=category, **row))
                used.add(row["sample_id"])
                used_scenes.add(row["scene"])
        selected.extend(chosen)
    for name, rows in (("per_image.csv",all_rows),("per_scene.csv",scenes)):
        with (output/name).open("w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(output/"summary.json", dict(metrics=summary, curves=curves, paired_input_checks="passed", sample_count=100))
    write(output/"selection.json", dict(metric="local_point_rel", diagnostic_only=True, samples=selected))
    write(output/"provenance.json", dict(code_sha256=sha(__file__), config_sha256=sha(args.config), source_files={str(p.relative_to(output)):sha(p) for p in sorted((output/"inputs").rglob("*.json*"))}))
    print(json.dumps({m:summary[m]["official_win_tie_loss"] for m in METRICS}, indent=2))
    print("selected", len(selected), flush=True)


def pixel_stats(sequence, target, valid, local, depth):
    import numpy as np
    if not valid.any() or not np.isfinite(sequence).all():
        raise ValueError("Empty GT or non-finite prediction")
    cut = np.quantile(depth[valid], [1/3,2/3])
    regions = {"full":valid, "local":valid & local, "complement":valid & ~local,
               "near":valid & (depth<=cut[0]), "middle":valid & (depth>cut[0]) & (depth<=cut[1]), "far":valid & (depth>cut[1])}
    errors = np.abs(sequence-target)
    result = dict(depth_tertiles_m=cut.tolist(), regions={})
    total = int(valid.sum())
    for name, mask in regions.items():
        count = int(mask.sum())
        if not count:
            result["regions"][name] = {"pixels":0, "fraction":0., "mae":None, "contribution":0., "updates":None}
            continue
        region = dict(pixels=count, fraction=count/total,
                      mae=[float(np.mean(e[mask],dtype=np.float64)) for e in errors],
                      contribution=float(np.sum((errors[0]-errors[3])[mask],dtype=np.float64)/total),
                      k3_improved_pixels=float(((errors[0]-errors[3])[mask]>EPS).mean()),
                      k3_degraded_pixels=float(((errors[3]-errors[0])[mask]>EPS).mean()), updates={})
        for k in range(len(sequence)-1):
            e=(target-sequence[k])[mask]
            u=(sequence[k+1]-sequence[k])[mask]
            nonzero=np.abs(e)>EPS
            gain=np.abs(e)-np.abs(e-u)
            crossed=e*(e-u)<0
            region["updates"][str(k+1)] = dict(
                wrong_direction=float((e*u<0).mean()), crossed_gt=float(crossed.mean()),
                harmful_crossing=float((crossed & (gain < -EPS)).mean()),
                improved=float((gain>EPS).mean()), degraded=float((gain < -EPS).mean()),
                mean_abs_update=float(np.abs(u).mean()), mean_abs_required=float(np.abs(e).mean()),
                near_limit_fraction=float((np.abs(u)>=.095).mean()),
                ratio_excluded=int((~nonzero).sum()),
                update_error_ratio_quantiles=np.quantile(np.abs(u[nonzero])/np.abs(e[nonzero]),[.1,.5,.9]).tolist() if nonzero.any() else None)
        result["regions"][name] = region
    reconstructed=result["regions"]["local"]["contribution"]+result["regions"]["complement"]["contribution"]
    if not math.isclose(reconstructed, result["regions"]["full"]["contribution"], abs_tol=1e-9):
        raise ValueError("Regional MAE contributions do not sum")
    return result


def infer(args):
    import os
    import numpy as np
    import torch
    sys.path.insert(0,str(args.project_root.resolve()))
    from experiment.exp6_4_official_ssr_disparity_adapter.evaluate_pair import load_protocol
    from training.disparity_refiner.train import _load_samples, _refinement_monitor
    from InfiniDepth.model import InfiniDepth
    from training.disparity_refiner.export_assets import load_refiner_checkpoint
    if not args.output.resolve().is_relative_to(args.experiment.resolve()) or not args.output.name.startswith("analysis_"):
        raise ValueError("Inference output must be an isolated experiment analysis directory")
    config_path=args.experiment/"arms"/args.arm/"config.json"
    config=read(config_path)
    checkpoint=args.experiment/"runs"/args.arm/"checkpoints/last.pt"
    reference=indexed(lines(args.output/"inputs"/args.arm/"per_image.jsonl"))
    ref_provenance=read(args.output/"inputs"/args.arm/"provenance.json")
    ckpt_hash=sha(checkpoint)
    if ckpt_hash!=ref_provenance["checkpoint_sha256"] or sha(config_path)!=ref_provenance["config_sha256"]:
        raise ValueError("Checkpoint/config do not match original terminal evaluation")
    metadata=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if metadata["stage_step"]!=20000 or metadata["total_step"]!=20000 or metadata["refiner_config"]["backend"]!=args.arm:
        raise ValueError("Expected the matching backend at terminal step 20,000")
    frozen_hash=metadata["frozen_base_sha256"]
    del metadata
    protocol,masks=load_protocol()
    run_output=args.output/"inference"/f"alpha{args.alpha:g}"/args.arm
    provenance=dict(checkpoint_sha256=ckpt_hash,config_sha256=sha(config_path),code_sha256=sha(__file__),
                    alpha=args.alpha,arm=args.arm,mask_sha256=sha(masks/"manifest.json"),
                    protocol_sha256=sha(protocol.__file__),selection_sha256=sha(args.output/"selection.json"),
                    model_sources={p:sha(args.project_root/p) for p in ("InfiniDepth/model/model.py","InfiniDepth/model/disparity_refiner.py","InfiniDepth/model/official_disparity_adapter.py")},
                    frozen_base_sha256=frozen_hash,stage_step=20000,
                    mae_recheck_tolerance={"k0":1e-6,"refined":1e-4},
                    initial_flex_cache_sha256=sha(args.output/"inputs/flex_gemm_autotune_cache.json"),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),torch=torch.__version__)
    if (run_output/"provenance.json").exists() and read(run_output/"provenance.json")!=provenance:
        raise ValueError("Analysis resume provenance changed; create a new revision")
    write(run_output/"provenance.json",provenance)
    samples=_load_samples(config,config["runs"][0],args.project_root,split="val",sample_ids=config["evaluation"]["full_sample_ids"])
    # Existing personal cache only; masks retain the audited Exp6-3 definition.
    all_samples=[samples[i] for i in range(len(samples))]
    local_assets=protocol.load_local_assets(masks,all_samples)
    chosen={r["sample_id"] for r in read(args.output/"selection.json")["samples"]}
    torch.cuda.set_device(0)
    model=InfiniDepth(model_path=config["model"]["checkpoint"]).cuda()
    model.attach_disparity_refiner(backend=args.arm,voxel_resolution=config["model"]["voxel_resolution"])
    load_refiner_checkpoint(model,checkpoint)
    model.eval()
    records_path=run_output/"per_image.jsonl"
    existing=indexed(lines(records_path)) if records_path.exists() else {}
    if not set(existing).issubset(reference):
        raise ValueError("Unexpected resume sample IDs")
    limit=args.limit or len(all_samples)
    for sample in all_samples[:limit]:
        sid=sample.sample_id
        if sid in existing:
            continue
        if protocol.image_hash(sample)!=reference[sid]["input_tensor_sha256"]:
            raise ValueError(f"Input tensor differs: {sid}")
        started=time.monotonic()
        with torch.no_grad():
            output=model.forward_dense_refined(sample.image[None].cuda(),query_hw=(config["model"]["height"],config["model"]["width"]),num_refinement_steps=5,chunk_size=config["model"]["query_chunk_size"],residual_scale=args.alpha)
            monitor=_refinement_monitor(output)
            sequence=np.stack([d[0].float().cpu().numpy() for d in output.disparity_sequence])
            target=sample.target_disparity.numpy()
            valid=sample.valid_mask.numpy().astype(bool)
            local=local_assets[sid]["local_mask"].numpy()
            stats=pixel_stats(sequence,target,valid,local,sample.radial_depth.numpy())
            checks=(0,1,3,5) if args.alpha==1 else (0,)
            diffs={str(k):stats["regions"]["full"]["mae"][k]-reference[sid]["metrics"][f"k{k}"]["native_disparity_mae"] for k in checks}
            if any(abs(v)>(1e-6 if k=="0" else 1e-4) for k,v in diffs.items()):
                raise ValueError(f"Inference differs from original evaluation: {sid}: {diffs}")
            geometry={}
            if args.alpha!=1:
                gt,rays=protocol.gt_for_sample(sample,torch.device("cuda:0"))
                gt["local_mask"]=local_assets[sid]["local_mask"].cuda()
                gt["local_segmentation"]=local_assets[sid]["local_segmentation"].cuda()
                low,high=sample.disparity_quantiles
                for k in STEPS:
                    raw=output.disparity_sequence[k][0].float()*(high-low)+low
                    if not bool((torch.isfinite(raw)&(raw>1e-6))[sample.valid_mask.cuda()].all()):
                        raise ValueError("Invalid radial reconstruction in intervention")
                    depth=raw.reciprocal()
                    geometry[f"k{k}"]=protocol.make_prediction_metrics(depth,rays*depth[...,None],gt,sharp_boundary=True)
                    geometry[f"k{k}"]["native_disparity_mae"]=stats["regions"]["full"]["mae"][k]
            row=dict(sample_id=sid,arm=args.arm,alpha=args.alpha,disparity_quantiles=list(sample.disparity_quantiles),
                     stats=stats,refinement_monitor=monitor,reference_mae_difference=diffs,metrics=geometry,
                     elapsed_seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
            if sid in chosen:
                path=run_output/"selected_arrays"/f"{sid}.npz"
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.with_suffix(".tmp").open("wb") as f:
                    np.savez_compressed(f,sequence=sequence,target=target,valid=valid,local=local,rgb=sample.image.permute(1,2,0).numpy())
                path.with_suffix(".tmp").replace(path)
            existing[sid]=row
            tmp=records_path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in existing.values()))
            tmp.replace(records_path)
            write(run_output/"status.json",dict(completed=len(existing),total=len(all_samples),status="completed" if len(existing)==len(all_samples) else "running",last_sample=sid))
            print(json.dumps(dict(arm=args.arm,alpha=args.alpha,count=len(existing),sample_id=sid,seconds=round(row["elapsed_seconds"],2),max_mae_difference=max(abs(v) for v in diffs.values()))),flush=True)
    if args.limit:
        print("Smoke gate passed; same provenance permits full continuation",flush=True)


def plots(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=args.output
    folder=out/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    summary=read(out/"summary.json")
    rows=list(csv.DictReader((out/"per_image.csv").open()))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axes,METRICS[:3]):
        values=sorted(float(r["official_minus_spconv"]) for r in rows if r["metric"]==metric)
        ax.bar(range(len(values)),values,color=["#238b8d" if v<0 else "#c76943" for v in values])
        ax.axhline(0,color="black",lw=.6)
        ax.set(title=metric,xlabel="Images sorted by difference",ylabel="official - spconv (lower better)")
    fig.tight_layout(); fig.savefig(folder/"paired_differences.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm in ARMS:
        curve=summary["curves"][arm]
        axes[0].plot([r["step"] for r in curve],[r["k3"] for r in curve],marker="o",label=arm)
        data=read(out/"inputs"/arm/"summary.json")["aggregate"]
        axes[1].plot(STEPS,[data[f"k{k}"]["local_point_rel"] for k in STEPS],marker="o",label=arm)
    axes[0].set(title="Val100 K3 native disparity MAE",xlabel="Training step",ylabel="MAE")
    axes[1].set(title="Local Point Rel (99 images)",xlabel="Refinement K",ylabel="Rel")
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/"curves.png",dpi=160); plt.close(fig)
    region_summary={}
    for alpha_dir in sorted((out/"inference").glob("alpha*")):
        region_summary[alpha_dir.name]={}
        for arm in ARMS:
            path=alpha_dir/arm/"per_image.jsonl"
            if not path.exists(): continue
            data=lines(path)
            regions={}
            for name in ("full","local","complement","near","middle","far"):
                rr=[r["stats"]["regions"][name] for r in data if r["stats"]["regions"][name]["pixels"]]
                means={key:statistics.mean(r[key] for r in rr) for key in ("fraction","contribution","k3_improved_pixels","k3_degraded_pixels")} if rr else {}
                # Empty regions contribute zero to image-macro full-MAE accounting.
                for key in ("fraction","contribution"):
                    means[key]=statistics.mean(r["stats"]["regions"][name][key] for r in data)
                means["contribution_denominator_images"]=len(data)
                means["mae"]=[statistics.mean(r["mae"][k] for r in rr) for k in range(6)] if rr else None
                means["images"]=len(rr)
                means["updates"]={str(k):{key:statistics.mean(r["updates"][str(k)][key] for r in rr) for key in ("wrong_direction","crossed_gt","harmful_crossing","improved","degraded","mean_abs_update","mean_abs_required","near_limit_fraction")} for k in range(1,6)} if rr else {}
                regions[name]=means
            region_summary[alpha_dir.name][arm]=dict(images=len(data),regions=regions,
                inference_seconds=sum(r["elapsed_seconds"] for r in data),steady_state_seconds=distribution([r["elapsed_seconds"] for r in data[2:]]),
                peak_cuda_memory_bytes=max(r["peak_cuda_memory_bytes"] for r in data),
                max_reference_mae_difference=max(abs(v) for r in data for v in r["reference_mae_difference"].values()),
                aggregate_metrics={f"k{k}":{m:statistics.mean(r["metrics"][f"k{k}"][m] for r in data if r["metrics"][f"k{k}"].get(m) is not None) for m in METRICS} for k in STEPS} if data[0]["metrics"] else {})
    write(out/"pixel_summary.json",region_summary)
    baseline=region_summary.get("alpha1",{})
    if len(baseline)==2:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for arm in ARMS:
            r=baseline[arm]["regions"]
            axes[0].plot(range(1,6),[100*r["full"]["updates"][str(k)]["wrong_direction"] for k in range(1,6)],marker="o",label=arm)
            axes[1].plot(range(1,6),[100*r["full"]["updates"][str(k)]["harmful_crossing"] for k in range(1,6)],marker="o",label=arm)
            axes[2].plot(range(6),r["local"]["mae"],marker="o",label=arm)
        axes[0].set(title="Update opposite to native GT correction",xlabel="Update number",ylabel="Pixels (%)")
        axes[1].set(title="Crossing GT and increasing error",xlabel="Update number",ylabel="Pixels (%)")
        axes[2].set(title="Native MAE inside Local mask (99 images)",xlabel="K",ylabel="MAE (not official Local Point Rel)")
        for ax in axes: ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder/"update_diagnostics.png",dpi=160); plt.close(fig)
    for selected in read(out/"selection.json")["samples"]:
        sid=selected["sample_id"]
        files=[out/"inference/alpha1"/a/"selected_arrays"/f"{sid}.npz" for a in ARMS]
        if not all(p.exists() for p in files): continue
        a,b=(np.load(p) for p in files)
        valid,target=a["valid"],a["target"]
        da,db=a["sequence"],b["sequence"]
        ea,eb,e0=np.abs(da[3]-target),np.abs(db[3]-target),np.abs(da[0]-target)
        error_limit=float(np.quantile(np.concatenate([e0[valid],ea[valid],eb[valid]]),.98))
        gain_limit=float(np.quantile(np.abs(np.concatenate([(e0-ea)[valid],(e0-eb)[valid],(ea-eb)[valid]])),.99))
        fig,axes=plt.subplots(4,4,figsize=(16,12))
        def draw(ax,value,title,kind="disparity"):
            ax.set_title(title,fontsize=9); ax.axis("off")
            if kind=="rgb": ax.imshow(value); return
            kw=dict(cmap="viridis",vmin=0,vmax=1)
            if kind=="error": kw=dict(cmap="magma",vmin=0,vmax=error_limit)
            if kind=="gain": kw=dict(cmap="RdBu",vmin=-gain_limit,vmax=gain_limit)
            if kind=="mask": kw=dict(cmap="gray",vmin=0,vmax=1)
            im=ax.imshow(np.ma.array(value,mask=~valid),**kw)
            fig.colorbar(im,ax=ax,fraction=.035,pad=.02)
        draw(axes[0,0],a["rgb"],"RGB","rgb")
        draw(axes[0,1],target,"GT normalized disparity (display clipped [0,1])")
        draw(axes[0,2],a["rgb"],"RGB + fixed Local mask outline","rgb")
        axes[0,2].contour(a["local"].astype(float),levels=[.5],colors=["#ff3838"],linewidths=.7)
        draw(axes[0,3],da[0],"Common K0")
        for row,seq,label,err in ((1,da,"spconv",ea),(2,db,"official",eb)):
            for col,k in enumerate((1,3,5)): draw(axes[row,col],seq[k],f"{label} K{k}")
            draw(axes[row,3],err,f"{label} K3 absolute error","error")
        draw(axes[3,0],e0,"K0 absolute error","error")
        draw(axes[3,1],e0-ea,"spconv gain over K0 (blue better)","gain")
        draw(axes[3,2],e0-eb,"official gain over K0 (blue better)","gain")
        draw(axes[3,3],ea-eb,"official gain over spconv (blue better)","gain")
        fig.suptitle(f"{selected['category']} | {sid}\nNative disparity maps; shared color scales, error p98 / gain p99 clipping",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(folder/f"{sid}.png",dpi=130); plt.close(fig)
        a.close(); b.close()
    print("plots and pixel_summary complete",flush=True)


def audit(args):
    import html
    out=args.output
    pixel=read(out/"pixel_summary.json")["alpha1"]
    details={}
    base_hashes=[]
    expected=set(indexed(lines(out/"inputs/spconv/per_image.jsonl")))
    def corr(x,y):
        mx,my=statistics.mean(x),statistics.mean(y)
        covariance=sum((a-mx)*(b-my) for a,b in zip(x,y))
        denom=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
        return covariance/denom if denom else None
    for arm in ARMS:
        root=out/"inference/alpha1"/arm
        records=indexed(lines(root/"per_image.jsonl"))
        assert set(records)==expected and len(records)==100
        assert read(root/"status.json")["status"]=="completed"
        prov=read(root/"provenance.json")
        assert sha(out/"analyze_results.py")==prov["code_sha256"]
        assert sha(out/"selection.json")==prov["selection_sha256"]
        reference_prov=read(out/"inputs"/arm/"provenance.json")
        for key in ("checkpoint_sha256", "config_sha256"):
            assert prov[key]==reference_prov[key]
        base_hashes.append(prov["frozen_base_sha256"])
        for r in records.values():
            regions=r["stats"]["regions"]
            assert math.isclose(regions["local"]["contribution"]+regions["complement"]["contribution"],regions["full"]["contribution"],abs_tol=1e-9)
            assert all(math.isfinite(v) for v in r["refinement_monitor"].values())
            for k in range(1,6):
                assert max(abs(r["refinement_monitor"][f"k{k}_bounded_residual_{s}"]) for s in ("min","max"))<=.1000001
        rr=pixel[arm]["regions"]
        assert math.isclose(rr["local"]["contribution"]+rr["complement"]["contribution"],rr["full"]["contribution"],abs_tol=1e-9)
        assert math.isclose(rr["local"]["fraction"]+rr["complement"]["fraction"],1.,abs_tol=1e-9)
        def gain(r):
            v=r["stats"]["regions"]["full"]["mae"]
            return v[0]-v[3]
        groups={}
        for label,predicate in (("improved",lambda r:gain(r)>EPS),("degraded",lambda r:gain(r)<-EPS)):
            group=[r for r in records.values() if predicate(r)]
            groups[label]=dict(images=len(group),
                k1_wrong_direction=statistics.mean(r["stats"]["regions"]["full"]["updates"]["1"]["wrong_direction"] for r in group),
                k3_wrong_direction=statistics.mean(r["stats"]["regions"]["full"]["updates"]["3"]["wrong_direction"] for r in group),
                k3_harmful_crossing=statistics.mean(r["stats"]["regions"]["full"]["updates"]["3"]["harmful_crossing"] for r in group),
                disparity_span_mean=statistics.mean(r["disparity_quantiles"][1]-r["disparity_quantiles"][0] for r in group))
        data=list(records.values())
        history=lines(out/"inputs"/arm/"history.jsonl")
        for h in history:
            assert all(math.isfinite(v) for v in h["training"].values())
            assert all(v is None or math.isfinite(v) for v in h["gradient_norms"].values())
        monitor_summary={}
        for label, monitors, steps in (("validation",[r["refinement_monitor"] for r in data],range(1,6)),
                                        ("training_logged_samples",[h["training"] for h in history],range(1,4))):
            monitor_summary[label]={str(k):dict(
                raw_min=min(r[f"k{k}_raw_residual_min"] for r in monitors),
                raw_max=max(r[f"k{k}_raw_residual_max"] for r in monitors),
                bounded_max_abs=max(abs(r[f"k{k}_bounded_residual_{s}"]) for r in monitors for s in ("min","max")),
                voxel_span_min=min(r[f"k{k}_disparity_span_max"] for r in monitors),
                voxel_span_max=max(r[f"k{k}_disparity_span_max"] for r in monitors)) for k in steps}
        details[arm]=dict(groups=groups,local_contribution_fraction=rr["local"]["contribution"]/rr["full"]["contribution"],
            residual_monitor=monitor_summary,training_logged_samples=len(history),
            max_recheck_difference_by_k={str(k):max(abs(r["reference_mae_difference"][str(k)]) for r in data) for k in STEPS},
            pearson_exploratory={
                "k3_gain_vs_disparity_span":corr([gain(r) for r in data],[r["disparity_quantiles"][1]-r["disparity_quantiles"][0] for r in data]),
                "k3_gain_vs_k3_wrong_direction":corr([gain(r) for r in data],[r["stats"]["regions"]["full"]["updates"]["3"]["wrong_direction"] for r in data])})
    assert base_hashes[0]==base_hashes[1] and base_hashes[0]
    write(out/"audit.json",dict(status="passed",paired_samples=100,local_nonempty=99,audit_code_sha256=sha(__file__),
          frozen_base_sha256=base_hashes[0],checks=["IDs and completion", "inference code and selection hashes", "checkpoint and config provenance", "per-image and aggregate contribution sums", "region pixel fractions", "common frozen Base", "finite logged training and inference statistics", "bounded inference residuals"],
          details=details,optional_intervention=dict(status="skipped",reason="Wrong-direction updates dominate harmful crossing; amplitude-only hypothesis lacks primary support.")))
    categories={"official_wins":"official_flex 优势样本", "spconv_wins":"spconv 优势样本", "near_tie":"Local 指标接近", "both_worse":"两组 Local 均退化"}
    sections=[]
    for item in read(out/"selection.json")["samples"]:
        sid=item["sample_id"]
        assert (out/"figures"/f"{sid}.png").is_file()
        title=html.escape(categories[item["category"]]+" · "+sid)
        sections.append(f'<section><h2>{title}</h2><a href="figures/{sid}.png"><img loading="lazy" src="figures/{sid}.png" alt="{title}"></a></section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Exp6-4 结果诊断</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#222}img{width:100%;height:auto}section{margin:40px 0}h2{font-size:19px}</style><h1>Exp6-4 结果诊断</h1><p>12 张 Val100 代表图，按 Local Point Rel 固定规则选择，不代表总体胜率。蓝色表示误差减少，红色表示误差增加；图中为原生 disparity 诊断，非对齐后的官方 Local Point Rel。每张图两方法共享色标，误差图 p98、差值图 p99 截断仅用于显示。点击图片查看原图。</p>'
    page+='<p><a href="REPORT.md">完整报告</a> · <a href="per_image.csv">逐图指标</a></p>'
    for name in ("paired_differences","curves","update_diagnostics"):
        page+=f'<section><img src="figures/{name}.png" alt="{name}"></section>'
    page+=''.join(sections)+'</html>'
    (out/"gallery.html").write_text(page)
    write(out/"artifact_manifest.json",{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"})
    print(json.dumps(details,indent=2))


def self_test():
    assert distribution([1,2,3,4])["median"]==2.5
    assert outcome(direction("local_point_delta_0.01")*(.5-.6))=="win"
    assert outcome(direction("native_disparity_mae")*(.5-.6))=="loss"
    assert distribution([])["mean"] is None
    try:
        indexed([dict(sample_id="x"),dict(sample_id="x")])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate IDs accepted")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="exp6_analysis_check_") as folder:
        out=Path(folder)
        for arm in ARMS:
            path=out/"inputs"/arm/"per_image.jsonl"
            write(path,dict(sample_id="a" if arm==ARMS[0] else "b"))
            # A one-record JSONL must be one line.
            path.write_text(json.dumps(read(path))+"\n")
        try:
            paired_inputs(out,{"evaluation":{"full_sample_ids":[str(i) for i in range(100)]}})
        except ValueError as exc:
            assert "Val100 IDs" in str(exc)
        else:
            raise AssertionError("Mismatched IDs accepted")
    print("standard-library checks passed")
    try:
        import numpy as np
    except ImportError:
        print("Pixel checks require the existing server NumPy environment")
        return
    seq=np.array([[[0.,0.]],[[1.5,.5]],[[1.2,.7]],[[1.1,.9]]])
    stats=pixel_stats(seq,np.ones((1,2)),np.ones((1,2),bool),np.array([[True,False]]),np.array([[1.,2.]]))
    first=stats["regions"]["full"]["updates"]["1"]
    assert first["crossed_gt"]==.5 and first["harmful_crossing"]==0 and first["improved"]==1
    assert math.isclose(sum(stats["regions"][r]["contribution"] for r in ("local","complement")),.9)
    empty=pixel_stats(seq,np.ones((1,2)),np.ones((1,2),bool),np.zeros((1,2),bool),np.ones((1,2)))
    assert empty["regions"]["local"]["mae"] is None
    print("pixel contribution, empty mask and harmless crossing checks passed")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command", choices=("summary","self-test","infer","plots","audit"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--alpha", type=float, choices=(1.0,0.5), default=1.0)
    parser.add_argument("--limit", type=int)
    args=parser.parse_args()
    if args.command=="self-test":
        self_test()
    elif args.command=="summary":
        summarize(args)
    elif args.command=="infer":
        infer(args)
    elif args.command=="audit":
        audit(args)
    else:
        plots(args)


if __name__=="__main__":
    main()

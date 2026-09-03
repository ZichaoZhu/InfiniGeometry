from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

from PIL import Image

from training.disparity_refiner.eth3d import center_crop_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a static ETH3D image selection gallery")
    parser.add_argument("--experiment", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gallery_html() -> str:
    return """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETH3D Exp5 选图</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#071017;color:#e7f1f3;font:14px system-ui,sans-serif}.top{position:sticky;top:0;z-index:2;padding:18px;background:#0c1b22eF;border-bottom:1px solid #28434b}h1{margin:0 0 6px;font-size:24px}p{margin:4px 0;color:#aabdc2}.controls{display:flex;flex-wrap:wrap;gap:9px;margin-top:13px}input,select,button{font:inherit;border:1px solid #42636a;border-radius:7px;background:#10262d;color:#e7f1f3;padding:8px}input{min-width:210px}button{cursor:pointer;background:#46d5cb;color:#062127;font-weight:700}.summary{margin-top:10px;color:#79e7dc}.warning{color:#f3c77a}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:12px;padding:14px}.card{border:1px solid #26454d;border-radius:10px;background:#0d1e25;padding:9px}.card.selected{border-color:#67eee4;box-shadow:0 0 0 1px #67eee4}.thumb{display:block;width:100%;border-radius:6px;cursor:pointer}.meta{display:grid;gap:4px;margin:8px 0;color:#b8c8cc;font-size:12px;overflow-wrap:anywhere}.row{display:flex;justify-content:space-between;gap:8px;align-items:center}.row a{color:#78e5dc}.row button{padding:6px 9px}.hidden{display:none}#output{white-space:pre-wrap;color:#b8c8cc;margin:8px 0 0}
</style><body><section class="top"><h1>ETH3D Exp5 · 选图</h1><p>这是 ETH3D high-res training DSLR 的 454 张、与正式评测完全一致的 512×384 中心裁剪 RGB 预览；不运行模型，不新增或显示逐图指标。</p><p>点击图片或“选择”勾选，最多选择 10 张。复制样本 ID 后发给我，即可导出对应的 K0/K1/K3/K5 点云。</p><div class="controls"><input id="search" placeholder="搜索序号、场景或样本 ID"><select id="scene"><option value="">全部场景</option></select><button id="copy">复制已选样本 ID</button><button id="clear">清空选择</button></div><div class="summary" id="summary">读取中…</div><div class="warning" id="warning"></div><div id="output"></div></section><main class="grid" id="grid"></main>
<script>
const maxSelection=10;const selected=new Set(JSON.parse(localStorage.getItem('eth3d-exp5-gallery-selection')||'[]'));let samples=[];
const byId=id=>document.getElementById(id);const values=key=>[...new Set(samples.map(x=>x[key]))].sort();
function addOptions(id,key){for(const value of values(key)){const option=document.createElement('option');option.value=value;option.textContent=value;byId(id).append(option)}}
function save(){localStorage.setItem('eth3d-exp5-gallery-selection',JSON.stringify([...selected]));}
function matches(x){const query=byId('search').value.trim().toLowerCase();return (!query||[x.order,x.id,x.scene].join(' ').toLowerCase().includes(query))&&(!byId('scene').value||x.scene===byId('scene').value)}
function update(){let visible=0;for(const card of document.querySelectorAll('.card')){const x=samples[Number(card.dataset.index)];const show=matches(x);card.classList.toggle('hidden',!show);card.classList.toggle('selected',selected.has(x.id));if(show)visible++}byId('summary').textContent=`显示 ${visible}/${samples.length} 张；已选择 ${selected.size}/${maxSelection} 张。`;}
function toggle(x){if(!selected.has(x.id)&&selected.size>=maxSelection){byId('warning').textContent=`最多选择 ${maxSelection} 张，请先取消已选图片。`;return}selected.has(x.id)?selected.delete(x.id):selected.add(x.id);byId('warning').textContent='';save();update()}
function card(x,index){const article=document.createElement('article');article.className='card';article.dataset.index=index;const image=document.createElement('img');image.className='thumb';image.loading='lazy';image.src=x.previewUrl;image.alt=`#${x.order} ${x.scene} ${x.id}`;image.onclick=()=>toggle(x);const meta=document.createElement('div');meta.className='meta';for(const text of [`#${String(x.order).padStart(3,'0')} · ${x.scene}`,x.id,`有效深度像素 ${x.valid_pixel_count.toLocaleString()} · held-out ${x.held_out_pixel_count.toLocaleString()}`]){const line=document.createElement('div');line.textContent=text;meta.append(line)}const row=document.createElement('div');row.className='row';const link=document.createElement('a');link.href=x.previewUrl;link.target='_blank';link.textContent='查看 512×384';const button=document.createElement('button');button.textContent='选择';button.onclick=()=>toggle(x);row.append(link,button);article.append(image,meta,row);return article}
for(const id of ['search','scene'])byId(id).oninput=update;
byId('clear').onclick=()=>{selected.clear();byId('warning').textContent='';save();update()};byId('copy').onclick=async()=>{const list=samples.filter(x=>selected.has(x.id)).map(x=>x.id).join(String.fromCharCode(10));byId('output').textContent=list||'尚未选择图片。';if(list&&navigator.clipboard)await navigator.clipboard.writeText(list)};
fetch('gallery.json').then(r=>r.json()).then(data=>{samples=data.samples;addOptions('scene','scene');const grid=byId('grid');samples.forEach((x,i)=>grid.append(card(x,i)));update()});
</script></body></html>"""


def save_preview(entry: Mapping[str, object], path: Path, *, output_hw: tuple[int, int], quality: int) -> tuple[int, int, int, int]:
    camera = entry["camera"]
    if not isinstance(camera, Mapping):
        raise ValueError(f"ETH3D sample {entry.get('id')} has no camera")
    width, height = int(camera["width"]), int(camera["height"])
    with Image.open(Path(str(entry["rgb_path"]))) as source:
        image = source.convert("RGB")
        if image.size != (width, height):
            raise ValueError(f"ETH3D calibration/image size mismatch for {entry.get('id')}")
        crop_box = center_crop_box((height, width), output_hw)
        preview = image.crop(crop_box).resize((output_hw[1], output_hw[0]), Image.Resampling.LANCZOS)
    preview.save(path, format="JPEG", quality=quality, optimize=True)
    return crop_box


def main() -> None:
    from training.disparity_refiner.data import ensure_within

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config = load_json(experiment / "config.json")
    gallery_config = load_json(experiment / "gallery.json")
    if config.get("experiment_id") != "exp5_ETH3D":
        raise ValueError("Unexpected Exp5 ETH3D config")
    if gallery_config.get("format") != "infinidepth-exp5-eth3d-gallery-v1":
        raise ValueError("Unexpected ETH3D gallery config")
    if int(gallery_config.get("selection_limit", 0)) != 10:
        raise ValueError("ETH3D gallery selection limit must be 10")
    safe_root = Path(str(config["server"]["safe_root"])).resolve()
    input_path = ensure_within(Path(str(config["data"]["input_manifest"])), safe_root, name="ETH3D input manifest")
    report_path = ensure_within(Path(str(gallery_config["report"])), safe_root, name="ETH3D report")
    inputs = load_json(input_path)
    report = load_json(report_path)
    samples = inputs.get("samples")
    per_image = report.get("per_image")
    if not isinstance(samples, list) or not isinstance(per_image, Mapping):
        raise ValueError("Invalid ETH3D manifest or report")
    if len(samples) != int(config["data"]["expected_sample_count"]) or len(per_image) != len(samples):
        raise ValueError("ETH3D report is incomplete")
    sample_ids = [str(entry["id"]) for entry in samples]
    if set(sample_ids) != set(per_image):
        raise ValueError("ETH3D report sample IDs do not match the input manifest")
    preview = gallery_config["preview"]
    output_hw = (int(preview["height"]), int(preview["width"]))
    quality = int(preview["jpeg_quality"])
    if min(*output_hw, quality) <= 0 or quality > 100:
        raise ValueError("Invalid ETH3D preview configuration")
    public_root = project_root / "experiment" / "viewer" / "public"
    output = public_root / "data" / str(gallery_config["asset_tag"])
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        gallery_samples = []
        for order, entry in enumerate(samples, start=1):
            sample_id = str(entry["id"])
            record = per_image[sample_id]
            preview_path = temporary / "previews" / f"{order:03d}.jpg"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            crop_box = save_preview(entry, preview_path, output_hw=output_hw, quality=quality)
            gallery_samples.append(
                {
                    "crop_xyxy": list(crop_box),
                    "held_out_pixel_count": int(record["held_out_pixel_count"]),
                    "id": sample_id,
                    "order": order,
                    "previewUrl": f"previews/{order:03d}.jpg",
                    "scene": str(entry["scene"]),
                    "valid_pixel_count": int(record["valid_pixel_count"]),
                }
            )
            if order % 50 == 0 or order == len(samples):
                print(f"[{order}/{len(samples)}] exported previews", flush=True)
        write_json(
            temporary / "gallery.json",
            {
                "format": "infinidepth-exp5-eth3d-gallery-v1",
                "input_manifest_sha256": sha256(input_path),
                "model_input_resolution": {"height": output_hw[0], "width": output_hw[1]},
                "report_sha256": sha256(report_path),
                "selection_limit": 10,
                "samples": gallery_samples,
                "source": "ETH3D high-res training DSLR inputs at the completed Exp5 evaluation resolution",
                "version": 1,
            },
        )
        (temporary / "index.html").write_text(gallery_html(), encoding="utf-8")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

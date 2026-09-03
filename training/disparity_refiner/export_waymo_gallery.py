from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
from typing import Mapping

from PIL import Image

from training.disparity_refiner.waymo import _reader_modules, list_tfrecords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a static Waymo side-camera image selection gallery")
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
<title>Waymo Val202 SIDE 选图</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#071017;color:#e7f1f3;font:14px system-ui,sans-serif}.top{position:sticky;top:0;z-index:2;padding:18px;background:#0c1b22eF;border-bottom:1px solid #28434b}h1{margin:0 0 6px;font-size:24px}p{margin:4px 0;color:#aabdc2}.controls{display:flex;flex-wrap:wrap;gap:9px;margin-top:13px}input,select,button{font:inherit;border:1px solid #42636a;border-radius:7px;background:#10262d;color:#e7f1f3;padding:8px}input{min-width:210px}button{cursor:pointer;background:#46d5cb;color:#062127;font-weight:700}.summary{margin-top:10px;color:#79e7dc}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:12px;padding:14px}.card{border:1px solid #26454d;border-radius:10px;background:#0d1e25;padding:9px}.card.selected{border-color:#67eee4;box-shadow:0 0 0 1px #67eee4}.thumb{display:block;width:100%;border-radius:6px;cursor:pointer}.meta{display:grid;gap:4px;margin:8px 0;color:#b8c8cc;font-size:12px;overflow-wrap:anywhere}.row{display:flex;justify-content:space-between;gap:8px;align-items:center}.row a{color:#78e5dc}.row button{padding:6px 9px}.hidden{display:none}#output{white-space:pre-wrap;color:#b8c8cc;margin:8px 0 0}
</style><body><section class="top"><h1>Waymo Val202 · SIDE 选图</h1><p>这是与 Exp5 同一批序列的 384×512 SIDE_LEFT / SIDE_RIGHT 图像预览；不运行模型，也不显示新指标。</p><p>点击图片或“选择”勾选；复制 TFRecord 文件名后发给我，并注明相机方向。</p><div class="controls"><input id="search" placeholder="搜索编号、地点、时段、文件名"><select id="camera"><option value="">全部相机</option></select><select id="location"><option value="">全部地点</option></select><select id="time"><option value="">全部时段</option></select><select id="weather"><option value="">全部天气</option></select><button id="copy">复制已选 TFRecord</button><button id="clear">清空选择</button></div><div class="summary" id="summary">读取中…</div><div id="output"></div></section><main class="grid" id="grid"></main>
<script>
const selected=new Set(JSON.parse(localStorage.getItem('waymo-exp5-side-gallery-selection')||'[]'));let samples=[];
const byId=id=>document.getElementById(id);const values=(key)=>[...new Set(samples.map(x=>x[key]))].sort();
function addOptions(id,key){for(const value of values(key)){const option=document.createElement('option');option.value=value;option.textContent=value;byId(id).append(option)}}
function save(){localStorage.setItem('waymo-exp5-side-gallery-selection',JSON.stringify([...selected]));}
function matches(x){const query=byId('search').value.trim().toLowerCase();return (!query||[x.order,x.id,x.source,x.camera,x.location,x.time_of_day,x.weather].join(' ').toLowerCase().includes(query))&&(!byId('camera').value||x.camera===byId('camera').value)&&(!byId('location').value||x.location===byId('location').value)&&(!byId('time').value||x.time_of_day===byId('time').value)&&(!byId('weather').value||x.weather===byId('weather').value)}
function update(){let visible=0;for(const card of document.querySelectorAll('.card')){const x=samples[Number(card.dataset.index)];const show=matches(x);card.classList.toggle('hidden',!show);card.classList.toggle('selected',selected.has(x.id));if(show)visible++}byId('summary').textContent=`显示 ${visible}/${samples.length} 张；已选择 ${selected.size} 张。`}
function toggle(x){selected.has(x.id)?selected.delete(x.id):selected.add(x.id);save();update()}
function card(x,index){const article=document.createElement('article');article.className='card';article.dataset.index=index;const image=document.createElement('img');image.className='thumb';image.loading='lazy';image.src=x.previewUrl;image.alt=`#${x.order} ${x.camera} ${x.location}`;image.onclick=()=>toggle(x);const meta=document.createElement('div');meta.className='meta';for(const text of [`#${String(x.order).padStart(3,'0')} · ${x.camera} · ${x.location} · ${x.time_of_day} · ${x.weather}`,`frame ${x.frame_index} · timestamp ${x.timestamp_micros}`,x.source]){const line=document.createElement('div');line.textContent=text;meta.append(line)}const row=document.createElement('div');row.className='row';const link=document.createElement('a');link.href=x.previewUrl;link.target='_blank';link.textContent='查看 512×384';const button=document.createElement('button');button.textContent='选择';button.onclick=()=>toggle(x);row.append(link,button);article.append(image,meta,row);return article}
for(const id of ['search','camera','location','time','weather'])byId(id).oninput=update;
byId('clear').onclick=()=>{selected.clear();save();update()};byId('copy').onclick=async()=>{const list=samples.filter(x=>selected.has(x.id)).map(x=>`${x.camera}\t${x.source}`).join(String.fromCharCode(10));byId('output').textContent=list||'尚未选择图片。';if(list&&navigator.clipboard)await navigator.clipboard.writeText(list)};
fetch('gallery.json').then(r=>r.json()).then(data=>{samples=data.samples;addOptions('camera','camera');addOptions('location','location');addOptions('time','time_of_day');addOptions('weather','weather');const grid=byId('grid');samples.forEach((x,i)=>grid.append(card(x,i)));update()});
</script></body></html>"""


def read_camera_image(
    path: Path, *, reader_root: Path, frame_index: int, camera_name: str
) -> bytes:
    WaymoDataFileReader, dataset_pb2 = _reader_modules(reader_root)
    camera_id = dataset_pb2.CameraName.Name.Value(camera_name)
    reader = WaymoDataFileReader(str(path))
    try:
        for _ in range(frame_index):
            reader.read_record(header_only=True)
        frame = reader.read_record()
    finally:
        reader.file.close()
    image = next((value for value in frame.images if value.name == camera_id), None)
    if image is None:
        raise ValueError(f"{path.name} frame {frame_index} has no {camera_name} image")
    return bytes(image.image)


def save_preview(image_bytes: bytes, path: Path, *, width: int, height: int, quality: int) -> None:
    with Image.open(io.BytesIO(image_bytes)) as image:
        preview = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, format="JPEG", quality=quality, optimize=True)


def main() -> None:
    from training.disparity_refiner.data import ensure_within
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    experiment = args.experiment.resolve()
    if experiment.parent != project_root / "experiment":
        raise PermissionError("Experiment must be a direct child of experiment/")
    config = load_json(experiment / "config.json")
    gallery_config = load_json(experiment / "gallery.json")
    if config.get("experiment_id") != "exp5_waymo_generalization":
        raise ValueError("Unexpected Exp5 config")
    if str(config["data"]["split"]) != "validation" or str(config["evaluation"]["lidar"]) != "TOP":
        raise ValueError("Waymo gallery is fixed to validation TOP LiDAR data")
    if gallery_config.get("format") != "infinidepth-exp5-waymo-gallery-v1":
        raise ValueError("Unexpected Waymo gallery config")
    safe_root = Path(str(config["server"]["safe_root"]))
    report_path = ensure_within(Path(str(gallery_config["report"])), safe_root, name="Waymo report")
    report = load_json(report_path)
    if report.get("format") != "infinidepth-exp5-waymo-evaluation-v1":
        raise ValueError("Unexpected Waymo evaluation report")
    per_image = report["per_image"]
    if len(per_image) != int(config["data"]["expected_count"]):
        raise ValueError("Waymo report is incomplete")
    data_root = Path(str(config["data"]["root"])).resolve()
    records = {path.name: path for path in list_tfrecords(data_root, expected_count=len(per_image))}
    if set(records) != set(per_image):
        raise ValueError("Waymo report sources do not match the validation TFRecords")
    reader_root = ensure_within(Path(str(config["data"]["reader_root"])), safe_root, name="Waymo reader")
    preview = gallery_config["preview"]
    width, height = int(preview["width"]), int(preview["height"])
    quality = int(preview["jpeg_quality"])
    if min(width, height, quality) <= 0 or quality > 100:
        raise ValueError("Invalid preview configuration")
    public_root = project_root / "experiment" / "viewer" / "public"
    output = public_root / "data" / str(gallery_config["asset_tag"])
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    cameras = [str(value) for value in gallery_config.get("cameras", [])]
    if set(cameras) != {"SIDE_LEFT", "SIDE_RIGHT"}:
        raise ValueError("Side gallery must contain SIDE_LEFT and SIDE_RIGHT")
    sources = sorted(per_image)
    random.Random(int(gallery_config["seed"])).shuffle(sources)
    samples = []
    try:
        order = 0
        for source in sources:
            item = per_image[source]
            metadata = item["metadata"]
            if str(metadata["camera"]) != "FRONT":
                raise ValueError(f"Unexpected camera in {source}")
            for camera_name in cameras:
                order += 1
                preview_path = temporary / "previews" / f"{order:03d}.jpg"
                save_preview(
                    read_camera_image(
                        records[source],
                        reader_root=reader_root,
                        frame_index=int(metadata["frame_index"]),
                        camera_name=camera_name,
                    ),
                    preview_path,
                    width=width,
                    height=height,
                    quality=quality,
                )
                samples.append(
                    {
                    "camera": camera_name,
                    "frame_index": int(metadata["frame_index"]),
                    "id": f"{source}:{metadata['frame_index']}:{camera_name}",
                    "location": str(metadata["location"]),
                    "order": order,
                    "previewUrl": f"previews/{order:03d}.jpg",
                    "source": source,
                    "time_of_day": str(metadata["time_of_day"]),
                    "timestamp_micros": int(metadata["timestamp_micros"]),
                    "weather": str(metadata["weather"]),
                    }
                )
            if order % 40 == 0 or order == len(sources) * len(cameras):
                print(f"[{order}/{len(sources) * len(cameras)}] exported previews", flush=True)
        write_json(
            temporary / "gallery.json",
            {
                "format": "infinidepth-exp5-waymo-gallery-v1",
                "model_input_resolution": {"height": height, "width": width},
                "report_sha256": sha256(report_path),
                "cameras": cameras,
                "samples": samples,
                "seed": int(gallery_config["seed"]),
                "source": "Waymo validation SIDE_LEFT and SIDE_RIGHT frames at the Exp5 Val202 frame indices",
                "version": 1,
            },
        )
        (temporary / "index.html").write_text(gallery_html(), encoding="utf-8")
        temporary.replace(output)
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise


if __name__ == "__main__":
    main()

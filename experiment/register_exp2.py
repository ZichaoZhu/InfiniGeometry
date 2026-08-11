from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


EXP2_NAME = "exp2_infinidepth_disparity_ssr_hypersim100_overfit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp1 通过后登记百图训练域 Exp2")
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2), required=True)
    return parser.parse_args()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    exp1 = root / "exp1_infinidepth_disparity_ssr_single_image_overfit"
    exp1_report = json.loads((exp1 / "metrics" / "report.json").read_text(encoding="utf-8"))
    if exp1_report.get("status") != "completed" or exp1_report.get("gate_passed") is not True:
        raise RuntimeError("Exp1 尚未完成 3/5 门槛，禁止登记 Exp2")
    target = root / EXP2_NAME
    if target.exists():
        raise FileExistsError(f"Exp2 已存在，拒绝覆盖: {target}")

    exp1_config = json.loads((exp1 / "config.json").read_text(encoding="utf-8"))
    config = deepcopy(exp1_config)
    config.update(
        {
            "claim_boundary": "仅验证固定 100 张训练图上的训练域拟合能力，不建立 validation/test 或泛化结论。",
            "experiment_id": EXP2_NAME,
            "research_question": "单图验证通过后，固定网格 disparity Refiner 能否扩展到 100 张 Hypersim 训练图？",
            "runs": [
                {
                    "directory": "runs/main",
                    "id": "main",
                    "sample_ids": None,
                    "seed_offset": 0,
                }
            ],
            "seed": 131,
            "status": "planned",
        }
    )
    config["data"]["expected_train_count"] = 100
    config["evaluation"].update(
        {
            "gate_improvement_fraction": 0.01,
            "gate_minimum_successes": 1,
            "minimum_improved_images": 80,
            "selection_metric": "k3_mean_full_disparity_mae_plus_locked_five_structure_disparity_mae",
        }
    )
    config["training"].update(
        {
            "global_batch_size": 8,
            "microbatch_size": args.microbatch_size,
            "stages": {
                "stage1": {
                    "checkpoint_every": 2500,
                    "dino_freeze_steps": 1000,
                    "dino_warmup_end": 2000,
                    "eval_every": 500,
                    "full_eval_every": 2500,
                    "learning_rates": {"dino": 5e-8, "head": 1e-6, "ssr": 1e-5},
                    "max_steps": 20000,
                    "min_steps": 20000,
                    "plateau_patience_evals": 0,
                    "plateau_relative_improvement": 0.0,
                },
                "joint": {
                    "checkpoint_every": 2500,
                    "dino_freeze_steps": 0,
                    "dino_warmup_end": 0,
                    "eval_every": 500,
                    "full_eval_every": 2500,
                    "learning_rates": {"dino": 1e-8, "head": 2.5e-7, "ssr": 1e-6},
                    "max_steps": 10000,
                    "min_steps": 10000,
                    "plateau_patience_evals": 0,
                    "plateau_relative_improvement": 0.0,
                },
            },
        }
    )

    write_json(target / "config.json", config)
    write_text(
        target / "README.md",
        """
# Exp2：InfiniDepth 原生 Disparity Refiner 百图训练域拟合

## 前置条件与边界

本目录由 `experiment/register_exp2.py` 在 Exp1 五个运行全部结束且至少 3/5 通过 1% 门槛后生成。实验只使用 MoGe-3 Exp30 对应清单中的固定 100 张 `train` 图片，不读取 val/test 形成结论，也不宣称泛化。

## 训练

阶段一执行 20,000 个 detached optimizer step；阶段二继续 10,000 个 joint step。全局 batch 固定为 8，microbatch 由登记时根据单卡显存选择 1 或 2，梯度累积保持全局 batch 不变。Stage1 前 1,000 step 冻结 DINO，随后在 step 1,000 至 2,000 线性预热到 $5\\times10^{-8}$；Joint 阶段从第一步起以 $10^{-8}$ 更新 DINO。`stage1_best.pt` 和 `joint_best.pt` 分别保留，联合终点不能覆盖更好的阶段一结果。

## 验收

在同一最佳 checkpoint 下，K3 平均综合分数需比 K0 低至少 1%，且至少 80/100 张训练图的 K3 全图 disparity MAE 优于 K0。联合阶段不强制优于阶段一；若退化，按 Base/Refiner 联合优化冲突报告。

## 服务器命令

```bash
bash experiment/exp2_infinidepth_disparity_ssr_hypersim100_overfit/run.sh
```
""",
    )
    write_json(
        target / "provenance.json",
        {
            "base_commit": "36c6e0c31887fafc210184ee43ca475230704095",
            "implementation_source": {
                "commit": "5796a09d24de5515bf9ea7ea14b1727f4bb37526",
                "repository": "https://github.com/ZichaoZhu/MoGe.git",
            },
            "manifest_sha256": config["data"]["manifest_sha256"],
            "origin_url": "https://github.com/ZichaoZhu/InfiniGeometry.git",
            "run_commit": None,
            "status": "planned",
            "upstream_url": "https://github.com/zju3dv/InfiniDepth.git",
        },
    )
    write_json(
        target / "metrics" / "report.json",
        {
            "acceptance": {
                "minimum_improved_images": 80,
                "required_k3_relative_improvement": 0.01,
            },
            "experiment_id": EXP2_NAME,
            "result": None,
            "runs": [],
            "status": "planned",
        },
    )
    write_json(
        target / "artifacts" / "manifest.json",
        {"assets": [], "experiment_id": EXP2_NAME, "status": "planned", "version": 1},
    )
    ignore = """
runs/*/checkpoints/*.pt
runs/*/checkpoints/*.tmp
runs/*/artifacts/*.ply
runs/*/artifacts/*.jpg
runs/*/logs/
runs/*/monitor/
*.tmp
"""
    write_text(target / ".gitignore", ignore)
    write_text(target / "runs" / "main" / ".gitignore", ignore)
    run_script = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/experiment/exp2_infinidepth_disparity_ssr_hypersim100_overfit/config.json"
OUTPUT="$ROOT/experiment/exp2_infinidepth_disparity_ssr_hypersim100_overfit/runs/main"
source "$ROOT/experiment/server_environment_guard.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/monitor"
python experiment/monitor_exp2.py \\
  --experiment "$(dirname "$(dirname "$OUTPUT")")" \\
  --training-pid "$$" \\
  --interval-seconds 300 \\
  --stale-seconds 7200 \\
  --daemon >>"$OUTPUT/monitor/daemon.log" 2>&1 &
ARGS=(
  --config "$CONFIG"
  --run-id main
  --output "$OUTPUT"
  --device "${INFINIDEPTH_DEVICE:-cuda:0}"
)
if [[ -f "$OUTPUT/checkpoints/last.pt" ]]; then
  ARGS+=(--resume "$OUTPUT/checkpoints/last.pt")
fi
exec python -m training.disparity_refiner.train "${ARGS[@]}"
"""
    run_path = target / "run.sh"
    write_text(run_path, run_script)
    run_path.chmod(0o755)

    index_path = root / "README.md"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith("| exp1 |")]
    if len(matches) != 1:
        raise ValueError("全局索引缺少唯一 Exp1 行")
    row = (
        "| exp2 | planned | 单图验证通过后，固定网格 disparity Refiner 能否扩展到 100 张 Hypersim 训练图？ "
        "| 待运行 | K3 训练域平均综合分数 | 待实验 | "
        "[exp2](exp2_infinidepth_disparity_ssr_hypersim100_overfit/) |"
    )
    lines.insert(matches[0] + 1, row)
    write_text(index_path, "\n".join(lines))


if __name__ == "__main__":
    main()

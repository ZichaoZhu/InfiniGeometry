
# Exp2：InfiniDepth 原生 Disparity Refiner 百图训练域拟合

## 前置条件与边界

本目录由 `experiment/register_exp2.py` 在 Exp1 五个运行全部结束且至少 3/5 通过 1% 门槛后生成。实验只使用 MoGe-3 Exp30 对应清单中的固定 100 张 `train` 图片，不读取 val/test 形成结论，也不宣称泛化。

## 训练

阶段一执行 20,000 个 detached optimizer step；阶段二继续 10,000 个 joint step。全局 batch 固定为 8，microbatch 由登记时根据单卡显存选择 1 或 2，梯度累积保持全局 batch 不变。Stage1 前 1,000 step 冻结 DINO，随后在 step 1,000 至 2,000 线性预热到 $5\times10^{-8}$；Joint 阶段从第一步起以 $10^{-8}$ 更新 DINO。`stage1_best.pt` 和 `joint_best.pt` 分别保留，联合终点不能覆盖更好的阶段一结果。

## 验收

在同一最佳 checkpoint 下，K3 平均综合分数需比 K0 低至少 1%，且至少 80/100 张训练图的 K3 全图 disparity MAE 优于 K0。联合阶段不强制优于阶段一；若退化，按 Base/Refiner 联合优化冲突报告。

## 结果

训练于 2026-08-11 正常完成。最终选择 Joint step 10,000，K3 相比同 checkpoint 的 K0 平均综合分数改善 23.98%，94/100 张训练图改善，通过验收。该结论仅适用于固定百图训练域。

- [最终汇总](metrics/report.json)
- [Stage1 详细报告](runs/main/metrics/stage1_report.json)
- [Joint 详细报告](runs/main/metrics/joint_report.json)
- [训练历史](runs/main/metrics/history.jsonl)
- [训练曲线](artifacts/training_curve.png)
- [Disparity 对比图](artifacts/disparity_comparison.png)
- [在线查看器](https://infinidepth-disparity-refiner-viewe.vercel.app)
- [本地查看器](http://127.0.0.1:3000)

## 服务器命令

```bash
bash experiment/exp2_infinidepth_disparity_ssr_hypersim100_overfit/run.sh
```

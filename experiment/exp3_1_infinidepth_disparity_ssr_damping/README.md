# Exp3-1：SSR 推理期 residual damping

## 目标

不重训模型，在每轮 SSR 更新中使用

\[
d_{k+1}=d_k+\alpha\,0.1\tanh(r_k/0.1),
\]

比较 \(\alpha\in\{0.25,0.5,0.75,1.0\}\)，判断 Exp3 的多轮退化是否主要来自迭代步长过大。

## 评测

固定使用 Exp3 的 Val100、K0/K1/K3/K5 和相同 disparity MAE 口径。评测当前实际保留的三个正式 checkpoint：Stage1 40k、Joint 2.5k、Joint 20k。damping 仅作用于 tanh 限幅后的 residual，下一轮 SSR 使用 damping 后的 disparity 重新体素化；\(\alpha=1.0\) 为原始推理。

主要判断条件是 K3 是否同时优于 K1 和 K0；K5 只作为训练迭代范围外的稳定性参考。若较小 \(\alpha\) 能恢复这一关系，则支持“修正方向有用但迭代步长过大”；若所有 \(\alpha\) 均无改善，则更可能是修正方向或状态分布存在问题。

## 运行

```bash
CUDA_VISIBLE_DEVICES=2 bash experiment/exp3_1_infinidepth_disparity_ssr_damping/run.sh
```

脚本支持从 `metrics/report.json` 中已完成的 checkpoint/系数组合继续运行，不覆盖 Exp3 的任何 checkpoint 或评测结果。

## 结果

- [完整逐图指标](metrics/report.json)
- [damping 曲线](artifacts/damping_curve.png)

下表分别选择每个 checkpoint 上 K3 MAE 最低的 \(\alpha\)。“K3 vs K0/K1”大于零表示 K3 更好。

| checkpoint | 最佳 \(\alpha\) | K0 | K1 | K3 | K5 | K3 vs K0 | K3 vs K1 | K3 优于 K0 | K3 优于 K1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage1 40k | 0.50 | 0.053121 | 0.053070 | 0.053052 | 0.053115 | +0.130% | +0.033% | 60/100 | 59/100 |
| Joint 2.5k | 0.25 | 0.053114 | 0.053093 | 0.053080 | 0.053099 | +0.063% | +0.025% | 59/100 | 57/100 |
| Joint 20k | 0.25 | 0.053080 | 0.053060 | 0.053077 | 0.053153 | +0.006% | -0.031% | 50/100 | 49/100 |

\(\alpha=1.0\) 复现了原始推理结果，数值差异不超过 \(3\times10^{-8}\)。Stage1 40k 和 Joint 2.5k 在较小步长下都恢复了 K3 优于 K1/K0，说明 SSR 早期 checkpoint 的修正方向有用，但原始多轮步长过大。Stage1 40k 的 \(\alpha=0.25\) 还得到 K5=0.053047，K5 进一步优于 K3=0.053057，表明较强 damping 能使训练范围外的第五轮保持稳定。

Joint 20k 的 \(\alpha=0.25\) 将原本 K3 比 K0 差 1.223% 缩小为 K3 比 K0 好 0.006%，但 K3 仍比 K1 差 0.031%，K5 也继续退化。因此 Exp3 的问题只得到部分确认：迭代步长过大是主要因素之一，但 Joint 后期还存在修正方向或迭代状态分布退化，不能只靠推理期统一 damping 完全解决。

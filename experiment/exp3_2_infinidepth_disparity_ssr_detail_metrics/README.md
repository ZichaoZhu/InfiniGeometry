# Exp3-2：SSR 细节指标评测

## 目标

验证 Exp3 的 SSR 是否在全图 normalized disparity MAE 变差时，仍改善了深度边界和局部细节。

## 评测

固定使用 Exp3 Val100、原始推理系数 1.0、K0/K1/K3/K5，以及 Stage1 40k、Joint 2.5k、Joint 20k 三个正式 checkpoint。不使用人工 crop，不修改或重训模型。

- `full_mae`：全图 normalized disparity MAE，越低越好。
- `multiscale_gradient_error`：1、2、4、8 四尺度 normalized disparity 梯度误差，越低越好。
- `boundary_f1`：以 GT radial depth 的 3% log-depth 变化定义边界，预测边界允许 1 像素偏差，越高越好。
- `edge_band_mae`：GT 边界向两侧扩张 3 像素后的 normalized disparity MAE，越低越好。

预测 radial depth 只在评测时使用每图 GT disparity 的 2%/98% 分位数还原；GT depth、边界和 mask 均不进入模型输入。每项指标同时报告 Val100 图像均值、K1/K3/K5 相对 K0 的变化和改善图数。

没有任何 GT 边界的图像不参与 `boundary_f1` 和 `edge_band_mae` 聚合，报告同时记录这两项指标的实际样本数；`full_mae` 和 `multiscale_gradient_error` 始终使用全部 100 张图。

## 运行

```bash
CUDA_VISIBLE_DEVICES=2 bash experiment/exp3_2_infinidepth_disparity_ssr_detail_metrics/run.sh
```

结果写入新的 `metrics/report.json`，不会覆盖 Exp3 或 Exp3-1 的任何结果。

## 结果

- [完整逐图指标](metrics/report.json)
- [指标变化图](artifacts/detail_metrics.png)

下表为 K3 相对同一 checkpoint K0 的变化，正值表示改善。

| checkpoint | full MAE | 四尺度梯度误差 | depth boundary F1 | edge-band MAE |
| --- | ---: | ---: | ---: | ---: |
| Stage1 40k | -0.104%（56/100） | +2.342%（89/100） | -0.913%（37/99） | -0.347%（47/99） |
| Joint 2.5k | -0.431%（44/100） | +2.339%（86/100） | -1.183%（31/99） | -0.388%（42/99） |
| Joint 20k | -1.223%（36/100） | +1.750%（73/100） | -1.740%（29/99） | -1.047%（31/99） |

Stage1 40k 的四尺度梯度误差从 K0 的 0.080714 降到 K3 的 0.078824，Joint 2.5k 从 0.080764 降到 0.078875，Joint 20k 从 0.080730 降到 0.079317。该指标与训练 loss 中的梯度项同口径，说明 SSR 确实学到了稳定的局部梯度修正，单独使用 full MAE 会遗漏这部分收益。

但 K3 的 boundary F1 和 edge-band MAE 在三个 checkpoint 上均变差，且 K5 进一步退化。Stage1 40k 和 Joint 2.5k 的 K3 梯度误差还分别比 K1 好 0.391% 和 0.280%；Joint 20k 的 K3 已比 K1 差 0.486%，再次表明 Joint 后期存在重复迭代退化。

因此“指标问题”只得到部分确认：SSR 改善的是局部梯度一致性，不等于深度不连续边界更准确。查看器中的视觉改善可能来自连续表面形状、局部平滑度或人工选取的 train 细结构区域；固定 Val100 不支持 SSR 普遍改善物体边界的结论。若要进一步覆盖连续表面细节，应另行评测 surface-normal angular error 或局部曲率误差，而不能用 boundary F1 代替。

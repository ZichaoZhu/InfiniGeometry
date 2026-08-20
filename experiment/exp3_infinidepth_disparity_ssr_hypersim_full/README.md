# Exp3：InfiniDepth Disparity Refiner 全量 Hypersim 训练

## 目标与边界

使用官方 Hypersim train split 中可产生有效 disparity 监督的 59,542 张图，验证 Exp2 的训练域拟合收益能否扩展到未参与训练的 Hypersim validation 图。本实验使用固定 100 张官方 validation 图选择 checkpoint，test split 保留；结果只支持 Hypersim 域内验证，不代表跨数据集泛化。

## 数据

RGB 和径向 depth 以 `/nas1/datasets/hypersim/raw` 为只读权威来源。正式训练前将 59,543 张 train 和固定 100 张 validation 对应的 RGB、径向 depth 原文件预复制到个人 `/mnt/data` 缓存。`ai_012_007_cam_01_frame.0000` 的 depth 中没有任何有限正值，不能形成 disparity 监督，因此从训练采样中排除，原文件仍保留在缓存中。不复制其他模态，不生成 tensor 副本，也不将全部样本预载入内存。训练采样器以固定 seed 打乱 59,542 个有效 train 索引，逐轮无放回遍历，并将顺序和位置写入可恢复 checkpoint。

## 训练

保持 Exp2 的损失、梯度路由和学习率。正式训练使用 S115 物理 GPU 2 单卡运行：全局 batch 为 8，microbatch 为 1，梯度累积 8 次。Stage1 执行 40,000 step，Joint 执行 20,000 step，共处理 480,000 个样本，约等于 8.1 个 train split 遍历。训练 K=0、1、2、3，评估 K=0、1、3、5。

每个 optimizer step 从同一可恢复采样器连续生成 8 个全局索引。Stage1 前 1,000 step 冻结 DINO，step 1,001 解冻。单个进程执行 validation 并写入日志、报告和 checkpoint。

每 500 step 在固定五张 validation 图上评估，每 2,500 step 在固定 100 张 validation 图上评估并保存 checkpoint。选择指标是 100 张 validation 图的平均 K3 全图 disparity MAE。

## 验收

同一最佳 checkpoint 下，K3 相比 K0 的 validation 平均全图 disparity MAE 至少改善 1%，且 100 张 validation 图中至少 80 张改善。联合阶段不强制优于 Stage1。

## 结果

训练完成 60,000 step。Stage1 40k 最佳 checkpoint 的 Val100 K0/K1/K3/K5 全图 disparity MAE 分别为 0.053121、0.053049、0.053176、0.053518；K3 比 K0 差 0.104%，仅 56/100 张图改善，因此未通过既定验收。Joint 2.5k 与 Joint 20k 的 K3 分别比同 checkpoint 的 K0 差 0.431% 和 1.223%，多轮 SSR 退化随 Joint 训练加重。

Exp3-1 表明较小 residual damping 能使早期 checkpoint 恢复 K3 优于 K0，但不能完全修复 Joint 后期退化。Exp3-2 表明 K3 的四尺度梯度误差改善 1.750% 至 2.342%，但 depth boundary F1 与 edge-band MAE 均变差。完整诊断见 [Exp3-1](../exp3_1_infinidepth_disparity_ssr_damping/) 和 [Exp3-2](../exp3_2_infinidepth_disparity_ssr_detail_metrics/)。

## 可视化

train 固定复用 Exp1/Exp2 的 5 张查看器样本及其细结构裁剪，便于对同一组训练图进行跨实验比较。validation 和 test 使用 seed 173，按 val、test 顺序从各官方 split 随机固定 5 张有效样本。样本 ID、选择策略、Stage1 best 与 revision 3 Joint best checkpoint 路径记录在 [viewer.json](viewer.json)。

- [在线三窗口查看器](https://infinidepth-disparity-refiner-viewe.vercel.app)
- [Val100 完整评估曲线](artifacts/training_curve_full_eval.png)
- [训练曲线](artifacts/training_curve_viewer_v2.png)
- [15 张 disparity 对比图](artifacts/disparity_comparison_viewer_v2.png)

`training_curve_full_eval.png` 的左图只使用每 2,500 step 的固定 Val100 full evaluation；右图单独显示每 500 step 记录一次的 batch=8 瞬时 training loss。旧训练曲线保留用于追溯，其中 K 曲线混合了 Val5 sampled evaluation 和 Val100 full evaluation，不用于趋势结论。

test 的 5 张样本仅用于定性展示，不参与 checkpoint 选择、调参或定量结论。validation/test 随机样本没有人工锁定细结构裁剪，只提供完整场景范围；train 保留 Exp1/Exp2 的细结构裁剪与完整场景切换。

## 运行

2026-08-14 起，Exp3 revision 2 在 S115 物理 GPU 2 上从官方 InfiniDepth 权重重新开始。revision 1 在 Stage1 step 4,953 读取到无有效 depth 样本后终止；其运行目录和 checkpoint 保留，不参与 revision 2 的 checkpoint 选择或结论。

```bash
bash experiment/exp3_infinidepth_disparity_ssr_hypersim_full/run.sh
```

# 实验总日志

## 2026-08-09 exp1

### 实验简述

目的：验证 MoGe-3 风格的稀疏三维 SSR 能否在固定 384×512 查询网格上，改善 InfiniDepth 的原生归一化 disparity；本实验只验证单图拟合能力与实现正确性，不验证泛化。

方法：对五张 Hypersim 图像分别从同一 InfiniDepth 官方 checkpoint 和零初始化 SSR 开始训练。SSR 以二维坐标、原生 disparity 和 DINOv3 特征构建稀疏三维体素，并迭代预测加性 disparity 残差。训练先进行 detach 阶段，再进行联合训练。

结果：五个独立运行全部完成并通过 1% 门槛；选定 checkpoint 均为 joint 阶段，K=3 相比同 checkpoint 的 K=0 综合 disparity MAE 分别下降 35.8%、60.4%、52.2%、87.0% 和 32.5%。结果只证明单图拟合和实现有效，不代表泛化能力。

### 实验结果

- [实验设计与验收标准](./exp1_infinidepth_disparity_ssr_single_image_overfit/README.md)
- [当前汇总结果](./exp1_infinidepth_disparity_ssr_single_image_overfit/metrics/report.json)
- [训练配置](./exp1_infinidepth_disparity_ssr_single_image_overfit/config.json)

### 其他

- 总损失为四个迭代输出损失的平均值：

  \[
  L = \frac{L_0 + L_1 + L_2 + L_3}{4}.
  \]

  每个 \(L_k\) 均由有效像素 disparity MAE 和四尺度梯度损失组成。
- K=0 是原始 InfiniDepth 输出 d0，不经过 SSR。因此 K=0 变好来自原始网络参数的微调，而非 SSR 的直接作用。
- Base Head 是训练脚本中的参数组简称，包含 BasicEncoder 和 depth_implicit_head；DINO 与 SSR 是独立参数组。
- detach 阶段中，L0 更新 Base Head；DINO 在前 1000 step 冻结，随后逐步解冻，并只接受 L0 的梯度。L1-L3 更新 SSR，但 d0 和 DINO visual feature 在输入 SSR 前 detach，因此精修损失不会回传到 Base Head 或 DINO。
- 联合阶段取消 detach：L0 继续更新 Base Head 与 DINO，L1-L3 同时更新 SSR、Base Head 和 DINO。
- GT disparity 的 2%/98% 指每张图有效 GT disparity 的第 2 与第 98 百分位，用于归一化目标与还原点云尺度；它不是模型输入，也不用于按 GT 修正预测。

## 2026-08-09 exp2

### 实验简述

目的：验证 Exp1 的 disparity SSR 能否从单图拟合扩展到固定 100 张 Hypersim 训练图，并测量同一 checkpoint 下 K=3 相对 K=0 的独立收益；本实验不使用 validation/test 得出结论。

方法：使用全局 batch 8、microbatch 1 和梯度累积 8。Stage1 训练 20,000 step，Joint 训练 10,000 step；训练监督 K=0、1、2、3，评估 K=0、1、3、5。每 500 step 进行抽样评估，每 2,500 step 进行百图完整评估并保存可恢复 checkpoint。

结果：实验已登记并进入服务器验证与训练，正式结果待训练完成。

### 实验结果

- [实验设计与验收标准](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/README.md)
- [当前汇总结果](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/metrics/report.json)
- [训练配置](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/config.json)

### 其他

- 总损失保持为：

  \[
  L = \frac{L_0 + L_1 + L_2 + L_3}{4}.
  \]

- Stage1 前 1,000 个 optimizer step 冻结 DINO，随后在 step 1,000 至 2,000 将学习率线性预热到 \(5\times10^{-8}\)。解冻后 DINO 在 detach 阶段只接收 \(L_0\) 梯度。
- Joint 阶段取消 SSR 到 Base Head 和 DINO 的梯度截断；DINO 从第一步起以 \(10^{-8}\) 更新。
- last.pt 保存 optimizer、当前阶段最佳结果、已完成阶段报告、采样器以及 Python、NumPy、PyTorch CPU 和当前训练 GPU 的 RNG 状态。
- 标准库监控器每 5 分钟检查一次，仅观察 Exp2 自己的 PID、工作目录、指标和输出目录；异常修复与恢复均记录到 Exp2 的 monitor 目录。

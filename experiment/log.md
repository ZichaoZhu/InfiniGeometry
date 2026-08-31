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

结果：实验已登记，CUDA smoke test 和 checkpoint 恢复验证通过；2026-08-10 已在 ZJU3DV-S115 的隔离 GPU 3 上启动正式训练，正式指标待训练完成。

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

## 2026-08-11 exp2

### 实验简述

目的：完成 Exp2 百图训练域实验，并验证 SSR 在同一 checkpoint 下相对 K=0 的独立收益。

方法：按既定配置完成 Stage1 20,000 step 和 Joint 10,000 step，每 2,500 step 执行百图完整评测与 checkpoint 保存，最终导出固定五图的 Exp2 viewer 资产。

结果：训练正常完成，最终选择 Joint step 10,000。K3 相比同 checkpoint 的 K0 平均综合分数改善 23.98%，94/100 张训练图改善，通过预设验收标准。该结果不代表泛化能力。

### 实验结果

- [实验设计与验收标准](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/README.md)
- [最终汇总报告](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/metrics/report.json)
- [Stage1 详细报告](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/runs/main/metrics/stage1_report.json)
- [Joint 详细报告](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/runs/main/metrics/joint_report.json)
- [训练曲线](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/artifacts/training_curve.png)
- [Disparity 对比图](./exp2_infinidepth_disparity_ssr_hypersim100_overfit/artifacts/disparity_comparison.png)
- [在线查看器](https://infinidepth-disparity-refiner-viewe.vercel.app)
- [本地查看器](http://127.0.0.1:3000)

### 其他

- 全训练过程没有出现 NaN、Inf、OOM 或有界残差越界，也没有触发 checkpoint 恢复。
- SSR 网络可以输出绝对值大于 0.1 的 raw residual，但实际加到 disparity 上的是

  \[
  r_{bounded}=0.1\tanh\left(\frac{r_{raw}}{0.1}\right),
  \]

  因此每轮实际更新始终满足 \(\lvert r_{bounded}\rvert\leq0.1\)。本次记录到的最大绝对值分别为 raw residual 0.773 和 bounded residual 0.100，属于正常的平滑限幅。
- 资产归档时发现汇总器未登记 `train.log` 和 `monitor/daemon.log`，已将 `runs/` 下的运行日志纳入资产清单后重新汇总，服务器完整性校验通过。该修复不改变训练语义、配置或 checkpoint。

## 2026-08-12 exp3

### 实验简述

目的：使用官方 Hypersim train 的全部 59,543 张图训练更久，验证 Exp2 的训练域收益能否扩展到未参与训练的 Hypersim validation 图。

方法：RGB 和径向 depth 从 NAS 按需加载；global batch 为 8，Stage1 训练 40,000 step，Joint 训练 20,000 step。使用固定 100 张官方 validation 图选择 checkpoint，test split 保留。

结果：流式加载、恢复与 CUDA smoke test 已通过；2026-08-12 在 ZJU3DV-S115 GPU 3 启动正式训练，结果待训练完成。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- 共享 NAS 是唯一数据来源；首次访问时将 RGB/depth 原压缩文件原子复制到个人 `/mnt/data` 缓存，以避免 NAS 随机读取瓶颈。训练不生成 tensor 磁盘副本，也不预载入内存。
- 训练采用固定 seed 的无放回打乱循环，采样顺序和当前位置进入 `last.pt`，恢复后不会改变样本序列。
- Exp3 只使用官方 train 更新参数；validation 只用于评估和 checkpoint 选择，test 不参与调参或结论。

## 2026-08-12 exp3

### 实验简述

目的：暂停 Exp3 单卡 pilot，将正式训练改为单机多 GPU DDP，保持 global batch、数据顺序、损失和优化口径不变。

方法：安全停止 S115 GPU 3 上的自有单卡进程，完整归档为 `single_gpu_pilot_20260812_202829`。正式实验调整为 V06 两张 RTX 4090，每卡 microbatch 为 1，梯度累积 4 次，global batch 仍为 8。新增 DDP 恢复、rank 0 单写、同步异常、协作暂停和标准库 GPU 调度器。

结果：S115 单卡 pilot 在 Stage1 step 1,000 后停止，没有 `last.pt`，不进入正式指标、checkpoint 选择或最终结论。V06 DDP smoke 和正式训练待调度器启动。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练与调度配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- S115 pilot 停止前最近一次 sampled evaluation 为 Stage1 step 1,000，有界残差绝对值最大为 0.0484，未出现 NaN、Inf 或 0.1 越界。
- V06 的 Hypersim 数据继续从 `/nas1/datasets/hypersim/raw` 只读懒加载。由于 V06 `/mnt/data` 空闲空间较少，本轮禁用 RGB/depth 本地缓存。
- 两个 rank 使用相同的全局采样状态，每次累积先生成 2 个索引，再按 rank 切分；每个 optimizer step 合计处理 8 张不重复分配的图像。
- Stage1 step 1 至 1,000 先冻结 DINO，step 1,001 同步解除 DDP 包装、启用 DINO 并围绕同一原始模型重建 DDP，optimizer 和已有参数状态不重建。

## 2026-08-12 exp3

### 实验简述

目的：在 V06 两张 GPU 空闲后完成多 GPU 验收，并启动 Exp3 正式训练。

方法：调度器确认两张 RTX 4090 连续三次满足无 compute process、空闲显存不少于 22 GiB、利用率不高于 10%。随后运行两卡 DDP smoke，验证安全暂停、同 world size 恢复、全局采样连续性和 rank 间参数一致性；验收通过后从官方 InfiniDepth 权重重新启动正式训练。

结果：两卡 DDP smoke 通过，恢复后的下一组全局样本序列一致，两个 rank 的参数 checksum 一致，单卡峰值 CUDA 显存约 16.93 GiB。正式 Exp3 于 2026-08-12 23:17 在 ZJU3DV-V06 启动，当前状态为 running。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练与调度配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- 正式训练使用 V06 的两张 RTX 4090，world size 为 2，每卡 microbatch 为 1，每卡梯度累积 4 次，global batch 为 8。
- smoke 的 median optimizer step 用时约 6.03 秒；该数值来自 8 张训练样本和 1 张 validation 图的短测，不用于估计完整实验的评估开销。
- 正式输出目录为服务器 Exp3 自有目录下的 `runs/main`。调度器仅管理本实验进程，外部任务占用所选 GPU 时请求本实验在 optimizer step 边界协作暂停，不操作其他用户进程。

## 2026-08-13 exp3

### 实验简述

目的：记录 Exp3 首次资源暂停后的恢复情况。

方法：训练推进至 Stage1 step 3,500 后，调度器检测到外部任务进入所选 GPU 并保存 `last.pt`。一次恢复启动期间又有外部进程进入 GPU，训练在 `optimizer.step()` 前因显存不足同步终止；随后重新启动调度器，等待两张 GPU 连续严格空闲后从最后一个有效 checkpoint 恢复。

结果：checkpoint 未被异常 step 污染，未发现 NaN、Inf 或有界残差越界。当前正式训练暂停在 Stage1 step 3,500，调度器状态为 waiting。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练与调度配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- Stage1 step 2,500 的首次百图 full evaluation 中，K0 full disparity MAE 为 0.05999，K3 为 0.05694，相对改善约 5.08%，100 张 validation 图中 64 张改善。该结果仍是中间结果。
- 恢复失败的直接原因是其他用户进程额外占用约 4.34 GiB 显存，不是模型 loss、梯度或 residual 异常；未停止或修改该外部进程。

## 2026-08-13 exp3

### 实验简述

目的：在实验室 GPU 资源紧张的条件下，将 Exp3 改为单卡正式训练并重新开始。

方法：删除本项目在 V06 和 S115 上的 Exp3 旧 runs、scheduler 及个人 NAS 上未完成的 checkpoint 中转目录，不修改其他用户文件或进程。新运行固定使用 S115 物理 GPU 3，从官方 InfiniDepth 权重和零初始化 SSR 开始；microbatch 为 1，梯度累积 8 次，global batch 保持为 8。

结果：此前 S115 单卡 pilot、V06 两卡 smoke 和训练结果全部作废并删除，不参与指标或结论。新的 S115 单卡正式训练从 step 0 开始。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- 单卡训练不初始化 DDP；world size 为 1。每个 optimizer step 连续处理 8 个全局样本，与原 global batch 口径一致。
- 删除范围仅包括个人项目的 Exp3 旧运行目录和个人 NAS 中转目录。官方模型权重、Hypersim 数据、Python 环境、源码及其他实验均保留。
- S115 GPU 3 启动前已有其他用户的常驻服务，占用约 9 GiB 显存。用户明确允许共享该卡；本实验不停止或修改该服务。

## 2026-08-13 exp3

### 实验简述

目的：消除 S115 单卡训练中的 NAS 随机读取瓶颈。

方法：停止尚未生成 checkpoint 的单卡运行并删除该次 `runs/main`。根据官方 split 索引，仅将 59,543 张 train 和固定 100 张 validation 对应的 RGB、径向 depth 原文件预复制到个人 `/mnt/data` 缓存；复制完成并验证文件计数后，从官方权重重新开始训练。

结果：数据缓存任务已启动，正式训练等待缓存完整性验证通过后自动开始。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- 缓存只保存原始 RGB 和径向 depth，不复制其他 Hypersim 模态，不生成 tensor 副本。
- 缓存目录为个人路径 `/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr/hypersim_exp3`，NAS 数据始终只读。
- 训练器优先读取本地缓存；只有目标文件缺失时才回退到 NAS。正式启动前要求清单中的文件全部存在，因此正常训练不会发生回退。

## 2026-08-14 exp3

### 实验简述

目的：使用已完整缓存到 S115 本地盘的 Hypersim 数据重新启动 Exp3 正式训练。

方法：逐文件校验 59,543 张 train 和固定 100 张 validation 的 119,286 个 RGB/depth 文件后，在 S115 物理 GPU 2 上启动单卡训练。不使用 `--resume`，从官方 `infinidepth.ckpt` 和零初始化 SSR 开始。

结果：训练状态为 running，GPU 利用率短时采样达到 98%，已进入实际前向和反向计算。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- world size 为 1，microbatch 为 1，梯度累积 8 次，global batch 为 8。
- 本地数据缓存大小为 20,765,777,881 字节；正常训练的 RGB 和 depth 不再回退读取 NAS。
- 启动时的 `xFormers not available` 为预期警告，模型使用现有兼容实现，不影响训练启动。

## 2026-08-14 exp3

### 实验简述

目的：处理官方 Hypersim train 中无法产生 disparity 监督的异常样本，恢复 Exp3 正式训练。

方法：revision 1 在 Stage1 step 4,953 读取 `ai_012_007_cam_01_frame.0000` 时终止。该样本的 786,432 个原始 depth 像素全部为非有限值，无法计算 2%/98% disparity 分位数。revision 2 在配置中显式排除该样本，使用其余 59,542 张有效 train 图像，并从官方权重重新开始。

结果：revision 1 的配置、源码、日志和 checkpoint 已完整归档，revision 2 已在 S115 物理 GPU 2 启动并进入实际训练。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [当前汇总结果](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [训练配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- revision 1 的最后一次完整记录为 Stage1 step 4,500；故障与 loss、梯度、residual、OOM 或 GPU 无关。
- revision 1 的归档目录为服务器 Exp3 `runs/revision1_invalid_depth_failure_20260814`，不参与 revision 2 的 checkpoint 选择和结论。
- 修复未删除异常原文件；本地缓存仍保留完整的 59,543 张 train 原始 RGB/depth。

## 2026-08-18 exp3

### 实验简述

目的：在 S115 服务器重启导致 Exp3 进程退出后恢复正式训练。

方法：保留原 `runs/main` 的日志、指标和 checkpoint，在独立目录 `runs/revision3_reboot_resume_20260818` 中从 Stage1 40,000 step 的 `last.pt` 恢复，继续使用 S115 物理 GPU 2、单卡 microbatch 1、梯度累积 8 和 global batch 8。

结果：checkpoint 已成功加载，训练重新进入 Joint 阶段，进程和 GPU 计算状态正常。

### 实验结果

- [实验设计与验收标准](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [训练配置](./exp3_infinidepth_disparity_ssr_hypersim_full/config.json)

### 其他

- S115 于 2026-08-18 02:10 重启；旧进程最后记录到 Joint step 1,500，但首次 Joint checkpoint 计划在 step 2,500 保存，因此恢复点为 Stage1 step 40,000。
- 旧 `runs/main` 未删除或覆盖。revision 3 的 provenance 明确记录了恢复 checkpoint 和新输出目录。

## 2026-08-20 exp3

### 实验简述

目的：为 Exp3 的 train、validation 和 test split 提供一致的定性点云对比。

方法：train 固定复用 Exp1/Exp2 的 5 张查看器样本、顺序和细结构裁剪；validation 与 test 使用 seed 173 从各官方 split 随机固定 5 张有效样本。对每张样本导出 GT、官方初始预测、Stage1 best 和 revision 3 Joint best 在 K=0、1、3、5 下的点云，并部署到现有三窗口查看器。

结果：修正版 `exp3_v2` 共导出 15 张 RGB 和 150 个实际 PLY 资产，服务器端检查无缺失或字节数不一致。生产网站已部署，Exp1/Exp2 回归、Exp3 三 split 切换、train 同图断言、桌面与移动端布局及三块 WebGL canvas 非空像素验收通过。

### 实验结果

- [可视化说明](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [固定样本与 checkpoint 记录](./exp3_infinidepth_disparity_ssr_hypersim_full/viewer.json)
- [训练曲线](./exp3_infinidepth_disparity_ssr_hypersim_full/artifacts/training_curve_viewer_v2.png)
- [15 张 disparity 对比图](./exp3_infinidepth_disparity_ssr_hypersim_full/artifacts/disparity_comparison_viewer_v2.png)
- [在线三窗口查看器](https://infinidepth-disparity-refiner-viewe.vercel.app)

### 其他

- test 的 5 张样本仅用于定性展示，不参与 checkpoint 选择、调参或定量结论；若后续据此修改模型，它们不得再视为完全未触碰的 test 样本。
- 预测点云使用每张图 GT disparity 的 2%/98% 分位数还原尺度，不是模型原生米制输出。
- train 保留 Exp1/Exp2 的人工锁定细结构 crop；validation/test 随机样本只显示完整场景范围，不将全图误标为细结构裁剪。
- 首次误将 train 也随机抽样的配置和 `exp3` 网站资产已保留用于追溯，但生产 catalog 只引用修正版 `exp3_v2`；旧配置归档为 [viewer_random_train_v1.json](./exp3_infinidepth_disparity_ssr_hypersim_full/viewer_random_train_v1.json)。

## 2026-08-20 exp3

### 实验简述

目的：修正 Exp3 训练曲线混合 Val5 sampled evaluation 与 Val100 full evaluation 的展示口径。

方法：保留旧曲线用于追溯，新增仅使用每 2,500 step 固定 Val100 完整评估的 K0、K1、K3、K5 曲线；batch=8 瞬时 training loss 改为独立子图。

结果：新曲线不再连接不可直接比较的 Val5 与 Val100 指标，可用于判断 Stage1 和 Joint 的 validation 趋势。

### 实验结果

- [Val100 完整评估曲线](./exp3_infinidepth_disparity_ssr_hypersim_full/artifacts/training_curve_full_eval.png)
- [实验说明](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)

### 其他

- 左图共包含 Stage1 16 个和 Joint 8 个 Val100 full evaluation 点。
- 右图 training loss 是评测时刻最后一个 global batch=8 的瞬时值，不是前 500 step 的均值。

## 2026-08-20 exp3-1

### 实验简述

目的：判断 Exp3 的 SSR 多轮退化是否主要由迭代步长过大导致。

方法：不重训模型，在 tanh 限幅后的 residual 上分别乘以 0.25、0.5、0.75、1.0，再将更新后的 disparity 用于下一轮重体素化。固定使用 Exp3 Val100，对 Stage1 40k、Joint 2.5k 和 Joint 20k 三个正式 checkpoint 评测 K0、K1、K3、K5。

结果：Stage1 40k 在系数 0.5、Joint 2.5k 在系数 0.25 时均恢复 K3 优于 K1/K0，支持早期 checkpoint 存在迭代步长过大。Joint 20k 在系数 0.25 时 K3 仅比 K0 好 0.006%，仍比 K1 差 0.031%，说明 Joint 后期还存在修正方向或迭代状态分布退化。

### 实验结果

- [实验说明与汇总](./exp3_1_infinidepth_disparity_ssr_damping/README.md)
- [完整逐图指标](./exp3_1_infinidepth_disparity_ssr_damping/metrics/report.json)
- [damping 曲线](./exp3_1_infinidepth_disparity_ssr_damping/artifacts/damping_curve.png)
- [实验配置](./exp3_1_infinidepth_disparity_ssr_damping/config.json)

### 其他

- 系数 1.0 复现原始推理结果，绝对差异不超过 3e-8；本实验没有修改或重写 Exp3 checkpoint。
- Stage1 40k 在系数 0.25 时 K5=0.053047，优于 K3=0.053057，说明强 damping 能改善训练迭代范围外的稳定性。
- damping 是推理期诊断，不是已确定的训练方案；若将系数作为超参数，应在 validation 上选择后使用未参与选择的数据确认。

## 2026-08-20 exp3-2

### 实验简述

目的：检验 Exp3 的 SSR 是否在全图 disparity MAE 变差时仍改善了局部几何细节。

方法：不重训模型，固定使用 Exp3 Val100、原始推理系数 1.0 和 Stage1 40k、Joint 2.5k、Joint 20k 三个正式 checkpoint。除 full MAE 外，自动计算四尺度 normalized disparity 梯度误差、3% log radial-depth boundary F1，以及 GT 边界 3 像素邻域的 disparity MAE。

结果：三个 checkpoint 的 K3 四尺度梯度误差均优于 K0，其中 Stage1 40k 改善 2.342%（89/100 图改善），Joint 2.5k 改善 2.339%（86/100），Joint 20k 改善 1.750%（73/100）。但 K3 的 boundary F1 和 edge-band MAE 在三个 checkpoint 上均变差。

### 实验结果

- [实验说明与汇总](./exp3_2_infinidepth_disparity_ssr_detail_metrics/README.md)
- [完整逐图指标](./exp3_2_infinidepth_disparity_ssr_detail_metrics/metrics/report.json)
- [指标变化图](./exp3_2_infinidepth_disparity_ssr_detail_metrics/artifacts/detail_metrics.png)
- [实验配置](./exp3_2_infinidepth_disparity_ssr_detail_metrics/config.json)

### 其他

- 细节 mask 只由 GT radial depth 自动生成，不使用 RGB、人工 crop 或模型预测，也不进入模型输入。
- Val100 中 99 张在 3% log-depth 阈值下包含 GT 边界；无边界的 `ai_004_003_cam_01_frame.0000` 不参与 boundary F1 和 edge-band MAE 聚合，full MAE 与梯度误差仍使用全部 100 张。
- 结论是 full MAE 会遗漏 SSR 的局部梯度收益，但现有证据不支持 SSR 普遍改善深度不连续边界。连续表面方向和曲率是否改善需要法线或局部曲率指标另行验证。

## 2026-08-20 exp3

### 实验简述

目的：归档 Exp3 全量训练的最终验收结论。

方法：完成 Stage1 40,000 step 和 Joint 20,000 step，并以固定 Val100 的全图 disparity MAE 比较同一 checkpoint 的 K0、K1、K3、K5。既定门槛为 K3 相对 K0 至少改善 1%，且至少 80/100 张图改善。

结果：训练正常完成，但 Stage1 40k 最佳 checkpoint 的 K3=0.053176，比 K0=0.053121 差 0.104%，仅 56/100 张改善，未通过验收。Joint 2.5k 和 Joint 20k 的 K3 分别比 K0 差 0.431% 和 1.223%。

### 实验结果

- [最终汇总](./exp3_infinidepth_disparity_ssr_hypersim_full/metrics/report.json)
- [实验说明](./exp3_infinidepth_disparity_ssr_hypersim_full/README.md)
- [Val100 完整评估曲线](./exp3_infinidepth_disparity_ssr_hypersim_full/artifacts/training_curve_full_eval.png)

### 其他

- `failed` 表示未通过既定 SSR 验收，不表示训练进程异常；60,000 个 optimizer step 已完整结束。
- K0 随训练缓慢改善，但原始 residual 步长下 K3/K5 普遍比 K0 更差。Exp3-1 与 Exp3-2 的 damping 和细节指标诊断分别独立归档。

## 2026-08-20 exp4

### 实验简述

目的：实现 LiDAR-conditioned InfiniDepth disparity refiner，并增加结合深度不连续与法线变化的三维边缘细节指标。

方法：删除单图过拟合和固定 Train100 训练阶段，正式配置只保留 59,542 张有效 Hypersim train 的单阶段 SSR-only 训练；冻结 `InfiniDepth_DepthSensor` 基座，以 64 线虚拟 LiDAR disparity 作为 prompt。新增 HEG-F1、每 500 step 不可变轻量 checkpoint、基础权重一次性归档和精确恢复。

结果：代码已部署到 S115 隔离目录。完整回归为 50 passed、7 skipped；两步 CUDA smoke、step 1 到 step 2 恢复、checkpoint SHA-256、基础参数冻结和 384×512 全评测均通过。正式训练尚未启动。

### 实验结果

- [实验设计与验收状态](./exp4_infinidepth_lidar_refiner_hypersim_full/README.md)
- [训练配置](./exp4_infinidepth_lidar_refiner_hypersim_full/config.json)
- [部署与测试记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment.json)

### 其他

- K0 是 LiDAR prompt conditioning 后的冻结 DepthSensor 输出；K1、K3、K5 是 SSR 迭代输出。稠密 GT 和 GT disparity 分位数不进入模型输入。
- HEG-F1 是本项目定义的混合三维边缘指标，不是既有 benchmark 的标准同名指标；同时保留 metric disparity MAE、radial-depth AbsRel、edge point-to-plane 和 normal angle 诊断。
- 官方 DepthSensor checkpoint 大小为 1,433,096,532 字节，SHA-256 为 `53230af7c46e987acc993f250e76f99914e7321ee8a13966136fdb0619f98267`。
- 当前只完成实现与测试；正式训练需在下一步获得授权后单独启动。

## 2026-08-21 exp4

### 实验简述

目的：按修订方案将 Exp4 的主训练 I/O 迁到 `/mnt/data`、将 `/nas1` 限定为校验备份，并用 MoGe-3 Local Point Rel 和 Local Point \(\delta_{0.01}\) 替换 HEG-F1。

方法：实现固定 Val100 的多尺度 disparity residual、top-hat/black-hat 与 SAM2 segment 筛选；局部点云评测采用每图共享尺度、每 segment 三维平移和 segment 等权汇总。新增每 500 steps 原子 checkpoint、SHA-256、限长异步 NAS 队列、失败重试、安全停止及本地/NAS 恢复逻辑。

结果：代码已部署到 S115 的 `20260821_impl7` 隔离目录；非 CUDA 回归 58 passed、8 deselected，本地缓存 Train 59,542 与 Val100 预检通过。SAM2 单样本生成 16 个细节 segment，主存储与 NAS 副本校验一致。服务器 NVIDIA 内核模块 580.159.03 与用户态 580.173.02 不一致，CUDA smoke 尚不能执行，正式训练未启动。

### 实验结果

- [实验设计与当前状态](./exp4_infinidepth_lidar_refiner_hypersim_full/README.md)
- [训练配置](./exp4_infinidepth_lidar_refiner_hypersim_full/config.json)
- [当前部署与测试记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260821_impl7.json)

### 其他

- 当前固定 mask manifest 仅含 1/100 个 smoke 样本，状态为 `partial`；正式训练路径会拒绝不完整 manifest。
- Val5 K0/K1/K3/K5、step 1 保存与备份、从 NAS 恢复到 step 2 的 CUDA 闭环仍待驱动恢复后执行。
- `START_EXP4_TRAINING=YES` 启动门保持关闭；本阶段未提交 Git，旧部署、checkpoint、日志和实验资产均未覆盖。

## 2026-08-21 exp4 训练启动

### 实验简述

目的：在管理员修复 NVIDIA 驱动后，使用空闲 GPU0 启动 Exp4 正式训练。

方法：先补齐固定 Val100 的 100 个 SAM2 mask，并校验 `/mnt/data` 与 NAS 逐文件 SHA-256；随后修复两处 metric disparity 边界判断：LiDAR prompt median 和 WarpMedian 均允许合法的 `1/m < 0.01` 数值。失败的启动目录和运行记录保留，正式训练使用独立 `main_retry2_20260821` run。

结果：截至当前，`main_retry2_20260821` 的 GPU0 训练进程 PID 为 341826，GPU0 显存约 16.2 GiB、利用率约 63%，尚未到达第一个 500-step checkpoint；正式训练仍在运行中。

### 实验结果

- [Exp4 配置](./exp4_infinidepth_lidar_refiner_hypersim_full/config.json)
- [训练主目录（服务器）](/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner/runs/main_retry2_20260821)

### 其他

- `main` 因环境启动脚本假设不存在 `bin/activate` 而失败；`main_retry_20260821` 因旧 WarpMedian 阈值而失败，均未覆盖。
- 当前运行使用 `/mnt/data/home/zhuzichao/projects/InfiniGeometry/deployments/exp4_lidar_refiner/20260821_impl11/InfiniDepth`，GPU 通过 `CUDA_VISIBLE_DEVICES=0` 固定为 GPU0。
- mask manifest 已完成：100/100 样本、1,117 个 segment；NAS 副本校验通过。

## 2026-08-22 exp4 备份策略调整

### 实验简述

目的：避免 NAS 延迟或权限故障阻塞并中断正式训练。

方法：每 500 steps 的 checkpoint 只在 S115 `/mnt/data` 原子保存；移除训练循环内的异步 NAS 队列。训练正常结束、Python 异常、`SIGINT` 或 `SIGTERM` 中断时，再同步备份最新本地 checkpoint 和运行元数据。

结果：新策略已部署到 S115 的 `20260822_impl12`；完整回归 61 passed、7 skipped。真实 S115→NAS 同步退出备份 smoke 用时 13.586 秒，local/NAS 状态均为 `completed`，完成标记校验通过。旧流程的 step 2500 补备份也已完成。

### 实验结果

- [Exp4 实验说明](./exp4_infinidepth_lidar_refiner_hypersim_full/README.md)
- [Exp4 训练配置](./exp4_infinidepth_lidar_refiner_hypersim_full/config.json)
- [部署与验证记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260822_impl12.json)

### 其他

- 硬断电和 `SIGKILL` 无法运行退出钩子；此时依赖 S115 上最近一次每 500 steps 保存的本地 checkpoint。
- NAS 上已原子发布的 checkpoint 仍保留，不覆盖、不删除。
- SIGTERM CUDA smoke 因检查时 GPU0 已占用 35.8/49.1 GiB、利用率 94% 而未启动，避免干扰其他任务。

## 2026-08-23 exp4 恢复训练

### 实验简述

目的：在 GPU2 空闲后，从已校验的 step 2500 checkpoint 恢复 Exp4 正式训练。

方法：首次使用 impl12 恢复时，旧 run 的 impl11 源码归档与新源码不同，恢复流程在进入训练前拒绝覆盖。impl13 将恢复版本源码保存到独立 `inputs/resume_step_000002500/`，保留原始源码归档；随后使用 `CUDA_VISIBLE_DEVICES=2` 从本地 step 2500 恢复。

结果：impl13 完整回归 62 passed、7 skipped。正式训练已恢复，Python PID 为 1559036；GPU2 显存约 14.2 GiB，连续采样利用率为 53%–61%，运行状态为 `running`，未发现 Traceback、OOM、NaN 或 Inf。

### 实验结果

- [Exp4 实验说明](./exp4_infinidepth_lidar_refiner_hypersim_full/README.md)
- [部署与恢复记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260823_impl13.json)
- [训练配置](./exp4_infinidepth_lidar_refiner_hypersim_full/config.json)

### 其他

- impl12 的失败恢复发生在训练循环之前，没有更新模型、optimizer 或 checkpoint；独立 launcher 日志保留。
- 训练期间 checkpoint 每 500 steps 仅写入 S115；正常结束或可捕获中断时，只向 NAS 同步最新 checkpoint。
- 本阶段未提交 Git。

## 2026-08-26 exp4 再次恢复训练

### 实验简述

目的：在 GPU3 可用后，从本地 step 8500 checkpoint 继续 Exp4 正式训练。

方法：复核 step 8500 的 checkpoint 与 metadata SHA256、GPU3 显存和现有进程后，使用 impl13、`CUDA_VISIBLE_DEVICES=3`、`DEVICE=cuda:0` 恢复同一 `main_retry2_20260821` run。恢复时创建独立 `inputs/resume_step_000008500/` 源码快照。

结果：训练进程 PID 为 1411064，`report.json` 状态为 `running`。GPU3 上 Exp4 显存约 12.5 GiB，总显存约 14.4 GiB，连续采样利用率为 11%–52%；未发现 Traceback、OOM、NaN 或 Inf。

### 实验结果

- [Exp4 实验说明](./exp4_infinidepth_lidar_refiner_hypersim_full/README.md)
- [恢复记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260826_resume_step8500.json)
- [训练配置](./exp4_infinidepth_lidar_refiner_hypersim_full/config.json)

### 其他

- step 8500 仍是本地最新 checkpoint；训练期间新的 checkpoint 只保存到 S115，本次恢复不触发 NAS I/O。
- NAS 上 step 8500 的旧临时目录保留，不作为可恢复 checkpoint 使用。
- 本阶段未提交 Git。

## 2026-08-27 exp4 素材导出后恢复训练

### 实验简述

目的：使用 GPU3 导出 Exp2 固定样本的 K0-K3 面试 pipeline 素材，并在导出后继续 Exp4 正式训练。

方法：确认 GPU3 上 PID 1411064 属于 Exp4 且本地最新完整 checkpoint 为 step 28,500 后，发送 `SIGTERM` 并等待同步退出备份完成。随后使用 Exp2 Joint step 10,000 selected checkpoint，对 `ai_002_003_cam_00_frame.0000` 执行一次 K0-K3 前向，导出 disparity、raw/bounded residual 和点云；导出完成后立即从 step 28,500 恢复 Exp4。

结果：Exp4 在 step 28,907 收到信号，退出备份状态为 `exit_backup_completed`；素材共 19,944,198 bytes，远端 SHA256 校验全部通过。2026-08-27 16:51:06 CST，Exp4 已在 GPU3 从 step 28,500 恢复，训练 PID 为 600937。

### 实验结果

- [本次暂停、素材导出与恢复记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260827_resume_step28500.json)
- [面试 pipeline 素材与生成脚本](../../../material/README.md)
- [素材来源与数值口径](../../../material/provenance.md)

### 其他

- step 28,907 不是完整 checkpoint；本次恢复回退到最近的 step 28,500，因此重新计算 407 steps。
- 素材 K0-K3 来自同一 Exp2 checkpoint 的同一次前向；GT 仅用于显示尺度还原和评测，不作为模型输入。
- 后续同类素材继续在实验室服务器生成，不使用 AutoDL；优先使用空闲 GPU，若需暂停正式训练则沿用“确认 checkpoint、可捕获中断、短时导出、立即恢复”的流程。
- Exp4 尚未结束，本阶段不提交 Git。

## 2026-08-31 exp4

### 实验简述

目的：完成 Exp4 正式训练的 checkpoint 选择、固定 Val100 复测与三 split 点云查看器部署。

方法：训练完成 40,000 step 后，仍以固定 Val100 的 K3 metric disparity MAE 选择 step 22,500 checkpoint；在空闲 GPU1 上重新评估 K0/K1/K3/K5，并使用确定性 LiDAR prompt 导出 train、val、test 各 5 张点云。train 复用 Exp1/Exp2 的细结构样本和裁剪，val/test 固定复用 Exp3 的 seed 173 样本。

结果：step 22,500 的 K3 相比 K0，metric disparity MAE 改善 6.979%，Local Point Rel 改善 9.584%，Local Point \(\delta_{0.01}\) 增加 3.226 个百分点；K3 在 98/100 张图的全局 MAE 更好。查看器资产共 15 个样本、106 个静态文件，Exp3 split、Exp4 split/K/三个 WebGL canvas 与移动布局验收均通过。Vercel 生产发布被账户授权拒绝，尚未对外生效。

### 实验结果

- [正式 Val100 评估](./exp4_infinidepth_lidar_refiner_hypersim_full/metrics/formal_eval_best_step_000022500.json)
- [训练与 Val100 曲线](./exp4_infinidepth_lidar_refiner_hypersim_full/artifacts/training_curve_full_eval.png)
- [15 张 disparity 对比图](./exp4_infinidepth_lidar_refiner_hypersim_full/artifacts/disparity_comparison_viewer.png)
- [资产 manifest](./exp4_infinidepth_lidar_refiner_hypersim_full/artifacts/manifest.json)
- [部署与验收记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260831_eval1.json)
- [查看器样本与 checkpoint 配置](./exp4_infinidepth_lidar_refiner_hypersim_full/viewer.json)

### 其他

- K0 是冻结 LiDAR-conditioned DepthSensor 输出，K1/K3/K5 是同一 best checkpoint 的 SSR 迭代结果；K5 相比 K3 回退，因此默认展示 K3。
- step 40,000 是最后保存点，不是选中的正式模型；其回退不会改变 best step 22,500 的 checkpoint 选择。
- 训练结束时 step 40,000 的本地与 NAS checkpoint 均已完成校验备份；正式评估单独使用已校验的 step 22,500。
- Vercel CLI 返回 `Not authorized`，需要项目账户重新授权后才可将已验证的本地 `exp4_best22500` 资产发布到现有线上查看器。

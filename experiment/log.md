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

## 2026-09-01 exp4 线上查看器发布

### 实验简述

目的：将已完成验收的 Exp4 best step 22,500 点云资产发布到现有生产查看器，并验证线上渲染。

方法：使用已授权的 Vercel 项目账户将现有查看器生产发布；在线检查 Exp4 manifest、代表性 PLY SHA-256，并以 Playwright 切换 train、val、test 和 K 值后读取三块 WebGL canvas 的非背景像素。

结果：生产地址已更新，Exp4 三个 split、K0/K3/K5 和三窗口点云均通过线上验收。离屏 canvas 在无头 Chromium 中不会稳定合成，验收测试已改为先滚动到画布可见区域再读像素，实际用户滚动到面板后渲染正常。

### 实验结果

- [生产发布与线上验收记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260901_vercel.json)

### 其他

- 本次仅发布既有 `exp4_best22500` 静态资产和测试稳定性修正，未重新导出点云、未修改模型或 checkpoint。

## 2026-09-01 exp4 RGB/LiDAR 版本对照查看器

### 实验简述

目的：在 Exp4 点云查看器中直接比较 RGB-only SSR 与 LiDAR-conditioned SSR，避免在不同实验页面之间切换样本和相机视角。

方法：确认 Exp3 `exp3_v2` 与 Exp4 `exp4_best22500` 的 train、val、test 各 5 张样本 ID 完全一致后，复用已有 PLY 资产新增第四个版本对照窗口。默认显示 Exp3 Stage1 best 的 RGB-only K3；窗口内提供 RGB/LiDAR、阶段和 K 值切换。中等桌面宽度使用 2×2 排版，大屏使用四列。

结果：本地类型检查、15 个单元测试、生产构建、Exp1–Exp4 与移动端浏览器回归均通过；生产网站已发布，线上 Exp4 的 RGB/LiDAR 切换、train/val/test、K 值和四块 WebGL canvas 验收通过。

### 实验结果

- [RGB/LiDAR 对照发布与验收记录](./exp4_infinidepth_lidar_refiner_hypersim_full/deployment_20260901_rgb_lidar_compare.json)

### 其他

- 对照窗展示的是同图、同相机坐标下的既有资产；它用于定性几何比较。RGB-only 与 LiDAR-conditioned 的输入、Base 和评测口径不同，绝对指标不作为跨版本排名。

## 2026-09-02 exp4 资产完整备份

### 实验简述

目的：将 Exp4 的完整训练资产归档至 NAS，同时保留 S115 的全部原始资产供后续实验使用。

方法：从 S115 主目录向既有 NAS 实验目录增量复制，不覆盖已有文件、不删除源文件。补齐全部 step 500 至 step 40,000 的 80 个 checkpoint、运行日志、指标、mask、provenance 和失败/烟雾运行记录；另行归档冻结的 `InfiniDepth_DepthSensor` 基础权重以及 Git 忽略的 `exp4_best22500` 查看器资产。

结果：主资产增量复制补充 261 个文件、47,347,431,270 bytes，删除文件数为 0。Base 权重 SHA-256 与配置一致；查看器资产为 106 个文件、266,495,802 bytes，源与 NAS 的整体内容摘要一致。80 个 checkpoint 的 `SHA256SUMS` 全量复算仍在 NAS 后台执行，完成前不将该项标为通过。

### 实验结果

- [本次备份记录](./exp4_infinidepth_lidar_refiner_hypersim_full/backup_20260902.json)

### 其他

- S115 的 `/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner` 未删除、未移动或覆盖，仍保留全部 80 个 checkpoint。
- Git 忽略的网站资产已独立备份；代码、配置、结果和实验日志已推送至 `feature/infinidepth-disparity-refiner`。

## 2026-09-02 exp5

### 实验简述

目的：使用 Exp4 step 22,500 checkpoint 在 Waymo validation 上进行零样本泛化评测。

方法：保留 Val5 smoke 后再运行 Val202 的顺序；将原 GPU0 等待会话改为 GPU2 安全门控会话。只当 GPU2 已用显存不超过 16,000 MiB、利用率不超过 5%，且连续三次检查满足时才启动。

结果：切换时 GPU2 已用 35,858 MiB，因此评测未启动，会话正在等待安全容量。

### 实验结果

- [Waymo 评测说明](./exp5_infinidepth_lidar_refiner_waymo_generalization/README.md)
- [GPU2 运行门控记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/runtime_20260902_gpu2_guard.json)

### 其他

- 本次只停止了自己的 GPU0 等待会话，未停止、修改或占用其他用户的进程。
- Exp4 同类任务历史显存占用约 12.5–16.2 GiB；当前 12.9 GiB 空闲显存不足以安全共卡，因此不直接启动。
- Exp5 尚未结束，当前不提交 Git。

## 2026-09-03 exp5

### 实验简述

目的：在可用 GPU 上启动 Waymo Val202 正式零样本评测。

方法：先确认 S115 GPU1 无计算进程、仅占用 25 MiB，再使用 Exp4 step 22,500 checkpoint 启动 batch=1 评测。设置显存看门狗：GPU1 空闲显存低于 4,500 MiB 时只终止本实验进程。

结果：Val5 smoke 已通过；Val202 运行至 15/202 时，本进程显存约 2,730 MiB，GPU1 仍有 45,749 MiB 空闲，未发现 OOM 或异常。

### 实验结果

- [Waymo 评测说明](./exp5_infinidepth_lidar_refiner_waymo_generalization/README.md)
- [GPU1 正式评测运行记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/runtime_20260903_gpu1_formal.json)

### 其他

- Val202 的 `progress.json` 每完成一张即原子更新；若进程中断，可跳过已完成样本继续。
- Exp5 尚未结束，当前不提交 Git。

## 2026-09-03 exp5 Waymo 评测完成

### 实验简述

目的：检验 Hypersim 训练的 Exp4 LiDAR Refiner 在真实 Waymo RGB 和 LiDAR 上的零样本泛化能力。

方法：固定 Exp4 step 22,500 checkpoint，对 Waymo validation 的 202 个序列各取一帧，使用分离的 TOP LiDAR prompt 点和 held-out 评测点比较 K0/K1/K3/K5，不训练、不微调、不重选 checkpoint。

结果：202/202 完成。K3 相比 K0 的 metric disparity MAE 下降 6.729%，point \(\delta_{0.01}\) 增加 5.665 个百分点；radial AbsRel 均值下降 5.976%，但其配对 bootstrap 95% 置信区间跨过 0。radial RMSE 从 18.515 m 增加到 26.075 m，表明仍有大误差样本需要排查。

### 实验结果

- [Waymo 评测说明](./exp5_infinidepth_lidar_refiner_waymo_generalization/README.md)
- [Val202 汇总指标](./exp5_infinidepth_lidar_refiner_waymo_generalization/metrics/waymo_val202_step22500_summary.json)
- [GPU1 运行记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/runtime_20260903_gpu1_formal.json)

### 其他

- 正式评测在 GPU1 完成，显存保护未触发，没有 OOM 或异常；结束后 GPU1 回到 25 MiB 占用。
- K3 在 disparity MAE 上改善 201/202 张，在 point \(\delta_{0.01}\) 上改善 202/202 张；K5 在主要指标上比 K3 略有回退。
- Waymo 子任务已完成；Exp5 还包含 ETH3D 子任务，因此当前不提交 Git。

## 2026-09-03 exp5 Waymo 可视化发布

### 实验简述

目的：在不补充新评测指标的前提下，定性检查 Exp4 step 22,500 模型在 Waymo 真实场景上的 zero-shot 点云输出。

方法：对 Val202 已完成序列按 seed 173 固定随机抽取 5 张，导出 held-out TOP LiDAR 参考点云和同一 checkpoint 的 K0/K1/K3/K5 预测。使用 Waymo 相机内参生成米制相机射线，统一显示 0–100 m；复用现有三窗口查看器并将指标栏标记为暂未评测。

结果：5/5 样本完成，共导出 25 个 PLY，资产大小 73,992,600 bytes，所有 PLY SHA-256 校验通过。线上 Exp5 入口、图像切换、K 切换和三块 WebGL canvas 验收通过。

### 实验结果

- [Waymo 可视化说明](./exp5_infinidepth_lidar_refiner_waymo_generalization/README.md)
- [可视化样本与导出配置](./exp5_infinidepth_lidar_refiner_waymo_generalization/viewer.json)
- [生产发布与验收记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/deployment_20260903_viewer.json)
- [在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)

### 其他

- 本批资产只用于可视化，未重跑 Val202，未新增计算 Local Point Rel、Local Point \(\delta_{0.01}\) 或其他指标。
- held-out TOP LiDAR 是稀疏参考点云，不是稠密 GT；页面已明确标注，不与 Hypersim 稠密 GT 混用。
- Exp5 还包含 ETH3D 子任务，当前不提交 Git。

## 2026-09-03 exp5 Waymo 侧视选图图库发布

### 实验简述

目的：针对前视图中细杆结构不易观察的问题，为用户提供 Waymo 侧视相机选图。

方法：使用与 Exp5 Val202 相同的 202 个序列和报告记录的实际帧索引，分别解码 `SIDE_LEFT` 和 `SIDE_RIGHT` 图像。每个相机使用自己的图像和标定信息进行预览，不运行模型。

结果：侧视图库共 404 张，左右各 202 张，包含相机、地点、时段、天气筛选，并可复制“相机 + TFRecord 文件名”。本地与线上清单一致，线上筛选和选择验收通过。

### 实验结果

- [Waymo Val202 SIDE 选图页](https://infinidepth-disparity-refiner-viewe.vercel.app/data/waymo_gallery_exp5_val202_side_20260903_r2/index.html)
- [侧视选图页发布与验收记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/deployment_20260903_side_gallery.json)
- [图库导出配置](./exp5_infinidepth_lidar_refiner_waymo_generalization/gallery.json)

### 其他

- 侧视图更可能看到路边护栏、电线杆、路牌杆和建筑边缘，但是否更适合需以实际图像为准。
- 用户选定后需同时保留相机方向；后续点云导出不能直接复用 FRONT 的标定。
- Exp5 还包含 ETH3D 子任务，当前不提交 Git。

## 2026-09-03 exp5 Waymo 选图图库发布

### 实验简述

目的：使用者可在不重跑模型的情况下，从 Waymo Val202 中主动挑选最适合点云定性对比的场景。

方法：仅解析 Val202 已完成报告记录的 FRONT JPEG，按报告中的实际 `frame_index` 取出同一帧，以模型输入一致的 384×512 尺寸生成静态图库。页面提供地点、时段、天气、文本筛选和 TFRecord 选中复制。

结果：202/202 张图像已导出，共 204 个静态文件、6,635,218 bytes。本地和线上的图库 JSON 的 SHA-256 一致；线上加载 202 张卡片、选中和 Night 筛选验收通过。

### 实验结果

- [Waymo Val202 FRONT 选图页](https://infinidepth-disparity-refiner-viewe.vercel.app/data/waymo_gallery_exp5_val202_front_20260903_r2/index.html)
- [选图页发布与验收记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/deployment_20260903_gallery.json)
- [图库导出配置](./exp5_infinidepth_lidar_refiner_waymo_generalization/gallery.json)

### 其他

- 图库仅用于选图，不运行 Base 或 SSR，不新增指标，也不更改已完成的 Val202 结果。
- 用户选定后将页面复制的 TFRecord 文件名发回，再单独导出对应 5 张点云资产。
- Exp5 还包含 ETH3D 子任务，当前不提交 Git。

## 2026-09-03 exp5 Waymo 手选 SIDE 点云可视化发布

### 实验简述

目的：针对用户从 SIDE_LEFT/SIDE_RIGHT 图库中手选的五个 Waymo 场景，定性检查 Exp4 LiDAR Refiner 在侧视细杆和场景边缘的 zero-shot 输出。

方法：固定 Exp4 step 22,500 checkpoint 与既有 Val202 报告的帧索引。每张图按指定 SIDE 相机重新解码 RGB、使用该相机标定将 TOP LiDAR 分为 prompt 与 held-out 点，并导出 K0/K1/K3/K5 点云。仅导出可视化，不重跑 Val202、不计算新指标、不选择 checkpoint。

结果：5/5 手选样本完成，SIDE_RIGHT 3 张、SIDE_LEFT 2 张；共导出 25 个 PLY，32 个文件共 74,050,932 bytes。25 个 PLY 的 SHA-256 均与 manifest 一致；本地桌面/移动端及生产站点的五样本、K=3、三窗口 WebGL 验收均通过。

### 实验结果

- [Waymo 手选 SIDE 配置](./exp5_infinidepth_lidar_refiner_waymo_generalization/viewer_side.json)
- [手选 SIDE 发布与验收记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/deployment_20260903_side_viewer.json)
- [在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)

### 其他

- SIDE 相机不是 FRONT 的复用标定；每张图均使用各自相机的图像、内参与 TOP LiDAR 投影。
- held-out TOP LiDAR 为稀疏参考点云，不是稠密 GT；页面明确标注该批资产仅用于定性观察。
- S115 使用新的不可变部署目录 `20260903_side_viewer1`；后续重试检测到已完成资产后停止，未覆盖任何文件。
- Exp5 仍包含 ETH3D 子任务，当前不提交 Git。

## 2026-09-03 exp5_waymo 十样例查看器合并

### 实验简述

目的：将固定随机选择的 5 张 Waymo FRONT 样例和用户手选的 5 张 SIDE 样例归入同一 `exp5_waymo` 查看器入口，便于按同一 K 值和阶段进行定性对比。

方法：不重新推理、不复制或覆盖既有 PLY。新增组合 manifest 与 provenance，逐项校验两个来源 manifest 中的既有 PLY SHA-256；查看器移除旧的两个独立入口，改为固定顺序的 10 张样例。

结果：10/10 样例可在生产站点切换，FRONT/SIDE、K0/K1/K3/K5 与三窗口 WebGL 均通过生产 Playwright 验收。

### 实验结果

- [Waymo 说明](./exp5_infinidepth_lidar_refiner_waymo_generalization/README.md)
- [组合样例来源](./viewer/public/data/exp5_waymo/selection.json)
- [生产发布与验收记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/deployment_20260903_waymo_combined_viewer.json)
- [在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)

### 其他

- FRONT 五张仍是 Val202 完成结果按 seed 173 的固定随机选择；SIDE 五张仍为用户手选，二者都不用于 checkpoint 选择。
- 旧的两个来源资产目录保留为不可变 provenance，不删除、不重写。
- Exp5_ETH3D 仍在运行，因此当前不提交 Git。

## 2026-09-03 exp5_ETH3D

### 实验简述

目的：检验固定 Exp4 LiDAR Refiner 在 ETH3D DSLR 多视图数据上的 zero-shot 泛化，不在 ETH3D 上训练、微调或选择 checkpoint。

方法：使用 ETH3D high-res training 的 13 个场景、454 张带 `THIN_PRISM_FISHEYE` 标定的图像。将原始相机 z-depth 依标定转换为 radial range，生成确定性的 64 线、stride 4 虚拟 LiDAR prompt；在排除 prompt 像素的 held-out 深度点上评测 K0/K1/K3/K5。细节区域使用独立的 SAM2 mask，并按 MoGe-3 的 Local Point Rel 与 Local Point \(\delta_{0.01}\) 诊断。

结果：K3 相比同 checkpoint K0 的 metric disparity MAE 下降 23.86%（0.0009067 至 0.0006904，446/454 图改善），Local Point Rel 下降 2.61%（0.01143 至 0.01114，351/444 图改善）。K5 的主指标与局部指标均略回退，K3 是该数据集上的最佳迭代次数。

### 实验结果

- [配置与运行脚本](./exp5_infinidepth_lidar_refiner_eth3d_generalization/)
- [454 图汇总指标](./exp5_infinidepth_lidar_refiner_eth3d_generalization/metrics/eth3d_highres_train_step22500_summary.json)
- [部署与数据 provenance](./exp5_infinidepth_lidar_refiner_eth3d_generalization/deployment_20260903_impl2.json)

### 其他

- 主聚合为图像宏平均；K3-K0 置信区间以 13 个场景 block bootstrap 计算，避免把同一场景的相关视角当作独立样本。
- ETH3D 测试标签不公开，本实验使用公开 training split，不属于官方 leaderboard 提交。
- 5,447 个细节 segment 来自 444 张有可用 segment 的图像；评测结束无 OOM、异常或显存保护触发。

## 2026-09-03 exp5_ETH3D 选图图库

### 实验简述

目的：让用户从已完成 ETH3D 正式评测的 454 张输入中挑选 10 张，后续导出对应 K0/K1/K3/K5 点云进行定性比较。

方法：复用正式评测的输入 manifest 和 RGB，按相同中心裁剪、LANCZOS 缩放到 512×384，生成 13 个场景的静态预览。页面支持场景/样本 ID 筛选、浏览器本地选择，最多选择 10 张并复制 `scene/DSC_xxxx`。

结果：454 张预览和选图页面已发布到生产查看器；S115 图库部署测试 7 passed，生产 Playwright 验收通过。该页面不运行模型、不新增指标、不修改正式评测结果。

### 实验结果

- [选图页](https://infinidepth-disparity-refiner-viewe.vercel.app/data/eth3d_gallery_exp5_highres_train_20260903_r1/index.html)
- [发布与校验记录](./exp5_infinidepth_lidar_refiner_eth3d_generalization/deployment_20260903_gallery.json)

### 其他

- 选择后请复制页面显示的样本 ID 并发回；后续点云导出将严格使用同一输入与 checkpoint。
- 静态预览资产保存在 S115 的 `deployments/exp5_ETH3D/20260903_gallery1`，生产别名保持 `infinidepth-disparity-refiner-viewe.vercel.app`。

## 2026-09-03 exp5_ETH3D 十张点云可视化

### 实验简述

目的：根据用户从 ETH3D 选图页挑选的十个样本，导出与 Exp5 正式评测一致的 K0/K1/K3/K5 点云并部署到查看器。

方法：固定 Exp4 step 22,500 checkpoint、ETH3D 输入 manifest 和 512×384 几何。每张导出 ETH3D dense ground truth、K0、K1、K3、K5 五个 PLY，点云显示范围为 0–100 m；不新增指标、不重新选择 checkpoint。

结果：10/10 样本、50 个 PLY 和 10 个 RGB 已发布。生产查看器可切换 `Exp5 ETH3D`、十张样本和 K0/K1/K3/K5，三窗口 WebGL 验收通过。

### 实验结果

- [在线十张点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)
- [选定样本与发布记录](./exp5_infinidepth_lidar_refiner_eth3d_generalization/deployment_20260903_selected_viewer.json)

### 其他

- 样本顺序严格按用户提供的十个 `scene/DSC_xxxx` ID；旧 r2 失败目录保留为诊断痕迹，生产使用新的不可变 r3 asset tag。
- 生产部署 ID 为 `2iE7HnoXLXuXgVcNqUssDotCaCn6`，别名保持不变。

## 2026-09-04 Exp5 查看器指标补充

### 实验简述

目的：为 Exp5 的定性点云查看器补充已有正式报告中的展示诊断指标，不改变 checkpoint 选择或正式实验结论。

方法：新增版本化 metadata-only manifest，复用既有 Waymo/ETH3D PLY 与 RGB 资产。ETH3D 显示 held-out Inverse-depth MAE、Radial AbsRel、Radial RMSE、Point \(\delta_{0.01}\) 及 MoGe-3 Local Point Rel/\(\delta_{0.01}\)；Waymo FRONT 显示前四项，SIDE 因没有对应正式逐图报告保留为无数值的定性样例。

### 实验结果

- [Exp5 ETH3D 指标清单](./viewer/public/data/exp5_eth3d_metrics_20260904_r1/manifest.json)
- [Exp5 Waymo 指标清单](./viewer/public/data/exp5_waymo_metrics_20260904_r1/manifest.json)
- [指标附加脚本](../training/disparity_refiner/attach_exp5_display_metrics.py)

### 其他

- 指标来源：ETH3D 报告 SHA-256 `a960a53d…85e0f7`；Waymo 报告 SHA-256 `0e4c03e4…e1db16`。
- `npm run typecheck`、`npm run build`、`npm test`（17 passed）和 Exp5 相关 Playwright（2 passed）通过；生产部署 `dpl_maRTD8bqBWBWxGkCdGwNdT3QxBDf` 已就绪，线上 Exp5 Waymo/ETH3D Playwright 2 passed。
- [生产发布与验收记录](./exp5_infinidepth_lidar_refiner_waymo_generalization/deployment_20260904_metrics_viewer.json)

## 2026-09-10 exp6-3 查看器切换修复

### 实验简述

目的：修复切换到 Exp6-3 时查看器偶发整页加载失败的问题。

方法：加载新实验 manifest 期间，禁止以新实验 ID 渲染上一个实验的 manifest 和样本；不重新推理或修改任何点云资产。

结果：生产站点的 Exp3 到 Exp6-3 切换可显示四路点云，随后切回 Exp3 可显示三路点云，均无页面异常或失败请求。

### 实验结果

- [生产发布与验收记录](./viewer/deployment_20260910_exp6_3_switch_fix.json)
- [在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)

### 其他

- 根因是新实验 ID 与旧 manifest 的短暂状态不一致，不是 Exp6-3 PLY 或 manifest 缺失。
- v8 的 10 个样本、100 个 PLY 和 10 个 RGB 引用均已核对存在；PLY 的大小和 SHA-256 与 manifest 一致。

## 2026-09-11 exp6-4 官方 SSR disparity 适配计划

### 实验简述

目的：在同一冻结的 RGB-only InfiniDepth Base 上，比较现有 SSR 与官方 Sparse3DUNet 的 disparity 适配版，判断网络主体差异是否影响局部收益与全局误差。

方法：计划重新训练 A/B 两组 SSR，各 20,000 steps，沿用相同 disparity 表达、视觉条件、损失、数据和采样顺序。当前仅完成计划，尚未实施或运行，无新增实验结果。

### 实验结果

- [实施与验收计划](./exp6_4_official_ssr_disparity_adapter/PLAN.md)

### 其他

- 本轮只移植官方网络主体，不迁移官方预训练权重或 log-depth 表达；不同结构的比较不等于单独验证 LayerNorm 的作用。
- Base 需同时冻结参数和训练态 buffers；评估结束后只恢复 SSR train。L0 保留为无梯度诊断项，损失仍按四项平均。

## 2026-09-11 exp6-4 适配与自动流程部署

### 实验简述

目的：实施冻结同一 RGB Base 的现有 SSR 与官方网络主体对照，避免将 Base 微调收益混入 SSR 收益。

方法：接入固定版本官方 Sparse3DUNet，新增 refiner-only 模式，复用现有恢复、GPU 查询、监控和 Exp6-3 指标口径；A/B smoke、恢复测试通过后自动顺序训练各 20,000 steps。

结果：源码及 CPU 回归测试已完成，S115 专用目录中的后台流程已启动。当前四张卡均有 compute 进程，正在等待严格空闲单卡；CUDA 测试和正式训练未开始，无新增训练结论。

### 实验结果

- [实现、执行顺序与产物说明](./exp6_4_official_ssr_disparity_adapter/README.md)
- [部署与验证记录](./exp6_4_official_ssr_disparity_adapter/deployment_20260911.json)
- [预登记实验协议](./exp6_4_official_ssr_disparity_adapter/PLAN.md)

### 其他

- 不改 disparity 表达、损失尺度或迭代间梯度连接；不加载官方 MoGe SSR 权重。
- Base 参数和 buffers 均冻结，保存前核对哈希；评估后恢复 SSR train，并核对固定样本 K0。
- 每 500 steps 保存恢复点，与每 2,500 steps 的 full evaluation 解耦；checkpoint 记录 backend，旧 checkpoint 保持 spconv 默认路径。
- 新增依赖在专用 venv，旧训练仓库和历史产物不覆盖。没有停止、修改他人的任务或文件，也没有执行 Git 提交。
- 后台每 300 秒检查一次，不调用模型；数值或代码错误写告警并停止盲目重启，资源暂停则从有效 checkpoint 自动恢复。
- 同日补齐最后一步协作暂停的恢复边界：恢复时重新执行终点 full evaluation，不沿用较早结果。仅停止了仍在等待空闲 GPU 的本次控制器，随后以新快照恢复后台等待；[最终部署与验证记录](./exp6_4_official_ssr_disparity_adapter/deployment_20260911_v4.json)。

## 2026-09-12 exp6-4 GPU 共享规则调整

### 实验简述

目的：遵循实验室允许多用户共享同一 GPU 的规则，在 GPU 3 显存充足、利用率较低时启动 Exp6-4。

方法：自动流程固定使用物理 GPU 3，允许其上存在其他用户 compute PID；启动门槛改为剩余显存至少 22 GiB、利用率不超过 10%。不向外部 PID 发送信号，也不因外部 PID 存在而暂停。

结果：规则修改完成，待新快照通过 CPU 回归后立即在 GPU 3 执行 CUDA 验证与 smoke；通过后进入正式 A/B 训练。

### 实验结果

- [执行说明](./exp6_4_official_ssr_disparity_adapter/README.md)
- [GPU 3 共享启动与验证记录](./exp6_4_official_ssr_disparity_adapter/deployment_20260912_shared_gpu3.json)

### 其他

- 旧控制器在 `waiting_for_idle_gpu` 状态下核验身份后停止；当时没有 Exp6-4 CUDA 子进程或训练进程，不影响其他任务。
- 共享造成的显存不足只处理 Exp6-4 自身任务，不停止或修改其他用户进程。
- 首次 CUDA 门禁在 GPU 3 运行，现有 spconv 路径及其余测试通过；官方 FlexGEMM 宽通道反向 kernel 请求 133,120 B shared memory，超过 RTX 4090 的 101,376 B 限制。将该后端改为等价的 `implicit_gemm` kernel 后复测，不改变网络结构或稀疏卷积数学定义。
- 复测表明单纯切换 algorithm 仍选中相同的大 tile。根因修正为 adaptive 模式在冷启动的前 99 次直接使用 A100 首选配置。恢复官方 algorithm 选择，在 Exp6-4 子进程内启用 `always` autotune，由 FlexGEMM 自身过滤 OOR tile；使用实验独立 cache，不修改共享环境。
- 完整 CUDA 门禁通过 10/10 后，首次 spconv smoke 在第一个 optimizer step 前生成 provenance 时发现部署使用的不含 `.git` 的不可变源码快照。非 Git 快照现明确记录 Git 字段为 `unavailable`；保留失败快照和日志，以新快照重新执行门禁和 smoke。
- `source_v8` 在 GPU 3 通过 CPU 82/82、CUDA 10/10 门禁；spconv smoke 已在 step 5 保存并恢复 checkpoint，恢复后继续产生 optimizer step，证明训练与恢复链路均已实际运行。

## 2026-09-14 exp6-4 官方 SSR 对照完成

### 实验简述

目的：在同一冻结 InfiniDepth Base、相同数据和采样顺序下，比较现有 spconv SSR 与移植的官方 Sparse3DUNet 主体。

方法：两组分别训练 20,000 steps，在固定 Hypersim validation 100 上评估，并按 Exp6-3 几何指标进行配对分析。

结果：两组均正常完成。official_flex 的 K3 full disparity MAE 为 0.054461，优于 spconv 的 0.058131；配对报告将其标记为候选改进方案。

### 实验结果

- [最终配对分析](./exp6_4_official_ssr_disparity_adapter/results_20260914/paired_report.json)
- [spconv 训练报告](./exp6_4_official_ssr_disparity_adapter/results_20260914/spconv/stage1_report.json)
- [spconv 几何指标](./exp6_4_official_ssr_disparity_adapter/results_20260914/spconv/geometry_summary.json)
- [official_flex 训练报告](./exp6_4_official_ssr_disparity_adapter/results_20260914/official_flex/stage1_report.json)
- [official_flex 几何指标](./exp6_4_official_ssr_disparity_adapter/results_20260914/official_flex/geometry_summary.json)
- [同步文件清单与哈希](./exp6_4_official_ssr_disparity_adapter/results_20260914/manifest.json)

### 其他

- 两组 K0 一致，冻结 Base 哈希一致，20,000-step 全局采样序列逐步一致。
- spconv 在 80/100 张图上改善，official_flex 在 71/100 张图上改善；后者平均误差更低，但不代表每张图都更好。
- disparity、affine point 和 local point 配对指标支持 official_flex；local depth 的置信区间跨过 0，暂不能确认稳定提升。
- 当前仅有单随机种子结果，不自动替换默认实现。

## 2026-09-14 exp6-4 三路点云可视化

### 实验简述

目的：在相同图片和观察口径下，定性比较 GT、现有 spconv SSR 与 official_flex SSR，避免选图差异干扰判断。

方法：复用 Exp3 的 seed 173 固定选图，train、val、test 各 5 张；其中 val/test 与 Exp6-3 完全一致。两种 SSR 使用各自 step 20,000 checkpoint，默认展示 K=3，并保留 K=0/1/3/5 切换。

结果：15 张 RGB、150 个 PLY 和 manifest 已发布到原生产查看器。线上三窗口渲染、K 切换及 train/val/test 切换均通过浏览器验收。

### 实验结果

- [在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)
- [部署与验收记录](./viewer/deployment_20260914_exp6_4_official_ssr.json)
- [实验结果与可视化说明](./exp6_4_official_ssr_disparity_adapter/README.md)

### 其他

- test 仅用于可视化，不参与训练、checkpoint 选择或定量结论。
- 导出资产清单中的 150 个 PLY 均完成字节数和 SHA-256 校验。
- 5 张 GT 含少量 Hypersim 无效深度像素，查看器在构建显示点云时过滤；150 个预测点云均为有限坐标。

## 2026-09-14 exp6-4 结果诊断计划

### 实验简述

目的：解释两种 SSR 的逐图波动，区分共同失效、迭代退化和区域收益，为下一项受控实验提供依据。

方法：先复用 Val100 逐图指标和训练曲线做 CPU 分析，再用现有两组终点 checkpoint 补充像素级推理，检查 Local 区域、深度分组及 residual 方向；仅在证据支持时做固定幅度的推理干预。

结果：已完成计划及本地配置、接口和样本交集核验。尚未实现分析入口或执行远程推理；本轮不补第二 seed、不启动新训练。

### 实验结果

- [结果诊断计划](./exp6_4_official_ssr_disparity_adapter/ANALYSIS_PLAN.md)

### 其他

- 网站 VAL 5 张与正式 Val100 的样本交集为 0；两者属于官方 validation 的不同选图，不能混为同一批样本。
- 网站固定 ROI 指标与正式 Local 分割及对齐口径不同；主诊断复用 Val100 和既有 Local masks，test 不参与方法选择。

## 2026-09-14 exp6-4 结果诊断完成

### 实验简述

目的：解释两种 SSR 的逐图波动，区分共同失效、迭代退化与区域收益。

方法：复算原 Val100 配对指标，读取训练曲线，并用两组固定终点 checkpoint 各补充一次 Val100 像素级推理；按预定规则输出代表图。

结果：official_flex 平均更好，但不是每图都更好；后续迭代存在退化。当前证据更支持优先检验更新方向与迭代监督，尚不足以将 disparity 表达或残差上限确定为根因。本轮仅分析，不新增训练。

### 实验结果

- [诊断报告](./exp6_4_official_ssr_disparity_adapter/analysis_20260914_r3/REPORT.md)
- [本地诊断图册](./exp6_4_official_ssr_disparity_adapter/analysis_20260914_r3/gallery.html)
- [逐图指标](./exp6_4_official_ssr_disparity_adapter/analysis_20260914_r3/per_image.csv)
- [像素与区域汇总](./exp6_4_official_ssr_disparity_adapter/analysis_20260914_r3/pixel_summary.json)
- [执行审计](./exp6_4_official_ssr_disparity_adapter/analysis_20260914_r3/audit.json)

### 其他

- 正式 Global/Local 几何指标与原生 disparity 区域诊断分开报告；空 Local 的区域 MAE 不记零，full-MAE 贡献按全部图像平均。
- sparse 迭代前向存在小幅非逐位一致，重复实验与容差调整记录在独立 r1/r2/r3；不以重算数值覆盖原正式评估。
- 没有满足过量更新主导的条件，按计划跳过残差减半干预；下一项建议为迭代非退化监督的受控实验，尚未开始。
- 未改动原 checkpoint、训练结果、网站选图和资产，不操作其他用户任务；复用现有接口和依赖，未添加持续模型监控。

## 2026-09-15 exp6-4 后续路线与历史 MoGe SSR 核查

### 实验简述

目的：明确旧 MoGe-2 + SSR 的实现和训练资产，避免把已有小规模训练误记为没有模型，并为后续公平对照确定边界。

方法：核查旧 MoGe 仓库、官方 MoGe-3 与 InfiniDepth 两种 SSR 源码，读取旧实验报告，并只读检查 S115 的 Exp30 checkpoint。

结果：旧实现有 100 张与 320 张 Hypersim 子集训练记录；Exp30 阶段一最佳和联合终点文件仍在，阶段一文件哈希与原报告一致。尚未找到全量 Hypersim 训练的旧 MoGe-2 + SSR 模型。优先规划官方模型 OOD 审计与同条件 MoGe 训练对照，暂不启动新的 InfiniDepth 损失实验。

### 实验结果

- [历史实现、模型资产与两项实验计划](../../../docs/current/2026-09-15_MoGe_SSR历史核查与两项实验计划.md)

### 其他

- InfiniDepth spconv 沿用旧 SSR 主体，但从 log-depth 改为 disparity；official_flex 移植官方 U-Net 主体，不等于完整官方 MoGe-3 流水线或官方已训练 SSR 权重。
- 官方 checkpoint 的十数据集评测已存在；应先核查 OOD 来源与复用条件，Hypersim 本身不是官方 MoGe-3 的 OOD。
- 两套 MoGe 的训练对照需统一 Base 初始化、数据、预算与指标；完整实现的差别不只在 U-Net，不能将整体结果单独归因于网络主体。
- 本轮仅新增本地核查文档并追加日志；远程只读，没有启动训练、部署或改动 checkpoint。

## 2026-09-21 exp6-4 方法口径与代码交接整理

### 实验简述

为交接后续 scaling up，将整理主线收缩到“InfiniDepth＋官方 SSR 网络主体的 disparity 适配版”。核对源码与历史记录，整理官方模块、接口适配、训练和导出入口，明确已完成实验与未验证变体。此次仅更新文档和索引，不修改算法或新增实验结果。

### 实验结果

- [代码交接、方法边界与工程待办](../docs/exp6_4_disparity_adapter_handoff.md)
- [Exp6-4 详细结果与原实验入口](./exp6_4_official_ssr_disparity_adapter/README.md)
- [Exp6-5 R1 损失修复记录](../../MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3/REVISION_R1.md)

### 其他

Exp6-4 只移植官方 Sparse3DUNet 主体，保留 normalized disparity、有界加法更新、disparity 监督和跨轮梯度；Base 全程冻结，未做 Joint。限幅是历史额外约束，不是官方必需项，也未证明去掉后等价或更好。Exp6-5 修正几何损失后无界 logZ 分支完成训练，不能据此认定 Exp6-4 去限幅已验证。

目前没有以 InfiniDepth 为基座、仅作必要接口适配并尽量保留官方完整 SSR 机制的实验。文档保留该结论边界，不把拟议实验当作现有结果。按 ponytail 沿用现有目录与训练器，仅整理说明；官方单组入口、可迁移路径和新环境验收仍列为待办。原实验计划、配置、源码、checkpoint、结果与网站不变；未连接服务器、部署、重训或提交 Git。

## 2026-09-21 exp6-4 本地代码与结果归档

### 实验简述

为后续交接，将已完成 Exp6-4 按代码、结果拆分归档。代码提交为 `dd252db`；结果与方法边界、入口和验证说明单独提交。保留历史 disparity 适配和限幅行为，不改算法、不补做实验，不混入 Exp6-1、Exp6-5 的其他改动。

### 实验结果

- [方法边界与代码交接](../docs/exp6_4_disparity_adapter_handoff.md)
- [详细结果](./exp6_4_official_ssr_disparity_adapter/README.md)
- [Git 与外部资产边界](./exp6_4_official_ssr_disparity_adapter/ARCHIVE.md)
- [本次验证与未重跑项目](./exp6_4_official_ssr_disparity_adapter/COMMIT_CHECKS.md)

### 其他

采用独立暂存快照验证 Exp6-4：查看器类型检查、19 项单元测试、生产构建和三窗口浏览器测试通过，调度器 7 项测试与分析脚本标准库自测通过。本机依赖不足，未重跑完整 PyTorch/CUDA 训练测试；历史验收单独标注。原指标与报告不重写，数组、案例图、权重和点云资产不删除；仅提交紧凑报告与两张汇总图。提交使用用户 Git 身份，不推送，不连接或更改服务器。

## 2026-09-23 exp6-4 单组运行交接

### 实验简述

为学长复用提供 official_flex 单组配置准备和显式机器路径。复用训练器，保留历史 A/B 入口、科学配置和严格恢复校验；新增路径保护及固定评估资源校验。本轮没有启动训练或修改已有结果。

### 实验结果

- [运行步骤、依赖与验证边界](./exp6_4_official_ssr_disparity_adapter/RUNNING.md)
- [方法与代码交接](../docs/exp6_4_disparity_adapter_handoff.md)
- 代码提交：`993e22c`。

### 其他

标准库 7 项测试通过，覆盖科学配置不变、路径边界、不可变配置、评估哈希、无启动检查与主 rank 报告保护。真实模型/CUDA 和新环境恢复验证未执行；旧 last.pt 不能因迁移路径而跳过配置哈希检查。仅本地整理与提交，不推送、不部署、不访问服务器。

# Exp6-4：官方 SSR 网络主体的 disparity 适配对照

状态：实施中。2026-09-11 方案获授权；实现与 CPU 验证已完成，后台等待空闲 GPU。CUDA 验证及正式训练尚未开始，执行记录见 [README.md](./README.md) 和 [部署记录](./deployment_20260911_v4.json)。

## 研究问题与范围

在同一冻结的 RGB-only InfiniDepth Base、相同数据和训练预算下，将当前 spconv SSR 换成官方 Sparse3DUNet 的 disparity 适配版，是否能改善局部几何，并减轻全局误差退化？

本轮只比较两套网络主体。保留原生 normalized disparity、体素化规则、有界加法更新、视觉条件和迭代间梯度连接；不接入 log-depth、官方预训练 SSR 权重、LiDAR 或新的损失。

本轮比较包含网络结构与稀疏算子实现的整体差异，不能将收益单独归因于 LayerNorm、池化或某一个模块。两组是固定训练预算对照，不承诺 FLOPs、参数量或耗时相等。

## 两个实验组

| 项目 | A：现有 SSR | B：官方网络主体适配版 |
| --- | --- | --- |
| Base | 同一官方 RGB-only `infinidepth.ckpt`，记录 SHA-256 | 同 A |
| Base 更新 | DINOv3、BasicEncoder、ImplicitHead 全部冻结 | 同 A |
| SSR 初始化 | 重新初始化现有网络，输出层零初始化 | 重新初始化官方网络，调用官方 `init_weights()`，输出层零初始化 |
| 网络 | 当前 SpconvDisparityUNet | 官方 Sparse3DUNet |
| 通道 | 32、64、128、256、512 | 相同通道序列 |
| 残差块 | 各级两个，BN + ReLU | 官方配置：编码/解码各级一个，额外一个 bottleneck block，LN + SiLU |
| 采样与跳连 | 步幅/逆稀疏卷积，拼接融合 | 池化/最近邻上采样，相加跳连 |
| 条件特征 | 同一 DINOv3 最后一层特征，1024 通道 | 同 A；官方 `encoder_channels` 适配为 1024 |

不加载 Exp3 已训练 SSR，也不使用两个不同 Base；Exp3 原报告仅作为历史背景。这是一组新的 frozen-Base 对照，不等于重跑 Exp3 原 detach/joint 训练。

两组使用相同随机种子和独立采样器种子，逐步记录全局样本 ID。不同结构不可能拥有逐参数相同的初始权重；相同 seed 用于复现及配对采样，不表示初始化数值相同。

## 接口与代码适配

复用已在 Exp6-1/6-3 验证的官方代码基线 `74fbce054ebed49800de42d0ad0e83495065719a`。使用其 `sparse_unet.py`、`flex_sparse_blocks.py` 和必要工具，保留来源、许可证及哈希；只调整本地 import，不改官方网络的数学运算。

在现有 `DisparitySparseRefiner` 的 backend 选择中增加 `official_flex`，保持默认 `spconv` 和测试用 `reference` 行为。新增一个薄适配文件，不复制完整 MoGe 模型或另写一套 U-Net。

固定接口：

- 输入 disparity 为 `[B,384,512]`，视觉特征为 `[B,1024,24,32]`。
- 复用 `voxelize_disparity`，特征仍为 `(x_norm, y_norm, d)`，量化倍率仍为 200；不改用官方 log-depth 的 256。
- spconv 坐标 `[batch, bin, row, col]` 转为官方 `[batch, row, col, bin]`；空间 shape 同步重排为 `[B,H,W,Z,3]`。保留原有 bin offset 和空间补齐规则。
- 官方条件图的高宽必须等于查询网格的 1/16；本轮直接使用已有 24×32 特征，不补 UV、不引入额外可训练条件适配器。
- 官方 U-Net 输出 `[B×H×W,1]`，验证恢复输入像素顺序后转回 `[B,H,W]`。
- 返回现有 residual/voxel statistics 接口；没有实测的可选统计不伪造。
- 复用现有外层更新：

\[
d^{k+1}=d^k+0.1\tanh\left(r^k/0.1\right).
\]

每次迭代重新体素化，三轮共享同一个 SSR。Base disparity 和视觉特征 detach；SSR 各轮之间不额外 detach。仅导入网络主体，不复制官方 `MoGeModel.forward` 的逐轮几何 detach。

预期实现范围：官方模块及薄适配器、现有 refiner 构造入口、RGB 训练器的 opt-in `refiner_only` 模式、实验配置/启动脚本和聚焦测试。共享 exporter 按 checkpoint 中的 backend/config 恢复模型，旧 checkpoint 缺少新字段时仍默认 spconv。

## 真正冻结 Base

复用现有 RGB 训练器的加载、采样、监控、评估和恢复能力，增加显式 `refiner_only` 模式；不复制完整 Exp4 LiDAR 训练器。

- Base 参数 `requires_grad=False`，optimizer 只登记 SSR 参数。
- 整个模型先 `eval()`，然后只对 SSR 调用 `train()`；冻结 Base 的参数和 buffers，避免训练态统计或随机行为改变 K0。
- Base 编码及查询使用 `no_grad()`，SSR 前向与反向正常开启梯度。
- 评估结束后显式恢复 SSR 的 train 状态，并继续保持 Base eval。现有 `evaluate()` 只看顶层 `model.training` 的恢复方法需要适配，不能让 A 组 BN 在第一次评估后一直停留在 eval。
- 在初始、恢复和最终状态核对 Base 参数/buffer 哈希；在固定样本上检查 K0 一致性。

## 数据与训练默认配置

| 参数 | 默认值 |
| --- | --- |
| 训练数据 | 既有 manifest 中 59,542 张有效 Hypersim train；保留 Exp3 的单张无效样本排除 |
| 验证数据 | Exp3/Exp6-3 固定 Val100 |
| Test | 本轮不使用 |
| 输入 | RGB，512×384；GT 仅参与监督和评测 |
| 数据读取 | 复用已经落盘的个人缓存，按 manifest 懒加载；不重新扫描 NAS、不新增全量 DINO feature cache |
| GPU | 单卡顺序运行 A、B，优先同一物理 GPU；启动前确认空闲资源 |
| Batch | 每卡 microbatch 1，梯度累积 8，global batch 8 |
| 正式预算 | 每组 20,000 optimizer steps，约 160,000 次样本呈现，即 2.69 个训练集遍历量 |
| 训练/评估 K | 训练 0/1/2/3；评估 0/1/3/5 |
| 优化器 | AdamW，SSR lr=1e-5，weight decay=0.01，梯度裁剪 1.0 |
| 精度 | Base 沿用现有 bfloat16/float32 路径；SSR 和几何计算 float32，两组一致 |
| 随机种子 | 主配对实验 seed=173；样本顺序由独立 ShuffledCycleSampler 控制 |
| 评估 | sampled 每 500 steps，Val100 full disparity 每 2,500 steps |
| 恢复点 | 每 500 steps 原子保存有效 checkpoint；full evaluation 周期与保存周期解耦 |

为保持现有优化尺度，两组均沿用原来的损失：

\[
L=\frac{1}{4}\sum_{k=0}^{3}\left[\operatorname{MAE}_{valid}(d^k,d_{GT})+0.5\,\operatorname{GradientLoss}_{4\ scales}(d^k,d_{GT})\right].
\]

冻结后 L0 是无梯度的诊断项；保留它和 1/4 系数，避免暗中将 SSR 梯度整体放大为原来的 4/3。只训练一个 SSR 阶段，不运行 DINO 解冻或 Joint 阶段，不根据单组验证曲线临时改学习率。

2,500 steps 是第一观察点，不能仅因一组暂时较好/较差就单独延长或停止该组。硬件或数值故障可以暂停；修复如改变数值语义，建立 revision，并保持两组协议可比。

## 环境、验证和恢复

先核对已用环境与 FlexGEMM 固定版本 `b2fadb29d41846c7981ade6801ffc689fae119cf`。本轮需要同时支持 spconv、FlexGEMM 和 InfiniDepth；在实验自己的环境内验证共存，不直接升级既有训练环境。官方推理通过不等于反向训练已验证。

先完成 CPU 可测的接口与恢复测试，再在空闲 GPU 上执行独立 smoke：

1. 坐标重排可逆、负 disparity/offset 正确、输出没有错位或漏点；小网格和正式分辨率均检查。
2. 两组零初始化时 K0/K1/K3/K5 恒等；同图 K0 两组一致。
3. 两组至少完成数次真实 optimizer update；SSR 有有效梯度，Base 无梯度且参数/buffers 不变。零输出层初始化会使早期内部层梯度为零，不能要求第一步所有层均非零。
4. 在训练→评估→继续训练之后，再检查 Base eval、SSR train 和 K0 不变。
5. 保存、退出、恢复，核对下一组样本 ID、RNG、optimizer、step 和 best 元数据连续；不同 backend/config 的 checkpoint 拒绝误载。
6. 剔除 JIT/预热后，分别测量不少于 50 个正式大小 optimizer steps 的耗时和峰值显存；确认 K5 评估也能运行。
7. 非有限 loss、raw residual、梯度在 optimizer step 前终止；保留 0.1 有界残差检查，并记录残差分布、接近边界比例及 disparity voxel span。

所有 smoke 写入单独目录，不作为正式结果；通过后两组重新从相同 Base 启动。若旧单卡恢复/导出路径受共享代码影响，执行既有对应回归测试。

预计顺序训练耗时由 smoke 实测计算：

\[
T_{train}=20{,}000(t_A+t_B),
\]

其中 t 为每个 global-batch optimizer step 的秒数，另加评估和 checkpoint I/O。当前未测量，不预报具体完成日期。

## 评估与决策

主比较固定为两组各自 step 20,000 的 K3，避免事后挑不同训练时长或最有利指标。另保存按 Val100 K3 full disparity MAE 选择的 best checkpoint，作为补充，不替换固定终点主比较。

终点评测直接接入 Exp6-3 统一评测器，至少报告：

- Full normalized disparity MAE、Affine Depth Rel、Affine Point Rel。
- Local Point Rel、Local Point delta_0.01，以及 Local Depth 两项辅助指标。
- 同一模型 K3 对 K0 的变化、B 对 A 的变化、逐图改善数量和 K5 稳定性。
- 复用相同 Local masks、first-segment-wins 和官方 Local 实现；99 张非空图为 Local 均值分母，空 mask 不记零分。
- 延续 Exp6-3 的几何还原/对齐协议并标明 GT 统计仅用于评测，不进入 Base 或 SSR，不宣称原生 metric-depth 能力。
- 从已有逐图结果按场景做配对 bootstrap，给出 B-A 的 95% 区间；同时报告参数量、峰值显存及推理/训练耗时。

预登记决策：若 B 的 Local Point Rel 低于 A，且 full disparity MAE 与 Affine Point Rel 均不高于 A，则列为候选改进；还需单独检查它们是否仍差于共同 K0。若仅 Local 更好而全局变差，记录为权衡，不宣布全面替代。差异区间跨零或单 seed 收益很小则视为证据不足。

若准备据此替换默认 SSR，再按相同配置补 seed=174 的 A/B 配对复核；单 seed 主实验只支持探索性结论。该复核最多增加两组 20,000 steps，不因某一组结果不理想而单独调参。

## 产物与执行边界

- 实验目录：`experiment/exp6_4_official_ssr_disparity_adapter/`；A/B 各用独立运行目录和 checkpoint 元数据。
- 保存配置、官方源码来源、环境锁定信息、Base 哈希、初始化方式、采样顺序摘要、报告和故障记录；旧实验资产不覆盖。
- 复用现有轻量状态监控；本轮不新建 GPU 调度系统、不做跨服务器或多卡训练、不重构查看器。结果完成后可在现有查看器增加 A/B 入口。
- 更新 `experiment/log.md`，详细数值只链接报告；实验结束后按代码/结果两次提交，使用用户 Git 身份。
- 本文件是计划。GPU 分配、依赖共存验证、训练耗时和最终结果尚未执行或确认。

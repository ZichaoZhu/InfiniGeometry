# Exp6-4：官方 SSR 网络主体的 disparity 适配版代码交接

2026-09-23 收尾：官方单组配置准备与显式机器路径已实现，见[运行交接](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md)。S115 现有环境的 CUDA、冻结和短步恢复验证通过；历史权重默认严格数值对照未通过，见[验收报告](../experiment/exp6_4_official_ssr_disparity_adapter/verification_20260923/README.md)。后续一小时内授权的[独立诊断](../experiment/exp6_4_official_ssr_disparity_adapter/diagnosis_20260923_determinism/README.md)定位了本例的池化顺序问题，组合控制在单图、单步上通过；未修改默认训练方法或历史产物。

初次整理日期为 2026-09-21，当前入口与验收状态以本次更新为准。没有重跑正式实验、重评 Val100 或部署网站；全新依赖环境与多卡扩训未验收。

## 1. 交付对象与结论边界

本轮主线是 `Project/InfiniDepth` 中的 `official_flex` 分支，准确名称为“InfiniDepth＋官方 SSR 网络主体的 disparity 适配版”。`spconv` 保留为历史对照，不改全局默认 backend，不整理其他实验的训练代码。

可用于汇报的表述：

> Exp6-4 在冻结的 InfiniDepth RGB Base 后接入官方 MoGe3 的 Sparse3DUNet 网络主体，保留本项目的 normalized disparity 表示、有界加法更新、disparity 监督及迭代间梯度连接。在固定 Hypersim Val100 上，该适配版的平均误差优于现有 spconv 对照，但不是逐图、逐区域均改善。该实验不等于完整官方 MoGe3 方法或训练配方的复现。

既有实验之间的关系：

| 实验 | 已完成内容 | 不能据此声称 |
| --- | --- | --- |
| Exp6-1 | 官方 MoGe3 模型与发布权重的推理评测 | InfiniDepth 已接入完整官方 SSR 机制 |
| Exp6-4 | 冻结 InfiniDepth，比较 spconv 与官方网络主体的 disparity 适配版 | 已复现官方 log-depth 表示、更新和训练流程 |
| Exp6-5 R1 | MoGe2 基座上的两套 SSR，使用修正后的共同几何损失完成训练 | 修复已经同步到 InfiniDepth，或完整官方多任务训练已复现 |

尚未完成“以 InfiniDepth 为基座，仅做必要接口适配，并尽量保留官方几何表示、更新、逐轮梯度处理和几何监督”的实验。也未完成 Exp6-4 去限幅的受控训练对照。不能在整理时将它们记成已有结果。

## 2. 官方部分与适配部分

官方来源锁定在 MoGe 提交 `74fbce054ebed49800de42d0ad0e83495065719a`，FlexGEMM 提交为 `b2fadb29d41846c7981ade6801ffc689fae119cf`。本地保留 [SOURCE.json](../InfiniDepth/model/official_moge_ssr/SOURCE.json) 和 [LICENSE](../InfiniDepth/model/official_moge_ssr/LICENSE)。

| 项目 | 官方 MoGe3 基线 | Exp6-4 official_flex |
| --- | --- | --- |
| 稀疏网络主体 | Sparse3DUNet、FlexGEMM、官方残差块及上下采样 | 复用相同主体；本地模块只调整 import |
| 主干通道 | 32、64、128、256、512 | 相同 |
| 几何特征 | 预测的 \((X/Z,Y/Z,\log Z)\) | 固定图像网格位置与预测 disparity：\((u_{grid},v_{grid},d)\) |
| 第三维量化 | \(\operatorname{round}(256\log Z)\) | \(\operatorname{round}(200d)\) |
| 视觉条件 | DINOv2 特征＋UV，配置为 1026 通道 | InfiniDepth DINOv3 特征，1024 通道，不额外拼 UV |
| 残差解释 | 直接加到 log-depth | 经 tanh 限幅后加到 normalized disparity |
| 逐轮几何梯度 | 每轮输入 SSR 的 `current_coord.detach()` | 不额外切断 SSR 迭代间的 disparity 梯度 |
| 训练监督 | 官方三维几何及按数据类型配置的多任务损失 | disparity MAE＋四尺度梯度损失 |
| 本实验初始化 | 不适用 | 官方 InfiniDepth Base 权重＋重新初始化 SSR；不加载官方 MoGe3 SSR 权重 |

这里的“零初始化 SSR”指输出层置零、初始残差为零，不是全部网络参数都为零。条件融合层的输入维度经过适配，因此不能把整个实例的参数形状或权重与官方 checkpoint 视作完全相同。

官方对照依据为锁定提交的 [模型](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py)、[配置](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/configs/train/v3.json) 和 [训练入口](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/train/train_moge3.py)，不是本地后续修改后的 MoGe 工作区。

### 几何与更新

两条路径均使用图像行列与量化几何标量构建稀疏壳，不是直接在欧氏 XYZ 中建立稠密体素。Exp6-4 保留自身 bin offset 和空间补齐规则，再将坐标从 `[batch,bin,row,col]` 重排为官方接口的 `[batch,row,col,bin]`。

官方更新在网络内部的 log-depth 坐标上进行：

\[
\ell^{k+1}=\ell^k+r^k,\qquad Z^{k+1}=Z^k\exp(r^k).
\]

Exp6-4 使用：

\[
\Delta d^k=0.1\tanh\left(\frac{r^k}{0.1}\right),\qquad d^{k+1}=d^k+\Delta d^k.
\]

这里的 0.1 是 normalized disparity 的单位，不是米或深度变化百分比。raw residual 不受这一边界直接限制；单步 applied residual 与多轮累积更新也不能混报。

GT 监督来自逆径向深度的逐图分位归一化：

\[
d_{GT}=\frac{1/R_{GT}-q_{0.02}}{q_{0.98}-q_{0.02}}.
\]

\(R_{GT}\) 是点到相机中心的距离；分位数取自有效逆径向深度。目标不裁剪到 \([0,1]\)。GT 分位数用于监督及既有导出的尺度还原，不作为 Base 或 SSR 输入；不能将依赖 GT 的显示尺度还原说成网络独立预测了绝对米制尺度。

### 损失与参数更新

\[
L_k=\operatorname{MAE}_{valid}(d^k,d_{GT})
+0.5\,L_{gradient}^{4\ scales}(d^k,d_{GT}),\qquad
L=\frac14\sum_{k=0}^{3}L_k.
\]

- Base 为 DINOv3、BasicEncoder 与 depth implicit head，全程冻结参数和 buffers，保持 eval 状态。
- Base 前向不建立梯度图，optimizer 只登记 SSR 参数；SSR 保持 train 状态。
- 三轮共享 SSR 参数，各轮间保留可微特征和更新路径的梯度；离散体素索引本身不可微。
- \(L_0\) 是无梯度的记录项。继续保留四项平均，不擅自改为三项平均，否则会改变 SSR 梯度尺度。
- 本实验仅有一个 refiner-only 阶段；没有 DINO 解冻或 Joint。不能套用 Exp1～Exp3 的 detach→joint 描述。

### 限幅与 Exp6-5 修复的关系

[Exp6-5 R1](../experiment/exp6_4_official_ssr_disparity_adapter/ARCHIVE.md#exp6-5)修正了历史几何损失的对齐梯度路径，并让后续 K 使用 K0 的尺度；同时替换了 local 算子的相关实现细节，不是只改一个布尔值的消融。修复后 A 仍保留限幅，B 仍直接更新 logZ，两组完成了既定训练。

这支持“不应靠限幅掩盖旧损失问题”，但不证明 Exp6-4 的限幅一定有用或一定无用。Exp6-4 的训练损失不经过那条独立几何对齐路径。

本次将限幅视为历史实验兼容行为，而非官方要求或已经验证的必要设计。保留旧 checkpoint 的原推理规则；去限幅、改 log-depth、改损失或改逐轮 detach 都属于新的方法变体，需独立配置与输出目录，不能静默改写 Exp6-4。

## 3. 代码阅读与复用入口

以下链接相对本文档，指向同一 InfiniDepth 仓库。它是阅读索引，不是仅复制表中文件即可运行的独立代码包，仍需 InfiniDepth 基础模型及其依赖。

| 入口 | 职责与边界 |
| --- | --- |
| [official_moge_ssr/sparse_unet.py](../InfiniDepth/model/official_moge_ssr/sparse_unet.py) | 官方网络主体；保留同目录依赖、许可证和来源清单 |
| [official_disparity_adapter.py](../InfiniDepth/model/official_disparity_adapter.py) | 配置官方网络、坐标重排、条件图形状校验、恢复像素顺序 |
| [disparity_refiner.py](../InfiniDepth/model/disparity_refiner.py) | disparity 特征、体素化、backend 选择及残差限幅函数 |
| [model.py](../InfiniDepth/model/model.py) | `attach_disparity_refiner` 与 `forward_dense_refined`；Base 编码、K0 和迭代更新 |
| [data.py](../training/disparity_refiner/data.py)、[losses.py](../training/disparity_refiner/losses.py) | manifest、懒加载缓存、GT 归一化及 disparity 监督 |
| [train.py](../training/disparity_refiner/train.py)、[frozen_base.py](../training/disparity_refiner/frozen_base.py) | 单卡 refiner-only、参数组、Base 哈希保护、评估与恢复 |
| [automate.py](../experiment/exp6_4_official_ssr_disparity_adapter/automate.py) | 生成两组锁定配置，运行 smoke、训练和评估；不是 official-only 启动器 |
| [prepare_official.py](../experiment/exp6_4_official_ssr_disparity_adapter/prepare_official.py)、[runtime_paths.py](../training/disparity_refiner/runtime_paths.py) | 准备单组配置和机器路径，校验输出边界；不自动启动训练 |
| [evaluate_pair.py](../experiment/exp6_4_official_ssr_disparity_adapter/evaluate_pair.py) | 固定 Exp6-3 协议的 Global/Local 评估与配对分析，不用于训练 loss |
| [export_assets.py](../training/disparity_refiner/export_assets.py)、[viewer.json](../experiment/exp6_4_official_ssr_disparity_adapter/viewer.json) | 已有 checkpoint 的加载和查看器导出、选图与模型映射 |
| [test_official_adapter.py](../tests/disparity_refiner/test_official_adapter.py) | 坐标、冻结、恢复、配置保护及 CUDA 残差/像素顺序测试 |

不要重命名 `disparity_refiner.unet.network` 等 checkpoint 参数路径，也不要将全局默认 `spconv` 改成 `official_flex` 来代替显式选组。

## 4. 已完成实验的固定配置

配置依据：[config.json](../experiment/exp6_4_official_ssr_disparity_adapter/config.json)。它是 A/B 模板，默认 `refiner_backend=spconv`；自动流程生成的 `arms/official_flex/config.json` 才是官方适配组配置。不得把模板直接传给训练器后称为官方组。

| 项目 | 已完成 Exp6-4 的设置 |
| --- | --- |
| Base | 官方 RGB-only `infinidepth.ckpt`，不是 Exp3 微调 Base，也不输入 LiDAR |
| 数据 | Hypersim 有效 train 59,542 张，排除既有清单中的 1 张无有效 depth 图 |
| 验证 | Exp3/Exp6-3 固定 Val100，选模指标为 K3 平均 full disparity MAE |
| Test | 不参与训练、调参或定量结论；仅使用固定选图进行可视化 |
| 查询网格与条件图 | disparity `[B,384,512]`；视觉条件 `[B,1024,24,32]` |
| Batch | 单卡 microbatch 1，累积 8，global batch 8 |
| 预算与随机性 | 每组 20,000 optimizer steps；seed 173，相同全局采样序列 |
| 优化器 | AdamW；SSR 学习率 \(10^{-5}\)，weight decay 0.01，梯度裁剪 1.0 |
| K | 训练 0/1/2/3，评估 0/1/3/5；K5 是迭代外推 |
| 频率 | sampled evaluation/checkpoint 每 500 步，full evaluation 每 2,500 步 |

主要比较固定在两组 20,000 步终点，best 单独保留。均值改善不等于全部样本改善，也不等于全部验收门槛通过；具体判断与数值仅引用原报告。

## 5. 运行、资产与环境边界

单组入口为 `prepare_official.py`，生成显式路径配置并打印共享训练器的 `--config`、`--output`、`--resume` 命令。原 `run.sh` 保留完整 A/B 自动流程，不是只运行官方组；按 [RUNNING.md](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md) 选择入口，不直接启动历史目录。

- 历史服务器源码为 `ZJU3DV-S115` 上的 `/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/exp6_4_20260911/source_v8`，依据[结果同步清单](../experiment/exp6_4_official_ssr_disparity_adapter/results_20260914/manifest.json)。2026-09-23 已只读加载其中 official_flex 的 stage1_best.pt 验收；不代表扫描过所有历史资产，旧 PID 不用于当前进程控制。
- 环境记录见[初期环境清单](../experiment/exp6_4_official_ssr_disparity_adapter/deployment_20260911_v4.json)与[后续 CUDA/共享 GPU 验收](../experiment/exp6_4_official_ssr_disparity_adapter/deployment_20260912_shared_gpu3.json)。保留固定 FlexGEMM 版本、进程级 `FLEX_GEMM_AUTOTUNE_MODE=always` 和实验独立缓存；不把旧环境记录当成新机器安装验收。
- `evaluate_pair.py` 支持显式配置 Exp6-3 协议、Local masks 与官方 MoGe 依赖位置，缺省保留历史路径。固定协议/清单哈希校验不变；迁移时补齐依赖，不关闭校验、不重写指标。
- checkpoint 格式为 `infinidepth-disparity-refiner-v1`，保存原模型 state dict；`last.pt` 另含 optimizer、RNG、采样与阶段状态。权重加载和完整恢复不是同一操作。
- 恢复会检查配置哈希、backend、训练模式和 world size。修改路径或配置后不能通过跳过检查冒充原训练的无缝恢复。
- 当前 `refiner_only` 显式拒绝 `world_size != 1`。共享训练器存在 DDP 不代表 Exp6-4 已验证支持多卡。
- 训练权重的源码来源仍以对应部署快照和哈希为准，当前整理提交不能冒充当时的训练版本。本次仅补交本地验收与交接材料，不改写历史或推送。

查看器复用历史字段：`stage1_best` 指 spconv，`joint_best` 指 official_flex；二者实际均加载各组的 `stage1_best.pt`。这是展示映射，不代表 Exp6-4 做过 Joint。保留旧 schema 和资产，在说明中解释映射，不将其改名后破坏历史加载。

Hypersim 可视化为 train/val/test 各 5 张；其中 val/test 与 Exp6-3 的选图一致。查看器的 val 5 张不等于用于选模的 Val100，不能互相替代。

## 6. 结果入口与剩余边界

Git 仅保存紧凑报告、指标和两张汇总图。原报告或图册引用的外部图片、数组、缓存及跨仓库材料见 [ARCHIVE.md](../experiment/exp6_4_official_ssr_disparity_adapter/ARCHIVE.md)，不是全部随仓库分发的资产。

- [实验 README 与详细数值](../experiment/exp6_4_official_ssr_disparity_adapter/README.md)
- [终点配对报告](../experiment/exp6_4_official_ssr_disparity_adapter/results_20260914/paired_report.json)
- [逐图及像素级诊断](../experiment/exp6_4_official_ssr_disparity_adapter/analysis_20260914_r3/REPORT.md)
- [查看器发布与验收记录](../experiment/viewer/deployment_20260914_exp6_4_official_ssr.json)
- [实验总日志](../experiment/log.md)

方法口径、单组入口、路径配置与结果索引已整理，验收按代码和结果分组归档。2026-09-21 的提交检查见 [COMMIT_CHECKS.md](../experiment/exp6_4_official_ssr_disparity_adapter/COMMIT_CHECKS.md)，新增真实 CUDA/恢复证据见[2026-09-23 验收](../experiment/exp6_4_official_ssr_disparity_adapter/verification_20260923/README.md)，不混用两次检查的时点。

1. 已完成 51 项相关 CPU、10 项 CUDA 测试，冻结及真实 checkpoint 的 RNG/采样/optimizer 恢复通过。
2. 默认 K0 精确一致，K1/3/5 严格对照未通过。后续诊断定位本例首次差异在稀疏池化的邻居枚举/求和顺序；临时固定池化坐标加 PyTorch 确定性设置后，预测、梯度和一次更新跨进程精确一致。仅验证一张图，未隔离全部反向算子、做多图或长训练验证；默认实现未更换，原限制不能删除。
3. 尚未验收全新机器安装环境、多卡 refiner-only 或新导出部署。接收方需确认外部资产权限，并先运行目标环境短测，不能将本轮交接称为多卡即开即用。

多 GPU、解冻 Base、去限幅、官方 log-depth 机制或几何损失移植属于独立适配或实验 revision，不混入无行为变化的代码整理。是否推进由后续实验目标决定，本轮不自动启动。

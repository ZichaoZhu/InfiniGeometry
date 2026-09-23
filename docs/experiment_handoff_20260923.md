# 实验代码交接索引

日期：2026-09-23。Exp6-4 单组入口、Exp6-5 R1 与近期可视化增量已整理；同日获授权在 S115 独立目录执行 CUDA 短测、恢复验收及限时数值诊断，并取回正式源码。本轮整理没有重训正式实验、追加 benchmark 或部署网站；复用原训练器和查看器，不改变默认方法或数据。

初次本地收尾时：功能/恢复通过，历史权重严格跨进程数值对照未通过，当时暂不继续诊断。随后按用户一小时内的授权完成[稀疏非确定性诊断](../experiment/exp6_4_official_ssr_disparity_adapter/diagnosis_20260923_determinism/README.md)：已定位所测样本的前向差异从第一次稀疏池化的邻居顺序开始；临时固定坐标并组合 PyTorch 确定性设置后，单图预测、梯度和一次更新均跨进程精确一致。正式默认设置未改，原失败记录保留，不宣称多图或完整训练均确定。

同步状态：此前两个仓库的交付分支已获授权推送，InfiniDepth 包含链接修复提交 `24d5e12`，MoGe 为 `ad9e4f0`。本次补充的实验总览、数值诊断和交付总结按代码/结果完成本地提交，尚未推送；权重、数据和外部源码包不随 Git 推送。

## 给师兄的摘要

- 推荐以 Exp6-4 official_flex 为扩训主线：在冻结的官方 InfiniDepth RGB Base 上，仅训练 SSR；已有固定 Hypersim Val100 对照支持其平均误差优于现有 spconv，但不是所有图片和局部指标均获益。
- 实际交付的是官方稀疏网络主体的 disparity 适配版，仍保留本项目的几何表示、限幅、损失和迭代间梯度；不是完整 MoGe3 机制或官方 SSR 权重的直接移植。Exp6-1～5 的目的、结果与限制见下表。
- 功能与短步恢复验收通过；默认数值不确定性已在单例中定位到稀疏池化顺序。独立组合控制通过单图/单步检查，但未应用到正式实现，也没有证明长期训练完全确定。
- 接手时先按[运行说明](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md)核对权重、数据和评估资产权限，完成目标环境单图推理及短步恢复；当前 Exp6-4 refiner-only 仅支持单卡，多卡扩训需另行适配和验证。

## Exp6-1～Exp6-5 分别在做什么

总体路线：先确认官方模型能否复现，再比较不同基座的表现，最后分别在 InfiniDepth 和 MoGe2 基座上比较 SSR 实现。下面是实验本身的目的与结果，不是本次代码整理的工作清单；详细指标保留在原报告。

| 实验 | 要回答的问题 | 实际做法 | 已有结论与边界 |
| --- | --- | --- | --- |
| [Exp6-1：官方权重复现](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_1_moge3_official_reproduction/README.md) | 官方 MoGe3 在我们的环境中能否稳定运行，SSR 精修效果如何？ | 不训练，直接用官方代码与发布的 ViT-L 权重完成 10 个可用数据集的评测和独立复核；后续加入人工挑选的跨数据集难例，与 InfiniDepth Exp3 RGB 权重作可视化对照。 | 推理、评测与复核完成，整体指标收益较小但可观察局部变化；人工难例只用于定性诊断。不是我们训练出的 MoGe3。 |
| [Exp6-2：论文结果可比性](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_2_moge3_paper_comparability/README.md) | Exp6-1 的结果与论文公开表格是否可比？ | 复用 Exp6-1 结果，对齐指标单位与同名数据集，不新增训练。 | 数据集构成不同，不能直接比较总均值；只比较重叠集。缺少官方 Local masks，未严格复现论文 Local 数值。 |
| [Exp6-3：同图 RGB 对照](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_3_moge3_vs_infinidepth_exp3/README.md) | 同一输入和评测口径下，官方 MoGe3 与我们的 InfiniDepth＋SSR 表现有何差别？ | 不新增训练，在固定 Hypersim Val100 上比较官方 MoGe3 与 Exp3 Stage1/Joint best；统一 RGB、分辨率和几何指标，补充固定细节区域评测。 | MoGe3 的主要全局及 Local 相对误差更低；Exp3 有局部改善但也有全局退化。基座与训练经历不同，不能将差距只归因于 SSR；自定义 Local 不是论文 Local 复现。 |
| [Exp6-4：InfiniDepth 上的 SSR 对照](../experiment/exp6_4_official_ssr_disparity_adapter/README.md) | 固定 InfiniDepth 基座后，换用官方稀疏网络主体是否优于现有 spconv SSR？ | 从同一官方 RGB-only InfiniDepth 权重开始，Base 全程冻结；两组使用相同 disparity 表示、损失及采样，SSR 分别训练 20,000 步，在 Hypersim Val100 上配对比较。 | official_flex 的 K3 平均 disparity MAE、Global Rel 和 Local Point Rel 更低，但并非逐图都好，Local Depth Rel 的差值区间跨 0。当前扩训主线；不是完整官方 MoGe3 方法。 |
| [Exp6-5：MoGe2 上的 SSR 对照](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_5_moge2_ssr_vs_official_moge3/HANDOFF.md) | 在共同 MoGe2 初始化、数据和预算下，自实现 SSR 与官方 SSR 流程有何差别？ | 正式 R1 两组都从共同 MoGe2 权重和新初始化 SSR 开始，在 Hypersim 上完成 Stage1 20,000＋Joint 10,000 步；共同采用修正后的几何损失与 K0 尺度锚定。 | 两组训练完成，35 图可视化已发布；训练中有 Val100 评估与选模，完成后按要求未追加最终 benchmark/Local/test 定量评估。不是只替换 U-Net 的单变量消融，不能仅凭展示宣称普遍胜出。 |

Exp6-1 最终的[二十张难例](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_1_moge3_official_reproduction/04_checkpoint_visualization/HARD20_RESULTS.md)与 Exp6-5 的[四窗口展示](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_5_moge2_ssr_vs_official_moge3/VISUALIZATION.md)应分开理解：Exp6-1 的 MoGe3 使用官方发布权重；Exp6-5 的 D 窗口是本次 Hypersim 训练的官方 SSR 实现组，不是官方发布权重。跨数据集图片未参与本次 Hypersim 微调，不等于已证明它们未参与任何基座预训练；不笼统称全部图片为严格 OOD。

Exp6-4/5 都使用 59,542 张有效 Hypersim train，验证沿用 Exp3 Val100。展示中的 train/val/test 各 5 张不是完整数据集，test 不用于训练或 checkpoint 选择。Exp6-4 的 Base 是原始官方 InfiniDepth，不是 Exp6-3 中的 Exp3 微调 Base；不能把两个实验的 K0 当成相同起点。

## Exp6-4 与官方 MoGe3 SSR 的差别

准确名称是“InfiniDepth＋官方 SSR 网络主体的 disparity 适配版”。复用的是 Sparse3DUNet、FlexGEMM、残差块及上下采样结构，不是整套官方输入、更新、训练机制或已训练权重。官方对照固定到 MoGe 提交 `74fbce054ebed49800de42d0ad0e83495065719a`，详细源码与配置证据见[方法边界](./exp6_4_disparity_adapter_handoff.md)。

| 方面 | 官方 MoGe3 | Exp6-4 official_flex |
| --- | --- | --- |
| Base 与视觉条件 | MoGe2 架构基座；DINOv2 特征拼 UV，1026 通道 | InfiniDepth：DINOv3＋BasicEncoder＋implicit head；1024 通道条件，不额外拼 UV |
| SSR 网络主体 | 官方 Sparse3DUNet 与 FlexGEMM | 复用主体；适配条件通道和坐标布局，不加载官方 MoGe3 SSR 权重 |
| 几何特征与量化 | 预测的 \((X/Z,Y/Z,\log Z)\)，第三维量化为 \(\operatorname{round}(256\log Z)\) | 固定网格与 normalized disparity：\((u_{grid},v_{grid},d)\)，第三维量化为 \(\operatorname{round}(200d)\) |
| 残差更新 | 直接加到 log-depth，无这里的 tanh 限幅 | 经 0.1 tanh 限幅后加到 normalized disparity |
| 逐轮梯度 | 每轮送入 SSR 的几何执行 `current_coord.detach()`；视觉条件按开关决定是否 detach | Base 冻结，但 SSR 迭代间的可微 disparity 路径保持连接；离散体素索引不可微 |
| 训练监督 | 官方几何损失，并按数据类型配置其他任务监督 | 有效像素 disparity MAE＋四尺度梯度损失，对 K0～K3 等权平均 |
| 本次训练范围 | 官方机制不能简化为“只训练 SSR” | 只训练 SSR；DINO、BasicEncoder、depth head 全程冻结，没有解冻或 Joint；K0 损失不反传 |

两种更新的含义不同，不能把 residual 数值直接横向比较：

\[
\text{MoGe3:}\quad \ell^{k+1}=\ell^k+r^k,\qquad Z^{k+1}=Z^k\exp(r^k),\quad \ell=\log Z.
\]

\[
\text{Exp6-4:}\quad d^{k+1}=d^k+0.1\tanh\left(\frac{r^k}{0.1}\right).
\]

这里的 \(Z\) 是官方网络内部、相机恢复前的深度坐标，不直接等于最终米制深度；\(d\) 是归一化逆径向深度，0.1 不是米或深度变化百分比。GT 的 2%/98% 分位数用于 disparity 监督和既有显示尺度还原，不作为 SSR 输入。

Exp6-5 R1 修复了其共同几何损失的对齐梯度与尺度锚定；A 仍有限幅，B 仍直接更新 logZ。Exp6-4 不经过那条训练损失路径，不能据此说它已去掉限幅，或已验证限幅一定必要/不必要。保留历史推理规则；去限幅、改 log-depth、逐轮 detach 或损失都应作为独立变体验证。

## 交付状态

| 范围 | 本轮完成 | 仍需注意 |
| --- | --- | --- |
| Exp6-4 | 单组入口、CUDA/恢复验收；限时定位本例池化顺序差异，临时组合控制通过 | 默认严格一致性仍未通过，诊断控制未用于正式代码；仍限单卡，未验收全新依赖环境 |
| Exp6-5 | R1 归档；正式 A/B 各 141 文件核验，原始清单/源码已取回 | 源码包单独交付，不包含权重、数据和运行环境 |
| Exp6-1 | 二十张难例、选图、RGB 对照导出及结果归档 | 大型点云、图片、原日志和 trace 为外部资产 |
| Exp6-3 | 补齐 MoGe 仓库的已有导出入口 | 本轮未重新运行模型导出 |
| 共享查看器 | 多实验三/四窗口、可取消加载与释放、测试输出隔离 | 本轮只验证本地已有资产，不是新部署 |

Exp1–Exp5、Exp6-2 继续使用原归档。本轮未实现多卡、去限幅或完整官方 SSR 机制移植。

## 从哪里开始

- [Exp6-4 单组运行](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md)：配置准备默认不启动任务，说明短测、正式启动、恢复和评估命令。
- [Exp6-4 方法边界](./exp6_4_disparity_adapter_handoff.md)：官方网络主体与本项目 disparity 适配的差别。
- [Exp6-5 R1 交接](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_5_moge2_ssr_vs_official_moge3/HANDOFF.md)：旧失败、pilot、正式 R1 的区分和外部源码依赖。
- [Exp6-1 归档](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_1_moge3_official_reproduction/04_checkpoint_visualization/ARCHIVE.md)与[Exp6-5 归档](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_5_moge2_ssr_vs_official_moge3/ARCHIVE.md)：哪些文件随 Git、哪些需补齐。
- [查看器说明](../experiment/viewer/README.md)与[本地验收](../experiment/viewer/acceptance_20260923.json)。

跨仓库文档使用 GitHub 完整链接，固定到已推送的 MoGe 交付提交 `ad9e4f0`，不依赖本地相邻目录布局。仅克隆其中一个仓库不能替代另一侧代码、历史源码包、数据、权重或公开点云资产。

## 配套提交

全部使用用户 Git 身份，未改写既有历史。下列为此前代码/结果提交；本次数值诊断代码先提交，新增报告、实验总览和交付总结随后作为结果提交保存。

| 仓库与范围 | 代码 | 结果/说明 |
| --- | --- | --- |
| InfiniDepth：Exp6-4 | `993e22c` | `542da9e` |
| MoGe-v3-reproduction：Exp6-1 | `b0a89ae` | `4e2e588` |
| MoGe-v3-reproduction：Exp6-3 | `97a7b81` | 无新实验结果，沿用已有报告 |
| MoGe-v3-reproduction：Exp6-5 | `90644e9` | `5a6659b` |
| InfiniDepth：共享查看器 | `00fcd58` | `30a2f89` |
| InfiniDepth：Exp6-4 验收收尾 | `fca46fd` | `7f691e2` |
| MoGe-v3-reproduction：Exp6-5 正式源码核验 | 无活动源码修改 | `ad9e4f0` |
| InfiniDepth：Exp6-4 数值诊断与交付总结 | `8e2dac9` | 本文与诊断 evidence 所在结果提交 |

MoGe 分支为 `experiment/exp6-moge3-reproduction`，InfiniDepth 分支为 `feature/infinidepth-disparity-refiner`；此前均已建立对应 origin 分支。本次新增提交仍仅在本地，未再次推送或修改其他分支。

## 初次整理验证（本地）

- Exp6-4：7 项标准库测试通过，科学配置与严格恢复检查保留；CLI 检查不启动进程、不写配置。
- Exp6-5：源码核验 2、可视化协议 3、并行控制 5、资源准入 7 项测试通过。
- Exp6-1：难例调度 2 项、图库 3 项通过；Pillow 相关 1 项跳过。导出器测试缺少 cv2 而无法导入，模型对照及完整训练/损失测试缺少 torch 等依赖，未重跑。
- 查看器：类型检查、22 项单元测试、生产构建、8 项浏览器回归通过。8 项中 1 项使用图库 fixture，其余使用真实资产，覆盖 Exp6-1 全20图、Exp6-5 全35图、Exp6-3/4及旧版三窗口。
- Python AST、Shell 语法、提交空白和结果 JSON/JSONL 检查；历史产物按清单逐字节核验，不改变原数值。新截图和内存记录保存在本次临时测试目录，汇总写入独立验收 JSON。

本机临时目录 `/private/tmp/infinigeometry-handoff.SDx3It` 保存工作区初始哈希基线与本次浏览器材料，不是长期服务器资产。报告与可重复命令已进入仓库；历史源文件、结果和外部资产未删除。

## 后续授权验证结果与剩余边界

1. [Exp6-4 CUDA 与恢复验收](../experiment/exp6_4_official_ssr_disparity_adapter/verification_20260923/README.md)：51 项 CPU、10 项 CUDA 测试通过；真实 3→6 步恢复及 RNG/采样通过。历史权重 K0 精确一致，但 K1/3/5 未通过 \(10^{-6}\) 对照；同版本重复运行也有差异，保留失败证据，不标为严格数值复现通过。
2. [Exp6-5 正式源码核验](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_5_moge2_ssr_vs_official_moge3/results/20260923_source_verification/README.md)：正式 A/B 清单均与历史 SHA 一致，各 141 文件通过，下载后再次通过；原始完整清单入库材料已补齐，源码包保留本地外部资产。
3. [限时数值诊断](../experiment/exp6_4_official_ssr_disparity_adapter/diagnosis_20260923_determinism/README.md)：8 个模型探针、3 项 CPU 统计自测完成；原选核清单一致，默认同进程/跨进程仍有差异。临时固定池化坐标并开启 PyTorch 确定性后，单图预测、梯度和一次更新精确一致；不将单例控制结果记为正式模型已修复。
4. 实际 scaling up、多卡适配、跨机器旧训练恢复迁移及新发布仍是独立任务。本次复用原 S115 venv，并非全新机器/依赖环境验收；未修改正式训练目标和默认稀疏算子配置。新增验收材料随本次本地提交归档，不推送。

## 给扩训接收方的交付清单

主线入口是 [Exp6-4 单组运行](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md)，不是把全部历史实验或网站作为训练依赖。先阅读[方法边界](./exp6_4_disparity_adapter_handoff.md)和验收限制，再准备个人路径配置、执行检查与目标环境短测。

| 外部资产 | 来源/交付位置 | 当前边界 |
| --- | --- | --- |
| 官方 InfiniDepth 初始权重 | [原配置](../experiment/exp6_4_official_ssr_disparity_adapter/config.json) 的 model.checkpoint | 不进入 Git；单组新训练从该 Base 开始 |
| Exp6-4 训练权重 | S115 历史 source_v8 下本实验 runs/official_flex/checkpoints，详见方法边界与验收 provenance | best 用于推理；完整恢复使用匹配配置的 last.pt |
| Hypersim 数据及缓存 | 原配置的 data.source_root、data.local_cache；[固定 manifest](../experiment/data/hypersim_full_manifest.json) | 数据不复制入 Git，不改变 train/val/test 划分 |
| 固定评估依赖 | [运行说明](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md)中的 Exp6-3 协议、Local masks 与官方 MoGe 路径 | 需单独提供，保留原哈希校验 |
| Exp6-5 正式源码包（可选对照） | MoGe 仓库本实验 source_archive/formal_26e6540a.tar.gz，见[核验记录](https://github.com/ZichaoZhu/MoGe/blob/ad9e4f0230c2380fdd4323cfeea8c27141ca8680/experiment/exp6_5_moge2_ssr_vs_official_moge3/results/20260923_source_verification/README.md) | 本地已取回且核验；外部资产单独交付 |

上述是交付位置记录，不代表已验证学长账号的访问权限。实际共享、传输、远程推送和接收方环境验收仍需协调；本次本地收尾不替接收方开通权限或启动训练。

InfiniDepth 总日志中原有一处与本任务无关的文本改动（Exp4 历史句尾的 `10`）保留在工作区、不纳入本轮提交。它不影响训练或查看器；不为获得干净状态擅自删除用户修改。

## 本地提交收尾检查

此前收尾重新执行 7 项路径准备测试、2 项源码核验工具测试，均通过；核验源码包内 141 个文件、原始清单及配置哈希，检查报告 JSON/JSONL、Python AST、文档链接和 Git 空白。复核已有 51 项 CPU、10 项 CUDA 的 XML 与恢复报告，不把证据复核记为再次执行 GPU 测试。

本次新增收尾重新执行 7 项路径准备测试、2 项源码核验工具测试，均通过；复核 40 个诊断证据文件的原始哈希、5 个脚本 AST、41 个 JSON 的有限性、文档链接及 Git 差异，不重跑 GPU 探针。诊断 scripts 仅用于独立复测，不能代替正式训练入口；evidence 是历史运行记录，不回写其配置、脚本、路径或数值。代码和结果分开提交，数据、权重、大张量与编译缓存不入库，用户原有日志修改不纳入提交。

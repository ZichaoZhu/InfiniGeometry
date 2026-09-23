# 实验代码交接索引

日期：2026-09-23。Exp6-4 单组入口、Exp6-5 R1 与近期可视化增量已整理；同日获授权在 S115 独立目录执行 CUDA 短测与恢复验收，并取回正式源码。没有重训正式实验、追加 benchmark、部署或推送。沿用 ponytail，复用原训练器和查看器，不改变已有方法或数据。

当前按现状完成本地提交收尾：功能/恢复通过，历史权重严格跨进程数值对照未通过。用户决定暂不继续非确定性诊断；保留失败证据，不声称根因已确认或全部验收通过。

## 交付状态

| 范围 | 本轮完成 | 仍需注意 |
| --- | --- | --- |
| Exp6-4 | 单组入口、路径保护、CUDA 功能与短步恢复验收 | 默认稀疏算子的严格跨进程数值一致性未通过；仍限单卡，未验收全新依赖环境 |
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

全部使用用户 Git 身份，未改写既有历史。下列为本轮代码/结果提交；本索引及查看器验收记录随 InfiniDepth 后续的交接记录提交保存。

| 仓库与范围 | 代码 | 结果/说明 |
| --- | --- | --- |
| InfiniDepth：Exp6-4 | `993e22c` | `542da9e` |
| MoGe-v3-reproduction：Exp6-1 | `b0a89ae` | `4e2e588` |
| MoGe-v3-reproduction：Exp6-3 | `97a7b81` | 无新实验结果，沿用已有报告 |
| MoGe-v3-reproduction：Exp6-5 | `90644e9` | `5a6659b` |
| InfiniDepth：共享查看器 | `00fcd58` | `30a2f89` |
| InfiniDepth：Exp6-4 验收收尾 | `fca46fd` | 本索引随验收结果提交保存 |
| MoGe-v3-reproduction：Exp6-5 正式源码核验 | 无活动源码修改 | `ad9e4f0` |

MoGe 分支为 `experiment/exp6-moge3-reproduction`，InfiniDepth 分支为 `feature/infinidepth-disparity-refiner`。这是本地交付，后续推送需确认各仓库目标分支，不自动创建或覆盖远程分支。

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
3. 实际 scaling up、多卡适配、跨机器旧训练恢复迁移及新发布仍是独立任务。本次复用原 S115 venv，并非全新机器/依赖环境验收；未修改正式训练目标和默认稀疏算子配置。新增验收材料随本次本地提交归档，不推送。

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

收尾时重新执行 7 项路径准备测试、2 项源码核验工具测试，均通过；核验源码包内 141 个文件、原始清单及配置哈希，检查报告 JSON/JSONL、Python AST、文档链接和 Git 空白。复核已有 51 项 CPU、10 项 CUDA 的 XML 与恢复报告，不把证据复核记为再次执行 GPU 测试。

验收 evidence 下的配置和脚本副本属于历史运行 provenance，随结果归档；活动入口仍是实验根目录的脚本。所有原始数值与证据字节保持不变，数据、权重和大资产不进入新增提交。三个收尾提交均使用用户身份，不改写历史、不推送。

# 实验代码交接索引

日期：2026-09-23。本轮整理 Exp6-4 单组入口、Exp6-5 R1 与近期可视化增量，未重训、追加评估、访问服务器、部署或推送。沿用 ponytail 的最小改动方式，复用原训练器和查看器；不整理其他历史模型仓库、不改变已有方法或数据。

## 交付状态

| 范围 | 本轮完成 | 仍需注意 |
| --- | --- | --- |
| Exp6-4 | official_flex 单组配置准备、机器路径、安全保护与运行说明 | 真实 checkpoint 数值一致性、完整 PyTorch/CUDA 与新环境恢复验收待执行；仍限单卡 |
| Exp6-5 | R1 训练/展示代码、最终协议、诊断和轻量结果归档；来源核验工具 | 正式不可变源码包和完整 source manifest 尚未同步到本地，只持有其哈希与运行记录 |
| Exp6-1 | 二十张难例、选图、RGB 对照导出及结果归档 | 大型点云、图片、原日志和 trace 为外部资产 |
| Exp6-3 | 补齐 MoGe 仓库的已有导出入口 | 本轮未重新运行模型导出 |
| 共享查看器 | 多实验三/四窗口、可取消加载与释放、测试输出隔离 | 本轮只验证本地已有资产，不是新部署 |

Exp1–Exp5、Exp6-2 继续使用原归档。本轮未实现多卡、去限幅或完整官方 SSR 机制移植。

## 从哪里开始

- [Exp6-4 单组运行](../experiment/exp6_4_official_ssr_disparity_adapter/RUNNING.md)：配置准备默认不启动任务，说明短测、正式启动、恢复和评估命令。
- [Exp6-4 方法边界](./exp6_4_disparity_adapter_handoff.md)：官方网络主体与本项目 disparity 适配的差别。
- [Exp6-5 R1 交接](../../MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3/HANDOFF.md)：旧失败、pilot、正式 R1 的区分和外部源码依赖。
- [Exp6-1 归档](../../MoGe-v3-reproduction/experiment/exp6_1_moge3_official_reproduction/04_checkpoint_visualization/ARCHIVE.md)与[Exp6-5 归档](../../MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3/ARCHIVE.md)：哪些文件随 Git、哪些需补齐。
- [查看器说明](../experiment/viewer/README.md)与[本地验收](../experiment/viewer/acceptance_20260923.json)。

跨仓库链接要求两个仓库仍同位于 `Project/` 下。仅克隆其中一个仓库不能替代另一侧代码、历史源码包、数据、权重或公开点云资产。

## 配套提交

全部使用用户 Git 身份，未改写既有历史。下列为本轮代码/结果提交；本索引及查看器验收记录随 InfiniDepth 后续的交接记录提交保存。

| 仓库与范围 | 代码 | 结果/说明 |
| --- | --- | --- |
| InfiniDepth：Exp6-4 | `993e22c` | `542da9e` |
| MoGe-v3-reproduction：Exp6-1 | `b0a89ae` | `4e2e588` |
| MoGe-v3-reproduction：Exp6-3 | `97a7b81` | 无新实验结果，沿用已有报告 |
| MoGe-v3-reproduction：Exp6-5 | `90644e9` | `5a6659b` |
| InfiniDepth：共享查看器 | `00fcd58` | 见本索引与 acceptance_20260923.json |

MoGe 分支为 `experiment/exp6-moge3-reproduction`，InfiniDepth 分支为 `feature/infinidepth-disparity-refiner`。这是本地交付，后续推送需确认各仓库目标分支，不自动创建或覆盖远程分支。

## 本次验证

- Exp6-4：7 项标准库测试通过，科学配置与严格恢复检查保留；CLI 检查不启动进程、不写配置。
- Exp6-5：源码核验 2、可视化协议 3、并行控制 5、资源准入 7 项测试通过。
- Exp6-1：难例调度 2 项、图库 3 项通过；Pillow 相关 1 项跳过。导出器测试缺少 cv2 而无法导入，模型对照及完整训练/损失测试缺少 torch 等依赖，未重跑。
- 查看器：类型检查、22 项单元测试、生产构建、8 项浏览器回归通过。8 项中 1 项使用图库 fixture，其余使用真实资产，覆盖 Exp6-1 全20图、Exp6-5 全35图、Exp6-3/4及旧版三窗口。
- Python AST、Shell 语法、提交空白和结果 JSON/JSONL 检查；历史产物按清单逐字节核验，不改变原数值。新截图和内存记录保存在本次临时测试目录，汇总写入独立验收 JSON。

本机临时目录 `/private/tmp/infinigeometry-handoff.SDx3It` 保存工作区初始哈希基线与本次浏览器材料，不是长期服务器资产。报告与可重复命令已进入仓库；历史源文件、结果和外部资产未删除。

## 尚未完成的验收

1. 在获授权环境，对 Exp6-4 同输入、同 checkpoint 比较 K0/1/3/5、冻结、梯度、短步恢复及采样序列。不得以路径测试或旧 CUDA 报告代替。
2. 获授权后取得 Exp6-5 最终正式源码包及清单，核对既有哈希，再使用只读 verify_source 工具。当前 pilot 清单不能替代正式清单。
3. 实际 scaling up、多卡适配、跨机器旧训练恢复迁移及新发布仍是独立任务。

InfiniDepth 总日志中原有一处与本任务无关的文本改动（Exp4 历史句尾的 `10`）保留在工作区、不纳入本轮提交。它不影响训练或查看器；不为获得干净状态擅自删除用户修改。

# InfiniDepth 实验登记表

实验编号在本项目内全局递增且不得复用。仅当研究问题、主要变量、数据范围或
训练策略发生变化时，才创建新的顶层实验目录。机器故障或等价重跑仍归入原
实验，并记录在该实验的来源信息中。

| 编号 | 研究问题 | 状态 | 基线提交 | 运行提交 | 主要指标 | 结论 |
|---|---|---|---|---|---|---|
| [exp1](exp1_infinidepth_ssr_fixed_grid_mvp/README.md) | 在单个 Hypersim 样本上，固定网格 SSR 能否改善冻结的官方 InfiniDepth 基线？ | `completed` | `36c6e0c31887fafc210184ee43ca475230704095` | `ae887f8c7c9f3ac857a0aec33e38f4e7cb622a52` | K1 Point Rel | 第 200 步通过验收：K1 改善 1.017%，几何损失下降 15.19%。 |
| [exp2](exp2_infinidepth_ssr_heldout_generalization/README.md) | exp1 的冻结 SSR checkpoint 能否改善未见 Hypersim 场景？ | `completed` | `36c6e0c31887fafc210184ee43ca475230704095` | `bed6db49b4f90ee3c7d06a5115f0c2d997f02551` | K1 平均 Point Rel | 未通过：平均指标略差 0.002%，样本胜率 53.33%，未证明跨场景泛化。 |

状态只允许使用 `planned`、`running`、`completed` 和 `failed`。尚未实际运行的
实验不得描述为已完成。

提交实验记录前运行 `python experiment/validate_experiment.py`。大型 checkpoint、
点云和日志只保留在服务器的同名实验目录中，并通过
`artifacts/manifest.json` 记录校验和。在实际保存这些资产的服务器上，增加
`--require-untracked-assets` 参数，以确认所有服务器专属文件均存在且与清单一致。

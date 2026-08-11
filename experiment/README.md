# InfiniDepth 实验索引

本目录只管理 InfiniDepth 原生 disparity Refiner 实验。编号按登记顺序全局递增，不能重排、复用或用机器故障重跑占用新编号。所有可读文档使用中文；训练域实验不得表述为泛化验证。

| 编号 | 状态 | 研究问题 | 代码提交 | 主指标 | 结论 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| exp1 | completed | 固定网格 disparity 稀疏三维 Refiner 能否在五张单图上完成稳定过拟合？ | `e38c65d105bc` | K3 全图 MAE + 锁定细结构 MAE | 5/5 个单图运行通过 1% 门槛，可登记 Exp2 | [exp1](exp1_infinidepth_disparity_ssr_single_image_overfit/) |
| exp2 | completed | 单图验证通过后，固定网格 disparity Refiner 能否扩展到 100 张 Hypersim 训练图？ | `26393474014e` | K3 全图 MAE + 锁定细结构 MAE | 百图训练正常完成，平均综合分数改善 23.98%，94/100 张改善 | [exp2](exp2_infinidepth_disparity_ssr_hypersim100_overfit/) |

状态只允许 `planned`、`running`、`completed`、`failed`。Exp2 已在 Exp1 的五个独立运行全部结束且通过登记门槛后完成；结果仅表示固定百图训练域的拟合能力，不作泛化性声明。

## 资产规则

- Git 只保存代码、中文文档、配置、紧凑 JSON/JSONL 指标、manifest 和最终两张汇总图。
- checkpoint、PLY、原始日志、查看器数据和临时图仅位于实验室服务器个人目录，由 `.gitignore` 排除。
- 每个运行最多保留 `stage1_best.pt`、`joint_best.pt`、`last.pt`；相同 SHA-256 必须登记为别名。
- Hypersim 原始 RGB/depth 只从 `/nas1/datasets/hypersim/raw` 读取，禁止复制到本目录。
- 环境、缓存和临时文件必须位于 `/mnt/data/home/zhuzichao/` 下的个人目录，不使用共享 `/tmp`。

## 验证

在服务器仓库根目录执行：

```bash
python experiment/validate_experiment.py --check-git
```

验证器会检查编号、状态、必需字段、运行登记、资产 SHA、重复内容、大文件误提交、空目录和路径边界。

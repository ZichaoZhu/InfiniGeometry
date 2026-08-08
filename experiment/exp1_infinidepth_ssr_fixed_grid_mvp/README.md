# exp1：InfiniDepth SSR 固定网格最小可行实验

## 实验假设

在冻结的官方 InfiniDepth RGB 模型后接入零初始化 SSR，可以在一个固定的
Hypersim 样本上学习有效的单步几何修正，同时不改变 K0，也不把真值引入推理
路径。

## 唯一受控变量

官方 InfiniDepth 基础模型与 MoGe2 尺度参考模型始终冻结。唯一可训练组件是新
初始化的 SSR，并在固定的 `384x512` 网格上以 `K=1` 进行训练。

## 验收标准

- 所选 checkpoint 的 K1 Point Rel 相比 K0 至少改善 1%。
- 确定性几何损失相比初始化至少下降 5%。
- K0 tensor 的 SHA-256 在整个训练过程中保持不变。
- Base 和 MoGe2 参数始终不接收梯度。
- 保存并重新加载后的结果满足 `rtol=1e-6`、`atol=1e-7`。

## 当前状态

实验于 2026-08-07 完成并通过验收。第 200 步得到所选 checkpoint：K1 Point Rel
从 `0.06967275` 改善到 `0.06896392`，相对改善 `1.017%`；确定性几何损失从
`0.43732962` 下降到 `0.37090552`，相对下降 `15.19%`。作为诊断指标，K3
Point Rel 为 `0.06822955`。

K0 保持逐位稳定；适配层和零初始化 SSR 的最大绝对回归误差均为 `0`；冻结的
Base/MoGe2 梯度始终为 `None`；checkpoint 重载结果满足规定容差。两个主要阈值
首次同时满足时，训练自动停止。完整结构化数值见 `metrics/report.json`；此前
两次运行故障及其修复均保留在同一实验编号的 `provenance.json` 中。

## 结论与局限

固定网格 SSR 最小可行实验在不改变冻结 InfiniDepth 预测路径的前提下，满足了
本实验范围内的验收标准。本实验只在单个样本上训练，只能说明优化过程可行，
不能证明其能够泛化到未参与训练的 Hypersim 场景或其他数据集。

## 复现方法

在 `ZJU3DV-S115` 的服务器代码目录中运行：

```bash
bash experiment/exp1_infinidepth_ssr_fixed_grid_mvp/run.sh
```

启动脚本会拒绝任何位于 `/mnt/data/home/zhuzichao/` 之外的输出路径。运行完成后，
使用以下命令严格验证服务器保留资产：

```bash
python experiment/validate_experiment.py --require-untracked-assets \
  experiment/exp1_infinidepth_ssr_fixed_grid_mvp
```

## 资产保留规则

只允许保留 `best.pt`、`last.pt`、`gt.ply`、`k0.ply`、`k1.ply`、`k3.ply`、一张
训练曲线、一张几何对比图和一份原始运行日志。大型资产只保留在服务器，并在
`artifacts/manifest.json` 中登记。

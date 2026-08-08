# exp2：InfiniDepth SSR 未见场景泛化评估

## 实验假设

在 exp1 单样本训练得到的 SSR checkpoint 完全冻结后，它仍能在未参与训练的
Hypersim 场景上稳定改善 InfiniDepth K0，说明学到的修正不只是对训练样本的
记忆。

## 唯一受控变量

模型、checkpoint、输入分辨率、MoGe2 尺度参考和 K 值均保持不变，只把评估数据
从 exp1 的训练样本替换为 10 个未见场景中的 30 个固定样本。实验不创建
optimizer，不更新任何参数。

## 固定评估集

- 场景：`ai_001_001`、`ai_005_001`、`ai_010_001`、`ai_015_001`、
  `ai_021_001`、`ai_026_001`、`ai_031_001`、`ai_036_001`、
  `ai_041_001`、`ai_044_001`。
- 每个场景固定使用 `cam_00` 的 `0000`、`0025`、`0050` 三帧。
- exp1 训练样本 `ai_053_018_cam_00_frame.0000` 被显式排除。
- 所有 RGB 和 radial depth 路径及 SHA-256 均固化在 `config.json` 中。

## 验收标准

- K1 平均 Point Rel 相比 K0 至少改善 1%。
- 至少 60% 的样本取得 K1 Point Rel 改善。
- 任一场景的平均 K1 Point Rel 相比 K0 退化不超过 5%。
- 重复 K1 推理的最大绝对误差不超过 `1e-7`。
- K0 在每个样本的评估前后保持稳定，InfiniDepth、MoGe2 和 SSR 均无梯度且参数
  不发生更新。

## 当前状态

`planned`。尚未执行服务器评估，不声明任何泛化结论。

## 复现方法

在 `ZJU3DV-S115` 的服务器代码目录中运行：

```bash
bash experiment/exp2_infinidepth_ssr_heldout_generalization/run.sh
```

启动脚本会拒绝任何位于 `/mnt/data/home/zhuzichao/` 之外的输出路径。运行完成后，
使用以下命令严格验证服务器保留资产：

```bash
python experiment/validate_experiment.py --require-untracked-assets \
  experiment/exp2_infinidepth_ssr_heldout_generalization
```

## 资产保留规则

Git 只保留配置、来源记录、紧凑指标、一张逐样本 Point Rel 对比图和一张 K1 相对
改善图。原始运行日志仅保留在服务器；不复制 exp1 checkpoint，不导出点云，
不生成逐样本截图或重复指标格式。

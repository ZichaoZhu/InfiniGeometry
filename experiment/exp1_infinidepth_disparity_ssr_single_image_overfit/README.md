# Exp1：InfiniDepth 原生 Disparity Refiner 单图过拟合

## 研究问题

在固定 `384x512` 像素中心网格上，MoGe-3 风格的动态稀疏三维上下文能否为 InfiniDepth 原生相对 disparity 学到有效的加法残差？本实验只验证五张训练图上的实现正确性与拟合能力，不验证泛化。

## 架构边界

- K0 是 InfiniDepth 原生归一化 disparity；不使用 MoGe-2、外部 reference depth 或 GT 对齐作为模型输入。
- Refiner 输入只有 `(x_norm, y_norm, disparity)` 和 InfiniDepth DINOv3 特征。
- 第三轴按 `round(200 * disparity)` 体素化；每轮预测最大绝对值为 `0.1` 的 additive disparity residual，并重新体素化。
- GT radial depth 只转换为训练目标、mask、细结构指标和查看器反归一化统计。

## 五个独立运行

五张图均从同一官方 checkpoint 和零初始化 Refiner 开始，互不继承权重：

| 运行 | 样本 | 锁定细结构 |
| --- | --- | --- |
| 01 | `ai_019_004_cam_00_frame.0000` | 楼梯扶手与平行栏杆 |
| 02 | `ai_054_008_cam_00_frame.0000` | 悬空楼梯踏板与支架 |
| 03 | `ai_053_018_cam_00_frame.0000` | 密集竖直隔断杆 |
| 04 | `ai_002_003_cam_00_frame.0000` | 楼梯栏杆与斜向扶手 |
| 05 | `ai_019_004_cam_00_frame.0099` | 玻璃楼梯细拉杆 |

阶段一对 K1–K3 切断 Base 与视觉特征梯度，但 K0 loss 继续训练 BasicEncoder、ImplicitHead 和按日程开放的 DINO。阶段二取消 detach，四轮 loss 等权联合训练全部模块。

## 验收

每个运行按 `K3全图disparity MAE + K3锁定细结构disparity MAE` 选优。五个运行必须全部正常结束，且至少 3/5 的最佳 checkpoint 中 K3 综合分数比同 checkpoint K0 低至少 1%，才能登记百图 Exp2。若失败，只归类报告，不通过临时调参覆盖结果。

## 服务器执行

先激活个人环境，再从仓库根目录逐个运行：

```bash
bash experiment/exp1_infinidepth_disparity_ssr_single_image_overfit/run.sh 01
```

运行 ID 为 `01` 至 `05`。完成五个运行后执行汇总命令；训练、评测、导出和查看器验收均不得在本地执行。

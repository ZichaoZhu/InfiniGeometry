# Exp4: InfiniDepth LiDAR Refiner on Hypersim Full

## 实验简述

Exp4 使用 `InfiniDepth_DepthSensor` 作为冻结基座，将 Hypersim radial range 投影为 64 线虚拟 LiDAR prompt，只训练 disparity SSR。正式配置直接使用全部 59,542 张有效 train 图像，不包含单图过拟合或固定 Train100 阶段。

K0 是 LiDAR prompt conditioning 后的 DepthSensor 原始输出；K1、K3、K5 是同一 checkpoint 的 SSR 迭代输出。模型输入只包含 RGB、稀疏 LiDAR disparity 和 mask；稠密 GT、细节 mask 和评测对齐参数均不进入模型。

## 局部点云指标

Exp4 使用 MoGe-3 v2 Appendix B 的 Local Point Rel 和 Local Point \(\delta_{0.01}\)，不再使用 HEG-F1。固定 Val100 的细节区域先离线生成：以尺度 \(\{8,16,32\}\) 的 disparity residual 和尺寸 \(\{3,5,9,17\}\) 的 top-hat/black-hat 响应构造粗 mask，再保留满足密度、重叠和面积阈值的 SAM2 segment。

对每张图所有有效点先求一个由所有 segment 共享的全局尺度：

\[
s^*=\frac{\sum_i w_i\,\mathbf p_i^\mathsf T\mathbf g_i}
{\sum_i w_i\,\lVert\mathbf p_i\rVert_2^2},
\qquad w_i=\frac{1}{\lVert\mathbf g_i\rVert_2}.
\]

每个 segment \(S\) 只单独拟合三维平移：

\[
\mathbf t_S^*=\frac{\sum_{i\in S}w_i(\mathbf g_i-s^*\mathbf p_i)}
{\sum_{i\in S}w_i},
\qquad \hat{\mathbf p}_i=s^*\mathbf p_i+\mathbf t_S^*.
\]

segment 内指标为：

\[
\operatorname{Rel}_S=\frac{1}{|S|}\sum_{i\in S}
\frac{\lVert\hat{\mathbf p}_i-\mathbf g_i\rVert_2}{\lVert\mathbf g_i\rVert_2},
\]

\[
\delta_{0.01,S}=\frac{1}{|S|}\sum_{i\in S}
\mathbb 1\!\left[
\lVert\hat{\mathbf p}_i-\mathbf g_i\rVert_2
<0.01\min(\lVert\hat{\mathbf p}_i\rVert_2,\lVert\mathbf g_i\rVert_2)
\right].
\]

最终在所有有效 segment 间等权平均；小于 10 个有效像素的 segment 跳过，空 mask 不产生局部指标。同时报告 K0/K1/K3/K5 的 metric disparity MAE 和 radial AbsRel。checkpoint 只按 K3 metric disparity MAE 选择，局部指标仅诊断 SSR 的细节收益。

论文没有发布这部分预处理的完整实现，因此 OpenCV 插值、椭圆形 morphology kernel 和 SAM2 参数已固定在配置与 mask manifest 中；实现遵循论文公式，但不宣称与未发布代码逐位一致。

## 存储与恢复

- 主运行目录：`/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner`。
- NAS 备份目录：`/nas1/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner`。
- 正式训练前扫描全部 Train 59,542 和 Val100 本地 RGB/depth 缓存，任何缺失都会拒绝启动，训练途中不从 NAS 补数据。
- 每 500 steps 在主目录原子发布不可变 `checkpoints/step_XXXXXXXXX/`，并生成 `SHA256SUMS`。
- 训练期间 checkpoint 只写入 S115 主目录，不访问 NAS，因此 NAS 延迟或故障不会中断训练。
- 训练正常结束、Python 异常、`SIGINT` 或 `SIGTERM` 中断时，同步备份最新本地 checkpoint 和运行元数据；硬断电或 `SIGKILL` 无法执行退出备份，此时从 S115 最近的本地 checkpoint 恢复。
- NAS 先写临时目录，校验文件数、大小和 SHA-256 后原子改名，并写入 `BACKUP_COMPLETE.json`。失败按 5、15、45 秒重试，四次尝试均失败则保留本地 checkpoint 并记录 `backup_failed`。
- 默认从本地最新校验通过的 checkpoint 恢复；仅在显式指定时从含完成标记且校验通过的 NAS checkpoint 恢复。
- `latest.json`、`best.json`、日志、指标、配置、源码/data/mask provenance 随 checkpoint 和最终状态同步。

## 正式结果

训练正常完成 40,000 step，最终 checkpoint 为 step 40,000；checkpoint 选择仍严格使用固定 Val100 的 K3 metric disparity MAE，因此正式模型为 step 22,500，而不是末尾发生回退的 step 40,000。step 22,500 的 checkpoint SHA-256 为 `eec2b698753bc4ab2059d89c153a5caf818bc21fa8655d8d728d1c1f0ef5c54a`。

固定 Val100（100 张、1,117 个保留 segment）上的正式重测结果如下：

| 输出 | Metric disparity MAE (1/m) | Radial AbsRel | Local Point Rel | Local Point \(\delta_{0.01}\) |
| --- | ---: | ---: | ---: | ---: |
| K0 | 0.003687 | 0.014911 | 0.017438 | 0.726185 |
| K1 | 0.003541 | 0.014276 | 0.016569 | 0.747763 |
| K3 | 0.003430 | 0.013750 | 0.015767 | 0.758443 |
| K5 | 0.003532 | 0.014087 | 0.015838 | 0.756741 |

K3 相比同 checkpoint 的 K0：metric disparity MAE 改善 6.979%，Local Point Rel 改善 9.584%，Local Point \(\delta_{0.01}\) 增加 3.226 个百分点。K3 在 98/100 张图的 metric disparity MAE 更好，在 88/100 张含有效细节 segment 的图中 Local Point Rel 更好。K5 相比 K3 回退，因此默认展示和结论使用 K3。

## 可视化

查看器新增 Exp4，可切换 train、val、test；每个 split 固定 5 张。train 复用 Exp1/Exp2/Exp3 的细结构样本和裁剪，val/test 使用 Exp3 的 seed 173 随机样本。左侧默认显示冻结 LiDAR-conditioned DepthSensor K0，右侧默认显示 step 22,500 的 K3；所有 K 值均使用同一确定性 LiDAR prompt。

- [正式 Val100 评估](metrics/formal_eval_best_step_000022500.json)
- [Val100 完整评估曲线](artifacts/training_curve_full_eval.png)
- [15 张 disparity 对比图](artifacts/disparity_comparison_viewer.png)
- [查看器样本与 checkpoint 配置](viewer.json)
- [资产 manifest](artifacts/manifest.json)
- [在线三窗口查看器](https://infinidepth-disparity-refiner-viewe.vercel.app)

2026-09-01 已发布至生产查看器。线上清单、点云文件 SHA-256 和 train/test 的三块 WebGL canvas 验收均通过，见 [deployment_20260901_vercel.json](deployment_20260901_vercel.json)。此前的本地/S115 验收及待授权状态保留在 [deployment_20260831_eval1.json](deployment_20260831_eval1.json)。

查看器的 Exp4 页面现为四窗口：GT、LiDAR K0、LiDAR SSR 和版本对照。版本对照默认复用同图的 Exp3 RGB-only Stage1 K3，并可切换为 Exp4 LiDAR；每个版本仍可独立选择阶段和 K 值。两种版本的 Base、输入条件和指标口径不同，页面明确提示不能直接横比绝对指标。线上发布与验收见 [deployment_20260901_rgb_lidar_compare.json](deployment_20260901_rgb_lidar_compare.json)。

## 最终验收状态

- 训练：40,000/40,000 step，训练进程已退出；final report 为 `completed`，冻结的 DepthSensor base 参数未变化。
- 恢复与备份：step 40,000 本地 checkpoint 与 NAS 备份均有完成标记并通过校验；best step 22,500 checkpoint 已单独校验。
- 数据：Train 59,542 与 Val100 本地缓存预检通过；固定 Val100 mask 为 100/100、1,117 个 segment。
- 代码：S115 `20260831_eval1` 隔离部署中的 Exp4 LiDAR 与 checkpoint 回归测试通过。
- 查看器：`exp4_best22500` 含 15 个样本和 106 个静态文件；本地 Next 构建、类型检查、单元测试及 Exp3/Exp4/移动端 Playwright 验收通过，并已在生产网站复验。

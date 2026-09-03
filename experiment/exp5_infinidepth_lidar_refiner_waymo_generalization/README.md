# Exp5 Waymo: Exp4 LiDAR Refiner Zero-shot Generalization

## 实验简述

Exp5 固定 Exp4 step 22,500 checkpoint，不训练、不微调、不在 Waymo 上重新选择 checkpoint。评测使用 Waymo Open Dataset v1.4.3 validation 的 202 个 TFRecord，每个序列取第 20 帧 FRONT 图像和 TOP LiDAR 第一回波。

本子实验的正式标识为 `exp5_waymo`。在线查看器将固定随机抽取的 5 张 FRONT 与用户手选的 5 张 SIDE 合并为同一十样例入口；旧的两个 manifest 仅保留为不可变资产来源，不再单独显示。

TOP LiDAR range-image 水平列满足 `column % 4 == 0` 的点作为 DepthSensor prompt，其余点只用于评测；缩放到 384x512 后与 prompt 落在同一像素的评测点被排除。LiDAR 点经过每点位姿补偿并转到相机时刻，prompt disparity 和指标都使用相机光心到点的欧氏距离倒数。

## 评测口径

- 输出：同一 checkpoint 的 K0、K1、K3、K5。
- 主指标：held-out sparse metric disparity MAE 和 radial AbsRel。
- 辅助指标：radial RMSE 和 held-out point \(\delta_{0.01}\) 。
- 聚合：每个 Waymo 序列一张图，按图等权平均；K3-K0 同时报告配对 bootstrap 95% 置信区间和改善图像数。
- 边界：Waymo 只提供稀疏 LiDAR 支持，本实验不计算 Hypersim 稠密 GT 口径的 Boundary F1 和 Local Point 指标，也不与 Hypersim Val100 绝对值横向比较。

## 数据与输出

- 只读输入：`/nas1/datasets/Waymo/waymo_open_dataset_v_1_4_3/individual_files/validation`
- 服务器输出：`/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp5_waymo_generalization`
- 轻量 reader：`gdlg/simple-waymo-open-dataset-reader` commit `d488196b3ded6574c32fad391467863b948dfd8e`

## 当前状态

Waymo 适配器、评测入口和服务器部署已完成。服务器 CPU 单元测试通过，真实
TFRecord 解析确认 validation 共 202 个文件；首个样本得到 3,404 个 prompt 点和
10,168 个 held-out 评测点。

2026-09-02 在 GPU2 完成 Val5 smoke；2026-09-03 改用空闲 GPU1 完成 Val202
正式评测。评测使用 batch=1，并设置仅终止本实验进程的显存保护；保护未触发，评测无 OOM 或异常。

K3 相比 K0：metric disparity MAE 下降 6.729%，201/202 张改善；
point \(\delta_{0.01}\) 增加 5.665 个百分点，202/202 张改善；radial
AbsRel 均值下降 5.976%，176/202 张改善，但配对 bootstrap 95% 置信区间跨过 0。
radial RMSE 从 18.515 m 增加到 26.075 m，需要继续检查大误差样本。

2026-09-03 完成 Waymo 定性可视化。从 Val202 已完成样本按 seed 173
固定随机抽取 5 张，导出 held-out TOP LiDAR、K0、K1、K3 和 K5 点云；
预测和 LiDAR 均使用 Waymo 相机内参反投影，可视化范围固定为
0–100 m。本批资产不新增计算 Local Point 等指标，页面明确显示
“评测指标稍后补充”。

同日另外导出 Val202 的全部 202 张 FRONT 图像预览，为实际送入模型的
384×512 版本。图库可按地点、时段和天气筛选，点选后可复制 TFRecord
文件名，用于后续替换点云可视化样本。

2026-09-03 根据手选清单另外导出 5 张 SIDE 点云：3 张 `SIDE_RIGHT`、2 张
`SIDE_LEFT`。每张均使用对应 SIDE 相机的 RGB、内参和 TOP LiDAR 到该相机的
投影，生成 K0、K1、K3、K5 与 held-out TOP LiDAR。该批仅用于定性观察细杆、
护栏、路牌等侧视细节；不重跑 Val202，不新增指标，也不用于选择 checkpoint。

同日将上述 FRONT 与 SIDE 资产合并为 `exp5_waymo`。合并只新增索引 manifest 与
provenance，不重新推理、不复制 PLY，且逐个核验既有 PLY 的 SHA-256。页面现在统一
显示 10 个样例，保留 K0/K1/K3/K5、初始/zero-shot 阶段和 held-out TOP LiDAR 标注。

## 实验结果

- [Val202 汇总指标](./metrics/waymo_val202_step22500_summary.json)
- [可视化样本与导出配置](./viewer.json)
- [生产发布与验收记录](./deployment_20260903_viewer.json)
- [手选 SIDE 可视化配置](./viewer_side.json)
- [手选 SIDE 发布与验收记录](./deployment_20260903_side_viewer.json)
- [合并十样例的选择来源](../viewer/public/data/exp5_waymo/selection.json)
- [合并十样例发布与验收记录](./deployment_20260903_waymo_combined_viewer.json)
- [在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)
- [Waymo Val202 FRONT 选图页](https://infinidepth-disparity-refiner-viewe.vercel.app/data/waymo_gallery_exp5_val202_front_20260903_r2/index.html)
- [选图页发布与验收记录](./deployment_20260903_gallery.json)
- [Waymo Val202 SIDE_LEFT/SIDE_RIGHT 选图页](https://infinidepth-disparity-refiner-viewe.vercel.app/data/waymo_gallery_exp5_val202_side_20260903_r2/index.html)
- [侧视选图页发布与验收记录](./deployment_20260903_side_gallery.json)
- 服务器完整报告：`/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp5_waymo_generalization/runs/waymo_val202_front_step22500/report.json`

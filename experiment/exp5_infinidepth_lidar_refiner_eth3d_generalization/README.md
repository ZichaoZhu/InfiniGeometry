# Exp5 ETH3D: Exp4 LiDAR Refiner Zero-shot Generalization

## 实验简述

固定 Exp4 step 22,500 checkpoint，在 ETH3D high-res multi-view training DSLR 的 13 个场景、454 张图像上进行零样本评测。不训练、不微调、不使用 ETH3D 选择 checkpoint；由于 ETH3D test 标签不公开，本实验不是官方 ETH3D benchmark leaderboard 提交。

每张图先以原始相机几何为准中心裁为 4:3，并缩放到 384×512。ETH3D 的 `ground_truth_depth/dslr_images/*.JPG` 文件虽然使用 JPG 文件名，内容实际为 raw `float32` 相机 z-depth；利用 COLMAP `THIN_PRISM_FISHEYE` 标定反投影为单位射线后，按

\[
r = \frac{z}{\hat{v}_z}
\]

转换为从相机光心出发的 radial range。虚拟 LiDAR prompt、预测尺度恢复和误差计算均使用该 radial range。

## 评测口径

- 输出：同一 checkpoint 的 K0、K1、K3、K5。
- 主结果：排除虚拟 LiDAR prompt 像素后的有效 depth 像素，报告 metric disparity MAE、radial AbsRel、radial RMSE 与 point \(\delta_{0.01}\)。
- 兼容性诊断：同时保留所有有效 depth 像素的 `all_valid` 结果；不作为主结论。
- 局部指标：MoGe-3 Local Point Rel、Local Point \(\delta_{0.01}\)，全局共享尺度、每个细节 segment 仅拟合三维平移、按保留 segment 等权。
- 细节区域：GT disparity 的 coarse detector 加 SAM2 候选。ETH3D 上原 Exp4 的 `min_density=0.30` 在 smoke 中保留 0 个 segment；因此本子实验固定为 `0.10`，并使用独立版本目录 `moge3_v2_sam2_1_small_eth3d_density010_v1`。该变化只影响评测区域，不进入模型输入或训练。
- 聚合：图像宏平均，另外报告场景宏平均；K3-K0 的置信区间按 13 个场景 block bootstrap，而不是把相关视角当独立样本。

## 数据、运行与恢复

- 归档：`/mnt/data/home/zhuzichao/datasets/ETH3D/high_res_training_exp5/downloads`
- 解压数据：`/mnt/data/home/zhuzichao/datasets/ETH3D/high_res_training_exp5/extracted_20260903`
- 实验输出：`/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp5_ETH3D`
- `extract.sh` 先校验 14 个 archive 的 SHA-256；解压到临时目录，确认 13 场景、454 RGB、454 depth 和 13 份相机标定后才原子发布。
- `prepare_eth3d_inputs.py` 固定输入 manifest；`prepare_eth3d_masks.py` 每图原子写 segment 与 preview，任务中断后跳过已校验条目；`evaluate_eth3d.py` 逐图原子写 `progress.json`。
- `run.sh` 的 smoke、masks 与 formal 在启动前要求 GPU1 至少 8/12/8 GiB 空余；生产启动器额外设置只终止本实验进程的显存 watchdog。

## 当前状态

输入结构、COLMAP fisheye 反投影、虚拟 LiDAR prompt、K0/K1/K3/K5 推理和主指标均已在 5 张 smoke 验证。完整评测已在 GPU1 完成：13 个场景、454 张图像、5,447 个保留细节 segment，运行 1,103 秒，无 OOM、异常或显存保护触发。

主评测范围内，K3 相比同 checkpoint 的 K0：metric disparity MAE 从 0.0009067 降至 0.0006904（-23.86%，446/454 图改善）；radial AbsRel 从 0.4931 降至 0.1517（436/454 图改善）；point \(\delta_{0.01}\) 从 0.9349 升至 0.9592（445/454 图改善）。细节区域中，Local Point Rel 从 0.01143 降至 0.01114（-2.61%，351/444 图改善），Local Point \(\delta_{0.01}\) 从 0.8001 升至 0.8081（302/444 图改善）。相应的 13 场景 block bootstrap 95% CI 均不跨 0。

K5 的 metric disparity MAE、point \(\delta_{0.01}\) 和局部指标都略逊于 K3，因此该 zero-shot 场景的有效迭代次数仍为 K3；`all_valid` 下 K5 的 AbsRel 更低，不改变以 held-out K3 metric disparity MAE 为主的选择口径。

## 实验结果

- [配置](./config.json)
- [下载脚本](./download.sh)
- [解压与一致性检查脚本](./extract.sh)
- [评测运行脚本](./run.sh)
- [454 图汇总指标](./metrics/eth3d_highres_train_step22500_summary.json)
- [正式运行记录](./runtime_20260903_gpu1_formal.json)
- [部署与数据 provenance](./deployment_20260903_impl2.json)
- [ETH3D 十张选图页发布记录](./deployment_20260903_gallery.json)
- [在线 ETH3D 选图页](https://infinidepth-disparity-refiner-viewe.vercel.app/data/eth3d_gallery_exp5_highres_train_20260903_r1/index.html)
- 服务器完整报告：`/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp5_ETH3D/runs/eth3d_highres_train_step22500/report.json`

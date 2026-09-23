# Exp6-4 稀疏数值非确定性的限时诊断

2026-09-23，按用户“一小时以内就做”的授权完成。计算与汇总约 20 分钟，不重训正式实验，不改默认模型、依赖或历史产物。此前[严格验收失败记录](../verification_20260923/README.md)保留。

## 结论

默认设置的差异已复现，并定位到本例第一次稀疏池化的坐标/邻居枚举及求和顺序。不是只有不同代码版本或不同进程才出现：同一进程连续推理也有差异。

临时固定池化输出坐标后，本例推理完全一致；再组合 PyTorch 确定性设置，跨进程的预测、loss、86 个梯度张量和一次 AdamW 更新均完全一致。这给出可行的诊断性处理方向，但没有将该处理应用到正式代码，也没有证明所有图片、长时间训练或其他硬件均确定。

## 范围与方法

- S115 GPU1，RTX 4090；启动时空闲 29,618 MiB，允许共享，不读取或控制其他用户任务。探针进程设置 10 GiB 显存上限，实际 PyTorch allocated 峰值约 3.59 GiB。
- 复用此前验收源码 `30a2f89`、既有 venv 和固定缓存图 `ai_003_010_cam_00_frame.0000`，分辨率 384×512。仅 1 张 Val100 图，不使用 test，不做全量评估。
- 同一已训练 `official_flex/stage1_best.pt`，SHA-256 为 `fb94443a4b77923283f59008e31e65a0dc9c9557007e400711eab3c52f4aac3c`；每个进程重新严格加载，Base 始终冻结。
- 每种设置两个独立进程，每个进程先连续推理两次，再做一次训练前向、反向和临时 optimizer step。复用[原验收工具](../verify_handoff_cuda.py)，额外记录第一轮 SSR 各层特征及后续轮次输入。
- 优化器是探针新建的 AdamW，不是恢复历史 optimizer；诊断 `probe.pt` 不是正式训练 checkpoint。
- 使用原选核缓存的只读副本，关闭自动保存，保留 `AUTOTUNE_MODE=always`。实际核验 8 个进程的完整选核清单都与输入缓存相同，因此本次对照不含选核变化。运行时 `USE_CUDA_EXTENSION=False`，实际走 Triton 路径。

四组设置：

| 设置 | 仅本次诊断中的变化 |
| --- | --- |
| baseline | 原始模型/稀疏算法；增加观测钩子和独立输出 |
| deterministic | 开启 `torch.use_deterministic_algorithms(True)`、cuDNN deterministic，关闭 cuDNN benchmark；进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| canonical | 通过临时 forward pre-hook，为非重叠 2×2×2 稀疏池化提供去重、排序后的输出坐标；仍调用原池化和邻域缓存 API |
| canonical_deterministic | 组合前两种诊断干预 |

canonical 不在池化完成后只重排特征，否则可能破坏邻域缓存。它使用 `floor(coords/2)` 的唯一坐标作为原 API 的 `output_coords`，前置检查 kernel=stride=2、padding=0；独立重放确认体素集合和每个体素的邻居集合未变。

## 跨进程结果

阈值沿用 `atol=rtol=1e-6`，没有放宽。下表是归一化 disparity 的最大绝对差，不是模型精度 MAE；K0 在所有对照中均逐元素完全一致。

| 设置 | K1 最大差 | K3 最大差 | K5 最大差 | 推理严格检查 |
| --- | ---: | ---: | ---: | --- |
| baseline | 4.0531e-6 | 0.00154608 | 0.00759773 | 未通过 |
| deterministic | 3.4571e-6 | 0.00109178 | 0.00348759 | 未通过 |
| canonical | 0 | 0 | 0 | 本例精确一致 |
| canonical_deterministic | 0 | 0 | 0 | 本例精确一致 |

四组的同进程重复推理也呈现同样的通过/失败模式。不能只开启 PyTorch 确定性开关，就假定第三方稀疏操作也会被覆盖。

| 设置 | 梯度最大差 | 86 个梯度张量中精确一致 / 阈值通过 | 更新后参数最大差 | 86 个更新张量中精确一致 / 阈值通过 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.2817e-4 | 0 / 36 | 1.9695e-5 | 14 / 54 |
| deterministic | 3.7171e-5 | 0 / 77 | 1.8754e-5 | 16 / 58 |
| canonical | 2.3842e-7 | 1 / 86 | 1.4831e-6 | 36 / 84 |
| canonical_deterministic | 0 | 86 / 86 | 0 | 86 / 86 |

canonical 单独使用时 loss 完全一致，但仍有反向/更新差异；不能将“梯度误差很小”写成参数更新严格通过。组合设置在本例通过，反向差异具体来自哪个低层算子未进一步逐项隔离。

## 差异链条的证据

1. 同进程默认对照的 K0、DINO 视觉条件、第一层残差块输出均精确相同。第一次差异在 `downsample_blocks.0.pool`；按坐标排序后比较仍有最大 `2.3842e-7` 的特征差异，不是把不同点错位比较。
2. 用相同的第一池化输入单独重放三次：体素集合和邻居集合相同，但两次重复分别有 18,317 / 17,982 个输出体素的邻居顺序变化。原 Triton reduction 按该顺序累加，浮点加法不满足严格结合律，因此产生不同舍入结果。
3. 独立池化使用 canonical 坐标后，邻居顺序变化数为 0，重复输出精确相同；与默认路径相比，邻居集合仍完全相同。
4. 默认整网同进程对照中，第 2～5 轮 SSR 输入分别有 11、43、257、528 个像素的 `round(200*d)` 分箱不同。小浮点差异改变后续稀疏邻域，伴随 K3/K5 的差异放大；并非训练数据或权重被换掉。
5. 所有进程的选核清单相同。至少在本次对照中，不能把误差简单归因于 autotune 选择了不同卷积内核。

这是对所测样本前向首个差异源的受控定位，不是对 FlexGEMM 全部内核的确定性证明，也不说明原始实验精度提升或下降。

## 证据、复用与剩余工作

- [汇总与选核一致性](./evidence/main/diagnosis_summary.json)、[原始跨进程统计](./evidence/main/runs/comparisons.json)、[组合设置统计](./evidence/followup/runs/comparisons.json)
- [固定池化输入的邻居/特征对照](./evidence/main/pool_isolation.json)、[三项 CPU 自测](./evidence/main/cpu_tests.json)、[同步证据哈希清单](./evidence/main/evidence_manifest.json)
- [诊断驱动与钩子](./scripts/diagnose_sparse.py)、[池化重放 v2](./scripts/isolate_pool_v2.py)、[统计自测](./scripts/test_diagnose_sparse.py)、[只读汇总脚本](./scripts/collect_summary.py)

精确命令、PID、输出目录和运行时间保存在 evidence 下各 `*.launch.json`、`*.exit.json` 与 `run_summary.json`。服务器诊断根目录为：

```text
/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/ssr_determinism_20260923_2052
/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/ssr_determinism_20260923_2052_followup
```

大 `trace.pt`、`probe.pt` 和编译缓存只保留在上述目录。复测必须使用新的输出目录；脚本不会覆盖已有结果。池化重放 v1 因遗漏 `output_shape` 参数失败，保留其脚本；v2 补齐与原调用相同的参数后通过。这是诊断脚本错误，不是正式模型故障。

`scripts/` 按原样归档服务器探针，不是可直接在任意机器运行的训练入口。原运行将脚本放在诊断根目录，`collect_summary.py` 依赖该布局及同级 `_followup` 目录；重新执行前需在新目录准备对应布局、环境、权重、缓存与大张量，不能直接对本仓库的 `evidence/` 回写。

8 个模型探针正常退出，3 项 CPU 统计测试通过；结束时 GPU1 恢复至验收前的 18,891 MiB 占用。没有停止其他任务、改动安装包、正式 checkpoint、实验配置或网站。

默认实现的严格检查仍应标记未通过，但可将“来源尚不明确”更新为“已定位本例前向稀疏池化顺序问题，组合控制在单图/单步上通过”。如果决定落地修复，应增加显式可选确定性模式，再验证多图、训练恢复、吞吐及历史 checkpoint 行为；本次不自动开启，不静默重导出或改写历史结果。

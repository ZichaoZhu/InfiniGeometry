# Exp6-4：冻结 Base 的官方 SSR 网络主体 disparity 适配对照

实现状态：网络适配、refiner-only 训练、恢复、A/B 正式训练和配对评估均已完成。详细实验协议见 [PLAN.md](./PLAN.md)，最终报告见 [results_20260914](./results_20260914/)。

## 方法口径与代码交接（2026-09-21）

本实验准确名称为“InfiniDepth＋官方 SSR 网络主体的 disparity 适配版”。复用官方 Sparse3DUNet 与 FlexGEMM，不加载官方 MoGe3 SSR 权重；保留本项目的 normalized disparity 表示、有界加法更新、disparity MAE＋梯度损失及 SSR 迭代间梯度连接。Base 全程冻结，没有 DINO 解冻或 Joint。

限幅 \(\Delta d=0.1\tanh(r/0.1)\) 是历史额外约束，不是官方要求，其必要性未单独验证。Exp6-5 R1 修正的是另一条几何损失路径，没有同步改动 Exp6-4；其无界 logZ 分支训练完成，不等于 Exp6-4 去限幅已经验证。

目前没有完成“以 InfiniDepth 为基座、仅作必要接口适配，并尽量保留官方几何表示、更新、逐轮梯度处理及几何监督”的实验。已有结果只支持本 disparity 体系内的网络主体对照，不能据此宣称完整官方机制已复现。

源码入口、官方/适配边界、配置、checkpoint 与运行限制见[代码交接文档](../../docs/exp6_4_disparity_adapter_handoff.md)。本次提交归档既有实现与结果，不改变训练算法、配置、权重或网站。Git 仅保留紧凑产物，图册的外部图片、数组和缓存见 [ARCHIVE.md](./ARCHIVE.md)；本次验证范围见 [COMMIT_CHECKS.md](./COMMIT_CHECKS.md)。

## 历史部署记录

2026-09-11 最终部署记录：[source_v4 与后台 PID 验收](./deployment_20260911_v4.json)。80 项 CPU 测试通过，10 项 CUDA 测试待空闲 GPU；此时无新增训练结果。

2026-09-12 首次 CUDA 门禁为 9 passed、1 failed：FlexGEMM 的 adaptive cold-start 直接使用面向 A100 的首个 backward tile，所需 shared memory 超过 RTX 4090 上限。实验进程改用 FlexGEMM 自带的 `always` autotune，让其过滤不可启动的 tile 并选择可用配置；缓存独立放在本实验目录，不修改官方网络或共享环境。

## 实验结果

两组均从同一官方 InfiniDepth Base 和零初始化 SSR 开始，Base 全程冻结，各训练 20,000 steps。两组的 Base 哈希、K0 输出和全局采样序列一致。以下结果来自固定 Hypersim validation 100；误差类指标越低越好，\(\delta\) 和 F1 越高越好。

相对 K0 的误差改善按下式计算：

\[
r = \frac{M_{K=0}-M_{K=3}}{M_{K=0}}.
\]

### 现有 spconv SSR

| K | 原生 disparity MAE ↓ | Global depth rel ↓ | Global point rel ↓ | Local depth rel ↓ | Local point rel ↓ | Local point \(\delta_{0.01}\) ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.063314 | 0.080735 | 0.086009 | 0.034293 | 0.029741 | 0.449072 |
| 1 | 0.060598 | 0.075701 | 0.081211 | 0.032631 | 0.027869 | 0.498147 |
| 3 | 0.058131 | 0.071136 | 0.077640 | 0.031799 | 0.026788 | 0.538317 |
| 5 | 0.059564 | 0.075685 | 0.082703 | 0.033816 | 0.028262 | 0.504235 |

K3 相对 K0：disparity MAE 改善 8.19%，Global depth rel 改善 11.89%，Global point rel 改善 9.73%，Local depth rel 改善 7.27%，Local point rel 改善 9.93%。按各指标逐图统计，K3 的改善数量依次为 80、82、80、85、87 张。

### official_flex SSR

| K | 原生 disparity MAE ↓ | Global depth rel ↓ | Global point rel ↓ | Local depth rel ↓ | Local point rel ↓ | Local point \(\delta_{0.01}\) ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.063314 | 0.080735 | 0.086009 | 0.034293 | 0.029741 | 0.449072 |
| 1 | 0.055457 | 0.069251 | 0.075121 | 0.031577 | 0.026859 | 0.529352 |
| 3 | 0.054461 | 0.068439 | 0.074237 | 0.031333 | 0.026040 | 0.552246 |
| 5 | 0.055964 | 0.073061 | 0.078707 | 0.032621 | 0.027442 | 0.514226 |

K3 相对 K0：disparity MAE 改善 13.98%，Global depth rel 改善 15.23%，Global point rel 改善 13.69%，Local depth rel 改善 8.63%，Local point rel 改善 12.44%。按各指标逐图统计，K3 的改善数量依次为 71、86、83、81、87 张。

### K3 配对比较

下表为 official_flex 减去 spconv 的逐图均值差。误差指标为负、准确率指标为正表示 official_flex 更好。

| 指标 | 均值差 | scene bootstrap 95% CI | 判断 |
|---|---:|---:|---|
| 原生 disparity MAE ↓ | -0.003670 | [-0.007789, -0.000118] | 支持 official_flex |
| Global depth rel ↓ | -0.002698 | [-0.004508, -0.001117] | 支持 official_flex |
| Global point rel ↓ | -0.003403 | [-0.005538, -0.001517] | 支持 official_flex |
| Local depth rel ↓ | -0.000466 | [-0.001431, 0.000628] | 区间跨 0，暂不确定 |
| Local point rel ↓ | -0.000748 | [-0.001236, -0.000364] | 支持 official_flex |
| Local point \(\delta_{0.01}\) ↑ | +0.013929 | [0.003888, 0.023966] | 支持 official_flex |

official_flex 的 K3 disparity MAE 比 spconv 再降低 6.31%，Global depth rel 降低 3.79%，Global point rel 降低 4.38%，Local point rel 降低 2.79%。配对报告给出 `candidate_improvement = true`，但当前只有单随机种子结果，不自动替换默认实现。

其他 K3 指标：spconv/official_flex 的 boundary F1 radius 1 分别为 0.366849/0.367941，\(\delta_{1.01}\) 分别为 0.213077/0.244732，Local depth \(\delta_{0.01}\) 分别为 0.496036/0.508989；两组无效预测率均为 0。

详细文件：

- [最终配对分析](./results_20260914/paired_report.json)
- [spconv 训练报告](./results_20260914/spconv/stage1_report.json)
- [spconv 几何指标](./results_20260914/spconv/geometry_summary.json)
- [official_flex 训练报告](./results_20260914/official_flex/stage1_report.json)
- [official_flex 几何指标](./results_20260914/official_flex/geometry_summary.json)
- [同步文件清单与哈希](./results_20260914/manifest.json)

## 结果诊断

2026-09-14 已完成 Val100 逐图配对分析、两组终点 checkpoint 的像素级补充推理和 12 张代表图。详见[诊断报告](./analysis_20260914_r3/REPORT.md)与[本地图册](./analysis_20260914_r3/gallery.html)。

结论：official_flex 平均误差更低，但存在逐图失败和后续迭代退化；目前优先检查更新方向与迭代监督。没有足够证据将过量跨越作为主机制，因此按计划跳过残差减半干预。本次未新增训练、未更换默认 SSR，原正式结果和网站资产不变。

## 点云可视化

结果已发布到[在线点云查看器](https://infinidepth-disparity-refiner-viewe.vercel.app/)。页面固定显示 GT、现有 spconv SSR 和 official_flex SSR 三个窗口，两种 SSR 默认均为 K=3，并可分别切换 K=0/1/3/5。

可视化固定复用 Exp3 的 train/val/test 选图：seed 173，每个集合 5 张，共 15 张；其中 val/test 的 10 张与 Exp6-3 完全一致。test 只用于可视化，不参与训练、checkpoint 选择或定量结论。详细发布信息见[部署与验收记录](../viewer/deployment_20260914_exp6_4_official_ssr.json)。

## 运行

S115 专用部署根目录为 `/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/exp6_4_20260911/`。旧服务器仓库、既有训练环境和历史产物不覆盖；新增 FlexGEMM 安装在专用 venv，PyTorch/spconv 等底层包只读复用原环境。

以下是历史完整 A/B 启动入口，不是 official-only 命令。模板 `config.json` 默认使用 spconv，自动流程生成 `arms/official_flex/config.json` 后才运行官方适配组；不要直接使用模板启动并将结果称为官方组。在独立、已授权的部署源码目录及专用 venv 中，原流程为：

```bash
bash experiment/exp6_4_official_ssr_disparity_adapter/run.sh
```

后台流程只依赖标准库进行编排。2026-09-12 根据实验室共享规则改为固定物理 GPU 3，允许同卡存在其他用户的 compute PID；启动时要求剩余显存至少 22 GiB、利用率不超过 10%，并在启动前再核对一次。

执行顺序：

1. 锁定源码哈希、包版本及两组配置，运行 CPU 回归测试。
2. 等待空闲单卡，执行 CUDA 回归测试；缺失权重、跳过测试或失败均不能通过门禁。
3. A/B 分别运行 56-step smoke，第 5 step 协作暂停并恢复，检查采样连续性；剔除预热后测量 50 个 optimizer steps。另各做一张图的统一指标 smoke。
4. 两组全部通过后，A/B 顺序各训练 20,000 steps；两组都重新从官方 Base 和零初始化 SSR 开始。
5. 各组自动评估固定终点和 best checkpoint 的 K0/K1/K3/K5，调用固定哈希的 Exp6-3 评测器和 Local masks，最后生成配对分析。

只训练 SSR，Base 参数和 buffers 不更新；每次保存检查 Base 哈希。固定样本 K0 在评估中复核。loss 保留原来的四项平均，L0 不反传。每 500 steps 保存恢复点，full evaluation 每 2,500 steps。

## 产物

- `automation/status.json`：等待、验证、训练、评估、失败或完成状态。
- `automation/alert.json`、`incident.jsonl`：故障/完成和启动、资源暂停、恢复记录。
- `automation/cpu_tests.xml`、`cuda_tests.xml`：实际测试结果。
- `automation/source_hashes.json`、`environment.json`：源码及环境记录。
- `automation/smoke_summary.json`：两组吞吐、显存、Base 哈希。
- `arms/{spconv,official_flex}/config.json`：锁定的正式配置。
- `smoke/{spconv,official_flex}/runs/main/`：独立 smoke，不纳入正式结论。
- `runs/{spconv,official_flex}/`：各组训练、checkpoint、轻量 monitor 和逐图评估。
- `metrics/paired_report.json`：固定 step 20,000 的 A/B 配对结果及场景 bootstrap 区间。

共享模式不会因 GPU 3 上已有或新增外部 compute PID 而暂停，更不会向其发送信号。数值、显存或代码错误只保留并处理自己的任务，写入告警后停止盲目重启；后台脚本本身不调用模型，也不消耗模型 token。

如果恰好在最后一个 optimizer step 暂停，恢复时补做完整终点评估并更新报告，不能用较早的 full evaluation 代替终点。

本轮不自动替换默认 SSR、不进行第二 seed 复核，也不提交实验中间状态。网站发布为训练完成后的定性对照，不改变实验结论。

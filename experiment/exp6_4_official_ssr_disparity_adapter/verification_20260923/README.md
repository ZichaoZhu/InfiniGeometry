# Exp6-4 CUDA 与恢复交接验收

2026-09-23，按用户授权在 S115 GPU2 空余资源上完成。结论：训练、冻结与 checkpoint 恢复功能通过；历史已训练权重跨进程的严格数值一致性检查未通过，不能标为全部通过。

## 环境与边界

整理前为 `ef378cb`，整理后为 `30a2f89`。两份已提交源码部署到本人新目录 `handoff_verify_20260923_1348/{before,current}`；模型及数据/损失等 223 个核心文件逐字节相同。使用现有 Exp6-4 venv：PyTorch 2.8.0+cu128、FlexGEMM 1.0.0、spconv-cu126 2.3.8，不安装依赖、不修改原环境。

保留原分辨率 384×512、microbatch 1、global batch 8、损失及学习率。先通过新增单组入口生成 56 步 smoke 配置，再派生明确标记的 6 步验收配置，仅调整总步数及评估/保存频率；8 张 train、1 张 val 全部确认已缓存，不使用 test。不是正式实验、全量评估、新机器从零安装或多卡验收。

报告及精确命令见 [summary.json](./summary.json)、[原始证据](./evidence/)与[验收工具](../verify_handoff_cuda.py)。配置、进程启动记录、XML、provenance 和轻量训练记录已同步；大 checkpoint 和张量探针仍保存在服务器本次独立目录，不重复下载。

## 已通过

- 51 项相关 CPU 测试、9 项子测试、10 项真实 CUDA 测试通过，无跳过。覆盖路径、采样/RNG/best 状态、冻结、spconv/official_flex 反传、旧加载与导出相关检查；不是整个项目全部测试。[CPU XML](./evidence/cpu.xml)、[CUDA XML](./evidence/cuda.xml)
- 同一历史 `official_flex/stage1_best.pt` 可被两版严格加载，Base 保持冻结，86 个 SSR 梯度张量有限，残差在 0.1 范围内。历史权重 SHA 为 `fb94443a4b77923283f59008e31e65a0dc9c9557007e400711eab3c52f4aac3c`。
- 第 3 步协作暂停成功保存 last.pt。独立加载后的模型、optimizer、Python/NumPy/CPU/CUDA RNG、sampler 状态精确相同，下一批索引一致。[恢复 round-trip](./evidence/roundtrip/result.json)
- 使用同一配置恢复到第 6 步，与连续 6 步相比，模型和 optimizer 最大绝对差均为 \(7.45\times10^{-9}\)，通过 \(10^{-6}\) 阈值。RNG、采样序列、最终 best step/score/evaluation 完全一致。[连续/恢复对照](./evidence/compare_resumes/result.json)
- 两条短测均完成；DINO 和 Base Head 均无梯度，没有 OOM、NaN/Inf 或 residual 越界。[连续报告](./evidence/continuous/metrics/report.json)、[恢复报告](./evidence/resumed/metrics/report.json)

最终交付工具改为从 sampler 状态读取样本数量后，已再次执行[恢复对照](./evidence/compare_resumes_final_tool.json)和[加载 round-trip](./evidence/roundtrip_final_tool.json)，结果相同；没有新增训练步骤。

## 未通过：历史权重的严格数值对照

使用同一张真实 val 图 `ai_003_010_cam_00_frame.0000`、同一历史权重、同一环境及 seed，按 \(\mathrm{atol}=\mathrm{rtol}=10^{-6}\) 比较。K0 完全一致，SSR 输出未通过。随后对同一整理版另开进程复测，不更换代码，也复现差异：

| K | 整理前/后最大绝对差 | 同一整理版复测最大绝对差 |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 0.00000411 | 0.00000417 |
| 3 | 0.000952 | 0.001546 |
| 5 | 0.007591 | 0.002200 |

上述为归一化 disparity 差，不是 MAE 性能指标。K5 的平均绝对差分别约为 \(2.50\times10^{-5}\)、\(3.00\times10^{-5}\)；损失差均为 \(2.24\times10^{-7}\)。不能只看均值而忽略局部最大差。

同版本复测说明严格差异不是只有改代码后才出现；模型、loss、data 等核心文件也未改变。当前 FlexGEMM 源码明确包含非确定性 hashmap 坐标顺序与 atomic_add 并行累加，且本次保留 `AUTOTUNE_MODE=always`。这些是可能来源，但本轮没有逐项隔离，不能断言差异只由自动选核造成。多轮稀疏处理可能放大微小差异，仍需单独确认。

保留[原始失败日志](./evidence/compare_probes.log)和全部探针统计。不放宽阈值，不替换内核、关掉限幅或修改正式模型来取得通过。若交接要求严格跨进程复现，需另做固定算子/坐标顺序的受控验证；当前只能宣称功能及短步恢复通过。

## 验收过程与安全记录

临时 smoke 驱动最初假设暂停的第 3 步会产生 best.pt，但原训练器在暂停步主动跳过评估，所以此时只有有效 last.pt。驱动因此停止，未破坏训练 checkpoint；随后直接使用原 CLI、同一配置恢复完成。初版脚本、暂停报告和错误原因保留，不将驱动错误当成训练故障。

真实任务均已退出，GPU2 显存恢复到验收前约 8,602 MiB 的占用。未关闭其他任务、修改他人文件、覆盖历史 checkpoint/结果，未启动长期训练、部署、提交或推送。

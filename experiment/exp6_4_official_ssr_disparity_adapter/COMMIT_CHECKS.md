# Exp6-4 本地提交验证

日期：2026-09-21。基线为 `e9afd2e809edc8af2ddfae1f4043c6550db6ff92`，代码提交为 `dd252db9230bb902e5b464546b79914b536d0af3`。本文件随紧随其后的结果提交保存。

## 隔离提交内容

工作区同时存在 Exp6-1、Exp6-5 及查看器其他修改。本次从 Git 基线建立临时快照，仅加入 Exp6-4 所需内容后验证；共享查看器文件与实验日志按内容拆分，没有将整个工作区一并提交。训练、适配、监控和测试源码与工作区已有实现逐文件核对，不在归档过程中改变算法。

提交作者与提交者均为用户已有身份 `ZichaoZhu <1606783699@qq.com>`，不附加助手署名。只创建新的本地提交，不重写已有提交、不推送。

## 本次实际执行

| 检查 | 结果 |
| --- | --- |
| 暂存代码与隔离快照逐文件比对 | 29 个文件一致 |
| Python AST、暂存 JSON 解析 | 通过 |
| `bash -n experiment/exp6_4_official_ssr_disparity_adapter/run.sh` | 通过 |
| `python3.12 experiment/exp6_4_official_ssr_disparity_adapter/analyze_results.py self-test` | 标准库自测通过 |
| `pytest -q tests/disparity_refiner/test_exp3_scheduler.py -p no:cacheprovider` | 7 passed；使用离线临时 pytest 环境 |
| 查看器 `npm run typecheck` | 通过 |
| 查看器 `npm test -- --reporter=dot` | 4 个文件、19 项测试通过 |
| 查看器 `next build --webpack` | 通过；未修改 package.json 或增加依赖 |
| Playwright `viewer.spec.ts --grep 'shows Exp6-4 GT' --workers=1` | 1 passed；验证 GT/spconv/official_flex 三窗口、K 切换与样本集合 |
| 原产物 SHA-256 与字节比对 | 108 项通过；64 项 Git 产物与原文件逐字节一致，44 项外部文件原位保留 |
| 结果数据解析与有限值检查 | 41 份 JSON、1,046 条 JSONL 通过 |
| 结果提交边界与文档链接 | 仅 10 个 Exp6-4 日志条目、2 张 PNG，无训练源码；无未说明的缺失链接 |

浏览器测试使用隔离快照的构建产物和既有本地 Exp6-4 数据，通过 `127.0.0.1:33264` 临时启动服务；验收后已停止。未导出新点云或部署网站。结果 JSON/JSONL、归档哈希、提交文件边界和新增交接链接已在结果提交前检查。

## 未重跑与例外

- 本机缺少完整 PyTorch/稀疏算子测试依赖，离线缓存也不足；本轮没有重跑模型完整 CPU/CUDA 测试、训练恢复或输出数值一致性测试，不将语法检查当成这些测试的替代。
- 2026-09-12 的 82 项 CPU、10 项 CUDA 验收属于[历史部署记录](./deployment_20260912_shared_gpu3.json)，不是本次执行结果。
- 官方 vendored `sparse_unet.py` 与 `utils.py` 保留 4 行原有行尾空白；完整 `git diff --check` 会报告它们。排除该官方源码目录后空白检查通过，不为通过格式检查改写锁定的上游源码。
- 未运行正式评估、推理或新训练；没有远程资源检查、服务器写入、公开发布或 Git push。

原诊断报告、指标、manifest 及已有分析产物保持字节不变。外部文件范围及克隆后需补齐的报告依赖见 [ARCHIVE.md](./ARCHIVE.md)。

提交前发现用户 Git 的 `core.autocrlf=input` 会规范化 6 份历史 CSV，从而改变归档哈希。已通过实验目录内的 `.gitattributes` 仅为这些 CSV 保留原 CRLF 字节；不修改全局 Git 配置、原 CSV 或清单中的哈希。

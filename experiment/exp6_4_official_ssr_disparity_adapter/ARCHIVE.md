# Exp6-4 Git 归档边界

整理日期：2026-09-21。本次只将既有 Exp6-4 源码与紧凑结果归档到本地 Git，不推送、不访问服务器、不重训或改写历史产物。方法与复用限制见[交接文档](../../docs/exp6_4_disparity_adapter_handoff.md)，验证范围见 [COMMIT_CHECKS.md](./COMMIT_CHECKS.md)。

## 随 Git 保存的内容

- 官方网络主体、来源与许可证，disparity 适配、冻结 Base 训练、配置、脚本、测试和 Exp6-4 查看器入口。
- 原部署记录、终点 A/B 训练与几何报告、配对结果、分析 JSON/JSONL/CSV、原诊断报告与图册 HTML。
- 最终两张汇总图：`analysis_20260914_r3/figures/paired_differences.png` 和 `update_diagnostics.png`。
- 实验 README、实验总日志中的 Exp6-4 条目及仓库内交接文档。

[git_archive_manifest.json](./git_archive_manifest.json) 为原产物的补充清单，路径相对本实验目录，记录字节数、SHA-256 与 `git` / `external` 分类。清单中 64 个 Git 产物共 8,207,629 字节；不计本次新增说明与源码。原 `artifact_manifest.json` 和结果同步清单保持原样，不能将其误读为本次 Git 文件列表。

实验目录的 `.gitattributes` 仅对既有分析 CSV 关闭换行归一化并认可 CRLF 行尾，以保留原始字节及 provenance 哈希；不改变其他文件的 Git 配置。

## 保留原位、不随 Git 分发的内容

清单中 44 个外部分析产物共 133,663,456 字节，包括 24 个 selected arrays NPZ、12 张案例图、`curves.png`、3 份设备 autotune cache 和 4 份执行阶段 Python 快照。它们均未删除或移动；运行入口源码已在代码提交中保留，历史快照不作为重复的活动源码提交。

权重、训练目录、数据缓存和查看器 PLY/RGB 资产也不随 Git 保存，继续沿用项目既有外部资产规则。清单仅覆盖本地已有的分析、结果与部署记录，不表示已经清点全部远程 checkpoint 或网站资产。

外部分析产物的本地位置：

```text
/Users/zzc/Desktop/科研/2026_TPAMI_InfiniGeometry/Project/InfiniDepth/experiment/exp6_4_official_ssr_disparity_adapter
```

历史服务器位置依据原部署与同步记录，未在本轮检查其当前可用性：

```text
ZJU3DV-S115:/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/exp6_4_20260911/source_v8/experiment/exp6_4_official_ssr_disparity_adapter
```

新机器仅克隆 Git 时，原 `REPORT.md` 的曲线、案例图片与 Python 快照链接，以及 `gallery.html` 的部分图片不会存在。阅读完整图册前，从已有本地副本或获授权的历史归档补齐所需文件到清单中的相对路径，并核对 SHA-256；不要覆盖已有且哈希不同的文件。无需重新推理，也不要修改原报告来假装这些文件随 Git 提供。

## 跨仓库与工作区文档

历史日志引用的 MoGe 复现、Exp6-3 评估协议和工作区 `docs/` 文档仍属于外部资料。本次不复制其他实验、不整理另一个仓库的提交。交接正文已放入当前仓库的 `docs/exp6_4_disparity_adapter_handoff.md`；主工作区 `docs/handoff/Exp6-4_官方SSR_Disparity适配版_代码交接.md` 仅保留本地入口。

### Exp6-5

Exp6-5 R1 的对齐梯度修复、两组更新方式及训练完成记录见相邻 MoGe 仓库，工作区相对位置为：

```text
Project/MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3/REVISION_R1.md
Project/MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3/README.md
```

它们不随当前 InfiniDepth 提交分发。Exp6-5 的无界 logZ 分支完成训练，不代表 Exp6-4 disparity 去限幅已验证；Exp6-4 仍按原配置与原 checkpoint 推理规则归档。

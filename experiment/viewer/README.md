# InfiniDepth 多实验点云查看器

本目录复用用户 MoGe 仓库 `agent/moge3-reproduction` 提交 `5796a09d24de5515bf9ea7ea14b1727f4bb37526` 中的 React/Three.js PLY 加载、相机同步、有限点过滤和测试框架，并将实验协议改为 InfiniDepth 原生 disparity。

窗口 A 固定显示 Hypersim GT；窗口 B/C 可独立选择 `initial`、`stage1_best`、`joint_best` 与 `K={0,1,3,5}`。预测点云使用每张图片的 GT 2%/98% disparity 统计反归一化，界面不得将其标注为模型原生米制输出。

Exp6-4 使用固定三窗口：GT、现有 spconv SSR、official_flex SSR。两种 SSR 只在各自窗口内切换 K=0/1/3/5，默认均为 K=3；train/val/test 各 5 张，并与 Exp3、Exp6-3 的对应选图保持一致。

Exp6-5 使用固定四窗口：A 为 GT，B 为共同的原始 MoGe2 基座（仅 K0），C 为自实现 SSR，D 为本次 Hypersim 训练的官方 SSR（不是已发布的 MoGe3 权重）。C/D 默认使用各自最佳 checkpoint，可切最终 30,000 步和 K0/1/3/5。分为 Hypersim 15 张与跨数据集 20 张两个入口，选图分别沿用 Exp6-4 和 Exp6-1。每个 checkpoint 的所有 K 共用 K0 的显示对齐，本轮仅可视化，不展示评估指标。详见 [Exp6-5 可视化记录](../../../MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3/VISUALIZATION.md)。

Exp6-1 的新清单若同时包含 `exp3_stage1` 和 `exp3_joint`，窗口 A 为 GT，B 为 InfiniDepth + SSR（RGB，可选 Stage1 best / Joint best，默认 Joint best K=3），C、D 都为官方 MoGe-3（分别默认 K=0、K=3）。各窗口独立切换 K；仅官方资产的旧清单仍保持原三窗口。Exp3 资产必须标注 GT 分位数尺度还原，并显示无效预测率。数据导出和发布记录见 [补充说明](../../../MoGe-v3-reproduction/experiment/exp6_1_moge3_official_reproduction/04_checkpoint_visualization/EXP3_RGB_COMPARISON.md)。

2026-09-15 二十张难例版本：点云改用可取消的原生 `fetch` 和既有 PLYLoader，不再使用全局解码缓存。切图时取消旧请求并释放源几何与显示几何；HTTP 缓存仍由浏览器管理。各窗口只持有当前点云，重复查看可能重新解析文件。导出器逐张读取输入，资源与验收记录见 [二十图报告](../../../MoGe-v3-reproduction/experiment/exp6_1_moge3_official_reproduction/04_checkpoint_visualization/HARD20_RESULTS.md)。

## 数据边界

`public/data/` 的 PLY、RGB 和实验 manifest 是外部导出资产，不随 Git 分发；`experiments.json` 是已版本化的入口目录。InfiniDepth 与 MoGe 各实验使用对应导出器，不应将所有资产都归为同一个训练器生成。新克隆需按各实验的发布清单补齐已有资产，不能用新推理覆盖历史目录。

## 本地或获授权的服务器验收

```bash
cd experiment/viewer
npm ci
npm run typecheck
npm test
npm run build:next
npm run test:e2e
```

端到端测试按实验覆盖三/四窗口、阶段/K 切换、预览和移动端。测试产物写入 Playwright 独立输出目录，不写回历史结果目录。运行 Exp6-5 真实资产验收需设置 `EXP65_REAL_ASSETS=1`；默认 fixture 测试不能代替真实资产验收。

2026-09-23：类型检查、22 项单元测试和 `next build --webpack` 通过；Exp6-1/3/4/5 的 8 项浏览器回归通过，其中 1 项为图库 fixture、7 项使用真实资产，包含 20/35 图连续切换。仅访问本机临时服务，未部署网站。详情见[本次验收](./acceptance_20260923.json)与[总交接索引](../../docs/experiment_handoff_20260923.md)。

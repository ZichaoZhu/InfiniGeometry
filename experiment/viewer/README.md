# InfiniDepth Disparity Refiner 三窗口查看器

本目录复用用户 MoGe 仓库 `agent/moge3-reproduction` 提交 `5796a09d24de5515bf9ea7ea14b1727f4bb37526` 中的 React/Three.js PLY 加载、相机同步、有限点过滤和测试框架，并将实验协议改为 InfiniDepth 原生 disparity。

窗口 A 固定显示 Hypersim GT；窗口 B/C 可独立选择 `initial`、`stage1_best`、`joint_best` 与 `K={0,1,3,5}`。预测点云使用每张图片的 GT 2%/98% disparity 统计反归一化，界面不得将其标注为模型原生米制输出。

## 数据边界

`public/data/` 由服务器上的 `training.disparity_refiner.export_assets` 生成，包含实验目录、PLY、RGB 和查看器 manifest，全部由 `.gitignore` 排除。Git 只保存查看器源码、锁文件和测试。Exp2 通过受保护脚本登记后会自动成为第二个可切换实验，不需要改前端源码。

## 服务器验收

```bash
cd experiment/viewer
npm ci
npm run typecheck
npm test
npm run build:next
npm run test:e2e
```

端到端测试覆盖桌面与移动端布局、三个真实 WebGL canvas 的非空像素、阶段/K 切换、交互模式和截图输出。上述命令只在实验室服务器执行。

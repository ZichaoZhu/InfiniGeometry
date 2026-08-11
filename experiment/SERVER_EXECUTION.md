# 实验室服务器执行规范

## 固定目录

- 仓库：`/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/InfiniDepth`
- 环境：`/mnt/data/home/zhuzichao/envs/infinidepth_disparity_ssr`
- 缓存：`/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr`
- 临时文件：`/mnt/data/home/zhuzichao/tmp/infinidepth_disparity_ssr`
- 只读数据：`/nas1/datasets/hypersim/raw`

不得使用 sudo、系统 Python、共享 `/tmp`，不得查看、终止、移动或修改其他用户的目录、进程和任务。环境安装只能写入上述个人目录。根据服务器 CUDA/PyTorch 版本安装匹配的 SpConv 2.x wheel，安装后先执行 `python -c "import spconv.pytorch"`；不能凭本地环境猜测 CUDA wheel。

## 首次准备

```bash
cd /mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/InfiniDepth
source /mnt/data/home/zhuzichao/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/data/home/zhuzichao/envs/infinidepth_disparity_ssr
export TMPDIR=/mnt/data/home/zhuzichao/tmp/infinidepth_disparity_ssr
export XDG_CACHE_HOME=/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export CONDA_PKGS_DIRS=$XDG_CACHE_HOME/conda_pkgs
export TORCH_HOME=$XDG_CACHE_HOME/torch
export HF_HOME=$XDG_CACHE_HOME/huggingface
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
export npm_config_cache=$XDG_CACHE_HOME/npm
export PLAYWRIGHT_BROWSERS_PATH=$XDG_CACHE_HOME/ms-playwright
export INFINIDEPTH_CHECKPOINT=$PWD/checkpoints/depth/infinidepth.ckpt
export INFINIDEPTH_TEST_TMP_ROOT=$TMPDIR/tests
```

确认当前分支、origin、工作区和 GPU 后，运行 `bash experiment/server_verify.sh`。禁止在本地电脑执行该脚本。
Playwright Chromium 如未安装，只能在上述环境变量生效后执行 `npx playwright install chromium`，不得写入其他用户或共享缓存。

## Exp1 顺序

1. 完成正式测试与 K0 回归。
2. 依次执行 `run.sh 01` 至 `run.sh 05`，每个运行使用独立初始化。
3. 执行资产导出与汇总。
4. 执行查看器类型检查、单元测试、生产构建和 Playwright。
5. 验证实验目录；只有 `gate_passed=true` 时才运行 `register_exp2.py`。

`stage1_best.pt` 和 `joint_best.pt` 只保存模型、评测和来源字段，不重复保存 AdamW 状态。只有 `last.pt` 用于 `--resume`；Exp1 每 1000 步及阶段终点写入，Exp2 每 2500 步及阶段终点写入。该周期只减少存储 I/O，不改变模型更新、评测频率或停止规则。

```bash
python -m training.disparity_refiner.export_assets \
  --experiment experiment/exp1_infinidepth_disparity_ssr_single_image_overfit
python -m training.disparity_refiner.summarize \
  --experiment experiment/exp1_infinidepth_disparity_ssr_single_image_overfit
python experiment/validate_experiment.py --check-git
```

查看器生成的 PLY、RGB 和 manifest 位于 `experiment/viewer/public/data/`，由 Git 排除；两张最终汇总 PNG 位于实验的 `artifacts/` 并纳入 Git。

## Exp2 登记

先根据单卡显存试跑正式前向，选择 microbatch 1 或 2；登记脚本不会接受其他值，也不会在 Exp1 门槛失败时创建目录。

```bash
python experiment/register_exp2.py --microbatch-size 1
```

# Exp6-4 单组运行交接

2026-09-23：新增 official_flex 配置准备入口，复用原训练器。仅完成本地准备逻辑验收；真实 checkpoint 数值一致性、完整 PyTorch/CUDA 与新环境短测仍待执行。不得据此宣称新机器训练已经验收。

## 方法与依赖

这是冻结 InfiniDepth RGB Base 的官方 SSR 网络主体 disparity 适配版，不是完整官方 MoGe3。模型、限幅、损失、batch、采样与 Val100 不变，仍限单卡。原 `config.json`、A/B `run.sh`、历史权重和结果不修改。

需要 InfiniDepth 原有 PyTorch、DINOv3、图像/数据读取依赖，以及锁定 FlexGEMM 与官方稀疏算子环境；spconv 是共享项目历史依赖，本轮没有拆除。版本来源见 [SOURCE.json](../../InfiniDepth/model/official_moge_ssr/SOURCE.json) 和[历史部署](./deployment_20260912_shared_gpu3.json)。不自动安装、下载权重或修改共享环境。

先根据 [paths.example.json](./paths.example.json) 准备自己的 `paths.local.json`。所有路径使用绝对路径；源码 checkout、只读数据、缓存和运行输出互不覆盖。Base 权重需在本人指定的 `safe_root` 内；运行输出必须在该目录的独立子目录，不能放入源码、缓存或输入目录。

`data_root` 必须包含原 manifest 指向的两个索引文件及原始数据树；已有缓存可继续复用。不会重扫数据集、复制整份数据或改变 train/val/test 的选择。固定 Local masks 和 Exp6-3 协议是外部依赖，不随 Git 提供；其 SHA-256 校验保持原值。

## 配置检查与生成

以下从 InfiniDepth 仓库根目录执行。先替换路径，不直接使用示例用户名。

```bash
python experiment/exp6_4_official_ssr_disparity_adapter/prepare_official.py \
  --paths experiment/exp6_4_official_ssr_disparity_adapter/paths.local.json \
  --output-config experiment/exp6_4_official_ssr_disparity_adapter/local/official.json \
  --output /mnt/data/home/YOUR_USER/exp64_work/runs/official_new \
  --check-only --check-assets
```

`--check-only` 不写文件、不启动任何进程。`--check-assets` 读取配置 manifest、两份索引并核验哈希，检查权重/缓存存在和固定评估资源；不是全量数据完整性扫描，也不核验全部模型依赖可导入。

检查通过后，去掉 `--check-only`，生成不可变的单组配置，并打印启动与恢复命令。只允许在本实验 `local/` 下生成配置，不能覆盖历史模板或 arms。已存在但内容不同的配置会拒绝；运行目录非空时也拒绝准备新训练。

## 短测、训练与恢复

短测在准备时增加 `--smoke`，同时使用独立的 `local/official_smoke.json` 和 `runs/official_smoke_new`。沿用历史 8 图、56 步短测配置，不占用正式目录。不要把 smoke checkpoint 当成正式实验恢复点。

准备器只打印命令。获得 GPU 运行授权后，显式设置 `CUDA_VISIBLE_DEVICES`，再运行打印出的训练命令；这里的 `--device cuda:0` 指选定可见 GPU 中的第一张，不负责自动选卡。沿用历史进程级设置：

```bash
export FLEX_GEMM_AUTOTUNE_MODE=always
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/mnt/data/home/YOUR_USER/exp64_work/tmp/official_flex_cache.json
```

先在本人目录创建缓存父目录；不使用他人的 cache。正式配置仍为 20,000 步、microbatch 1、累积 8，仅 SSR 更新。共享 GPU 时先检查可用显存，不停止他人任务。

恢复直接使用生成时的同一配置和命令，追加 `--resume <该运行>/checkpoints/last.pt`，不要再次生成配置或换目录冒充原运行。完整恢复仍要求配置文件 SHA、backend、refiner 配置、refiner-only 身份、world size 及 optimizer 状态一致。

历史 `stage1_best.pt` 可按原方法加载用于推理，但不是完整训练恢复点。路径迁移产生的新配置不能绕过旧 `last.pt` 的哈希校验；新环境从官方 Base 开始的新运行与旧运行严格区分。

## 已有权重评估

准备的配置可交给原评估器；它使用显式路径但不改变固定协议和 Local masks。评估需要新的空输出目录，示例先用 `--limit 1` 验收，正式 Val100 评估须另行确认。

```bash
python experiment/exp6_4_official_ssr_disparity_adapter/evaluate_pair.py \
  --config experiment/exp6_4_official_ssr_disparity_adapter/local/official.json \
  --checkpoint /path/to/official_flex/stage1_best.pt \
  --output /mnt/data/home/YOUR_USER/exp64_work/evaluations/check_one \
  --limit 1
```

网站导出仍保留历史 A/B 的 `stage1_best=spconv`、`joint_best=official_flex` 映射，本轮没有改成官方单组网站部署工具。独立训练与展示发布是不同操作。

## 本次验证与待办

- 标准库 7 项测试通过：科学配置不变、旧 NAS 限制、新路径边界、不可变配置、固定评估哈希、CLI 无写入/无训练，以及终止报告只写入已验证的主 rank 输出位置。
- Python 语法和差异空白检查通过；模型、loss、数据文件及原配置未修改。
- 本机缺少 torch/h5py 等依赖，完整训练 CPU/CUDA 测试未重跑。已有历史验收不是本轮验收。
- 尚需在获授权的新环境比较原版/整理版相同 checkpoint 的 K0/1/3/5 输出，核验 Base 冻结、短步训练、暂停恢复、RNG 与下一批索引；新配置不得在此之前标为正式训练就绪。
- GPU 扩展、去限幅和新训练目标不属于本次整理。

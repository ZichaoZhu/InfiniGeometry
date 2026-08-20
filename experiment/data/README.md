# 数据清单

`hypersim100_train_manifest.json` 从用户 MoGe 仓库 `agent/moge3-reproduction` 的提交 `5796a09d24de5515bf9ea7ea14b1727f4bb37526` 逐字迁移，SHA-256 为 `801f544ef7e0326ff2df0aa2d604e8ab01a6d0ba001762696b85a040fafd8a24`。

该文件只保存样本来源、相机射线元数据和哈希，不包含 RGB 或 depth。正式训练只选择其中 `split=train` 的条目，并直接只读访问 `/nas1/datasets/hypersim/raw`；清单中的 val/test 条目不用于本项目结论。

`hypersim_full_manifest.json` 只锁定 Hypersim 官方 scene split CSV 和相机参数 CSV 的路径、计数与 SHA-256，不枚举或复制 59,543 张 train 图。Exp3 启动时从两份官方 CSV 构造只读路径索引，训练时按需读取 RGB 和 radial depth。

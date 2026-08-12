# Realtime Pose Plan

- Python 本地主链路的数据字段、维度和时序语义只以仓库根目录的 `contract.md` 为准。
- source、Task Store、normalizer、训练、采样和评估都读取调用方明确传入的实际目录。
- source 由相对路径和 split 选择；Task Store 由 `<task_dir>/<split>/shards/shard_*` 定义；normalizer 由指定目录中的八个 `.pt` 文件定义。
- 数据链路不读取 manifest、meta、hash、时间戳目录或 `latest_*` 指针，也不保留旧 Task Store 兼容入口。
- 训练 checkpoint 的 `latest_run` 与 `--resume_latest` 保留，仅用于训练恢复。
- 长序列评估直接读取 `source_dir + split_dir + split`，不构建复制版 eval set。
- Unity runtime 与导出链路后续单独同步，不作为当前 Python 主链路的运行保证。

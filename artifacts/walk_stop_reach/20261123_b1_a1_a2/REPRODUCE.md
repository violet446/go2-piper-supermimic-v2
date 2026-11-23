# 复现 B1 同回合串联

从 WSL Ubuntu 的仓库根目录运行。先按项目首页的说明放置已训练的 A1 与 A2 权重，留出评估会核对冻结配置中的权重哈希。 [权重下载与放置位置](../../../README.md#下载已训练权重)

```bash
entry=experiments/walk_stop_reach/run.sh
run=20261123_b1_repro
config=artifacts/walk_stop_reach/20261123_b1_a1_a2/config.json
bash "$entry" development --run "$run" --config "$config"
bash "$entry" holdout --run "$run"
bash "$entry" video --run "$run"
```

[实验口径与对照结果](../../../experiments/walk_stop_reach/README.md) · [留出指标](holdout/metrics.json)

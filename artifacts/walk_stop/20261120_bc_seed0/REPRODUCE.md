# 复现 20261120_bc_seed0

本运行使用 `data/walk_stop/20261120_crawl_seed0`。在 WSL Ubuntu 仓库根目录运行下列命令会在新的目录复现同一配置、种子和评估顺序。

```bash
entry=experiments/walk_stop/run.sh
run=20261120_bc_repro_seed0
dataset=20261120_crawl_repro_seed0
config=artifacts/walk_stop/20261120_bc_seed0/config.json
bash "$entry" prepare --run "$run" --dataset "$dataset" --config "$config"
bash "$entry" train --run "$run" --dataset "$dataset" --device cuda
bash "$entry" evaluate --run "$run" --dataset "$dataset" --split development --no-video
bash "$entry" evaluate --run "$run" --dataset "$dataset" --split holdout
```

环境版本见 [environment.json](environment.json)，实际训练参数见 [config.json](config.json)，GPU 统计见 [training_metrics.json](training_metrics.json)。

# 复现 B2 联合模仿对照

从 WSL Ubuntu 的仓库根目录运行。以下命令使用验收通过的正式 B1 配置，重新采集 12 条训练示范并检查物理回放。

```bash
entry=experiments/walk_stop_reach/run.sh
run=20261123_b2_repro_seed0
dataset=20261123_joint_repro_seed0
bash "$entry" joint train --run "$run" --b1 20261123_b1_a1_a2 --dataset "$dataset" --device cuda
bash "$entry" joint development --run "$run"
bash "$entry" joint holdout --run "$run"
bash "$entry" video --run "$run"
```

[实验口径与对照结果](../../../experiments/walk_stop_reach/README.md) · [配置](config.json) · [下载权重](https://github.com/violet446/go2-piper-supermimic-v2/releases/download/walk-stop-reach-20261123/go2-piper-b2-joint.pt) · [训练指标](training_metrics.json)

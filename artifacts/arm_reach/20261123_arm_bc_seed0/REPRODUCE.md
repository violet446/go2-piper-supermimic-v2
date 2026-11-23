# 复现 A2 独立六关节伸收

从 WSL Ubuntu 的仓库根目录执行，使用现有校准训练和留出 NPZ。采用新运行名保留结果。

```bash
run=20261123_arm_bc_repro_seed0
bash experiments/arm_reach/run.sh train --run "$run"
bash experiments/arm_reach/run.sh holdout --run "$run"
bash experiments/arm_reach/run.sh video --run "$run"
```

[实验口径与对照结果](../../../experiments/arm_reach/README.md) · [配置](config.json) · [下载权重](https://github.com/violet446/go2-piper-supermimic-v2/releases/download/walk-stop-reach-20261123/go2-piper-a2-arm-reach.pt) · [训练指标](training_metrics.json)

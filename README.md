# Go2 + PiPER 带臂走停与伸收

项目目标是在同一个 MuJoCo 仿真回合中，让 Go2 携带收好的 PiPER 走到指定位置、停稳，然后向固定目标伸手、停留并收回。**截至 2026-11-23，同回合走停伸收在两个留出场景通过验收，当前组合基线是 B1。**

## 实验路线与结果

A1 和 A2 分别验证腿和臂。B1 把两者接成同一个任务，B2 用整段示范做联合训练对照。

| 阶段 | 机器人做什么 | 实测结果与报告 |
| --- | --- | --- |
| **带臂走停（A1）** | 机械臂收拢，学得的 12 关节腿部策略控制直行和停车。 | 两条留出都停稳，停车位置误差均约 0.21 cm。[走停报告](experiments/walk_stop/README.md) |
| **站稳伸收（A2）** | 独立控制器维持腿部站姿，学得的 6 关节臂策略控制伸出、停留和收回。 | 两条留出的停留平均到点误差为 0.23 和 0.13 cm。[伸收报告](experiments/arm_reach/README.md) |
| **同回合走停伸收（B1）** | 先用 A1 走到位并停稳，再从停车后的实际仿真状态切到站立与 A2 伸收。 | 两条留出走完全程，无跌倒，停留平均到点误差为 0.54 和 0.73 cm。[串联报告](experiments/walk_stop_reach/README.md#b1-模块串联) |
| **整机联合模仿对照（B2）** | 用 B1 采集的整段示范，联合拟合 18 关节动作，再与 B1 在相同场景对照。 | 两条留出均通过，到点和停车后偏移与 B1 基本相同。[联合训练对照](experiments/walk_stop_reach/README.md#b2-联合模仿对照) |

想先看机器人做完整动作，可以直接打开 [B1 短距视频](artifacts/walk_stop_reach/20261123_b1_a1_a2/holdout/offset_short/walk_stop_reach.mp4) 或 [B1 长距视频](artifacts/walk_stop_reach/20261123_b1_a1_a2/holdout/offset_long/walk_stop_reach.mp4)。这两条是在平地、短距离、固定已知目标下完成的本地仿真实验。

### 下载已训练权重

从 [走停伸收 Release](https://github.com/violet446/go2-piper-supermimic-v2/releases/tag/walk-stop-reach-20261123) 下载权重，再放到对应位置。B1 串联使用 A1 和 A2 两份权重。

| Release 附件 | 放到仓库内的位置 |
| --- | --- |
| `go2-piper-a1-walk-stop.pt` | `artifacts/walk_stop/20261120_bc_seed0/policy.pt` |
| `go2-piper-a2-arm-reach.pt` | `artifacts/arm_reach/20261123_arm_bc_seed0/policy.pt` |
| `go2-piper-b2-joint.pt` | `artifacts/walk_stop_reach/20261123_b2_joint_bc_seed0/policy.pt` |

## 文件与文档

- 想按运行顺序读源码，或用 GPT 分轮学习，见 [核心代码导读与提示词](CODE_WALKTHROUGH.md)。
- `experiments/` 放各任务的实验报告、配置和复现入口。上表每个报告都有对应命令。
- `data/` 放训练与评估所用的示范。`artifacts/` 放实际运行产生的配置、权重、日志、指标和视频。
- [PLAN.md](PLAN.md) 记录当时的计划、验收口径和实施进度。2026-11-19 的历史结果见 [左右平面伸手实验](REACH_LR_RESULTS.md)，其中的 18 关节整机站立伸手策略用于 A2 对照。

- `vendor/go2_piper/` 是固定版本的组合模型。源 XML 保留，`reference.py` 生成供本项目使用的派生 XML。派生版本改用 mjlab 的执行器、默认接触选项和零碰撞 margin，以适配 MuJoCo Warp。
- `vendor/supermimic-quadruped-0.2.0/` 包含四足 SuperMimic 和本项目添加的 Go2 + PiPER 机器人配置。

## 运行环境

在 WSL Ubuntu 24.04 中运行。项目使用 Python 3.12、CUDA 和支持 EGL 的 MuJoCo 渲染环境。新克隆可用 `uv` 根据随附的锁文件创建环境。

```bash
runtime="$HOME/.cache/go2-piper-supermimic-v2"
UV_PROJECT_ENVIRONMENT="$runtime/venv" uv sync --locked --no-dev --python 3.12 \
  --project vendor/supermimic-quadruped-0.2.0
uv pip install --python "$runtime/venv/bin/python" imageio-ffmpeg==0.6.0 matplotlib==3.11.2
```

## 历史站立伸手基线

<details>
<summary>展开早期 SuperMimic 结果、公开下载与复现命令</summary>

2026-11-19 公开发布了站立伸手的 [对比视频（MP4）](https://github.com/violet446/go2-piper-supermimic-v2/releases/download/reach-lr-20261119/go2_piper_demo_vs_policy.mp4) 和 [评估权重](https://github.com/violet446/go2-piper-supermimic-v2/releases/tag/reach-lr-20261119)。对比视频左侧是 IK 与 PD 生成的参考，右侧是策略在物理仿真中的表现。

早期策略每个控制周期同时输出 12 个腿关节和 6 个臂关节的指令。示范的腿关节目标设为固定站姿，臂关节目标随伸手动作变化，夹爪开合固定。用 IK 生成关节指令，在带地面的 MuJoCo 物理仿真里用 PD 执行，再以 50 Hz 记录实际运动。`data/train/` 保存 6 条训练参考，`data/holdout/` 保存 2 条留出参考。8 条参考都全程保持四脚着地，最大机身倾角为 2.45°。

| 模型 | SuperMimic 迭代 / 更新 | 留出 1 到点误差 | 留出 2 到点误差 | 最大倾角 |
|---|---:|---:|---:|---:|
| 从头短训练 | 300 / 104 | 8.64 cm | 4.71 cm | 4.30° |
| 仅示范初始化 | 0 / 0 | 5.98 cm | 4.69 cm | 2.45° |
| 示范初始化后做 SuperMimic 更新 | 150 / 44 | 5.59 cm | 4.66 cm | 2.42° |

每个策略都完整跑完两条 3 秒留出轨迹。到点误差是伸臂停留段的手爪位置与 IK 请求点的距离。参考本身经 PD 执行后到请求点也还差约 2.9 和 2.6 cm。短训练足以验证更新和回放，该实验只涵盖站立状态下少量近距离目标。示范初始化后继续做 44 次更新带来的增益很小。

从头实验用 48 个并行环境、两个 64 样本的训练批次、12 步想象视野。在训练期间做的 `nvidia-smi` 采样，整张卡峰值为 1,745 MiB，含桌面和同时运行的评估。上游裸 Rover 的 72 GiB 和 20,480 个环境是完整实验配置。

要在公开仓库中评估 2026-11-19 的校准策略，先下载示范初始化检查点。以下命令使用随仓库发布的两个校准留出参考。

```bash
model_dir=artifacts/reach_lr_20261119/control_calibrated/training/lr_calibrated_100_initial
mkdir -p "$model_dir"
curl -fL https://github.com/violet446/go2-piper-supermimic-v2/releases/download/reach-lr-20261119/go2-piper-calibrated-initial.pt \
  -o "$model_dir/policy.pt"
bash run.sh reach-lr compare artifacts/reach_lr_replay \
  --data-dir artifacts/reach_lr_20261119/holdout_data/calibrated \
  --checkpoint calibrated_initial "$model_dir/policy.pt"
```

从头重跑早期短训的命令如下。已有本机实验可用新运行名保存新结果。2026-11-19 校准对照的命令见 [左右伸手实验](REACH_LR_RESULTS.md#复现)。

```bash
bash run.sh reference
bash run.sh train reach_smoke 300
bash run.sh warm reach_warm 150
bash run.sh evaluate --checkpoint artifacts/training/reach_warm/ckpt_latest.pt --clip 0
bash run.sh evaluate --checkpoint artifacts/training/reach_warm/ckpt_latest.pt --clip 1 --no-video
```

</details>

## 模型来源

机器人资产来自 [LeggedManip Lab 的 Go2 + PiPER 模型](https://github.com/zzzJie-Robot/LeggedManip_Lab/tree/4e12109d1eccb1d9da2e4a35f189fabea3f13677/mujoco/robots/go2_piper)。

`vendor/go2_piper/training_model.xml` 为项目修改的派生模型。随附 [上游机器人资产许可证](vendor/go2_piper/LICENSE) 和 [SuperMimic 第三方说明](vendor/supermimic-quadruped-0.2.0/THIRD_PARTY_NOTICES.md)。

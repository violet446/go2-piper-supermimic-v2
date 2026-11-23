# A1 带臂直行与停车 · 2026-11-20

**A1 首版通过，具备启动独立 A2 的条件。** 学得的 12 关节腿部策略在完整 Go2＋PiPER 模型上通过两条留出测试，停车位置误差约 2.1 mm，末 0.5 s 四脚持续接触地面。A1 验证使用平地、低速单脚交替迈步和一个训练种子。B 还需要在同一回合中验证控制器切换，并从 A1 停稳后的真实状态开始伸手。

## 输入和示范选择

使用原有 `training_model.xml`、`reference.floor_model()`、仿真步长和关节 PD 参数。机身有六个自由度，地面与脚产生的接触力决定机身如何移动。PiPER 装在原来的位置并保留质量，六个臂关节由 PD 保持 `(0, 0.25, −0.15, 0, 0, 0)` rad 的收拢目标。该姿态把腕和爪收在上臂附近。模型的夹爪没有活动关节。每条先从配置中的坐标和朝向物理站立 1.2 s，再以站定后的机身位置为测距起点，沿指定朝向设一个固定的世界停车点。

按 [Kine2Go 数据卡](https://huggingface.co/datasets/MIMUW-Robotics/kine2go/blob/7f03a3fb7f6b2e684899530862ba103e81570363/README.md) 检查了 `solo8_walk` 和 `vhdc_horse1_s1_walk_01` 两个 60 Hz 小样。逐片元数据分别标明 BSD-3-Clause 和 CC BY-SA 4.0。实际 `motion.npy` 分别为 120 和 600 帧，两份 `clip.json` 均写 571 帧。逐帧动作和运动学参考是两类数据。上游 PPO 的 12 个动作以 FR、FL、RR、RL 排列，每腿 hip、thigh、calf。默认关节角加 `0.25 × action` 形成目标，显式 PD 的 `Kp=30`、`Kd=1.5`，启用延迟时执行上一控制步动作，训练配置还包含电机偏置和增益随机化。见 [动作和力矩实现](https://github.com/nomagiclab/kine2go-pipeline/blob/65b39f104706c1b4307dfc2a7df2aa8bed8d20aa/go2_genesis/locomotion_env.py#L572-L584) 与 [延迟实现](https://github.com/nomagiclab/kine2go-pipeline/blob/65b39f104706c1b4307dfc2a7df2aa8bed8d20aa/go2_genesis/locomotion_env.py#L918-L925)。

两份实测 `motion.npy` 各自以关节类型分组，每组是 FL、FR、RL、RR。按这些列和文件里的机身位姿重算脚的位置，平均误差低于 `8e-8 m`。直接按所宣传的动作顺序解释同一数组时误差达 29–32 cm。以 0 为起始列号，转为本机按腿排列的列是 `[6,10,14,7,11,15,8,12,16,9,13,17]`。上游导出参考使用 Genesis 完整自由度数组。见 [运动学导出实现](https://github.com/nomagiclab/kine2go-pipeline/blob/65b39f104706c1b4307dfc2a7df2aa8bed8d20aa/motion_retargeting/env.py#L167-L180)。

重排并线性插值至 50 Hz 后，用当地 PD 把参考角度当作位置目标，在带臂模型上跑了一次直接迁移诊断。以源片段初始位姿、零速度开始，必要时抬升初始机身以清除地面穿透，之后完全靠物理推进。Solo8 跑完 100/100 步，前进 60.1 cm，有 7 个四脚同时离地的采样点，末速度 0.586 m/s。马步段在第 438/500 步出现非足部触地，前进 82.3 cm 后按跌倒结束，末速度 0.284 m/s。源文件末半秒平均速度分别是 0.852 和 0.338 m/s，缺少站定段。诊断只测试这两份角度参考的直接迁移。A1 最终使用本机生成的完整走停示范。见 [外部样本实测](../../artifacts/walk_stop/20261120_kine2go_probe/metrics.json) 和 [样本文件与许可](../../data/external/kine2go/README.md)。

## 本机走停与模仿学习

1. **先让脚承重，再迈一步。** 规划器用当前站姿和停车距离安排四轮步幅，比例是 15%、35%、35%、15%。每次抬一只脚，先将机身参考移到其余三脚支撑的区域。示范控制器根据实际机身位置、速度和姿态，分配三脚或四脚应承担的地面反力。抬起的脚用逆运动学求腿关节目标，也就是把「脚该去哪」换成三个关节角。最初仅按假定机身路径算关节角的试走，在 20 cm 和 25 cm 指令下分别倒退约 6.5 cm 和 5.1 cm，在 30 cm 指令下侧翻。用实时机身状态分配支撑力后，25 cm 开发示范走到了位，至少三脚支撑，最大倾角 1.58°。
2. **保存动作并重新推进物理。** 从平衡后的完整积分状态复位，每 20 ms 仅重放一次记录的 12 个目标角，底层以 5 ms 的步长执行 PD。物理回放逐帧得到与录制一致的 `qpos`、`qvel` 和脚部接触。25 cm 开发示范末位置误差 0.210 cm，末窗最大速度 0.00152 m/s。据此在训练和生成留出前冻结验收。六个训练距离各有一条原样示范和五条小幅腿部指令扰动示范。36 条中有 18 条通过。剔除的 18 条中 17 条末窗速度超线，1 条末窗四脚支撑不全。扰动样本保留实际执行指令供回放，监督标签用同一观测下示范控制器给出的无扰动目标，以学习偏离轨迹后的修正。
3. **只用示范训练腿部。** 纯行为克隆把「当时的机身与腿的状态、离目标姿态或落脚点有多远、哪只脚该迈步」映射到 12 个腿部目标角。网络预测相对当前关节角和阻尼项的修正。评估每一步重新读策略自己走出来的状态，规划器仍提供机身和脚部参考，示范反力分配和逆运动学退出控制。训练共 14,634 个观测－动作对、5,500 次优化，动作均方根误差 0.00542 rad。训练测量用时 7.26 s。RTX 4060 Laptop 整卡 200 ms 采样首值 321 MiB、峰值 459 MiB，PyTorch 峰值已分配 35.62 MiB、保留 44.00 MiB。

## 指标和结果

位置误差为末帧机身在水平面到固定停车点的直线距离。朝向误差为末帧与指定航向的最小绝对角差。停稳速度取最后 0.5 s 的 26 个采样点的最大水平速度，包含两个端点。跌倒判据为非脚部触地、机身高度低于 0.17 m 或倾角超过 45°。完成以 [冻结配置](../../artifacts/walk_stop/20261120_bc_seed0/config.json) 中的全部检查为准。核心门槛为位置 ≤5 cm、朝向 ≤10°、停稳速度 ≤0.05 m/s。另要求跑完整段、末窗四脚着地、全程无四脚离地，并检查倾角、侧偏、收臂、触地脚的滑移速度、16 次迈步的抬脚净空，以及起步与减速段的平均前进速度均低于巡航段。

| 场景 | 控制器 | 停车误差 cm | 朝向误差 ° | 停稳速度 m/s | 跌倒 | 完成 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 开发 25.0 cm | 示范回放 | 0.210 | 0.014 | 0.00152 | 否 | 是 |
| 开发 25.0 cm | 学得策略 | 0.214 | 0.034 | 0.00123 | 否 | 是 |
| 留出 21.5 cm | 示范回放 | 0.214 | 0.013 | 0.00155 | 否 | 是 |
| 留出 21.5 cm | 学得策略 | 0.211 | 0.040 | 0.00122 | 否 | 是 |
| 留出 28.5 cm | 示范回放 | 0.206 | 0.014 | 0.00149 | 否 | 是 |
| 留出 28.5 cm | 学得策略 | 0.215 | 0.029 | 0.00123 | 否 | 是 |

短、长两条留出的置位坐标分别为 `(0.11, −0.07) m` 和 `(−0.09, 0.08) m`，指定朝向分别为 +4°、−5°。各回合 813 个控制步、16.26 s。两条策略全程至少三脚接触，最大侧偏 4.92 cm，最大倾角 1.60°。16 次摆脚中最小的峰值抬脚高度约 4.72 cm，接触脚水平速度的 P95 约 0.022 m/s。单次摆脚的触地采样速度峰值最高约 0.237 m/s，明显高于 P95。稳定站立的证据来自末窗四脚接触与速度实测。逐脚和逐段数据见 [短距指标](../../artifacts/walk_stop/20261120_bc_seed0/holdout/offset_short/metrics.json) 与 [长距指标](../../artifacts/walk_stop/20261120_bc_seed0/holdout/offset_long/metrics.json)。视频中左栏为示范，右栏为策略，黄点是站定起点，绿点是停车目标。见 [短距对比](../../artifacts/walk_stop/20261120_bc_seed0/holdout/offset_short/comparison.mp4) 与 [长距对比](../../artifacts/walk_stop/20261120_bc_seed0/holdout/offset_long/comparison.mp4)。

两条留出距离均在训练距离之间。位置和航向输入以任务坐标表示，变换起点和朝向主要检查坐标处理。现有结果支持本组合模型在平地慢走到点，可以继续验证独立站立支撑下的 A2。B 的衔接仍需要实际测量停车状态交接、臂从收拢姿态起动，以及伸臂时腿部独立支撑的效果。历史 18 关节校准伸手基线的 79.30% 和 82.56% 是手爪伸出方向完成率，见 [历史伸手结果](../../REACH_LR_RESULTS.md)。

## 复现和产物

从 WSL Ubuntu 仓库根目录执行。入口复用历史项目的 Python 环境 `~/.cache/go2-piper-supermimic-v2/venv`，可通过 `GO2_RUNTIME_DIR` 指定其他已有环境。运行名和数据名请选新的以避免覆盖。`evaluate` 默认生成视频，加 `--no-video` 可只跑数值评估。

```bash
entry=experiments/walk_stop/run.sh
bash "$entry" external --run 20261120_kine2go_repro
run=20261120_bc_repro_seed0
dataset=20261120_crawl_repro_seed0
bash "$entry" prepare --run "$run" --dataset "$dataset"
bash "$entry" train --run "$run" --dataset "$dataset" --device cuda
bash "$entry" evaluate --run "$run" --dataset "$dataset" --split development --no-video
bash "$entry" evaluate --run "$run" --dataset "$dataset" --split holdout
```

- 数据位于 `data/walk_stop/20261120_crawl_seed0/`。`manifest.json` 记录来源、模型哈希、关节顺序、动作含义和数据划分。NPZ 的第 0 行是初始状态，动作第 `i` 行用于从状态 `i` 推进到状态 `i+1`。`target` 是实际执行的腿部目标，`label_residual_rad` 是训练标签。`schedule` 列依次为阶段、抬腿索引、阶段内移重心或抬脚标记、该段进度。速度和位置是 MuJoCo 的 SI 单位。
- 正式运行位于 `artifacts/walk_stop/20261120_bc_seed0/`。`config.json`、`environment.json`、`policy.pt`、`train.log`、`gpu.csv` 和 `training_metrics.json` 记录训练。`prepare.log` 记录示范筛选，`development/` 和 `holdout/` 保存逐条指标和策略轨迹。GPU CSV 列为本机采样时刻、已用 MiB、总量 MiB。

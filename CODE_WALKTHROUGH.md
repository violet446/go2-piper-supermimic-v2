# Go2 + PiPER 核心代码导读

从一次真实的走停伸收回合入手，沿着代码看清数据、控制、训练和评估各自做什么。主线是带臂走停（A1）、站稳伸收（A2）和同回合串联（B1）。整机联合模仿（B2）放在最后，作为已经理解 B1 之后的对照。项目结果与视频见 [README.md](README.md)。

## 复制给网页版 GPT 的提示词

```text
公开 GitHub 仓库：[粘贴仓库链接]
版本：[可填提交 SHA 或分支；留空时使用仓库首页显示的版本]

请你当我的代码导师，用中文带我分多轮读懂这个 Go2 + PiPER 项目。我第一次接触四足机器人和模仿学习，会基本 Python。我的目标是能自己说清设计原理、顺着文件和函数追到实际实现，并能独立验证一个判断。

先打开这个版本的 README.md 和 CODE_WALKTHROUGH.md，按导读的主线读代码。用任务中文名沟通，第一次出现时附上 A1、A2、B1 或 B2。首轮给我一份简短目录地图，然后只讲第 1 课。以后我说“继续”再进入下一课；我问细节时先讲清当前课。

每轮请做到以下几点：
1. 先给一句话结论和一个具体问题。用本项目的一拍动作或一个场景做例子，按输入、代码做了什么、输出交给谁的顺序讲。
2. 列出实际打开的仓库相对路径和函数或类名，给同版本的 GitHub 源码链接。能核实行号时附上行号。把理解本课必需的代码原样贴进对话，标明每段出自哪里。通常每次 1 到 3 个短片段。跨文件时请同时贴出关键的调用处和实现处，让我在聊天里就能对上。示意或简化代码要明确标为示意。
3. 至少挑一个变量沿调用链走一遍，交代它的含义、单位或形状，以及它在哪一步更新。首次遇到规划器、示范控制器、PD、行为克隆、闭环或留出，用一句白话和一个实际例子解释。
4. 每轮最后给我一道需要预测或解释的小题，先等我回答，再根据源码或实测检查我的推理。我说“给提示”时先给一条提示，我说“继续”时可以跳过题目。

优先用读到的真实源码解释。若无法打开文件，告诉我已读到哪里，尝试同版本的 GitHub Raw 内容。仍读不到时，仅列出当前课缺的具体文件和链接，请我补充这些文件。实测数字来自报告或指标文件；对原因的推测明确写成推测。

现在开始第 1 课。以 B1 为总览，带我从启动脚本跟到 execute_case，再解释为什么完成一次仿真回合需要规划、控制和物理推进三层。
```

## 分轮读代码

| 课 | 问题 | 入口与要跟的代码 |
| --- | --- | --- |
| **1. 从整机任务进仓库** | 一条运行命令如何启动同回合走、停、伸、收？ | `holdout` 由 [组合运行脚本](experiments/walk_stop_reach/run.sh) 进入 [组合入口](experiments/walk_stop_reach/experiment.py) 的 `main`，加载走停与伸收的检查点，再调用 `execute_case`。先看流程骨架和 [B1 的配置](experiments/walk_stop_reach/config.json)。 |
| **2. 从目标角到真实运动** | 关节目标、实际关节角、力矩和接触有什么关系？ | [仿真基础](experiments/walk_stop/simulation.py) 的 `Simulation.reset`、`advance` 和 `torque_step`。顺着 `floor_model` 到 [地面模型构造](reference.py) 和 [组合模型 XML](vendor/go2_piper/training_model.xml)。追踪一次 20 ms 控制周期内的四个 5 ms 物理步。 |
| **3. 示范怎样走起来** | 为什么抬一只脚前要移重心，谁把落脚点换成关节目标？ | [走停规划与示范控制](experiments/walk_stop/simulation.py) 的 `WalkPlan.plan` 和 `Teacher.action`。追踪一次从计划移重心、抬脚到输出腿目标的过程。先用三只支撑脚解释，再看代码里的支撑边和反力分配。 |
| **4. 运动怎样变成训练数据** | 观测和动作如何对齐，为什么要重放？ | [走停实验](experiments/walk_stop/experiment.py) 的 `demo`、`replay`、`prepare`，再到 [逐拍记录](experiments/walk_stop/rollout.py) 的 `rollout`。追踪同拍观测、用于重放的 `target` 和用于训练的 `label_residual_rad`。外部样本诊断可在本课末看 [Kine2Go 检查](experiments/walk_stop/external.py) 的 `inspect_kine2go` 与 `physical_replay`。 |
| **5. 腿部策略怎么学、怎么验收** | 训练损失低，为什么还要跑完整回合？ | [走停实验](experiments/walk_stop/experiment.py) 的 `train`、`policy_target`、`evaluate`，以及 [观测与指标](experiments/walk_stop/rollout.py) 的 `features`、`target_anchor` 和 `measure`。追踪观测到目标角，再追踪策略动作到下一拍自己的观测。 [走停报告](experiments/walk_stop/README.md) |
| **6. 六关节臂怎么学** | 如何从手爪的平面目标偏移生成整段伸收？ | [历史示范生成](reference.py) 的 `arm_ik`、`interpolate` 和 `record`，接到 [六关节实验](experiments/arm_reach/experiment.py) 的 `cases`、`train`、`arm_targets` 和 `arm_rollout`。追踪六个峰值角增量如何乘以时间曲线、加到预备姿态，再由 PD 执行。 [伸收报告](experiments/arm_reach/README.md) |
| **7. 停车状态怎么交给伸手** | 如何确认停稳、保持指令连续，并让世界目标保持固定？ | 回到 [组合入口](experiments/walk_stop_reach/experiment.py) 的 `execute_case`，按实际执行顺序跟踪 `goal`、`stopped_state`、`transition`、`reach_initial`、`arm_targets` 和最后的收拢。用 [B1 留出指标](artifacts/walk_stop_reach/20261123_b1_a1_a2/holdout/metrics.json) 区分停车误差、交接后最大偏差和停留到点误差。 |
| **8. 联合模仿对照，可选** | 为什么 B2 可以复现 B1，却几乎没有减少交接偏移？ | [联合训练](experiments/walk_stop_reach/joint.py) 的 `collect_demos`、`check_replay`、`JointPolicy` 与 `train_joint`。沿同一条 B1 轨迹追到两个训练损失，再比较相同留出场景。 [B2 对照结果](experiments/walk_stop_reach/README.md#b2-联合模仿对照) |

前三个任务的启动入口在各自的 `experiments/` 子目录。`data/` 保存示范，`artifacts/` 保存运行结果和配置，`vendor/` 提供共享模型和框架。2026-11-19 的站立伸手 SuperMimic 对照可作为背景阅读，在理解六关节实验后查看。 [历史对照报告](REACH_LR_RESULTS.md)

## 学完后的亲手练习

选一条 B1 留出，从 `rollout.npz` 的 `time`、`qpos` 和 `dock_xy` 计算三个时刻的停车点误差：确认停车、腿部交接结束、机械臂预备完成。先预测哪一段偏移最大，再用 `metrics.json` 的阶段时间戳取样。向 GPT 发你的预测、计算方式和三组结果，请它核对你量的是机身的世界坐标，并帮你为其中一个原因设计只改一项的对照。 [短距留出记录](artifacts/walk_stop_reach/20261123_b1_a1_a2/holdout/offset_short/metrics.json)

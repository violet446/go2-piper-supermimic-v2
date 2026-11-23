# Kine2Go 小样本检查

本地保存 MIMUW Robotics Lab 的 Kine2Go 数据集两个小样的 `clip.json`、`config.json` 和 `motion.npy`。采样自 [数据版本 7f03a3f](https://huggingface.co/datasets/MIMUW-Robotics/kine2go/tree/7f03a3fb7f6b2e684899530862ba103e81570363)。两条样本只用于本项目的格式与物理回放诊断。

- `data/solo8_walk/` 来自 Cassi 的 Solo8 数据，逐片标记 BSD-3-Clause。出处 C. Li et al., *Versatile Skill Control via Self-supervised Adversarial Imitation of Unlabeled Mixed Motions*, ICRA 2023。[上游项目](https://github.com/martius-lab/cassi) · [本地许可文本](LICENSES/cassi_BSD-3-Clause.txt)
- `data/vhdc_horse1_s1_walk_01/` 来自 Vienna Horse Data Collection，逐片标记 CC BY-SA 4.0。[上游项目](https://horse.cs.uni-bonn.de/vhdc-home.html) · [本地许可文本](LICENSES/VHDC_CC-BY-SA-4.0.txt)

数据集的分片许可见 [LICENSE](LICENSE)。采样频率、参考数组与动作顺序的差异、物理诊断和输入文件哈希见 [本地检查结果](../../../artifacts/walk_stop/20261120_kine2go_probe/metrics.json)。

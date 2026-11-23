"""B2: joint 18-axis behavior cloning from validated B1 rollouts."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from torch import nn

from reference import ROOT
from supermimic.robots.go2_piper import JOINTS, TIP

from experiments.arm_reach.experiment import AMOUNTS, ARM_HOME, SCALE_M, heading, load_policy as load_arm
from experiments.walk_stop.experiment import LOG, gpu_monitor, load_policy as load_walk, start_log, write_json
from experiments.walk_stop.rollout import STATE, features, target_anchor
from experiments.walk_stop.simulation import FPS, LEG_HOME, STOW, Scenario, Simulation
from .experiment import execute_case

B1_RUN = "20261123_b1_a1_a2"
A2_SOURCE_OFFSETS = ((.045, .025, 0.), (.05, .04, 0.), (.065, .05, 0.),
                     (.045, -.025, 0.), (.05, -.04, 0.), (.065, -.05, 0.))


class JointPolicy(nn.Module):
    def __init__(self, walking):
        super().__init__()
        self.trunk = nn.Sequential(*list(walking.children())[:-1])
        last = walking[-1]
        self.walk_head = nn.Linear(last.in_features, 18)
        self.reach_head = nn.Linear(last.in_features + 2, 18)
        with torch.no_grad():
            self.walk_head.weight.zero_()
            self.walk_head.bias.zero_()
            self.walk_head.weight[:12].copy_(last.weight)
            self.walk_head.bias[:12].copy_(last.bias)
            self.reach_head.weight.zero_()
            self.reach_head.bias.zero_()

    def forward(self, walk_observation, reach_observation, request):
        return self.walk_head(self.trunk(walk_observation)), self.arm_delta(reach_observation, request)

    def arm_delta(self, observation, request):
        return self.reach_head(torch.cat((self.trunk(observation), request), dim=-1))

    def walk_target(self, sim, plan, index):
        with torch.inference_mode():
            correction = self.walk_head(self.trunk(torch.from_numpy(features(sim, plan, index)))).numpy()
        return np.clip(np.r_[target_anchor(sim), STOW] + correction, *sim.model.jnt_range[1:].T)

    def reach_targets(self, sim, observation, goal):
        displacement = (goal - sim.data.body(TIP).xpos) @ heading(sim)
        request = torch.tensor(displacement[:2] / SCALE_M, dtype=torch.float32)
        with torch.inference_mode():
            delta = self.arm_delta(torch.from_numpy(observation), request).numpy()
        return np.clip(np.r_[LEG_HOME, ARM_HOME] + AMOUNTS[:, None] * delta, *sim.model.jnt_range[1:].T), displacement


def check_replay(sim, trace):
    mujoco.mj_setState(sim.model, sim.data, trace["a1_initial_state"], STATE)
    mujoco.mj_forward(sim.model, sim.data)
    for i, target in enumerate(trace["target"], 1):
        sim.advance(target[:12], target[12:])
        support, hit = sim.contacts()
        if not (np.array_equal(sim.data.qpos, trace["qpos"][i]) and np.array_equal(sim.data.qvel, trace["qvel"][i]) and
                hit == trace["nonfoot_hit"][i] and support.sum() == trace["contact_count"][i]):
            raise RuntimeError(f"Physical command replay diverged at tick {i}")


def collect_demos(directory, walk, arm, walk_config, arm_config, b1):
    folder = directory / "train"
    folder.mkdir(parents=True, exist_ok=False)
    sim = Simulation()
    examples = []
    manifest = {"source": "B1 modular A1+A2 policy on independent training scenarios", "checkpoints": b1["checkpoints"],
                "fps": FPS, "joint_names": JOINTS, "action": "18 absolute joint PD targets in rad",
                "model_license": "vendor/go2_piper/LICENSE", "train": [],
                "evaluation": [entry["name"] for entry in walk_config["holdout"]]}
    for i, distance in enumerate(walk_config["train_distances_m"]):
        for side in (0, 1):
            offset = A2_SOURCE_OFFSETS[(i + side) % 3 + 3 * side]
            scenario = Scenario(f"d{round(distance * 1000)}_{side}", distance)
            trace, report = execute_case(scenario, np.array(offset), sim, walk, arm, walk_config, arm_config, b1, collect=True)
            write_json(folder / f"{scenario.name}.json", report)
            if not report["passed"]:
                raise RuntimeError(f"B1 training demonstration {scenario.name} failed: {report['checks']}")
            check_replay(sim, trace)
            np.savez_compressed(folder / f"{scenario.name}.npz", **trace)
            manifest["train"].append({"name": scenario.name, "distance_m": distance, "offset_m": offset, "exact_physical_replay": True})
            examples.append(trace)
            LOG.info("%s passes whole-body gates and exact command replay", scenario.name)
    write_json(directory / "manifest.json", manifest)
    return examples


def train_joint(model, run, examples, config):
    device = config["device"]
    model.to(device)
    torch.manual_seed(config["seed"])
    tensors = [torch.as_tensor(array, dtype=torch.float32, device=device) for array in (
        np.concatenate([e["walk_observation"] for e in examples]), np.concatenate([e["walk_label"] for e in examples]),
        np.stack([e["reach_observation"] for e in examples]), np.stack([e["reach_request"][:2] for e in examples]) / SCALE_M,
        np.stack([e["reach_label"] for e in examples]), AMOUNTS[None, :, None],
    )]
    x, y, reach_observation, request, reach_label, phase = tensors
    optimizer = torch.optim.AdamW([
        {"params": [*model.trunk.parameters(), *model.walk_head.parameters()], "lr": 1e-5},
        {"params": model.reach_head.parameters(), "lr": .001},
    ], weight_decay=0.)
    started = time.monotonic()
    csv = run / "gpu.csv"
    with gpu_monitor(csv):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for update in range(1, config["updates"] + 1):
            selection = torch.randint(len(x), (512,), device=device)
            walk, reach = model(x[selection], reach_observation, request)
            walk_loss = (walk - y[selection]).square().mean()
            reach_loss = (reach[:, None, :] * phase - reach_label).square().mean()
            optimizer.zero_grad(set_to_none=True)
            ((walk_loss + reach_loss) / .01).backward()
            optimizer.step()
            if update == 1 or update % 500 == 0 or update == config["updates"]:
                LOG.info("update %s, walk / reach RMSE %.7f / %.7f rad", update,
                         float(walk_loss.detach().sqrt()), float(reach_loss.detach().sqrt()))
        with torch.inference_mode():
            walk, reach = model(x, reach_observation, request)
            walk_rmse = float((walk - y).square().mean().sqrt())
            reach_rmse = float((reach[:, None, :] * phase - reach_label).square().mean().sqrt())
    gpu = [int(row.split(",")[1]) for row in csv.read_text().splitlines() if len(row.split(",")) == 3] if csv.exists() else []
    write_json(run / "training_metrics.json", {"device": device, "clips": len(examples), "updates": config["updates"],
               "walk_frames": len(x), "reach_frames": int(len(examples) * len(AMOUNTS)), "walk_rmse_rad": walk_rmse,
               "reach_rmse_rad": reach_rmse, "wall_time_s": time.monotonic() - started,
               "gpu_peak_sample_mib": max(gpu) if gpu else None,
               "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.})
    torch.save({"weights": model.cpu().state_dict(), "config": config}, run / "policy.pt")


def evaluate(run, split, b1, walk_config, arm_config, joint):
    sim = Simulation()
    dest = run / split
    dest.mkdir(exist_ok=False)
    entries = [(walk_config["development"], b1["development_offset_m"])] if split == "development" else list(
        zip(walk_config["holdout"], b1["holdout_offsets_m"], strict=True))
    reports = []
    for entry, offset in entries:
        scenario = Scenario(**entry)
        trace, report = execute_case(scenario, np.array(offset), sim, None, None, walk_config, arm_config, b1, joint=joint)
        folder = dest / scenario.name
        folder.mkdir()
        np.savez_compressed(folder / "rollout.npz", **trace)
        write_json(folder / "metrics.json", report)
        reports.append(report)
        LOG.info("%s pass %s, failed gates %s, park %.2f mm, after-stop maximum %s m, reach %s m",
                 scenario.name, report["passed"], [k for k, ok in report["checks"].items() if not ok],
                 report["walk"]["park_error_m"] * 1000., report["max_park_error_after_stop_m"],
                 report["reach"]["reach_goal_error_m"] if report["reach"] else None)
    write_json(dest / "metrics.json", reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "development", "holdout"))
    parser.add_argument("--run", default="20261123_b2_joint_bc_seed0")
    parser.add_argument("--b1", default=B1_RUN)
    parser.add_argument("--dataset", default="20261123_joint_seed0")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = ROOT / "artifacts" / "walk_stop_reach"
    run = output / args.run
    if args.stage == "train":
        run.mkdir(parents=True, exist_ok=False)
        config = {"b1_run": args.b1, "dataset": args.dataset, "device": args.device, "seed": 0, "updates": 2500}
    else:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    start_log(run / f"{args.stage}.log")
    b1 = json.loads((output / config["b1_run"] / "config.json").read_text(encoding="utf-8"))
    for check in b1["checkpoints"].values():
        if hashlib.sha256((ROOT / check["path"]).read_bytes()).hexdigest() != check["sha256"]:
            raise ValueError("B1 checkpoint changed since its development run")
    walk, walk_config = load_walk(ROOT / b1["checkpoints"]["walk"]["path"])
    arm_path = ROOT / b1["checkpoints"]["arm"]["path"]
    arm_config = json.loads((arm_path.parent / "config.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    if args.stage == "train":
        dataset = ROOT / "data" / "walk_stop_reach" / args.dataset
        examples = collect_demos(dataset, walk, load_arm(arm_path), walk_config, arm_config, b1)
        config["b1_config"] = b1
        write_json(run / "config.json", config)
        train_joint(JointPolicy(walk), run, examples, config)
    else:
        state = torch.load(run / "policy.pt", map_location="cpu", weights_only=True)
        if state["config"] != config or b1 != config["b1_config"]:
            raise ValueError("Joint checkpoint and evaluation configuration differ")
        joint = JointPolicy(walk)
        joint.load_state_dict(state["weights"])
        evaluate(run, args.stage, b1, walk_config, arm_config, joint.eval())


if __name__ == "__main__":
    main()

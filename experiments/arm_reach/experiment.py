"""A2: imitate six arm targets while independent leg PD holds a stance."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate import measure, read_reference, render
from reference import ROOT, FPS, interpolate
from supermimic.robots.go2_piper import HOME, MODEL, TIP

from experiments.walk_stop.experiment import LOG, gpu_monitor, start_log, write_json
from experiments.walk_stop.simulation import LEG_HOME, Scenario, Simulation

ARM_HOME = np.array(list(HOME.values())[-6:])
AMOUNTS = np.array([interpolate(i / FPS) for i in range(150)])
SCALE_M = .05
SCALE_RAD = .1
HISTORY = ROOT / "artifacts" / "reach_lr_20261119"
GATE = {"direction_min": .70, "dwell_reference_max_m": .025, "tilt_max_deg": 5.0}


def heading(sim):
    cosine, sine = sim.data.xmat[sim.base, [0, 3]]
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., np.hypot(cosine, sine)]]) / np.hypot(cosine, sine)


def network():
    # Six training goals sample a small planar region. An affine map interpolates there.
    return torch.nn.Linear(2, 6)


def load_policy(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    if state["model_sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Arm checkpoint model does not match the current model")
    policy = network()
    policy.load_state_dict(state["weights"])
    return policy.eval()


def arm_targets(sim, policy, world_goal):
    displacement = (world_goal - sim.data.body(TIP).xpos) @ heading(sim)
    request = torch.tensor(displacement[:2] / SCALE_M, dtype=torch.float32)
    with torch.inference_mode():
        delta = SCALE_RAD * policy(request).numpy()
    targets = ARM_HOME + AMOUNTS[:, None] * delta
    return np.clip(targets, *sim.model.jnt_range[-6:].T), displacement


def arm_rollout(sim, targets, leg_target=LEG_HOME):
    """Start from the live state, and never reset during the reach."""
    poses, tips, contacts, ground, velocities, times = [], [], [], [], [], []
    for arm in targets:
        sim.advance(leg_target, arm)
        support, hit = sim.contacts()
        poses.append(sim.data.qpos.copy())
        tips.append(sim.data.body(TIP).xpos.copy())
        contacts.append(support.sum())
        ground.append(hit)
        velocities.append(sim.data.qvel.copy())
        times.append(sim.data.time)
    return dict(qpos=np.array(poses), tip=np.array(tips), support=np.array(contacts), nonfoot_hit=np.array(ground),
                qvel=np.array(velocities), arm_target=targets, time=np.array(times))


def assess(trace, reference_tip, goal, gate):
    metrics = measure(reference_tip, goal, trace["qpos"], trace["tip"], trace["support"])
    metrics["nonfoot_ground_samples"] = int(trace["nonfoot_hit"].sum())
    metrics["checks"] = {
        "complete": metrics["steps_completed"] == 150,
        "direction": metrics["direction_completion"] >= gate["direction_min"],
        "dwell_tracking": metrics["dwell_tip_tracking_error_m"] <= gate["dwell_reference_max_m"],
        "tilt": metrics["max_tilt_deg"] <= gate["tilt_max_deg"],
        "four_feet": metrics["full_support_fraction"] == 1.,
        "no_body_ground": metrics["nonfoot_ground_samples"] == 0,
    }
    metrics["passed"] = all(metrics["checks"].values())
    return metrics


def cases(run, split, config, policy=None):
    out = run / split
    out.mkdir(exist_ok=False)
    directory = ROOT / config[f"{split}_source"]
    sim = Simulation()
    reports = []
    samples = []
    for path in sorted(directory.glob("reach_*.npz")):
        folder = out / path.stem
        folder.mkdir()
        poses, tip, goal, target = read_reference(path)
        if len(target) != len(AMOUNTS) or not np.allclose(target[:, :12], LEG_HOME, atol=1e-7):
            raise ValueError(f"Incompatible source standing controller or timing: {path}")
        scenario = Scenario(path.stem, 0.)
        # Match reference.record's 160 physics-step pre-stand and pre-action frame convention.
        sim.reset(scenario, ARM_HOME, settle_steps=160)
        displacement = (goal - sim.data.body(TIP).xpos) @ heading(sim)
        teacher_trace = arm_rollout(sim, target[:, -6:])
        teacher = assess(teacher_trace, tip, goal, config["gate"])
        teacher["max_tip_replay_difference_m"] = float(np.linalg.norm(teacher_trace["tip"] - tip, axis=1).max())
        teacher["max_pose_replay_difference"] = float(np.abs(teacher_trace["qpos"] - poses).max())
        teacher["passed"] &= teacher["max_tip_replay_difference_m"] <= config["source_replay_max_m"]
        report = {"clip": path.stem, "source": str(path.relative_to(ROOT)), "offset_local_m": displacement.tolist(),
                  "goal_world_m": goal.tolist(), "demo_replay": teacher}
        if policy is not None:
            sim.reset(scenario, ARM_HOME, settle_steps=160)
            targets, _ = arm_targets(sim, policy, goal)
            actual = arm_rollout(sim, targets)
            report["arm_policy"] = assess(actual, tip, goal, config["gate"])
            np.savez_compressed(folder / "policy.npz", **actual)
            if split == "holdout":
                legacy = HISTORY / "holdout" / path.stem / "metrics.json"
                h = json.loads(legacy.read_text(encoding="utf-8"))
                report["historical_18_joint_calibrated_initial"] = h["calibrated_initial"]
                LOG.info("%s A2 vs historical dwell-to-goal %.4f / %.4f m", path.stem,
                         report["arm_policy"]["reach_goal_error_m"], h["calibrated_initial"]["reach_goal_error_m"])
        np.savez_compressed(folder / "demo_replay.npz", **teacher_trace)
        write_json(folder / "metrics.json", report)
        reports.append(report)
        if not teacher["passed"]:
            raise RuntimeError(f"Source demonstration failed revalidation in {path}: {teacher}")
        samples.append((displacement, target[:, -6:]))
        selected = report.get("arm_policy", teacher)
        LOG.info("%s/%s pass %s direction %.3f, dwell-to-goal %.4f m, return %.4f m", split, path.stem,
                 selected["passed"], selected["direction_completion"], selected["reach_goal_error_m"],
                 selected["return_residual_m"])
    if len(reports) != (6 if split == "train" else 2):
        raise RuntimeError(f"Unexpected number of clips in {directory}")
    write_json(out / "metrics.json", reports)
    return samples


def train(run, config, samples):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    policy = network()
    x = torch.tensor(np.array([offset[:2] for offset, _ in samples]) / SCALE_M, dtype=torch.float32)
    y = torch.tensor((np.array([target for _, target in samples]) - ARM_HOME) / SCALE_RAD, dtype=torch.float32)
    phase = torch.tensor(AMOUNTS[None, :, None], dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.025)
    started = time.monotonic()
    gpu_csv = run / "gpu.csv"
    with gpu_monitor(gpu_csv):
        for update in range(config["updates"]):
            loss = (policy(x)[:, None, :] * phase - y).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if update == 0 or (update + 1) % 300 == 0:
                LOG.info("Update %d, six-axis action RMSE %.6f rad", update + 1, float(loss.detach().sqrt()) * SCALE_RAD)
        # This CPU fit uses no CUDA allocation. Keep the whole-card background sample separately.
        time.sleep(.23)
    samples_gpu = [int(row.split(",")[1]) for row in gpu_csv.read_text().splitlines() if len(row.split(",")) == 3] if gpu_csv.exists() else []
    with torch.inference_mode():
        rmse = float((policy(x)[:, None, :] * phase - y).square().mean().sqrt()) * SCALE_RAD
    torch.save({"weights": policy.state_dict(), "config": config,
                "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()}, run / "policy.pt")
    write_json(run / "training_metrics.json", {"device": "cpu", "clips": len(samples), "transitions": int(y.shape[0]*y.shape[1]),
               "updates": config["updates"], "action_rmse_rad": rmse, "pytorch_training_cuda_allocated_mib": 0,
               "gpu_peak_sample_mib": max(samples_gpu) if samples_gpu else None, "gpu_samples": len(samples_gpu),
               "wall_time_s": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "holdout", "video"))
    parser.add_argument("--run", default="20261123_arm_bc_seed0")
    args = parser.parse_args()
    run = ROOT / "artifacts" / "arm_reach" / args.run
    if args.stage == "train":
        run.mkdir(parents=True, exist_ok=False)
        start_log(run / "train.log")
        config = {
            f"{split}_source": str((HISTORY / f"{split}_data" / "calibrated").relative_to(ROOT)) for split in ("train", "holdout")
        }
        config.update(gate=GATE, source_replay_max_m=.0001, seed=0, updates=1200, fps=FPS,
                      stand_leg_target_rad=LEG_HOME.tolist(), arm_home_target_rad=ARM_HOME.tolist())
        samples = cases(run, "train", config)
        write_json(run / "config.json", config)
        train(run, config, samples)
    elif args.stage == "holdout":
        start_log(run / "holdout.log")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        cases(run, "holdout", config, load_policy(run / "policy.pt"))
    else:
        for folder in sorted(path for path in (run / "holdout").iterdir() if path.is_dir()):
            with np.load(folder / "demo_replay.npz") as demo, np.load(folder / "policy.npz") as policy:
                render(folder / "comparison.mp4", demo["qpos"], policy["qpos"], "Six-joint arm policy")


if __name__ == "__main__":
    main()

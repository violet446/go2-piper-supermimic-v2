"""A1 data preparation, supervised training, and physical replay entry point."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess
import time

import mujoco
import numpy as np

from reference import ROOT
from supermimic.robots.go2_piper import JOINTS, MODEL

from .rollout import STATE, features, rollout, target_anchor
from .simulation import FPS, LEGS, SIM, STOW, Scenario, Simulation, Teacher, WalkPlan


LOG = logging.getLogger("walk_stop")
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DATA_ROOT = ROOT / "data" / "walk_stop"
RUN_ROOT = ROOT / "artifacts" / "walk_stop"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def start_log(path):
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[file_handler, logging.StreamHandler()])


def require_pass(result):
    if not result["completed"]:
        raise RuntimeError(f'{result["scenario"]} failed: {result["checks"]}')


def log_result(kind, report):
    LOG.info("%s %s, %.3f cm, %.3f deg, stop max %.5f m/s, fall %s, complete %s %s",
             kind, report["scenario"], 100 * report["park_error_m"], report["heading_error_deg"],
             report["stop_speed_max_mps"], report["fell"], report["completed"],
             [name for name, passed in report["checks"].items() if not passed])


def demo(sim, scenario, config, rng=None):
    sim.reset(scenario)
    plan = WalkPlan(sim, scenario)
    teacher = Teacher(plan)
    noise = np.zeros(12)
    taps = rng.normal(0, config["perturbation_std_rad"], (len(plan.commands), 12)) if rng else None

    def action(index):
        label = teacher.action(index)
        if taps is not None:
            # Smooth action perturbations expose the teacher's corrections to off-nominal states.
            noise[:] = .60 * noise + .80 * taps[index]
        return np.clip(label + noise, sim.limit[:, 0], sim.limit[:, 1]), label

    return rollout(sim, plan, scenario, action, config["thresholds"], collect=True)


def replay(sim, scenario, config, recorded):
    mujoco.mj_setState(sim.model, sim.data, recorded["initial_state"], STATE)
    mujoco.mj_forward(sim.model, sim.data)
    plan = WalkPlan(sim, scenario)
    out, report = rollout(sim, plan, scenario,
                          lambda i: (recorded["target"][i], None), config["thresholds"])
    for key in ("qpos", "qvel", "contacts"):
        if not np.array_equal(out[key], recorded[key]):
            raise RuntimeError(f"Stored targets did not reproduce {scenario.name}: {key}")
    report["exact_physics_replay"] = True
    return out, report


def save_clip(folder, simulation, scenario, config, rng=None):
    arrays, _ = demo(simulation, scenario, config, rng)
    _, report = replay(simulation, scenario, config, arrays)
    log_result("demo/replay", report)
    write_json(folder / (scenario.name + ".json"), report)
    if report["completed"]:
        np.savez_compressed(folder / (scenario.name + ".npz"), **arrays)
    return report


def prepare(data_dir, run_dir, config):
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir.mkdir(parents=True, exist_ok=False)
    start_log(run_dir / "prepare.log")
    write_json(run_dir / "requested_config.json", config)
    sim = Simulation()
    dev = Scenario(**config["development"])
    arrays, _ = demo(sim, dev, config)
    _, report = replay(sim, dev, config, arrays)
    write_json(run_dir / "development_demo.json", report)
    np.savez_compressed(run_dir / "development_demo.npz", **arrays)
    log_result("development demo/replay", report)
    require_pass(report)
    write_json(run_dir / "config.json", config)
    LOG.info("Accepted and froze parking/physics gates before training and before holdout generation")
    train_dir = data_dir / "train"
    train_dir.mkdir()
    rng = np.random.default_rng(config["seed"])
    manifest = {
        "source": "Local Go2+PiPER MuJoCo support-force/foot-IK teacher",
        "provenance": "Generated locally from the vendored model. Model terms: vendor/go2_piper/LICENSE.",
        "external_data_used": False,
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "joint_names": JOINTS, "action_joint_names": JOINTS[:12], "feet_order": LEGS,
        "physics_dt_s": SIM.timestep, "fps": FPS, "arm_target_rad": STOW.tolist(),
        "action": "Absolute leg PD position target in radians, applied after the recorded observation for four substeps",
        "label": "Unperturbed teacher target minus current leg q and KD/KP times current leg velocity",
        "train": [], "excluded_train": [], "holdout": [entry["name"] for entry in config["holdout"]],
    }
    for distance in config["train_distances_m"]:
        for index in range(1 + config["perturbed_per_distance"]):
            name = f"d{round(distance * 1000):03}_{index:02}"
            scenario = Scenario(name, distance)
            report = save_clip(train_dir, sim, scenario, config, rng if index else None)
            if index == 0:
                require_pass(report)
            manifest["train" if report["completed"] else "excluded_train"].append(asdict(scenario))
    write_json(data_dir / "manifest.json", manifest)
    write_json(data_dir / "config.json", config)
    LOG.info("Qualified training clips %s; excluded %s", len(manifest["train"]), len(manifest["excluded_train"]))


def network(torch, input_size, hidden_dim):
    nn = torch.nn
    return nn.Sequential(nn.Linear(input_size, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 12))


@contextmanager
def gpu_monitor(csv_path):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        yield
        return
    with csv_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([executable, "--query-gpu=timestamp,memory.used,memory.total",
                                    "--format=csv,noheader,nounits", "-lms", "200"], stdout=output)
        try:
            yield
        finally:
            process.terminate()
            process.wait(timeout=5)


def train(data_dir, run_dir, device):
    import torch

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config != json.loads((data_dir / "config.json").read_text(encoding="utf-8")):
        raise ValueError("Run and dataset configurations differ")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("The training model changed after demonstration validation")
    start_log(run_dir / "train.log")
    torch.manual_seed(config["seed"])
    clips = [data_dir / "train" / f'{entry["name"]}.npz' for entry in manifest["train"]]
    observations, actions = [], []
    for path in clips:
        with np.load(path, allow_pickle=False) as clip:
            observations.append(clip["observation"])
            actions.append(clip["label_residual_rad"])
    started = time.monotonic()
    use_cuda = torch.device(device).type == "cuda"
    LOG.info("%s clips, %s transitions, device %s", len(clips), sum(map(len, actions)), device)
    write_json(run_dir / "environment.json", {
        "system": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "mujoco": mujoco.__version__, "numpy": np.__version__, "device": device,
        "gpu": torch.cuda.get_device_name(device) if use_cuda else None,
    })
    csv = run_dir / "gpu.csv"
    with gpu_monitor(csv):
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        x = torch.as_tensor(np.concatenate(observations), device=device)
        y = torch.as_tensor(np.concatenate(actions), device=device)
        policy = network(torch, x.shape[1], config["hidden_dim"]).to(device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["updates"], eta_min=.00005)
        for update in range(1, config["updates"] + 1):
            indices = torch.randint(x.shape[0], (config["batch_size"],), device=device)
            loss = (policy(x[indices]) - y[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if update == 1 or update % 500 == 0 or update == config["updates"]:
                LOG.info("update %d, action RMSE %.6f rad", update, loss.detach().sqrt().item())
        with torch.inference_mode():
            train_rmse = (policy(x) - y).square().mean().sqrt().item()
        torch.save({"weights": policy.cpu().state_dict(), "input_size": x.shape[1], "hidden_dim": config["hidden_dim"],
                    "model_sha256": manifest["model_sha256"], "config": config}, run_dir / "policy.pt")
    samples = [int(line.split(",")[1]) for line in csv.read_text().splitlines() if len(line.split(",")) == 3] if csv.exists() else []
    summary = {
        "clips": len(clips), "transitions": x.shape[0], "updates": config["updates"],
        "training_action_rmse_rad": train_rmse, "wall_time_s": time.monotonic() - started,
        "pytorch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if use_cuda else None,
        "pytorch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if use_cuda else None,
        "gpu_first_sample_mib": samples[0] if samples else None,
        "gpu_peak_sample_mib": max(samples) if samples else None, "gpu_sample_count": len(samples),
    }
    write_json(run_dir / "training_metrics.json", summary)
    LOG.info("%s", summary)


def load_policy(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Checkpoint model must match the current model")
    policy = network(torch, checkpoint["input_size"], checkpoint["hidden_dim"])
    policy.load_state_dict(checkpoint["weights"])
    policy.eval()
    torch.set_num_threads(1)
    return policy, checkpoint["config"]


def policy_target(sim, plan, policy, index):
    import torch

    with torch.inference_mode():
        observation = torch.from_numpy(features(sim, plan, index))
        target = target_anchor(sim) + policy(observation).numpy()
    return np.clip(target, sim.limit[:, 0], sim.limit[:, 1])


def evaluate(data_dir, run_dir, split, video):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    start_log(run_dir / f"evaluate_{split}.log")
    policy, saved_config = load_policy(run_dir / "policy.pt")
    if saved_config != config:
        raise ValueError("Checkpoint and run config must match")
    sim = Simulation()
    reports = []
    entries = [config["development"]] if split == "development" else config["holdout"]
    target_dir = run_dir / split
    target_dir.mkdir(exist_ok=False)
    if split == "holdout":
        holdout_dir = data_dir / "holdout"
        holdout_dir.mkdir(exist_ok=False)
    for entry in entries:
        scenario = Scenario(**entry)
        folder = target_dir / scenario.name
        folder.mkdir()
        if split == "development":
            with np.load(run_dir / "development_demo.npz", allow_pickle=False) as clip:
                recorded = dict(clip)
        else:
            recorded, _ = demo(sim, scenario, config)
        demonstration, demo_report = replay(sim, scenario, config, recorded)
        log_result("demo/replay", demo_report)
        if split == "holdout":
            write_json(holdout_dir / f"{scenario.name}.json", demo_report)
            if demo_report["completed"]:
                np.savez_compressed(holdout_dir / f"{scenario.name}.npz", **recorded)

        # Start the learned controller from the same independently settled state.
        sim.reset(scenario)
        plan = WalkPlan(sim, scenario)

        def action(index):
            return policy_target(sim, plan, policy, index), None

        actual, policy_report = rollout(sim, plan, scenario, action, config["thresholds"])
        if not np.array_equal(actual["initial_state"], recorded["initial_state"]):
            raise RuntimeError(f"Policy and teacher have different initial states: {scenario.name}")
        log_result("policy", policy_report)
        combined = {"scenario": entry, "demo": demo_report, "policy": policy_report}
        reports.append(combined)
        write_json(folder / "metrics.json", combined)
        np.savez_compressed(folder / "policy.npz", **actual)
        if video:
            from .video import render

            render(folder / "comparison.mp4", sim, scenario, demonstration, actual)
    write_json(target_dir / "metrics.json", reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("external", "prepare", "train", "evaluate"))
    parser.add_argument("--run", default="20261120_bc_seed0")
    parser.add_argument("--dataset", default="20261120_crawl_seed0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    data_dir, run_dir = DATA_ROOT / args.dataset, RUN_ROOT / args.run
    if args.stage == "prepare":
        prepare(data_dir, run_dir, json.loads(args.config.read_text(encoding="utf-8")))
    elif args.stage == "train":
        train(data_dir, run_dir, args.device)
    elif args.stage == "evaluate":
        evaluate(data_dir, run_dir, args.split, not args.no_video)
    else:
        from .external import inspect_kine2go

        run_dir.mkdir(parents=True, exist_ok=False)
        start_log(run_dir / "external.log")
        write_json(run_dir / "metrics.json", inspect_kine2go())


if __name__ == "__main__":
    main()

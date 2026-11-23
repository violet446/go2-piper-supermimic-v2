"""Run A1 and A2 without resetting physics between walking and reaching."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import ROOT, DWELL_START, DWELL_END
from supermimic.robots.go2_piper import TIP

from experiments.arm_reach.experiment import AMOUNTS, ARM_HOME, arm_targets, assess, load_policy as load_arm_policy
from experiments.walk_stop.experiment import LOG, load_policy as load_walk_policy, policy_target, start_log, write_json
from experiments.walk_stop.rollout import features, integration_state, rollout, STAGES, target_anchor
from experiments.walk_stop.simulation import FPS, LEG_HOME, STOW, Scenario, Simulation, WalkPlan, smooth

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
PHASES = (*STAGES, "leg_handoff", "prepare_arm", "ready_hold", "pre_reach", "extend", "dwell", "retract", "restow", "stow_hold")


def execute_case(scenario, offset, sim, walk, arm, walk_config, arm_config, config, *, joint=None, collect=False):
    sim.reset(scenario)
    d = sim.data
    clock_zero = d.time
    plan = WalkPlan(sim, scenario)
    dock = plan.start[:2] + plan.axes[:, 0] * scenario.distance
    # Commit the spatial goal before any walking. Its XY is anchored to the nominal dock.
    waypoint = np.array(config["ready_tip_from_nominal_dock_local_m"]) + offset
    goal = np.r_[dock + plan.axes @ waypoint[:2], waypoint[2]]

    walking_observations = []
    walking_labels = []

    def walk_action(index):
        if joint is not None:
            return joint.walk_target(sim, plan, index), None
        target = policy_target(sim, plan, walk, index)
        if collect:
            walking_observations.append(features(sim, plan, index))
            walking_labels.append(np.r_[target - target_anchor(sim), np.zeros(6)])
        return target, None

    a1, a1_report = rollout(sim, plan, scenario, walk_action, walk_config["thresholds"])
    stopped_state = integration_state(sim)
    times = list(np.arange(len(a1["qpos"])) / FPS)
    poses, velocities = list(a1["qpos"]), list(a1["qvel"])
    contact, nonfoot = list(a1["contacts"].sum(axis=1)), list(a1["body_ground"])
    cursor = a1["schedule"][:len(a1["target"]), 0].astype(int)
    steps = [np.r_[q, STOW] if len(q) == 12 else q for q in a1["target"]]
    phase = ["ready", *(STAGES[index] for index in cursor)]
    phase_boundaries = {"walk_start_s": 0., "stop_confirmed_s": d.time - clock_zero if a1_report["completed"] else None}

    def record(name, legs, arm_target):
        sim.advance(legs, arm_target)
        support, hit = sim.contacts()
        poses.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        times.append(d.time - clock_zero)
        contact.append(int(support.sum()))
        nonfoot.append(hit)
        steps.append(np.r_[legs, arm_target])
        phase.append(name)

    def transition(name, duration, start, end):
        ticks = round(duration * FPS)
        for amount in np.linspace(0., 1., ticks):
            target = start + smooth(amount) * (end - start)
            record(name, target[:12], target[12:])
        phase_boundaries[f"{name}_end_s"] = d.time - clock_zero

    first_arm_masks = len(poses)
    reach_report = None
    reference_tip = np.empty((0, 3))
    reach_initial = np.empty(0)
    local_request = None
    reach_observation = None
    reach_labels = None
    reached_ready = False
    if a1_report["completed"]:
        # Read the last command and the live state from A1. Keep all integration state.
        transition("leg_handoff", config["leg_handoff_s"], steps[-1], np.r_[LEG_HOME, STOW])
        transition("prepare_arm", config["prepare_arm_s"], steps[-1], np.r_[LEG_HOME, ARM_HOME])
        stand = np.r_[LEG_HOME, ARM_HOME]
        transition("ready_hold", config["ready_hold_s"], stand, stand)
        final_window = round(FPS * walk_config["thresholds"]["stop_window_s"]) + 1
        ready_speed = np.linalg.norm(np.array(velocities[-final_window:])[:, :2], axis=1).max()
        ready_error = np.linalg.norm(d.qpos[:2] - dock)
        reached_ready = bool(ready_speed <= walk_config["thresholds"]["stop_speed_mps"] and
                             ready_error <= walk_config["thresholds"]["park_error_m"] and min(contact[-final_window:]) == 4 and
                             not any(nonfoot[first_arm_masks:]))
        if reached_ready:
            reach_initial = integration_state(sim)
            ready_tip = d.body(TIP).xpos.copy()
            ready_qpos = d.qpos.copy()
            reference_tip = ready_tip + AMOUNTS[:, None] * (goal - ready_tip)
            reach_observation = features(sim, plan, len(plan.commands) - 1)
            if joint is None:
                arm_sequence, local_request = arm_targets(sim, arm, goal)
                targets = np.column_stack((np.tile(LEG_HOME, (len(AMOUNTS), 1)), arm_sequence))
            else:
                targets, local_request = joint.reach_targets(sim, reach_observation, goal)
            if collect:
                reach_labels = targets - np.r_[LEG_HOME, ARM_HOME]
            phase_boundaries["reach_start_s"] = d.time - clock_zero
            begin = len(poses)
            tips = []
            for index, target in enumerate(targets):
                t = index / FPS
                label = "pre_reach" if t < .25 else "extend" if t < 1.10 else "dwell" if t < 1.75 else "retract"
                record(label, target[:12], target[12:])
                tips.append(d.body(TIP).xpos.copy())
            phase_boundaries["dwell_metric_start_s"] = times[begin + int(np.ceil(DWELL_START * FPS))]
            phase_boundaries["dwell_metric_end_s"] = times[begin + int(np.floor(DWELL_END * FPS))]
            phase_boundaries["reach_end_s"] = d.time - clock_zero
            trace = dict(qpos=np.array(poses[begin:]), tip=np.array(tips), support=np.array(contact[begin:]),
                         nonfoot_hit=np.array(nonfoot[begin:]))
            reach_report = assess(trace, reference_tip, goal, arm_config["gate"])
            reach_report["true_start_tip_w_m"] = ready_tip.tolist()
            reach_report["true_start_qpos"] = ready_qpos.tolist()
            reach_report["return_from_true_start_m"] = float(np.linalg.norm(trace["tip"][-1] - ready_tip))
            transition("restow", config["restow_s"], steps[-1], np.r_[LEG_HOME, STOW])
            stow = np.r_[LEG_HOME, STOW]
            transition("stow_hold", config["stow_hold_s"], stow, stow)
    qpos, qvel = np.array(poses), np.array(velocities)
    contact = np.array(contact)
    arm_region = qpos[first_arm_masks:]
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (qpos[:, 4]**2 + qpos[:, 5]**2), -1., 1.)))
    fell = bool(any(nonfoot) or qpos[:, 2].min() < walk_config["thresholds"]["fall_height_m"] or
                tilt.max() > walk_config["thresholds"]["fall_tilt_deg"] or not np.isfinite(qpos).all())
    final_park = float(np.linalg.norm(qpos[-1, :2] - dock))
    after_park = float(np.linalg.norm(arm_region[:, :2] - dock, axis=1).max()) if len(arm_region) else None
    stow_error = float(np.abs(qpos[-1, -6:] - STOW).max())
    controls = np.array(steps)
    stance_support = bool(len(arm_region) > 0 and np.all(contact[first_arm_masks:] == 4))
    checks = {
        "walk_and_stop": a1_report["completed"],
        "ready_and_stopped": reached_ready,
        "reach": bool(reach_report and reach_report["passed"] and reach_report["reach_goal_error_m"] <= config["reach_error_max_m"]),
        "no_fall": not fell, "four_feet_after_walk": stance_support,
        "stay_in_dock": after_park is not None and after_park <= walk_config["thresholds"]["park_error_m"],
        "final_stow": stow_error <= walk_config["thresholds"]["max_stow_error_rad"],
        "handoff_continuous": bool(len(arm_region) and np.array_equal(controls[first_arm_masks - 2], controls[first_arm_masks - 1])),
    }
    report = {"scenario": scenario.name, "dock_xy_m": dock.tolist(), "fixed_goal_world_m": goal.tolist(),
              "reach_request_local_m": local_request.tolist() if local_request is not None else None,
              "ticks": len(steps), "stages_s": phase_boundaries, "walk": a1_report, "reach": reach_report,
              "fell": fell, "final_park_error_m": final_park, "max_park_error_after_stop_m": after_park,
              "ready_park_error_m": float(ready_error) if len(arm_region) else None,
              "ready_max_stop_speed_mps": float(ready_speed) if len(arm_region) else None,
              "max_tilt_deg": float(tilt.max()), "final_stow_error_rad": stow_error,
              "leg_handoff_max_step_rad": float(np.abs(np.diff(controls[first_arm_masks-2:first_arm_masks-1+round(config["leg_handoff_s"]*FPS), :12], axis=0)).max()) if len(arm_region) else None,
              "checks": checks, "passed": all(checks.values())}
    trace = dict(qpos=qpos, qvel=qvel, time=np.array(times), phase=np.array(phase), target=controls,
                 contact_count=contact, nonfoot_hit=np.array(nonfoot), a1_initial_state=a1["initial_state"],
                 stopped_state=stopped_state, reach_initial_state=reach_initial, reference_tip_w=reference_tip,
                 start_xyz=a1["start_xyz"], dock_xy=dock, fixed_goal_w=goal)
    if collect and report["passed"]:
        trace.update(walk_observation=np.array(walking_observations), walk_label=np.array(walking_labels),
                     reach_observation=reach_observation, reach_request=local_request, reach_label=reach_labels)
    return trace, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "holdout", "video"))
    parser.add_argument("--run", default="20261123_b1_a1_a2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = ROOT / "artifacts" / "walk_stop_reach" / args.run
    if args.stage == "video":
        from .video import render

        for folder in sorted(path for path in (run / "holdout").iterdir() if path.is_dir()):
            with np.load(folder / "rollout.npz", allow_pickle=False) as trace:
                render(folder / "walk_stop_reach.mp4", trace)
        return
    if args.stage == "development":
        run.mkdir(parents=True, exist_ok=False)
        config = json.loads(args.config.read_text(encoding="utf-8"))
    else:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    start_log(run / f"{args.stage}.log")
    a1 = ROOT / "artifacts" / "walk_stop" / config["walk_run"] / "policy.pt"
    a2 = ROOT / "artifacts" / "arm_reach" / config["arm_run"] / "policy.pt"
    walk, walk_config = load_walk_policy(a1)
    arm = load_arm_policy(a2)
    arm_config = json.loads((a2.parent / "config.json").read_text(encoding="utf-8"))
    checkpoints = {name: {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in (("walk", a1), ("arm", a2))}
    if args.stage == "holdout" and checkpoints != config["checkpoints"]:
        raise ValueError("Holdout checkpoints differ from the development run")
    config["checkpoints"] = checkpoints
    cases = [(walk_config["development"], config["development_offset_m"])] if args.stage == "development" else list(
        zip(walk_config["holdout"], config["holdout_offsets_m"], strict=True))
    dest = run / args.stage
    dest.mkdir(exist_ok=False)
    reports = []
    sim = Simulation()
    for entry, offset in cases:
        scenario = Scenario(**entry)
        trace, report = execute_case(scenario, np.array(offset), sim, walk, arm, walk_config, arm_config, config)
        folder = dest / scenario.name
        folder.mkdir()
        write_json(folder / "metrics.json", report)
        np.savez_compressed(folder / "rollout.npz", **trace)
        reports.append(report)
        LOG.info("%s pass %s, failed gates %s, park %.2f mm, after-stop maximum %s m, reach %s m",
                 scenario.name, report["passed"], [k for k, ok in report["checks"].items() if not ok],
                 report["walk"]["park_error_m"] * 1000., report["max_park_error_after_stop_m"],
                 report["reach"]["reach_goal_error_m"] if report["reach"] else None)
    write_json(dest / "metrics.json", reports)
    if args.stage == "development":
        write_json(run / "requested_config.json", config)
        if not all(report["passed"] for report in reports):
            raise RuntimeError("Development rollout did not pass the planned gates")
        write_json(run / "config.json", config)


if __name__ == "__main__":
    main()

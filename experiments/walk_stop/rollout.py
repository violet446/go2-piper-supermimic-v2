"""Measure live physics and replay stored joint targets without a controller."""

import mujoco
import numpy as np

from .simulation import DT, FPS, KD, KP, LEG_HOME, STOW, active_swing


STATE = mujoco.mjtState.mjSTATE_INTEGRATION
STAGES = ("ready", "start", "walk_1", "walk_2", "brake", "settle")


def integration_state(sim):
    state = np.empty(mujoco.mj_stateSize(sim.model, STATE))
    mujoco.mj_getState(sim.model, sim.data, state, STATE)
    return state


def features(sim, plan, index):
    d = sim.data
    body, feet, phase = plan.commands[index]
    rotation = d.xmat[sim.base].reshape(3, 3)
    stance = np.ones(4)
    swing = active_swing(phase)
    if swing >= 0:
        stance[swing] = 0.
    orientation = .5 * np.cross(rotation.T, plan.rotation.T).sum(axis=0) @ plan.rotation
    centers = d.geom_xpos[sim.feet]
    # Path references are in meters. The feature scales keep inputs near one.
    return np.r_[
        22 * ((body - d.qpos[:3]) @ plan.rotation), 6 * orientation,
        10 * (d.qvel[:3] @ plan.rotation), 2 * d.qvel[3:6],
        2 * (d.qpos[7:19] - LEG_HOME), .5 * d.qvel[6:18],
        5 * ((centers - d.qpos[:3]) @ rotation).flatten(),
        14 * ((feet - centers) @ rotation).flatten(),
        stance, phase[0] == 0,
    ].astype(np.float32)


def target_anchor(sim):
    return sim.data.qpos[7:19] + (KD[:12] / KP[:12]) * sim.data.qvel[6:18]


def rollout(sim, plan, scenario, choose_action, thresholds, *, collect=False):
    d = sim.data
    out = {key: [] for key in (
        "qpos", "qvel", "contacts", "body_ground", "foot_centers_w", "foot_velocity_w",
    )}
    controls, observations, labels = [], [], []
    initial = integration_state(sim)
    start = plan.start.copy()
    direction = plan.axes[:, 0]
    goal = start[:2] + scenario.distance * direction

    def snapshot():
        contacts, hit = sim.contacts()
        state = (d.qpos, d.qvel, contacts, hit, d.geom_xpos[sim.feet], sim.foot_velocity())
        for key, value in zip(out, state):
            out[key].append(np.array(value))
        return hit or d.qpos[2] < thresholds["fall_height_m"] or (
            d.xmat[sim.base, 8] < np.cos(np.deg2rad(thresholds["fall_tilt_deg"]))
        ) or not np.isfinite(d.qpos).all()

    fell = snapshot()
    for index in range(len(plan.commands)):
        if fell:
            break
        target, label = choose_action(index)
        if collect:
            observations.append(features(sim, plan, index))
            labels.append((label - target_anchor(sim)).astype(np.float32))
        controls.append(target)
        if len(target) == 12:
            sim.advance(target)
        else:
            sim.advance(target[:12], target[12:])
        fell = snapshot()
    out = {key: np.array(values) for key, values in out.items()}
    out.update(
        target=np.asarray(controls), schedule=np.array([command[2] for command in plan.commands]),
        initial_state=initial, start_xyz=start, goal_xy=goal,
    )
    if collect:
        out.update(observation=np.array(observations), label_residual_rad=np.array(labels))
    return out, measure(out, scenario, plan.axes, thresholds, fell)


def measure(out, scenario, axes, gate, fell):
    qpos, qvel = out["qpos"], out["qvel"]
    contacts, feet = out["contacts"], out["foot_centers_w"]
    path = (qpos[:, :2] - out["start_xyz"][:2]) @ axes
    speed = np.linalg.norm(qvel[:, :2], axis=1)
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (qpos[:, 4]**2 + qpos[:, 5]**2), -1., 1.)))
    w, x, y, z = qpos[-1, 3:7]
    yaw = np.rad2deg(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
    heading_error = abs((yaw - scenario.yaw_deg + 180.) % 360. - 180.)
    foot_speed = np.linalg.norm(out["foot_velocity_w"][..., :2], axis=2)
    on_ground = foot_speed[contacts]
    schedule = out["schedule"][:len(qpos)-1]
    steps = []
    for stage in range(1, 5):
        for leg in range(4):
            indices = np.flatnonzero((schedule[:, 0] == stage) & (schedule[:, 1] == leg) & (schedule[:, 2] == 1))
            if indices.size:
                k = indices + 1
                steps.append({
                    "stage": STAGES[stage], "leg_index": leg,
                    "peak_clearance_m": float(max(feet[k, leg, 2] - feet[indices[0], leg, 2])),
                    "swing_airborne_samples": int(np.count_nonzero(~contacts[k, leg])),
                    "peak_landing_speed_mps": float(foot_speed[k, leg][contacts[k, leg]].max(initial=0.)),
                })
    phases = {}
    for phase, name in enumerate(STAGES):
        indices = np.flatnonzero(schedule[:, 0] == phase)
        if indices.size:
            dx = path[indices[-1] + 1, 0] - path[indices[0], 0]
            phases[name] = {"progress_m": float(dx), "avg_forward_speed_mps": float(dx / (indices.size * DT))}
    stop_frames = round(FPS * gate["stop_window_s"]) + 1
    result = {
        "scenario": scenario.name, "distance_m": scenario.distance, "start_xyz_m": out["start_xyz"].tolist(),
        "goal_xy_m": out["goal_xy"].tolist(), "final_xy_m": qpos[-1, :2].tolist(),
        "ticks": len(qpos) - 1, "expected_ticks": len(out["schedule"]),
        "park_error_m": float(np.linalg.norm(qpos[-1, :2] - out["goal_xy"])),
        "heading_error_deg": float(heading_error),
        "stop_speed_max_mps": float(speed[-stop_frames:].max()),
        "stop_speed_mean_mps": float(speed[-stop_frames:].mean()),
        "final_speed_mps": float(speed[-1]),
        "max_lateral_m": float(np.abs(path[:, 1]).max()),
        "max_tilt_deg": float(tilt.max()), "min_base_height_m": float(qpos[:, 2].min()),
        "max_stow_error_rad": float(np.abs(qpos[:, -6:] - STOW).max()),
        "contact_slip_p95_mps": float(np.percentile(on_ground, 95)) if on_ground.size else None,
        "non_foot_ground_samples": int(out["body_ground"].sum()),
        "airborne_samples": int(np.count_nonzero(~contacts.any(axis=1))),
        "min_support_feet": int(contacts.sum(axis=1).min()),
        "four_feet_during_stop": bool(len(qpos) >= stop_frames and contacts[-stop_frames:].all()),
        "min_peak_swing_clearance_m": min((step["peak_clearance_m"] for step in steps), default=0.),
        "max_landing_speed_mps": max((step["peak_landing_speed_mps"] for step in steps), default=0.),
        "fell": bool(fell), "phases": phases, "swings": steps,
    }
    cruise = [phases.get(name, {}).get("avg_forward_speed_mps", 0.) for name in ("walk_1", "walk_2")]
    result["checks"] = {
        "full_rollout": result["ticks"] == result["expected_ticks"],
        "park_position": result["park_error_m"] <= gate["park_error_m"],
        "heading": heading_error <= gate["heading_error_deg"],
        "stop_speed": result["stop_speed_max_mps"] <= gate["stop_speed_mps"],
        "support": result["four_feet_during_stop"] and result["airborne_samples"] == 0,
        "stable": not fell and result["max_tilt_deg"] <= gate["max_tilt_deg"],
        "straight": result["max_lateral_m"] <= gate["max_lateral_m"],
        "stowed": result["max_stow_error_rad"] <= gate["max_stow_error_rad"],
        "slip": on_ground.size > 0 and result["contact_slip_p95_mps"] <= gate["max_contact_slip_p95_mps"],
        "start_walk_brake": all(0 < phases.get(name, {}).get("avg_forward_speed_mps", 0.) < min(cruise)
                                for name in ("start", "brake")),
        "foot_lift": len(steps) == 16 and result["min_peak_swing_clearance_m"] >= gate["min_swing_clearance_m"],
    }
    result["checks"] = {name: bool(passed) for name, passed in result["checks"].items()}
    result["completed"] = all(result["checks"].values())
    return result

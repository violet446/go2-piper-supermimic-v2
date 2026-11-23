"""Audit two Kine2Go kinematic references on the complete A1 model."""

import hashlib
import json
import logging
from itertools import permutations

import mujoco
import numpy as np

from reference import ROOT
from supermimic.robots.go2_piper import LEG_JOINTS

from .simulation import FPS, LEGS, STOW, Simulation


SOURCE = ROOT / "data" / "external" / "kine2go" / "data"
CLIPS = ("solo8_walk", "vhdc_horse1_s1_walk_01")
LOG = logging.getLogger("walk_stop")


def fk_error(sim, motion, columns):
    d = sim.data
    ids = [sim.model.body(f"{leg}_foot").id for leg in LEGS]
    reference = motion[:, 36:48].reshape(-1, 4, 3)[:, [0, 2, 1, 3]]
    feet = []
    for frame in motion:
        d.qpos[:7] = frame[48:55]
        d.qpos[7:19] = frame[columns]
        mujoco.mj_kinematics(sim.model, d)
        feet.append(d.xpos[ids].copy())
    return float(np.linalg.norm(np.array(feet) - reference, axis=2).mean())


def physical_replay(sim, motion, columns, fps):
    d, model = sim.data, sim.model
    time = np.arange(len(motion)) / fps
    control_time = np.arange(int(time[-1] * FPS) + 1) / FPS
    targets = np.column_stack([np.interp(control_time, time, motion[:, j]) for j in columns])
    mujoco.mj_resetData(model, d)
    d.qpos[:7] = motion[0, 48:55]
    d.qpos[7:19] = targets[0]
    d.qpos[19:] = STOW
    mujoco.mj_forward(model, d)
    # Use the source pose and clear any initial floor penetration. All DoFs then step freely.
    correction = max(0., float(np.max(model.geom_size[sim.feet, 0] - d.geom_xpos[sim.feet, 2])))
    d.qpos[2] += correction
    mujoco.mj_forward(model, d)
    origin = d.qpos[:2].copy()
    rotation = d.xmat[sim.base].reshape(3, 3)
    forward = rotation[:2, 0] / np.linalg.norm(rotation[:2, 0])
    max_tilt, body_hit, fell = 0., False, False
    airborne, speeds = 0, []
    for tick, target in enumerate(targets, 1):
        sim.advance(np.clip(target, sim.limit[:, 0], sim.limit[:, 1]))
        contacts, body_hit = sim.contacts()
        airborne += not contacts.any()
        speeds.append(float(np.linalg.norm(d.qvel[:2])))
        tilt = float(np.rad2deg(np.arccos(np.clip(d.xmat[sim.base, 8], -1., 1.))))
        max_tilt = max(max_tilt, tilt)
        fell = body_hit or tilt > 45. or d.qpos[2] < .17
        if fell:
            break
    return {
        "ticks": tick, "expected_ticks": len(targets), "fallen": bool(fell), "body_ground": bool(body_hit),
        "airborne_samples": airborne, "max_tilt_deg": max_tilt, "initial_floor_correction_m": correction,
        "forward_travel_m": float((d.qpos[:2] - origin) @ forward), "end_speed_mps": speeds[-1],
        "last_0p5s_max_speed_mps": max(speeds[-(FPS // 2 + 1):]),
    }


def inspect_kine2go():
    sim = Simulation()
    results = []
    for clip in CLIPS:
        folder = SOURCE / clip
        meta = json.loads((folder / "clip.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        env = config["env_cfg"]
        path = folder / "motion.npy"
        motion = np.load(path, allow_pickle=False)
        assert motion.ndim == 2 and motion.shape[1] == 61 and np.isfinite(motion).all()
        advertised = [6 + env["dof_names"].index(name) for name in LEG_JOINTS]
        groups = np.arange(6, 18).reshape(3, 4).T
        candidates = [groups[list(order)].flatten().tolist() for order in permutations(range(4))]
        errors = sorted((fk_error(sim, motion[::5], columns), columns) for columns in candidates)
        verified_error, verified = errors[0]
        if verified_error > .001:
            raise ValueError(f"None of the tested column orders reproduce the feet in {clip}")
        fps = env["control_freq"]
        velocity = np.linalg.norm(np.diff(motion[:, 48:50], axis=0), axis=1) * fps
        result = {
            "clip": clip, "source_license": meta["license"], "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "motion_shape": list(motion.shape), "metadata_frames": meta["n_frames"], "control_hz": fps,
            "motion_range_in_policy_config": config["policy_cfg"].get("motion_range"),
            "source_motion_start_to_end_m": (motion[-1, 48:51] - motion[0, 48:51]).tolist(),
            "speed_from_position_first_0p5s_mean_mps": float(velocity[:fps // 2].mean()),
            "speed_from_position_last_0p5s_mean_mps": float(velocity[-fps // 2:].mean()),
            "config_action_joint_names": env["dof_names"], "action_scale": env["action_scale"],
            "action_latency_cfg_s": env["action_latency"], "control_decimation": env["decimation"],
            "kp": env["PD_stiffness"], "kd": env["PD_damping"],
            "model_leg_joint_names": LEG_JOINTS, "advertised_motion_columns": advertised,
            "advertised_fk_mean_error_m": fk_error(sim, motion[::5], advertised),
            "verified_motion_columns": verified, "verified_fk_mean_error_m": verified_error,
            "physics_position_target_probe": physical_replay(sim, motion, verified, fps),
        }
        LOG.info("%s %s, FK error %.6g m; %s", clip, motion.shape, verified_error, result["physics_position_target_probe"])
        results.append(result)
    return results

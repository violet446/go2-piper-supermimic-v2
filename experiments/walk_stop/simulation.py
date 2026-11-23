"""Free-base MuJoCo plant and a slow, single-swing walk/stop teacher."""

from dataclasses import dataclass

import mujoco
import numpy as np

from reference import FPS, floor_model
from supermimic.robots.go2_piper import (
    ARM_KD, ARM_KP, BASE, EFFORT, HOME, JOINTS, LEGS, LEG_KD, LEG_KP, SIM,
)

STOW = np.array([0., .25, -.15, 0., 0., 0.])
LEG_HOME = np.array(list(HOME.values())[:12])
KP = np.repeat([LEG_KP, ARM_KP], [12, 6])
KD = np.repeat([LEG_KD, ARM_KD], [12, 6])
DT = SIM.control_dt
SHIFT = 23
SWING = 22
WAIT = 20
FEET_ORDER = (2, 0, 3, 1)
FRACTIONS = (.15, .35, .35, .15)


@dataclass(frozen=True)
class Scenario:
    name: str
    distance: float
    start_x: float = 0.
    start_y: float = 0.
    yaw_deg: float = 0.


def smooth(u):
    return u**3 * (10 + u * (-15 + 6 * u))


class Simulation:
    def __init__(self):
        self.model = floor_model()
        assert self.model.nq == 25 and self.model.nv == 24
        assert self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert tuple(self.model.joint(i).name for i in range(1, self.model.njnt)) == JOINTS
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body(BASE).id
        self.feet = np.array([self.model.geom(leg).id for leg in LEGS])
        self.hips = np.array([self.model.body(f"{leg}_hip").pos for leg in LEGS])
        self.floor = self.model.geom("floor").id
        self.limit = self.model.jnt_range[1:13]
        self.jac = np.zeros((3, self.model.nv))

    def torque_step(self, leg_target, arm_target=STOW):
        d = self.data
        target = np.r_[leg_target, arm_target]
        d.qfrc_applied[6:] = np.clip(KP * (target - d.qpos[7:]) - KD * d.qvel[6:], -np.array(EFFORT), EFFORT)
        mujoco.mj_step(self.model, d)

    def reset(self, scenario, arm_target=STOW, settle_steps=240):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:2] = scenario.start_x, scenario.start_y
        angle = np.deg2rad(scenario.yaw_deg)
        d.qpos[3:7] = np.cos(angle / 2), 0, 0, np.sin(angle / 2)
        d.qpos[7:] = np.r_[LEG_HOME, arm_target]
        mujoco.mj_forward(m, d)
        d.qpos[2] += .025 - d.geom_xpos[self.feet, 2].min()
        for _ in range(settle_steps):
            self.torque_step(LEG_HOME, arm_target)
        mujoco.mj_forward(m, d)

    def advance(self, goal, arm_target=STOW):
        for _ in range(SIM.decimation):
            self.torque_step(goal, arm_target)
        mujoco.mj_forward(self.model, self.data)

    def contacts(self):
        touching = set()
        body_ground = False
        for c in self.data.contact:
            a, b = c.geom
            if self.floor in (a, b):
                geom = b if a == self.floor else a
                touching.add(geom)
                body_ground |= geom not in self.feet
        return np.isin(self.feet, list(touching)), body_ground

    def foot_velocity(self):
        speeds = []
        for geom in self.feet:
            mujoco.mj_jac(self.model, self.data, self.jac, None, self.data.geom_xpos[geom], self.model.geom_bodyid[geom])
            speeds.append(self.jac @ self.data.qvel)
        return np.array(speeds)

    def leg_ik(self, centers):
        # The contact spheres sit 2 mm forward of the lower-link axes.
        x, y, z = (centers - self.hips).T
        offset = np.array([.0955, -.0955, .0955, -.0955])
        z_sagittal = -np.sqrt(np.maximum(y*y + z*z - offset*offset, 1e-8))
        abduction = np.arctan2(z, y) - np.arctan2(z_sagittal, offset)
        lower = np.hypot(.213, .002)
        knee = -np.arccos(np.clip((x*x + z_sagittal*z_sagittal - .213**2 - lower**2) / (2 * .213 * lower), -1., 1.))
        thigh = np.arctan2(-x, -z_sagittal) - np.arctan2(lower * np.sin(knee), .213 + lower * np.cos(knee))
        q = np.column_stack((abduction, thigh, knee - np.arctan2(.002, .213))).flatten()
        return np.clip(q, self.limit[:, 0], self.limit[:, 1])


def active_swing(phase):
    _, leg, section, u = phase
    return leg if section and leg >= 0 and u < .96 else -1


class WalkPlan:
    """Plan foot centers and a chassis path in the reset heading frame."""
    def __init__(self, sim, scenario):
        self.sim = sim
        d = sim.data
        yaw = np.deg2rad(scenario.yaw_deg)
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        self.axes = np.array([forward, [-forward[1], forward[0]]]).T
        self.start = d.qpos[:3].copy()
        self.com = (d.subtree_com[sim.base, :2] - self.start[:2]) @ self.axes
        self.rotation = np.eye(3)
        self.rotation[:2, :2] = self.axes
        self.commands = []
        self.plan(scenario.distance)

    def plan(self, distance):
        feet = self.sim.data.geom_xpos[self.sim.feet].copy()
        feet[:, :2] = (feet[:, :2] - self.start[:2]) @ self.axes
        origin = np.zeros(2)

        def append(at, points, stage, leg, section, u):
            base = np.r_[self.start[:2] + self.axes @ at, self.start[2]]
            world = points.copy()
            world[:, :2] = self.start[:2] + points[:, :2] @ self.axes.T
            self.commands.append((base, world, (stage, leg, section, u)))

        for _ in range(WAIT):
            append(origin, feet, 0, -1, 0, 0.)
        travel = 0.
        for stage, fraction in enumerate(FRACTIONS, 1):
            stride = fraction * distance
            for slot, leg in enumerate(FEET_ORDER):
                diagonal = (3, 2, 1, 0)[leg]
                edge = np.delete(np.arange(4), [leg, diagonal])
                tangent = feet[edge[1], :2] - feet[edge[0], :2]
                inward = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
                inward *= np.sign((feet[diagonal, :2] - feet[edge[0], :2]) @ inward)
                center = np.array([travel + stride * (slot + .5) / 4, 0.])
                margin = (center + self.com - feet[edge[0], :2]) @ inward
                destination = center + inward * max(0., .043 - margin)
                for index in range(SHIFT):
                    u = (index + 1) / SHIFT
                    append(origin + smooth(u) * (destination - origin), feet, stage, leg, 0, u)
                origin = destination
                moved = feet.copy()
                for index in range(SWING):
                    u = (index + 1) / SWING
                    moved[leg, 0] = feet[leg, 0] + stride * smooth(u)
                    moved[leg, 2] = feet[leg, 2] + .053 * np.sin(np.pi * u)**2
                    append(origin, moved, stage, leg, 1, u)
                feet = moved
            travel += stride

        tail = np.array([distance, 0.])
        for index in range(SHIFT):
            u = (index + 1) / SHIFT
            append(origin + smooth(u) * (tail - origin), feet, 5, -1, 0, u)
        for _ in range(FPS):
            append(tail, feet, 5, -1, 1, 1.)

class Teacher:
    def __init__(self, plan):
        self.plan = plan
        self.mass = plan.sim.model.body_subtreemass[plan.sim.base]

    def action(self, index):
        plan = self.plan
        sim, d = plan.sim, plan.sim.data
        base, feet, phase = plan.commands[index]
        if phase[0] == 0:
            return LEG_HOME
        swing = active_swing(phase)
        stance = np.flatnonzero(np.arange(4) != swing)
        rotation = d.xmat[sim.base].reshape(3, 3)
        orientation_error = .5 * np.cross(rotation.T, plan.rotation.T).sum(axis=0)
        angular_velocity = rotation @ d.qvel[3:6]
        acceleration = np.array([75., 90., 160.]) * (base - d.qpos[:3]) - np.array([12., 13., 17.]) * d.qvel[:3]
        force = self.mass * (acceleration + [0, 0, 9.81])
        moment = 155. * orientation_error - 16. * angular_velocity
        positions = d.geom_xpos[sim.feet]
        arms = positions[stance] - d.subtree_com[sim.base]
        # Map world-frame ground forces to total force and torque about the CoM.
        mapping = np.vstack((np.tile(np.eye(3), len(stance)), np.cross(arms[:, None, :], np.eye(3)).reshape(-1, 3).T))
        wrench = np.r_[force, moment]
        seed = np.tile([0., 0., max(0., force[2]) / len(stance)], len(stance))
        ground = (seed + mapping.T @ np.linalg.solve(mapping @ mapping.T + .003 * np.eye(6), wrench - mapping @ seed)).reshape(-1, 3)
        ground[:, 2] = np.maximum(ground[:, 2], 0.)
        ground[:, :2] = np.clip(ground[:, :2], -.5 * ground[:, 2, None], .5 * ground[:, 2, None])
        q = d.qpos[7:19]
        vel = d.qvel[6:18]
        target = q + (d.qfrc_bias[6:18] + LEG_KD * vel) / LEG_KP
        for foot, reaction in zip(stance, ground):
            geom = sim.feet[foot]
            mujoco.mj_jac(sim.model, d, sim.jac, None, positions[foot], sim.model.geom_bodyid[geom])
            joints = slice(3 * foot, 3 * foot + 3)
            target[joints] -= sim.jac[:, 6 + joints.start:6 + joints.stop].T @ reaction / LEG_KP
        if swing >= 0:
            joints = slice(3 * swing, 3 * swing + 3)
            target[joints] = sim.leg_ik((feet - d.qpos[:3]) @ rotation)[joints] + d.qfrc_bias[6:18][joints] / LEG_KP
        return np.clip(target, sim.limit[:, 0], sim.limit[:, 1])

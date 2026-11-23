"""Render a continuous rollout with its fixed dock and fixed spatial goal."""

import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from reference import FPS, floor_model
from supermimic.robots.go2_piper import TIP


def render(path, trace):
    model = floor_model()
    data = mujoco.MjData(model)
    start, dock, goal = trace["start_xyz"], trace["dock_xy"], trace["fixed_goal_w"]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (*((start[:2] + dock) / 2), .47)
    camera.azimuth = 130 + np.rad2deg(np.arctan2(*(dock - start[:2])[::-1]))
    camera.elevation, camera.distance = -15, 1.90
    width, height = 640, 400
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(FPS // 2), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    qpos, qvel, times, phase = (trace[key] for key in ("qpos", "qvel", "time", "phase"))
    indices = list(range(0, len(qpos), 2))
    if indices[-1] < len(qpos) - 1:
        indices.append(len(qpos) - 1)
    with mujoco.Renderer(model, height, width) as renderer, subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in indices:
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            scene = renderer.scene
            for point, kind, size, color in (
                (np.r_[dock, .002], mujoco.mjtGeom.mjGEOM_CYLINDER, [.044, .002, 0], [.17, .78, .60, .94]),
                (goal, mujoco.mjtGeom.mjGEOM_SPHERE, [.014, 0, 0], [1., .55, .13, .72]),
            ):
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.array(size), point,
                                    np.eye(3).flatten(), np.array(color, dtype=np.float32))
                scene.ngeom += 1
            panel = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, width, 32), fill="#172331")
            draw.text((14, 9), f"{times[i]:05.2f} s   |   {phase[i].replace('_', ' ')}", fill="#f8fbff")
            draw.rectangle((0, height - 28, width, height), fill="#172331")
            park = 100 * np.linalg.norm(qpos[i, :2] - dock)
            speed = np.linalg.norm(qvel[i, :2])
            reach = 100 * np.linalg.norm(data.body(TIP).xpos - goal)
            draw.text((14, height - 21), f"Dock {park:.1f} cm   |   Speed {speed:.3f} m/s   |   Hand to target {reach:.1f} cm", fill="#f8fbff")
            encoder.stdin.write(np.asarray(panel).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed to write {path}")

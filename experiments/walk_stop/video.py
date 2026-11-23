"""Render the same world view of the physically simulated expert and policy."""

import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .rollout import STAGES
from .simulation import FPS


def render(path, sim, scenario, expert, policy):
    model, data = sim.model, mujoco.MjData(sim.model)
    start, goal = expert["start_xyz"][:2], expert["goal_xy"]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (*((start + goal) * .5), .27)
    camera.azimuth, camera.elevation, camera.distance = 128 + scenario.yaw_deg, -21, 1.48
    width, height = 480, 360
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width * 2}x{height}", "-r", str(FPS // 2), "-i", "-",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    count = max(len(expert["qpos"]), len(policy["qpos"]))
    # Include the last physics sample even when it falls between video ticks.
    indices = list(range(0, count, 2))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    with mujoco.Renderer(model, height, width) as renderer, subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for tick in indices:
            panels = []
            for label, trace in (("Demonstration", expert), ("Learned policy", policy)):
                stopped = tick >= len(trace["qpos"])
                i = min(tick, len(trace["qpos"]) - 1)
                data.qpos[:] = trace["qpos"][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                scene = renderer.scene
                for point, color, radius in ((start, [.95, .77, .29, .92], .023), (goal, [.2, .85, .6, .94], .036)):
                    marker = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, .002, 0.]),
                                        np.array([*point, .002]), np.eye(3).flatten(), np.array(color, dtype=np.float32))
                    scene.ngeom += 1
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                stage = STAGES[int(trace["schedule"][max(0, i - 1), 0])]
                heading = f"{label}   {i / FPS:05.2f} s   {stage if not stopped else 'ended'}"
                speed = np.linalg.norm(trace["qvel"][i, :2])
                remaining = np.linalg.norm(data.qpos[:2] - goal)
                draw.rectangle((0, 0, width, 32), fill="#172331")
                draw.text((12, 9), heading, fill="#f8fbff")
                draw.rectangle((0, height - 26, width, height), fill="#172331")
                draw.text((12, height - 20), f"Distance to goal {100 * remaining:5.1f} cm   |   Speed {speed:.3f} m/s",
                          fill="#f8fbff")
                panels.append(np.asarray(panel))
            encoder.stdin.write(np.concatenate(panels, axis=1).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed to write {path}")

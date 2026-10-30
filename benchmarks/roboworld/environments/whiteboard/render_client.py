"""One persistent, render-only Isaac worker per explicitly opened renderer."""
from gym_anything_robotics.paths import worker_env

from gym_anything_robotics.render_lifecycle import close_worker, watch_worker

import base64
import hashlib
import io
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
import uuid
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .ipc import send, receive
from .runtime import configure, ROOT

# The render interpreter (Isaac Sim's Python), as every other environment uses. Render-only.
ISAAC_PYTHON = Path(os.environ.get("ISAAC_RENDER_PYTHON", ""))


def available_cameras(env):
    cameras = {"classroom", "board", "tools", "duster", "student", "nib", "stroke"}
    if env.config.robot:
        cameras.add("g1_rgb")
    return cameras


class LiveRenderer:
    def __init__(self, env, *, width=960, height=540, samples=16, interpreter=ISAAC_PYTHON):
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (width, height, samples)
        ):
            raise ValueError("Width, height, samples must be positive integers")
        if not Path(interpreter).is_file():
            raise FileNotFoundError(f"Set ISAAC_RENDER_PYTHON to the Isaac Sim render interpreter (got {interpreter})")
        self.env, self.width, self.height, self.samples = env, width, height, samples
        self.lock, self.sequence, self.closed = threading.RLock(), 0, False
        self.folder = configure() / ("renderer_" + uuid.uuid4().hex[:12])
        self.folder.mkdir()
        scene = env.export(self.folder)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(180)
        child_env = dict(os.environ)
        child_env["WHITEBOARD_RENDER_PORT"] = str(listener.getsockname()[1])
        child_env["WHITEBOARD_RENDER_SECRET"] = secrets.token_urlsafe(32)
        child_env["LOCALAPPDATA"] = str(configure() / "localappdata")
        child_env["APPDATA"] = str(configure() / "appdata")
        self.log = (self.folder / "worker.log").open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen(
                [
                    str(interpreter),
                    "-m",
                    "whiteboard.render_worker",
                    "--scene",
                    str(scene.resolve()),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--samples",
                    str(samples),
                ],
                cwd=ROOT,
                env=worker_env(child_env),
                stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.socket, _ = listener.accept()
            self.socket.settimeout(180)
            hello = receive(self.socket)
            if not secrets.compare_digest(
                hello.get("secret", ""), child_env["WHITEBOARD_RENDER_SECRET"]
            ):
                raise PermissionError("Renderer authentication failed")
            ready = receive(self.socket)
            if ready.get("phase") != "ready":
                raise RuntimeError(ready)
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)
        self.last_texture = None

    def _poses(self):
        e = self.env
        q = e.state_0.body_q.numpy()
        poses = {f"/Classroom/{name}": q[i].tolist() for name, i in e.body_ids.items()}
        b = e.builder
        for i in range(len(e.shape_ids), b.shape_count):
            if not b.shape_flags[i] & int(e.newton.ShapeFlags.VISIBLE) or b.shape_body[i] < 0:
                continue
            pose, local = q[b.shape_body[i]], np.asarray(b.shape_transform[i])
            r = Rotation.from_quat(pose[3:])
            poses[f"/Classroom/G1/shape_{i}"] = np.r_[
                pose[:3] + r.apply(local[:3]), (r * Rotation.from_quat(local[3:])).as_quat()
            ].tolist()
        return poses

    def observe(self, cameras=("g1_rgb",)):
        with self.lock, self.env._lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            available = available_cameras(self.env)
            if not cameras or len(set(cameras)) != len(cameras) or set(cameras) - available:
                raise ValueError("Unknown, duplicate or empty camera selection")
            self.sequence += 1
            tick = self.env.tick
            request = {
                "op": "observe",
                "sequence": self.sequence,
                "tick": tick,
                "cameras": list(cameras),
                "poses": self._poses(),
            }
            from .deformable_contact import render_meshes

            request["elastic_vertices"] = {
                f"/Classroom/{name}/elastic_material": data["vertices"].tolist()
                for name, data in render_meshes(self.env).items()
            }
            pixels = self.env.ink.rgb()
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest != self.last_texture:
                buf = io.BytesIO()
                Image.fromarray(pixels).save(buf, format="PNG")
                request["board_png"] = base64.b64encode(buf.getvalue()).decode()
            if self.env.config.robot:
                from .sensors import optical_pose

                request["optical_pose"] = [v.tolist() for v in optical_pose(self.env)]
            start = time.perf_counter()
            send(self.socket, request)
            response = receive(self.socket)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Renderer failed"))
            if (
                response["sequence"] != self.sequence
                or response["tick"] != tick
                or self.env.tick != tick
            ):
                raise RuntimeError("Stale frame or render-induced clock advance")
            self.last_texture = digest
            response["wall_time_s"] = time.perf_counter() - start
            response["board_texture_sha256"] = digest
            return response

    def close(self):
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

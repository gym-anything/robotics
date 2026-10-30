"""One hidden render-only worker; sequence/tick checked, no physics stepping."""
from gym_anything_robotics.paths import worker_env

from gym_anything_robotics.render_lifecycle import close_worker, watch_worker

import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
from whiteboard.ipc import send, receive
from .render import export, optical_pose


class Renderer:
    def __init__(
        self, world, output, *, width=1280, height=720, samples=16, interpreter=None
    ):
        for value in (width, height, samples):
            if type(value) is not int or value <= 0:
                raise ValueError("Positive integer render dimensions/samples required")
        self.world, self.width, self.height, self.samples = (
            world,
            width,
            height,
            samples,
        )
        self.folder = Path(output).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        interpreter = (interpreter or os.environ.get("FOOTBALL_RENDER_PYTHON")
                       or os.environ.get("ISAAC_RENDER_PYTHON") or "")
        if not interpreter or not Path(interpreter).is_file():
            raise FileNotFoundError(
                "Set FOOTBALL_RENDER_PYTHON or ISAAC_RENDER_PYTHON to the Isaac Sim 5.1 render interpreter"
                + (f" ({interpreter} is not a file)" if interpreter else "")
            )
        scene = export(world, self.folder)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(240)
        secret = secrets.token_urlsafe(32)
        env = dict(
            os.environ,
            FOOTBALL_RENDER_PORT=str(listener.getsockname()[1]),
            FOOTBALL_RENDER_SECRET=secret,
            FOOTBALL_RENDER_SESSION=str(self.folder),
        )
        self.log = (self.folder / "worker.log").open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen(
                [
                    str(interpreter),
                    "-m",
                    "football.newton15.render_worker",
                    "--scene",
                    str(scene),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--samples",
                    str(samples),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env=worker_env(env),
                stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.socket, _ = listener.accept()
            self.socket.settimeout(240)
            if not secrets.compare_digest(
                receive(self.socket).get("secret", ""), secret
            ):
                raise PermissionError("Unrecognized renderer")
            if receive(self.socket).get("phase") != "ready":
                raise RuntimeError("Renderer not ready")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)
        self.sequence, self.closed, self.lock = 0, False, threading.RLock()

    def observe(self, cameras=("g1_rgb",)):
        with self.lock, self.world.lock:
            if self.closed:
                raise RuntimeError("Renderer closed")
            allowed = {"overview", "sideline", "contact", "goal"} | (
                {"g1_rgb"} if self.world.robot else set()
            )
            if (
                not cameras
                or len(set(cameras)) != len(cameras)
                or set(cameras) - allowed
            ):
                raise ValueError("Invalid cameras")
            self.sequence += 1
            w = self.world
            request = {
                "op": "observe",
                "sequence": self.sequence,
                "tick": w.tick,
                "cameras": list(cameras),
                "poses": w.poses.tolist(),
                "nets": {name: net.position.tolist() for name, net in w.nets.items()},
            }
            from .deformation import shell_state

            request["ball_surface"] = {"body": w.ball_id, "state": shell_state(w)}
            if w.robot:
                request["optical_pose"] = optical_pose(w)
            start = time.perf_counter()
            send(self.socket, request)
            result = receive(self.socket)
            if not result.get("ok"):
                raise RuntimeError(result)
            if (
                result["sequence"] != self.sequence
                or result["tick"] != w.tick
                or result["physics_steps"] != 0
            ):
                raise RuntimeError("Stale render or render-induced physics advancement")
            result["wall_s"] = time.perf_counter() - start
            return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

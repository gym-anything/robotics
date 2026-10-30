"""Authenticated local persistent rendering, independent of environment ticks."""
from gym_anything_robotics.paths import worker_env

from gym_anything_robotics.render_lifecycle import close_worker, watch_worker

import os
import json
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
import uuid
from whiteboard.ipc import send, receive
from .runtime import configure


class Renderer:
    """One observation interface; volume transport is the titration default.

    RTX remains explicitly selectable for non-liquid diagnostic fixtures only.
    Its tested boundary/volume limitations must not silently discard chemistry.
    """
    def __new__(cls, scene, width=1280, height=720, samples=32, interpreter=None, backend='volume'):
        if backend == 'volume':
            if interpreter is not None:
                raise ValueError('Volume rendering runs in the pinned Newton Python; no Kit interpreter override')
            from .transport_renderer import TransportRenderer
            return TransportRenderer(scene,width,height,samples)
        if backend == 'rtx':
            return RtxRenderer(scene,width,height,samples,interpreter)
        raise ValueError('Unknown rendering backend')


class RtxRenderer:
    def __init__(self, scene, width=1280, height=720, samples=32, interpreter=None):
        if any(type(x) is not int or x <= 0 for x in (width, height, samples)):
            raise ValueError("Positive integer image dimensions and samples required")
        configured = configure()/"runtime.json"
        selected = interpreter or os.environ.get("TITRATE_RENDER_PYTHON")
        if not selected and configured.is_file():
            selected = json.loads(configured.read_text(encoding='utf-8'))['render_interpreter']
        if not selected:
            raise FileNotFoundError('Run python -m titrate.newton15.bootstrap --isaac-root <Isaac-Sim-5.1-root>, '
                'or set TITRATE_RENDER_PYTHON to its render interpreter')
        interpreter = Path(selected)
        if not interpreter.is_file():
            raise FileNotFoundError(interpreter)
        self.folder = configure()/("render_"+uuid.uuid4().hex[:10])
        self.folder.mkdir()
        self.log = (self.folder/"worker.log").open("wb")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(240)
        env = dict(os.environ)
        secret = secrets.token_urlsafe(32)
        env.update(TITRATE_RENDER_PORT=str(listener.getsockname()[1]), TITRATE_RENDER_SECRET=secret,
                   TITRATE_RENDER_SESSION=str(self.folder),
                   LOCALAPPDATA=str(configure()/"localappdata"), APPDATA=str(configure()/"appdata"))
        self.process = None
        try:
            self.process = subprocess.Popen([str(interpreter), "-m", "titrate.newton15.render_worker",
                "--scene", str(Path(scene).resolve()), "--width", str(width), "--height", str(height),
                "--samples", str(samples)], cwd=Path(__file__).resolve().parents[2], env=worker_env(env),
                stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.connection, _ = listener.accept()
            self.connection.settimeout(240)
            hello = receive(self.connection)
            if not secrets.compare_digest(hello.get("secret", ""), secret):
                raise PermissionError("Unrecognized render worker")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Render worker did not become ready")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)
        self.sequence, self.closed = 0, False
        self.lock = threading.Lock()

    def observe(self, tick, body_poses, cameras, liquid=None, *, visual_signals=None, indicators=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            self.sequence += 1
            request = {"op": "observe", "sequence": self.sequence, "tick": tick,
                       "body_poses": body_poses.tolist(), "cameras": list(cameras)}
            if indicators:
                # Exported indicator lamps (/Materials/<name> with customData
                # indicator_rgb) switched on/off from simulator truth.
                if not isinstance(indicators,dict) or any(
                        type(k) is not str or not k.isidentifier() or type(v) is not bool
                        for k,v in indicators.items()):
                    raise ValueError("Indicators are {material_name: bool}")
                request["indicators"]=dict(indicators)
            if visual_signals is not None:
                allowed={"power","sensor","pass_","fail"}
                if not isinstance(visual_signals,dict) or set(visual_signals)!=allowed:
                    raise ValueError("Control-cable indicator schema mismatch")
                if any(type(v) is not bool for v in visual_signals.values()):
                    raise ValueError("Indicator states must be booleans")
                request["control_cable_indicators"]=visual_signals
            if liquid is not None:
                if liquid.pink_volume is not None or liquid.pink_molar is not None:
                    raise ValueError('RTX cannot render titration interior chemistry correctly; use backend=volume')
                import trimesh
                from .render_payload import encode

                normals = trimesh.Trimesh(liquid.vertices, liquid.triangles, process=False).vertex_normals
                request["liquid"] = {"vertices": encode(liquid.vertices, 'float32'),
                    "triangles": encode(liquid.triangles, 'int32'), "normals": encode(normals, 'float32')}
            start = time.perf_counter()
            send(self.connection, request)
            result = receive(self.connection)
            if not result.get("ok"):
                raise RuntimeError(result)
            if result["sequence"] != self.sequence or result["tick"] != tick or result["physics_steps"] != 0:
                raise RuntimeError("Render clock/sequence mismatch")
            result["wall_s"] = time.perf_counter()-start
            return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

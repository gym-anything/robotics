"""Small authenticated HTTP boundary for the persistent whiteboard simulator.

Every mutating request is serialized. No renderer owns the physics clock.
Errors report whether physics advanced; a render failure never silently retries
an action. Action idempotency keys are scoped to an episode and request body.
"""

import argparse
import base64
from collections import OrderedDict
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit
import uuid

from .env import WhiteboardEnv
from .tasks import Task


class ApiError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def strict_fields(body, allowed, required=()):
    if not isinstance(body, dict) or set(body) - set(allowed) or set(required) - set(body):
        raise ApiError(400, f"Expected fields {list(allowed)}, required {list(required)}")


def diagnostic_state(state):
    """Keep corrupt historical states retrievable without inventing finite data."""
    missing = []

    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            missing.append(path)
            return None
        if isinstance(value, dict):
            return {k: clean(v, f"{path}/{k}") for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, f"{path}/{i}") for i, v in enumerate(value)]
        return value

    result = clean(state, "")
    result["nonfinite_fields"] = missing
    return result


class Service:
    def __init__(self, env, teacher_key, policy_key, *, renderer_options=None, recording_dir=None):
        if not teacher_key or not policy_key or teacher_key == policy_key:
            raise ValueError("Two different, nonempty keys are required")
        self.env, self.teacher_key, self.policy_key = env, teacher_key, policy_key
        self.renderer_options = dict(renderer_options or {})
        self.lock = threading.RLock()
        self.episode = uuid.uuid4().hex
        self.commands = OrderedDict()
        self.checkpoints = OrderedDict()
        self.history = OrderedDict()
        self.evicted_history = 0
        self.command_sequence = 0
        self.store = None
        if recording_dir is not None:
            from .storage import Store

            self.store = Store(recording_dir)
            env.trace.attach(self.store)
            for key in self.store.keys("command"):
                record = self.store.get("command", key)
                self.command_sequence = max(self.command_sequence, record["command_id"])
                if record.get("phase") == "started":
                    record.update(
                        phase="interrupted",
                        execution="unknown; process ended before durable completion; never automatically retry",
                    )
                    self.store.put("command", key, record)
                if "result" in record:
                    self.history[record["command_id"]] = record
                    self._remember(
                        tuple(record["cache_key"]), record["fingerprint"], record["result"]
                    )
        from .geometry_api import GeometryView

        self.geometry = GeometryView(env)

    def role(self, authorization):
        value = authorization.removeprefix("Bearer ")
        if authorization.startswith("Bearer ") and secrets.compare_digest(value, self.teacher_key):
            return "teacher"
        if authorization.startswith("Bearer ") and secrets.compare_digest(value, self.policy_key):
            return "policy"
        raise ApiError(401, "Bearer authentication required")

    def status(self):
        return {
            "episode_id": self.episode,
            "tick": self.env.tick,
            "time_s": self.env.tick * self.env.config.physics_dt,
            "ready": self.env.healthy,
            "truncated": self.env.tick >= self.env.config.episode_ticks,
            "recording": "durable" if self.store is not None else "memory_only",
        }

    def help(self, role):
        common = {
            "GET /v1/health": "Process and simulation readiness, with authoritative tick",
            "GET /v1/help": "This contract; role-specific routes",
            "GET /v1/task": "Public task instruction and desired reference image",
            "GET /v1/robot": "Joint names, ranges, actuator efforts and camera calibration",
            "GET /v1/proprioception": "Joint encoders; does not advance physics",
            "POST /v1/observe": {"cameras": ["g1_rgb"]},
            "POST /v1/action": {
                "id": "unique string",
                "episode_id": self.episode,
                "targets": {"left_elbow_joint": 0.1},
                "ticks": 100,
            },
            "POST /v1/reset": {"id": "unique string", "episode_id": self.episode},
        }
        if role == "teacher":
            common.update(
                {
                    "GET /v1/state": "Privileged physical state, including robot",
                    "GET /v1/geometry": "Shared metric object/robot geometry descriptor",
                    "POST /v1/geometry/mesh": {
                        "mesh_id": 0,
                        "array": "vertices",
                        "start": 0,
                        "limit": 1000,
                    },
                    "GET /v1/evaluate": "Private reference-overlap reward diagnostics",
                    "GET /v1/history": "Bounded before/after action state history",
                    "POST /v1/history/trace": {"after": 0, "limit": 100},
                    "POST /v1/checkpoints/save": {"name": "checkpoint"},
                    "GET /v1/checkpoints": "List saved checkpoints (survive process restart when durable recording is enabled)",
                    "POST /v1/checkpoints/restore": {
                        "name": "checkpoint",
                        "id": "unique string",
                        "episode_id": self.episode,
                    },
                    "POST /v1/variation": {
                        "id": "unique string",
                        "episode_id": self.episode,
                        "seed": 1,
                        "task": {"kind": "bridge"},
                    },
                    "POST /v1/variations/preview": {
                        "variation_id": 0,
                        "split": "optional",
                        "kind": "optional",
                        "selection_seed": 0,
                    },
                    "POST /v1/variations/start": {
                        "id": "unique",
                        "episode_id": self.episode,
                        "variation_id": 0,
                    },
                }
            )
        return {
            "schema": "whiteboard.http.v1",
            "role": role,
            "routes": common,
            "action_units": "radians and integer physics ticks",
            "physics_dt_s": self.env.config.physics_dt,
            "max_action_ticks": 4000,
            "episode_ticks": self.env.config.episode_ticks,
            "clock": "Only explicit actions advance physics. A failed action may have advanced ticks; inspect ticks_executed. Never retry under a new id automatically.",
            "idempotency": "Same id+episode+body returns the cached result; changed body is 409. Last 1024 commands retained.",
            "concurrency": "One serialized owner; observations are read-only and reflect one exact tick",
            "camera_roles": {
                "policy": ["g1_rgb"],
                "teacher": [
                    "classroom",
                    "board",
                    "tools",
                    "duster",
                    "student",
                    "nib",
                    "stroke",
                    "g1_rgb",
                ],
            },
            "limits": "No real-world material calibration claim",
        }

    def _remember(self, key, fingerprint, result):
        self.commands[key] = (fingerprint, result)
        while len(self.commands) > 1024:
            self.commands.popitem(last=False)

    def _mutation(self, route, body, role):
        if not isinstance(body.get("id"), str) or not 1 <= len(body["id"]) <= 128:
            raise ApiError(400, "id must be a nonempty string, at most 128 characters")
        episode = body.get("episode_id")
        if not isinstance(episode, str):
            raise ApiError(400, "episode_id must be a string")
        key = (episode, body["id"])
        fingerprint = hashlib.sha256(
            json.dumps([route, body], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if key in self.commands:
            old_fingerprint, result = self.commands[key]
            if old_fingerprint != fingerprint:
                raise ApiError(409, "Idempotency id reused with different request")
            return result
        if episode != self.episode:
            raise ApiError(409, "Stale episode_id; read /v1/health")
        before = self.env.get_state()
        self.command_sequence += 1
        command_id = self.command_sequence
        self.env.trace.command_id = command_id
        record = {
            "command_id": command_id,
            "cache_key": list(key),
            "fingerprint": fingerprint,
            "request": body,
            "route": route,
            "role": role,
            "before": diagnostic_state(before),
            "phase": "started",
            "started_ns": time.time_ns(),
        }
        if self.store is not None:
            self.store.put("command", command_id, record)
            self.store.retain("command", 1024)
        failure = None
        failure_detail = None
        control_result = None
        started = time.perf_counter()
        try:
            if route == "/v1/action":
                strict_fields(body, ("id", "episode_id", "targets", "ticks"), ("targets", "ticks"))
                if (
                    isinstance(body["ticks"], bool)
                    or not isinstance(body["ticks"], int)
                    or not 1 <= body["ticks"] <= 4000
                ):
                    raise ApiError(400, "ticks must be an integer in [1,4000]; no clamping")
                self.env.act(body["targets"], ticks=body["ticks"])
            elif route == "/v1/cartesian":
                strict_fields(body, ("id", "episode_id", "action"), ("action",))
                from gym_anything_robotics.action_codec import action_from_dict
                from gym_anything_robotics import CartesianPose, CartesianDelta
                from .ik import execute_cartesian

                action = action_from_dict(body["action"])
                if not isinstance(action, (CartesianPose, CartesianDelta)):
                    raise ApiError(400, "Expected cartesian_pose or cartesian_delta")
                control_result = execute_cartesian(self.env, action)
            elif route == "/v1/reset":
                strict_fields(body, ("id", "episode_id"))
                self.env.reset()
                self.episode = uuid.uuid4().hex
            elif route == "/v1/checkpoints/restore":
                strict_fields(body, ("id", "episode_id", "name"), ("name",))
                if self.store is not None and body["name"] in self.store.keys("checkpoint"):
                    from .storage import restore_bytes

                    restore_bytes(self.env, self.store.get("checkpoint", body["name"], raw=True))
                elif body["name"] in self.checkpoints:
                    self.env.restore(self.checkpoints[body["name"]])
                else:
                    raise ApiError(404, "Checkpoint not found")
                self.episode = uuid.uuid4().hex
            elif route in ("/v1/variation", "/v1/variations/start"):
                if route == "/v1/variation":
                    strict_fields(body, ("id", "episode_id", "seed", "task"), ("seed", "task"))
                    config = replace(self.env.config, seed=body["seed"])
                    task = Task(**body["task"])
                else:
                    from .variations import resolve, select

                    strict_fields(
                        body,
                        ("id", "episode_id", "variation_id", "split", "kind", "selection_seed"),
                    )
                    selection = select(
                        **{k: v for k, v in body.items() if k not in ("id", "episode_id")}
                    )
                    config, task = resolve(selection["variation_id"], base=self.env.config)
                    control_result = {"variation": selection}
                replacement = WhiteboardEnv(config, task=task)
                previous = self.env
                self.env = replacement
                if self.store is not None:
                    replacement.trace.attach(self.store)
                from .geometry_api import GeometryView

                self.geometry = GeometryView(replacement)
                previous.close()
                self.episode = uuid.uuid4().hex
                self.checkpoints.clear()
            else:
                raise ApiError(404, "Unknown mutation route")
        except ApiError as exc:
            record.update(
                phase="rejected",
                http_status=exc.status,
                error=str(exc),
                after=diagnostic_state(self.env.get_state()),
                finished_ns=time.time_ns(),
            )
            self.history[command_id] = record
            if self.store is not None:
                self.store.put("command", command_id, record)
            raise
        except (ValueError, TypeError, KeyError) as exc:
            if self.env.tick == before["tick"] and self.env.healthy:
                record.update(
                    phase="rejected",
                    http_status=400,
                    error=str(exc),
                    after=diagnostic_state(self.env.get_state()),
                    finished_ns=time.time_ns(),
                )
                self.history[command_id] = record
                if self.store is not None:
                    self.store.put("command", command_id, record)
                raise ApiError(400, str(exc)) from exc
            failure = type(exc).__name__
            failure_detail = str(exc)
        except Exception as exc:
            failure = type(exc).__name__
            failure_detail = str(exc)
        after = self.env.get_state()
        result = {
            "ok": failure is None,
            **self.status(),
            "command_id": command_id,
            "ticks_executed": after["tick"] - before["tick"]
            if route in ("/v1/action", "/v1/cartesian")
            else 0,
            "control": control_result,
            "error": failure,
            "wall_time_s": time.perf_counter() - started,
        }
        self.history[command_id] = {
            **record,
            "phase": "finished",
            "command_id": command_id,
            "request": body,
            "route": route,
            "role": role,
            "before": diagnostic_state(before),
            "after": diagnostic_state(after),
            "result": result,
            "failure_detail": failure_detail,
        }
        if self.store is not None:
            self.store.put("command", command_id, self.history[command_id])
        while len(self.history) > 128:
            self.history.popitem(last=False)
            self.evicted_history += 1
        self._remember(key, fingerprint, result)
        return result

    def handle(self, method, route, body, role):
        privileged = {
            "/v1/state",
            "/v1/geometry",
            "/v1/geometry/mesh",
            "/v1/evaluate",
            "/v1/history",
            "/v1/history/trace",
            "/v1/checkpoints/save",
            "/v1/checkpoints",
            "/v1/checkpoints/restore",
            "/v1/variation",
            "/v1/variations/preview",
            "/v1/variations/start",
        }
        if route in privileged and role != "teacher":
            raise ApiError(403, "Teacher key required")
        with self.lock:
            if method == "GET":
                if route == "/v1/health":
                    return {"ok": True, **self.status()}
                if route == "/v1/help":
                    return self.help(role)
                if route == "/v1/task":
                    from .tasks import reference_png

                    task, ink = self.env.task, self.env.ink
                    return {
                        "description": task.description,
                        "reference_png": base64.b64encode(
                            reference_png(task, ink.board, ink.width, ink.height)
                        ).decode(),
                        "reference_role": task.reference_role,
                        "episode_id": self.episode,
                    }
                if route == "/v1/proprioception":
                    return self.env.proprioception()
                if route == "/v1/robot":
                    return {
                        "joints": {
                            name: {k: v for k, v in j.items() if k not in ("q", "dof")}
                            for name, j in self.env.robot_joints.items()
                        },
                        "cameras": {
                            "g1_rgb": {
                                "mount": "G1 URDF d435_joint",
                                "horizontal_fov_deg": 69,
                                "calibration": "Ideal pinhole; optical center offset approximated as zero",
                            }
                        },
                        "control": "Bounded position/velocity servos on 43 joints; no hidden task policy",
                    }
                if route == "/v1/state":
                    return self.env.get_state()
                if route == "/v1/checkpoints":
                    return {
                        "names": self.store.keys("checkpoint")
                        if self.store is not None
                        else list(self.checkpoints),
                        "durable": self.store is not None,
                    }
                if route == "/v1/evaluate":
                    return self.env.evaluate()
                if route == "/v1/geometry":
                    return self.geometry.describe()
                if route == "/v1/history":
                    return {
                        "entries": [
                            self.store.get("command", k) for k in self.store.keys("command")
                        ]
                        if self.store is not None
                        else list(self.history.values()),
                        "evicted_entries": self.evicted_history,
                        "retention_entries": 1024 if self.store is not None else 128,
                        "disk": self.store.metadata() if self.store is not None else None,
                        "sampling": "Before/after commands plus per-tick /v1/history/trace",
                        "body_labels": list(self.env.model.body_label),
                        "geom_labels": [
                            self.env.solver.mj_model.geom(i).name
                            for i in range(self.env.solver.mj_model.ngeom)
                        ],
                    }
                raise ApiError(404, "Unknown GET route")
            if method != "POST":
                raise ApiError(405, "Unsupported method")
            if route == "/v1/variations/preview":
                from .variations import select

                strict_fields(body, ("variation_id", "split", "kind", "selection_seed"))
                return select(**body)
            if route == "/v1/geometry/mesh":
                strict_fields(body, ("mesh_id", "array", "start", "limit"), ("mesh_id", "array"))
                return self.geometry.mesh(**body)
            if route == "/v1/history/trace":
                strict_fields(body, ("after", "limit"))
                return self.env.trace.page(body.get("after", 0), body.get("limit", 100))
            if route == "/v1/observe":
                strict_fields(body, ("cameras",))
                cameras = body.get("cameras", ["g1_rgb"])
                if not isinstance(cameras, list) or any(not isinstance(c, str) for c in cameras):
                    raise ApiError(400, "cameras must be a list of names")
                if role != "teacher" and cameras != ["g1_rgb"]:
                    raise ApiError(403, "Policy key permits only g1_rgb")
                from .render_client import available_cameras

                if (
                    not cameras
                    or len(set(cameras)) != len(cameras)
                    or set(cameras) - available_cameras(self.env)
                ):
                    raise ApiError(400, "Unknown, duplicate or empty camera selection")
                if self.env._renderer is None:
                    self.env.open_renderer(**self.renderer_options)
                result = {**self.env.obs(cameras), "episode_id": self.episode}
                if role != "teacher":
                    result.pop("board_texture_sha256", None)
                return result
            if route == "/v1/checkpoints/save":
                strict_fields(body, ("name",), ("name",))
                name = body["name"]
                if not isinstance(name, str) or not 1 <= len(name) <= 64:
                    raise ApiError(400, "Checkpoint name length must be 1..64")
                if not self.env.healthy:
                    raise ApiError(409, "Cannot save an unhealthy state as a recovery checkpoint")
                if name not in self.checkpoints and len(self.checkpoints) >= 8:
                    raise ApiError(409, "Eight checkpoints already retained")

                def byte_size(snapshot):
                    import numpy as np

                    return (
                        sum(v.nbytes for v in snapshot.values() if isinstance(v, np.ndarray))
                        + snapshot["ink"].metadata()["allocated_bytes"]
                    )

                candidate_bytes = (
                    self.env.ink.metadata()["allocated_bytes"]
                    + sum(
                        getattr(self.env.state_0, n).numpy().nbytes
                        for n in ("body_q", "body_qd", "joint_q", "joint_qd")
                    )
                    + 1024 * 1024
                )
                if (
                    sum(byte_size(v) for k, v in self.checkpoints.items() if k != name)
                    + candidate_bytes
                    > 256 * 1024 * 1024
                ):
                    raise ApiError(409, "Checkpoint retention would exceed 256 MiB")
                candidate = self.env.snapshot()
                if self.store is not None:
                    from .storage import snapshot_bytes

                    if (
                        name not in self.store.keys("checkpoint")
                        and len(self.store.keys("checkpoint")) >= 8
                    ):
                        raise ApiError(409, "Eight durable checkpoints already retained")
                    self.store.put("checkpoint", name, snapshot_bytes(self.env), encoded=True)
                self.checkpoints[name] = candidate
                return {"ok": True, "name": name, **self.status()}
            if route not in {
                "/v1/action",
                "/v1/reset",
                "/v1/checkpoints/restore",
                "/v1/variation",
                "/v1/variations/start",
            }:
                raise ApiError(404, "Unknown POST route")
            return self._mutation(route, body, role)


def make_server(service, host="127.0.0.1", port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log authorization headers or request contents.

        def dispatch(self):
            route = urlsplit(self.path).path
            if self.command == "GET" and route in (
                "/",
                "/app.js",
                "/styles.css",
                "/vendor/three.module.js",
                "/vendor/three.core.js",
                "/vendor/OrbitControls.js",
            ):
                root = Path(__file__).resolve().parent
                file = root / "static" / ("index.html" if route == "/" else route[1:])
                if route.startswith("/vendor/"):
                    file = root / ".cache/web_vendor" / route.split("/")[-1]
                if not file.is_file():
                    self.send_error(
                        404, "Viewer asset unavailable; run python -m whiteboard.web_assets"
                    )
                    return
                content = file.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/html"
                    if route == "/"
                    else "text/css"
                    if route.endswith(".css")
                    else "text/javascript",
                )
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            try:
                role = service.role(self.headers.get("Authorization", ""))
                route = urlsplit(self.path).path
                body = {}
                if self.command == "POST":
                    if self.headers.get("Transfer-Encoding"):
                        raise ApiError(400, "Chunked requests are not accepted")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 256 * 1024:
                        raise ApiError(413, "Request must contain 1..262144 bytes")
                    body = json.loads(
                        self.rfile.read(length),
                        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
                    )
                    if not isinstance(body, dict):
                        raise ApiError(400, "JSON body must be an object")
                result = service.handle(self.command, route, body, role)
                status = 200 if result.get("ok", True) else 503
            except ApiError as exc:
                status, result = exc.status, {"ok": False, "error": str(exc)}
            except (ValueError, TypeError) as exc:
                status, result = 400, {"ok": False, "error": str(exc)}
            except Exception:
                status, result = (
                    503,
                    {
                        "ok": False,
                        "error": "Operation unavailable; no implicit retry",
                        "ready": service.env.healthy,
                    },
                )
            try:
                payload = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
            except (ValueError, TypeError):
                status = 503
                service.env.healthy = False
                payload = b'{"ok":false,"ready":false,"error":"Non-finite or invalid state; explicit recovery required"}'
            self.send_response(status)
            if "gzip" in self.headers.get("Accept-Encoding", "") and len(payload) > 8192:
                import gzip

                payload = gzip.compress(payload, compresslevel=1)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = dispatch
        do_POST = dispatch

    return ThreadingHTTPServer((host, port), Handler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recording-dir", type=Path, default=Path("whiteboard/output/server_history"))
    p.add_argument(
        "--restore",
        help="Explicitly restore this durable checkpoint; creates a new discontinuous episode",
    )
    p.add_argument("--task", choices=["text", "bridge"], default="text")
    p.add_argument(
        "--credentials", type=Path, default=Path("whiteboard/output/api_credentials.json")
    )
    a = p.parse_args()
    if a.credentials.exists():
        keys = json.loads(a.credentials.read_text())
    else:
        keys = {"teacher_key": secrets.token_urlsafe(32), "policy_key": secrets.token_urlsafe(32)}
        a.credentials.parent.mkdir(parents=True, exist_ok=True)
        a.credentials.write_text(json.dumps(keys, indent=2))
    service = Service(
        WhiteboardEnv(seed=a.seed, task=Task(kind=a.task)), **keys, recording_dir=a.recording_dir
    )
    if a.restore:
        from .storage import restore_bytes

        restore_bytes(service.env, service.store.get("checkpoint", a.restore, raw=True))
    server = make_server(service, a.host, a.port)
    print(
        f"Whiteboard API: http://{a.host}:{server.server_port}; credentials: {a.credentials.resolve()}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.env.close()
        if service.store is not None:
            service.store.close()


if __name__ == "__main__":
    main()

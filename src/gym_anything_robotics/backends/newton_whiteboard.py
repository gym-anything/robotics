"""Shared GAR adapter, with lazy Newton/renderer imports and no task policy."""

import base64
from dataclasses import replace, asdict
from collections import deque
from pathlib import Path
import time
import uuid

from .base import RoboticsBackend, reject_unexecuted
from ..actions import NamedJointPosition, CartesianPose, CartesianDelta
from ..errors import UnsupportedActionError, UnsupportedConfigurationError
from ..types import BackendStep, Capabilities, CameraSpec, ImageFrame, Observation


class NewtonWhiteboardBackend(RoboticsBackend):
    def __init__(
        self,
        *,
        robot="g1",
        config=None,
        task=None,
        cameras=("g1_rgb",),
        renderer_options=None,
        output_dir=None,
        recording=True,
        variation_id=None,
        **config_options,
    ):
        if robot != "g1":
            raise UnsupportedConfigurationError(
                "whiteboard uses the pinned Unitree G1/Dex3; robot must be g1"
            )
        from whiteboard.config import Config
        from whiteboard.env import WhiteboardEnv
        from whiteboard.storage import Store

        self.config = config or Config(**config_options)
        if config is not None and config_options:
            raise ValueError("Pass Config or individual options, not both")
        if variation_id is not None:
            if task is not None or "seed" in config_options:
                raise ValueError(
                    "Catalogue variation supplies its seed/task; do not combine explicit overrides"
                )
            from whiteboard.variations import resolve

            self.config, task = resolve(variation_id, base=self.config)
        if not self.config.robot:
            raise UnsupportedConfigurationError(
                "GAR whiteboard includes G1; robot-free fixtures use WhiteboardEnv directly"
            )
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - {
            "g1_rgb"
        }:
            raise ValueError(
                "GAR obs supports g1_rgb only; use render() for diagnostic views"
            )
        self.output = Path(
            output_dir or Path("whiteboard/output/sessions") / uuid.uuid4().hex
        ).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self._image_paths = deque()
        self.renderer_options = dict(renderer_options or {})
        self.env = WhiteboardEnv(self.config, task=task)
        self.store = Store(self.output / "history") if recording else None
        if self.store is not None:
            self.env.trace.attach(self.store)
        self._command_sequence = (
            max((int(k) for k in self.store.keys("command")), default=0)
            if self.store
            else 0
        )
        width, height = (
            self.renderer_options.get("width", 960),
            self.renderer_options.get("height", 540),
        )
        import math

        focal = 0.018 / math.tan(math.radians(69 / 2))
        self.camera = CameraSpec(
            "g1_rgb",
            "d435_link",
            width,
            height,
            1280,
            720,
            (0.036, 0.036 * height / width),
            (focal, focal),
        )
        self.capabilities = Capabilities(
            "whiteboard",
            "newton",
            "g1",
            frozenset({"named_joint_position", "cartesian_pose", "cartesian_delta"}),
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True,
            True,
            True,
            self.cameras,
            (self.camera,) if self.cameras else (),
            notes={
                "clock": "Exact explicit physics ticks; no background advancement",
                "camera": "Ideal 69deg D435 RGB: 36mm VIRTUAL filmback, not measured lens/sensor dimensions; URDF mount, zero optical offset approximation",
                "control": "43 named joints; Cartesian hand targets via bounded IK, no object state controller",
            },
        )

    def reset(self, *, seed=None, options=None):
        if options and (set(options) != {"variation_id"} or seed is not None):
            raise ValueError(
                "Only reset option variation_id is supported; do not combine a seed override"
            )
        if options or (seed is not None and seed != self.config.seed):
            from whiteboard.env import WhiteboardEnv

            if options:
                from whiteboard.variations import resolve

                config, task = resolve(options["variation_id"], base=self.config)
            else:
                config, task = replace(self.config, seed=seed), self.env.task
            candidate = WhiteboardEnv(config, task=task)
            self.env.close()
            self.config, self.env = config, candidate
            if self.store is not None:
                candidate.trace.attach(self.store)
        else:
            self.env.reset()
        self._score_cache = None
        return self.get_state(), {"seed": self.config.seed, "continuity": "reset"}

    def evaluate(self):
        if not self.env.healthy:
            raise RuntimeError("Cannot evaluate a faulted whiteboard")
        return self.env.evaluate()

    def get_state(self):
        return self.env.get_state()

    def _render(self, cameras):
        from ..render_lifecycle import renderer_operation
        with self.env._lock, renderer_operation(self.env, "_renderer"):
            if self.env._renderer is None:
                self.env.open_renderer(**self.renderer_options)
            result = self.env.obs(cameras)
        paths = {}
        for name, payload in result["images"].items():
            path = self.output / f"{name}_{uuid.uuid4().hex}.png"
            path.write_bytes(base64.b64decode(payload))
            paths[name] = path
            self._image_paths.append(path)
            while len(self._image_paths) > 64:
                old = self._image_paths.popleft()
                if old.parent != self.output:
                    raise RuntimeError("Image retention path escaped session output")
                old.unlink(missing_ok=True)
        return paths

    def observation_from_state(self, state, *, sequence):
        if state["tick"] != self.env.tick:
            raise RuntimeError("Observation state is not current")
        images = {
            name: ImageFrame(name, path, True, camera=self.camera)
            for name, path in (
                self._render(self.cameras).items() if self.cameras else []
            )
        }
        return Observation(
            sequence, state["time_s"], "g1", {"joints": state["joints"]}, images
        )

    def policy_access(self):
        from ..policy_access import PolicyAccess

        return PolicyAccess(
            "g1",
            (self.camera,) if self.cameras else (),
            {
                ("joints", name, field): None
                for name in self.env.robot_joints
                for field in ("position_rad", "velocity_rad_s")
            },
            frozenset({"named_joint_position"}),
            lambda: self.env.task.description,
            hardware={
                "joints": {
                    name: {
                        k: v
                        for k, v in j.items()
                        if k in ("lower", "upper", "effort_nm")
                    }
                    for name, j in self.env.robot_joints.items()
                }
            },
            task_images=self._task_images if self.env.task.kind == "bridge" else None,
        )

    def _task_images(self):
        """The bridge diagram to copy, as the board is seen from the front (Task.reference_view)."""
        from whiteboard.tasks import reference_png

        task, ink = self.env.task, self.env.ink
        return {"reference": (task.reference_role, reference_png(task, ink.board, ink.width, ink.height))}

    def _execute(self, action):
        if isinstance(action, NamedJointPosition):
            self.env.act(dict(action.targets_rad), ticks=action.ticks)
            info = {"ticks_executed": action.ticks}
        elif isinstance(action, (CartesianPose, CartesianDelta)):
            from whiteboard.ik import execute_cartesian

            info = execute_cartesian(self.env, action)
        else:
            raise UnsupportedActionError(
                "Use NamedJointPosition or a supported Cartesian hand command"
            )
        return info

    def _checked(self, action, role):
        # A rejected target (unknown joint, outside the joint limits, past the episode) cancels the command only.
        return reject_unexecuted(lambda: self._recorded(action, role), lambda: (
            self.env.tick, self.env.control.joint_target_q.numpy().copy()))

    def _recorded(self, action, role):
        from whiteboard.api import diagnostic_state

        if self.store is None:
            return self._execute(action)
        with self.env._lock:
            self._command_sequence += 1
            command = self._command_sequence
            record = {
                "command_id": command,
                "source": "gar",
                "role": role,
                "phase": "started",
                "action_type": type(action).__name__,
                "request": asdict(action),
                "wall_time_ns": time.time_ns(),
                "before": diagnostic_state(self.env.get_state()),
            }
            self.store.put("command", command, record)
            old = self.env.trace.command_id
            self.env.trace.command_id = command
            try:
                info = self._execute(action)
            except Exception as exc:
                record.update(
                    phase="failed",
                    error=type(exc).__name__,
                    detail=str(exc),
                    after=diagnostic_state(self.env.get_state()),
                )
                try:
                    self.store.put("command", command, record)
                except Exception as recording_error:
                    self.env.healthy = False
                    exc.add_note(f"Could not record failure: {recording_error}")
                raise
            else:
                record.update(
                    phase="complete",
                    after=diagnostic_state(self.env.get_state()),
                    info=info,
                )
                try:
                    self.store.put("command", command, record)
                except Exception:
                    self.env.healthy = False
                    raise
                return info
            finally:
                self.env.trace.command_id = old

    def act(self, action):
        # Bridge tasks: the step reward is the change in score, so an episode's return is its final score
        # whatever the drawing order. Text tasks keep the absolute score.
        bridge = self.env.task.kind == "bridge"
        key = (id(self.env.ink), self.env.ink.revision, self.env.task)
        cached = getattr(self, "_score_cache", None)
        before = (cached[1] if cached and cached[0] == key else self.env.evaluate()["score"]) if bridge else 0.0
        info = self._checked(action, "teacher")
        reward = self.env.evaluate()
        self._score_cache = ((id(self.env.ink), self.env.ink.revision, self.env.task), reward["score"])
        return BackendStep(
            self.get_state(),
            info,
            reward["score"] - before,
            reward["success"],
            self.env.tick >= self.config.episode_ticks,
        )

    def act_policy(self, action):
        if not isinstance(action, NamedJointPosition):
            raise UnsupportedActionError(
                "Restricted whiteboard interface supports named hardware joints"
            )
        info = self._checked(action, "policy")
        return BackendStep(
            self.get_state(),
            info,
            None,
            False,
            self.env.tick >= self.config.episode_ticks,
        )

    def render(self, *, label="classroom"):
        camera = "classroom" if label == "interactive" else label
        return self._render((camera,))[camera]

    def get_geometry_index(self):
        from whiteboard.geometry_api import GeometryView

        return GeometryView(self.env).describe()

    def get_geometry(self, geometry_id):
        return self.get_geometry_index()["geometries"][int(geometry_id)]

    def get_geometry_asset(self, asset_id):
        from whiteboard.geometry_api import GeometryView

        geometry = GeometryView(self.env)
        result = {}
        for name in ("vertices", "hull_triangles"):
            start, values = 0, []
            while True:
                page = geometry.mesh(int(asset_id), name, start, 10000)
                values.extend(page["values"])
                if page["complete"]:
                    break
                start = page["next_start"]
            result[name] = values
        return result

    def save_checkpoint(self, *, label):
        from whiteboard.storage import snapshot_bytes

        if self.store is None:
            raise RuntimeError("Durable recording is disabled")
        if not isinstance(label, str) or not 1 <= len(label) <= 64:
            raise ValueError("Checkpoint label length must be 1..64")
        self.store.put("checkpoint", label, snapshot_bytes(self.env), encoded=True)
        return {"name": label, "tick": self.env.tick, "durable": True}

    def restore_checkpoint(self, checkpoint):
        from whiteboard.storage import restore_bytes

        if self.store is None:
            raise RuntimeError("Durable recording is disabled")
        state = restore_bytes(
            self.env, self.store.get("checkpoint", checkpoint, raw=True)
        )
        self._score_cache = None
        return state, {"checkpoint": checkpoint, "continuity": "restored_discontinuity"}

    def list_checkpoints(self):
        return (
            tuple({"name": k} for k in self.store.keys("checkpoint"))
            if self.store
            else ()
        )

    def close(self):
        self.env.close()
        if self.store is not None:
            self.store.close()
            self.store = None

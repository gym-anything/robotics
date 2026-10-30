"""Persistent G1 football; shared role-safe HTTP, history and robot actions."""

import base64
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import re
import uuid
from .base import RoboticsBackend, reject_unexecuted
from ..actions import NamedJointPosition, NamedJointTarget
from ..errors import (
    ActionRejectedError,
    UnsupportedConfigurationError,
    UnsupportedActionError,
    BackendProtocolError,
)
from ..policy_access import PolicyAccess
from ..types import BackendStep, Capabilities, CameraSpec, ImageFrame, Observation

# goal's instruction; every task's instruction is football/newton15/variants.INSTRUCTIONS (served per task).
DESCRIPTION = "The football is on the penalty mark in front of the Unitree G1 humanoid. Kick it into the goal."
# Interface fact, not part of the job: served in the policy's capabilities (hardware.control), not in the task text.
CONTROL_NOTE = "No balance or kicking controller is supplied; command the robot's named joints directly."


def same(a, b):
    return tuple(a) == tuple(b) if isinstance(a, (list, tuple)) else a == b


def select_task(config, task, seed, explicit=frozenset()):
    """`config` with the catalogue task's start state (football/newton15/variants.py); refuses to override a start
    field the caller set explicitly to something else."""
    from football.newton15.variants import apply

    selected = apply(config, task, seed)
    clash = sorted(k for k in explicit if hasattr(selected, k) and not same(getattr(config, k), getattr(selected, k)))
    if clash:
        raise ValueError(f"config sets {clash}, which task {task!r} seed {seed} defines; drop them or the selection")
    return selected


class NewtonFootballBackend(RoboticsBackend):
    def __init__(
        self,
        *,
        robot="g1",
        config=None,
        device=None,
        cameras=("g1_rgb",),
        output_dir=None,
        renderer_options=None,
        recording=True,
        task=None,
        seed=None,
    ):
        from football.newton15.config import Config
        from football.newton15.world import FootballWorld
        from football.newton15.catalog import Catalog
        from football.newton15.trace import Trace

        if robot != "g1":
            raise UnsupportedConfigurationError("Football uses the standard G1/Dex3")
        explicit = frozenset(config) if isinstance(config, dict) else frozenset()
        config = Config(**config) if isinstance(config, dict) else config
        config = config or Config(net_layout="enclosure")
        if task is None and seed is None and config.task != "goal":
            task, seed, explicit = config.task, config.seed, explicit - {"task", "seed"}
        if task is not None or seed is not None:
            # Catalogue selection (benchmark.json resolver): the task's start state, props and verifier.
            config = select_task(config, task or config.task, 0 if seed is None else seed, explicit)
        if device is not None:
            config = replace(config, device=device)
        if not config.robot:
            raise ValueError("Robot-free fixtures are not policy environments")
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - {
            "g1_rgb"
        }:
            raise ValueError(
                "Policy camera is g1_rgb; other angles are privileged diagnostic renders"
            )
        self.output = Path(
            output_dir
            or Path.home() / ".cache/football-newton15/sessions" / uuid.uuid4().hex
        ).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.options = dict(renderer_options or {})
        self.env = FootballWorld(config)
        self.catalog = Catalog(self.env)
        self.trace = (
            Trace(self.env, self.output / "physics.sqlite3") if recording else None
        )
        self.renderer = None
        self.images = deque()
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
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
        self.kinds = frozenset({"named_joint_position", "named_joint_target"})
        self.capabilities = Capabilities(
            "football",
            "newton",
            "g1",
            self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True,
            False,
            True,
            self.cameras,
            (self.camera,) if self.cameras else (),
            notes={
                "clock": "One tick = 0.002 s by default; idle does not advance physics",
                "actions": "Named joint targets in radians; empty targets advances with existing targets",
                "control": CONTROL_NOTE,
                "evaluation": "success, score and progress: football/newton15/task.py (goal, free-kick) and "
                "drills.py (the other drills) (owner only)",
                "camera": "Ideal 69-degree RGB pinhole at pinned URDF D435 mount; virtual filmback",
                "checkpoint": "Exact native plus constitutive state; restore starts a new continuity generation",
                "physics": "CPU Newton 1.5 MuJoCo; 20 kHz shoe/pitch, 24 kHz cable subcycles",
                "tasks": "Reset options {'task': <catalogue task>} with the reset seed select a task of "
                "football/newton15/benchmark.json; reset without them restarts the current task",
            },
        )

    @property
    def description(self):
        from football.newton15.variants import INSTRUCTIONS

        return INSTRUCTIONS[self.env.config.task]

    def reset(self, *, seed=None, options=None):
        from football.newton15.variants import validate

        values = dict(options or {})
        if set(values) - {"task"}:
            raise ActionRejectedError("Football reset options are {'task': ...}; the seed is reset's seed")
        current = self.env.config
        task = values.get("task", current.task)
        if seed is None:
            seed = current.seed if task == current.task else 0
        try:
            validate(task, seed)
        except ValueError as error:
            # a refusal that changed nothing: the session stays ready
            raise ActionRejectedError(str(error)) from None
        if self.trace and self.trace.current:
            raise RuntimeError("Cannot reset during action recording")
        config = current if (task, seed) == (current.task, current.seed) else select_task(current, task, seed)
        if self.env.tick or self.env.fault or config != current:
            from football.newton15.world import FootballWorld
            from football.newton15.catalog import Catalog

            if self.renderer:
                self.renderer.close()
                self.renderer = None
            candidate = FootballWorld(config)
            candidate.generation = self.env.generation + 1
            self.env = candidate
            self.catalog = Catalog(candidate)
            if self.trace:
                self.trace.world = candidate
        return self.get_state(), {"continuity": "reset", "task": task, "seed": seed}

    def get_state(self):
        state = self.env.state()
        state["evaluation"] = {**self.env.goal.evaluation(), "instruction": self.description}
        if self.env.config.task != "goal":
            from football.newton15.variants import task_id

            state["evaluation"]["task_id"] = task_id(self.env.config.task, self.env.config.seed)
        return state

    def variation_contract(self):
        from football.newton15.variants import contract

        return dict(**contract(), current={"task": self.env.config.task, "seed": self.env.config.seed})

    def select_variation(self, request):
        from football.newton15.variants import layout

        if set(request) - {"task", "seed"}:
            raise ActionRejectedError("Football selectors are task and seed")
        task, seed = request.get("task", "goal"), request.get("seed", 0)
        try:
            chosen = layout(task, seed)
        except ValueError as error:
            raise ActionRejectedError(str(error)) from None
        return dict(variation={"task": task, "seed": seed, "task_id": chosen["task_id"]},
                    task=chosen["instruction"], reset_request=dict(seed=seed, options={"task": task}))

    def _render(self, names):
        from football.newton15.render_client import Renderer

        with self.env.lock:
            from ..render_lifecycle import renderer_operation
            with renderer_operation(self, "renderer"):
                if self.renderer is None:
                    self.renderer = Renderer(
                        self.env,
                        self.output / ("render_" + uuid.uuid4().hex[:8]),
                        **self.options,
                    )
                result = self.renderer.observe(names)
            frames = {}
            for name, blob in result["images"].items():
                path = (
                    self.output
                    / f"{name}_{self.env.generation}_{self.env.tick}_{uuid.uuid4().hex}.png"
                )
                path.write_bytes(base64.b64decode(blob))
                self.images.append(path)
                frames[name] = path
            while len(self.images) > 64:
                old = self.images.popleft()
                if old.parent != self.output:
                    raise RuntimeError("Image retention path escaped session")
                old.unlink(missing_ok=True)
            return frames

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (
                self.env.tick,
                self.env.generation,
            ):
                raise RuntimeError("Cannot mix stale state with new camera frames")
            frames = self._render(self.cameras) if self.cameras else {}
            proprio = {
                "joints": {
                    n: {
                        k: v
                        for k, v in j.items()
                        if k in ("position_rad", "velocity_rad_s")
                    }
                    for n, j in state["joints"].items()
                }
            }
            return Observation(
                sequence,
                self.env.time,
                "g1",
                proprio,
                {
                    n: ImageFrame(n, path, True, camera=self.camera)
                    for n, path in frames.items()
                },
            )

    def act(self, action):
        # A rejected target (unknown joint, outside the published limits) cancels the command, not the session.
        return reject_unexecuted(lambda: self._act(action), lambda: (self.env.tick, self.env.targets.copy()))

    def _act(self, action):
        if isinstance(action, NamedJointPosition):
            targets, ticks = action.targets_rad, action.ticks
        elif isinstance(action, NamedJointTarget):
            targets, ticks = action.targets_si, action.ticks
        else:
            raise UnsupportedActionError(
                "Football supports named_joint_position and named_joint_target"
            )
        with self.env.lock:
            if self.env.fault:
                raise BackendProtocolError(self.env.fault)
            before = self.env.tick
            self.env.set_joint_targets(targets)
            try:
                self.env.step(ticks)
            except (RuntimeError, FloatingPointError, OSError) as exc:
                raise BackendProtocolError(
                    f"Stopped at tick {self.env.tick}: {exc}"
                ) from exc
            state = self.get_state()
            evaluation = state["evaluation"]
            return BackendStep(
                state,
                {"requested_ticks": ticks, "executed_ticks": self.env.tick - before},
                evaluation["score"],
                evaluation["reward"]["attempt_over"],
            )

    def policy_access(self):
        return PolicyAccess(
            "g1",
            (self.camera,) if self.cameras else (),
            {
                ("joints", n, f): None
                for n in self.env.joints
                for f in ("position_rad", "velocity_rad_s")
            },
            self.kinds,
            lambda: self.description,
            {
                "dt_s": self.env.config.dt,
                "control": CONTROL_NOTE,
                "joints": {
                    n: {k: v for k, v in a.items() if k not in ("q", "dof")}
                    for n, a in self.env.joints.items()
                },
            },
        )

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, False, result.truncated)

    def render(self, *, label="interactive"):
        name = "overview" if label in {"interactive", "http"} else label
        return self._render((name,))[name]

    def get_geometry_index(self):
        return self.catalog.index()

    def get_geometry(self, geometry_id):
        return self.catalog.get(str(geometry_id))

    def get_geometry_asset(self, asset_id):
        return self.catalog.assets[str(asset_id)]

    def begin_action_trace(self, context):
        if not self.trace:
            return {"available": False, "reason": "recording_disabled"}
        try:
            return self.trace.begin(context)
        except Exception as error:
            # Shared HTTP records trace-start errors; this backend additionally
            # requires recording, so do not let the subsequent action advance.
            self.env.fault = (
                f"Recording could not start: {type(error).__name__}: {error}"
            )
            raise

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def _checkpoint_path(self, label):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", label) or label in {".", ".."}:
            raise ValueError(
                "Checkpoint label must be 1..100 plain filename characters"
            )
        folder = self.output / "checkpoints"
        folder.mkdir(exist_ok=True)
        return folder / (label + ".bin")

    def save_checkpoint(self, *, label):
        from football.newton15.checkpoint import save

        with self.env.lock:
            path = self._checkpoint_path(label)
            if path.exists():
                raise FileExistsError(label)
            payload = save(self.env)
            with path.open("xb") as f:
                f.write(payload)
            return {"name": label, "tick": self.env.tick, "bytes": len(payload)}

    def restore_checkpoint(self, checkpoint):
        from football.newton15.checkpoint import restore

        with self.env.lock:
            if self.trace and self.trace.current:
                raise RuntimeError("Cannot restore during an action trace")
            restore(self.env, self._checkpoint_path(checkpoint).read_bytes())
            return self.get_state(), {
                "checkpoint": checkpoint,
                "continuity": "restored_discontinuity",
            }

    def list_checkpoints(self):
        return tuple(
            {"name": p.stem, "bytes": p.stat().st_size}
            for p in sorted((self.output / "checkpoints").glob("*.bin"))
        )

    def close(self):
        if self.trace:
            self.trace.close()
            self.trace = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None

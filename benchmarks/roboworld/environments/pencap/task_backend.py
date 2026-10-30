"""GAR adapter for the catalogue; legacy input/action semantics are inherited.

The constructor follows newton_pencap.py verbatim except task validation and the
TaskPenCapEnv factory. That file is left unchanged.
"""

from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import uuid
from gym_anything_robotics.backends.newton_pencap import NewtonPenCapBackend
from gym_anything_robotics.errors import UnsupportedConfigurationError
from gym_anything_robotics.types import CameraSpec, Capabilities
from .tasks import DEFAULT_TASK, INSTRUCTIONS, validate


class NewtonPenCapTaskBackend(NewtonPenCapBackend):
    def __init__(self, *, robot="dual-panda", config=None, device=None, cameras=("bench_rgb",),
                 output_dir=None, renderer_options=None, recording=True, task="seat-cap"):
        from pencap.newton15.config import Config
        from pencap.task_env import TaskPenCapEnv
        validate(task)
        from pencap.newton15.catalog import Catalog
        from pencap.newton15.kinematics import Kinematics
        if robot != "dual-panda":
            raise UnsupportedConfigurationError("Pen-cap uses two manufacturer-description Panda arms")
        if isinstance(config, dict):
            config = Config(**config)
        config = config or Config(scene="studio")
        if device is not None:
            config = replace(config, device=device)
        if config.dt > 1/960:
            raise UnsupportedConfigurationError("Pen-cap contacts require dt <= 1/960 s; the 1/240 s migration penetrated the shoulder")
        if task != DEFAULT_TASK and config.scene != "studio":
            raise ValueError("The seven new pencap tasks require the studio scene")
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras)-{"bench_rgb"}:
            raise ValueError("Policy camera is bench_rgb; other views are diagnostic render labels")
        if self.cameras and config.scene != "studio":
            raise ValueError("Mounted policy camera requires the studio scene")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir or Path(__file__).resolve().parent/"output"/uuid.uuid4().hex)
        self.output.mkdir(parents=True, exist_ok=False)
        self.env = TaskPenCapEnv(config, task)
        self.catalog, self.ik = Catalog(self.env), Kinematics()
        self.images = deque()
        self.trace = None
        if recording:
            from pencap.newton15.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        focal = .018/math.tan(math.radians(69/2))
        self.camera = CameraSpec("bench_rgb", "floor_stand_D435_enclosure", width, height, 1920, 1080,
            (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        self.capabilities = Capabilities("pencap", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, (self.camera,) if self.cameras else (), notes={
                "clock": "Exact requested ticks; duration must be a whole number of physics ticks. Simulated time "
                         "advances only while a command runs: between commands and after your last command the scene "
                         "is frozen, and the task is judged on the scene as your last command leaves it",
                "control": "Physical servos; IK is not a collision-free path planner",
                "objects": "Actions control the robot only; pen, cap and notebook move through contact",
                "camera": "Ideal pinhole, D435-like RGB FOV; filmback is virtual, not a measured sensor size",
                "gripper": "0..40 mm per finger; fixed baseline 8 N effort limit; no effort action",
                "joint_units": "panda_joint1..7 radians; panda_finger_joint1/2 metres"})

    def policy_access(self):
        return replace(super().policy_access(), instruction=lambda: INSTRUCTIONS[self.env.task])

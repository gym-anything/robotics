"""Gymnasium facade over the same GAR backend; no second physics implementation."""

import gymnasium as gym
import numpy as np
from PIL import Image


class WhiteboardGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *, action_ticks=20, render_mode=None, cameras=("g1_rgb",), **options):
        from gym_anything_robotics.backends.newton_whiteboard import NewtonWhiteboardBackend

        if isinstance(action_ticks, bool) or not isinstance(action_ticks, int) or action_ticks < 1:
            raise ValueError("action_ticks must be a positive integer")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.backend = NewtonWhiteboardBackend(cameras=cameras, **options)
        self.action_ticks, self.render_mode = action_ticks, render_mode
        self.joints = tuple(self.backend.env.robot_joints)
        bounds = self.backend.env.robot_joints
        self.action_space = gym.spaces.Box(
            np.asarray([bounds[n]["lower"] for n in self.joints], np.float64),
            np.asarray([bounds[n]["upper"] for n in self.joints], np.float64),
            dtype=np.float64,
        )
        spaces = {
            "joint_position": gym.spaces.Box(-np.inf, np.inf, (43,), np.float64),
            "joint_velocity": gym.spaces.Box(-np.inf, np.inf, (43,), np.float64),
        }
        for name in self.backend.cameras:
            camera = self.backend.camera
            spaces[name] = gym.spaces.Box(0, 255, (camera.height_px, camera.width_px, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    def _observation(self):
        state = self.backend.env.proprioception()["joints"]
        result = {
            "joint_position": np.asarray([state[n]["position_rad"] for n in self.joints]),
            "joint_velocity": np.asarray([state[n]["velocity_rad_s"] for n in self.joints]),
        }
        if self.backend.cameras:
            for name, path in self.backend._render(self.backend.cameras).items():
                result[name] = np.asarray(Image.open(path).convert("RGB"))
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, info = self.backend.reset(seed=seed, options=options)
        return self._observation(), info

    def step(self, action):
        from gym_anything_robotics import NamedJointPosition

        value = np.asarray(action)
        if not self.action_space.contains(value):
            raise ValueError(
                "Expected 43 finite joint targets within named URDF limits; no clipping"
            )
        step = self.backend.act(
            NamedJointPosition(dict(zip(self.joints, value.tolist())), self.action_ticks)
        )
        return self._observation(), step.reward, step.terminated, step.truncated, dict(step.info)

    def render(self):
        return np.asarray(Image.open(self.backend.render(label="classroom")).convert("RGB"))

    def close(self):
        self.backend.close()


gym.register("Whiteboard-G1-v0", entry_point="whiteboard.gym_env:WhiteboardGym")

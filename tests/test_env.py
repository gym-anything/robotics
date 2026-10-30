from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gym_anything_robotics.actions import CartesianDelta, Gripper
from gym_anything_robotics.backends.base import RoboticsBackend
from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.errors import EnvironmentClosedError
from gym_anything_robotics.types import BackendStep, Capabilities, Observation


class FakeBackend(RoboticsBackend):
    def __init__(self, render_path: Path) -> None:
        self.render_path = render_path
        self.tick = 0
        self.closed = False
        self.actions = []
        self.checkpoints = {}
        self.capabilities = Capabilities(
            environment_id="fake",
            backend="test",
            robot="yam",
            action_types=frozenset(("cartesian_delta", "gripper")),
            observation_channels=frozenset(("joint_positions",)),
            privileged_state=True,
            seeded_reset=True,
            diagnostic_render=True,
        )

    def _state(self):
        return {"tick": self.tick, "secret": {"object_pose": [1, 2, 3]}}

    def reset(self, *, seed=None, options=None):
        self.tick = 0
        return self._state(), {"seed": seed, "options": dict(options or {})}

    def get_state(self):
        return self._state()

    def observation_from_state(self, state, *, sequence):
        return Observation(
            sequence=sequence,
            sim_time_s=float(state["tick"]),
            robot="yam",
            proprioception={"tick": state["tick"]},
            metadata={"privileged_state_excluded": True},
        )

    def act(self, action):
        self.actions.append(action)
        self.tick += 1
        return BackendStep(self._state(), info={"index": self.tick})

    def render(self, *, label="interactive"):
        return self.render_path

    def save_checkpoint(self, *, label):
        self.checkpoints[label] = self.tick
        return {"checkpoint": f"{label}.garckpt", "source_tick": self.tick}

    def restore_checkpoint(self, checkpoint):
        self.tick = self.checkpoints[checkpoint]
        return self._state(), {"checkpoint": checkpoint, "byte_exact": True}

    def list_checkpoints(self):
        return tuple(
            {"name": name, "source_tick": tick}
            for name, tick in self.checkpoints.items()
        )

    def close(self):
        self.closed = True


class RoboticsEnvTest(unittest.TestCase):
    def test_failed_initial_reset_closes_the_backend(self) -> None:
        backend = FakeBackend(Path("unused.png"))

        def fail_reset(*, seed=None, options=None):
            raise RuntimeError("reset failed")

        backend.reset = fail_reset
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            RoboticsEnv(backend)
        self.assertTrue(backend.closed)

    def test_lifecycle_and_information_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(b"png")
            backend = FakeBackend(path)
            env = RoboticsEnv(backend)
            self.assertEqual(env.last_observation.sequence, 0)
            state = env.get_state()
            state["secret"]["object_pose"][0] = 99
            self.assertEqual(env.get_state()["secret"]["object_pose"][0], 1)
            self.assertNotIn("secret", env.obs().to_dict())
            result = env.act(
                CartesianDelta("right", (0.0, 0.0, 0.001)),
            )
            self.assertEqual(result.observation.sequence, 1)
            self.assertEqual(result.info["index"], 1)
            self.assertFalse(env.render().deployable)
            env.close()
            self.assertTrue(backend.closed)
            with self.assertRaises(EnvironmentClosedError):
                env.get_state()

    def test_action_sequence_stops_at_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(Path(directory) / "frame.png")

            def terminal_act(action):
                backend.actions.append(action)
                backend.tick += 1
                return BackendStep(
                    backend._state(), terminated=backend.tick == 1
                )

            backend.act = terminal_act
            env = RoboticsEnv(backend)
            result = env.act(
                (
                    Gripper("right", inward_effort_n=1.0),
                    Gripper("right", inward_effort_n=2.0),
                )
            )
            self.assertTrue(result.terminated)
            self.assertEqual(len(backend.actions), 1)
            env.close()

    def test_checkpoint_save_list_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(Path(directory) / "frame.png")
            env = RoboticsEnv(backend)
            env.act(CartesianDelta("right", (0.0, 0.0, 0.001)))
            saved = env.save_checkpoint(label="grasp")
            env.act(CartesianDelta("right", (0.0, 0.0, 0.001)))

            observation, restored = env.restore_checkpoint("grasp")

            self.assertEqual(saved["source_tick"], 1)
            self.assertEqual(env.list_checkpoints()[0]["name"], "grasp")
            self.assertEqual(observation.sequence, 0)
            self.assertEqual(observation.proprioception["tick"], 1)
            self.assertTrue(restored["byte_exact"])
            env.close()


if __name__ == "__main__":
    unittest.main()

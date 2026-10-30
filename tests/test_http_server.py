from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from gym_anything_robotics.backends.base import RoboticsBackend
from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.errors import ActionRejectedError, BackendUnavailableError
from gym_anything_robotics.http_server import GARHTTPServer, GARHTTPService
from gym_anything_robotics.types import (
    BackendStep,
    CameraSpec,
    Capabilities,
    ImageFrame,
    Observation,
)


class HTTPFakeBackend(RoboticsBackend):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tick = 0
        self.closed = False
        self.actions = []
        self.last_reset = None
        self.saved = {}
        self.fail_observation = False
        self.fail_action = False
        self.nonfinite_state = False
        self.camera_path = root / "camera.png"
        self.camera_path.write_bytes(b"camera-bytes")
        self.render_path = root / "render.png"
        self.render_path.write_bytes(b"render-bytes")
        camera = CameraSpec(
            name="robot_camera",
            mount_frame="wrist",
            width_px=16,
            height_px=9,
            native_width_px=16,
            native_height_px=9,
            sensor_size_m=(0.01, 0.01),
            focal_length_m=(0.01, 0.01),
        )
        self.capabilities = Capabilities(
            environment_id="fake",
            backend="test",
            robot="yam",
            action_types=frozenset(("cartesian_delta", "gripper")),
            observation_channels=frozenset(("joint_positions", "rgb")),
            privileged_state=True,
            seeded_reset=True,
            diagnostic_render=True,
            robot_cameras=(camera.name,),
            camera_specs=(camera,),
        )

    def _state(self):
        return {
            "tick": self.tick,
            "secret_object_pose": (
                [1.0, float("nan"), 3.0]
                if self.nonfinite_state
                else [1.0, 2.0, 3.0]
            ),
        }

    def reset(self, *, seed=None, options=None):
        self.tick = 0
        self.last_reset = {"seed": seed, "options": dict(options or {})}
        return self._state(), dict(self.last_reset)

    def get_state(self):
        return self._state()

    def get_geometry_index(self):
        return {
            "schema_version": "gar.geometry.v1",
            "revision_sha256": "a" * 64,
            "object_count": 1,
            "asset_count": 1,
            "objects": [{"id": "block_06", "shape_count": 2}],
        }

    def get_geometry(self, geometry_id):
        if geometry_id != "block_06":
            raise KeyError(geometry_id)
        return {
            "schema_version": "gar.geometry.v1",
            "id": geometry_id,
            "stud_grid": {"nx": 6, "ny": 12},
            "dimensions_m": {"body": [0.0478, 0.0958, 0.0096]},
            "shapes": [{"geometry": {"asset_id": "mesh-sha256-test"}}],
        }

    def get_geometry_asset(self, asset_id):
        if asset_id != "mesh-sha256-test":
            raise KeyError(asset_id)
        return {
            "id": asset_id,
            "vertices_m": [[0.0, 0.0, 0.0]],
            "triangle_indices": [],
        }

    def observation_from_state(self, state, *, sequence):
        if self.fail_observation:
            raise OSError("camera storage unavailable")
        camera = self.capabilities.camera_specs[0]
        return Observation(
            sequence=sequence,
            sim_time_s=float(state["tick"]),
            robot="yam",
            proprioception={"tick": state["tick"]},
            images={
                camera.name: ImageFrame(
                    name=camera.name,
                    path=self.camera_path,
                    deployable=True,
                    camera=camera,
                )
            },
            metadata={"privileged_state_excluded": True},
        )

    def act(self, action):
        if self.fail_action:
            raise BackendUnavailableError("non-finite rigid state for block_00")
        self.actions.append(action)
        self.tick += 1
        return BackendStep(self._state(), info={"action_index": self.tick}, reward=0.25)

    def render(self, *, label="interactive"):
        return self.render_path

    def save_checkpoint(self, *, label):
        self.saved[label] = self.tick
        return {"name": label, "tick": self.tick}

    def restore_checkpoint(self, checkpoint):
        if checkpoint == "from-another-variation":
            raise ActionRejectedError("checkpoint was saved in another variation")
        self.tick = self.saved[checkpoint]
        return self._state(), {"name": checkpoint, "byte_exact": True}

    def list_checkpoints(self):
        return tuple({"name": name, "tick": tick} for name, tick in self.saved.items())

    def close(self):
        self.closed = True


class HTTPServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.backend = HTTPFakeBackend(Path(self.temporary_directory.name))
        self.env = RoboticsEnv(self.backend)
        self.server = GARHTTPServer(
            ("127.0.0.1", 0),
            GARHTTPService(self.env),
            api_token="test-token",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.root = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.env.close()
        self.temporary_directory.cleanup()

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body=None,
        token: str | None = "test-token",
        idempotency_key: str | None = None,
    ):
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.urlopen(
            urllib.request.Request(
                self.root + path,
                method=method,
                data=data,
                headers=headers,
            ),
            timeout=5,
        )

    @staticmethod
    def json(response):
        return json.loads(response.read().decode("utf-8"))

    def test_help_is_complete_and_authentication_is_required(self) -> None:
        self.assertTrue(self.json(self.request("/v1/health", token=None))["ok"])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/help", token=None)
        self.assertEqual(caught.exception.code, 401)

        help_payload = self.json(self.request("/v1/help"))
        self.assertIn("POST /v1/action", help_payload["routes"])
        self.assertIn("bimanual_cartesian", help_payload["action_types"])
        self.assertEqual(
            help_payload["effective_capabilities"]["environment_id"],
            "fake",
        )

    def test_observation_hides_paths_and_state_is_explicitly_privileged(self) -> None:
        observation = self.json(self.request("/v1/observation"))["observation"]
        image = observation["images"]["robot_camera"]
        self.assertNotIn("path", image)
        self.assertIn("observation_id=", image["url"])
        self.assertNotIn("secret_object_pose", observation["proprioception"])

        state = self.json(self.request("/v1/state"))["state"]
        self.assertEqual(state["secret_object_pose"], [1.0, 2.0, 3.0])

        with self.request(image["url"]) as response:
            self.assertEqual(response.read(), b"camera-bytes")
            self.assertEqual(
                int(response.headers["X-GAR-Observation-ID"]),
                observation["observation_id"],
            )

    def test_geometry_index_object_and_exact_asset_are_exposed(self) -> None:
        index = self.json(self.request("/v1/geometry"))["geometry"]
        self.assertEqual(index["objects"][0]["id"], "block_06")

        geometry = self.json(self.request("/v1/geometry/block_06"))["geometry"]
        self.assertEqual(geometry["stud_grid"], {"nx": 6, "ny": 12})
        self.assertEqual(geometry["dimensions_m"]["body"][1], 0.0958)

        asset = self.json(
            self.request("/v1/geometry/assets/mesh-sha256-test")
        )["asset"]
        self.assertEqual(asset["vertices_m"], [[0.0, 0.0, 0.0]])
        self.assertEqual(asset["triangle_indices"], [])

        help_payload = self.json(self.request("/v1/help"))
        self.assertIn("GET /v1/geometry/{geometry_id}", help_payload["routes"])

    def test_action_is_atomic_and_idempotent_retry_does_not_act_twice(self) -> None:
        action = {
            "type": "cartesian_delta",
            "arm": "right",
            "delta_m": [0.0, 0.0, 0.005],
            "duration_s": 0.75,
        }
        first = self.json(
            self.request(
                "/v1/action",
                method="POST",
                body=action,
                idempotency_key="action-1",
            )
        )
        second = self.json(
            self.request(
                "/v1/action",
                method="POST",
                body=action,
                idempotency_key="action-1",
            )
        )

        self.assertEqual(first, second)
        self.assertEqual(len(self.backend.actions), 1)
        self.assertEqual(first["reward"], 0.25)
        self.assertEqual(first["observation"]["proprioception"]["tick"], 1)

    def test_committed_action_is_never_reported_as_unexecuted(self) -> None:
        self.backend.fail_observation = True
        action = {
            "type": "cartesian_delta",
            "arm": "right",
            "delta_m": [0.0, 0.0, 0.005],
        }

        response = self.json(
            self.request(
                "/v1/action",
                method="POST",
                body=action,
                idempotency_key="committed-without-camera",
            )
        )

        self.assertTrue(response["ok"])
        self.assertTrue(response["committed"])
        self.assertIsNone(response["observation"])
        self.assertEqual(response["observation_error"]["type"], "OSError")
        self.assertEqual(self.backend.tick, 1)
        self.assertFalse(self.json(self.request("/v1/health", token=None))["ok"])

        self.backend.fail_observation = False
        self.json(self.request("/v1/observation"))
        self.assertTrue(self.json(self.request("/v1/health", token=None))["ok"])

    def test_refused_restore_keeps_session_ready(self) -> None:
        # a restore refused before touching the simulator (ActionRejectedError) must not latch a runtime fault
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/checkpoints/restore", method="POST", body={"checkpoint": "from-another-variation"})
        self.assertEqual(caught.exception.code, 400)
        health = self.json(self.request("/v1/health", token=None))
        self.assertTrue(health["ok"])
        self.assertTrue(health["runtime_ok"])
        response = self.json(self.request("/v1/action", method="POST",
                                          body={"type": "cartesian_delta", "arm": "left", "delta_m": [0.0, 0.0, 0.005]}))
        self.assertTrue(response["committed"])

    def test_runtime_failure_marks_health_unready_until_reset(self) -> None:
        self.backend.fail_action = True
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                "/v1/action",
                method="POST",
                body={
                    "type": "cartesian_delta",
                    "arm": "left",
                    "delta_m": [0.0, 0.0, 0.005],
                },
            )
        self.assertEqual(caught.exception.code, 502)

        health = self.json(self.request("/v1/health", token=None))
        self.assertFalse(health["ok"])
        self.assertFalse(health["runtime_ok"])
        self.assertEqual(health["runtime_fault_type"], "BackendUnavailableError")

        self.backend.fail_action = False
        self.json(self.request("/v1/reset", method="POST", body={}))
        self.assertTrue(self.json(self.request("/v1/health", token=None))["ok"])

    def test_nonfinite_state_serialization_marks_health_unready(self) -> None:
        self.backend.nonfinite_state = True
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/state")
        self.assertEqual(caught.exception.code, 400)

        health = self.json(self.request("/v1/health", token=None))
        self.assertFalse(health["ok"])
        self.assertFalse(health["runtime_ok"])
        self.assertEqual(health["runtime_fault_type"], "ValueError")

    def test_reset_preserves_exact_seed_and_options(self) -> None:
        response = self.json(
            self.request(
                "/v1/reset",
                method="POST",
                body={
                    "seed": 17,
                    "options": {"dataset_count": 10000, "episode_index": 42},
                },
            )
        )

        self.assertEqual(
            self.backend.last_reset,
            {
                "seed": 17,
                "options": {"dataset_count": 10000, "episode_index": 42},
            },
        )
        self.assertEqual(response["info"], self.backend.last_reset)

    def test_checkpoint_and_diagnostic_render_endpoints(self) -> None:
        self.json(
            self.request(
                "/v1/checkpoints/save",
                method="POST",
                body={"label": "grasp"},
            )
        )
        checkpoints = self.json(self.request("/v1/checkpoints"))["checkpoints"]
        self.assertEqual(checkpoints[0]["name"], "grasp")
        restored = self.json(
            self.request(
                "/v1/checkpoints/restore",
                method="POST",
                body={"checkpoint": "grasp"},
            )
        )
        self.assertTrue(restored["info"]["byte_exact"])
        with self.request("/v1/render") as response:
            self.assertEqual(response.read(), b"render-bytes")
            self.assertEqual(response.headers["X-GAR-Deployable"], "false")


if __name__ == "__main__":
    unittest.main()

"""Real loopback HTTP, actual Newton state; no render/physics mocks."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
from whiteboard.env import WhiteboardEnv
from whiteboard.config import Config
from whiteboard.api import Service, make_server


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WhiteboardEnv(
            Config(
                robot=False,
                furniture=False,
                texture_width=256,
                texture_height=128,
                episode_ticks=100,
            )
        )
        cls.service = Service(cls.env, "teacher-test-key", "policy-test-key")
        cls.server = make_server(cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.service.env.close()

    def setUp(self):
        self.env.reset()
        self.service.commands.clear()

    def request(self, path, body=None, key="policy-test-key"):
        req = Request(
            f"http://127.0.0.1:{self.server.server_port}" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as r:
                return r.status, json.load(r)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def action(self, **updates):
        return {
            "id": "test",
            "episode_id": self.service.episode,
            "targets": {},
            "ticks": 2,
            **updates,
        }

    def test_unauthorized(self):
        self.assertEqual(self.request("/v1/health", key="wrong")[0], 401)

    def test_all_privileged_routes_forbidden(self):
        for route in (
            "state",
            "geometry",
            "geometry/mesh",
            "evaluate",
            "history",
            "history/trace",
            "checkpoints/save",
            "checkpoints/restore",
            "variation",
        ):
            for body in (None, {}):
                with self.subTest(route=route, body=body):
                    self.assertEqual(self.request("/v1/" + route, body)[0], 403)

    def test_help_and_policy_camera_gate(self):
        _, help = self.request("/v1/help")
        self.assertNotIn("GET /v1/state", help["routes"])
        self.assertEqual(self.request("/v1/observe", {"cameras": ["classroom"]})[0], 403)
        self.assertIsNone(self.env._renderer)

    def test_invalid_teacher_camera_does_not_launch_renderer(self):
        for cameras in ([], ["not-a-camera"], ["board", "board"]):
            self.assertEqual(
                self.request("/v1/observe", {"cameras": cameras}, key="teacher-test-key")[0], 400
            )
        self.assertIsNone(self.env._renderer)

    def test_corrupt_history_remains_available(self):
        q = self.env.state_0.body_q.numpy()
        q[0, 0] = np.nan
        self.env.state_0.body_q.assign(q)
        self.env.healthy = False
        self.assertEqual(self.request("/v1/action", self.action())[0], 503)
        status, history = self.request("/v1/history", key="teacher-test-key")
        self.assertEqual(status, 200)
        self.assertTrue(history["entries"][-1]["after"]["nonfinite_fields"])

    def test_geometry_description(self):
        status, geometry = self.request("/v1/geometry", key="teacher-test-key")
        self.assertEqual(status, 200)
        self.assertTrue(geometry)

    def test_checkpoint_restore_and_stale_episode(self):
        self.assertEqual(
            self.request("/v1/checkpoints/save", {"name": "origin"}, key="teacher-test-key")[0], 200
        )
        old = self.service.episode
        self.request("/v1/action", self.action())
        status, result = self.request(
            "/v1/checkpoints/restore",
            {"name": "origin", "id": "restore", "episode_id": old},
            key="teacher-test-key",
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["tick"], 0)
        self.assertNotEqual(result["episode_id"], old)

    def test_invalid_actions_never_mutate(self):
        before = self.env.state_0.body_q.numpy().copy()
        for change in (
            {"ticks": 0},
            {"ticks": True},
            {"ticks": 4001},
            {"ticks": 101},
            {"targets": []},
            {"targets": {"foo": 1}},
            {"episode_id": []},
            {"extra": 1},
            {"targets": {"foo": float("nan")}},
        ):
            with self.subTest(change=change):
                self.assertEqual(self.request("/v1/action", self.action(**change))[0], 400)
                self.assertEqual(self.env.tick, 0)
                np.testing.assert_array_equal(before, self.env.state_0.body_q.numpy())

    def test_idempotent_action_and_conflict(self):
        first = self.request("/v1/action", self.action())
        second = self.request("/v1/action", self.action())
        self.assertEqual(first, second)
        self.assertEqual(first[0], 200)
        self.assertEqual(self.env.tick, 2)
        self.assertEqual(self.request("/v1/action", self.action(ticks=3))[0], 409)

    def test_horizon_is_explicit_without_partial_action(self):
        status, result = self.request("/v1/action", self.action(ticks=100))
        self.assertEqual(status, 200)
        self.assertTrue(result["truncated"])
        self.assertEqual(self.request("/v1/action", self.action(id="past-end", ticks=1))[0], 400)
        self.assertEqual(self.env.tick, 100)

    def test_concurrent_same_command_runs_once(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.request("/v1/action", self.action()), range(4)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(self.env.tick, 2)

    def test_trace_every_tick_and_no_read_advance(self):
        sequence = self.env.trace.sequence
        _, result = self.request("/v1/action", self.action(ticks=5))
        status, page = self.request(
            "/v1/history/trace", {"after": sequence, "limit": 10}, key="teacher-test-key"
        )
        self.assertEqual(status, 200)
        self.assertEqual([r["tick"] for r in page["frames"]], [1, 2, 3, 4, 5])
        self.assertTrue(all(r["command_id"] == result["command_id"] for r in page["frames"]))
        self.assertEqual(self.env.tick, 5)
        self.assertEqual(page["failed_frames"], 0)

    def test_reset_is_idempotent_and_old_episode_stale(self):
        body = {"id": "reset", "episode_id": self.service.episode}
        first = self.request("/v1/reset", body)
        self.assertEqual(first, self.request("/v1/reset", body))
        self.assertNotEqual(body["episode_id"], first[1]["episode_id"])
        self.assertEqual(
            self.request("/v1/action", self.action(episode_id=body["episode_id"]))[0], 409
        )

    def test_public_target_not_current_ink(self):
        status, result = self.request("/v1/task")
        self.assertEqual(status, 200)
        self.assertIn("reference_png", result)
        self.assertNotIn("ink", result)

    def test_failed_state_requires_explicit_recovery(self):
        self.env.healthy = False
        status, result = self.request("/v1/action", self.action())
        self.assertEqual(status, 503)
        self.assertFalse(result["ready"])
        self.assertEqual(result["ticks_executed"], 0)
        self.assertEqual(self.request("/v1/health")[1]["ready"], False)

    def test_nonfinite_state_is_structured_failure(self):
        q = self.env.state_0.body_q.numpy()
        q[0, 0] = np.nan
        self.env.state_0.body_q.assign(q)
        status, result = self.request("/v1/state", key="teacher-test-key")
        self.assertEqual(status, 503)
        self.assertFalse(result["ready"])


if __name__ == "__main__":
    unittest.main()

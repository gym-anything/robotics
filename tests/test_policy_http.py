"""Generic credential boundary checks over a fake robot backend."""
from dataclasses import replace
import json
import urllib.error
from gym_anything_robotics.access import APIKeys
from gym_anything_robotics.policy_access import PolicyAccess, ROBOT_ACTIONS
from tests.test_http_server import HTTPServerTest
from gym_anything_robotics.types import ImageFrame

SECRET = "HIDDEN-object-pose-connection-mesh-seed"
DELTA = {"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, 0.001]}


class PolicyHTTPTest(HTTPServerTest):
    """Real localhost HTTP, with deliberately contaminated backend responses."""

    def setUp(self):
        super().setUp()
        camera = self.backend.capabilities.camera_specs[0]
        self.profile = PolicyAccess(
            robot="yam", cameras=(camera,),
            proprioception={("joint_positions_rad", "left"): 6},
            action_types=ROBOT_ACTIONS,
            instruction=lambda: "Place the red target brick in the red bin.",
            hardware={"frame": "station"},
        )
        self.backend.policy_access = lambda: self.profile
        self.backend.act_policy = self.backend.act
        self.server.keys = APIKeys("test-token", "policy-token")
        original = self.backend.observation_from_state

        def observation(state, *, sequence):
            obs = original(state, sequence=sequence)
            return replace(obs,
                proprioception={**obs.proprioception, "joint_positions_rad": {
                    "left": [0.0] * 6, "secret": SECRET}, "hidden": SECRET},
                metadata={"hidden": SECRET},
                images={**obs.images, "free": ImageFrame(
                    "free", self.backend.render_path, False,
                )},
            )
        self.backend.observation_from_state = observation

    def policy(self, path, **kwargs):
        return self.request(path, token="policy-token", **kwargs)

    def denied(self, path, *, status=403, **kwargs):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.policy(path, **kwargs)
        self.assertEqual(caught.exception.code, status)
        body = caught.exception.read().decode()
        self.assertNotIn(SECRET, body)
        return body

    def test_policy_help_and_capabilities_are_allowlisted(self):
        help = self.json(self.policy("/v1/help"))
        self.assertEqual(help["role"], "policy")
        self.assertNotIn("GET /v1/state", help["routes"])
        self.assertNotIn("raw", help["action_types"])
        self.assertNotIn("safety", help["action_types"])
        self.assertNotIn("variation_selection", help)
        self.assertNotIn(SECRET, json.dumps(help))
        caps = self.json(self.policy("/v1/capabilities"))["capabilities"]
        self.assertFalse(caps["privileged_state"])
        self.assertEqual(caps["robot_cameras"], ["robot_camera"])
        self.assertNotIn("notes", caps)
        self.assertEqual(self.json(self.policy("/v1/task"))["instruction"], self.profile.instruction())

    def test_task_pictures_are_listed_and_served_only_when_the_task_has_them(self):
        self.assertEqual(set(self.json(self.policy("/v1/task"))), {"ok", "instruction"})
        self.assertNotIn("GET /v1/task/images/{name}", self.json(self.policy("/v1/help"))["routes"])
        self.denied("/v1/task/images/reference", status=404)
        self.profile = replace(self.profile, task_images=lambda: {"reference": ("Drawing to copy", b"png-bytes")})
        self.assertEqual(self.json(self.policy("/v1/task"))["reference_images"], {"reference": {
            "url": "/v1/task/images/reference", "mime_type": "image/png", "description": "Drawing to copy"}})
        self.assertIn("GET /v1/task/images/{name}", self.json(self.policy("/v1/help"))["routes"])
        with self.policy("/v1/task/images/reference") as response:
            self.assertEqual(response.headers["Content-Type"], "image/png")
            self.assertEqual(response.read(), b"png-bytes")
        self.denied("/v1/task/images/other", status=404)

    def test_all_privileged_routes_are_denied_without_mutation(self):
        for path in ("/v1/state", "/v1/geometry", "/v1/geometry/block_06",
                     "/v1/geometry/assets/mesh-sha256-test", "/v1/render",
                     "/v1/checkpoints", "/v1/variations", "/v1/unknown",
                     "/v1/%73tate", "/v1/state?role=privileged"):
            with self.subTest(path=path):
                self.denied(path)
        for path in ("/v1/reset", "/v1/checkpoints/save", "/v1/checkpoints/restore",
                     "/v1/variations/select", "/v1/variations/start"):
            with self.subTest(path=path):
                self.denied(path, method="POST", body={})
        self.assertEqual(self.backend.tick, 0)
        self.assertFalse(self.backend.actions)
        self.assertFalse(self.backend.saved)

    def test_observation_strips_nested_metadata_and_debug_images(self):
        obs = self.json(self.policy("/v1/observation"))["observation"]
        self.assertNotIn(SECRET, json.dumps(obs))
        self.assertEqual(obs["metadata"], {})
        self.assertEqual(obs["proprioception"], {"joint_positions_rad": {"left": [0] * 6}})
        self.assertEqual(set(obs["images"]), {"robot_camera"})
        with self.policy(obs["images"]["robot_camera"]["url"]) as response:
            self.assertEqual(response.read(), b"camera-bytes")
        self.denied("/v1/cameras/free?observation_id=" + str(obs["observation_id"]), status=400)

    def test_policy_cannot_fetch_privileged_or_other_credential_observations(self):
        admin = self.json(self.request("/v1/observation"))["observation"]
        self.denied(admin["images"]["robot_camera"]["url"])
        self.denied(admin["images"]["free"]["url"])
        self.denied("/v1/cameras/robot_camera")
        own = self.json(self.policy("/v1/observation"))["observation"]
        self.server.keys = APIKeys("test-token", "replacement-policy")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(own["images"]["robot_camera"]["url"], token="replacement-policy")
        self.assertEqual(caught.exception.code, 403)

    def test_action_filter_and_idempotency_are_credential_scoped(self):
        admin = self.json(self.request("/v1/action", method="POST", body=DELTA,
                                       idempotency_key="shared"))
        self.assertIn("reward", admin)
        first = self.json(self.policy("/v1/action", method="POST", body=DELTA,
                                      idempotency_key="shared"))
        second = self.json(self.policy("/v1/action", method="POST", body=DELTA,
                                       idempotency_key="shared"))
        self.assertEqual(first, second)
        self.assertEqual(len(self.backend.actions), 2)
        self.assertEqual(set(first), {"ok", "committed", "observation", "request_id"})
        self.assertNotIn(SECRET, json.dumps(first))
        self.assertFalse(self.backend.actions[-1].safety.stop_on_arm_arm_contact)

    def test_whole_batch_is_validated_before_execution(self):
        self.denied("/v1/action", method="POST", body={"actions": [DELTA,
                    {"type": "raw", "request": {"op": "policy_next"}}]})
        self.assertEqual(len(self.backend.actions), 0)

    def test_hidden_contact_guards_and_spoofed_roles_rejected(self):
        for safety in ({"stop_on_contact": True}, {"stop_on_robot_contact": True},
                       {"stop_on_arm_arm_contact": True}, {"stop_on_arm_arm_contact": "false"},
                       {"stop_on_grip_loss": "block_06"}, {"hidden": False}):
            self.denied("/v1/action", method="POST", body={**DELTA, "safety": safety})
        self.denied("/v1/action", method="POST", body={**DELTA, "role": "privileged"}, status=400)
        self.denied("/v1/action", method="POST", body={
            "type": "bimanual_cartesian", "left": {"delta_m": [0, 0, 0]},
            "right": {"delta_m": [0, 0, 0]}, "stop_on_arm_arm_contact": True,
        })
        self.assertFalse(self.backend.actions)

    def test_hidden_success_does_not_shorten_policy_batch(self):
        act = self.backend.act
        self.backend.act_policy = lambda action: replace(act(action), terminated=True, reward=1.0)
        result = self.json(self.policy("/v1/action", method="POST", body={"actions": [DELTA] * 3}))
        self.assertEqual(len(self.backend.actions), 3)
        self.assertNotIn("terminated", result)

    def test_backend_failure_and_capture_failure_do_not_leak(self):
        self.backend.fail_action = True
        body = self.denied("/v1/action", method="POST", body=DELTA, status=502)
        self.assertNotIn("block_00", body)
        self.backend.fail_action = False
        self.request("/v1/recover", method="POST", body={"acknowledge_uncertain_actions":True})
        self.backend.fail_observation = True
        result = self.json(self.policy("/v1/action", method="POST", body=DELTA,
                                       idempotency_key="capture-fail"))
        self.assertTrue(result["committed"])
        self.assertIsNone(result["observation"])
        self.assertNotIn("info", result)
        self.assertNotIn("reward", result)
        self.assertEqual(self.json(self.policy("/v1/action", method="POST", body=DELTA,
                                               idempotency_key="capture-fail")), result)
        self.assertEqual(len(self.backend.actions), 1)

    def test_projection_failure_after_motion_preserves_commit_semantics(self):
        self.profile = replace(self.profile, proprioception={("nonexistent",): 6})
        result = self.json(self.policy("/v1/action", method="POST", body=DELTA,
                                       idempotency_key="projection-fail"))
        self.assertTrue(result["committed"])
        self.assertIsNone(result["observation"])
        self.assertNotIn("nonexistent", json.dumps(result))
        self.assertEqual(self.json(self.policy("/v1/action", method="POST", body=DELTA,
                                               idempotency_key="projection-fail")), result)
        self.assertEqual(len(self.backend.actions), 1)

    def test_bad_deployable_flag_fails_closed(self):
        original = self.backend.observation_from_state
        def observation(state, *, sequence):
            obs = original(state, sequence=sequence)
            return replace(obs, images={"robot_camera": replace(obs.images["robot_camera"], deployable=False)})
        self.backend.observation_from_state = observation
        self.denied("/v1/observation", status=400)

    def test_unauthenticated_camera_and_wrong_keys_stay_denied(self):
        for token in (None, "wrong", "policy-token-suffix"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/v1/cameras/robot_camera", token=token)
            self.assertEqual(caught.exception.code, 401)

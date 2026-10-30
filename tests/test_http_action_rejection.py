"""Pre-execution rejection must not latch the shared HTTP runtime fault."""
import unittest
import urllib.error
from unittest.mock import patch

from gym_anything_robotics.access import APIKeys
from gym_anything_robotics.errors import ActionRejectedError, BackendProtocolError, UnsupportedActionError
from gym_anything_robotics.policy_access import PolicyAccess
from tests import test_http_server as helpers


COMMAND = {"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]}


class ActionRejectionHTTPTest(unittest.TestCase):
    tearDown = helpers.HTTPServerTest.tearDown
    request = helpers.HTTPServerTest.request
    json = staticmethod(helpers.HTTPServerTest.json)

    def setUp(self):
        helpers.HTTPServerTest.setUp(self)
        self.server.keys = APIKeys("test-token", "policy-token")
        self.backend.policy_access = lambda: PolicyAccess(
            "yam", self.backend.capabilities.camera_specs, {("tick",): None},
            frozenset({"cartesian_delta"}), lambda: "Test robot control.", {},
        )
        self.backend.act_policy = lambda action: self.backend.act(action)

    def rejected(self, body, *, token="test-token", status=400):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/action", method="POST", body=body, token=token)
        self.assertEqual(caught.exception.code, status)
        return self.json(caught.exception)

    def test_rejected_command_preserves_state_and_next_action_works_for_both_roles(self):
        for token in ("test-token", "policy-token"):
            with self.subTest(token=token):
                before = self.env.get_state()
                with patch.object(self.backend, "act", side_effect=ActionRejectedError(
                    "IK target not reached: 6.50 mm; secret-internal-detail; no action executed"
                )):
                    error = self.rejected(COMMAND, token=token)
                self.assertFalse(error["committed"])
                self.assertEqual(error["error_type"], "ActionRejectedError")
                if token == "policy-token":
                    self.assertNotIn("secret-internal-detail", error["error"])
                self.assertEqual(self.env.get_state(), before)
                self.assertTrue(self.json(self.request("/v1/health"))["ready"])
                records = self.server.service.history.store.list()["records"]
                record = self.server.service.history.store.get(records[-1]["id"])
                self.assertEqual(record["before"]["state"], record["after"]["state"])
                self.assertEqual(record["error"]["type"], "ActionRejectedError")
                result = self.json(self.request("/v1/action", method="POST", body=COMMAND, token=token))
                self.assertTrue(result["committed"])
                self.assertEqual(self.backend.tick, before["tick"] + 1)

    def test_first_command_rejection_in_batch_does_not_run_later_commands(self):
        with patch.object(self.backend, "act", side_effect=ActionRejectedError("IK rejected")) as act:
            result = self.rejected({"actions": [COMMAND, COMMAND]}, token="policy-token")
        self.assertFalse(result["committed"])
        self.assertEqual(act.call_count, 1)
        self.assertEqual(self.backend.tick, 0)
        self.assertTrue(self.server.service.health()["ready"])

    def test_unsupported_commands_are_nonfatal_for_both_roles(self):
        for token in ("test-token", "policy-token"):
            with self.subTest(token=token):
                before = self.env.get_state()
                with patch.object(self.backend, "act", side_effect=UnsupportedActionError("secret-detail")):
                    error = self.rejected(COMMAND, token=token, status=422)
                self.assertFalse(error["committed"])
                self.assertEqual(error["error_type"], "UnsupportedActionError")
                if token == "policy-token":
                    self.assertNotIn("secret-detail", error["error"])
                self.assertEqual(self.env.get_state(), before)
                self.assertTrue(self.server.service.health()["ready"])
                result = self.json(self.request("/v1/action", method="POST", body=COMMAND, token=token))
                self.assertTrue(result["committed"])

    def test_first_unsupported_command_in_batch_leaves_server_ready(self):
        with patch.object(self.backend, "act", side_effect=UnsupportedActionError("unsupported")) as act:
            self.rejected({"actions": [COMMAND, COMMAND]}, status=422)
        self.assertEqual(act.call_count, 1)
        self.assertEqual(self.backend.tick, 0)
        self.assertTrue(self.server.service.health()["ready"])


    def test_rejected_reset_preserves_state_and_existing_fault_status(self):
        self.backend.tick = 42
        for existing_fault in (False, True):
            with self.subTest(existing_fault=existing_fault):
                if existing_fault:
                    self.server.service._record_runtime_fault(BackendProtocolError("existing failure"))
                with patch.object(self.backend, "reset", side_effect=ActionRejectedError("unsupported seed")):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        self.request("/v1/reset", method="POST", body={"seed": 7})
                self.assertEqual(caught.exception.code, 400)
                self.assertFalse(self.json(caught.exception)["committed"])
                self.assertEqual(self.backend.tick, 42)
                self.assertEqual(self.server.service.health()["ready"], not existing_fault)


    def test_unknown_value_error_still_faults_even_without_tick_change(self):
        with patch.object(self.backend, "act", side_effect=ValueError("unclassified runtime failure")):
            self.rejected(COMMAND)
        self.assertFalse(self.server.service.health()["ready"])
        self.rejected(COMMAND, status=502)

    def test_post_step_failure_still_faults(self):
        def fail(action):
            self.backend.tick += 1
            raise BackendProtocolError("physics failed after step")

        with patch.object(self.backend, "act", side_effect=fail):
            self.rejected(COMMAND, status=502)
        self.assertFalse(self.server.service.health()["ready"])
        self.assertEqual(self.backend.tick, 1)
        self.rejected(COMMAND, status=502)
        self.assertEqual(self.backend.tick, 1)


if __name__ == "__main__":
    unittest.main()

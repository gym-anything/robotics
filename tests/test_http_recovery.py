"""Owner reconciliation of the generic HTTP service, without a simulator."""
import unittest
import urllib.error
from unittest.mock import MagicMock, patch
from gym_anything_robotics.access import APIKeys
from gym_anything_robotics.errors import BackendTimeoutError, BackendUnavailableError
from gym_anything_robotics import http_server
from tests import test_http_server as helpers


class RecoveryHTTPTest(unittest.TestCase):
    setUp = helpers.HTTPServerTest.setUp
    tearDown = helpers.HTTPServerTest.tearDown
    request = helpers.HTTPServerTest.request
    json = staticmethod(helpers.HTTPServerTest.json)
    acknowledgement = {"acknowledge_uncertain_actions":True}

    def test_owner_recovery_preserves_state_and_history_without_motion(self):
        initial_reset = self.backend.last_reset
        self.backend.tick = 53
        self.server.service._record_runtime_fault(BackendTimeoutError("uncertain old command"))
        self.assertFalse(self.json(self.request("/v1/health"))["ready"])
        # State retrieval alone must not silently authorize another action.
        self.request("/v1/state")
        self.assertFalse(self.json(self.request("/v1/health"))["ready"])
        response = self.json(self.request("/v1/recover", method="POST", body=self.acknowledgement))
        self.assertTrue(response["health"]["ready"])
        self.assertFalse(response["reset_performed"])
        self.assertEqual(response["state"]["tick"], 53)
        self.assertEqual(self.backend.tick, 53)
        self.assertEqual(self.backend.actions, [])
        self.assertIs(self.backend.last_reset, initial_reset)
        records = self.json(self.request("/v1/history"))["records"]
        self.assertEqual(records[-1]["kind"], "owner_recovery")

    def test_acknowledgement_and_privileged_key_required(self):
        self.server.keys = APIKeys("test-token", "policy-token")
        for body in ({}, {"acknowledge_uncertain_actions":False}, {"acknowledge_uncertain_actions":1}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/v1/recover", method="POST", body=body)
            self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/recover", method="POST", body=self.acknowledgement, token="policy-token")
        self.assertEqual(caught.exception.code, 403)
        self.assertEqual(self.backend.actions, [])

    def test_new_motion_blocked_until_recovery_and_not_replayed_by_recovery(self):
        self.server.service._record_runtime_fault(BackendTimeoutError("uncertain old command"))
        body={"type":"cartesian_delta","arm":"left","delta_m":[0,0,.001]}
        with self.assertRaises(urllib.error.HTTPError):
            self.request("/v1/action", method="POST", body=body)
        self.assertEqual(self.backend.actions, [])
        self.request("/v1/recover", method="POST", body=self.acknowledgement)
        self.assertEqual(self.backend.actions, [])
        self.request("/v1/action", method="POST", body=body)
        self.assertEqual(len(self.backend.actions), 1)

    def test_bad_state_camera_or_history_does_not_become_ready(self):
        for failure in ("nan", "finite_flag", "dead", "camera", "history"):
            with self.subTest(failure=failure):
                self.backend.nonfinite_state = failure == "nan"
                self.backend.fail_observation = failure == "camera"
                self.server.service.history.recording_fault = "disk failure" if failure == "history" else None
                def state():
                    if failure == "dead":
                        raise BackendUnavailableError("controller dead")
                    return {**self.backend._state(), "finite":failure != "finite_flag"}
                with patch.object(self.backend, "get_state", side_effect=state):
                    with self.assertRaises(urllib.error.HTTPError):
                        self.request("/v1/recover", method="POST", body=self.acknowledgement)
                self.assertFalse(self.json(self.request("/v1/health"))["ready"])
                self.assertEqual(self.backend.actions, [])

    def test_disconnect_does_not_turn_a_completed_action_into_runtime_fault(self):
        handler = object.__new__(http_server.GARHTTPHandler)
        handler.send_response = MagicMock(side_effect=ConnectionAbortedError("reader left"))
        handler._send_json({"ok":True,"committed":True})
        self.assertTrue(handler.close_connection)

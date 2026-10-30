from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

from gym_anything_robotics.access import APIKeys
from gym_anything_robotics.action_history import ActionHistory, HistoryRecorder, HistoryUnavailableError
from gym_anything_robotics.errors import BackendUnavailableError
from gym_anything_robotics.policy_access import PolicyAccess, ROBOT_ACTIONS
from tests.test_http_server import HTTPServerTest


DELTA = {"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]}


class HistoryHTTPTest(HTTPServerTest):
    def setUp(self):
        super().setUp()
        profile = PolicyAccess(robot="yam", cameras=self.backend.capabilities.camera_specs,
                               proprioception={("tick",): 1}, action_types=ROBOT_ACTIONS,
                               instruction=lambda: "Move the brick.", hardware={})
        self.backend.policy_access = lambda: profile
        self.backend.act_policy = self.backend.act
        self.server.keys = APIKeys("test-token", "policy-token")

    def records(self, query=""):
        summaries = self.json(self.request("/v1/history" + query))["records"]
        return [self.json(self.request("/v1/history/" + row["id"]))["record"] for row in summaries]

    def action(self, payload=DELTA, **kwargs):
        return self.json(self.request("/v1/action", method="POST", body=payload, **kwargs))

    def test_atomic_batch_state_id_and_idempotent_retries(self):
        result = self.action({"actions": [DELTA] * 3}, token="policy-token", idempotency_key="three")
        rows = self.records("?request_id=" + result["request_id"])
        self.assertEqual([r["action_index"] for r in rows], [0, 1, 2])
        self.assertEqual([r["before"]["state"]["tick"] for r in rows], [0, 1, 2])
        self.assertEqual([r["after"]["state"]["tick"] for r in rows], [1, 2, 3])
        self.assertEqual(len({r["id"] for r in rows}), 3)
        for row in rows:
            self.assertEqual(row["role"], "policy")
            self.assertEqual(row["command"], DELTA)
            self.assertEqual(row["outcome"], "completed")
            self.assertLessEqual(row["started_at"], row["finished_at"])
            self.assertIn("secret_object_pose", row["after"]["state"])
        repeated = self.action({"actions": [DELTA] * 3}, token="policy-token", idempotency_key="three")
        self.assertEqual(result, repeated)
        self.assertEqual(len(self.records()), 3)
        self.assertEqual(len(self.backend.actions), 3)

    def test_denied_routes_no_leaks_even_with_known_id(self):
        result = self.action(token="policy-token")
        row = self.records()[0]
        self.assertNotIn("secret_object_pose", json.dumps(result))
        for path in ("/v1/history", "/v1/history/status", "/v1/history/" + row["id"],
                     "/v1/history?request_id=" + result["request_id"]):
            for token, status in (("policy-token", 403), (None, 401)):
                with self.subTest(path=path, token=token):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        self.request(path, token=token)
                    self.assertEqual(caught.exception.code, status)
                    self.assertNotIn("secret_object_pose", caught.exception.read().decode())
        help = self.json(self.request("/v1/help", token="policy-token"))
        self.assertNotIn("history", json.dumps(help["routes"]))

    def test_failure_retains_partial_state_and_error_correlation(self):
        original = self.backend.act
        def fail_second(action):
            result = original(action)
            if self.backend.tick == 2:
                self.backend.nonfinite_state = True
                raise BackendUnavailableError("invalid state for hidden brick")
            return result
        self.backend.act_policy = fail_second
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.action({"actions": [DELTA] * 3}, token="policy-token")
        self.assertEqual(caught.exception.code, 502)
        request_id = caught.exception.headers["X-GAR-Request-ID"]
        rows = self.records("?request_id=" + request_id)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["outcome"], "failed")
        self.assertEqual(rows[-1]["after"]["state"]["tick"], 2)
        self.assertEqual(rows[-1]["after"]["nonfinite_paths"], ["/secret_object_pose/1"])
        self.assertEqual(rows[-1]["after"]["state"]["secret_object_pose"][1], "NaN")
        self.assertIn("hidden brick", rows[-1]["error"]["message"])
        self.assertNotIn("hidden brick", caught.exception.read().decode())

    def test_unreachable_after_state_preserves_before_and_original_failure(self):
        def fail(_action):
            self.backend.get_state = lambda: (_ for _ in ()).throw(OSError("controller died"))
            raise BackendUnavailableError("original command failure")
        self.backend.act_policy = fail
        with self.assertRaises(urllib.error.HTTPError):
            self.action(token="policy-token")
        row = self.records()[0]
        self.assertEqual(row["before"]["state"]["tick"], 0)
        self.assertIsNone(row["after"]["state"])
        self.assertEqual(row["after"]["error"]["message"], "controller died")
        self.assertEqual(row["error"]["message"], "original command failure")

    def test_failed_pre_capture_blocks_action_and_invalid_batch_creates_no_records(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.action({"actions": [DELTA, {"type": "not_a_command"}]})
        self.assertEqual(self.records(), [])
        self.backend.nonfinite_state = True
        with self.assertRaises(urllib.error.HTTPError):
            self.action()
        self.assertEqual(self.backend.actions, [])
        self.assertEqual(self.records()[0]["outcome"], "not_executed")

    def test_observation_failure_does_not_erase_physical_action_record(self):
        self.backend.fail_observation = True
        result = self.action(token="policy-token", idempotency_key="capture")
        self.assertTrue(result["committed"])
        self.assertIsNone(result["observation"])
        row = self.records()[0]
        self.assertEqual(row["outcome"], "completed")
        self.assertEqual(row["after"]["state"]["tick"], 1)
        self.assertEqual(self.action(token="policy-token", idempotency_key="capture"), result)
        self.assertEqual(len(self.records()), 1)

    def test_storage_failures_do_not_report_committed_motion_as_unexecuted(self):
        store = self.server.service.history.store
        original = store.write
        def fail_after(entry, *, update=False):
            if update and entry.get("after") is not None:
                raise OSError("disk full")
            original(entry, update=update)
        with patch.object(store, "write", side_effect=fail_after):
            result = self.action(idempotency_key="commit")
        self.assertTrue(result["committed"])
        self.assertEqual(self.records()[0]["outcome"], "incomplete")
        status = self.json(self.request("/v1/history/status"))["history"]
        self.assertIn("disk full", status["recording_fault"])
        self.assertFalse(self.json(self.request("/v1/health"))["ready"])
        self.assertEqual(self.action(idempotency_key="commit"), result)
        with self.assertRaises(urllib.error.HTTPError):
            self.action()
        self.assertEqual(len(self.backend.actions), 1)

    def test_failed_storage_before_motion_blocks_execution(self):
        with patch.object(self.server.service.history.store, "write", side_effect=OSError("disk full")):
            with self.assertRaises(urllib.error.HTTPError):
                self.action()
        self.assertEqual(self.backend.actions, [])

    def test_reset_restore_and_privileged_early_stop_are_explicit(self):
        self.json(self.request("/v1/checkpoints/save", method="POST", body={"label": "zero"}))
        original = self.backend.act
        self.backend.act = lambda action: replace(original(action), terminated=True)
        self.action({"actions": [DELTA] * 3})
        self.json(self.request("/v1/checkpoints/restore", method="POST", body={"checkpoint": "zero"}))
        self.action()
        self.json(self.request("/v1/reset", method="POST", body={"seed": 9}))
        rows = self.records()
        self.assertEqual([r["kind"] for r in rows], ["action", "checkpoint_restore", "action", "reset"])
        self.assertEqual(rows[1]["before"]["state"]["tick"], 1)
        self.assertEqual(rows[1]["after"]["state"]["tick"], 0)
        self.assertEqual(rows[3]["command"], {"seed": 9})

    def test_pagination_invalid_filters_missing_records_and_header(self):
        with self.request("/v1/action", method="POST", body=DELTA) as response:
            request_id = response.headers["X-GAR-Request-ID"]
            self.assertEqual(self.json(response)["request_id"], request_id)
        self.action()
        page = self.json(self.request("/v1/history?limit=1"))
        self.assertTrue(page["has_more"])
        page2 = self.json(self.request("/v1/history?after_seq=" + str(page["next_after_seq"])))
        self.assertEqual(len(page2["records"]), 1)
        for query in ("?limit=201", "?limit=0", "?after_seq=-1", "?limit=1&limit=2", "?wat=1"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/v1/history" + query)
            self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/history/not-here")
        self.assertEqual(caught.exception.code, 404)

    def test_started_record_is_visible_while_action_is_running(self):
        entered, release = threading.Event(), threading.Event()
        original = self.backend.act
        result = []
        def wait_action(action):
            entered.set()
            if not release.wait(4):
                raise RuntimeError("test action not released")
            return original(action)
        self.backend.act_policy = wait_action
        worker = threading.Thread(target=lambda: result.append(self.action(token="policy-token")))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            rows = self.records()
            self.assertEqual(rows[0]["outcome"], "incomplete")
            self.assertIsNone(rows[0]["after"])
            self.assertEqual(rows[0]["before"]["state"]["tick"], 0)
        finally:
            release.set()
            worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result[0]["committed"])
        self.assertEqual(self.records()[0]["outcome"], "completed")


class HistoryStorageTest(unittest.TestCase):
    def test_binary_arrays_freeze_before_state_and_reopen_as_same_json(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'arrays.sqlite3'
            state = {'q': np.arange(12, dtype=np.float32).reshape(4, 3)}
            before = state['q'].tolist()
            store = ActionHistory(path, binary_arrays=True)
            recorder = HistoryRecorder(store, lambda: state)
            with recorder.span(kind='action', payload=DELTA, request_id='r', role='policy'):
                state['q'] += .5
            key = store.list()['records'][0]['id']
            store.close()
            reader = ActionHistory(path)
            try:
                row = reader.get(key)
                self.assertEqual(row['before']['state']['q'], before)
                self.assertEqual(row['after']['state']['q'], state['q'].tolist())
                self.assertEqual(row['before']['nonfinite_paths'], [])
            finally:
                reader.close()

    def test_binary_mode_keeps_nonfinite_preflight_guard(self):
        import numpy as np
        store = ActionHistory(binary_arrays=True)
        recorder = HistoryRecorder(store, lambda: {'q': np.array([1., np.nan])})
        try:
            with self.assertRaises(HistoryUnavailableError):
                with recorder.span(kind='action', payload=DELTA, request_id='r', role='policy'):
                    self.fail('Nonfinite before-state must prevent execution')
            row = store.get(store.list()['records'][0]['id'])
            self.assertEqual(row['before']['nonfinite_paths'], ['/q/1'])
            self.assertEqual(row['before']['state']['q'], [1., 'NaN'])
            self.assertEqual(row['outcome'], 'not_executed')
        finally:
            store.close()

    def test_durable_reopen_and_explicit_retention_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            store = ActionHistory(path, max_records=2)
            recorder = HistoryRecorder(store, lambda: {"tick": 4})
            first_session = recorder.session_id
            for i in range(4):
                with recorder.span(kind="action", payload=DELTA, request_id=str(i), role="policy"):
                    pass
            self.assertEqual(store.status()["evicted_records"], 2)
            rows = store.list()["records"]
            self.assertEqual([r["seq"] for r in rows], [3, 4])
            store.close()
            store = ActionHistory(path, max_records=2)
            try:
                new = HistoryRecorder(store, lambda: {"tick": 4})
                self.assertNotEqual(new.session_id, first_session)
                self.assertEqual(store.get(rows[0]["id"])["session_id"], first_session)
                self.assertEqual(store.status()["evicted_records"], 2)
            finally:
                store.close()

    def test_incomplete_record_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            store = ActionHistory(path)
            recorder = HistoryRecorder(store, lambda: {"tick": 7})
            # Deliberately leave a committed start row, as after a process crash.
            with patch.object(store, "write", wraps=store.write) as write:
                with recorder.span(kind="action", payload=DELTA, request_id="r", role="policy"):
                    entry = json.loads(json.dumps(write.call_args.args[0]))
            store.write({**entry, "id": "interrupted"})
            store.close()
            store = ActionHistory(path)
            try:
                row = store.get("interrupted")
                self.assertEqual(row["outcome"], "incomplete")
                self.assertIsNone(row["after"])
                self.assertEqual(store.status()["incomplete_records"], 1)
            finally:
                store.close()

    def test_byte_budget_and_snapshot_copy(self):
        store = ActionHistory(max_bytes=1500, max_records=100)
        state = {"tick": 0, "nested": [1]}
        recorder = HistoryRecorder(store, lambda: state)
        try:
            for i in range(8):
                with recorder.span(kind="action", payload=DELTA, request_id=str(i), role="policy"):
                    state["tick"] += 1
                    state["nested"].append(i)
            status = store.status()
            self.assertLessEqual(status["payload_bytes"], 1500)
            self.assertGreater(status["evicted_records"], 0)
            last = store.get(store.list()["records"][-1]["id"])
            self.assertEqual(last["before"]["state"]["tick"], 7)
            self.assertEqual(last["after"]["state"]["tick"], 8)
            self.assertEqual(len(last["before"]["state"]["nested"]), 8)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()

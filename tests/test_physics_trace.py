"""Generic durable trace storage and HTTP access tests."""
from pathlib import Path
import tempfile
import unittest
import urllib.error
import numpy as np

from gym_anything_robotics.physics_trace import PhysicsTraceStore, missing_ranges, pack, unpack
from tests.test_http_server import HTTPServerTest
from gym_anything_robotics.access import APIKeys


class TraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "trace.sqlite3"
        self.store = PhysicsTraceStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_retention_totals_survive_rollback_eviction_and_writer_reopen(self):
        def check():
            db=self.store.db
            actual=tuple(db.execute('SELECT data_bytes,items,metadata_bytes,headers FROM retention_totals').fetchone())
            data=tuple(db.execute('SELECT COALESCE(SUM(size),0),COUNT(*) FROM item').fetchone())
            meta=tuple(db.execute('SELECT COALESCE(SUM(metadata_size),0),COUNT(*) FROM trace').fetchone())
            self.assertEqual(actual,data+meta)
        self.store.begin('a','r',0,.001,{})
        for tick in range(1,8):
            self.store.append('a',tick,{'tick':tick},[dict(type='contact',tick=tick)])
        check()
        db=self.store.db
        db.execute('BEGIN')
        db.execute('DELETE FROM item WHERE seq IN (SELECT seq FROM item LIMIT 3)')
        check()
        db.rollback()
        check()
        self.store.close()
        self.store=PhysicsTraceStore(self.path,max_items=3)
        check()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM item').fetchone()[0],3)
        self.assertGreater(self.store.summary('a')['evicted_samples'],0)
        # A separate writer retains the same transactional accounting.
        other=PhysicsTraceStore(self.path,max_items=3)
        try:
            other.append('a',8,{'tick':8},[])
            check()
        finally:
            other.close()
        db=self.store.db
        with db:db.execute('UPDATE trace SET metadata_size=metadata_size+1 WHERE id=\'a\'')
        check()

    def test_zstd_roundtrip_and_readonly_reopen(self):
        value = {"pose": np.arange(21, dtype=np.float32).reshape(3, 7),
                 "invalid": np.array([np.nan, np.inf, -np.inf]), "nested": [None, 2]}
        self.assertEqual(unpack(pack(value, compression="zstd")), unpack(pack(value)))
        self.store.close()
        self.store = PhysicsTraceStore(self.path, compression="zstd")
        self.store.begin("a", "r", 0, .001, {"codec": "lossless"})
        self.store.append("a", 1, value, [])
        self.store.finish("a", 1)
        reader = PhysicsTraceStore(self.path, readonly=True)
        try:
            self.assertEqual(reader.page("a")["items"][0]["data"], unpack(pack(value))["data"])
        finally:
            reader.close()

    def test_live_progress_missing_steps_and_window(self):
        self.store.begin("a", "r", 0, .001, {})
        self.store.append("a", 1, {"tick": 1}, [])
        self.assertFalse(self.store.summary("a")["complete"])
        self.store.append("a", 3, {"tick": 3}, [{"type": "event", "tick": 3}])
        self.store.finish("a", 3)
        summary = self.store.summary("a")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing_tick_ranges"], [[2, 2]])
        page = self.store.page("a", start_tick=3, end_tick=3)
        self.assertEqual([row["tick"] for row in page["items"]], [3])
        self.assertEqual(self.store.page("a", kind="event")["items"][0]["data"]["type"], "event")

    def test_eviction_is_not_reported_as_complete(self):
        self.store.close()
        self.store = PhysicsTraceStore(self.path, max_items=3)
        self.store.begin("a", "r", 0, .001, {})
        for tick in range(1, 7):
            self.store.append("a", tick, {"tick": tick}, [])
        self.store.finish("a", 6)
        summary = self.store.summary("a")
        self.assertGreater(summary["evicted_samples"], 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["expected_samples_through_last_tick"], 6)

    def test_nonfinite_diagnostics_and_ranges(self):
        self.assertEqual(missing_ranges(10, 15, [11, 13, 14]), [[12, 12], [15, 15]])
        self.assertEqual(missing_ranges(0, 10**12, []), [[1, 10**12]])
        self.store.begin("a", "r", 0, .001, {})
        self.store.append("a", 1, {"bad": [float("nan")]}, [])
        item = self.store.page("a")["items"][0]
        self.assertEqual(item["nonfinite_paths"], ["/bad/0"])
        self.assertEqual(item["data"]["bad"], ["NaN"])

    def test_scalar_fast_path_preserves_exceptional_locations(self):
        import math
        from unittest.mock import patch
        from gym_anything_robotics.action_history import diagnostic_json
        finite = {"a/b~c": [None, True, 17, 0.0, -0.0, 1e300, "NaN"],
                  "array": np.arange(6, dtype=np.float32)}
        with patch("gym_anything_robotics.physics_trace.diagnostic_json",
                   wraps=diagnostic_json) as convert:
            blob = pack(finite)
        self.assertEqual(convert.call_count, 0)
        result = unpack(blob)
        self.assertEqual(result["data"]["a/b~c"], finite["a/b~c"])
        self.assertEqual(math.copysign(1, result["data"]["a/b~c"][4]), -1)
        bad = {"a/b~c": [float("nan"), float("inf"), -float("inf")]}
        for codec in ("zlib", "zstd"):
            result = unpack(pack(bad, compression=codec))
            self.assertEqual(result["nonfinite_paths"], ["/a~1b~0c/0", "/a~1b~0c/1", "/a~1b~0c/2"])
            self.assertEqual(result["data"]["a/b~c"], ["NaN", "+Infinity", "-Infinity"])

    def test_binary_arrays_preserve_shape_and_nonfinite_locations(self):
        value = {"matrix": np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
                 "scalar": np.array(1.5), "empty": np.empty((0, 3), dtype=np.float32),
                 "bad": np.array([np.nan, np.inf, -np.inf], dtype=np.float32)}
        restored = unpack(pack(value))
        self.assertEqual(restored["data"]["matrix"], [[0, 2], [4, 6], [8, 10]])
        self.assertEqual(restored["data"]["scalar"], 1.5)
        self.assertEqual(restored["data"]["empty"], [])
        self.assertEqual(restored["nonfinite_paths"], ["/bad/0", "/bad/1", "/bad/2"])
        self.assertEqual(restored["data"]["bad"], ["NaN", "+Infinity", "-Infinity"])


class TraceHTTPTest(HTTPServerTest):
    def test_privileged_trace_route_and_policy_denials(self):
        path = Path(self.temporary_directory.name) / "trace.sqlite3"
        trace = PhysicsTraceStore(path)
        def begin(context):
            trace.begin(context["action_id"], context["request_id"], 0, 1/480, {"private": True})
            trace.append(context["action_id"], 1, {"secret": "full-private-state", "tick": 1},
                         [{"type": "test_event", "tick": 1}])
            trace.finish(context["action_id"], 1)
            return {"available": True, "store_path": str(path)}
        self.backend.begin_action_trace = begin
        try:
            result = self.json(self.request("/v1/action", method="POST", body={
                "type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]}))
            records = self.json(self.request("/v1/history?request_id=" + result["request_id"]))["records"]
            prefix = "/v1/history/" + records[0]["id"] + "/trace"
            page = self.json(self.request(prefix))
            self.assertTrue(page["summary"]["complete"])
            self.assertEqual(page["items"][0]["data"]["secret"], "full-private-state")
            self.assertEqual(self.json(self.request(prefix + "/events"))["items"][0]["data"]["type"], "test_event")
            self.server.keys = APIKeys("test-token", "policy-token")
            for suffix in ("", "/events", "?start_tick=1&end_tick=2", "/events?limit=1"):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    self.request(prefix + suffix, token="policy-token")
                self.assertEqual(caught.exception.code, 403)
                self.assertNotIn("full-private-state", caught.exception.read().decode())
            for query in ("?limit=129", "?wat=1", "?limit=1&limit=2", "?start_tick=-1"):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    self.request(prefix + query)
                self.assertEqual(caught.exception.code, 400)
        finally:
            trace.close()

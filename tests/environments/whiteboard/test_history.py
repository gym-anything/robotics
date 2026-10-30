import unittest
import numpy as np
from whiteboard.trace_codec import encode, decode
from whiteboard.api import diagnostic_state


class HistoryTests(unittest.TestCase):
    def test_codec_preserves_numeric_arrays_and_empty_shapes(self):
        value = {
            "float": np.array([[0.1, 1e-30, -1e30]], np.float64),
            "index": np.array([[3, 2], [5, 4]], np.int32),
            "empty": np.zeros((0, 6)),
            "list": [None, True, "test"],
        }
        restored = decode(encode(value))
        for key in ("float", "index", "empty"):
            self.assertEqual(restored[key], value[key].tolist())
        self.assertEqual(restored["list"], value["list"])

    def test_codec_rejects_nonfinite(self):
        for value in ({"x": np.array([np.nan])}, {"x": float("inf")}, {"x": np.array([object()])}):
            with self.assertRaises(ValueError):
                encode(value)

    def test_legacy_zlib_trace_is_readable(self):
        import json
        import struct
        import zlib

        value = {"legacy": [1, 2, 3], "tick": 17}
        header = json.dumps(value).encode()
        self.assertEqual(decode(zlib.compress(struct.pack("!I", len(header)) + header)), value)
        self.assertTrue(encode(value).startswith(b"WBT2"))


    def test_corrupt_history_is_retrievable_with_explicit_missing_values(self):
        result = diagnostic_state({"body_q": [[1.0, float("nan"), float("inf")]], "ready": False})
        self.assertEqual(result["body_q"], [[1.0, None, None]])
        self.assertEqual(result["nonfinite_fields"], ["/body_q/0/1", "/body_q/0/2"])
        self.assertFalse(result["ready"])

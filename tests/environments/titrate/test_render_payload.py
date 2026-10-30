import base64
import unittest
import numpy as np
from titrate.newton15.render_payload import encode, decode, mesh_arrays


class RenderPayloadTest(unittest.TestCase):
    def test_exact_usd_precision(self):
        for dtype in ('int32', 'float32'):
            q = np.arange(30000).reshape(-1, 3).astype(dtype)
            if dtype == 'float32':
                q *= .0012345
            restored = decode(encode(q, dtype), dtype, 3)
            np.testing.assert_array_equal(q, restored)
            self.assertEqual(q.tobytes(), restored.tobytes())

    def test_invalid_packets(self):
        payload = encode(np.zeros((3, 3)), 'float32')
        for field, value in (('shape', [2**30, 3]), ('shape', [3, True]),
                ('dtype', 'object'), ('data', 'x'), ('data', '!'*len(payload['data']))):
            with self.assertRaises(ValueError):
                decode({**payload, field: value}, 'float32', 3)
        with self.assertRaises(ValueError):
            encode([2**32], 'int32')
        with self.assertRaises(ValueError):
            decode(dict(dtype='float32', shape=[1],
                data=base64.b64encode(np.array([np.nan], '<f4').tobytes()).decode()), 'float32')
        mesh = dict(vertices=payload, normals=payload, triangles=encode([[0, 1, 3]], 'int32'))
        with self.assertRaises(ValueError):
            mesh_arrays(mesh)


if __name__ == '__main__':
    unittest.main()

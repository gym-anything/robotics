"""Wire compatibility for producer-validated JSON; no physics claims."""
import unittest
import numpy as np
from gym_anything_robotics.physics_trace import pack,unpack,ValidatedJSON

class ValidatedJSONTest(unittest.TestCase):
    def test_exact_bytes_with_surrounding_arrays(self):
        rows=[dict(surface='can_000',body='left/finger',shape='pad/~0',
            point_world_m=[.1,-0.,1.23456789012345],force_on_surface_n=[1.,-2.,3.],
            source='Finite contact — µm')]
        before=np.arange(12,dtype=np.float32).reshape(4,3)
        after=np.array([1,-1],np.int64)
        for compression in ('zlib','zstd'):
            ordinary=dict(before=before,contacts=rows,after=after)
            optimized=dict(before=before,contacts=ValidatedJSON(rows),after=after)
            self.assertEqual(pack(ordinary,compression=compression),pack(optimized,compression=compression))
            decoded=unpack(pack(optimized,compression=compression),arrays=True)
            self.assertEqual(decoded['data']['contacts'],rows)
            self.assertEqual(decoded['nonfinite_paths'],[])
            np.testing.assert_array_equal(decoded['data']['before'],before)
            np.testing.assert_array_equal(decoded['data']['after'],after)

    def test_invalid_fast_path_is_not_silently_sanitized(self):
        for value in ([{'x':float('nan')}],[{'x':float('inf')}],[{'x':np.zeros(1)}]):
            with self.assertRaises((TypeError,ValueError)):
                pack(ValidatedJSON(value))

    def test_ordinary_diagnostics_still_record_nonfinite(self):
        result=unpack(pack({'bad':float('nan')}))
        self.assertEqual(result['data']['bad'],'NaN')
        self.assertIn('/bad',result['nonfinite_paths'])

if __name__=='__main__':unittest.main()

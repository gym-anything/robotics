"""Numerical/asset guards complement (not replace) inspected optical renders."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from titrate.newton15.volume_optics import pack, unpack, install, read_field, sample_field


class VolumeOpticsTest(unittest.TestCase):
    def field(self):
        return dict(origin_m=[0.,0.,1.],spacing_m=.001,counts_xyz=[2,3,4],
                    values=np.arange(24,dtype=np.float32)*1e-7)

    def test_payload_roundtrip_and_reject_invalid(self):
        field = self.field()
        np.testing.assert_array_equal(unpack(pack(field))['values'],field['values'])
        for change in (dict(counts_xyz=[2,3,5]),dict(spacing_m=0),
                       dict(origin_m=[0.,float('nan'),0.]),dict(counts_xyz=[True,3,4])):
            with self.assertRaises(ValueError):
                unpack({**pack(field),**change})

    def test_archived_field_roundtrip_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stage = Usd.Stage.CreateNew(str(folder/'scene.usda'))
            UsdGeom.SetStageMetersPerUnit(stage,1.)
            record = install(stage,self.field(),folder/'optics')
            stage.GetRootLayer().Save()
            restored = read_field(Usd.Stage.Open(str(folder/'scene.usda')))
            np.testing.assert_array_equal(restored['values'],self.field()['values'])
            self.assertEqual(restored['origin_m'],self.field()['origin_m'])
            changed = self.field()
            changed['values'][0] = 99
            np.savez_compressed(folder/'optics'/(record['sha256']+'.npz'),**changed)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                read_field(stage)
            UsdGeom.SetStageMetersPerUnit(stage,.01)
            with self.assertRaises(ValueError):
                install(stage,self.field(),folder/'optics')

    def test_surface_only_input_rejected(self):
        from titrate.newton15.renderer import Renderer
        with self.assertRaises(ValueError):
            Renderer('unused',backend='unknown')
        with self.assertRaises(ValueError):
            Renderer('unused',interpreter='ignored.exe')

    def test_isolated_coloured_parcel_is_not_erased(self):
        # Its 0.5 isosurface is unresolved and represented by a parcel glyph.
        # The glyph still requires the parcel's concentration in its interior.
        result = sample_field(np.zeros((1,3)),np.array([1e-5]),.003,.001,'cpu')
        self.assertAlmostEqual(float(result['values'].max()),1e-5,places=10)
        self.assertGreater(np.count_nonzero(result['values']),1)


if __name__ == '__main__':
    unittest.main()

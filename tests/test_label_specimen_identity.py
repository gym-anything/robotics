"""Visual tube-identity regression checks; no simulator or GPU required."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
from label_application import dressing


class SpecimenIdentityTest(unittest.TestCase):

    def test_manufacturer_and_pack_agree(self):
        texts = []
        original = ImageDraw.ImageDraw.text
        def capture(draw, xy, text, *args, **kwargs):
            texts.append(text)
            return original(draw, xy, text, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Room scans are unrelated; only replace their input textures, not the printed artwork.
            texture = root / "scan.png"
            Image.new("RGB", (32, 32), (120, 120, 120)).save(texture)
            maps = dict(diff=texture, rough=texture, nor_gl=texture)
            with patch.object(dressing.va, "texture", return_value=maps), patch.object(ImageDraw.ImageDraw, "text", capture):
                generated = dressing._derived(root)
            self.assertIn("VACUTEST  K2EDTA  6.0 mL", texts)
            self.assertIn("VACUTEST K2EDTA 6.0 mL", texts)
            self.assertIn("13 x 100 mm  •  100 TUBES", texts)
            self.assertEqual(sum("REF VT-13100-6" in t for t in texts), 2)
            self.assertFalse(any("3.0 mL" in t or "13 x 75" in t for t in texts))
            for key in ("tube_label", "tube_pack"):
                self.assertTrue(Path(generated[key]["diff"]).is_file())


if __name__ == "__main__":
    unittest.main()

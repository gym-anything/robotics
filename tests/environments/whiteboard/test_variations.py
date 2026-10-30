import unittest

from whiteboard.variations import resolve, select
from whiteboard.config import Config
from whiteboard.geometry import classroom, tools


class VariationTests(unittest.TestCase):

    def test_constraints_are_exact(self):
        a = select(split="test", kind="bridge", selection_seed=44)
        self.assertEqual((a["split"], a["task"]["kind"]), ("test", "bridge"))
        self.assertEqual(a, select(split="test", kind="bridge", selection_seed=44))
        with self.assertRaises(ValueError):
            select(variation_id=a["variation_id"], kind="text")
        for invalid in (-1, 10000, True, 1.5):
            with self.assertRaises(ValueError):
                resolve(invalid)

    def test_placement_geometry_and_product_dimensions(self):
        baseline = {o.name: o for o in tools(Config())}
        for i in (0, 7, 99, 511, 1024, 4999, 9999):
            c, _ = resolve(i)
            objects = tools(c)
            markers = sorted(
                (o for o in objects if o.name.startswith("marker_")), key=lambda o: o.position[0]
            )
            self.assertEqual(
                tuple(o.name.removeprefix("marker_") for o in markers), c.layout.marker_order
            )
            for a, b in zip(markers, markers[1:]):
                self.assertGreater(b.position[0] - a.position[0], 0.138)
            for obj in objects:
                self.assertEqual(obj.mass, baseline[obj.name].mass)
                self.assertEqual(
                    [p.size for p in obj.parts], [p.size for p in baseline[obj.name].parts]
                )
            self.assertEqual(len(classroom(c)), len(classroom(Config())))


if __name__ == "__main__":
    unittest.main()

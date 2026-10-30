import unittest
from titrate.newton15.variations import COUNT, resolve, select


class VariationTest(unittest.TestCase):
    def test_catalogue_unique_reproducible_and_exact_volumes(self):
        rows = set()
        for i in range(COUNT):
            cfg, r = resolve(i, spacing_m=.000625)
            cfg2, r2 = resolve(f'titration-n15-{i:04d}', spacing_m=.000625)
            self.assertEqual((cfg, r), (cfg2, r2))
            rows.add((cfg.flask_x, cfg.flask_y, r.aliquot_ml, r.burette_ml))
            self.assertEqual(r.aliquot_ml+r.dilution_ml, 25.)
        self.assertEqual(len(rows), COUNT)
        cfg, r = resolve(0)
        self.assertEqual((cfg.flask_x, cfg.flask_y, r.aliquot_ml, r.burette_ml), (.15, .035, 10., 20.))

    def test_constraints_and_no_rounding(self):
        result = select({'constraints': {'aliquot_ml': 12., 'burette_ml': 22.}}, spacing_m=.000625)
        self.assertEqual(result['reagents']['aliquot_ml'], 12.)
        self.assertEqual(result['reagents']['burette_ml'], 22.)
        for bad in ({'variation_id': COUNT}, {'variation_id': True}, {'seed': 4},
                {'constraints': {'aliquot_ml': 12.1}}, {'variation_id': 0, 'constraints': {'aliquot_ml': 12.}}):
            with self.assertRaises(ValueError):
                select(bad)


if __name__ == '__main__':
    unittest.main()

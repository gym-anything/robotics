import unittest
import copy
from titrate.newton15.task import EndpointEvaluator, description
from titrate.newton15.liquid_charge import Reagents


class EndpointTest(unittest.TestCase):
    def state(self, **changes):
        values = dict(pink_fraction=.95, base_in_flask_ml=10.03, acid_in_flask_mol=.001,
            spill_ml=0., invariants_ok=True, indicator_retained=1., loose_base_ml=0., loose_acid_mol=0.,
            pending_base_ml=0., arriving_ml=0., stray_moving_ml=0.)
        values.update(changes)
        return values

    def run_constant(self, ticks=31, **changes):
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(ticks):
            result = e.update(t, **self.state(**changes))
        return result

    def test_continuous_hold_and_restore(self):
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(16):
            self.assertFalse(e.update(t, **self.state())['success'])
        saved = e.snapshot()
        for t in range(16, 30):
            self.assertFalse(e.update(t, **self.state())['success'])
        self.assertTrue(e.update(30, **self.state())['success'])
        e.restore(saved)
        self.assertEqual(e.last_tick, 15)
        self.assertFalse(e.update(16, **self.state(pink_fraction=.84))['success'])
        self.assertIsNone(e.pink_start_tick)
        with self.assertRaises(ValueError):
            e.update(18, **self.state())

    def test_endpoint_band_is_one_drop_past_the_flasks_own_acid(self):
        for base, accepted in ((9.99, False), (10.001, True), (10.03, True), (10.049, True),
                (10.05+3.e-14, True), (10.051, False), (10.08, False), (10.099, False)):
            result = self.run_constant(base_in_flask_ml=base)
            self.assertEqual(result['success'], accepted, base)
            self.assertEqual(result['checks']['endpoint'], accepted, base)
        # Back-titration: an overshoot followed by standard acid, ending inside the band of the flask's own acid.
        result = self.run_constant(base_in_flask_ml=10.63, acid_in_flask_mol=.00106)
        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['flags']['acid_added_ml'], .6)

    def test_one_more_drop_after_a_completed_endpoint_fails(self):
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(31):
            result = e.update(t, **self.state(base_in_flask_ml=10.03))
        self.assertTrue(result['success'])
        for t in range(31, 70):
            result = e.update(t, **self.state(base_in_flask_ml=10.08))
        self.assertFalse(result['success'])
        self.assertGreater(result['score'], .8)

    def test_titrant_still_in_transit_counts_against_the_band_and_in_progress(self):
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(31):
            e.update(t, **self.state(base_in_flask_ml=10.048))
        # A sub-threshold flick in the tip does not restart the window, but counts as landed.
        result = e.update(31, **self.state(base_in_flask_ml=10.048, loose_base_ml=.004, pending_base_ml=.004))
        self.assertFalse(result['success'])
        self.assertFalse(result['checks']['endpoint'])
        self.assertGreater(result['stable_pink_seconds'], 30.)
        self.assertAlmostEqual(result['flags']['overshoot_ml'], .002)
        self.assertAlmostEqual(self.run_constant(base_in_flask_ml=5., pink_fraction=0., pending_base_ml=1.)['reward']['completion'], .5)
        self.assertAlmostEqual(self.run_constant(base_in_flask_ml=10.03, pending_base_ml=.17)['reward']['accuracy'], 1-.15/.6)

    def test_unmixed_flask_below_equivalence_never_starts_the_hold(self):
        result = self.run_constant(base_in_flask_ml=9.66, pink_fraction=.9001)
        self.assertFalse(result['success'])
        self.assertEqual(result['stable_pink_seconds'], 0.)

    def test_titrant_or_acid_leaving_its_container_restarts_the_persistence_window(self):
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(20):
            e.update(t, **self.state())
        for t in range(20, 50):
            # A drip below the restart threshold per tick accumulates past it.
            result = e.update(t, **self.state(base_in_flask_ml=10.03+.004*(t-19)))
            if t < 22:
                self.assertGreater(result['stable_pink_seconds'], 0.)
        self.assertLess(result['stable_pink_seconds'], 30.)
        self.assertFalse(result['success'])
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(20):
            e.update(t, **self.state())
        result = e.update(20, **self.state(acid_in_flask_mol=.001002, base_in_flask_ml=10.05))
        self.assertEqual(result['stable_pink_seconds'], 0.)
        # Titrant past the stopcock but not yet in the flask restarts the window before it lands.
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(30):
            e.update(t, **self.state(loose_base_ml=.002))
        result = e.update(30, **self.state(loose_base_ml=.045))
        self.assertFalse(result['success'])
        self.assertEqual(result['stable_pink_seconds'], 0.)
        # Flask liquid moving out of the flask is not an addition.
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(30):
            e.update(t, **self.state())
        self.assertTrue(e.update(30, **self.state(base_in_flask_ml=10.01, loose_base_ml=.02))['success'])
        # A small decrease does not hide a later addition.
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(5):
            e.update(t, **self.state())
        e.update(5, **self.state(base_in_flask_ml=10.02))
        self.assertEqual(e.update(6, **self.state(base_in_flask_ml=10.035))['stable_pink_seconds'], 0.)

    def test_aliquot_retention_is_measured_by_the_indicator_tracer(self):
        self.assertTrue(self.run_constant(indicator_retained=.996)['success'])
        topped_up = self.run_constant(indicator_retained=.5)
        self.assertFalse(topped_up['success'])
        self.assertFalse(topped_up['checks']['acid_retained'])
        self.assertAlmostEqual(topped_up['acid_loss_ml'], 5.)
        self.assertEqual(topped_up['progress'], 0.)
        replaced = self.run_constant(indicator_retained=0., pink_fraction=0.)
        self.assertEqual(replaced['score'], 0.)

    def test_spill_and_invalid_state(self):
        self.assertFalse(self.run_constant(spill_ml=.051)['success'])
        result = self.run_constant(invariants_ok=False)
        self.assertFalse(result['success'])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['score'])
        self.assertIn('invalid_reason', result)

    def test_progress_measures_the_flask_not_the_route(self):
        score = self.run_constant
        untouched = score(base_in_flask_ml=0., pink_fraction=0.)
        self.assertEqual((untouched['progress'], untouched['score']), (0., 0.))
        half = score(base_in_flask_ml=5., pink_fraction=0.)
        self.assertAlmostEqual(half['progress'], .25)
        self.assertAlmostEqual(half['score'], .225)
        unmixed = score(base_in_flask_ml=10.03, pink_fraction=.5)
        self.assertAlmostEqual(unmixed['progress'], .5)
        held = self.run_constant(ticks=16)
        self.assertAlmostEqual(held['progress'], .5*(1+15/30))
        four_drops = score(base_in_flask_ml=10.2)
        self.assertAlmostEqual(four_drops['reward']['accuracy'], 1-.15/.6)
        self.assertAlmostEqual(four_drops['flags']['overshoot_ml'], .15)
        self.assertAlmostEqual(score(base_in_flask_ml=10.7)['progress'], 0.)
        messy = score(spill_ml=.15)
        self.assertAlmostEqual(messy['reward']['accuracy'], 1-.1/.6)
        self.assertAlmostEqual(messy['score'], .9*(1-.1/.6))
        self.assertAlmostEqual(score(indicator_retained=.96)['reward']['accuracy'], 1-.35/.6)
        success = score()
        self.assertTrue(success['success'])
        self.assertEqual(success['score'], 1.)

    def test_reject_inconsistent_snapshot_without_mutation(self):
        e = EndpointEvaluator(1., 10., .1, .1)
        e.update(0, **self.state())
        original = e.snapshot()
        for field, value in (('success', True), ('score', 1.), ('stable_pink_seconds', 30.),
                ('equivalence_ml', 1.), ('pink_fraction', float('nan')), ('progress', 1.),
                ('indicator_retained', .5), ('loose_base_ml', -1.), ('pending_base_ml', .2), ('stray_moving_ml', -1.)):
            altered = copy.deepcopy(original)
            altered['result'][field] = value
            with self.assertRaises(ValueError, msg=field):
                e.restore(altered)
            self.assertEqual(e.snapshot(), original)
        altered = copy.deepcopy(original)
        altered['window_base_ml'] = 9.
        with self.assertRaises(ValueError):
            e.restore(altered)
        self.assertEqual(e.snapshot(), original)

    def test_snapshot_after_restart_and_during_hold_validates(self):
        e = EndpointEvaluator(1., 10., .1, .1)
        for t in range(10):
            e.update(t, **self.state())
        e.update(10, **self.state(base_in_flask_ml=10.045))
        e.validate(e.snapshot())
        e.update(11, **self.state(base_in_flask_ml=10.04, loose_base_ml=.003))
        e.validate(e.snapshot())
        e.update(12, **self.state(pink_fraction=.5))
        e.validate(e.snapshot())

    def test_description_states_the_job_not_the_checklist(self):
        self.assertEqual(description(Reagents()),
            'Titrate the 10 mL acid aliquot in the flask, diluted with 15 mL water and phenolphthalein, '
            'using 0.1 mol/L sodium hydroxide. Stop at the endpoint: the first faint pink that persists '
            'throughout the solution for 30 seconds.')


if __name__ == '__main__':
    unittest.main()

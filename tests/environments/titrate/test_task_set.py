"""Pure Python/NumPy tests: no Newton, Warp, MuJoCo, servers or simulation."""

import copy
from dataclasses import asdict
import json
import math
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from titrate.newton15 import benchmark, variations
from titrate.newton15.lab_config import LabConfig
from titrate.newton15.liquid_charge import Reagents
from titrate.newton15.task import EndpointEvaluator, READINGS, description
from titrate.newton15.task_catalogue import TASKS, TASK_SET, contract
from titrate.newton15.task_evaluators import CatalogueEvaluator, motor_rpm
from titrate.newton15.task_presets import preset
from tests.packages import PACKAGES

ROOT = PACKAGES / "titrate" / "newton15"


def initial(name, config=None, reagents=None):
    c, r = config or LabConfig(), reagents or Reagents()
    p = preset(name, c, r, .000625)
    m = dict(pink_fraction=0., base_in_flask_ml=p['base0_ml'], acid_in_flask_mol=r.aliquot_ml*r.acid_molar/1000,
             spill_ml=0., invariants_ok=True, indicator_retained=1., loose_base_ml=0., loose_acid_mol=0.,
             pending_base_ml=0., arriving_ml=0., stray_moving_ml=0., beaker_base_ml=0., beaker_acid_ml=0.,
             burette_volume_ml=p['reagents']['burette_ml'], delivery_pending_ml=0., valve_rad=math.pi/2,
             motor_rpm=motor_rpm(p['dial_rad']), bar_rpm=0.,
             flask_distance_m=math.dist(p['flask_start'][:2], c.tip_position[:2]), flask_start_distance_m=0.,
             flask_tilt_rad=0., flask_plate_error_m=.0002, flask_speed_m_s=0., flask_spin_rad_s=0.,
             beaker_tilt_rad=0., bottle_tilt_rad=0., bottle_bench_error_m=.0002,
             bottle_speed_m_s=0., bottle_spin_rad_s=0., flask_robot_contact=False,
             stirrer_robot_contact=False, bottle_robot_contact=False)
    return p, m


def evaluator(name, dt=1., r=None):
    r = r or Reagents()
    p, m = initial(name, reagents=r)
    return CatalogueEvaluator(name, dt, r, p), m


def result(name, changes, ticks=32):
    e, m = evaluator(name)
    for t in range(ticks):
        out = e.update(t, **{**m, **changes})
    return out


SUCCESS = {
    'start-stirring': dict(motor_rpm=500., bar_rpm=490.),
    'centre-flask': dict(flask_distance_m=.003),
    'prime-burette': dict(beaker_base_ml=1.),
    'run-in': dict(base_in_flask_ml=9.),
    'drain-burette': dict(beaker_base_ml=19.8, burette_volume_ml=.1, valve_rad=.1),
    'decant-acid': dict(beaker_acid_ml=10.),
    'finish-titration': dict(base_in_flask_ml=10.03, pink_fraction=.95),
}


class CatalogueTest(unittest.TestCase):
    def setUp(self):
        # The sparse checkout intentionally omits robots/. Only metadata is stubbed.
        self.stub = patch.dict(sys.modules, {'robots': NS(), 'robots.catalog': NS(
            default_robot=lambda name: 'panda', environment_robots=lambda name: ['panda'], ENVIRONMENTS={})})
        self.stub.start()
        self.addCleanup(self.stub.stop)

    def test_exact_task_set_and_gar_roundtrip(self):
        from gym_anything_robotics import tasks
        from gym_anything_robotics.tasks import TaskSpec
        cat = json.loads((ROOT/'benchmark.json').read_text())
        entries = cat['task_set']['tasks']
        self.assertEqual([e['selection'] for e in entries], list(TASK_SET))
        self.assertEqual(cat['count'], 8*variations.COUNT)
        self.assertEqual(len({e['task_id'] for e in entries}), 8)
        for entry in entries:
            resolved = tasks.resolve('titration', entry['selection'])
            self.assertEqual(resolved.task_id, entry['task_id'])
            self.assertEqual(TaskSpec.from_dict(resolved.to_dict()).to_dict(), resolved.to_dict())
            self.assertEqual(contract(entry['selection']['task']), entry['task_contract'])
            self.assertEqual(resolved.launch.backend_options.get('task', 'titration'), entry['selection']['task'])

    def test_legacy_default_id_description_and_options(self):
        ref = json.loads((ROOT/'reference.json').read_text())
        c, r = variations.resolve(0)
        got = benchmark.resolve({})
        self.assertEqual(got, dict(task_id='titration/titration-n15-0000', description=description(r),
            backend_options={**ref, 'config': asdict(c), 'reagents': asdict(r)},
            selection=dict(variation_id='titration-n15-0000', config=asdict(c), reagents=asdict(r),
                           task=description(r), reset_request={'options': {'variation_id': 0}}), split=None))
        self.assertEqual(benchmark.resolve({'task': 'titration', 'variation_id': 0})['task_id'], got['task_id'])

    def test_constraints_and_input_rejections_preserved(self):
        for name in TASKS:
            selected = variations.select({'task': name, 'constraints': {'aliquot_ml': 12., 'burette_ml': 22.}})
            self.assertEqual(selected['reagents']['aliquot_ml'], 12.)
            self.assertEqual(selected['reagents']['burette_ml'], 22.)
        for bad in ({'task': None}, {'task': []}, {'task': 'run_in'}, {'seed': 0}, {'variation_id': True},
                    {'variation_id': '0'}, {'variation_id': 0.}, {'variation_id': -1},
                    {'constraints': {'aliquot_ml': '10'}}, {'task': 'run-in', 'extra': 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                variations.select(bad)

    def test_all_native_placements_and_precharge_volumes(self):
        worst = 0.
        for i in range(variations.COUNT):
            c, r = variations.resolve(i)
            for name in SUCCESS:
                p = preset(name, c, r, .000625)
                worst = max(worst, max(p['reach_m'].values()))
                self.assertEqual(p, preset(name, c, r, .000625))
                self.assertAlmostEqual(sum(p['reagents'][k] for k in ('aliquot_ml', 'dilution_ml', 'burette_ml')),
                                       r.aliquot_ml+r.dilution_ml+r.burette_ml)
                if name == 'centre-flask':
                    distance = math.dist(p['flask_start'][:2], c.flask_position[:2])
                    self.assertTrue(.013 <= distance <= .0165)
                    self.assertLessEqual(distance+.0335, .05+1e-12)
        self.assertLess(worst, .855)


class EvaluationTest(unittest.TestCase):
    def test_every_start_and_idle_is_unsuccessful_zero(self):
        for name in SUCCESS:
            e, m = evaluator(name)
            for t in range(40):
                out = e.update(t, **m)
                self.assertEqual((out['success'], out['score']), (False, 0.), (name, t, out))

    def test_every_task_can_pass_in_a_constructed_state_and_reports_contract(self):
        for name, changes in SUCCESS.items():
            out = result(name, changes)
            self.assertEqual((out['success'], out['score']), (True, 1.), (name, out))
            self.assertEqual(out['contract'], contract(name))
            self.assertEqual(out['reward']['contract'], contract(name))

    def test_continuous_partial_credit_and_common_fail_scale(self):
        fields = {'start-stirring': ('bar_rpm', [0., 50., 100.]),
                  'centre-flask': ('flask_distance_m', [.012, .010, .008]),
                  'prime-burette': ('beaker_base_ml', [.01, .1, .2]),
                  'run-in': ('base_in_flask_ml', [1., 4., 7.]),
                  'drain-burette': ('beaker_base_ml', [1., 7., 14.]),
                  'decant-acid': ('beaker_acid_ml', [1., 3., 5.])}
        for name, (key, values) in fields.items():
            extra = {'motor_rpm': 110.} if name == 'start-stirring' else {}
            if name == 'start-stirring':
                values = [60., 80., 100.]
            scores = [result(name, {**extra, key: v})['score'] for v in values]
            self.assertTrue(0 <= scores[0] < scores[1] < scores[2] < 1, (name, scores))
            a = result(name, {**extra, key: values[1]})
            b = result(name, {**extra, key: values[1]+1e-7})
            self.assertLess(abs(a['score']-b['score']), .0001)
            self.assertAlmostEqual(a['score'], .9*a['progress'])

    def test_failed_invariants_spill_loss_and_success_unlatches(self):
        for name, success in SUCCESS.items():
            for changes in ({'invariants_ok': False}, {'spill_ml': .051}, {'indicator_retained': .994}):
                out = result(name, {**success, **changes})
                self.assertFalse(out['success'], (name, changes))
                if changes.get('invariants_ok') is False:
                    self.assertIsNone(out['score'])
                else:
                    self.assertAlmostEqual(out['score'], .9*out['progress'])
            e, m = evaluator(name)
            for tick in range(32):
                e.update(tick, **{**m, **success})
            self.assertFalse(e.update(32, **{**m, **success, 'spill_ml': .2})['success'])

    def test_valve_closure_window_pending_delivery_and_open_drain(self):
        for name in ('prime-burette', 'run-in'):
            e, m = evaluator(name)
            m.update(SUCCESS[name])
            for t in range(2):
                self.assertFalse(e.update(t, **m)['success'])
            self.assertTrue(e.update(2, **m)['success'])
            self.assertFalse(e.update(3, **{**m, 'delivery_pending_ml': .003})['success'])
            self.assertFalse(e.update(4, **m)['success'])
            self.assertFalse(result(name, {**SUCCESS[name], 'valve_rad': .5})['success'])
        self.assertFalse(result('drain-burette', {**SUCCESS['drain-burette'], 'burette_volume_ml': .201})['success'])
        self.assertFalse(result('drain-burette', {**SUCCESS['drain-burette'], 'valve_rad': .201})['success'])

    def test_centre_release_upright_support_and_spurious_contact(self):
        for change in ({'flask_robot_contact': True}, {'flask_tilt_rad': .1}, {'flask_plate_error_m': .002},
                       {'flask_speed_m_s': .02}, {'base_in_flask_ml': .011}):
            self.assertFalse(result('centre-flask', {**SUCCESS['centre-flask'], **change})['success'])
        self.assertFalse(result('prime-burette', {**SUCCESS['prime-burette'], 'flask_robot_contact': True})['success'])

    def test_stirring_five_second_window_coupling_and_release(self):
        e, m = evaluator('start-stirring')
        m.update(SUCCESS['start-stirring'])
        for t in range(5):
            self.assertFalse(e.update(t, **m)['success'])
        self.assertTrue(e.update(5, **m)['success'])
        self.assertFalse(e.update(6, **{**m, 'stirrer_robot_contact': True})['success'])
        for change in ({'bar_rpm': 800.}, {'bar_rpm': 149.}, {'motor_rpm': 801.}, {'flask_start_distance_m': .011}):
            self.assertFalse(result('start-stirring', {**SUCCESS['start-stirring'], **change})['success'])

    def test_transfer_bands_flask_protection_and_nested_inventory_inputs(self):
        for name, field, values in (('prime-burette', 'beaker_base_ml', [.299, 2.001]),
                                   ('decant-acid', 'beaker_acid_ml', [5.999, 14.001]),
                                   ('run-in', 'base_in_flask_ml', [8.499, 9.501])):
            for value in values:
                self.assertFalse(result(name, {**SUCCESS[name], field: value})['success'])
        for name in ('prime-burette', 'drain-burette'):
            self.assertFalse(result(name, {**SUCCESS[name], 'base_in_flask_ml': .051})['success'])
        for change in ({'bottle_robot_contact': True}, {'bottle_tilt_rad': .3}, {'bottle_bench_error_m': .002},
                       {'acid_in_flask_mol': .001006}, {'base_in_flask_ml': .011}):
            self.assertFalse(result('decant-acid', {**SUCCESS['decant-acid'], **change})['success'])
        self.assertFalse(result('run-in', dict(base_in_flask_ml=9.49, pending_base_ml=.02))['success'])

    def test_finish_rebase_and_unchanged_endpoint_and_back_titration(self):
        e, m = evaluator('finish-titration')
        base0 = e.setup['base0_ml']
        out = e.update(0, **m)
        self.assertEqual(out['score'], 0.)
        halfway = e.update(1, **{**m, 'base_in_flask_ml': (base0+10)/2})
        self.assertAlmostEqual(halfway['score'], .225)
        for change in (SUCCESS['finish-titration'], dict(base_in_flask_ml=10.08, pink_fraction=.95),
                       dict(base_in_flask_ml=10.63, acid_in_flask_mol=.00106, pink_fraction=.95),
                       dict(base_in_flask_ml=10.049, pending_base_ml=.004, pink_fraction=.95)):
            e, m = evaluator('finish-titration')
            original = EndpointEvaluator(1., 10., .1, .1)
            for t in range(35):
                values = {**m, **change}
                new = e.update(t, **values)
                old = original.update(t, **{k: values[k] for k in (*READINGS, 'invariants_ok')})
                self.assertEqual(new['checks'], old['checks'])
                self.assertEqual(new['success'], old['success'])

    def test_checkpoint_roundtrip_rejection_and_consecutive_ticks(self):
        for name, changes in SUCCESS.items():
            e, m = evaluator(name, dt=.25)
            m.update(changes)
            for t in range(30):
                e.update(t, **m)
            snap = json.loads(json.dumps(e.snapshot()))
            other, _ = evaluator(name, dt=.25)
            other.restore(snap)
            for t in range(30, 140):
                a, b = e.update(t, **m), other.update(t, **m)
                self.assertEqual(a, b)
            for path, value in (('task', 'wrong'), ('last_tick', -1), ('closed_start', 300)):
                bad = copy.deepcopy(snap)
                bad[path] = value
                before = other.snapshot()
                with self.assertRaises(ValueError):
                    other.restore(bad)
                self.assertEqual(other.snapshot(), before)
            bad = copy.deepcopy(snap)
            bad['result']['score'] = .123
            with self.assertRaises(ValueError):
                other.restore(bad)
            with self.assertRaises(ValueError):
                other.update(150, **m)


if __name__ == '__main__':
    unittest.main()

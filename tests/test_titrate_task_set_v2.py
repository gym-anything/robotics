"""Contract `titration-tasks-v2` (the render-checked eight): pure Python, no Newton, Warp, servers or simulation.

The fix round changed one thing in titrate's eight: the seven tasks other than titration report the
single contract string (`titration-tasks-v2`) instead of `<contract>/<task>`. These tests check that
v2 is exactly the released tasks with that contract, and that selections without a contract never
reach the v2 code. The released behaviour keeps its own tests (tests/environments/titrate/).
"""

from contextlib import ExitStack
import copy
import inspect
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from titrate.newton15 import benchmark, task_set_v2 as V2
from titrate.newton15.lab_config import LabConfig
from titrate.newton15.task import REWARD_CONTRACT
from titrate.newton15.task_catalogue import TASKS, TASK_SET as V1_TASK_SET, contract as v1_contract
from titrate.newton15.task_evaluators import CatalogueEvaluator
from tests.environments.titrate.test_task_set import ROOT, SUCCESS, evaluator
from tests.environments.titrate.test_task_set_wiring import Liquid, Readings, Session
from tests.packages import REPO

SEVEN = tuple(s['task'] for s in V2.TASK_SET[:-1])


def wrapped(name, dt=1.):
    e, m = evaluator(name, dt)
    return V2.ContractEvaluator(e), m


def relabelled(result, label=V2.CONTRACT):
    result = copy.deepcopy(result)
    result['contract'] = label
    result['reward']['contract'] = label
    return result


class CatalogueTest(unittest.TestCase):
    def test_benchmark_json_lists_v2_and_keeps_v1(self):
        cat = json.loads((ROOT/'benchmark.json').read_text())
        self.assertEqual(cat['task_set']['selections'], list(V2.TASK_SET))
        self.assertEqual(cat['task_set']['contract'], V2.CONTRACT)
        self.assertEqual(cat['task_set']['titration_contract'], REWARD_CONTRACT)
        self.assertEqual([e['selection'] for e in cat['task_set_v1']['tasks']], list(V1_TASK_SET))
        self.assertIn('contract', cat['selectors'])
        self.assertEqual(cat['selector_choices']['contract'], [V2.CONTRACT])
        self.assertEqual(set(SEVEN) | {'titration'}, set(TASKS))
        # The audited hardest task keeps its released split selection exactly.
        test = json.loads((REPO/'benchmarks/roboworld/splits/test.json').read_text())['tasks']
        self.assertEqual(V2.TASK_SET[-1], next(t['selection'] for t in test if t['environment_id'] == 'titration'))

    def test_one_contract_string_for_the_seven(self):
        for name in SEVEN:
            self.assertEqual(V2.contract(name), 'titration-tasks-v2')
            self.assertEqual(v1_contract(name), 'titration-tasks-v1/'+name)
        self.assertEqual(V2.contract('titration'), REWARD_CONTRACT)
        self.assertEqual(v1_contract('titration'), REWARD_CONTRACT)

    def test_v2_selections_resolve_as_released_with_the_v2_contract(self):
        from gym_anything_robotics import tasks
        from gym_anything_robotics.tasks import TaskSpec
        ids = set()
        for s in V2.TASK_SET:
            spec = tasks.resolve('titration', s)
            self.assertEqual(TaskSpec.from_dict(spec.to_dict()).to_dict(), spec.to_dict())
            ids.add(spec.task_id)
            got = benchmark.resolve(s)
            if 'contract' not in s:
                self.assertEqual(spec.task_id, 'titration/titration-n15-0000')
                self.assertNotIn('contract', got['backend_options'])
                continue
            old = benchmark.resolve({k: v for k, v in s.items() if k != 'contract'})
            self.assertEqual(old['task_id'], f"titration/{s['task']}/titration-n15-0000")
            self.assertEqual(got['task_id'], f"titration/{s['task']}/{V2.CONTRACT}/titration-n15-0000")
            self.assertEqual((got['description'], got['split']), (old['description'], old['split']))
            self.assertEqual(got['backend_options'], {**old['backend_options'], 'contract': V2.CONTRACT})
            self.assertEqual(old['selection']['task_contract'], v1_contract(s['task']))
            self.assertEqual(got['selection'], {**old['selection'], 'task_contract': V2.CONTRACT})
        self.assertEqual(len(ids), 8)

    def test_selections_without_contract_never_reach_v2(self):
        with patch.object(V2, 'resolve', side_effect=AssertionError('released selection reached v2')):
            for s in (*V1_TASK_SET, {}, {'variation_id': 7}, {'task': 'run-in', 'constraints': {'aliquot_ml': 12.}}):
                benchmark.resolve(s)

    def test_contract_refusals(self):
        for bad in ({'contract': 'titration-tasks-v1'}, {'contract': None}, {'contract': 'bogus', 'task': 'run-in'},
                    {'contract': 'titration-tasks-v1/run-in', 'task': 'run-in'}, {'contract': V2.CONTRACT},
                    {'contract': V2.CONTRACT, 'task': 'titration', 'variation_id': 0},
                    {'contract': V2.CONTRACT, 'task': 'run_in'}, {'contract': V2.CONTRACT, 'task': 'run-in', 'extra': 0},
                    {'contract': V2.CONTRACT, 'task': 'run-in', 'variation_id': -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                benchmark.resolve(bad)


class EvaluationTest(unittest.TestCase):
    def test_scores_are_the_released_ones_under_the_single_contract(self):
        for name, changes in SUCCESS.items():
            plain, m = evaluator(name)
            w, _ = wrapped(name)
            for t in range(32):
                a, b = plain.update(t, **{**m, **changes}), w.update(t, **{**m, **changes})
                self.assertEqual(b, relabelled(a))
            self.assertEqual((b['success'], b['score'], b['contract'], b['reward']['contract']),
                             (True, 1., V2.CONTRACT, V2.CONTRACT), name)
            self.assertEqual(w.result, b)
            self.assertEqual(w.name, name)

    def test_every_start_and_idle_is_unsuccessful_zero(self):
        for name in SUCCESS:
            w, m = wrapped(name)
            for t in range(40):
                out = w.update(t, **m)
                self.assertEqual((out['success'], out['score'], out['contract']), (False, 0., V2.CONTRACT), (name, t))

    def test_checkpoint_roundtrip_carries_the_contract(self):
        for name, changes in SUCCESS.items():
            w, m = wrapped(name, dt=.25)
            m.update(changes)
            for t in range(30):
                w.update(t, **m)
            snap = json.loads(json.dumps(w.snapshot()))
            self.assertEqual((snap['contract'], snap['result']['contract'], snap['result']['reward']['contract']),
                             (V2.CONTRACT,)*3)
            other, _ = wrapped(name, dt=.25)
            other.validate(snap)
            other.restore(snap)
            for t in range(30, 60):
                self.assertEqual(w.update(t, **m), other.update(t, **m))
            # A released checkpoint is not this contract's, nor the reverse; refusals change nothing.
            plain, _ = evaluator(name, dt=.25)
            for t in range(30):
                plain.update(t, **m)
            fresh, _ = wrapped(name, dt=.25)
            before = fresh.snapshot()
            with self.assertRaises(ValueError):
                fresh.restore(json.loads(json.dumps(plain.snapshot())))
            self.assertEqual(fresh.snapshot(), before)
            with self.assertRaises(ValueError):
                evaluator(name, dt=.25)[0].restore(snap)
            bad = copy.deepcopy(snap)
            bad['result']['score'] = .123
            with self.assertRaises(ValueError):
                other.restore(bad)
            with self.assertRaises(ValueError):
                other.update(150, **m)


class WiringTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        Session.instances = []
        self.stack.enter_context(patch.dict(sys.modules, {
            'titrate.newton15.session': NS(ApparatusSession=Session),
            'titrate.newton15.sph': NS(SphNumerics=lambda **kw: NS(**kw)),
            'titrate.newton15.laboratory_liquid': NS(LaboratoryLiquid=Liquid),
            'titrate.newton15.task_observations': NS(CatalogueReadings=Readings),
            'titrate.newton15.geometry_api': NS(GeometryCatalog=lambda world: NS()),
            'titrate.newton15.session_trace': NS(SessionTrace=Mock()),
            'whiteboard.storage': NS(Store=Mock()),
        }))
        self.stack.enter_context(patch('titrate.newton15.task_session.apply_rigid_start'))
        self.numerics = NS(spacing_m=.000625, dt_s=.000125, device='cuda:0')

    def backend(self, task='centre-flask', **kwargs):
        from gym_anything_robotics.backends.newton_titration import NewtonTitrationBackend
        return NewtonTitrationBackend(task=task, variation_id=0, liquid_numerics=self.numerics,
                                      recording=False, cameras=(), **kwargs)

    def test_factory_wraps_the_seven_and_leaves_titration_released(self):
        for name in TASKS:
            s = V2.make_session(LabConfig(), task=name, dt_s=.000125, liquid_numerics=self.numerics)
            if name == 'titration':
                self.assertNotIsInstance(getattr(s, 'evaluator', None), V2.ContractEvaluator)
                self.assertFalse(hasattr(s, 'task_setup'))
                continue
            self.assertIsInstance(s.evaluator, V2.ContractEvaluator)
            self.assertIs(type(s.evaluator.inner), CatalogueEvaluator)
            self.assertEqual((s.evaluator.result['score'], s.evaluator.result['contract']), (0., V2.CONTRACT))
            self.assertEqual(s._initial['contract'], V2.CONTRACT)
            state = s.reset()['evaluation']
            self.assertEqual((state['score'], state['contract']), (0., V2.CONTRACT))

    def test_backend_contract_dispatch_reset_and_preview(self):
        b = self.backend(contract=V2.CONTRACT)
        self.assertEqual(b.contract, V2.CONTRACT)
        self.assertIsInstance(b.session.evaluator, V2.ContractEvaluator)
        self.assertEqual(b.get_state()['evaluation']['contract'], V2.CONTRACT)
        preview = b.select_variation({'task': 'finish-titration', 'variation_id': 25})
        self.assertEqual(preview['task_contract'], V2.CONTRACT)
        b.reset(**preview['reset_request'])
        self.assertEqual(b.task, 'finish-titration')
        self.assertEqual(b.session.evaluator.result['contract'], V2.CONTRACT)
        b.reset(options={'task': 'titration'})
        self.assertNotIsInstance(getattr(b.session, 'evaluator', None), V2.ContractEvaluator)
        b.reset(options={'task': 'run-in'})
        self.assertEqual(b.reset()[0]['evaluation']['contract'], V2.CONTRACT)
        # Without a contract the backend is the released one.
        old = self.backend()
        self.assertIsNone(old.contract)
        self.assertIs(type(old.session.evaluator), CatalogueEvaluator)
        self.assertEqual(old.get_state()['evaluation']['contract'], 'titration-tasks-v1/centre-flask')
        self.assertEqual(old.select_variation({'task': 'run-in', 'variation_id': 0})['task_contract'],
                         'titration-tasks-v1/run-in')
        for bad in ('titration-tasks-v1', 'titration-tasks-v1/centre-flask', 'bogus'):
            with self.subTest(contract=bad), self.assertRaises(ValueError):
                self.backend(contract=bad)

    def test_resolved_v2_options_fit_the_backend(self):
        from gym_anything_robotics.backends.newton_titration import NewtonTitrationBackend
        for s in V2.TASK_SET:
            options = benchmark.resolve(s)['backend_options']
            inspect.signature(NewtonTitrationBackend).bind(**options)
            self.assertEqual(options.get('contract'), V2.CONTRACT if 'contract' in s else None)


if __name__ == '__main__':
    unittest.main()

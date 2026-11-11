"""sewing-tasks-v2, the render-checked eight-task set: synthetic ledgers, geometry and a stubbed adapter only.

Never imports a simulator runtime. The ledger fixtures are the released task-set test's
(tests/environments/sewing/test_task_set.py); corner-seam, which only this contract defines, takes the pivot's setup.
"""
import ast
import dataclasses
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from gym_anything_robotics import tasks
from sewing import task, task_set as V1, task_set_v2 as V2, variations
from sewing import task_set_scoring
from sewing.task_set_v2 import CONTRACT, TASK_SET, TASKS
from sewing.task_set_v2_scoring import TaskEvaluator, failure_factor, seam_path
from tests.environments.sewing import test_task_set as T
from tests.packages import PACKAGES, REPO

ROOT = PACKAGES
BENCH = json.loads((ROOT/'sewing/benchmark.json').read_text())
SEAM_TASKS = ('plain-seam', 'seam-to-notch', 'pivot', 'corner-seam', 'load-and-seam')
RELEASED_DIGEST = 'ff4967564c0ea08724149e11af28d848c8dca20eef58af4f58f74d8acdc55a5f'


@pytest.fixture(autouse=True)
def corner_seam_fixtures():
    # The released ledger fixtures read task.TASK_SETUP and the released seam_path; corner-seam is the pivot's plies
    # on v2's path. Only the fixtures see this: the evaluator and the resolver are the modules under test.
    with patch.dict(task.TASK_SETUP, {'corner-seam': task.TASK_SETUP['pivot']}), patch.object(T, 'seam_path', seam_path):
        yield


def evaluator(env):
    ev = TaskEvaluator(env)
    if env.config.task == 'load-and-seam':
        ev.load_progress = lambda: 0.
    return ev


def test_benchmark_lists_the_eight_and_keeps_the_released_set():
    v2 = BENCH['task_set']
    assert v2['selections'] == [dict(s) for s in TASK_SET] and v2['note']
    assert [s['task'] for s in v2['selections']] == ['load', 'seam-to-notch', 'plain-seam', 'square-plies', 'pivot',
                                                     'corner-seam', 'load-and-seam', 'curve']
    split = json.loads((REPO/'benchmarks/roboworld/splits/test.json').read_text())['tasks']
    assert v2['selections'][-1] in [t['selection'] for t in split if t['environment_id'] == 'sewing']
    assert CONTRACT in BENCH['selector_choices']['contract'] and 'corner-seam' in BENCH['selector_choices']['task']
    for e in BENCH['task_set_v1']['tasks']:                      # the released set resolves as released
        r = variations.task_resolve(e['selection'])
        assert r['task_id'] == e['task_id'] and r == V1.task_resolve(e['selection'])
        assert V1.choose_contract(r['selection']['task'], r['selection'].get('contract')) == e['task_contract']
    # the released catalogue is unchanged: 90 entries, the same digest, no corner-seam
    assert variations.COUNT == 90 and variations.digest() == RELEASED_DIGEST
    assert 'corner-seam' not in task.TASKS and 'corner-seam' not in variations.TASK_ORDER
    assert tasks.resolve('sewing').task_id == BENCH['default_task_id']


def test_the_eight_resolve_with_their_own_ids():
    ids = []
    for s in TASK_SET:
        spec = tasks.resolve('sewing', s)
        assert tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict() == spec.to_dict()
        if s['task'] == 'curve':
            assert spec.task_id == 'sewing/v1/sample-room/curve/seed-0' and spec.selection == dict(s, preset='sample-room', seed=0)
            assert spec.description == task.INSTRUCTIONS['curve']
        else:
            assert spec.task_id == f"sewing/v1/sample-room/{s['task']}/{CONTRACT}/seed-0"
            assert spec.selection == spec.launch.backend_options == s
            want = (V1.PIVOT_INSTRUCTION if s['task'] == 'pivot' else V2.CORNER_SEAM_INSTRUCTION
                    if s['task'] == 'corner-seam' else task.INSTRUCTIONS[s['task']])
            assert spec.description == want
        ids.append(spec.task_id)
    released = {e['task_id'] for e in BENCH['task_set_v1']['tasks']}
    assert len(set(ids)) == 8 and not (set(ids[:-1]) & released) and BENCH['default_task_id'] not in ids


@pytest.mark.parametrize('name', TASKS)
def test_contract_selection_by_task_or_variation_id(name):
    new = V1.task_resolve(dict(task=name, contract=CONTRACT))
    assert new == V2.task_resolve(dict(task=name, contract=CONTRACT))
    assert new['task_id'] == f'sewing/v1/sample-room/{name}/{CONTRACT}/seed-0'
    assert new['backend_options'] == dict(preset='sample-room', task=name, seed=0, contract=CONTRACT)
    idx = variations.index_of(name, 0)
    entry = variations.specification(idx, CONTRACT)
    assert entry['task_id'] == new['task_id'] and entry['contract'] == CONTRACT
    assert entry.get('task_contract', CONTRACT) == CONTRACT
    r = variations.task_resolve(dict(variation_id=idx, contract=CONTRACT))
    assert r['task_id'] == new['task_id'] and r['backend_options'] == new['backend_options']
    if name != 'corner-seam':                                   # the released entry and launch are untouched
        assert idx < variations.COUNT and 'contract' not in variations.specification(idx)
        assert V1.task_resolve(dict(task=name))['task_id'] == f'sewing/v1/sample-room/{name}/seed-0'
        assert variations.config_values(entry) == variations.config_values(variations.specification(idx))
    else:
        assert idx == 90 and entry['physical_task'] == 'pivot' and entry['time_limit_s'] == 180.
        pivot = variations.config_values(variations.specification(variations.index_of('pivot', 0)))
        assert variations.config_values(entry) == dict(pivot, task='corner-seam')


@pytest.mark.parametrize('selection', [dict(task='corner-seam'), dict(task='A14'), dict(task='corner-seam', contract=V1.CONTRACT),
                                       dict(task='curve', contract=CONTRACT), dict(task='baste', contract=CONTRACT),
                                       dict(task='three-sides', contract=CONTRACT), dict(task='load', seed=True, contract=CONTRACT),
                                       dict(variation_id=90), dict(variation_id=100, contract=CONTRACT),
                                       dict(variation_id=40, contract=CONTRACT), dict(variation_id=90, task='pivot', contract=CONTRACT),
                                       dict(task='load', contract='sewing-tasks-v3')])
def test_invalid_selections(selection):
    with pytest.raises(ValueError):
        variations.task_resolve(selection)


def test_corner_seam_code_and_seeds():
    assert V2.task_resolve(dict(task='A14', seed=4, contract=CONTRACT)) == V2.task_resolve(dict(task='corner-seam', seed=4, contract=CONTRACT))
    for seed in range(10):
        c, p = V2.resolve(None, 'corner-seam', seed, CONTRACT), V2.resolve(None, 'pivot', seed, CONTRACT)
        assert c == dict(p, task='corner-seam')                 # the pivot start at every seed
        assert variations.index_of('corner-seam', seed) == 90 + seed


@pytest.mark.parametrize('name', SEAM_TASKS)
def test_good_seams_and_untouched_starts(name):
    good = evaluator(T.good_env(name)).evaluate()
    assert good['success'] and good['score'] == 1 and good['task_contract'] == CONTRACT, good
    empty = evaluator(T.fake_env(name)).evaluate()
    assert not empty['success'] and empty['score'] == 0 and empty['task_contract'] == CONTRACT


@pytest.mark.parametrize('name', SEAM_TASKS)
def test_reverse_sewing_order_is_valid(name):
    env = T.good_env(name)
    env.world.machine.stitches.reverse()
    assert evaluator(env).evaluate()['success']


def test_corner_seam_is_the_pivot_path_with_both_ends_backtacked():
    assert np.array_equal(seam_path('corner-seam', .2, .2), seam_path('pivot', .2, .2))
    assert V2.BASE_TASK['corner-seam'] == 'pivot' and V2.TIME_LIMIT_S['corner-seam'] == 180.
    good = evaluator(T.good_env('corner-seam')).evaluate()
    assert {'corner_1', 'backtack_start', 'backtack_end'} <= set(good['checks'])
    # the pivot's ledger (no reverse stitches) is not a corner-seam: both locks are missing
    r = evaluator(T.fake_env('corner-seam', T.seam_points('corner-seam'))).evaluate()
    assert not r['success'] and not r['checks']['backtack_start']['ok'] and not r['checks']['backtack_end']['ok']
    assert 0 < r['score'] < .95 and r['score'] == pytest.approx(.95*r['info']['continuous_progress'])
    env = T.good_env('corner-seam')
    corner = np.array([.19, .01])
    for h in env.world.machine.holes:
        if np.linalg.norm(np.array(h.uv) - corner) < .004:
            h.uv = tuple(np.array(h.uv) + [-.004, .004])
    r = evaluator(env).evaluate()
    assert not r['checks']['corner_1']['ok'] and not r['success']


def test_failure_factor_follows_the_spec():
    for name in ('load', 'square-plies'):
        assert failure_factor(name) == .9
    for name in ('seam-to-notch', 'plain-seam', 'pivot', 'three-sides', 'load-and-seam', 'corner-seam'):
        assert failure_factor(name) == task.CURVE_PARTIAL_MAX == .95
    env = T.fake_env('load'); ev = evaluator(env)
    ev.ply_registration = lambda: (0., 100)
    ev.under_needle = lambda k: [(.186, .415)]
    r = ev.evaluate()
    assert not r['success'] and r['score'] == pytest.approx(.9*r['info']['continuous_progress']) and r['score'] > 0
    released = task_set_scoring.TaskEvaluator(env)
    released.ply_registration, released.under_needle = ev.ply_registration, ev.under_needle
    assert released.evaluate()['score'] == pytest.approx(.95*r['info']['continuous_progress'])   # v1 kept 0.95
    env = T.fake_env('square-plies', offset=(.040, .025)); ev = evaluator(env)
    env.positions[1][:, :2] -= [.020, .0125]
    r = ev.evaluate()
    assert not r['success'] and r['score'] == pytest.approx(.9*r['info']['continuous_progress']) and r['score'] > 0
    r = evaluator(T.fake_env('plain-seam', T.seam_points('plain-seam')[:25])).evaluate()
    assert r['score'] == pytest.approx(.95*r['info']['continuous_progress'])


def _start_mesh(name):
    # As world.py _build_cloth meshes the physical base task's plies (the pivot adds its near edge to the band).
    from sewing import params
    from sewing.mesh import rect_panel, mesh_panel
    base = V2.BASE_TASK.get(name, name)
    c = task.TASK_SETUP[base]
    W, L, A = c['ply_width'], c['ply_length'], c.get('seam_allowance', .010)
    near = [np.array([[0., 0.], [W, 0.]]), np.array([[0., A], [W, A]])]
    return mesh_panel(rect_panel('ply', W, L, seam_offset=A, extra_curves=near if base == 'pivot' else (),
                                 band=params.FINE_BAND))


def _coarsest_triangle_on_path(mesh, path, lateral=.0016):
    from sewing.task import _triangle_under
    X, Tri = np.asarray(mesh['X']), np.asarray(mesh['tri'])
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.linalg.norm(b - a)/.001))
        d = (b - a)/np.linalg.norm(b - a)
        for t in np.linspace(0., 1., n):
            for off in (-lateral, 0., lateral):
                pts.append(a + t*(b - a) + off*np.array([-d[1], d[0]]))
    pts = np.clip(np.array(pts), X.min(axis=0) + 1e-6, X.max(axis=0) - 1e-6)
    t, *_ = _triangle_under(pts, X[Tri[:, 0]], X[Tri[:, 1]], X[Tri[:, 2]])
    e = np.max([np.linalg.norm(X[Tri[t, i]] - X[Tri[t, (i + 1) % 3]], axis=1) for i in range(3)], axis=0)
    return float(e.max())


def test_every_selected_seam_lies_in_the_stitchable_band_and_three_sides_does_not():
    # machine.py ends the episode as a world fault when the needle pierces a triangle coarser than 2 FINE_H.
    from sewing import params
    limit = 2*params.FINE_H
    for name in SEAM_TASKS:
        assert name in TASKS
        c = task.TASK_SETUP[name]
        assert _coarsest_triangle_on_path(_start_mesh(name), seam_path(name, c['ply_width'], c['ply_length'])) <= limit
    c = task.TASK_SETUP['three-sides']
    path = seam_path('three-sides', c['ply_width'], c['ply_length'])
    mesh = _start_mesh('three-sides')
    assert _coarsest_triangle_on_path(mesh, path[:3]) <= limit
    assert _coarsest_triangle_on_path(mesh, path[2:]) > 2*limit
    assert 'three-sides' not in TASKS


def test_v2_session_variation_contract():
    entry = variations.specification(95, CONTRACT)
    c = variations.contract(dict(variation=entry, task='corner-seam', seed=5))
    assert (c['count'], c['integer_id_range'], c['tasks'][-1]) == (100, [0, 99], 'corner-seam')
    assert c['task_set'] == [dict(s) for s in TASK_SET] and c['task_contracts']['task_set'] == CONTRACT
    released = variations.contract(dict(variation=variations.specification(10), task='plain-seam', seed=0))
    assert (released['count'], released['task_set']) == (90, [dict(s) for s in V1.TASK_SET])
    assert variations.select(task='corner-seam', seed=5, contract=CONTRACT) == entry
    assert variations.select(variation_id=95, task='A14', seed=5, contract=CONTRACT) == entry
    assert variations.select_request(dict(variation_id=10, contract=CONTRACT))['task_id'] == \
        f'sewing/v1/sample-room/plain-seam/{CONTRACT}/seed-0'


@pytest.fixture
def adapters():
    """task_set_env.py and task_set_v2_env.py over a stubbed SewingEnv and the real Config (no engine)."""
    tree = ast.parse((ROOT/'sewing/world.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config')
    from sewing import params, layout
    ns = dict(dataclass=dataclasses.dataclass, P=params, LY=layout)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'Config', 'exec'), ns)
    world = ModuleType('sewing.world'); world.Config = ns['Config']
    base = ModuleType('sewing.env')

    class Base:
        def __init__(self, config, output_dir=None):
            self.config = config; self.start_state = config.start; self.assisted_log = []
            self.world = NS(config=config, cloth=None)
            self.evaluator = task.Evaluator(self)
            self.sim_time = 0.
            self._make_start()

        def _make_start(self, use_cache=True): self.prepared_config = self.config

        def get_state(self, evaluate=True):
            return dict(instruction=task.INSTRUCTIONS.get(self.config.task), time_s=self.sim_time,
                        terminated=self.sim_time >= task.TIME_LIMIT_S.get(self.config.task, 1e9), evaluation={})
    base.SewingEnv = Base
    mods = {}
    # sys.modules is restored afterwards; the released adapter's make_environment imports task_set_v2_env when called
    with patch.dict(sys.modules, {'sewing.world': world, 'sewing.env': base}):
        for short in ('task_set_env', 'task_set_v2_env'):
            spec = importlib.util.spec_from_file_location(f'sewing.{short}', ROOT/f'sewing/{short}.py')
            mods[short] = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mods[short])
        yield mods, ns['Config']


def test_v2_adapter_builds_the_physical_start_and_judges_under_v2(adapters):
    mods, Config = adapters
    v1, v2 = mods['task_set_env'], mods['task_set_v2_env']
    for name, physical in [('load', 'load'), ('seam-to-notch', 'plain-seam'), ('plain-seam', 'plain-seam'),
                           ('pivot', 'pivot'), ('corner-seam', 'pivot'), ('load-and-seam', 'load')]:
        c = Config(**V2.resolve(None, name, 0, CONTRACT), robot='g1')
        e = v1.make_environment(c, contract=CONTRACT)               # the released entry point hands v2 over
        assert type(e) is v2.TaskEnvironment and e.task_contract == CONTRACT
        assert e.config.task == name and e.world.config.task == physical == e.prepared_config.task
        assert type(e.evaluator).__module__.endswith('task_set_v2_scoring')
        s = e.get_state()
        assert s['instruction'] == V2.task_resolve(dict(task=name, contract=CONTRACT))['description'] + task.G1_MACHINE_NOTE
        assert s['task_contract'] == s['evaluation']['task_contract'] == CONTRACT
        e.sim_time = 150.
        assert e.get_state()['terminated'] == (V2.TIME_LIMIT_S[name] <= 150.)
    assert V2.TIME_LIMIT_S['corner-seam'] == 180.
    # released selections keep the released adapter
    legacy = v1.make_environment(Config(task='plain-seam'))
    assert type(legacy) is v1.TaskEnvironment and type(legacy.evaluator) is task.Evaluator
    released = v1.make_environment(Config(task='pivot'), contract=V1.CONTRACT)
    assert type(released) is v1.TaskEnvironment and released.task_contract == V1.CONTRACT
    with pytest.raises(ValueError):
        v1.make_environment(Config(**V2.resolve(None, 'corner-seam', 0, CONTRACT)))
    with pytest.raises(ValueError):
        v2.make_environment(Config(task='curve'), contract=CONTRACT)

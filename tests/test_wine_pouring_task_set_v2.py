"""wine-tasks-v2, the render-checked eight-task set, on constructed state only. Never import/build a simulator.

The state fixture is the released task-set test's (tests/test_wine_pouring_task_set.py).
"""
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from wine_pouring import task as LEGACY
from wine_pouring import task_contracts as T
from wine_pouring import task_set_v2 as V2
from wine_pouring.config import TASKS
from wine_pouring.contracts import TaskConfig as Config
from wine_pouring.contracts import (LEGACY_REWARD_CONTRACT, SPLIT_REWARD_CONTRACT, TASKS_V1_REWARD_CONTRACT,
                                    PROGRESS_TASKS, eight_task_selections, reward_contract)
from wine_pouring.variants import config_values, contract, resolve, task_resolve
from tests.packages import PACKAGES, REPO
from tests.test_wine_pouring_task_set import StateEnv

ROOT = PACKAGES
BENCH = json.loads((ROOT/'wine_pouring/benchmark.json').read_text())
SELECTIONS = V2.selections()
CONTRACT = V2.CONTRACT


def test_benchmark_lists_the_eight_and_keeps_the_released_set():
    assert CONTRACT == 'wine-tasks-v2'
    v2 = BENCH['task_set']
    assert v2['selections'] == SELECTIONS and v2['note']
    assert [(s.get('preset'), s['task']) for s in SELECTIONS] == [(p, t) for p, t, _ in __import__(
        'wine_pouring.contracts', fromlist=['x']).EIGHT_TASKS]
    split = json.loads((REPO/'benchmarks/roboworld/splits/test.json').read_text())['tasks']
    assert SELECTIONS[-1] in [t['selection'] for t in split if t['environment_id'] == 'wine-pouring']
    for e in BENCH['task_set_v1']['tasks']:                    # the released set resolves as released
        assert task_resolve(e['selection'])['task_id'] == e['task_id']
    assert [e['selection'] for e in BENCH['task_set_v1']['tasks']] == eight_task_selections() == contract()['task_set']


def test_the_eight_resolve_with_their_own_ids():
    ids = []
    for s in SELECTIONS:
        r = task_resolve(s)
        bare = {k: v for k, v in s.items() if k != 'contract'}
        if s['task'] == 'split-bottle':
            assert r['task_id'] == 'wine-pouring/v1/service/split-bottle/seed-0' == task_resolve(bare)['task_id']
        else:
            assert r['task_id'] == f"wine-pouring/v1/{s['preset']}/{s['task']}/{CONTRACT}/seed-0"
            assert r['selection'] == r['backend_options'] == s
            v1 = task_resolve(dict(s, contract=TASKS_V1_REWARD_CONTRACT))
            assert r['task_id'] != v1['task_id'] == task_resolve(bare)['task_id']
            assert r['description'] == v1['description']
            assert r['split'] == v1['split']
            assert Config(**config_values(resolve(**s))).task_contract == CONTRACT
        ids.append(r['task_id'])
    assert len(set(ids)) == 8 and BENCH['default_task_id'] not in ids
    r = task_resolve(dict(task='lined-measure', seed=123, contract=CONTRACT))
    assert r['task_id'] == f'wine-pouring/v1/lined/lined-measure/{CONTRACT}/seed-123'


@pytest.mark.parametrize('selection', [
    {'task': 'split-bottle', 'contract': CONTRACT}, {'task': 'decant-carafe', 'contract': CONTRACT},
    {'task': 'tasting-flight', 'contract': CONTRACT}, {'task': 'pour-glass', 'seed': True, 'contract': CONTRACT},
    {'preset': 'lined', 'task': 'pour-glass', 'contract': CONTRACT}, {'task': 'pour-glass', 'contract': 'wine-tasks-v3'},
    {'task': 'pour-glass', 'contract': CONTRACT, 'typo': 1}])
def test_invalid_selection_rejected(selection):
    with pytest.raises(ValueError):
        task_resolve(selection)


def test_contract_choices():
    for names in TASKS.values():
        for t in names:
            if t in PROGRESS_TASKS:
                assert reward_contract(t, CONTRACT) == CONTRACT
            else:
                with pytest.raises(ValueError):
                    reward_contract(t, CONTRACT)
            assert reward_contract(t) in (LEGACY_REWARD_CONTRACT, SPLIT_REWARD_CONTRACT)
    c = V2.contract()
    assert c['task_set'] == SELECTIONS and c['reward_contracts']['pour-glass'][-1] == CONTRACT
    assert c['reward_contracts']['split-bottle'] == [SPLIT_REWARD_CONTRACT]


@pytest.mark.parametrize('selection', SELECTIONS, ids=lambda s: s['task'])
@pytest.mark.parametrize('seed', [0, 1, 7, 131, 9999])
def test_untouched_is_zero_and_success_is_one(selection, seed):
    env = StateEnv(dict(selection, seed=seed))
    initial = env.evaluator.evaluate()
    assert not initial['success'] and initial['score'] == 0.
    if selection['task'] != 'split-bottle':
        assert initial['reward']['progress'] == 0. and initial['reward']['contract'] == CONTRACT
    assert env.finish()['success']
    assert env.evaluator.evaluate()['score'] == 1.


@pytest.mark.parametrize('selection', SELECTIONS[:-1], ids=lambda s: s['task'])
def test_same_verdict_as_wine_tasks_v1_but_its_own_contract(selection):
    a, b = StateEnv(dict(selection, contract=TASKS_V1_REWARD_CONTRACT)), StateEnv(selection)
    for step in range(2):                                      # at reset, then finished
        if step:
            a.finish(), b.finish()
        ra, rb = a.evaluator.evaluate(), b.evaluator.evaluate()
        assert ra['reward'].pop('contract') == TASKS_V1_REWARD_CONTRACT and rb['reward'].pop('contract') == CONTRACT
        assert ra == rb and ra['success'] == bool(step)
        assert T.instruction(a.config) == T.instruction(b.config)


def test_backend_reset_select_and_checkpoint_identity_under_v2(monkeypatch):
    from wine_pouring.task_backend import NewtonWinePouringBackend

    class FakeEnv:
        def __init__(self, config, **kw):
            self.config, self.world, self.lock, self.generation = config, object(), threading.RLock(), 0

        def reset(self):
            return 'reset'

        def get_state(self):
            return 'state'
    stub = ModuleType('wine_pouring.kinematics')
    stub.ArmKinematics = lambda w: 'ik'
    monkeypatch.setitem(sys.modules, 'wine_pouring.kinematics', stub)
    backend = NewtonWinePouringBackend.__new__(NewtonWinePouringBackend)
    backend._make_env = lambda config: FakeEnv(config)
    backend.config = Config()
    backend.env = FakeEnv(backend.config)
    backend.trace = None
    backend.output = None
    backend._end_operation = lambda **kw: None
    backend.capabilities = SimpleNamespace(notes={})
    legacy_key = backend._variation_key()
    assert backend.variation_contract()['task_set'] == eight_task_selections()     # a released session: as released
    selected = backend.select_variation(SELECTIONS[2])
    assert selected['reset_request']['options']['contract'] == CONTRACT
    assert selected['task'] == task_resolve(SELECTIONS[2])['description']
    backend.reset(**selected['reset_request'])
    assert backend.config.task_contract == CONTRACT
    v2_key = backend._variation_key()
    assert v2_key != legacy_key and v2_key['contract'] == CONTRACT
    vc = backend.variation_contract()
    assert vc['current']['contract'] == CONTRACT and vc['task_set'] == SELECTIONS
    backend.reset(seed=3)
    assert backend.config.task_contract == CONTRACT and backend.config.seed == 3
    backend.reset(options={'contract': TASKS_V1_REWARD_CONTRACT})
    assert backend._variation_key() not in (v2_key, legacy_key)
    backend.reset(options={'contract': CONTRACT})
    original = backend.env
    with pytest.raises(ValueError):
        backend.reset(options={'task': 'split-bottle', 'contract': CONTRACT})
    assert backend.env is original
    backend.reset(options={'task': 'taste-pour'})
    assert backend.config.task_contract == ''


def test_registry_launch_judges_under_v2(monkeypatch):
    from gym_anything_robotics import registry
    from gym_anything_robotics.policy_access import PolicyAccess
    from wine_pouring.config import Config as PhysicsConfig
    from wine_pouring.task_backend import NewtonWinePouringBackend, PhysicsBackend

    def construct(backend, *, config=None, preset='service', task=None, seed=0):
        assert 'task_contract' not in config
        physics_config = PhysicsConfig(**dict(config, preset=preset, task=task, seed=seed))
        env = StateEnv(dict(preset=preset, task=task, seed=seed))
        env.config, env.lock, env.world.config = physics_config, threading.RLock(), physics_config
        env.evaluator = LEGACY.Evaluator(env)
        backend.config, backend.env = physics_config, env
    monkeypatch.setattr(PhysicsBackend, '__init__', construct)
    monkeypatch.setattr(PhysicsBackend, 'policy_access', lambda b: PolicyAccess(
        b.config.robot, (), {}, frozenset(), lambda: LEGACY.instruction(b.config), {}))
    for selection in SELECTIONS[:-1]:
        backend = registry._newton_wine_pouring_factory(**task_resolve(selection)['backend_options'])
        assert isinstance(backend, NewtonWinePouringBackend) and isinstance(backend.env.evaluator, T.Evaluator)
        assert backend.config.task_contract == CONTRACT
        assert backend.policy_access().instruction() == task_resolve(selection)['description']
        result = backend.evaluate()
        assert result['reward']['contract'] == CONTRACT and result['score'] == 0. and not result['success']

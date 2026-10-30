"""Eight-task contracts on constructed state only. Never import/build a simulator.

Run with Python 3.12 + numpy + pytest; robots/Newton/Warp/Isaac are not required.
"""
import json
import math
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wine_pouring import params as P
from wine_pouring import task as LEGACY
from wine_pouring import task_contracts as T
from wine_pouring.config import TASKS
from wine_pouring.contracts import TaskConfig as Config
from wine_pouring.contracts import (EIGHT_TASKS, LEGACY_REWARD_CONTRACT, SPLIT_REWARD_CONTRACT,
                                    TASKS_V1_REWARD_CONTRACT, eight_task_selections, reward_contract)
from wine_pouring.layout import task_layout
from wine_pouring.variants import config_values, contract, resolve, task_resolve
from tests.packages import PACKAGES

ROOT = PACKAGES
SELECTIONS = eight_task_selections()


class StateEnv:
    """Only the evaluator's inventory, pose and contact interface; no dynamics."""
    def __init__(self, selection):
        self.config = Config(**config_values(selection))
        self.layout = task_layout(self.config)
        self.positions = {i['name']: np.array([*i['xy'], P.BAR_TOP_Z_M]) for i in self.layout['items']}
        self.angles = {}
        self.world = SimpleNamespace(fault=None, body_pose=self.body_pose)
        self.inv = dict(containers={}, bar={'total': 0.}, floor={'total': 0.}, elsewhere={'total': 0.},
                        film={}, speed_mean_m_s=0.)
        for item in self.layout['items']:
            self.set_volume(item['name'], item.get('liquid', 'red_wine'), item.get('volume', 0.)*1e6)
        self.contacts = {}
        self.still = 9.
        self.source_touch = dict(max_n=0., ticks=0)
        self.evaluator = T.Evaluator(self)

    def body_pose(self, name):
        a = math.radians(self.angles.get(name, 0.))
        return self.positions[name], np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                                              [-math.sin(a), 0, math.cos(a)]])

    def inventory(self):
        return self.inv

    def robot_contacts(self):
        return self.contacts

    def robot_still_s(self):
        return self.still

    def geometry(self, name):
        # Only the lined glass's label capacity; no mesh or runtime imported.
        return SimpleNamespace(lines={'line_175': .1}, volume_below=lambda z: 175e-6)

    def set_volume(self, name, liquid, amount):
        self.inv['containers'][name] = {liquid: amount, 'total': amount}

    def finish(self):
        g, task = self.layout['goal'], self.config.task
        src = next((i for i in self.layout['items'] if i.get('role') == 'source'), None)
        if task == 'split-bottle':
            amount = (self.evaluator.source_initial_mL - 8.)/2
            for name in g['targets']:
                self.set_volume(name, src['liquid'], amount)
            self.set_volume(src['name'], src['liquid'], 8.)
        elif task == 'carry-glass':
            self.positions[g['target']][:2] = P.DELIVERY_SPOTS[g['deliver_spot']]
        else:
            if task == 'dump-and-pour':
                self.set_volume(g['dump'], g['old'], self.inv['containers'][g['target']][g['old']])
            amount = 180. if 'line' in g else g['volume']*1e6
            initial = self.inv['containers'][g['target']].get(src['liquid'], 0.)
            self.set_volume(g['target'], src['liquid'], amount)
            self.set_volume(src['name'], src['liquid'], self.inv['containers'][src['name']]['total'] - amount + initial)
        return self.evaluator.evaluate()


@pytest.mark.parametrize('selection', SELECTIONS, ids=lambda s: s['task'])
@pytest.mark.parametrize('seed', [0, 1, 7, 131, 9999])
def test_untouched_is_zero_and_success_is_one(selection, seed):
    env = StateEnv(dict(selection, seed=seed))
    initial = env.evaluator.evaluate()
    assert not initial['success']
    assert initial['score'] == initial['reward']['progress'] == 0.
    assert initial['reward']['contract'] == selection['contract']
    assert env.finish()['success']
    assert env.evaluator.evaluate()['score'] == 1.
    json.dumps(env.evaluator.evaluate(), allow_nan=False)


def test_catalogue_and_resolver_keep_ids_defaults_and_all_ten_tasks():
    raw = json.loads((ROOT/'wine_pouring/benchmark.json').read_text())
    entries = raw['task_set']['tasks']
    assert [e['selection'] for e in entries] == SELECTIONS == contract()['task_set']
    assert raw['count'] == sum(map(len, TASKS.values())) == 10
    assert task_resolve({})['task_id'] == raw['default_task_id']
    assert task_resolve({})['backend_options'] == dict(preset='service', task='pour-glass', seed=0)
    for selection, entry in zip(SELECTIONS, entries):
        resolved = task_resolve(selection)
        assert resolved['selection'] == resolved['backend_options'] == selection
        assert resolved['task_id'] == entry['task_id']
        cfg = Config(**config_values(resolve(**selection)))
        assert cfg.task_contract == entry['task_contract']
        bare = {k: v for k, v in selection.items() if k != 'contract'}
        assert task_resolve(bare)['task_id'] == resolved['task_id']
        assert task_resolve(bare)['split'] == resolved['split']
        assert task_layout(Config(**bare)) == task_layout(cfg)
    assert reward_contract('split-bottle') == SPLIT_REWARD_CONTRACT
    assert reward_contract('pour-glass') == LEGACY_REWARD_CONTRACT
    assert resolve(None, 'lined-measure', 9999)['preset'] == 'lined'
    assert task_resolve({'task': 'tasting-flight'})['selection']['preset'] == 'tasting'
    assert task_resolve({'task': 'decant-carafe'})['selection']['preset'] == 'tasting'


@pytest.mark.parametrize('selection', [
    {'task': 'split-bottle', 'contract': TASKS_V1_REWARD_CONTRACT},
    {'task': 'decant-carafe', 'contract': TASKS_V1_REWARD_CONTRACT},
    {'task': 'pour-glass', 'contract': SPLIT_REWARD_CONTRACT},
    {'contract': 'unknown'}, {'contract': 0}, {'preset': 'lined', 'task': 'pour-glass'},
    {'seed': True}, {'seed': -1}, {'seed': 10000}, {'typo': 'x'},
])
def test_invalid_selection_rejected(selection):
    with pytest.raises(ValueError):
        task_resolve(selection)


def test_instructions_and_legacy_are_selectable():
    for selection in SELECTIONS:
        text = task_resolve(selection)['description']
        for tolerance in ('135-165', '225-275', 'within 3 cm', '42-58', 'at most 2', '175-185', '10 mL over'):
            assert tolerance not in text
    bare = task_resolve({'task': 'pour-glass'})
    assert '(135-165 mL)' in bare['description']
    env = StateEnv(bare['selection'])
    ev = env.evaluator.evaluate()
    assert ev['score'] > 0. and ev['reward']['contract'] == LEGACY_REWARD_CONTRACT
    assert ev['score'] == sum(ev['checks'].values())/len(ev['checks'])
    assert task_resolve(SELECTIONS[-1])['description'] == task_resolve({'task': 'split-bottle'})['description']


@pytest.mark.parametrize('selection', SELECTIONS, ids=lambda s: s['task'])
def test_failure_gates_and_world_fault(selection):
    env = StateEnv(selection)
    env.finish()
    env.contacts = {'left': True}
    ev = env.evaluator.evaluate()
    assert not ev['success'] and ev['score'] == pytest.approx(.9*ev['reward']['progress'], abs=5e-5)
    env.contacts = {'right': True}
    assert env.evaluator.evaluate()['score'] == ev['score']
    env.world.fault = 'constructed fault'
    ev = env.evaluator.evaluate()
    assert not ev['success'] and ev['score'] == ev['reward']['progress'] == 0.
    assert not any(ev['checks'].values())


@pytest.mark.parametrize('task,low,high', [('taste-pour', 30., 45.), ('pour-glass', 135., 165.),
                                           ('refill-glass', 135., 165.), ('water-tumbler', 225., 275.),
                                           ('lined-measure', 175., 185.)])
def test_continuous_pour_progress_and_overfill(task, low, high):
    env = StateEnv(next(s for s in SELECTIONS if s['task'] == task))
    name = env.layout['goal']['target']
    liquid = 'water' if task == 'water-tumbler' else 'red_wine'
    initial = env.inv['containers'][name].get(liquid, 0.)
    env.contacts = {'right': True}  # keep verdict failed across the entire fill window
    for fraction in [0., .1, .25, .333, .7, 1.]:
        amount = initial + fraction*(low-initial)
        env.set_volume(name, liquid, amount)
        ev = env.evaluator.evaluate()
        assert ev['score'] == pytest.approx(.9*fraction)
    env.set_volume(name, liquid, high + .5*(low-initial))
    ev = env.evaluator.evaluate()
    assert not ev['checks']['volume'] and ev['score'] == pytest.approx(.45)
    env.set_volume(name, liquid, high + low-initial)
    assert env.evaluator.evaluate()['score'] == pytest.approx(0.)
    env.set_volume(name, liquid, low)
    env.angles[name] = 11.
    assert env.evaluator.evaluate()['score'] == 0.


def test_refill_uses_actual_reset_inventory_not_requested_volume():
    env = StateEnv(SELECTIONS[3])
    env.set_volume('glass', 'red_wine', 73.008)  # actual simulated particle count can differ from layout volume
    env.evaluator = T.Evaluator(env)
    assert env.evaluator.evaluate()['score'] == 0.
    env.set_volume('glass', 'red_wine', (135. + 73.008)/2.)
    assert env.evaluator.evaluate()['score'] == pytest.approx(.45)
    assert env.evaluator.initial_containers['glass']['red_wine'] == 73.008


def test_carry_distance_from_actual_reset_and_retained_volume():
    env = StateEnv(SELECTIONS[0])
    env.positions['glass'][:2] += [.001, -.001]  # rigid settle, before evaluator construction
    env.evaluator = T.Evaluator(env)
    start = env.positions['glass'].copy()
    spot = np.asarray(P.DELIVERY_SPOTS[env.layout['goal']['deliver_spot']])
    distance = np.linalg.norm(start[:2] - spot)
    assert env.evaluator.evaluate()['score'] == 0.
    for f in (.1, .25, .5, .75):
        env.positions['glass'][:2] = start[:2] + (spot-start[:2])*(f*(distance-P.SPOT_RADIUS_M)/distance)
        assert env.evaluator.evaluate()['score'] == pytest.approx(.9*f)
    env.set_volume('glass', 'red_wine', 149.3/2.)
    assert env.evaluator.evaluate()['score'] == pytest.approx(.9*.75*.5)
    env.angles['glass'] = 90.
    assert env.evaluator.evaluate()['score'] == 0.


@pytest.mark.parametrize('place', ['bar', 'floor', 'elsewhere', 'coaster'])
def test_waste_reduces_progress_and_fails_success(place):
    env = StateEnv(SELECTIONS[2])
    env.finish()
    if place == 'coaster':
        env.set_volume(place, 'red_wine', 150.)
    else:
        env.inv[place] = {'red_wine': 150., 'total': 150.}
    ev = env.evaluator.evaluate()
    assert not ev['success'] and not ev['checks']['no_spill']
    assert ev['score'] == pytest.approx(.45)


def test_dump_rewards_actual_bucket_contents_and_both_liquids():
    env = StateEnv(SELECTIONS[5])
    old = env.inv['containers']['iso']['red_wine']
    env.set_volume('iso', 'red_wine', old/2.)
    env.set_volume('dump', 'red_wine', old/2.)
    assert env.evaluator.evaluate()['score'] == pytest.approx(.9*.4*.5)
    env.set_volume('iso', 'white_wine', 0.)
    env.set_volume('dump', 'red_wine', old)
    assert env.evaluator.evaluate()['score'] == pytest.approx(.36)
    env.set_volume('iso', 'white_wine', 21.)
    assert env.evaluator.evaluate()['score'] == pytest.approx(.9*(.4+.6*.5))
    env.set_volume('iso', 'white_wine', 50.)
    assert env.evaluator.evaluate()['score'] == 1.
    env.set_volume('dump', 'red_wine', 0.)
    # Old wine hidden in another container cannot pass disposal or receive dump credit.
    env.inv['containers']['bottle']['red_wine'] = old
    env.inv['containers']['bottle']['total'] += old
    ev = env.evaluator.evaluate()
    assert not ev['success'] and not ev['checks']['old_in_bucket']
    assert ev['score'] == pytest.approx(.9*.6)
    env.set_volume('dump', 'red_wine', old/2.)
    env.inv['floor'] = {'red_wine': old/2., 'total': old/2.}
    assert env.evaluator.evaluate()['score'] == pytest.approx(.9*(.4*.5*.5 + .6))
    env.inv['floor'] = {'white_wine': 50., 'total': 50.}
    assert env.evaluator.evaluate()['score'] == pytest.approx(.9*(.4*.5 + .6*.5))


def test_success_threshold_minimal_pairs():
    env = StateEnv(SELECTIONS[2])
    env.finish()
    for film, passes in [(.149, True), (.151, False)]:
        env.inv['film']['bottle'] = film
        assert env.evaluator.evaluate()['checks']['no_drip'] == passes
    env.inv['film'].clear()
    env.source_touch = dict(max_n=.51, ticks=10)
    assert not env.evaluator.evaluate()['success']
    env.source_touch = dict(max_n=0., ticks=0)
    env.positions['bottle'][0] += .021
    assert not env.evaluator.evaluate()['checks']['source_on_coaster']
    env.positions['bottle'][0] -= .021
    env.still = 1.99
    assert not env.evaluator.evaluate()['checks']['settled']
    lined = StateEnv(SELECTIONS[6])
    for amount, passes in [(174.999, False), (175., True), (185., True), (185.001, False)]:
        lined.set_volume('glass', 'red_wine', amount)
        assert lined.evaluator.evaluate()['checks']['volume'] == passes


def test_split_exchangeable_targets_and_original_reward():
    env = StateEnv(SELECTIONS[-1])
    env.finish()
    a, b = env.layout['goal']['targets']
    env.positions[a], env.positions[b] = env.positions[b].copy(), env.positions[a].copy()
    assert env.evaluator.evaluate()['success']
    env.contacts = {'left': True}
    ev = env.evaluator.evaluate()
    assert ev['score'] == .9 and ev['reward']['contract'] == SPLIT_REWARD_CONTRACT
    env.set_volume('coaster', 'red_wine', 1.)
    assert not env.evaluator.evaluate()['checks']['no_spill']
    assert LEGACY.touch_rule('split-bottle') == (1, LEGACY.TOUCH_FLOOR_N)
    for _, task, _ in EIGHT_TASKS[:-1]:
        assert LEGACY.touch_rule(task) == (10, P.CONTACT_FORCE_TOL_N)


@pytest.mark.parametrize('selection', SELECTIONS[:-1], ids=lambda s: s['task'])
@pytest.mark.parametrize('held_progress,success,fault', [(.8, False, False), (.1, False, False), (1., True, False),
                                                       (.8, False, True)])
def test_settle_branch_restores_live_state_and_caps_failed_credit(selection, held_progress, success, fault):
    def result(p, ok, settled):
        return dict(success=ok, score=1. if ok else .9*p, checks=dict(settled=settled),
                    reward=dict(contract=TASKS_V1_REWARD_CONTRACT, progress=p), meta={})
    cfg = Config(**config_values(selection))
    world = SimpleNamespace(fault=None, gas=None, liquid=None, lumped=None, phantom_log=['original'], _prev_cvel={7: 1.})
    callback = object()
    env = SimpleNamespace(config=cfg, world=world, trace_callback=callback, restored=[], phase=0)
    env.robot_still_s = lambda: 9.
    env.checkpoint = lambda: env.phase
    def restore(s):
        env.phase, env.world.fault = s, None
        env.restored.append(s)
    env.restore = restore
    def act(targets, ticks):
        assert targets == {} and env.trace_callback is None
        env.phase = 1
        world._prev_cvel = {7: 99.}
        world.phantom_log.append('branch')
        if fault:
            world.fault = 'stub fault'
    env.act = act
    ev = T.Evaluator.__new__(T.Evaluator)
    ev.env = env
    ev.evaluate = lambda: result(held_progress, success, True) if env.phase else result(.2, False, False)
    out = ev.verify()
    expected = .2 if fault else 1. if success else min(.2, held_progress)
    assert out['score'] == pytest.approx(1. if success and not fault else .9*expected)
    assert bool(out['read']['infrastructure_fault']) == fault
    assert env.phase == 0 and env.restored == [0] and env.trace_callback is callback
    assert world.phantom_log == ['original'] and world._prev_cvel == {7: 1.}


@pytest.mark.parametrize('selection', SELECTIONS[:-1], ids=lambda s: s['task'])
@pytest.mark.parametrize('gas_kind', ['none', 'disabled', 'enabled'])
def test_settle_branch_with_the_worlds_gas_model(selection, gas_kind):
    """A real world always has a GasModel (Config.gas): without a bottle (carry-glass, water-tumbler) it is disabled and
    has no event/history logs, which the settle branch's plain save reads (AttributeError, and GET /v1/evaluation
    would return no reply). With a bottle its clocks and logs are restored."""
    def result(settled):
        return dict(success=False, score=0., checks=dict(settled=settled),
                    reward=dict(contract=TASKS_V1_REWARD_CONTRACT, progress=0.), meta={})
    if gas_kind == 'disabled':      # what GasModel.__init__ leaves when the scene has no bottle
        gas = SimpleNamespace(world=None, liquid=None, bottles=[], enabled=False)
    elif gas_kind == 'enabled':
        gas = SimpleNamespace(enabled=True, events=[{'event': 'seal'}], history=[(1., [0.])], _t_last=1., _t_prev=1.)
    else:
        gas = None
    cfg = Config(**config_values(selection))
    world = SimpleNamespace(fault=None, gas=gas, liquid=None, lumped=None, phantom_log=[], _prev_cvel=None)
    env = SimpleNamespace(config=cfg, world=world, trace_callback=None, phase=0)
    env.robot_still_s = lambda: 9. if env.phase else 0.
    env.checkpoint = lambda: env.phase
    env.restore = lambda s: setattr(env, 'phase', s)
    def act(targets, ticks):
        env.phase = 1
        if gas_kind == 'enabled':
            gas.events.append({'event': 'vent'}); gas.history.append((4., [0.])); gas._t_last = gas._t_prev = 4.
    env.act = act
    ev = T.Evaluator.__new__(T.Evaluator)
    ev.env = env
    ev.evaluate = lambda: result(bool(env.phase))
    out = ev.verify()
    assert out['read']['stopped'] == 'settled' and out['read']['infrastructure_fault'] is None
    assert env.phase == 0 and world.gas is gas
    if gas_kind == 'enabled':
        assert gas.events == [{'event': 'seal'}] and gas.history == [(1., [0.])] and gas._t_last == gas._t_prev == 1.
    if gas_kind == 'disabled':
        assert vars(gas) == dict(world=None, liquid=None, bottles=[], enabled=False)


def test_backend_selection_reset_and_checkpoint_contracts_without_runtime(monkeypatch):
    from wine_pouring.task_backend import NewtonWinePouringBackend
    class FakeEnv:
        def __init__(self, config, **kw):
            self.config, self.world, self.lock, self.generation = config, object(), threading.RLock(), 0
        def reset(self):
            return 'reset'
        def get_state(self):
            return 'state'
    for module, name, value in [('wine_pouring.kinematics', 'ArmKinematics', lambda w: 'ik')]:
        stub = ModuleType(module)
        setattr(stub, name, value)
        monkeypatch.setitem(sys.modules, module, stub)
    backend = NewtonWinePouringBackend.__new__(NewtonWinePouringBackend)
    backend._make_env = lambda config: FakeEnv(config)
    backend.config = Config()
    backend.env = FakeEnv(backend.config)
    backend.trace = None
    backend.output = None
    backend._end_operation = lambda **kw: None
    backend.capabilities = SimpleNamespace(notes={})
    legacy_key = backend._variation_key()
    selected = backend.select_variation(SELECTIONS[2])
    assert selected['reset_request']['options']['contract'] == TASKS_V1_REWARD_CONTRACT
    backend.reset(**selected['reset_request'])
    assert backend.config.task_contract == TASKS_V1_REWARD_CONTRACT
    assert backend._variation_key() != legacy_key
    assert backend.variation_contract()['current']['contract'] == TASKS_V1_REWARD_CONTRACT
    backend.reset(seed=3)
    assert backend.config.task_contract == TASKS_V1_REWARD_CONTRACT
    backend.reset()
    assert backend.config.task_contract == TASKS_V1_REWARD_CONTRACT
    original = backend.env
    with pytest.raises(ValueError):
        backend.reset(options={'task': 'split-bottle', 'contract': TASKS_V1_REWARD_CONTRACT})
    assert backend.env is original
    backend.reset(options={'task': 'taste-pour'})
    assert backend.config.task_contract == ''  # no stale opt-in when a different task is selected
    backend.reset(options={'task': 'split-bottle'})
    assert backend._variation_key()['contract'] == SPLIT_REWARD_CONTRACT


def test_layouts_have_no_new_placements_and_nearest_mount_distances():
    # A horizontal workspace screen, not collision/IK qualification. No layouts change.
    # The reach check in wine_pouring/REPORT.pdf section 12.4 covers bottle poses at all four spots.
    for selection in SELECTIONS:
        for seed in (0, 1, 7, 131, 9999):
            bare = {k: v for k, v in dict(selection, seed=seed).items() if k != 'contract'}
            layout = task_layout(Config(**config_values(dict(selection, seed=seed))))
            assert layout == task_layout(Config(**bare))
            for item in layout['items']:
                distance = min(math.dist(item['xy'], mount) for mount in P.ARM_BASE_XY.values())
                assert distance < .65, (selection['task'], seed, item['name'], distance)


@pytest.mark.parametrize('selection', SELECTIONS + [dict(preset='service', task='pour-glass', seed=0)],
                         ids=lambda s: s['task'] + ('-opt-in' if 'contract' in s else '-legacy'))
def test_registry_launch_attaches_contract_without_changing_physics_config(monkeypatch, selection):
    from dataclasses import asdict, fields
    from gym_anything_robotics import registry
    from gym_anything_robotics.policy_access import PolicyAccess
    from wine_pouring.config import Config as PhysicsConfig
    from wine_pouring.task_backend import NewtonWinePouringBackend, PhysicsBackend

    built = []
    def construct(backend, *, config=None, preset='service', task=None, seed=0):
        assert 'task_contract' not in config
        physics_config = PhysicsConfig(**dict(config, preset=preset, task=task, seed=seed))
        env = StateEnv(dict(preset=preset, task=task, seed=seed))
        env.config = physics_config
        env.lock = threading.RLock()
        env.world.config = physics_config
        env.evaluator = LEGACY.Evaluator(env)
        backend.config, backend.env = physics_config, env
        built.append(physics_config)
    monkeypatch.setattr(PhysicsBackend, '__init__', construct)
    monkeypatch.setattr(PhysicsBackend, 'policy_access', lambda b: PolicyAccess(
        b.config.robot, (), {}, frozenset(), lambda: LEGACY.instruction(b.config), {}))
    options = task_resolve(selection)['backend_options']
    backend = registry._newton_wine_pouring_factory(**options, config={'solver_iterations': 50})
    assert isinstance(backend, NewtonWinePouringBackend)
    assert backend.config is backend.env.config
    assert isinstance(backend.env.evaluator, T.Evaluator)
    assert backend.env.world.config is built[0]
    assert type(built[0]) is PhysicsConfig
    assert all(getattr(backend.config, f.name) == getattr(built[0], f.name) for f in fields(PhysicsConfig))
    assert 'task_contract' not in asdict(built[0])
    assert backend.policy_access().instruction() == task_resolve(selection)['description']
    result = backend.evaluate()
    assert result['reward']['contract'] == reward_contract(selection['task'], selection.get('contract', ''))
    assert not result['success']
    assert result['score'] == (0. if 'contract' in selection else sum(result['checks'].values())/len(result['checks']))


def test_rebuild_attaches_evaluator_after_physical_preparation(monkeypatch):
    from wine_pouring.task_backend import NewtonWinePouringBackend
    from wine_pouring.config import Config as PhysicsConfig
    config = Config(**config_values(SELECTIONS[3]))
    stub = ModuleType('wine_pouring.env')
    def make_env(physics_config, **kwargs):
        assert type(physics_config) is PhysicsConfig
        env = StateEnv(dict(preset=physics_config.preset, task=physics_config.task, seed=physics_config.seed))
        env.set_volume('glass', 'red_wine', 73.008)
        env.world.config = physics_config
        return env
    stub.WineEnv = make_env
    monkeypatch.setitem(sys.modules, 'wine_pouring.env', stub)
    backend = NewtonWinePouringBackend.__new__(NewtonWinePouringBackend)
    backend.output = None
    env = backend._make_env(config)
    assert env.config is config
    assert env.evaluator.initial_containers['glass']['red_wine'] == 73.008
    assert env.evaluator.evaluate()['score'] == 0.
    assert type(env.world.config) is PhysicsConfig


def test_invalid_contract_is_rejected_before_backend_construction(monkeypatch):
    from wine_pouring.task_backend import NewtonWinePouringBackend, PhysicsBackend
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid contract reached the physics constructor')
    monkeypatch.setattr(PhysicsBackend, '__init__', forbidden)
    with pytest.raises(ValueError):
        NewtonWinePouringBackend(task='split-bottle', contract=TASKS_V1_REWARD_CONTRACT)
    with pytest.raises(ValueError):
        NewtonWinePouringBackend(task='pour-glass', config={'task_contract': 'typo'})

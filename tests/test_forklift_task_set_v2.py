"""forklift-tasks-v2 (forklift/task_set_v2.py): the fix branch's eight-task tests, ported to the v2 contract.

Constructed states and stubbed backend only. Never builds a world or advances physics. Released selections (bare, v1,
forklift-tasks-v1, load-bales) are checked to keep their released behaviour; their own tests are
tests/environments/forklift/test_task_set.py.
"""
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from forklift import evaluator as E
from forklift import params as P
from forklift.catalogue import CONTRACT as V1_CONTRACT
from forklift.events import Events
from forklift.task import TIME_LIMIT_S, instruction
from forklift.task_progress import measurements, workmanship
from forklift.task_progress_v2 import TaskProgress, approach, progress
from forklift.task_set_v2 import BALE_CONTRACT, CONTRACT, DIFFICULTY, TASK_SET, TASKS, resolve_contract
from forklift.variants import digest, resolved_start, task_resolve
from tests.packages import PACKAGES, REPO

ROOT = PACKAGES
SPLITS = REPO / 'benchmarks/roboworld/splits'


def world_for(task, seed=0):
    """Authored start geometry, with array-backed rigid poses and empty physical latches."""
    from forklift import facility
    cfg, start = resolved_start(task, seed)
    from forklift.task import reserved_locations
    layout = getattr(facility, cfg.preset)(np.random.default_rng(1000 + seed), reserved=reserved_locations(cfg))
    poses, names, loads, trailers = [], {}, {}, {}

    def body(name, xyz, yaw):
        names[name] = len(poses)
        poses.append([*xyz, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])
        return names[name]

    x, y, yaw = start['truck']
    z = start.get('truck_z', 0.0)
    body('chassis', (x, y, z + .3), yaw)
    body('sideshift', (x, y, z + start['lift'] - .104), yaw)
    for ld in start['loads']:
        if 'pose' in ld:
            lx, ly, lyaw = ld['pose']
            xyz = (lx, ly, ld.get('z0', 0.0))
        else:
            xyz, lyaw = ld['centre'], 0.0
        loads[ld['name']] = dict(body=body(ld['name'], xyz, lyaw), mass=ld.get('mass', 900), size=ld.get('size'))
    if start.get('load_on_forks'):
        loads['carried'] = dict(body=body('carried', (x + 1.17 * math.cos(yaw), y + 1.17 * math.sin(yaw),
                                                   z + start['lift']), yaw), mass=start['load_on_forks']['mass'])
    for door, tr in start.get('trailers', {}).items():
        d = layout.zones['doors'][door]
        trailers[door] = dict(body=body(door, (d['x'], d['face_y'] + facility.BUMPER + .106, tr['bed_z']), math.pi / 2),
                              W=P.TRAILER_INSIDE['W'], L=P.TRAILER_INSIDE['L'])
    q = np.array(poses, dtype=float)
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    w = NS(events=ev, sim_time=0.0, tick=0, fault=None, on_tick=[], signals=dict(park=1.0, direction=0, key=start['key']),
           drivetrain=NS(running=start['engine'] != 'off'),
           T=NS(bodies={'sideshift': names['sideshift']}, anchors={'sideshift': (0, 0, 0)}, shifts={}),
           state=NS(body_q=NS(numpy=lambda: q)),
           scene=NS(c=cfg, zones=layout.zones, layout=layout, loads=loads, trailers=trailers, task_start=start,
                    ground_height=lambda x, y: 0.0, load_start_pose={n: q[ld['body'], :3].tolist() for n, ld in loads.items()}),
           solver=NS(mj_model=NS(ngeom=0), mj_data=NS(ncon=0, contact=[])), mj_geom_label=lambda g: '')
    w.body_pose = lambda name: E.body_frame(w, names[name])
    w.tyres = NS(telemetry={n: {'contact': None} for n in ('wheel_fl', 'wheel_fr')})
    w.q, w.names = q, names
    return w, start


def test_exact_selections_and_versioned_resolver():
    raw = json.loads((ROOT / 'forklift/benchmark.json').read_text())
    rows = raw['task_set']['selections']
    test_split = [r['selection'] for r in json.loads((SPLITS / 'test.json').read_text())['tasks']
                  if r['environment_id'] == 'forklift']
    # the audited hardest task keeps its released legacy selection, exactly as splits/test.json lists it
    assert rows == [dict(s) for s in TASK_SET[:-1]] + test_split == [*map(dict, TASK_SET[:-1]), {'task': 'load-bales'}]
    assert [r['task'] for r in rows] == list(TASKS) and len(set(TASKS)) == 8
    assert list(DIFFICULTY.values())[-1] == 'hardest'
    assert raw['count'] == 14
    assert CONTRACT in raw['selector_choices']['contract']
    released = raw['task_set_v1']['tasks']
    assert [r['selection']['task'] for r in released] == list(TASKS)  # the same eight work orders as v1
    for row, v1 in zip(rows, released):
        new = task_resolve(row)
        old = task_resolve(dict(task=row['task'], seed=0))
        assert new['description'] == old['description'] == task_resolve(v1['selection'])['description']
        if row['task'] != 'load-bales':
            assert new['task_id'] == f"forklift/v1/{new['selection']['preset']}/{row['task']}/{CONTRACT}/seed-0"
            assert new['backend_options'] == dict(task=row['task'], seed=0, contract=CONTRACT)
            assert new['selection'] == dict(old['selection'], contract=CONTRACT)
            assert new['split'] == old['split'] == 'train'
            assert resolve_contract(row['task'], CONTRACT) == CONTRACT
        else:
            assert new == old == task_resolve(v1['selection']) == task_resolve({'task': 'load-bales'})
            assert resolve_contract(row['task']) == BALE_CONTRACT == v1['task_contract']
    default = task_resolve({})
    assert default['task_id'] == raw['default_task_id'] == 'forklift/v1/warehouse/start-and-park/seed-0'
    assert default['backend_options'] == dict(task='start-and-park', seed=0)
    assert task_resolve({'task': 'start-and-park', 'contract': 'v1'})['task_id'] == default['task_id']
    for seed in (True, '0', 1.5, -1, 1000):
        with pytest.raises(ValueError):
            task_resolve({'seed': seed, 'contract': CONTRACT})
    for task, contract in (('load-bales', CONTRACT), ('yard-route', CONTRACT), ('carry-loose-stack', CONTRACT),
                           ('start-and-park', ''), ('start-and-park', []), ('start-and-park', 'unknown')):
        with pytest.raises(ValueError):
            task_resolve(dict(task=task, contract=contract))


def test_frozen_starts_and_instructions_preserved():
    frozen = json.loads((ROOT / 'forklift/data/variations.json').read_text())
    for t in TASKS:
        for seed in (0, 3, 4, 800, 900, 999):
            dg, start = digest(t, seed)
            assert dg[:16] == frozen['tasks'][t]['entries'][seed]['digest']
            assert task_resolve(dict(task=t, seed=seed))['description'] == instruction(start)
            if t != 'load-bales':
                assert task_resolve(dict(task=t, seed=seed, contract=CONTRACT))['description'] == instruction(start)


@pytest.mark.parametrize('task', TASKS[:-1])
@pytest.mark.parametrize('seed', [0, 4, 900, 999])
def test_untouched_and_waiting_start_zero(task, seed):
    w, st = world_for(task, seed)
    adapter = TaskProgress(w, st)
    r = adapter.evaluate(w)
    assert not r['success'] and r['score'] == r['progress'] == 0, (task, seed, r)
    w.sim_time = 2.0
    w.events.rest_s = 2.0
    if task == 'aisle-cruise':
        x, y, yaw = st['truck']
        w.events.traj = [(0, x, y, yaw, 0), (2, x, y, yaw, 0)]
    r = adapter.evaluate(w)
    assert not r['success'] and r['score'] == r['progress'] == 0, (task, seed, r)


@pytest.mark.parametrize('task', TASKS[:-1])
def test_transport_progress_is_continuous_and_reversible(task):
    w, st = world_for(task)
    adapter = TaskProgress(w, st)
    initial = adapter.initial
    if task == 'aisle-cruise':
        def reading(f):
            return dict(initial, sections=[f, 0, 0])
    else:
        def reading(f):
            return dict(initial, distance=initial['distance'] * (1 - f))
    values = [progress(task, reading(f), initial) for f in (0, .25, .5, .50001, .75)]
    assert values[0] == 0
    assert all(a < b for a, b in zip(values, values[1:])), (task, values)
    assert values[3] - values[2] < .0001
    assert progress(task, initial, initial) == 0


def test_floor_placement_is_continuous_at_painted_lines_and_floor():
    w, st = world_for('carry-to-floor-bay')
    adapter = TaskProgress(w, st)
    bay = w.scene.zones['bays'][st['bay']]
    b = w.names['carried']
    w.q[b, 3:] = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    w.q[b, :3] = [np.mean(bay['x']), np.mean(bay['y']), .01]
    full = adapter.evaluate(w)
    scores = []
    for outside in (.05, .01, .00001, 0):
        w.q[b, 0] = bay['x'][1] - P.GMA_PALLET['W'] / 2 + outside
        scores.append(adapter.evaluate(w)['score'])
    assert all(a < b for a, b in zip(scores, scores[1:])), scores
    assert scores[-1] - scores[-2] < .0001
    assert full['score'] > 0
    w.q[b, 2] = .15
    raised = adapter.evaluate(w)
    w.q[b, 2] = .10
    assert adapter.evaluate(w)['score'] > raised['score']


def test_cruise_short_band_motion_is_not_full_section():
    w, st = world_for('aisle-cruise')
    TaskProgress(w, st)
    a, b = E.CRUISE_SECTIONS[st['cruise']][0]
    w.events.traj = [(0, a + .1, 28.25, 0, 2.0), (1, a + .11, 28.25, 0, 2.0)]
    now = measurements(w, st, E.evaluate(w, st))
    assert 0 < now['sections'][0] < .01
    assert now['crossings'] == [0, 0]


def test_rack_height_starts_at_zero_and_uses_actual_030_travel_limit():
    w, st = world_for('put-away-rack')
    adapter = TaskProgress(w, st)
    before = adapter.evaluate(w)
    w.q[w.names['carried'], 2] += .2
    assert adapter.evaluate(w)['score'] > before['score'] == 0


def test_approach_gives_no_credit_for_a_start_already_at_the_goal():
    from forklift import task_progress as released
    assert approach(0.0, 0.0) == 0.0 and approach(0.3, 0.0) == 0.0     # nothing to close at the start: no credit
    assert approach(1.0, 2.0) == 0.5 and approach(0.0, 2.0) == 1.0 and approach(3.0, 2.0) == 0.0
    w, st = world_for('start-and-park')
    initial = dict(TaskProgress(w, st).initial, distance=0.0)           # a seed whose truck starts inside the bay
    assert progress(st['task'], dict(initial), initial) == 0.0
    # forklift-tasks-v1 keeps its released approach (the credit this fix removes under v2)
    assert released.approach(0.0, 0.0) == 1.0 and released.progress(st['task'], dict(initial), initial) > 0


def test_pushed_pallet_loses_approach_credit():
    w, st = world_for('drive-to-pallet')
    adapter = TaskProgress(w, st)
    i = adapter.initial
    near = dict(i, distance=i['distance'] / 2)
    assert progress(st['task'], near, i) > 0
    near['pallet'] = [i['pallet'][0] + .31, *i['pallet'][1:]]
    assert progress(st['task'], near, i) == 0


def test_time_limit_is_the_evaluator_criterion_and_red_reads_the_trajectory():
    from tests.test_forklift_tasks import _park_world
    w, st = world_for('start-and-park')
    adapter = TaskProgress(w, st)
    limit = TIME_LIMIT_S[st['task']]
    good = _park_world()
    good.sim_time = limit
    done = adapter.evaluate(good)
    assert done['success'] and done['score'] == 1 and done['within_time_limit'] and done['time_limit_s'] == limit
    assert done['task_contract'] == done['details']['reward']['contract'] == CONTRACT
    assert 'episode_done' not in done and 'end_reason' not in done
    good.sim_time = limit + .01                   # the same end state, judged after the limit
    late = adapter.evaluate(good)
    by = {c['name']: c for c in late['criteria']}
    assert not late['success'] and not by['time_limit']['passed'] and not late['within_time_limit']
    assert late['score'] == pytest.approx(.9 * late['progress'] * late['details']['reward']['workmanship'])
    assert late['score'] < 1
    good.sim_time = limit                         # no latch: an evaluation reads the world as it is now
    assert adapter.evaluate(good)['success']
    assert not hasattr(adapter, 'terminal') and not hasattr(adapter, 'on_tick')
    w.fault = 'nonfinite'
    assert adapter.evaluate(w)['score'] == 0 and not adapter.evaluate(w)['success']
    w, st = world_for('load-trailer', 4)
    adapter = TaskProgress(w, st)
    edge = w.scene.zones['doors'][st['target']['door']]['face_y'] - P.LEVELER['length']
    w.q[w.names['chassis'], 1] = edge
    p, R = w.body_pose('chassis')
    w.events.traj.append((1.0, float(p[0]), float(p[1]), E.yaw_of(R), 0.0))   # the 10 ms trajectory sample
    red = adapter.evaluate(w)
    assert red['score'] == red['progress'] == 0 and not red['success']
    w.q[w.names['chassis'], 1] -= 10              # backing out does not undo entering on red
    assert adapter.evaluate(w)['score'] == 0


def test_workmanship_severity_and_missing_floor_bay_details():
    w, st = world_for('carry-to-floor-bay')
    base = E.evaluate(w, st)
    assert base['details'] == {}  # regression: reconstruct shared diagnostics for this old evaluator
    assert workmanship(w, st, base) == (1, {})
    w.events.load_floor = {'carried': [(1.0, (0, 0)), (1.25, (0, 0))]}
    w.events.load_floor_onset = {'carried': [(1.0, .05)]}
    gentle = workmanship(w, st, E.evaluate(w, st))[0]
    w.events.load_floor_onset = {'carried': [(1.0, 1.0)]}
    rough = workmanship(w, st, E.evaluate(w, st))[0]
    assert 0 < rough < gentle < 1
    w.events.overturn = 1.0
    assert workmanship(w, st, E.evaluate(w, st))[0] == 0


@pytest.mark.parametrize('task', TASKS[:-1])
def test_constructed_success_keeps_one_and_exact_predicates(task):
    from tests.test_forklift_tasks import _park_world, _put_away_world
    w, st = world_for(task)
    adapter = TaskProgress(w, st)
    w.sim_time, w.events.rest_s = 30.0, 3.0
    if task == 'start-and-park':
        w = _park_world()
    elif task == 'put-away-rack':
        w, st = _put_away_world(0)
    elif task in ('carry-to-floor-bay', 'ramp-down-loaded'):
        zone = w.scene.zones['bays'][st['bay']] if task == 'carry-to-floor-bay' else w.scene.zones[E.DROP_ZONE[st['drop_zone']]]
        w.q[w.names['carried']] = [np.mean(zone['x']), np.mean(zone['y']), 0, 0, 0, math.sin(math.pi/4), math.cos(math.pi/4)]
        w.q[w.names['sideshift'], 2] = .10 - .104
    elif task == 'drive-to-pallet':
        _, o, R = E.pallet_corners(w, 'target')
        goal = o - R[:, 0] * (P.GMA_PALLET['L']/2 + .25)
        goal[2] = .04
        w.q[w.names['sideshift'], :3] = goal - R @ np.array([1.601, 0, .104])
        w.q[w.names['sideshift'], 3:] = w.q[w.names['target'], 3:]
    elif task == 'aisle-cruise':
        way = st['cruise']
        rows = []
        for a, b in E.CRUISE_SECTIONS[way]:
            rows.extend((len(rows)+i, x, 28.25, 0, 2.0) for i, x in enumerate(np.linspace(a, b, 100)))
        w.events.traj = rows
        w.events.crossings = [dict(zone=n, speed=1.0, since_horn=1.0) for n in ('I1', 'I2')]
        w.q[w.names['chassis'], 0] = w.scene.zones['stop_lines'][E.CRUISE_END[way]]['x']
    elif task == 'load-trailer':
        door = st['target']['door']
        tr, to, TR = E._trailer_frame(w, door)
        ac, _, _ = E.pallet_corners(w, st['target']['behind'])
        ahead = ((ac-to) @ TR)[:, 0].min()
        sign = 1 if st['target']['side'] == 'L' else -1
        local = np.array([ahead - P.GMA_PALLET['L']/2 - .025, sign * (tr['W']/2 - P.GMA_PALLET['W']/2 - .10), 0])
        w.q[w.names['carried'], :3] = to + TR @ local
        w.q[w.names['carried'], 3:] = w.q[w.names[door], 3:]
    legacy = E.evaluate(w, st)
    assert legacy['success'], [c for c in legacy['criteria'] if not c['passed']]
    revised = adapter.evaluate(w)
    assert revised['success'] and revised['score'] == 1
    assert revised['criteria'] == legacy['criteria']
    # The very same achieved state remains below success after a real shared-rule violation.
    w.events.seat_open_max_s = 3.0
    revised = adapter.evaluate(w)
    assert not revised['success'] and 0 < revised['score'] <= .9


def test_constructed_red_wait_success():
    w, st = world_for('load-trailer', 4)
    adapter = TaskProgress(w, st)
    lane = w.scene.zones['staging']['lane' + st['target']['door'][4:]]
    y = lane['y'][1] - E.REACH_LOADED + .1
    dy = y - w.q[w.names['chassis'], 1]
    for name in ('chassis', 'carried', 'sideshift'):
        w.q[w.names[name], 1] += dy
    w.events.traj = [(10, w.q[w.names['chassis'], 0], y, math.pi/2, 0)]
    w.events.rest_s = 3
    w.sim_time = 10
    assert E.evaluate(w, st)['success']
    assert adapter.evaluate(w)['score'] == 1


def test_actual_park_success_and_legacy_score():
    from tests.test_forklift_tasks import _park_world
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    w, st = world_for('start-and-park')
    adapter = TaskProgress(w, st)
    good = _park_world()
    assert adapter.evaluate(good)['score'] == 1
    good.signals['park'] = 0
    bad = adapter.evaluate(good)
    assert not bad['success'] and 0 < bad['score'] <= .9
    backend = NewtonForkliftBackend.__new__(NewtonForkliftBackend)
    backend.config = NS(task='start-and-park')
    backend.env = NS(world=w)
    legacy = backend._evaluate_task()             # a bare selection: the released evaluation, unchanged
    assert legacy['score'] == E.evaluate(w, st)['score'] > 0
    assert legacy.pop('task_contract') == 'v1' and legacy == E.evaluate(w, st)
    backend._task_progress = TaskProgress(w, st)
    assert backend._evaluate_task()['score'] == 0


def _stub_session(cls, w, st):
    import threading
    backend = cls.__new__(cls)
    backend.config = NS(task=st['task'], seed=0, dt=.0005)
    backend._checkpoints = {}
    backend._prestart_worker = lambda ticks: None
    backend.validate_action = lambda action: ({}, 4000)
    calls = []

    def snapshot():
        return dict(tick=w.tick, sim_time=w.sim_time, q=w.q.copy())

    initial = snapshot()

    def restore(s):
        w.tick, w.sim_time = s['tick'], s['sim_time']
        w.q[:] = s['q']

    def reset():
        restore(initial)
        return {}

    def act(cmd, ticks):
        # Stub clock only, with no physics implementation imported or called.
        calls.append(ticks)
        w.tick += ticks
        w.sim_time = w.tick * backend.config.dt
        for hook in w.on_tick:
            hook(w)
        return {}

    backend.env = NS(world=w, lock=threading.RLock(), generation=0, checkpoint=snapshot,
                     restore=restore, reset=reset, get_state=lambda: {},
                     owner_state=lambda: {'events': {'traj': []}}, act=act)
    return backend, calls


def test_backend_reset_checkpoint_timeout_and_owner_evaluation():
    from forklift.task_set_v2_backend import NewtonForkliftTaskSetV2Backend
    w, st = world_for('start-and-park')
    backend, calls = _stub_session(NewtonForkliftTaskSetV2Backend, w, st)
    backend._task_progress = adapter = TaskProgress(w, st)
    backend.save_checkpoint(label='before')
    assert backend.get_state()['evaluation']['score'] == 0
    assert backend.evaluate()['task_contract'] == CONTRACT
    w.tick = round((TIME_LIMIT_S[st['task']] - .01) / backend.config.dt)
    w.sim_time = w.tick * backend.config.dt
    result = backend.act(None)                    # command execution is the base's: the whole command runs
    assert calls == [4000] and not result.truncated and not result.terminated
    assert result.info['requested_ticks'] == result.info['executed_ticks'] == 4000
    backend.act(None)                             # and nothing is refused after the limit
    assert calls == [4000, 4000]
    late = backend.evaluate()
    assert not late['success'] and not late['within_time_limit']
    backend.save_checkpoint(label='after')
    assert 'task_progress' not in backend._checkpoints['after']
    backend.restore_checkpoint('before')
    assert backend.evaluate()['score'] == 0 and backend.evaluate()['within_time_limit']
    backend.reset()                               # the released reset (the v2 adapter has nothing to clear)
    assert backend.evaluate()['score'] == 0 and backend.evaluate()['within_time_limit']
    w.fault = 'nonfinite'                         # a faulted world: the adapter's 0, as the released routing gives
    assert backend.evaluate()['score'] == 0 and backend.get_state()['evaluation']['score'] == 0
    assert adapter.initial == TaskProgress(*world_for('start-and-park')).initial


def test_released_v1_session_still_cuts_at_the_limit():
    """forklift-tasks-v1 keeps the released lifecycle: the command is cut at the limit and later ones refused."""
    from forklift.task_progress import TaskProgress as V1TaskProgress
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    from gym_anything_robotics.errors import ActionRejectedError
    w, st = world_for('start-and-park')
    backend, calls = _stub_session(NewtonForkliftBackend, w, st)
    backend._task_progress = adapter = V1TaskProgress(w, st)
    w.on_tick.append(adapter.on_tick)
    w.tick = round((TIME_LIMIT_S[st['task']] - .01) / backend.config.dt)
    w.sim_time = w.tick * backend.config.dt
    result = backend.act(None)
    assert calls == [20] and result.truncated and backend.evaluate()['task_contract'] == V1_CONTRACT
    with pytest.raises(ActionRejectedError):
        backend.act(None)


def test_bales_is_not_wrapped_and_reports_its_own_contract():
    import threading
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    from tests.test_forklift_tasks import _bales_world
    w, st, *_ = _bales_world()
    backend = NewtonForkliftBackend.__new__(NewtonForkliftBackend)
    backend.config = NS(task='load-bales')
    w.scene.task_start = st
    backend.env = NS(world=w, lock=threading.RLock())
    original = E.evaluate(w, st)
    result = backend.evaluate()                   # the released audited evaluation, unchanged
    assert result.pop('task_contract') == BALE_CONTRACT
    assert result == original and result['details']['reward']['contract'] == BALE_CONTRACT
    with pytest.raises(ValueError):
        TaskProgress(w, st)


def _patch_session(monkeypatch, w, notes=None):
    import sys
    import threading
    import types
    import forklift.robot_setup as setup
    import gym_anything_robotics.backends.newton_forklift as B
    made = []

    def fake_env(config, *, output_dir):
        made.append(config)
        return NS(world=w, generation=0, lock=threading.RLock())

    env_module = types.ModuleType('forklift.env')
    env_module.ForkliftEnv = fake_env
    render_module = types.ModuleType('forklift.render')
    render_module.DIAGNOSTIC_CAMERAS = ()
    monkeypatch.setitem(sys.modules, 'forklift.env', env_module)
    monkeypatch.setitem(sys.modules, 'forklift.render', render_module)
    monkeypatch.setattr(setup, 'validate_setup', lambda x: x)
    monkeypatch.setattr(setup, 'describe', lambda x: x)
    monkeypatch.setattr(B, '_accepted_setup', lambda requested, accepted: None)
    monkeypatch.setattr(B, '_robot_notes', lambda w: dict(notes or {}))
    return B, made


@pytest.mark.parametrize('contract', [None, 'v1', V1_CONTRACT, CONTRACT])
def test_backend_constructor_wires_only_selected_adapter(monkeypatch, tmp_path, contract):
    from forklift import task_progress as released
    from forklift.task_set_v2_backend import NewtonForkliftTaskSetV2Backend
    from gym_anything_robotics.registry import _newton_forklift_factory
    w, st = world_for('start-and-park')
    w.operator = NS(handle=NS(instance='operator'))
    B, made = _patch_session(monkeypatch, w)
    backend = _newton_forklift_factory(task='start-and-park', contract=contract, cameras=(),
                                       renderer_options={'trace': False}, output_dir=tmp_path/'session')
    assert len(made) == 1
    assert made[0].to_dict() == w.scene.c.to_dict() | {'robot': 'h1-2'}
    assert backend.task_contract == (contract or 'v1')
    assert (type(backend) is NewtonForkliftTaskSetV2Backend) == (contract == CONTRACT)
    assert type(backend) is (NewtonForkliftTaskSetV2Backend if contract == CONTRACT else B.NewtonForkliftBackend)
    if contract == CONTRACT:
        assert not w.on_tick                      # the v2 adapter adds no tick hook
        assert type(backend._task_progress) is TaskProgress
    elif contract == V1_CONTRACT:                 # released: v1's adapter with its tick hook
        assert type(backend._task_progress) is released.TaskProgress and len(w.on_tick) == 1
    else:
        assert backend._task_progress is None and not w.on_tick
    r = backend.evaluate()
    assert r['score'] == 0 if contract in (CONTRACT, V1_CONTRACT) else r['score'] > 0


def test_backend_constructor_refusals_keep_the_released_order(monkeypatch, tmp_path):
    from gym_anything_robotics.errors import UnsupportedConfigurationError
    w, st = world_for('start-and-park')
    w.operator = NS(handle=NS(instance='operator'))
    B, made = _patch_session(monkeypatch, w)
    kw = dict(cameras=(), renderer_options={'trace': False})
    with pytest.raises(UnsupportedConfigurationError, match='Unknown forklift task'):
        B.NewtonForkliftBackend(task='no-such-task', contract=CONTRACT, output_dir=tmp_path/'a', **kw)
    with pytest.raises(UnsupportedConfigurationError, match='belongs to preset'):
        B.NewtonForkliftBackend(task='start-and-park', preset='dock', contract=CONTRACT, output_dir=tmp_path/'b', **kw)
    for task in ('load-bales', 'yard-route'):
        with pytest.raises(UnsupportedConfigurationError, match='Unknown forklift contract'):
            B.NewtonForkliftBackend(task=task, contract=CONTRACT, output_dir=tmp_path/task, **kw)
    with pytest.raises(UnsupportedConfigurationError, match='seed must be an integer'):
        B.NewtonForkliftBackend(task='start-and-park', seed='0', contract=CONTRACT, output_dir=tmp_path/'c', **kw)
    assert made == []


@pytest.mark.parametrize('contract', [None, V1_CONTRACT, CONTRACT])
def test_policy_capabilities_declare_the_time_limit(monkeypatch, tmp_path, contract):
    """Under forklift-tasks-v2 the task's simulated-time limit (TIME_LIMIT_S) reaches the policy through
    /v1/capabilities and /v1/help with the rule the evaluation applies; bare and forklift-tasks-v1 selections keep the
    released declaration. The instruction text itself is unchanged."""
    w, st = world_for('start-and-park')
    names = ('left_shoulder_pitch_joint',)
    w.operator = NS(handle=NS(instance='operator'),
                    motors=NS(names=names, lower=[-1.0], upper=[1.0], effort=[10.0], w_nl=[5.0], kp_max=[100.0],
                              kd_max=[5.0], group=['left_arm'], followers=()))
    B, _ = _patch_session(monkeypatch, w, notes={'control': 'c'})
    backend = B.NewtonForkliftBackend(task='start-and-park', contract=contract, cameras=(),
                                     renderer_options={'trace': False}, output_dir=tmp_path/'session')
    pa = backend.policy_access()
    hw = pa.capabilities()['hardware']
    assert pa.instruction() == instruction(st)          # the instruction text is not changed
    assert hw['time_limit_s'] == TIME_LIMIT_S['start-and-park'] == 90.0
    assert pa.help()['effective_capabilities']['hardware']['time_limit_s'] == 90.0
    note = hw['episode']
    assert note == backend.capabilities.notes['episode']
    assert '90 s of simulated time' in note and 'only while a command runs' in note and 'time_limit' in note
    if contract == CONTRACT:
        assert 'not cut or refused' in note and 'attempt ends' not in note
        assert 'at most 90 s' in note
    else:                                               # released declarations
        assert 'not cut or refused' not in note
        assert ('attempt ends at the limit' in note) == (contract == V1_CONTRACT)


def test_v2_session_overrides_only_its_wiring():
    """The v2 session class changes the constructor's adapter, act, the checkpoint pair and the episode note only;
    the released reset, evaluation routing, owner state and policy access serve it unchanged."""
    from forklift.task_set_v2_backend import NewtonForkliftTaskSetV2Backend
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    own = {k for k in vars(NewtonForkliftTaskSetV2Backend)
           if not (k.startswith('__') and k.endswith('__')) and k != '_abc_impl'}
    assert own == {'_episode_note', 'act', 'save_checkpoint', 'restore_checkpoint'}
    for name in ('reset', '_evaluate_task', 'get_state', 'evaluate', 'policy_access', '_notes', 'act_policy'):
        assert getattr(NewtonForkliftTaskSetV2Backend, name) is getattr(NewtonForkliftBackend, name), name

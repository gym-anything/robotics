"""Constructed states and stubbed backend only. Never builds a world or advances physics."""
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from forklift import evaluator as E
from forklift import params as P
from forklift.catalogue import BALE_CONTRACT, CONTRACT, SET_TASKS, TASK_SET, resolve_contract
from forklift.events import Events
from forklift.task import TIME_LIMIT_S, instruction
from forklift.task_progress import TaskProgress, measurements, progress, workmanship
from forklift.variants import resolved_start, task_resolve
from tests.packages import PACKAGES

ROOT = PACKAGES


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
    rows = raw['task_set']['tasks']
    assert [r['selection'] for r in rows] == list(TASK_SET)
    assert len(rows) == len(set(SET_TASKS)) == 8
    assert raw['count'] == 14
    for row in rows:
        st = row['selection']
        new = task_resolve(st)
        old = task_resolve(dict(task=st['task'], seed=st['seed']))
        assert new['description'] == old['description']
        assert row['task_id'] == new['task_id']
        assert row['task_contract'] == resolve_contract(st['task'], st.get('contract'))
        if st['task'] != 'load-bales':
            assert new['backend_options']['contract'] == CONTRACT
            assert new['task_id'] != old['task_id']
        else:
            assert new == old
    default = task_resolve({})
    assert default['task_id'] == raw['default_task_id'] == 'forklift/v1/warehouse/start-and-park/seed-0'
    assert default['backend_options'] == dict(task='start-and-park', seed=0)
    assert task_resolve({'task': None, 'seed': 0}) == default
    assert task_resolve({'task': 'start-and-park', 'contract': 'v1'})['task_id'] == default['task_id']
    for seed in (True, '0', 1.5, -1, 1000):
        with pytest.raises(ValueError):
            task_resolve({'seed': seed})
    for task, contract in (('load-bales', CONTRACT), ('yard-route', CONTRACT), ('start-and-park', ''),
                           ('start-and-park', []), ('start-and-park', 'unknown')):
        with pytest.raises(ValueError):
            task_resolve(dict(task=task, contract=contract))


@pytest.mark.parametrize('task', SET_TASKS[:-1])
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


@pytest.mark.parametrize('task', SET_TASKS[:-1])
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


def test_pushed_pallet_loses_approach_credit():
    w, st = world_for('drive-to-pallet')
    adapter = TaskProgress(w, st)
    i = adapter.initial
    near = dict(i, distance=i['distance'] / 2)
    assert progress(st['task'], near, i) > 0
    near['pallet'] = [i['pallet'][0] + .31, *i['pallet'][1:]]
    assert progress(st['task'], near, i) == 0


def test_timeout_fault_and_red_freeze_without_reading_later_poses():
    w, st = world_for('start-and-park')
    adapter = TaskProgress(w, st)
    w.q[w.names['chassis'], 0] -= 1
    w.sim_time = TIME_LIMIT_S[st['task']]
    adapter.on_tick(w)
    at_limit = adapter.evaluate(w)
    assert at_limit['episode_done'] and at_limit['end_reason'] == 'time_limit'
    w.q[:] = np.nan
    w.sim_time += 100
    assert adapter.evaluate(w) == at_limit
    adapter.reset()
    w.fault = 'nonfinite'
    assert adapter.evaluate(w)['score'] == 0
    assert adapter.evaluate(w)['end_reason'] == 'fault'
    w, st = world_for('load-trailer', 4)
    adapter = TaskProgress(w, st)
    edge = w.scene.zones['doors'][st['target']['door']]['face_y'] - P.LEVELER['length']
    w.q[w.names['chassis'], 1] = edge
    adapter.on_tick(w)
    red = adapter.evaluate(w)
    assert red['score'] == red['progress'] == 0 and red['end_reason'] == 'entered_on_red'
    w.q[w.names['chassis'], 1] -= 10
    assert adapter.evaluate(w) == red


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


@pytest.mark.parametrize('task', SET_TASKS[:-1])
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
    legacy = backend._evaluate_task()
    assert legacy['score'] == E.evaluate(w, st)['score'] > 0
    assert legacy['task_contract'] == 'v1'
    backend._task_progress = TaskProgress(w, st)
    assert backend._evaluate_task()['score'] == 0


def test_backend_reset_checkpoint_timeout_and_owner_evaluation():
    import threading
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    from gym_anything_robotics.errors import ActionRejectedError
    w, st = world_for('start-and-park')
    backend = NewtonForkliftBackend.__new__(NewtonForkliftBackend)
    backend.config = NS(task=st['task'], seed=0, dt=.0005)
    backend._task_progress = adapter = TaskProgress(w, st)
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
        adapter.on_tick(w)
        return {}

    backend.env = NS(world=w, lock=threading.RLock(), generation=0, checkpoint=snapshot,
                     restore=restore, reset=reset, get_state=lambda: {},
                     owner_state=lambda: {'events': {'traj': []}}, act=act)
    backend.save_checkpoint(label='before')
    assert backend.get_state()['evaluation']['score'] == 0
    assert backend.evaluate()['task_contract'] == CONTRACT
    w.tick = round((TIME_LIMIT_S[st['task']] - .01) / backend.config.dt)
    w.sim_time = w.tick * backend.config.dt
    result = backend.act(None)
    assert calls == [20] and result.truncated and not result.terminated
    assert adapter.terminal['end_reason'] == 'time_limit'
    backend.save_checkpoint(label='after')
    with pytest.raises(ActionRejectedError):
        backend.act(None)
    backend.restore_checkpoint('before')
    assert adapter.terminal is None and backend.evaluate()['score'] == 0
    backend.restore_checkpoint('after')
    assert adapter.terminal['end_reason'] == 'time_limit'
    backend.reset()
    assert adapter.terminal is None and backend.evaluate()['score'] == 0


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
    result = backend.evaluate()
    assert result.pop('task_contract') == BALE_CONTRACT
    assert result == original
    with pytest.raises(ValueError):
        TaskProgress(w, st)


@pytest.mark.parametrize('contract', [None, CONTRACT])
def test_backend_constructor_wires_only_selected_adapter(monkeypatch, tmp_path, contract):
    import sys
    import threading
    import types
    import forklift.robot_setup as setup
    import gym_anything_robotics.backends.newton_forklift as B
    w, st = world_for('start-and-park')
    w.operator = NS(handle=NS(instance='operator'))
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
    monkeypatch.setattr(B, '_robot_notes', lambda w: {})
    backend = B.NewtonForkliftBackend(task='start-and-park', contract=contract, cameras=(),
                                     renderer_options={'trace': False}, output_dir=tmp_path/'session')
    assert len(made) == 1
    assert made[0].to_dict() == w.scene.c.to_dict() | {'robot': 'h1-2'}
    assert backend.task_contract == (contract or 'v1')
    assert bool(w.on_tick) == (contract == CONTRACT)
    assert (backend._task_progress is not None) == (contract == CONTRACT)
    r = backend.evaluate()
    assert r['score'] == 0 if contract == CONTRACT else r['score'] > 0


@pytest.mark.parametrize('contract', [None, CONTRACT])
def test_policy_capabilities_declare_the_time_limit(monkeypatch, tmp_path, contract):
    """The task's simulated-time limit (TIME_LIMIT_S) reaches the policy through /v1/capabilities and /v1/help, as
    filleting and locomotive-driving declare theirs; the instruction text itself is unchanged."""
    import sys
    import threading
    import types
    import forklift.robot_setup as setup
    import gym_anything_robotics.backends.newton_forklift as B
    w, st = world_for('start-and-park')
    names = ('left_shoulder_pitch_joint',)
    w.operator = NS(handle=NS(instance='operator'),
                    motors=NS(names=names, lower=[-1.0], upper=[1.0], effort=[10.0], w_nl=[5.0], kp_max=[100.0],
                              kd_max=[5.0], group=['left_arm'], followers=()))
    env_module = types.ModuleType('forklift.env')
    env_module.ForkliftEnv = lambda config, *, output_dir: NS(world=w, generation=0, lock=threading.RLock())
    render_module = types.ModuleType('forklift.render')
    render_module.DIAGNOSTIC_CAMERAS = ()
    monkeypatch.setitem(sys.modules, 'forklift.env', env_module)
    monkeypatch.setitem(sys.modules, 'forklift.render', render_module)
    monkeypatch.setattr(setup, 'validate_setup', lambda x: x)
    monkeypatch.setattr(setup, 'describe', lambda x: x)
    monkeypatch.setattr(B, '_accepted_setup', lambda requested, accepted: None)
    monkeypatch.setattr(B, '_robot_notes', lambda w: {'control': 'c'})
    backend = B.NewtonForkliftBackend(task='start-and-park', contract=contract, cameras=(),
                                     renderer_options={'trace': False}, output_dir=tmp_path/'session')
    pa = backend.policy_access()
    hw = pa.capabilities()['hardware']
    assert hw['time_limit_s'] == TIME_LIMIT_S['start-and-park'] == 90.0
    assert pa.help()['effective_capabilities']['hardware']['time_limit_s'] == 90.0
    note = hw['episode']
    assert note == backend.capabilities.notes['episode']
    assert '90 s of simulated time' in note and 'only while a command runs' in note and 'time_limit' in note
    assert ('attempt ends at the limit' in note) == (contract == CONTRACT)
    assert pa.instruction() == instruction(st)          # the instruction text is not changed

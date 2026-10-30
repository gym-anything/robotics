"""Synthetic ledgers, geometry and stubbed lifecycle only. Never import a simulator runtime."""
import ast
import hashlib
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from sewing import task, variations
from sewing.task_set import CONTRACT, CURVE, LEGACY, TASK_SET, REVISED, task_resolve, choose_contract
from sewing.task_set_scoring import TaskEvaluator, layer_count, seam_path, weighted_coverage
from tests.packages import PACKAGES

ROOT = PACKAGES


def rectangle(width=.2, length=.42):
    X = np.array([(x, y) for y in np.linspace(0, length, 43) for x in np.linspace(0, width, 21)])
    tri = []
    for j in range(42):
        for i in range(20):
            k = j*21 + i
            tri.extend(((k, k+1, k+22), (k, k+22, k+21)))
    return dict(X=X, tri=np.array(tri))


def fake_env(name, points=(), reverse=(), *, offset=(0., 0.), noref=True):
    c = NS(task=name, seam_allowance=.010, robot='dual-fr3', **task.TASK_SETUP[name])
    mesh = rectangle(c.ply_width, c.ply_length)
    positions = [np.column_stack([mesh['X'] + [-.4, -.17], np.full(len(mesh['X']), .0004 + .0006*k)])
                 for k in (0, 1)]
    positions[1][:, :2] += offset
    M = NS(stitches=[], holes=[], threads=[], needle_state='ok', tether='trimmed',
           tails=dict(start=.003, end=.003), foot_N=0., presser='down')
    for i, uv in enumerate(points):
        holes = []
        for k in (0, 1):
            holes.append(len(M.holes))
            M.holes.append(NS(ply=k, uv=tuple(uv)))
        M.stitches.append(dict(formed=True, cycle=i, holes=holes, reverse=i in reverse,
                               reason=None, balance='balanced', seam=0))
    w = NS(config=c, machine=M, mesh=mesh, ply_ids=[0, 1], hand_arms=('left', 'right'),
           reflexes=[] if noref else [dict(tick=1)], finger_force=lambda arm: np.zeros(3),
           ply_positions=lambda k: positions[k])
    env = NS(config=c, world=w, history=dict(reflex_while_sewing=0), positions=positions)
    return env


def seam_points(name):
    c = task.TASK_SETUP[name]
    line = seam_path(name, c['ply_width'], c['ply_length'])
    points = []
    for i, (a, b) in enumerate(zip(line[:-1], line[1:])):
        length = np.linalg.norm(b-a)
        lo = .002 if i == 0 else 0.
        hi = length - .002 if i == len(line)-2 else length
        n = round((hi-lo)/.0025)
        seg = a + np.linspace(lo, hi, n+1)[:, None]*(b-a)/length
        points.extend(seg if i == 0 else seg[1:])
    return np.array(points)


def good_env(name):
    points = seam_points(name)
    if name == 'pivot':
        return fake_env(name, points)
    # Three reverse stitches over the initial and terminal 5 mm of the finished seam.
    all_points = np.vstack([points[:3][::-1], points, points[-3:][::-1]])
    return fake_env(name, all_points, reverse={0, 1, 2, len(all_points)-3, len(all_points)-2, len(all_points)-1})


def evaluator(env):
    ev = TaskEvaluator(env)
    # The load-seam tests below exercise actual loading geometry separately.
    if env.config.task == 'load-and-seam':
        ev.load_progress = lambda: 0.
    return ev


def test_catalogue_and_frozen_first_fifty():
    raw = json.loads((ROOT/'sewing/benchmark.json').read_text())
    assert raw['count'] == 9 and variations.COUNT == 90
    assert raw['variation_catalogue']['sha256'] == variations.digest()
    payload = ''.join(json.dumps(variations.specification(i), sort_keys=True, separators=(',', ':'))+'\n'
                      for i in range(50)).encode()
    assert hashlib.sha256(payload).hexdigest() == '500f97ec88c1a0aa0594440223a85241861bd452722d6011061147368ce75a3d'
    entries = raw['task_set']['tasks']
    assert [e['selection'] for e in entries] == list(TASK_SET)
    for e in entries:
        r = variations.task_resolve(e['selection'])
        assert r['task_id'] == e['task_id']
        assert choose_contract(r['selection']['task'], r['selection'].get('contract')) == e['task_contract']
    assert task_resolve({})['backend_options'] == dict(preset='sample-room', task='plain-seam', seed=0)


@pytest.mark.parametrize('name', REVISED)
def test_contract_choice_keeps_ids_and_defaults(name):
    old = task_resolve(dict(task=name))
    new = task_resolve(dict(task=name, contract=CONTRACT))
    assert old['task_id'] == new['task_id']
    assert 'contract' not in old['backend_options']
    assert new['backend_options']['contract'] == CONTRACT
    assert task_resolve(dict(task=name, contract=LEGACY)) == old
    idx = variations.index_of(name, 0)
    assert variations.task_resolve(dict(variation_id=idx, contract=CONTRACT))['backend_options'] == new['backend_options']
    assert variations.specification(idx)['sha256'] != variations.specification(idx, CONTRACT)['sha256']


@pytest.mark.parametrize('selection', [dict(task='curve',contract=CONTRACT), dict(task='baste',contract=CONTRACT),
                                      dict(task='load',seed=True), dict(task='load',seed='0'),
                                      dict(variation_id=90), dict(variation_id=10,task='curve'),
                                      dict(task='load',contract='bogus')])
def test_invalid_selections(selection):
    with pytest.raises(ValueError):
        variations.task_resolve(selection)


def test_aliases_and_new_ids():
    for i, name in enumerate(variations.TASK_ORDER):
        for seed in range(10):
            assert variations.task_seed(i*10+seed) == (name, seed)
            assert task_resolve(dict(task=task.TASKS[name],seed=seed)) == task_resolve(dict(task=name,seed=seed))
    assert task_resolve(dict(task='pivot',contract=CONTRACT))['description'].startswith('Join the two plies')
    assert task.INSTRUCTIONS['curve'] == task_resolve(dict(task='curve'))['description']
    assert '120 mm' in task.INSTRUCTIONS['seam-to-notch'] and 'chalk' not in task.INSTRUCTIONS['seam-to-notch']


@pytest.mark.parametrize('name', ['plain-seam','seam-to-notch','pivot','three-sides','load-and-seam'])
def test_good_seams_and_untouched_starts(name):
    good = evaluator(good_env(name)).evaluate()
    assert good['success'], good
    assert good['score'] == 1.
    empty = evaluator(fake_env(name)).evaluate()
    assert not empty['success'] and empty['score'] == 0.
    assert empty['task_contract'] == CONTRACT


@pytest.mark.parametrize('name', ['plain-seam','seam-to-notch','pivot','three-sides','load-and-seam'])
def test_reverse_sewing_order_is_valid(name):
    env = good_env(name)
    env.world.machine.stitches.reverse()
    ev = evaluator(env).evaluate()
    assert ev['success'], ev


def test_gap_registration_stray_and_cancelled_pinch():
    for defect, check in [('gap','no_gaps'),('registration','plies_registered'),('stray','seam_line'),
                           ('pinch','released'),('needle','needle_intact'),('thread','no_thread_break')]:
        env = good_env('plain-seam'); M=env.world.machine
        if defect == 'gap':
            del M.stitches[40:45]
        elif defect == 'registration':
            h = M.holes[M.stitches[45]['holes'][0]]
            h.uv = (h.uv[0], h.uv[1]+.004)
        elif defect == 'stray':
            h = M.holes[M.stitches[45]['holes'][1]]
            h.uv = (h.uv[0]-.004, h.uv[1])
        elif defect == 'pinch':
            env.world.pads = [(0,None,None,None),(1,None,None,None)]
            env.world.cloth=NS(shapes=[dict(name='left_f1'),dict(name='left_f2')],
                               shape_reaction=lambda i: (np.array([(-1)**i,0.,0.]),np.zeros(3)))
        elif defect == 'needle': M.needle_state='bent'
        else: M.threads=[dict(broken=True,cycle=45)]
        r = evaluator(env).evaluate()
        assert not r['checks'][check]['ok'], (defect,r)
        assert not r['success'] and 0 <= r['score'] <= .95


def test_notch_overrun_and_unformed_punctures():
    for formed in (True,False):
        env=good_env('seam-to-notch');M=env.world.machine
        M.holes.append(NS(ply=0,uv=(.19,.290)))
        M.stitches.append(dict(formed=formed,cycle=1000,holes=[len(M.holes)-1],reverse=False,
                               reason='missed_loop',balance='balanced',seam=1))
        r=evaluator(env).evaluate()
        assert not r['success'] and not r['checks']['opening_clear']['ok']
        assert r['score'] < .95


def test_notch_longitudinal_tolerance_does_not_change_lateral_tolerance():
    env=good_env('seam-to-notch')
    # Move the finished seam by 4 mm toward its stopping point: its first stitch
    # still reaches within 6 mm of the far end; the last is 2 mm past the mark.
    for h in env.world.machine.holes:
        h.uv=(h.uv[0],h.uv[1]-.004)
    r=evaluator(env).evaluate()
    assert r['success'],r
    for h in env.world.machine.holes:
        h.uv=(h.uv[0]-.002,h.uv[1])
    r=evaluator(env).evaluate()
    assert not r['checks']['seam_line']['ok']


def test_backtacks_must_be_at_ends_and_corner_on_both_plies():
    env=good_env('plain-seam')
    for s in env.world.machine.stitches: s['reverse']=False
    for s in env.world.machine.stitches[30:40]: s['reverse']=True
    r=evaluator(env).evaluate()
    assert not r['checks']['backtack_start']['ok'] and not r['checks']['backtack_end']['ok']
    env=good_env('three-sides')
    corner=np.array([.01,.01])
    for h in env.world.machine.holes:
        if np.linalg.norm(np.array(h.uv)-corner)<.004:
            h.uv=tuple(np.array(h.uv)+[.004,.004])
    r=evaluator(env).evaluate()
    assert not r['checks']['corner_2']['ok']


def test_coverage_continuous_and_overlap_not_extra_work():
    a,_=weighted_coverage([.01],[1.],.006,.1)
    b,_=weighted_coverage([.01,.01],[1.,1.],.006,.1)
    assert a == b
    values=[weighted_coverage([.0043+d],[1.],.006,.1)[0] for d in (0,1e-6,2e-6)]
    assert values[0] < values[1] < values[2]
    assert values[2]-values[1] == pytest.approx(values[1]-values[0])
    short=evaluator(fake_env('plain-seam', seam_points('plain-seam')[:3])).evaluate()
    longer=evaluator(fake_env('plain-seam', seam_points('plain-seam')[:25])).evaluate()
    assert 0 < short['score'] < .03 < longer['score'] < .3


def test_load_progress_zero_without_plies_and_graded_with_alignment():
    env=fake_env('load');ev=evaluator(env)
    ev.ply_registration=lambda: (0.,100)
    ev.under_needle=lambda k: []
    assert ev.evaluate()['score'] == 0.
    env.world.machine.foot_N=2.
    assert ev.evaluate()['score'] == 0.  # foot on an empty plate
    ev.under_needle=lambda k: [(.183,.415)]
    a=ev.evaluate()['score']
    ev.under_needle=lambda k: [(.186,.415)]
    b=ev.evaluate()['score']
    ev.under_needle=lambda k: [(.190,.415)]
    r=ev.evaluate()
    assert 0 < a < b < 1 and r['success'] and r['score']==1.


def test_square_start_success_fold_lift_reflex_and_release():
    env=fake_env('square-plies',offset=(.040,.025));ev=evaluator(env)
    assert ev.evaluate()['score'] == 0.
    env.positions[1][:,:2] -= [.020,.0125]
    halfway=ev.evaluate()['score']
    env.positions[1][:,:2]=env.positions[0][:,:2]
    r=ev.evaluate()
    assert 0 < halfway < 1 and r['success'] and r['score']==1.
    env.positions[1][:,2] += .010
    r=ev.evaluate()
    assert not r['success'] and r['score'] < .95
    env.positions[1][:,2] -= .010
    env.world.reflexes=[{}]
    assert not ev.evaluate()['checks']['no_reflex']['ok']
    env.world.reflexes=[]
    env.world.finger_force=lambda arm: np.array([1.,0.,0.])
    assert not ev.evaluate()['checks']['released']['ok']
    env.world.finger_force=lambda arm: np.zeros(3)
    env.positions[1][env.world.mesh['X'][:,0]>.1,0] = -.3 - (env.world.mesh['X'][env.world.mesh['X'][:,0]>.1,0]-.1)
    assert not ev.evaluate()['success']


def test_vertical_intersection_deduplicates_shared_triangle_edges():
    tri=np.array([[0,1,2],[0,2,3]])
    p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
    assert layer_count([.5,.5],tri,p)==1
    doubled=np.vstack([p,p+[0,0,.01]])
    assert layer_count([.5,.5],np.vstack([tri,tri+4]),doubled)==2
    # A flat fold at the SAME height still crosses two different material points.
    folded=np.vstack([p,p])
    uv=np.vstack([p[:,:2],p[:,:2]+[2.,0.]])
    assert layer_count([.5,.5],np.vstack([tri,tri+4]),folded,uv)==2


def adapter_module():
    # Real Config definition only: no import of world.py or any engine dependency.
    tree=ast.parse((ROOT/'sewing/world.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Config')
    import dataclasses
    from sewing import params, layout
    ns=dict(dataclass=dataclasses.dataclass,P=params,LY=layout)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'Config','exec'),ns)
    world=ModuleType('sewing.world');world.Config=ns['Config']
    base=ModuleType('sewing.env')
    class Base:
        def __init__(self,config,output_dir=None):
            self.config=config;self.start_state=config.start;self.assisted_log=[]
            self.world=NS(config=config,cloth=None)
            self.evaluator=task.Evaluator(self)
            self._make_start()
        def _make_start(self,use_cache=True): self.prepared_config=self.config
        def get_state(self,evaluate=True): return dict(instruction=task.INSTRUCTIONS[self.config.task],evaluation={})
    base.SewingEnv=Base
    name='sewing._task_set_env_stubbed'
    spec=importlib.util.spec_from_file_location(name,ROOT/'sewing/task_set_env.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'sewing.world':world,'sewing.env':base,name:module}):
        spec.loader.exec_module(module)
    return module,ns['Config']


def test_adapter_reuses_physics_starts_and_contract_selection():
    module,Config=adapter_module()
    for name,physical in [('plain-seam','plain-seam'),('curve','curve'),('seam-to-notch','plain-seam'),
                          ('three-sides','pivot'),('load-and-seam','load')]:
        c=Config(**__import__('sewing.variants',fromlist=['resolve']).resolve(task=name))
        e=module.make_environment(c,contract=CONTRACT if name=='plain-seam' else None)
        assert e.config.task==name and e.world.config.task==physical
        assert e.prepared_config.task==physical
        assert e.get_state()['instruction']==task_resolve(dict(task=name,contract=e.task_contract))['description']
        assert isinstance(e.evaluator,TaskEvaluator) == (name!='curve')
    legacy=module.make_environment(Config(task='plain-seam'))
    assert type(legacy.evaluator) is task.Evaluator
    assert legacy.task_contract==LEGACY


def test_slipped_placement_and_reach_against_existing_grid():
    module,_=adapter_module()
    from sewing.variants import stack_offset
    from sewing.layout import TABLE, BED
    # All seeds: rectangle footprint and a near-right top-ply grasp patch. The E0b
    # left-arm grid spans x [-.35,0], y [-.18,.04], z [.02,.08]. This is a static
    # workspace comparison, not a new IK/collision qualification.
    for seed in range(10):
        (dx,dy,yaw),_=stack_offset('square-plies',seed)
        X=np.array([[0,0],[.2,0],[.2,.42],[0,.42],[.16,.07]])
        a=np.deg2rad(yaw);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        centre=np.array([-.2+dx,-.17+dy+.21])
        xy=(X+[-.4+dx,-.17+dy]-centre)@R.T+centre
        p=np.column_stack([xy,np.full(5,.001)])
        slipped=module.slip_top(p)
        assert np.allclose(np.linalg.norm(slipped[1]-slipped[0]),.2)
        assert (slipped[:,0]>TABLE['x'][0]).all() and (slipped[:,0]<0.).all()
        on_bed=(slipped[:,0]>=BED['x'][0]) & (slipped[:,1]>=BED['y'][0]) & (slipped[:,1]<=BED['y'][1])
        assert not on_bed.any()
        assert (slipped[:,1]>TABLE['y'][0]).all() and (slipped[:,1]<TABLE['y'][1]).all()
        x,y=slipped[-1,:2]
        assert -.35 <= x <= 0 and -.18 <= y <= .04


def test_gar_catalogue_regression_with_robot_metadata_stub():
    # Stub the robot catalogue's metadata only; real task/CLI
    # resolution and exact-option merge run through the public GAR entry points.
    import unittest
    catalog = ModuleType('robots.catalog')
    catalog.default_robot = lambda name: 'dual-fr3'
    catalog.environment_robots = lambda name: ['dual-fr3', 'dual-panda', 'single-fr3', 'g1']
    catalog.select_for_backend = lambda name, robot: robot
    catalog.ENVIRONMENTS = {}
    with patch.dict(sys.modules, {'robots': ModuleType('robots'), 'robots.catalog': catalog}):
        from tests.test_sewing import CatalogueTest
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(CatalogueTest).run(result)
        assert not result.failures and not result.errors, (result.failures, result.errors)
        from gym_anything_robotics import tasks
        for selection in TASK_SET:
            spec = tasks.resolve('sewing', selection)
            assert spec.launch.backend_options == task_resolve(selection)['backend_options']


def test_backend_preview_and_reset_propagate_contract_without_building():
    from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
    from gym_anything_robotics.errors import ActionRejectedError
    b = object.__new__(NewtonSewingBackend)
    b.config = NS(robot='dual-fr3', seed=0)
    result = b.select_variation(dict(variation_id=30, contract=CONTRACT))
    assert result['description'].startswith('Join the two plies')
    assert result['reset_request'] == dict(options=dict(variation_id=30, contract=CONTRACT))
    calls = []
    b._start_variation = lambda i, c=None: calls.append((i,c))
    b.reset(**result['reset_request'])
    assert calls == [(30,CONTRACT)]
    b.reset(options=dict(variation_id=40))
    assert calls[-1] == (40,None)  # explicit variation switch takes its own default
    with pytest.raises(ActionRejectedError): b.reset(seed=2, options=dict(variation_id=30))


def test_square_placement_is_before_preparation_and_preserves_other_particles():
    module, Config = adapter_module()
    c=Config(task='square-plies',start='tray')
    e=object.__new__(module.TaskEnvironment)
    e.catalogue_config=e.config=c
    e.assisted_log=[]
    x=np.array([[0.,0.,.1],[.2,0.,.1],[0.,.4,.1],[.2,.4,.1],[9.,8.,7.]])
    saved={}
    cl=NS(plies=[{},dict(v0=0,nv=4)],snapshot=lambda:dict(x=x.copy(),other='preserved'),
          restore=lambda snap:saved.update(snap))
    e.world=NS(cloth=cl,ply_ids=[0,1])
    module.TaskEnvironment._make_start(e)
    assert np.allclose(saved['x'][:4],module.slip_top(x[:4]))
    assert np.array_equal(saved['x'][4],x[4]) and saved['other']=='preserved'
    assert e.prepared_config is c
    assert e.assisted_log[-1]['initial_condition']==module.SQUARE_SETUP
    assert module.SquareConfig(task='square-plies').task_setup_revision==module.SQUARE_SETUP


def test_backend_switch_installs_new_contract_and_plain_reset_keeps_it():
    import threading
    from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
    from sewing.task_set import instruction
    _, Config = adapter_module()
    world_module=ModuleType('sewing.world');world_module.Config=Config
    env_module=ModuleType('sewing.task_set_env')
    built=[]
    def make(config,contract=None,output_dir=None):
        e=NS(config=config,lock=threading.RLock(),generation=0,steps=0,world=NS(tick=0),
             task_contract=choose_contract(config.task,contract),start_state=config.start,close=lambda:None)
        e.get_state=lambda:dict(tick=e.steps,generation=e.generation,task=config.task,
                               evaluation=dict(task_contract=e.task_contract,success=False,score=0.))
        def reset():
            e.generation+=1
            return e.get_state()
        e.reset=reset
        built.append(e)
        return e
    env_module.make_environment=make
    b=object.__new__(NewtonSewingBackend)
    b.config=Config(task='curve',seed=0,robot='dual-fr3')
    b.env=make(b.config)
    b.trace=None;b._config_overrides={};b._device=None;b.output=None
    b._take_prestart=lambda:(None,None);b._take_retained=lambda:None
    b.capabilities=NS(notes={});b.last_timing={};b._last_valid_evaluation=None;b._state_cache=None
    with patch.dict(sys.modules,{'sewing.world':world_module,'sewing.task_set_env':env_module}):
        state,meta=b._start_variation(30,CONTRACT)
        assert b.task_contract==CONTRACT and b.config.task=='pivot'
        assert state['evaluation']['task_contract']==CONTRACT
        assert b.capabilities.notes['instruction']==instruction('pivot',contract=CONTRACT)
        assert meta['variation']['contract']==CONTRACT
        old=b.env
        state,_=b.reset()
        assert b.env is old and b.task_contract==CONTRACT and state['generation']==2
        b._start_variation(40)
        assert b.task_contract==CURVE and b.config.task=='curve'
        assert len(built)==3

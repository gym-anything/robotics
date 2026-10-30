"""Stubbed constructor/reset, initialization, and measurement checks only.

No simulation dependency is imported. Kernel bodies are executed as ordinary
Python over tiny arrays with a scalar Warp stand-in; this is not GPU validation.
"""

import ast
from contextlib import ExitStack
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from titrate.newton15.lab_config import LabConfig
from titrate.newton15.liquid_charge import Reagents
from titrate.newton15.task_catalogue import TASKS
from titrate.newton15.task_presets import preset
from titrate.newton15.task_session import apply_rigid_start, make_session, precharge_chemistry
from .test_task_set import initial, ROOT


class Array:
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.array(value, copy=True)


class Chemistry:
    def __init__(self, n, volume_ml):
        self.volume_l = volume_ml/1000
        self.moles = Array(np.zeros((n, 5)))
        self.next = Array(np.zeros((n, 5)))
        self.reacted = Array(np.zeros(n))
        self.next_reacted = Array(np.zeros(n))

    def charge(self, indices, acid_m=0., base_m=0., indicator_m=0.):
        n = self.moles.numpy()
        n[indices] = self.volume_l*np.array([acid_m, base_m, base_m, acid_m, indicator_m])
        self.moles.assign(n)

    def observe(self):
        return dict(totals_mol=dict(zip(('H', 'OH', 'Na', 'Cl', 'indicator'), self.moles.numpy().sum(0))))


class Liquid:
    def __init__(self, world, numerics, reagents):
        self.world, self.reagents = world, reagents
        r, vml = reagents, .0625
        self.initial_groups, first = {}, 0
        for name, volume in (('flask', r.aliquot_ml+r.dilution_ml), ('burette', r.burette_ml),
                             ('acid_bottle', r.acid_bottle_ml), ('base_bottle', r.base_bottle_ml)):
            last = first+round(volume/vml)
            self.initial_groups[name] = (first, last)
            first = last
        self.chemistry = Chemistry(first, vml)
        for name, (a, b) in self.initial_groups.items():
            if name == 'flask':
                self.chemistry.charge(slice(a, b), acid_m=r.acid_molar*r.aliquot_ml/(r.aliquot_ml+r.dilution_ml),
                                      indicator_m=r.indicator_molar)
            else:
                self.chemistry.charge(slice(a, b), acid_m=r.acid_molar if name == 'acid_bottle' else 0.,
                                      base_m=0. if name == 'acid_bottle' else r.base_molar)
        self.initial_totals = self.chemistry.observe()['totals_mol']


class Session:
    instances = []

    def __init__(self, config=None, **kwargs):
        self.config = config or LabConfig()
        self.dt_s = kwargs.get('dt_s', .001)
        self.output = Path(kwargs.get('output_dir') or '/unused')
        self.renderer_options = kwargs.get('renderer_options')
        self.liquid = NS(reagents=kwargs.get('reagents') or Reagents()) if kwargs.get('liquid_numerics') else None
        self.closed, self.tick, self.trace_liquid_stride = False, 0, 20
        self.world = NS(lab=NS(robot_joints={}))
        self.magnet = NS(rpm=0.)
        self.kwargs = kwargs
        self.instances.append(self)

    def close(self):
        self.closed = True

    def get_state(self):
        return {'tick': self.tick, 'evaluation': self.evaluator.result if hasattr(self, 'evaluator') else None}

    def snapshot(self):
        return self.evaluator.snapshot()

    def reset(self):
        self.evaluator.restore(self._initial)
        return self.get_state()


class Readings:
    def __init__(self, liquid, setup):
        self.liquid, self.setup = liquid, setup

    def read(self):
        r = Reagents(**self.setup['nominal_reagents'])
        _, m = initial(self.setup['task'], reagents=r)
        return m


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
        self.apply = self.stack.enter_context(patch('titrate.newton15.task_session.apply_rigid_start'))
        self.numerics = NS(spacing_m=.000625, dt_s=.000125, device='cuda:0')

    def test_factory_all_tasks_initial_checkpoint_and_default_passthrough(self):
        for name in TASKS:
            s = make_session(LabConfig(), task=name, dt_s=.000125, liquid_numerics=self.numerics)
            if name == 'titration':
                self.assertIs(s.kwargs['liquid_numerics'], self.numerics)
                self.assertFalse(hasattr(s, 'task_setup'))
            else:
                self.assertIsNone(s.kwargs['liquid_numerics'])
                self.assertEqual(s.evaluator.result['score'], 0.)
                self.assertEqual(s.reset()['evaluation']['score'], 0.)
                self.assertEqual(s.task_setup['task'], name)
        self.assertEqual(self.apply.call_count, 7)

    def test_factory_failure_closes_candidate_and_dry_tasks_reject(self):
        with self.assertRaises(ValueError):
            make_session(task='centre-flask')
        with self.assertRaises(ValueError):
            make_session(task='centre-flask', dt_s=.001, liquid_numerics=self.numerics)
        self.assertTrue(Session.instances[-1].closed)

    def test_precharged_species_and_tracer_conservation(self):
        r = Reagents()
        p = preset('finish-titration', LabConfig(), r)
        liquid = Liquid(NS(), self.numerics, Reagents(**p['reagents']))
        precharge_chemistry(liquid, p)
        a, b = liquid.initial_groups['flask']
        n = liquid.chemistry.moles.numpy()
        reacted = liquid.chemistry.reacted.numpy()
        self.assertAlmostEqual(n[a:b, 2].sum(), .1*p['base0_ml']/1000)
        self.assertAlmostEqual(n[a:b, 3].sum(), .001)
        self.assertAlmostEqual(n[a:b, 4].sum(), .025*r.indicator_molar)
        self.assertEqual(float(n[a:b, 1].max()), 0.)  # Still acid, uniformly premixed.
        self.assertTrue((n[a:b, 0] > 0).all())
        for k, field in enumerate(('H', 'OH', 'Na', 'Cl', 'indicator')):
            total = n[:, k].sum()+(reacted.sum() if k < 2 else 0.)
            self.assertAlmostEqual(total, liquid.initial_totals[field])
        self.assertAlmostEqual(liquid.initial_totals['Na'], .004)
        self.assertAlmostEqual(liquid.initial_totals['Cl'], .003)

    def backend(self, task='centre-flask'):
        from gym_anything_robotics.backends.newton_titration import NewtonTitrationBackend
        return NewtonTitrationBackend(task=task, variation_id=0, liquid_numerics=self.numerics,
                                     recording=False, cameras=())

    def test_backend_constructor_reset_preview_and_instruction(self):
        b = self.backend()
        self.assertEqual(b.task, 'centre-flask')
        self.assertEqual(b.session.evaluator.name, b.task)
        access = b.policy_access()
        self.assertIn('flask', access.instruction())
        preview = b.select_variation({'task': 'finish-titration', 'variation_id': 25})
        self.assertEqual(b.task, 'centre-flask')
        old = b.session
        b.reset(**preview['reset_request'])
        self.assertTrue(old.closed)
        self.assertEqual(b.task, 'finish-titration')
        self.assertEqual(b.session.task_setup['nominal_reagents']['aliquot_ml'], 5.)
        b.reset(options={'task': 'run-in'})
        self.assertEqual(b.session.liquid.reagents.aliquot_ml, 5.)
        self.assertEqual(b.session.liquid.reagents.dilution_ml, 20.)
        self.assertEqual(b.session.liquid.reagents.burette_ml, 20.)
        b.reset(options={'variation_id': 0})
        self.assertEqual(b.task, 'run-in')
        self.assertEqual(b.reset()[0]['evaluation']['score'], 0.)
        preview = b.select_variation({'variation_id': 0})
        self.assertEqual(preview['reset_request']['options']['task'], 'titration')
        b.reset(**preview['reset_request'])
        self.assertEqual(b.task, 'titration')
        self.assertFalse(hasattr(b.session, 'task_setup'))

    def test_backend_bad_reset_keeps_live_session(self):
        b = self.backend()
        s = b.session
        for request in ({'options': {'task': 'unknown'}}, {'options': {'variation_id': True}},
                        {'options': {'variation_id': 1}, 'seed': 0}, {'options': {'other': 0}}):
            with self.assertRaises(ValueError):
                b.reset(**request)
            self.assertIs(b.session, s)
            self.assertFalse(s.closed)
        with patch('titrate.newton15.task_session.make_session', side_effect=ValueError('bad initial placement')):
            with self.assertRaises(ValueError):
                b.reset(options={'task': 'decant-acid'})
        self.assertIs(b.session, s)
        self.assertEqual(b.task, 'centre-flask')

    def test_initial_poses_update_both_buffers_and_joint_state_without_retuning(self):
        # Directly call the real function (the factory hook above remains stubbed).
        poses = np.array([[.15, .035, 1.0052, 0, 0, 0, 1], [.15, .035, 1.0102, 0, 0, 0, 1]], float)
        def state():
            return NS(body_q=Array(poses), joint_q=Array(np.r_[poses[0], 0.]), joint_qd=Array(np.zeros(7)))
        builder = NS(joint_child=[0, 9], joint_q_start=[0, 7], joint_type=[4, 1])
        world = NS(lab=NS(builder=builder, apparatus=NS(bodies={'flask': 0, 'stir_bar': 1}, joints={'stirrer_knob': 1})),
                   state=state(), next_state=state(), model=NS(), _validate_initial_contacts=Mock())
        fake_newton = NS(JointType=NS(FREE=4), eval_fk=Mock())
        with patch.dict(sys.modules, {'titrate.newton15.runtime': NS(require_newton=lambda: (fake_newton, None))}):
            p = preset('centre-flask', LabConfig(), Reagents())
            apply_rigid_start(world, p)
        for buffer in (world.state, world.next_state):
            np.testing.assert_allclose(buffer.body_q.numpy()[0, :3], p['positions']['flask'])
            np.testing.assert_allclose(buffer.joint_q.numpy()[:3], p['positions']['flask'])
        self.assertEqual(fake_newton.eval_fk.call_count, 2)
        world._validate_initial_contacts.assert_called_once()


def python_functions(file, names, namespace):
    namespace.update(__name__='titrate.newton15.scalar_test', __package__='titrate.newton15')
    tree = ast.parse((ROOT/file).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
        for arg in (*node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
        node.returns = None
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/file), 'exec'), namespace)
    return namespace


class InventoryTest(unittest.TestCase):
    def test_rigid_readings_use_current_pose_angular_speed_and_all_robot_links(self):
        ns = python_functions('task_observations.py', ['tilt', 'rigid_measurements'],
                              {'math': math, 'np': np, 'motor_rpm': lambda angle: 500.})
        names = dict(zip(('flask', 'acid_bottle', 'rinse_beaker', 'stirrer', 'stir_bar'), range(5)))
        c = LabConfig()
        poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (5, 1))
        poses[0, :3] = c.flask_position
        poses[1, :3] = (-.05, -.28, .9002)
        poses[2, :3] = (.12, -.28, .9002)
        poses[3, :3] = (.15, .035, .9002)
        poses[4, :3] = (.15, .035, 1.0102)
        vel = np.zeros((5, 6))
        vel[4, 5] = 500*2*math.pi/60
        labels = ['flask', 'panda/panda_link3', 'acid_bottle', 'stirrer_knob']
        contacts = [NS(geom=(0, 1), dist=0.), NS(geom=(1, 3), dist=.00001), NS(geom=(1, 2), dist=.1)]
        world = NS(lab=NS(config=c, apparatus=NS(bodies=names)),
                   state=NS(body_q=Array(poses), body_qd=Array(vel)), joint=lambda name: (1.4, 0.),
                   solver=NS(mj_model=NS(geom_bodyid=np.arange(4), body=lambda i: NS(name=labels[i])),
                             mj_data=NS(contact=contacts)))
        def inverse(points, pose):
            return Rotation.from_quat(pose[3:]).inv().apply(points-pose[:3])
        with patch.dict(sys.modules, {'titrate.newton15.solid_sampling': NS(inverse_transformed=inverse)}):
            p = preset('start-stirring', c, Reagents())
            m = ns['rigid_measurements'](world, p)
            self.assertAlmostEqual(m['bar_rpm'], 500.)
            self.assertAlmostEqual(m['flask_plate_error_m'], 0.)
            self.assertTrue(m['flask_robot_contact'])
            self.assertTrue(m['stirrer_robot_contact'])
            self.assertFalse(m['bottle_robot_contact'])
            poses[0, :3] += [.06, 0., .01]
            poses[0, 3:] = Rotation.from_euler('x', .3).as_quat()
            world.state.body_q.assign(poses)
            m = ns['rigid_measurements'](world, p)
            self.assertAlmostEqual(m['flask_distance_m'], .06)
            self.assertAlmostEqual(m['flask_plate_error_m'], .035)
            self.assertAlmostEqual(m['flask_tilt_rad'], .3)

    def test_actual_kernel_body_with_scalar_warp_stub_and_nested_vessels(self):
        # Execute the production kernel as plain Python, no Warp import/JIT.
        class WP:
            pi, float64, abs, floor, atan2, cos, sin = math.pi, float, abs, math.floor, math.atan2, math.cos, math.sin
            current = 0
            def tid(self): return self.current
            def vec2(self, *v): return np.array(v)
            vec3 = vec4 = vec2
            def length(self, v): return np.linalg.norm(v)
            def transform_inverse(self, p):
                r = Rotation.from_quat(p[3:]).inv()
                return np.r_[-r.apply(p[:3]), r.as_quat()]
            def transform_point(self, p, v): return Rotation.from_quat(p[3:]).apply(v)+p[:3]
        wp = WP()
        # Port-space samples are tested with the actual host CSG expressions.
        from titrate.newton15.bored_geometry import plug_sdf, socket_sdf
        def field(p, kind, dims):
            sdf = plug_sdf(p, dims[0], dims[2], dims[3]) if kind == 1 else socket_sdf(p, *dims)
            return (0., 0., 0., float(sdf))
        ns = python_functions('task_readings.py', ['cavity'], {'wp': wp})
        ns.update(field=field, REST_SPEED_M_S=.05)
        python_functions('task_observations.py', ['inventory'], ns)
        profiles = [[(.02, 0), (.02, .1)]]*5 + [[(.005, .008), (.005, .7)], [(.001, -.055), (.001, -.008)]]
        flat = np.array([v for p in profiles for v in p])
        layout = np.array([(i*2, 2) for i in range(7)])
        poses = np.tile([3., 3., 3., 0., 0., 0., 1.], (6, 1))
        poses[3] = [0., 0., 0., 0., 0., 0., 1.]  # Beaker.
        poses[1] = [0., 0., .05, 0., 0., 0., 1.]  # Bottle neck inside it.
        valve = np.array([1., 0., 1., 0., 0., 0., 1.])
        poses[5] = valve
        points = np.array([[0., 0., .02], [0., 0., .07], [1., 0., 1.1], [1., 0., .96], [2., 0., .5], [1., 0., 1.0075]])
        vel = np.zeros_like(points)
        vel[3:, 2] = -.1
        n = np.zeros((len(points), 5))
        n[:, 2], n[:, 3] = np.arange(1., 7.), np.arange(11., 17.)
        out = np.zeros((4, len(points)))
        for i in range(len(points)):
            wp.current = i
            ns['inventory'](points, vel, n, poses, np.array([0, 1, 2, 3, 4, -1, -1]), flat, layout,
                            valve, 5, .007, .014, .001, .1, out)
        self.assertEqual(out[0].sum(), 1.)  # Bottle contents cannot count as beaker transfer.
        self.assertEqual(out[1].sum(), 11.)
        self.assertAlmostEqual(out[2].sum(), .3)  # Column, tip, socket port; each once.
        self.assertEqual(out[3].sum(), 9.)  # Moving tip + loose sodium, not inside the bottle.


if __name__ == '__main__':
    unittest.main()

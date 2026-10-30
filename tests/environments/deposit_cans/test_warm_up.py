"""Warm-up before ready: recording the first tick's graphs executes nothing, and that tick launches them."""
from types import SimpleNamespace
import unittest
from unittest import mock
import warp as wp
from laundry.newton15.graph import CapturedClothStep, CapturedCollision
from deposit_cans.simulation import Simulation


@wp.kernel
def bump(a: wp.array[float]):
    a[0] = a[0] + 1.0


class Native:
    def __init__(self, device):
        self.device, self.calls, self._contact_matcher = device, 0, None

    def step(self, state_in, state_out, control, contacts, dt):
        self.calls += 1
        wp.launch(bump, 1, [state_out], device=self.device)

    def collide(self, state, contacts):
        self.calls += 1
        wp.launch(bump, 1, [contacts], device=self.device)


@unittest.skipUnless(wp.is_cuda_available(), 'CUDA graph capture needs a GPU')
class RecordTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device('cuda:0')
        self.out = wp.zeros(1, dtype=float, device=self.device)

    def check(self, native, record, run):
        record()
        wp.synchronize_device(self.device)
        self.assertEqual(self.out.numpy()[0], 0.)   # recorded only: no kernel ran
        run()
        self.assertEqual(self.out.numpy()[0], 1.)   # the tick launches the recorded graph once
        self.assertEqual(native.calls, 1)           # and does not record it again

    def test_step(self):
        native, state, contacts = Native(self.device), object(), object()
        step = CapturedClothStep(native)
        self.check(native, lambda: step.record_graph(state, self.out, None, contacts, 1 / 240),
                   lambda: step.step(state, self.out, None, contacts, 1 / 240))

    def test_collision(self):
        native, state = Native(self.device), object()
        pipeline = CapturedCollision(native)
        self.check(native, lambda: pipeline.record_graph(state, self.out), lambda: pipeline.collide(state, self.out))


class WarmUpTests(unittest.TestCase):
    def sim(self, *, capture=True, coupled=False):
        calls = []
        sim = SimpleNamespace(fault=None, config=SimpleNamespace(capture=capture, device='cuda:0', dt=1 / 240),
                              state_0=object(), state_1=object(), control=object(), contacts=object())
        sim._collision_states = {id(sim.state_0): object(), id(sim.state_1): object()} if coupled else None
        sim.pipeline = SimpleNamespace(record_graph=lambda *a: calls.append(('collide', a)))
        sim.solver = SimpleNamespace(record_graph=lambda *a: calls.append(('step', a)))
        return sim, calls

    def warm_up(self, sim, cuda=True):
        with mock.patch('deposit_cans.simulation.wp.get_device', return_value=SimpleNamespace(is_cuda=cuda)), \
             mock.patch('deposit_cans.simulation.wp.load_module') as load:
            Simulation.warm_up(sim, modules=('m',))
        return load

    def test_records_the_first_ticks_graph_keys(self):
        for coupled in (False, True):
            sim, calls = self.sim(coupled=coupled)
            load = self.warm_up(sim)
            collision = sim._collision_states[id(sim.state_0)] if coupled else sim.state_0
            # The arguments step() passes on the next tick: state_0 -> state_1 with the model's dt.
            self.assertEqual(calls, [('collide', (collision, sim.contacts)),
                                     ('step', (sim.state_0, sim.state_1, sim.control, sim.contacts, sim.config.dt))])
            load.assert_called_once_with('m', device='cuda:0')

    def test_nothing_without_graph_dispatch(self):
        for capture, cuda in ((False, True), (True, False)):
            sim, calls = self.sim(capture=capture)
            load = self.warm_up(sim, cuda=cuda)
            self.assertEqual(calls, [])
            load.assert_not_called()

    def test_environment_modules_are_imported_and_loadable(self):
        # wp.load_module refuses a module that is not imported or holds no Warp code; the environment
        # warms up without a trace recorder too, so it must import them itself.
        from warp._src.context import user_modules
        from deposit_cans.env import DepositCansEnv
        with mock.patch.object(Simulation, 'warm_up') as warm_up:
            object.__new__(DepositCansEnv).warm_up()
        modules = warm_up.call_args.kwargs['modules']
        self.assertEqual([m.__name__ for m in modules],
                         ['deposit_cans.unified_candidate', 'deposit_cans.self_contact_evidence', 'deposit_cans.trace'])
        for module in modules:
            self.assertIn(module.__name__, user_modules)

    def test_refuses_a_faulted_simulation(self):
        sim, calls = self.sim()
        sim.fault = 'FloatingPointError after tick 3'
        with self.assertRaises(RuntimeError):
            self.warm_up(sim)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()

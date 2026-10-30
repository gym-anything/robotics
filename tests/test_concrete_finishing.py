"""Concrete finishing: discovery, robots, renderer options, and the backend's evaluation, capabilities, wait command and
reset contract.

CPU-only tests run by default (seconds). The backend tests build the full room on the CPU without cameras (about a
minute each on a loaded machine) and run only with GAR_SLOW_TESTS=1:
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_concrete_finishing.py
  GAR_SLOW_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_concrete_finishing.py
"""
import math
import os
import unittest

SLOW = os.environ.get("GAR_SLOW_TESTS") == "1"


class Discovery(unittest.TestCase):
    def test_launch_and_robots(self):
        from gym_anything_robotics.launch import describe
        from robots.catalog import for_environment, environment_robots
        d = describe("concrete-finishing")
        self.assertEqual(d["default_robot"], "mobile-fr3-duo")
        self.assertEqual({r["id"] for r in environment_robots("concrete-finishing")}, {"mobile-fr3-duo", "mobile-fr3-single"})
        with self.assertRaises(ValueError) as cm:
            for_environment("concrete-finishing", "g1")
        self.assertIn("floor level", str(cm.exception))                 # the refusal names its reason

    def test_renderer_options_refused_with_reason(self):
        from gym_anything_robotics.backends.newton_sweeping import check_renderer_options
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        for bad, word in (({"idle_relase_s": 5}, "Unknown renderer_options"), ({"samples": 100000}, "1..256"),
                          ({"idle_release_s": -1}, "idle_release_s")):
            with self.assertRaises(UnsupportedConfigurationError) as cm:
                check_renderer_options(bad)
            self.assertIn(word, str(cm.exception))

    def test_refusals_note_says_what_the_gateway_does_with_a_request_stopped_part_way(self):
        # The gateway answers a request stopped part-way with partial true and keeps the session ready (http_server.py
        # recover_known_partial_batches, test_eval_partial_batches.py), so the note must not ask for a recovery.
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_concrete_finishing import NewtonConcreteFinishingBackend
        from gym_anything_robotics.http_server import GARHTTPService
        stub = SimpleNamespace(min_action_s=0.05, cameras=(), options=dict(width=640, height=360, samples=16),
                               arms=("left", "right"), config=SimpleNamespace(reflex_profile="libfranka default"),
                               _long_command_note=lambda: "", _worker_note=lambda: "")
        text = NewtonConcreteFinishingBackend._base_notes(stub)["refusals"]
        self.assertTrue(GARHTTPService.recover_known_partial_batches)
        self.assertNotIn("recover the session", text)
        for word in ("no recovery needed", "partial true", "completed_actions", "rejected_action_index",
                     "unexecuted_action_indices", "refused wait"):
            self.assertIn(word, text)


class Deposits(unittest.TestCase):
    """What a tool lays down (a set-down roll, a spill, a heave) conserves volume, is no thicker where the region's edge
    cuts its footprint, lies where the blade's leading face was, and a spill stands no higher than its roll."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from concrete_finishing import layout as Lay, params as P
        from concrete_finishing.coupling import Coupler, Contactor
        from concrete_finishing.paste import Sole
        cls.np, cls.P = np, P
        cls.reg = Lay.patch_region(0, 21.0)
        cls.c = Contactor("float", 0, np.zeros(3), np.eye(3), Sole("blade", 0.203, 0.0445), 1.0)
        cls.cp = Coupler({"patch": cls.reg}, [cls.c], 0.0005)

    def _set_down(self, x, y, lead, ml):
        np, reg = self.np, self.reg
        tb0, off0 = reg.tb.copy(), reg.off_m3
        q = np.zeros((1, 7)); q[0, :3] = (x, y, 0.0); q[0, 6] = 1.0
        self.cp._last_q = q
        self.c.roll, self.c.lead = ml*1e-6, np.array(lead, float)
        self.cp._set_down(self.c, reg)
        add = reg.tb - tb0
        reg.tb[:] = tb0
        laid, off = float(add.sum())*reg.cell**2*1e6, (reg.off_m3 - off0)*1e6
        reg.off_m3 = off0
        return add, laid, off

    def test_set_down_conserves_and_is_not_piled_at_the_edge(self):
        reg = self.reg
        cx, cy = self.P.PATCH_CENTRE
        whole, laid, off = self._set_down(cx, cy, (0, 1), 20.0)
        self.assertAlmostEqual(laid + off, 20.0, places=6)
        self.assertAlmostEqual(off, 0.0, places=9)
        cut, laid2, off2 = self._set_down(reg.o[0] + reg.extent()[0] + 0.15, cy, (0, 1), 20.0)
        self.assertAlmostEqual(laid2 + off2, 20.0, places=6)
        self.assertGreater(off2, 10.0)
        self.assertLessEqual(cut.max(), whole.max()*1.0001)     # no thicker where the edge cuts the line

    def test_set_down_lengthwise_lies_at_the_leading_end(self):
        cx, cy = self.P.PATCH_CENTRE
        add, laid, off = self._set_down(cx, cy, (1, 0), 5.0)
        ii, jj = self.np.nonzero(add > 1e-9)
        self.assertLess((jj.max() - jj.min() + 1)*self.reg.cell, 0.10)      # across: the blade's width, not the region
        self.assertGreater(self.reg.u[ii.min()] + self.reg.o[0], cx + 0.2)  # ahead of the leading end

    def test_spill_of_a_blade_moving_lengthwise_lies_beside_its_leading_end(self):
        # an edger run along a form pushes its roll with its end: the overflow runs off beside its leading corners, not
        # onto the track ahead (re-cut every tick) or behind it (the worked track)
        np, reg = self.np, self.reg
        cx, cy = self.P.PATCH_CENTRE
        x, y = cx, cy + 0.25                                     # on the slab north of the cut
        tb0 = reg.tb.copy()
        self.c.lead = np.array([1.0, 0.0])                       # along the float's long axis (+x)
        self.cp._spill(self.c, reg, np.array([x, y, 0.0]), np.eye(3), 2e-6)
        add = reg.tb - tb0
        reg.tb[:] = tb0
        ii, jj = np.nonzero(add > 1e-9)
        X = reg.o[0] + reg.u[ii]
        Y = reg.o[1] + reg.v[jj]
        self.assertTrue(len(ii) > 0)
        self.assertTrue(np.all(np.abs(Y - y) > 0.0445 - 1e-3))      # beside the blade, not on its track
        self.assertTrue(np.all(X > x + 0.203 - 0.03))                 # at its leading end

    def test_spill_stands_no_higher_than_its_roll(self):
        np, reg = self.np, self.reg
        cx, cy = self.P.PATCH_CENTRE
        x, y = cx - 0.1, cy + 0.25                               # on the slab north of the cut
        pr = reg.to_region(np.array([x, y, 0.0]))
        z = float(reg.h[int(pr[0]/reg.cell), int(pr[1]/reg.cell)])
        tb0 = reg.tb.copy()
        top = z + 0.010
        for _ in range(6):
            left = self.cp._spill(self.c, reg, np.array([x, y, z]), np.eye(3), 8e-6, top=top)
            self.assertGreater(left, 0.0)
        self.assertLessEqual(float(reg.h.max()), max(top, float((reg.b + tb0).max())) + 1e-9)
        reg.tb[:] = tb0


@unittest.skipUnless(SLOW, "builds the room (set GAR_SLOW_TESTS=1)")
class Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gym_anything_robotics.backends.newton_concrete_finishing import NewtonConcreteFinishingBackend
        cls.b = NewtonConcreteFinishingBackend(task="tool-sealant", seed=0, cameras=())

    @classmethod
    def tearDownClass(cls):
        cls.b.close()

    def test_evaluation_has_success_score_and_criteria(self):
        # GET /v1/evaluation and `gar verify` read these keys from the evaluator's result
        ev = self.b.evaluate()
        self.assertIn("success", ev)
        self.assertFalse(ev["success"])                                 # an untouched start state is not finished
        self.assertTrue(0.0 <= ev["score"] < 1.0)
        for name, c in ev["criteria"].items():
            for k in ("value", "threshold", "passed", "basis"):
                self.assertIn(k, c, name)

    def test_shortest_command(self):
        # what the HTTP capabilities give a policy as hardware.min_command_s: 0.05 s without cameras, the policy's
        # shortest command
        self.assertAlmostEqual(float(self.b.min_action_s), 0.05, places=6)

    def test_wait_refused_out_of_range_and_changes_nothing(self):
        from gym_anything_robotics.actions import RawAction
        from gym_anything_robotics.errors import ActionRejectedError
        w = self.b.env.world
        t0, clock0 = w.sim_time, self.b.env.material_clock()
        for bad in (3601.0, -5.0, "soon", math.nan):
            with self.assertRaises((ActionRejectedError, ValueError)):
                self.b.act(RawAction({"op": "wait", "duration_s": bad}))
        self.assertEqual(w.sim_time, t0)
        self.assertEqual(self.b.env.material_clock(), clock0)


if __name__ == "__main__":
    unittest.main()

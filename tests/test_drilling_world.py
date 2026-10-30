"""Drilling world mechanics.

One world build (bench preset, take-drill): a target step is followed within the FR3's joint acceleration and jerk
limits; a cartesian_wrench that helps the servo against an external twist leaves the wrist joint's total motor torque
within its 12 N m rating; a bit that breaks is shortened by the piece it loses, and a checkpoint restores it whole.
"""

from __future__ import annotations

import unittest

try:
    import newton  # noqa: F401
    import ruckig  # noqa: F401
    HAVE_ENGINE = True
except ImportError:
    HAVE_ENGINE = False


@unittest.skipUnless(HAVE_ENGINE, "Newton engine or ruckig not importable in this interpreter")
class DrillingWorldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from drilling.config import Config
        from drilling.env import DrillingEnv
        cls.env = DrillingEnv(Config(preset="bench", task="take-drill", seed=0))

    def test_target_step_within_acceleration_and_jerk(self) -> None:
        import numpy as np
        from drilling import params as P
        env, w = self.env, self.env.world
        g = w._arm["right"]
        names = g["names"]
        q0 = w.jq[g["q"]].copy()
        env.act({names[0]: float(q0[0] + 0.3), names[4]: float(q0[4] + 0.5)}, ticks=1)
        acc = []
        for _ in range(700):
            env.act({}, ticks=1)
            acc.append(np.abs(np.array(g["inp"].current_acceleration)))
        acc = np.array(acc)
        jerk = np.abs(np.diff(acc, axis=0)).max(0)/w.dt
        self.assertLessEqual(acc.max(), max(P.FR3_MAX_JOINT_ACC) + 1e-6)
        self.assertLessEqual(jerk.max(), max(P.FR3_MAX_JOINT_JERK)*(1 + 1e-6))
        self.assertGreater(acc.max(), 0.99*max(P.FR3_MAX_JOINT_ACC))            # the limit is reached, not avoided
        self.assertLess(abs(w.jq[g["q"][0]] - (q0[0] + 0.3)), 2e-3)
        env.act({names[0]: float(q0[0]), names[4]: float(q0[4])}, ticks=1500)

    def test_wrench_shares_the_joint_cap(self) -> None:
        env, w = self.env, self.env.world
        m, d = w.solver.mj_model, w.solver.mj_data
        mdofs = [dof for _, dof in w._arm_mj_dofs["right"]]
        hb = w._mj_to_newton[w._arm_hand["right"]]
        axis = d.xaxis[int(m.dof_jntid[mdofs[6]])].copy()

        def twist(world, wrench):
            wrench[hb, 3:] -= 20.*axis
            return wrench
        w.set_wrench("right", [0., 0., 0.], (10.*axis).tolist(), 300)
        w.fixture = twist
        peak = 0.
        try:
            for _ in range(300):
                env.act({}, ticks=1)
                peak = max(peak, abs(float(d.qfrc_actuator[mdofs[6]] + d.qfrc_applied[mdofs[6]])))
        finally:
            w.fixture = None
        self.assertLessEqual(peak, 12.0 + 1e-6)
        self.assertGreater(peak, 11.9)                   # the servo and the feed-forward together reach the rating

    def test_broken_bit_is_shortened_and_restored(self) -> None:
        import numpy as np
        env, w = self.env, self.env.world
        it = w.scene.tools["drill"].item
        m = w.solver.mj_model
        snap = env.checkpoint()
        tip0, length0 = it.tip.copy(), it.length
        size0, contype0 = m.geom_size.copy(), m.geom_contype.copy()
        w.hm._break(it, None, None, "test")
        s = 2.5e-3*it.d_mm
        self.assertTrue(it.state["broken"])
        self.assertAlmostEqual(float(np.linalg.norm(tip0 - it.tip)), s, places=9)
        self.assertAlmostEqual(length0 - it.length, s, places=9)
        self.assertLess(m.geom_size[:, 1].sum(), size0[:, 1].sum())          # the body cylinder is shorter
        self.assertLess(int(m.geom_contype.sum()), int(contype0.sum()))      # the point no longer collides
        env.act({}, ticks=50)
        self.assertIsNone(w.fault)
        env.restore(snap)
        self.assertFalse(it.state["broken"])
        self.assertTrue(np.array_equal(it.tip, tip0))
        self.assertTrue(np.array_equal(m.geom_size, size0) and np.array_equal(m.geom_contype, contype0))

    def test_solver_settings_come_from_the_ledger(self) -> None:
        """The line search that finishes the Newton solve and the convex-collision settings are the parameter
        ledger's, in the built model."""
        from drilling import params as P
        opt = self.env.world.solver.mj_model.opt
        self.assertEqual(int(opt.ls_iterations), P.SOLVER_LS_ITERATIONS)
        self.assertAlmostEqual(float(opt.ls_tolerance), P.SOLVER_LS_TOLERANCE, places=12)
        self.assertEqual(int(opt.ccd_iterations), P.SOLVER_CCD_ITERATIONS)
        self.assertAlmostEqual(float(opt.ccd_tolerance), P.SOLVER_CCD_TOLERANCE, places=12)
        self.assertGreaterEqual(P.SOLVER_LS_ITERATIONS, 200)

    def test_a_bar_against_a_stopped_bit_gets_no_hole(self) -> None:
        """A part carried against the drill press's stopped bit, its top face 0.9 mm above the tip, is met by a
        contact; it does not receive a hole record."""
        import numpy as np
        from scipy.spatial.transform import Rotation
        from drilling.config import Config
        from drilling.env import DrillingEnv
        env = DrillingEnv(Config(preset="bench", task="press-drill", seed=0))
        w = env.world
        it = w.hm.items["drill_press_bit"]
        q = w.bq[it.body]
        tip = q[:3] + Rotation.from_quat(q[3:]).apply(it.tip)
        bar = w.scene.parts["flat_bar"]["body"]
        w.place_free(bar, tip + np.array([0., 0., 0.0009 - 0.003]), np.array([0., 0., 0., 1.]))
        env.act({}, ticks=40)
        holes = [h for h in w.hm.drillables["flat_bar"].holes if h.kind == "drilled"]
        self.assertEqual(holes, [])


if __name__ == "__main__":
    unittest.main()

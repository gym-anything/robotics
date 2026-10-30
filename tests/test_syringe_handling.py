"""Syringe handling: discovery, robot setups, the task catalogue, the layouts of every seed, the evaluator's tolerance
table, and (one built scene, about 20 s on the processor) the start state, the engine corrections and a checkpoint.

  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_syringe_handling.py
"""
import json
import math
import unittest
from pathlib import Path

import numpy as np
from tests.packages import PACKAGES

ROOT = PACKAGES


class Discovery(unittest.TestCase):
    def test_launch_and_robots(self):
        from gym_anything_robotics.launch import describe
        from robots.catalog import environment_robots, for_environment
        d = describe("syringe-handling")
        self.assertEqual(d["default_robot"], "dual-fr3")
        self.assertEqual({r["id"] for r in environment_robots("syringe-handling")}, {"dual-fr3", "dual-panda", "single-fr3", "g1"})
        self.assertEqual(for_environment("syringe-handling").id, "syringe-fr3")
        self.assertEqual(for_environment("syringe-handling", "g1").id, "syringe-g1")
        from syringe_handling.robot_setup import validate, check_task
        self.assertEqual(validate(for_environment("syringe-handling", "g1").to_dict()), "g1")
        check_task("g1", "sharps-away")
        with self.assertRaises(ValueError):
            check_task("g1", "set-dose")

    def test_single_arm_is_admitted_for_sharps_away_only(self):
        from syringe_handling.robot_setup import tasks_for
        self.assertEqual(tuple(tasks_for("single-fr3")), ("sharps-away",))
        self.assertIsNone(tasks_for("dual-fr3"))                                    # None: every task


class Trace(unittest.TestCase):
    def test_world_events_and_stops_are_filed_under_their_tick_and_type(self):
        # A world event is dict(t=..., event=...) (devices.py log_event) and the store reads event["tick"]. Passed on as
        # it is, the event raises KeyError after physics has advanced, so every command in which a Luer fit or a touch
        # is logged would stop part-way.
        import tempfile
        from types import SimpleNamespace
        from syringe_handling.trace import Trace
        w = SimpleNamespace(tick=43000, reflexes=[], events=[], model=SimpleNamespace(body_label=["a", "b"]),
                            scene=SimpleNamespace(bodies={"a": 0, "b": 1}), bq=np.zeros((2, 7)), bqd=np.zeros((2, 6)),
                            jq=np.zeros(3), jqd=np.zeros(3), tq=np.zeros(3), grip_force={}, grip_applied_n=lambda k: 0.)
        env = SimpleNamespace(world=w, config=SimpleNamespace(rigid_dt=0.001), trace_callback=None,
                              equipment_state=lambda: {"syringes": {}})
        with tempfile.TemporaryDirectory() as tmp:
            trace = Trace(env, Path(tmp)/"physics.sqlite3")
            try:
                trace.begin(dict(action_id="a1", request_id="r1"))
                for event in (dict(t=43.001, event="luer_captured", syringe="flush", fitting="port"),
                              dict(t=43.001, event="touch", part="flush", place="tray")):
                    w.events.append(event)
                w.reflexes.append(dict(tick=43001, arm="left", kind="collision", joint="fr3_joint4"))
                w.tick = 43001
                env.trace_callback(env)                                 # Trace replaced the env's per-tick callback
                w.tick = 43002
                env.trace_callback(env)
                summary = trace.finish()
                page = trace.store.page("a1", kind="event")
            finally:
                trace.close()
        self.assertEqual(summary["retained_samples"], 2)
        self.assertEqual([(e["tick"], e["data"]["type"]) for e in page["items"]],
                         [(43001, "protective_stop"), (43001, "luer_captured"), (43001, "touch")])
        self.assertEqual(page["items"][1]["data"]["fitting"], "port")          # the event's own fields are kept
        self.assertEqual(page["items"][1]["data"]["t"], 43.001)


class OffersAndForceLaw(unittest.TestCase):
    """The catalogue's offers, the policy camera's view of the wall clock and the force law's closed form, all without
    building a world."""

    def test_two_pandas_are_offered_for_sharps_away_only(self):
        from syringe_handling.robot_setup import tasks_for, check_task
        self.assertEqual(tuple(tasks_for("dual-panda")), ("sharps-away",))
        check_task("dual-panda", "injection", "sharps-away")
        with self.assertRaises(ValueError):
            check_task("dual-panda", "injection", "set-dose")

    def test_a_policy_camera_sees_the_whole_wall_clock(self):
        # slow-push's instruction times the push "by the wall clock": the ceiling camera bay_rgb is offered to a policy
        # and the clock's face lies inside its image, at least 80 px across (a pinhole of its pose; fov is horizontal)
        from syringe_handling import params as P
        from gym_anything_robotics.backends.newton_syringe_handling import POLICY_CAMERAS
        self.assertIn("bay_rgb", POLICY_CAMERAS)
        bc, (kx, kw, kh, kd, kz) = P.BAY_CAMERA, P.WALL_FITTINGS["clock"]
        eye, tgt = np.asarray(bc["eye"], float), np.asarray(bc["target"], float)
        f_ = (tgt - eye)/np.linalg.norm(tgt - eye)
        r_ = np.cross(f_, (0., 0., 1.))
        r_ /= np.linalg.norm(r_)
        u_ = np.cross(r_, f_)
        W, H = 1280, 720
        fpx = 0.5*W/math.tan(math.radians(0.5*bc["fov"]))
        centre = np.array([kx, P.ROOM["y1"] - kd, kz + 0.5*kh])
        pts = [centre + 0.5*kw*np.array([math.cos(a), 0., math.sin(a)]) for a in np.linspace(0, 2*math.pi, 16, endpoint=False)]
        px = [(0.5*W + fpx*np.dot(p - eye, r_)/np.dot(p - eye, f_), 0.5*H - fpx*np.dot(p - eye, u_)/np.dot(p - eye, f_)) for p in pts]
        self.assertTrue(all(5 < x < W - 5 and 5 < y < H - 5 for x, y in px), px)
        self.assertGreater(max(x for x, _ in px) - min(x for x, _ in px), 80.0)

    def test_poiseuille_is_hagen_poiseuille(self):
        from syringe_handling import hydraulics as H
        for mu, L, d in ((1.0e-3, 0.0254, 0.26e-3), (11.6e-3, 0.0381, 0.838e-3)):
            self.assertAlmostEqual(H.poiseuille(mu, L, d)/(128*mu*L/(math.pi*d**4)), 1.0, places=12)


class Catalogue(unittest.TestCase):
    def test_benchmark_counts_and_selectors(self):
        b = json.loads((ROOT/"syringe_handling/benchmark.json").read_text())
        from syringe_handling.config import TASKS, VARIANT_TASKS
        pairs = sum(len(t) for t in TASKS.values()) + sum(len(t) for t in VARIANT_TASKS.values())
        self.assertEqual(b["count"], pairs)
        self.assertEqual(pairs, 25)
        self.assertEqual(b["cli_task_selector"], "task")
        self.assertEqual(set(b["selector_choices"]["task"]), {t for ts in TASKS.values() for t in ts})

    def test_a_variant_belongs_to_its_tasks(self):
        from syringe_handling.config import Config
        Config(task="inject", variant="subcut")
        with self.assertRaises(ValueError):
            Config(task="set-dose", variant="subcut")
        with self.assertRaises(ValueError):
            Config(task="set-dose", variant="no-such")

    def test_every_fill_fits_its_barrel_over_the_seeds(self):
        from syringe_handling import params as P
        from syringe_handling.config import Config, TASKS, VARIANT_TASKS
        from syringe_handling.scene import task_layout
        pairs = [(t, "") for ts in TASKS.values() for t in ts] + [(t, v) for v, ts in VARIANT_TASKS.items() for t in ts]
        for task, variant in pairs:
            for seed in range(120):
                for (name, kind, liq, ml, air, fit, slot) in task_layout(Config(task=task, variant=variant, seed=seed))["syringes"]:
                    sp = P.SYRINGES[kind]
                    x0 = max(((ml + air) if liq else 0.0) - sp["dead_ml"], 0.0)*1e-6/(math.pi*sp["bore"]**2/4)
                    self.assertLess(x0, sp["stroke_max"] - 0.0005, f"{task}/{variant} seed {seed}: {name}")

    def test_a_filled_syringe_holds_its_dose_above_the_dead_space(self):
        from syringe_handling import params as P
        from syringe_handling.config import Config
        from syringe_handling.scene import task_layout
        L = task_layout(Config(task="inject"))
        (name, kind, liq, ml, air, fit, slot), = L["syringes"]
        self.assertAlmostEqual(ml - P.SYRINGES[kind]["dead_ml"], L["dose_ml"], places=6)


class Evaluator(unittest.TestCase):
    def test_dose_tolerance_is_the_standards_table(self):
        from syringe_handling.task import dose_tolerance_ml
        self.assertAlmostEqual(dose_tolerance_ml(2.5, 3.0), 0.125)                 # at or above half the capacity: 5 %
        self.assertAlmostEqual(dose_tolerance_ml(0.5, 3.0), 0.015*3.0 + 0.02*0.5)   # below it: 1.5 % of capacity + 2 %

    def test_instructions_name_their_numbers(self):
        from syringe_handling.task import INSTRUCTIONS
        from syringe_handling.config import TASKS
        self.assertEqual(set(INSTRUCTIONS), {t for ts in TASKS.values() for t in ts})
        self.assertIn("{dose_ml", INSTRUCTIONS["set-dose"])


class BuiltScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from syringe_handling.config import Config
        from syringe_handling.env import SyringeEnv
        cls.env = SyringeEnv(Config(task="draw-dose", robot="none"))

    def test_start_state_does_not_pass_and_breaks_no_rule(self):
        from syringe_handling.task import RESULTS
        ev = self.env.evaluate()
        self.assertFalse(ev["success"])
        self.assertEqual(ev["score"], 0.0)
        self.assertEqual([k for k, c in ev["criteria"].items() if not c["ok"] and k not in RESULTS], [])

    def test_solver_tolerance_and_subtree_masses(self):
        from syringe_handling import params as P
        w = self.env.world
        m = w.solver.mj_model
        self.assertLessEqual(float(m.opt.tolerance), P.SOLVER_TOLERANCE*1.01)
        for s in w.fluid.syringes.values():
            b = s.mj_barrel
            sub = [k for k in range(m.nbody) if k == b or self._under(m, k, b)]
            self.assertAlmostEqual(float(m.body_subtreemass[b]), float(sum(m.body_mass[k] for k in sub)), places=9)

    @staticmethod
    def _under(m, k, root):
        while k > 0:
            k = int(m.body_parentid[k])
            if k == root:
                return True
        return False

    def test_an_armature_rewrite_keeps_track_and_fit_anchors(self):
        import mujoco
        from syringe_handling.env import SyringeEnv
        from syringe_handling.config import Config
        # This deliberately corrupts anchors: do not pollute the class scene
        # used by the independent checkpoint test. Zero is the normal armature.
        w = SyringeEnv(Config(task="draw-dose", robot="none")).world
        m = w.solver.mj_model
        s = next(iter(w.fluid.syringes.values()))
        connects = [e for e in range(m.neq) if m.eq_type[e] in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD)]
        self.assertTrue(connects)
        for e in connects:
            m.eq_data[e, 3:6] = (0.011, 0.012, 0.013)
        w._arm_pending[s.d_pl] = float(m.dof_armature[s.d_pl]) + .002
        w._apply_const()
        for e in connects:
            np.testing.assert_allclose(m.eq_data[e, 3:6], (0.011, 0.012, 0.013))


    def test_zz_checkpoint_restores_bit_for_bit(self):
        env = self.env
        d = env.world.solver.mj_data
        env.run(100)
        snap = env.checkpoint()
        env.run(100)
        a = (d.qpos.copy(), d.qvel.copy())
        env.restore(snap)
        env.run(100)
        np.testing.assert_array_equal(a[0], d.qpos)
        np.testing.assert_array_equal(a[1], d.qvel)
        self.assertTrue(np.isfinite(d.qpos).all())


if __name__ == "__main__":
    unittest.main()

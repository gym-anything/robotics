"""The syringe_handling task set. No simulator: real catalogue/evaluator/ledger with device and rigid-state stubs.

Run: python -m pytest tests/environments/syringe_handling/test_task_set.py
"""
import json
import math
import unittest
from types import SimpleNamespace as NS

import numpy as np

from syringe_handling import hydraulics as H, params as P, task as T, variants as V
from syringe_handling.config import Config, EIGHT_TASK_CONTRACT as NEW, VIAL_TO_PAD_CONTRACT, LEGACY_CONTRACT
from syringe_handling.scene import task_layout
from tests.packages import PACKAGES

ROOT = PACKAGES / "syringe_handling"


def fake(task, variant="", seed=0, contract=None):
    c = Config(task=task, variant=variant, seed=seed,
               contract=contract or (LEGACY_CONTRACT if task == "vial-to-pad" else NEW))
    layout = task_layout(c)
    bodies, poses = {}, []

    def body(name, p=(-0.03, 0.16, P.COUNTER_Z + 0.02)):
        bodies[name] = len(poses)
        poses.append([*p, 0., 0., 0., 1.])
        return bodies[name]

    syringes, needles = {}, {}

    def needle(name):
        b = body(name)
        n = NS(body=b, records=[], touch={}, resheathed=[], track=None, flags={"bent": False}, sheathed=True)
        n.tip_world = lambda w: (w.bq[b, :3] - [0.04, 0, 0], np.eye(3))
        needles[name] = n
        return n

    for name, kind, liquid, ml, air, fitting, slot in layout["syringes"]:
        barrel = body(name)
        chamber = NS(liquid=ml*H.ML, dead=P.SYRINGES[kind]["dead_ml"]*H.ML, gas0=air*H.ML)
        s = NS(s=P.SYRINGES[kind], chamber=chamber, barrel=barrel, touch={}, mate=None,
               flags={"extracted": False, "overridden": False})
        syringes[name] = s
        if fitting in P.NEEDLES:
            needle(name + "_needle")
        body(name + "_sheath")  # distractor; no simulator objects are constructed
    if task == "vial-to-pad":
        needle("needle_im")
    vials = {}
    if "vial_ml" in layout:
        b = body("vial", (*P.LAYOUT["vial"], P.COUNTER_Z))
        v = NS(body=b, female=None, site=NS(name="vial_stopper"), cos=1.0,
               ml=layout["vial_ml"], gauge=layout["vial_gauge_kpa"]*1000)
        v.vessel = NS(liquid_volume=v.ml*H.ML, gas_pressure=lambda: P.P_ATM + v.gauge)
        v.reading = lambda w: {"liquid_ml": v.ml, "upright_cos": v.cos}
        vials["vial"] = v
    locks = {}
    if layout["trainer"] == "lock":
        b = body("lock", (*P.LAYOUT["work"], P.COUNTER_Z + 0.04))
        lk = NS(body=b, blocked=layout["blocked"], max_shift=0.0, peak_gauge=0.0,
                female=NS(mate=None), clamp_mm=8.0 if layout["clamp_closed"] else 0.0)
        lk.reading = lambda w: dict(clamp_open=lk.clamp_mm < 4, clamp_mm=lk.clamp_mm,
                                    mate=None if lk.female.mate is None else "syringe")
        locks["lock"] = lk
    fl = NS(syringes=syringes, needles=needles, vials=vials, locks=locks, ledger=H.Ledger(),
            females={}, caps={}, penetrables=[], conservation_error_ml=lambda: 0.,
            spray=NS(hanging=lambda: 0., airborne=lambda: 0.))
    sc = NS(bodies=bodies, _robot_set=set(), sharps_cavity={"lo": (0.37, 0.395, 0.92), "hi": (0.63, 0.485, 1.17)},
            flap_joint=NS(q=0), tray_floor=P.COUNTER_Z + 0.004)
    w = NS(fluid=fl, scene=sc, bq=np.array(poses), jq=np.array([0.]), sim_time=0.0, reflexes=[], events=[])
    w.body_pose = lambda b: (w.bq[b, :3], np.eye(3))
    start = dict(touch={}, vial_ml={n: v.ml for n, v in vials.items()},
                 vial_xy={n: list(w.bq[v.body, :2]) for n, v in vials.items()},
                 syringe_ml={n: s.chamber.liquid/H.ML for n, s in syringes.items()},
                 syringe_dose_ml={n: max(s.chamber.liquid - s.chamber.dead, 0.0)/H.ML for n, s in syringes.items()},
                 syringe_air_ml={n: s.chamber.gas0/H.ML for n, s in syringes.items()})
    return NS(config=c, spec=T.task_spec(c, layout), layout=layout, world=w, start=start, _history=NS(last_move=0.))


def arrival(env, place, species, ml, at):
    if place == "lock:in":
        env.world.fluid.ledger.note(place, H.parcel(ml*H.ML, species), t=at)
    else:
        env.world.fluid.ledger.add(place, H.parcel(ml*H.ML, species), t=at)


def dispose(env, names=None):
    names = names or [*env.world.fluid.syringes, *env.world.fluid.needles]
    for n in names:
        env.world.bq[env.world.scene.bodies[n], :3] = (0.49, 0.44, 1.05)


def puncture(needle=None):
    return dict(target="pad", t=1.0, t_out=25.0, offset_mm=0., angle_deg=0., exit_angle_deg=0.,
                lever_deg=0., deepest_mm=20.)


def finish(env):
    task, fl, spec = env.spec["task"], env.world.fluid, env.spec
    if task == "sharps-away":
        dispose(env)
    elif task in ("set-dose", "draw-dose"):
        s = fl.syringes[spec["dose_syringe"]]
        s.chamber.liquid = spec["dose_ml"]*H.ML + s.chamber.dead
        s.chamber.gas0 = 0.0
        if fl.vials:
            fl.vials["vial"].ml -= spec["dose_ml"]
    elif task in ("inject", "vial-to-pad"):
        name = "needle_im" if task == "vial-to-pad" else spec["dose_syringe"] + "_needle"
        fl.needles[name].records = [puncture()]
        for t in range(2, 22):
            arrival(env, "pad:" + spec["layer"], "medication", spec["dose_ml"]/20, t)
        if task == "vial-to-pad":
            fl.needles["syringe_needle"].records = [dict(target="vial_stopper", t=0., t_out=0.5)]
            env.world.events = [dict(t=.8, event="luer_captured", syringe="syringe", fitting="needle_im")]
        dispose(env)
    else:
        for t in range(10):
            arrival(env, "lock:in", "saline", spec["flush_ml"]/10, 200 + t)
        if task != "flush-lock":
            sp = "medication" if task == "slow-push" else "contrast"
            for t in range(100):
                at = t*2 if task == "slow-push" else t*.06
                arrival(env, "vein", sp, spec["dose_ml"]/100, at)
                arrival(env, "lock:in", sp, spec["dose_ml"]/100, at)


class Catalogue(unittest.TestCase):
    def test_exact_set_and_roundtrip(self):
        data = json.loads((ROOT/"benchmark.json").read_text())
        for k, value in V.catalogue_fields().items():
            self.assertEqual(data[k], value, k)
        rows = data["task_set"]["selections"]
        self.assertEqual(len(rows), 8)
        self.assertEqual([r["task"] for r in rows], list(V.EIGHT_TASKS))
        for row in rows:
            resolved = V.task_resolve(row)
            cfg = Config(**resolved["backend_options"])
            self.assertEqual(cfg.task, row["task"])
            self.assertEqual(cfg.contract, row["contract"])
            self.assertEqual(resolved["description"], T.instruction(cfg))
            self.assertEqual(V.task_resolve(resolved["selection"]), resolved)

    def test_defaults_ids_instructions_and_layouts(self):
        row = V.task_resolve({})
        self.assertEqual(row["task_id"], "syringe-handling/v1/injection/set-dose/seed-0")
        self.assertEqual(row["backend_options"], dict(preset="injection", task="set-dose", seed=0))
        for task in V.EIGHT_TASKS[:-1]:
            old = V.task_resolve({"task": task})
            new = V.task_resolve({"task": task, "contract": NEW})
            self.assertNotEqual(new["task_id"], old["task_id"])
            self.assertEqual(new["split"], old["split"])
            self.assertEqual(task_layout(Config(task=task)), task_layout(Config(task=task, contract=NEW)))
            self.assertNotIn("Scoring", new["description"])
            self.assertNotIn("measured entering", new["description"])
        self.assertIn("Scoring tolerance", T.instruction(Config(task="inject")))

    def test_bad_selections_fail(self):
        for row in ({"contract": "typo"}, {"task": "vial-to-pad", "contract": NEW}, {"contracts": NEW},
                    {"task": "inject", "variant": "blocked"}, {"seed": True}, {"preset": "iv-lock", "task": "set-dose"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                V.task_resolve(row)

    def test_backend_reset_and_selection_without_build(self):
        from gym_anything_robotics.backends.newton_syringe_handling import NewtonSyringeHandlingBackend as Backend
        b = Backend.__new__(Backend)
        b.config = Config(task="inject", contract=NEW)
        self.assertEqual(b._reset_selection(5, {})["contract"], NEW)
        self.assertEqual(b._reset_selection(None, {"contract": None})["contract"], NEW)
        self.assertEqual(b.select_variation({"task": "inject", "contract": None})["variation"]["contract"], "v1")
        self.assertEqual(b._reset_selection(None, {"contract": "v1"})["contract"], "v1")
        self.assertEqual(b._reset_selection(None, {"task": "vial-to-pad"})["contract"], "v1")
        row = b.select_variation({"task": "flush-lock", "contract": NEW})
        self.assertEqual(row["reset_request"]["options"]["contract"], NEW)
        for options in ({"contract": "typo"}, {"task": "vial-to-pad", "contract": NEW}):
            with self.assertRaises(ValueError):
                b._reset_selection(None, options)


class Rewards(unittest.TestCase):
    def test_every_start_and_variant_is_zero(self):
        for _, task, variant in V.pairs():
            for seed in (0, 1, 11, 9999):
                with self.subTest(task=task, variant=variant, seed=seed):
                    e = T.evaluate(fake(task, variant, seed))
                    self.assertFalse(e["success"])
                    self.assertEqual(e["score"], 0.)
                    self.assertEqual(e["contract"], VIAL_TO_PAD_CONTRACT if task == "vial-to-pad" else NEW)

    def test_every_complete_state_is_one_and_failed_state_below_one(self):
        for task in V.EIGHT_TASKS:
            with self.subTest(task=task):
                env = fake(task)
                finish(env)
                e = T.evaluate(env)
                self.assertTrue(e["success"], e)
                self.assertEqual(e["score"], 1.)
                env.world.reflexes = ["physical reflex"]
                e = T.evaluate(env)
                self.assertFalse(e["success"])
                self.assertGreater(e["score"], 0.)
                self.assertLessEqual(e["score"], .9)

    def test_set_dose_continuity_empty_regression_and_start_baseline(self):
        env = fake("set-dose")
        s = env.world.fluid.syringes["syringe"]
        scores = []
        for volume, gas in ((2.8, .3), (2.75, .25), (2.7, .20), (2.65, .15)):
            s.chamber.liquid, s.chamber.gas0 = (volume + .05)*H.ML, gas*H.ML
            scores.append(T.evaluate(env)["score"])
        self.assertEqual(scores, sorted(set(scores)))
        s.chamber.liquid = s.chamber.gas0 = 0.
        self.assertEqual(T.evaluate(env)["score"], 0.)
        for volume in (2.62499, 2.62501):
            s.chamber.liquid, s.chamber.gas0 = (volume + .05)*H.ML, .1*H.ML
            scores.append(T.evaluate(env)["score"])
        self.assertLess(abs(scores[-1] - scores[-2]), .0001)

    def test_draw_volume_not_boolean_and_no_overdraw_reward(self):
        env = fake("draw-dose")
        s = env.world.fluid.syringes["syringe"]
        scores = []
        for v in (0, .1, .2, .5, 1., 2.):
            s.chamber.liquid = (v + .05)*H.ML
            scores.append(T.evaluate(env)["score"])
        self.assertEqual(scores, sorted(set(scores)))
        finish(env)
        s.chamber.liquid = 5.0*H.ML
        self.assertFalse(T.evaluate(env)["success"])
        self.assertLess(T.evaluate(env)["score"], scores[-1])

    def test_sharps_distance_and_flap_are_continuous(self):
        env = fake("sharps-away")
        scores = []
        for x in (.30, .32, .34, .36):
            for b in env.world.scene.bodies.values():
                env.world.bq[b, :3] = (-.03, .16, .94)
            for n in (*env.world.fluid.syringes, *env.world.fluid.needles):
                env.world.bq[env.world.scene.bodies[n], :3] = (x, .44, 1.05)
            scores.append(T.evaluate(env)["score"])
        self.assertEqual(scores, sorted(set(scores)))
        dispose(env)
        env.world.jq[0] = math.radians(60.)
        a = T.evaluate(env)["score"]
        env.world.jq[0] = math.radians(59.9)
        self.assertGreater(T.evaluate(env)["score"], a)
        b = env.world.scene.bodies["syringe_a_sheath"]
        env.world.bq[b, :3] = (.49, .44, 1.05)
        self.assertLess(T.evaluate(env)["score"], a)

    def test_inject_unused_disposal_wrong_layer_and_delivered_site(self):
        env = fake("inject")
        dispose(env)
        self.assertEqual(T.evaluate(env)["score"], 0.)
        fl = env.world.fluid
        arrival(env, "pad:muscle", "medication", .2, 5.)
        self.assertEqual(T.evaluate(env)["score"], 0.)  # a later dry stick cannot explain earlier liquid
        r = puncture()
        fl.needles["syringe_needle"].records = [r]
        good = T.evaluate(env)["score"]
        self.assertGreater(good, 0.)
        r["offset_mm"] = 15.
        self.assertLess(T.evaluate(env)["score"], good)
        env = fake("inject")
        arrival(env, "pad:fat", "medication", .5, 5.)
        env.world.fluid.needles["syringe_needle"].records = [puncture()]
        self.assertEqual(T.evaluate(env)["score"], 0.)
        env = fake("inject", "subcut")
        finish(env)
        env.world.fluid.needles["syringe_needle"].records[0]["angle_deg"] = 40.
        self.assertTrue(T.evaluate(env)["success"])

    def test_flush_counts_entry_not_initial_saline_and_grades_cleanup(self):
        env = fake("flush-lock")
        arrival(env, "vein", "saline", 1.1, 1.)
        self.assertEqual(T.evaluate(env)["score"], 0.)
        scores = []
        for i in range(4):
            arrival(env, "lock:in", "saline", .25, 2. + i)
            scores.append(T.evaluate(env)["score"])
        self.assertEqual(scores, sorted(set(scores)))
        lk = env.world.fluid.locks["lock"]
        lk.clamp_mm = 2.
        self.assertLess(T.evaluate(env)["score"], scores[-1])
        lk.clamp_mm = 0.
        lk.peak_gauge = 600e3
        self.assertEqual(T.evaluate(env)["score"], 0.)

    def test_blocked_line_requires_try_and_penalizes_forcing(self):
        env = fake("flush-lock", "blocked")
        lk = env.world.fluid.locks["lock"]
        scores = []
        for p in (0., 100., 200., 500.):
            lk.peak_gauge = p
            scores.append(T.evaluate(env)["score"])
        self.assertEqual(scores, sorted(set(scores)))
        lk.peak_gauge = 2000.
        self.assertEqual(T.evaluate(env)["score"], 1.)
        lk.peak_gauge = 200e3
        self.assertEqual(T.evaluate(env)["score"], 0.)

    def test_iv_timing_air_and_saline_only(self):
        for task in ("fast-bolus", "slow-push"):
            env = fake(task)
            arrival(env, "lock:in", "saline", 5., 1.)
            self.assertEqual(T.evaluate(env)["score"], 0.)
            env = fake(task)
            finish(env)
            led = env.world.fluid.ledger
            place = "lock:in" if task == "fast-bolus" else "vein"
            factor = 3. if task == "fast-bolus" else .1
            led.arrivals[place] = [(t*factor, m, k) for t, m, k in led.arrivals[place]]
            e = T.evaluate(env)
            self.assertFalse(e["success"])
            self.assertLess(e["score"], .9)
            arrival(env, "vein", "air", .5, 999.)
            self.assertEqual(T.evaluate(env)["score"], 0.)

    def test_evaluation_is_read_only_and_strategy_independent(self):
        env = fake("draw-dose")
        finish(env)
        env.world.reflexes = ["stop"]
        original = T.evaluate(env)
        env.world.command_history = ["left", "right", "arbitrary route"]
        env.world.fluid.syringes = dict(reversed(list(env.world.fluid.syringes.items())))
        self.assertEqual(T.evaluate(env), original)
        self.assertEqual(T.evaluate(env), original)
        json.dumps(original, allow_nan=False)

    def test_legacy_discrete_reward_and_vial_to_pad_score_parts(self):
        env = fake("set-dose", contract="v1")
        env.world.fluid.syringes["syringe"].chamber.gas0 = 0.
        e = T.evaluate(env)
        self.assertEqual(e["score"], .5)
        self.assertEqual(e["contract"], "v1")
        env = fake("vial-to-pad")
        self.assertEqual(T.evaluate(env)["score_parts"]["partial"], 0.)
        finish(env)
        e = T.evaluate(env)
        self.assertEqual(e["score_parts"]["partial"], .9)
        self.assertEqual(e["score"], 1.)

    def test_start_record_captures_settled_air_and_expellable_volume(self):
        env = fake("set-dose")
        s = env.world.fluid.syringes["syringe"]
        s.chamber.liquid, s.chamber.gas0 = 2.79*H.ML, .217*H.ML
        env.start = T.start_record(env.world)
        self.assertEqual(env.start["syringe_air_ml"]["syringe"], .217)
        self.assertEqual(T.evaluate(env)["score"], 0.)
        s.chamber.gas0 *= .9
        self.assertGreater(T.evaluate(env)["score"], 0.)

    def test_attached_lock_cleanup_and_strict_failure_cap(self):
        env = fake("flush-lock")
        finish(env)
        lk = env.world.fluid.locks["lock"]
        depth = [P.LUER["d_seat"]]
        lk.female.mate = NS(depth=lambda w: depth[0])
        scores = []
        for fraction in (1., .75, .5, .25, 0.):
            depth[0] = fraction*P.LUER["d_seat"]
            e = T.evaluate(env)
            self.assertFalse(e["success"])
            self.assertLessEqual(e["score"], .9)
            scores.append(e["score"])
        self.assertEqual(scores, sorted(set(scores)))


if __name__ == "__main__":
    unittest.main()

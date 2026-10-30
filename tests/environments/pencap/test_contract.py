"""Small non-simulation checks for input handling and coordinate conventions."""

import json
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from pencap.newton15.config import Config
from pencap.newton15.task import DESCRIPTION, Evaluator
from pencap.newton15.state_validation import validate_state
from tests.packages import PACKAGES


class Array:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)
    def numpy(self):
        return self.value


class ContractTests(unittest.TestCase):
    def test_finite_runaway_is_rejected_without_mutating_state(self):
        model = SimpleNamespace(body_label=["notebook"])
        state = SimpleNamespace(body_q=Array([[0, 0, .758, 0, 0, 0, 1]]),
            body_qd=Array([[0, 0, 0, 0, 0, 0]]), joint_q=Array([0]), joint_qd=Array([0]))
        validate_state(model, state, 1)
        state.body_qd.value[0, 3] = 1e7
        original = state.body_qd.numpy().copy()
        with self.assertRaisesRegex(FloatingPointError, "notebook.*attempted tick 2.*angular_speed"):
            validate_state(model, state, 2)
        np.testing.assert_array_equal(state.body_qd.numpy(), original)
        state.body_q.value[0, 0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "Non-finite body_q for notebook"):
            validate_state(model, state, 3)

    def test_defaults_preserve_object_dimensions(self):
        c = Config()
        self.assertEqual((c.pen_length, c.cap_length, c.nub_count), (.140, .058, 4))
        self.assertAlmostEqual(c.shoulder_u, .04027)

    def test_invalid_physics_configuration_rejected(self):
        for options in ({"dt": float("nan")}, {"pen_mass": -1}, {"nub_count": 3.5},
                        {"scene": "missing"}, {"plastic_friction": float("inf")}, {"lead_deg": 90}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options)


def pair(u, tilt_deg=0., axis_deg=0., lateral=(0., 0.), world=Rotation.identity(), at=(0., 0., .76)):
    """Pen and cap poses (xyzw) with the cap mouth at pen-frame (lateral[0], u, lateral[1]), axes opposed,
    then tilted about an axis in the pen x-z plane; the pair is placed at `at` with orientation `world`."""
    a = np.radians(axis_deg)
    rel = Rotation.from_rotvec(np.radians(tilt_deg)*np.array([np.cos(a), 0., np.sin(a)]))*Rotation.from_rotvec([0., 0., np.pi])
    at = np.asarray(at, float)
    return (np.r_[at, world.as_quat()],
            np.r_[at+world.apply([lateral[0], u, lateral[1]]), (world*rel).as_quat()])


class EvaluatorTests(unittest.TestCase):
    """Pose-level evaluator and reward cases. No physics."""

    def setUp(self):
        self.c = Config()
        self.need = round(.5/self.c.dt)
        self.e = SimpleNamespace(config=self.c, bodies={"pen": 0, "cap": 1},
            state=SimpleNamespace(body_q=Array(np.zeros((2, 7))), body_qd=Array(np.zeros((2, 6)))))

    def run_poses(self, frames, evaluator=None):
        evaluator = evaluator or Evaluator()
        for pen, cap in frames:
            self.e.state.body_q = Array([pen, cap])
            result = evaluator.sample(self.e)
        return result

    def hold(self, poses, seconds=.6, evaluator=None):
        return self.run_poses([poses]*round(seconds/self.c.dt), evaluator)

    def test_catalogue_and_policy_serve_one_job_description(self):
        catalogue = json.loads((PACKAGES / "pencap" / "newton15" / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(catalogue["description"], DESCRIPTION)
        for internal in ("nub", "bead", "shoulder", "Pick up", "Panda", "mm", "0.5 s", "contact"):
            self.assertNotIn(internal, DESCRIPTION)
        self.assertIn("writing tip", DESCRIPTION)

    def test_seated_cap_succeeds_whatever_the_route(self):
        aligned = self.run_poses([pair(u) for u in np.linspace(.030, .040, 100)]+[pair(.040)]*self.need)
        self.assertTrue(aligned["success"])
        self.assertEqual(aligned["score"], 1.)
        cocked = [pair(u, 3.) for u in np.linspace(.030, .0402, 200)]+[pair(.0402, t) for t in np.linspace(3., 0., 50)]
        self.assertTrue(self.run_poses(cocked+[pair(.0402)]*self.need)["success"])
        # Return-face hold; nubs on the crest's trailing corner, past its edge (retained); a bench-rest pose; pair
        # held vertical; feasible fully snapped caps cocked 2.75 and 2.5 deg inside their play (all nubs past the
        # edge); pressed past the collar face by contact compliance.
        for case in (pair(.0398), pair(.0397), pair(.039945, 1.509, 0., (.000146, 0.)),
                     pair(.0400, world=Rotation.from_rotvec([np.pi/2, 0., 0.]), at=(.2, -.1, 1.1)),
                     pair(.03994, 2.75, 0., (-.000075, .000225)), pair(.03996, 2.5, 0., (-.0001, .0002)),
                     pair(self.c.shoulder_u+.0015)):
            with self.subTest(case=case):
                result = self.hold(case)
                self.assertTrue(result["success"])
                self.assertEqual(result["nubs_snapped"], 4)
        self.assertAlmostEqual(self.hold(pair(self.c.shoulder_u+.0015))["mouth_past_collar_m"], .0015)

    def test_dwell_and_stillness_are_needed_for_success_but_not_for_credit(self):
        evaluator = Evaluator()
        early = self.run_poses([pair(.040)]*(self.need-1), evaluator)
        self.assertFalse(early["success"])
        self.assertAlmostEqual(early["score"], .9)
        self.assertTrue(evaluator.sample(self.e)["success"])
        sliding = [pair(.040, at=(.3*k*self.c.dt, 0., .76)) for k in range(self.need+100)]
        moving = self.run_poses(sliding)
        self.assertFalse(moving["success"])
        self.assertAlmostEqual(moving["score"], .9)
        evaluator = Evaluator()
        self.run_poses(sliding, evaluator)
        self.assertTrue(self.hold(pair(.040, at=sliding[-1][0][:3]), .5, evaluator)["success"])

    def test_unsnapped_removed_posted_and_beside_caps_fail(self):
        c = self.c
        self.assertFalse(self.hold(pair(.0395))["success"])            # nubs on the crest flat
        self.assertFalse(self.hold(pair(.03964))["success"])           # nubs 10 um short of the crest's edge
        self.assertFalse(self.hold(pair(.03966))["success"])           # 10 um past it, still resting on the corner
        # Cocked 2 deg (both feasible): nub 1 is 25 um past the edge but rests on the corner
        # (normal 6.7 deg from radial); 60 um more press clears it (21.7 deg) and all four count.
        cocked = self.hold(pair(.03986, 2., 0., (0., .0002)))
        self.assertEqual((cocked["success"], cocked["nubs_snapped"]), (False, 3))
        self.assertTrue(self.hold(pair(.03992, 2., 0., (0., .0002)))["success"])
        evaluator = Evaluator()
        self.hold(pair(.0401), evaluator=evaluator)
        ramp = self.hold(pair(.0393), evaluator=evaluator)               # removed past the bead, left on the ramp
        self.assertFalse(ramp["success"])
        self.assertAlmostEqual(ramp["progress"], .1+.9*.5*.0393/c.seat_u)
        posted = (np.r_[0., 0., .76, 0., 0., 0., 1.], np.r_[0., c.pen_length-.040, .76, 0., 0., 0., 1.])
        beside = pair(.0399, lateral=(.015, 0.))
        start = (np.r_[-.07, 0., .7575, Rotation.from_rotvec([0., 0., -np.pi/2]).as_quat()],
                 np.r_[-.08, .07, .7587, Rotation.from_rotvec([0., 0., -np.pi/2]).as_quat()])
        turned_far = (start[0], np.r_[-.07+.07, .07, .7587, Rotation.from_rotvec([0., 0., np.pi/2]).as_quat()])
        for case in (posted, beside, start, turned_far, pair(-.0251), pair(.002, 95.)):
            with self.subTest(case=case):
                result = self.hold(case)
                self.assertFalse(result["success"])
                self.assertEqual(result["score"], 0.)
                self.assertIsNone(result["tip_depth_in_cap_m"])
        # A recorded final state (tick 19200): nub 3 still on the crest flat.
        recorded = (np.r_[0., 0., .76, 0., 0., 0., 1.], np.r_[-.000133697, .039924428, .76-.000173436,
                   0.030043855, 0.028845087, 0.999121711, -0.004597235])
        result = self.hold(recorded)
        self.assertFalse(result["success"])
        self.assertEqual(result["nubs_snapped"], 3)

    def test_progress_rises_with_the_cap_on_the_tip(self):
        scores = [self.hold(pair(u), .05)["score"] for u in np.linspace(-.030, .0405, 80)]
        self.assertEqual(scores[0], 0.)
        self.assertTrue(all(b >= a for a, b in zip(scores, scores[1:])))
        self.assertTrue(all(s < 1. for s in scores))
        seat = self.c.seat_u
        for u, expected in ((-.010, .9*.1*.6), (.010, .9*(.1+.9*.5*.010/seat)), (.0395, .9*(.1+.9*.5*.0395/seat))):
            self.assertAlmostEqual(self.hold(pair(u), .05)["score"], expected)
        # Presented but not yet on the tip: below every engaged state, and no drop on entering the mouth.
        outside, inside = (self.hold(pair(u), .05)["score"] for u in (-1e-6, 1e-6))
        self.assertLess(outside, inside)
        self.assertLess(inside-outside, 1e-4)
        self.assertEqual(self.hold(pair(-.010, 1.5, 0., (.03, 0.)), .05)["score"], 0.)   # 30 mm to the side


if __name__ == "__main__":
    unittest.main()

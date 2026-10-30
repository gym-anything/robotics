"""Contract and regression checks for label-application.

The first class needs no GPU and no renderer (seconds). The physics class builds the real Newton scene and
is skipped unless GAR_LABEL_PHYSICS_TESTS=1 (minutes): restore contract, policy redaction, grasp force.
The render class (GAR_LABEL_RENDER_TESTS=1, Isaac) checks that every render worker is released when its
operation ends.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gym_anything_robotics.errors import UnsupportedConfigurationError
from tests.packages import PACKAGES

ROOT = PACKAGES


class LabelContractTest(unittest.TestCase):
    def test_ten_tasks_resolve(self):
        from label_application.variants import task_resolve, contract
        bench = json.loads((ROOT/"label_application"/"benchmark.json").read_text())
        entries = bench.get("tasks") or bench.get("entries") or bench
        self.assertEqual(sum(len(v) for v in contract()["tasks"].values()), 10)
        self.assertEqual(bench["count"], 10)
        self.assertEqual(set(contract()["seeds"]), {t for v in contract()["tasks"].values() for t in v})
        for preset, tasks in contract()["tasks"].items():
            for task in tasks:
                spec = task_resolve(dict(preset=preset, task=task, seed=3))
                self.assertEqual(spec["task_id"], f"label-application/v1/{preset}/{task}/seed-3")
                self.assertEqual(spec["backend_options"], dict(preset=preset, task=task, seed=3))
                self.assertTrue(spec["description"])
        self.assertTrue(entries)
        for bad in (dict(preset="garage"), dict(preset="parcel", task="tube-wrap"), dict(seed=-1), dict(seed=1.5),
                    dict(preset="part", task="part-id-steel")):
            with self.assertRaises(ValueError):
                task_resolve(bad)
        with self.assertRaisesRegex(ValueError, "retired"):
            task_resolve(dict(preset="part", task="part-id-steel"))

    def test_seeded_layouts(self):
        """Seed 0 keeps fixed reference layouts; every seed gives payloads of a fixed format (the SSCC with a
        valid GS1 check digit), the old label inside the carrier zone and off the centred carrier-top spot, and a
        plate set whose label part number names exactly one plate."""
        from label_application import layouts as Lay
        self.assertEqual(Lay.payload("carrier-top", 0), "961201950123456789")
        self.assertEqual(Lay.payload("tube-wrap", 0), "S2609230742")
        self.assertEqual(Lay.gs1_check_digit("10614141234567890"), "8")         # GS1 General Specifications example
        self.assertEqual(Lay.tube_roll(0), 0.)
        self.assertEqual(Lay.canister_offset(0), (0., 0.))
        for seed in range(40):
            for task, n in (("carrier-top", 18), ("cover-old-label", 18), ("press-down", 18), ("gs1-case-side", 20),
                            ("tube-wrap", 11), ("part-id-hdpe", 19), ("part-id-match", 19),
                            ("smooth-corner", 18), ("finish-securing", 18), ("tube-label", 11)):
                self.assertEqual(len(Lay.payload(task, seed)), n, (task, seed))
            sscc = Lay.payload("gs1-case-side", seed)[2:]
            self.assertEqual(Lay.gs1_check_digit(sscc[:-1]), sscc[-1])
            o = Lay.old_label(seed)
            v = np.abs(np.asarray(o["v"]))
            self.assertTrue(v.min() > .024 and v.max() < 6*.0254 - .005, o)
            self.assertGreaterEqual(abs(o["centre"][0]), .035)
            pl = Lay.plates(seed)
            label_no = Lay.payload("part-id-match", seed).split("-", 1)[1].rsplit("-", 1)[0]
            self.assertEqual([k for k, num in enumerate(pl["part_numbers"]) if num == label_no], [pl["target"]])

    def test_ladder_layouts(self):
        """The task ladder: smooth-corner has exactly one lifted corner on the robots' flap half; finish-securing
        two to four lifted areas on either half; every lifted area is a half-plane of the label, >= 300 mm2, clear of
        the printed symbol, apart from the others, curled at most 100/m (below the paper's 119/m yield); the
        lifted share is 12-35 % on finish-securing."""
        from label_application import layouts as Lay
        G = Lay._grid(1e-3)
        sides, kinds, counts = set(), set(), set()
        for seed in range(120):
            a = Lay.partly_applied("smooth-corner", seed)
            self.assertEqual(a["side"], 1)
            self.assertEqual([f["kind"] for f in a["regions"]], ["corner"])
            self.assertLessEqual(abs(a["yaw"]), math.radians(2.) + 1e-12)
            b = Lay.partly_applied("finish-securing", seed)
            sides.add(b["side"]); counts.add(len(b["regions"]))
            for lay in (a, b):
                masks = [Lay.flap_mask(G, f)[0] for f in lay["regions"]]
                for f, m in zip(lay["regions"], masks):
                    kinds.add(f["kind"])
                    self.assertGreaterEqual(m.sum()*1e-6, 300e-6)
                    self.assertGreaterEqual(Lay._symbol_distance(G[m]).min(), Lay.KEEPOUT)
                    self.assertLessEqual(f["curvature"], Lay.MAX_CURL + 1e-9)
                    self.assertTrue(60. - 1e-9 <= f["tip_deg"] <= 90. + 1e-9 or f["curvature"] >= Lay.MAX_CURL - 1e-9)
                self.assertFalse(np.any(np.sum(masks, 0) > 1))
            frac = np.any([Lay.flap_mask(G, f)[0] for f in b["regions"]], 0).mean()
            self.assertTrue(.12 <= frac <= .35, (seed, frac))
        self.assertEqual(sides, {1, -1})
        self.assertEqual(kinds, {"corner", "end", "side"})
        self.assertTrue({2, 3} <= counts <= {2, 3, 4})
        self.assertEqual(Lay.partly_applied("finish-securing", 5), Lay.partly_applied("finish-securing", 5))

    def test_edge_regions_cover_every_lifted_area(self):
        """The eight fixed perimeter regions the repair rungs hold to the standard (layouts.edge_regions) are the
        largest areas the generator draws, so every lifted area drawn lies inside the region of its kind and place, and
        none reaches the printed symbol."""
        from label_application import layouts as Lay
        G = Lay._grid(1e-3)
        regions = Lay.edge_regions()
        R = {f["name"]: Lay.flap_mask(G, f)[0] for f in regions}
        self.assertEqual(len(R), 8)
        for m in R.values():
            self.assertGreaterEqual(Lay._symbol_distance(G[m]).min(), 0.)
            self.assertFalse((Lay._symbol_distance(G[m]) == 0).any())
        for seed in range(60):
            for task in ("smooth-corner", "finish-securing"):
                for f in Lay.partly_applied(task, seed)["regions"]:
                    m = Lay.flap_mask(G, f)[0]
                    self.assertFalse((m & ~R[f["name"]]).any(), (task, seed, f["name"]))

    def test_reward_contracts(self):
        """Every task is scored by a progress reward: the ladder's contract progress-v1 (the repair rungs),
        progress-v2 for the seven other tasks (the mean of the checks is reported beside it as
        reward.mean_of_checks), and tube-label's own progress-v3 (the take-and-apply rule, its window
        read from where the label is going)."""
        from label_application.task import REWARD_CONTRACTS
        from label_application.config import LADDER_TASKS, START_CREDIT_TASKS
        from label_application.variants import contract
        self.assertEqual(REWARD_CONTRACTS, contract()["reward_contracts"])
        for t, c in REWARD_CONTRACTS.items():
            self.assertEqual(c, "progress-v3" if t == "tube-label" else
                             "progress-v1" if t in LADDER_TASKS else "progress-v2", t)
        self.assertEqual(sorted(LADDER_TASKS), ["finish-securing", "smooth-corner", "tube-label"])
        self.assertEqual(sorted(START_CREDIT_TASKS), ["finish-securing", "press-down", "smooth-corner"])

    def test_other_seven_registration_ramps(self):
        """Contract progress-v2: gs1-case-side's and part-id-hdpe's registration factors are 1 wherever the
        placement check passes and fall linearly to 0 where the label would leave the face (the carton's base or top,
        the canister wall's end), at the vertical edge (symbol clearance 0) or at a diagonal label (45 deg)."""
        from label_application.task import gs1_registration_factors as gs1, canister_registration_factors as can
        H = .2032
        for b in (.026, .032, .038):                                   # the 32 +- 6 mm band
            self.assertEqual(gs1(b, .019, 5., .01, .16, H), (1., 1., 1.))
        self.assertAlmostEqual(gs1(.020, .03, 0., .002, .15, H)[0], .25)       # label 2 mm off the base: 18 mm is 0
        self.assertEqual(gs1(.010, .03, 0., 0., .14, H)[0], 0.)                 # label down on the base
        self.assertAlmostEqual(gs1(.044, .03, 0., .02, .17, H)[0], 33.2/39.2)  # 33.2 mm below the top edge
        self.assertEqual(gs1(.060, .03, 0., .05, H, H)[0], 0.)                  # label up to the top edge
        self.assertAlmostEqual(gs1(.032, .0095, 0., .01, .16, H)[1], .5)       # symbol half the 19 mm off the edge
        self.assertEqual(gs1(.032, 0., 0., .01, .16, H)[1], 0.)
        self.assertAlmostEqual(gs1(.032, .03, 25., .01, .16, H)[2], .5)
        self.assertEqual(gs1(.032, .03, 45., .01, .16, H)[2], 0.)
        self.assertEqual(can(.0, .1, .1, 0.), (1., 1., 1.))
        self.assertEqual(can(.0299, .07, .13, 4.9), (1., 1., 1.))
        self.assertAlmostEqual(can(.040, .060, .13, 0.)[0], 6/7)                 # 10 mm past the band, 60 mm of wall left
        self.assertAlmostEqual(can(-.040, .13, .060, 0.)[0], 6/7)                # the same downward
        self.assertEqual(can(.0996, 0., .2, 0.)[:2], (0., 0.))                   # label at the wall's top
        self.assertAlmostEqual(can(.0, .01, .2, 0.)[1], .5)                      # 10 mm from the top (rule: 20 mm)
        self.assertAlmostEqual(can(.0, .1, .1, 25.)[2], .5)

    def test_tube_label_registration_ramps(self):
        """tube-label's registration factors are 1 inside tube-wrap's band (top edge 17.5-20.0 mm from the cap end,
        skew <= 5 deg, window >= 5 mm) and reach 0 only where the placement stops being a labelled tube: the cap (top
        2.0 mm), the far end at the 9 mm bottom clearance (top edge 40.2 mm for the 50.8 mm label), 45 deg skew, no
        window. The window / 5 mm factor checked here is tube-wrap's (tube-label's own is
        test_tube_label_window_ramp)."""
        from label_application.task import registration_factors
        g = dict(L=.100, mfr_label_top_from_cap_end=.018)
        f = lambda top, skew=0., win=.01: registration_factors(top, skew, win, g, .0508)[:3]
        self.assertEqual(f(.0175), (1., 1., 1.))
        self.assertEqual(f(.0199), (1., 1., 1.))
        self.assertAlmostEqual(f(.017)[0], 1. - .5/15.5)            # 1 mm above the manufacturer label's edge: 0.968
        self.assertAlmostEqual(f(.022)[0], 1. - 2./20.2)            # 4 mm below it: 0.901
        self.assertAlmostEqual(f(.0025)[0], .5/15.5)                # lined up with the cap end: 0.032
        self.assertEqual(f(.002)[0], 0.)
        self.assertAlmostEqual(f(.0402)[0], 0.)
        self.assertEqual(f(.0185, 5.)[1], 1.)
        self.assertAlmostEqual(f(.0185, 10.)[1], .875)
        self.assertEqual(f(.0185, 45.)[1], 0.)
        self.assertAlmostEqual(f(.0185, 0., 4.3e-3)[2], .86)
        self.assertEqual(f(.0185, 0., 0.)[2], 0.)

    def test_tube_label_window_ramp(self):
        """tube-label's window factor is 1 while at most TUBE_STRIP_TOL (2 mm) of the manufacturer label's bare strip
        is covered (its over_mfr_label check) and falls linearly to 0 with the whole strip covered; tube-wrap's stays
        window / 5 mm (the cap-end and skew factors are shared)."""
        from label_application.task import registration_factors, TUBE_STRIP_TOL
        g = dict(L=.100, mfr_label_top_from_cap_end=.018)
        bare = .01355
        f = lambda win: registration_factors(.0185, 0., win, g, .0508, bare_strip=bare)[2]
        self.assertEqual(TUBE_STRIP_TOL, 2e-3)
        self.assertEqual(f(bare), 1.)
        self.assertEqual(f(bare - 2e-3), 1.)
        self.assertAlmostEqual(f(bare - 3e-3), (bare - 3e-3)/(bare - 2e-3))
        self.assertAlmostEqual(f(5.45e-3), 5.45/11.55)            # turned 80 deg: 59 % of the strip covered
        self.assertEqual(f(0.), 0.)
        self.assertEqual(registration_factors(.0185, 0., 5.45e-3, g, .0508)[2], 1.)   # tube-wrap: 5 mm left is enough
        self.assertEqual(registration_factors(.0185, 7.5, 5.45e-3, g, .0508, bare_strip=bare)[:2],
                         registration_factors(.0185, 7.5, 5.45e-3, g, .0508)[:2])

    def test_tube_label_lost_label_rule(self):
        """Evaluator._label_lost calls a label lost only when it lies off the bench top (area centroid, the item's
        off-bench test) with no node on the tube, no alive node bonded to anything and no node within reach (bounding
        sphere + 1.5 mm) of a robot or tool collider. Constructed label state, no physics."""
        from types import SimpleNamespace
        from label_application.task import Evaluator, TABLE, BENCH_HALF, LOST_LABEL_ZEROES
        self.assertEqual(LOST_LABEL_ZEROES, ("tube-label",))
        n = 4
        pad = SimpleNamespace(role="pad", p=np.array([0., 0., 5.]), size=(.01, .005, .005))
        lab = SimpleNamespace(n=n, x=None, bond=-np.ones(n, int), vertex_area=np.full(n, 1e-6),
                              vertex_alive=np.ones(n, bool), colliders=[SimpleNamespace(role="static"), pad])
        ev = Evaluator.__new__(Evaluator)
        ev.world = SimpleNamespace(label=lab)
        none = np.zeros(n, bool)
        quad = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.], [.02, .02, 0.]])
        lost = lambda _=None: ev._label_lost(none, [1])
        lab.x = quad + [BENCH_HALF[0] + .3, 0., .5]                       # beside the bench, unbonded
        self.assertTrue(lost(None))
        lab.x = quad + [0., 0., -.2]                                       # below the floor plane
        self.assertTrue(lost(None))
        lab.x = quad + [.5, .1, TABLE + .001]                              # lying on the bench
        self.assertFalse(lost(None))
        lab.x = quad + [BENCH_HALF[0] + .3, 0., .5]
        lab.bond[0] = 0                                                    # one node still stuck to the bench
        self.assertFalse(lost(None))
        lab.bond[0] = -1
        self.assertFalse(ev._label_lost(np.r_[True, np.zeros(n - 1, bool)], [1]))   # a node on the tube
        pad.p = lab.x[0] + [.01, 0., 0.]                                   # a fingertip pad at the label
        self.assertFalse(lost(None))
        pad.p = lab.x[0] - [float(np.linalg.norm(pad.size)) + 1.6e-3, 0., 0.]   # just beyond reach of the nearest node
        self.assertTrue(lost(None))

    def test_config_rejects_unknown_fields(self):
        from label_application.config import Config
        Config(preset="part", task="part-id-hdpe")
        with self.assertRaises(TypeError):
            Config(roller_magic=1.)

    def test_catalog_setups_and_own_controller_profile(self):
        from robots.catalog import ENVIRONMENTS, resolve
        item = ENVIRONMENTS["label-application"]
        self.assertEqual(item["default"], "dual-fr3")
        self.assertEqual(item["aliases"], {"dual-fr3": "label-fr3", "dual-panda": "label-panda",
                                           "single-fr3": "label-fr3-single"})
        for rig in item["aliases"].values():
            for inst in resolve(rig).to_dict()["instances"].values():
                self.assertEqual(inst["controller_profile"], "label_application")

    def test_g1_rejected_with_reason(self):
        from gym_anything_robotics.launch import LaunchSpec
        with self.assertRaisesRegex(ValueError, "does not support robot 'g1'.*fingertip"):
            LaunchSpec("label-application", robot="g1")
        from gym_anything_robotics.backends.newton_label_application import NewtonLabelApplicationBackend
        for bad in ("g1", {"instances": {}}):
            with self.assertRaises(UnsupportedConfigurationError):
                NewtonLabelApplicationBackend(robot=bad)
        with self.assertRaises(UnsupportedConfigurationError):
            NewtonLabelApplicationBackend(robot="single-fr3", cameras=("right_wrist_rgb",))
        with self.assertRaises(ValueError):
            NewtonLabelApplicationBackend(cameras=("carton_top",))       # privileged diagnostic camera

    def test_doctor_declares_label_dependencies(self):
        from gym_anything_robotics.launch import LaunchSpec, doctor
        rep = doctor(LaunchSpec("label-application"))
        names = [c["check"] for c in rep["environment_checks"]["checks"]]
        for mod in ("barcode", "zxingcpp", "pypardiso"):
            self.assertIn(f"import {mod}", names)
        self.assertFalse(any("label_application_pylib" in str(p) for p in __import__("sys").path))

    def test_pad_bonds_only_from_the_adhesive_side_of_the_sheet(self):
        """A node of a nearly level label inside a closed fingertip that presses it from above, nearest the pad's
        pinching face, is read against that face's horizontal normal, and the label's small tilt toward it passes the
        adhesive-side test, which alone would bond the pad to the PRINTED side (smooth-corner tears). Geometry of a
        recorded smooth-corner robot event (pinch face, ln . n = 0.0195), through the real Collider.sdf: the
        adhesive-side test alone bonds, the pad body is on the printed side, the rule refuses it. Still bonded: a
        recorded smooth-corner seed-0 event (node 1785 at tick 3013, just outside the pad's outer side face, its
        adhesive facing the pad: pad body 2.11 mm on the adhesive side), a pinched sheet's adhesive-side pad, a sheet
        wrapped over the rim, and non-pad colliders."""
        from types import SimpleNamespace
        from label_application import labelsim as L, materials as M
        from label_application.scene import FINGERTIP_ROUNDING
        mk = lambda name: L.Collider(name, "box", (.00875, .0076, .00925), M.SURFACES["finger_pad"], role="pad",
                                     rounding=FINGERTIP_ROUNDING, pinch_axis=1, pinch_sign=-1.)
        # (1) printed-side event: pad pressing from above, end face (local +z) down
        face_n = np.array([-0.8180053940606037, -0.5750535992776646, 0.013428376669521291])
        ln1 = np.array([-0.09450469359875688, 0.12357653418000968, 0.9878247329804322])
        y = -face_n/np.linalg.norm(face_n)                        # pinch_sign -1: outward pinch normal = -R e_y
        z = np.array([0., 0., -1.]) - (np.array([0., 0., -1.]) @ y)*y
        z /= np.linalg.norm(z)
        R = np.column_stack([np.cross(y, z), y, z])
        pad = mk("right_fr3_leftfinger_pad_112")
        pad.set_body_pose(R, np.zeros(3), first=True)
        x1 = R @ np.array([-7.212627515725556, -7.270882907452875, 8.24171316912617])*1e-3
        d, n1, _ = pad.sdf(x1[None])
        self.assertTrue(pad.last_pinch[0])                         # classified against the pinching face
        self.assertLess(d[0], 0.)                                  # inside the pad
        np.testing.assert_allclose(n1[0], face_n/np.linalg.norm(face_n), atol=1e-9)
        self.assertGreater(float(ln1 @ n1[0]), 0.)                 # the adhesive-side test alone: bond
        self.assertGreater(float((pad.p - x1) @ ln1), 5e-3)        # pad body on the printed side
        # (2) side-face event: pad at the origin, node just outside its outer side face (local +x)
        pad2 = mk("right_fr3_rightfinger_pad_118")
        pad2.set_body_pose(np.eye(3), np.zeros(3), first=True)
        x2 = np.array([8.802, 0.214, 4.238])*1e-3
        ln2 = np.array([0.5897, -0.3909, -0.7067]); ln2 /= np.linalg.norm(ln2)
        d2, n2, _ = pad2.sdf(x2[None])
        self.assertGreater(d2[0], 0.)
        np.testing.assert_allclose(n2[0], [1., 0., 0.], atol=1e-9)
        self.assertAlmostEqual(float((pad2.p - x2) @ ln2)*1e3, -2.112, places=2)
        carton = L.Collider("carton", "box", (.1, .1, .1), M.SURFACES["kraft"], role="item")
        sim = SimpleNamespace(colliders=[pad, pad2, carton])
        down = R[:, 2]                                             # end face outward normal
        wrap = (-R[:, 1] + R[:, 2])/np.sqrt(2)                     # rim fillet between pinching and end face
        x = np.array([x1,                                          # printed side (above)
                      pad.p - .0076*R[:, 1],                       # pinched: on the pinching face, adhesive on pad
                      pad.p + .00925*down,                         # pressed by the end face, printed side up
                      pad.p + (.0076 - .002)*-R[:, 1] + (.00925 - .002)*R[:, 2] + .002*wrap,   # on the rim fillet
                      x2,                                          # side face: adhesive on the pad
                      x1, x1])                                     # carton pairs (any geometry)
        Q = dict(v=np.arange(7), c=np.array([0, 0, 0, 0, 1, 2, 2]),
                 n=np.array([n1[0], -R[:, 1], down, wrap, n2[0], [0., 0., 1.], [0., 0., -1.]]))
        lns = np.array([ln1, -R[:, 1], -down, wrap, ln2, ln1, ln1])
        ok = L.LabelSim._pad_on_adhesive_side(sim, Q, x, lns)
        self.assertEqual(ok.tolist(), [False, True, False, True, True, True, True])
        adh = (lns*Q["n"]).sum(1) > 0.
        self.assertEqual(adh.tolist(), [True, True, False, True, True, True, False])
        self.assertEqual((adh & ok).tolist(), [False, True, False, True, True, True, False])
        self.assertTrue(L.PAD_BOND_ADHESIVE_SIDE)


    def test_liner_web_blocks_unbonded_points(self):
        from label_application import labelsim as L, materials as M
        w = L.LinerWeb("web", M.SURFACES["liner"], start=(-.10, 0., 0.), edge_x=0., radius=1e-3, z_top=0., y_half=.03)
        d, n, _ = w.sdf(np.array([[-.05, 0, .001], [.0005, 0, .0005], [.0015, 0, -.003], [-.011, 0, -.0042]]))
        np.testing.assert_allclose(d[:3], [.001, np.hypot(.0005, .0015) - .001, .0005], atol=1e-9)
        self.assertGreater(d[3], 1.)                       # far behind the web, under the plate: no contact
        np.testing.assert_allclose(n[2], [1., 0., 0.], atol=1e-9)


@unittest.skipUnless(os.environ.get("GAR_LABEL_PHYSICS_TESTS") == "1", "set GAR_LABEL_PHYSICS_TESTS=1 (minutes, GPU)")
class LabelPhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gym_anything_robotics as gar
        cls.env = gar.create("label-application", robot="single-fr3", preset="part", task="part-id-match", cameras=())
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_restore_returns_state_and_info(self):
        from gym_anything_robotics.actions import NamedJointTarget
        be = self.env._backend
        self.env.save_checkpoint(label="t")
        tick = be.env.world.steps
        self.env.act(NamedJointTarget({"left/fr3_joint1": .03}, ticks=25))
        out = be.restore_checkpoint("t")
        self.assertIsInstance(out, tuple)
        state, info = out
        self.assertEqual(info["tick"], tick)
        self.assertEqual(be.env.world.steps, tick)

    def test_policy_access_is_redacted(self):
        access = self.env._backend.policy_access()
        keys = {k[0] for k in access.state_fields} if hasattr(access, "state_fields") else set()
        self.assertFalse(any("label" in str(k) or "bond" in str(k) for k in keys))

    def test_force_mode_grasp_holds_commanded_force(self):
        """Franka Hand grasp: the pinch between the pads holds the commanded per-finger force."""
        from gym_anything_robotics.actions import Gripper
        be = self.env._backend
        # from fully open (40 mm per finger) the Franka Hand closes at its 50 mm/s datasheet speed: ~0.8 s to
        # contact, then the force-controlled hold; read the pinch after it has settled
        self.env.act(Gripper(arm="left", inward_effort_n=20., half_width_m=None, duration_s=1.5))
        w = be.env.world
        pads = {c.body for c in w.scene.colliders if c.role == "pad" and c.name.startswith("left")}
        pinch = .5*sum(f for b in pads for ob, f in w.last_pad_pair_loads.get(b, {}).items() if ob in pads)
        # API contract: inward_effort_n is the TOTAL inward force of both fingers; the pad-to-pad squeeze is
        # each finger's share, 10 N for 20 N
        self.assertLess(abs(pinch - 10.)/10., .05)


@unittest.skipUnless(os.environ.get("GAR_LABEL_PHYSICS_TESTS") == "1", "set GAR_LABEL_PHYSICS_TESTS=1 (builds Newton scenes)")
class LabelStartStateTest(unittest.TestCase):
    """Every task's start state on the real Newton world (no stepping, no robot): the evaluator judges it a failure
    with no progress, and does not raise."""

    def test_every_start_state_fails_without_raising(self):
        from label_application.config import Config, TASKS
        from label_application.world import World
        from label_application.env import make_art, prepare_start
        from label_application.task import Evaluator
        from label_application.config import START_CREDIT_TASKS
        with tempfile.TemporaryDirectory() as tmp:
            for preset, tasks in TASKS.items():
                for task in tasks:
                    c = Config(preset=preset, task=task, seed=7, device="cpu", record_trace=False)
                    w = World(c)
                    prepare_start(w)
                    art = make_art(c, tmp)
                    if task in START_CREDIT_TASKS:
                        # as LabelEnv does after its settle: the start credit the net-work progress is measured from
                        w.scene.task_geometry["start_credit"] = Evaluator(w, art).node_credit()
                    r = Evaluator(w, art).evaluate()
                    self.assertFalse(r["success"], task)
                    self.assertTrue(r["checks"]["item_resting_on_bench"], task)
                    self.assertEqual(r["progress"], 0., task)      # nothing done yet (every task)
                    self.assertEqual(r["score"], 0., task)


@unittest.skipUnless(os.environ.get("GAR_LABEL_RENDER_TESTS") == "1", "set GAR_LABEL_RENDER_TESTS=1 (minutes, GPU, Isaac)")
class LabelRenderLifecycleTest(unittest.TestCase):
    """Render-only workers are released when their operation ends (observation, evaluation, diagnostic render),
    including on errors; none is kept idle between operations."""

    @classmethod
    def setUpClass(cls):
        import label_application.render as R
        cls.made = []
        orig = R.Renderer.__init__

        def init(self, *a, **k):
            cls.made.append(self)
            orig(self, *a, **k)
        cls._orig, R.Renderer.__init__ = orig, init
        import gym_anything_robotics as gar
        cls.env = gar.create("label-application", robot="single-fr3", preset="part", task="part-id-match",
                             cameras=("bench_rgb",))

    @classmethod
    def tearDownClass(cls):
        import label_application.render as R
        R.Renderer.__init__ = cls._orig
        cls.env.close()

    def assert_released(self):
        be = self.env._backend
        self.assertIsNone(getattr(be, "_prestart", None))
        for t in getattr(be, "_reapers", []):
            t.join(60)
        alive = [r.process.pid for r in self.made if getattr(r, "process", None) is not None and r.process.poll() is None]
        self.assertEqual(alive, [], "render workers still alive after their operation")

    def test_workers_released_after_each_operation(self):
        from gym_anything_robotics.actions import NamedJointTarget
        obs, _ = self.env.reset()
        self.assertIn("bench_rgb", obs.images)
        self.assert_released()
        n = len(self.made)
        step = self.env.act(NamedJointTarget({"left/fr3_joint1": .03}, ticks=25))
        self.assertIn("bench_rgb", step.observation.images)
        self.assertEqual(len(self.made), n + 1, "one worker per action operation (started during the physics)")
        self.assert_released()
        self.env._backend.evaluate()
        self.assert_released()

    def test_start_states_get_a_verdict_through_evaluate(self):
        """Every task's start state must return success False through the GAR path."""
        from gym_anything_robotics.backends.newton_label_application import NewtonLabelApplicationBackend
        from label_application.config import TASKS
        with tempfile.TemporaryDirectory() as tmp:
            for preset, tasks in TASKS.items():
                for task in tasks:
                    be = NewtonLabelApplicationBackend(robot="dual-fr3", preset=preset, task=task, seed=7,
                                                       output_dir=Path(tmp)/task, cameras=(), recording=False)
                    try:
                        self.assertFalse(be.evaluate()["success"], task)
                    finally:
                        be.close()

    def test_failed_action_releases_its_worker(self):
        from gym_anything_robotics.actions import NamedJointTarget
        be = self.env._backend
        orig = be.env.act

        def fail(*a, **k):
            raise RuntimeError("injected physics failure")
        be.env.act = fail
        try:
            with self.assertRaises(Exception):
                self.env.act(NamedJointTarget({"left/fr3_joint1": .03}, ticks=25))
        finally:
            be.env.act = orig
        self.assert_released()


if __name__ == "__main__":
    unittest.main()

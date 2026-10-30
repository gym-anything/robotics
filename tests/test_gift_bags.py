"""Contract and regression checks for gift-bags.

The first two classes need no GPU, no renderer and no physics stepping (seconds; the second judges bag-purchase on
constructed states). The physics class builds the real scenes
and is skipped unless GIFT_BAGS_PHYSICS_TESTS=1 (minutes): exact checkpoint restore, the contact caliper's resting
distance, the evaluator's negative controls on a live state.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from gym_anything_robotics.errors import UnsupportedConfigurationError
from tests.packages import PACKAGES

ROOT = PACKAGES


class GiftBagsContractTest(unittest.TestCase):
    def test_six_tasks_resolve(self):
        from gift_bags.variants import task_resolve, contract
        from gift_bags import task_set_catalog
        bench = json.loads((ROOT/"gift_bags"/"benchmark.json").read_text())
        self.assertEqual(sum(len(v) for v in contract()["tasks"].values()), 6)
        # benchmark.json lists the task-set catalogue: the six tasks first, then the three it appends
        catalogue = [t for ts in task_set_catalog.contract()["tasks"].values() for t in ts]
        self.assertEqual(bench["count"], len(catalogue))
        self.assertEqual(contract()["tasks"]["receiving"], ["open-carton"])
        for preset, tasks in contract()["tasks"].items():
            for task in tasks:
                spec = task_resolve(dict(preset=preset, task=task, seed=2))
                self.assertEqual(spec["task_id"], f"gift-bags/v1/{preset}/{task}/seed-2")
                self.assertEqual(spec["backend_options"], dict(preset=preset, task=task, seed=2))
                self.assertTrue(spec["description"])
        for bad in (dict(preset="dev"), dict(preset="counter", task="lift-lid"), dict(seed=-1), dict(seed=1.5),
                    dict(task="no-such-task")):
            with self.assertRaises(ValueError):
                task_resolve(bad)
        # a task name alone selects its preset (names are unique), and the CLI can list and take the names
        self.assertEqual(bench["cli_task_selector"], "task")
        self.assertEqual(bench["selector_choices"]["preset"], list(contract()["tasks"]))
        self.assertEqual(bench["selector_choices"]["task"][:6], [t for ts in contract()["tasks"].values() for t in ts])
        self.assertEqual(sorted(bench["selector_choices"]["task"]), sorted(catalogue))
        for preset, tasks in contract()["tasks"].items():
            for task in tasks:
                self.assertEqual(task_resolve(dict(task=task))["backend_options"], dict(preset=preset, task=task, seed=0))

    def test_instructions_state_every_rule_their_evaluator_applies(self):
        """An instruction states what its checks require that a reader could not take for granted. Ordinary
        workmanship may stay implicit (a task description is not the verifier): bag-purchase does not say "Do not tear
        the bag", which nobody bagging a customer's purchase needs to be told, and its tear check stays. It states its
        end state without a sequence ("then")."""
        from gift_bags.task import INSTRUCTIONS
        self.assertIn("Do not tear the bag", INSTRUCTIONS["open-bag"])
        purchase = INSTRUCTIONS["bag-purchase"]
        for words in ("square its bottom", "one of the bottom boards", "standing open on the dark rubber counter mat",
                      "boxed candle inside, upright on the board", "Release everything when you are done"):
            self.assertIn(words, purchase)
        self.assertNotIn("then", purchase.split())
        self.assertIn("Do not lift or tip the box's base", INSTRUCTIONS["lift-lid"])
        self.assertIn("you may pick the present up", INSTRUCTIONS["open-present"])
        self.assertIn("base standing upright on the table", INSTRUCTIONS["open-present"])

    def test_slack_law_is_an_energy_with_zero_force_between_crushed_and_true_length(self):
        """Crumpled triangles (ipcworld.slack_response): the force is the gradient of the energy, the tangent its
        derivative; along a slack axis nothing is carried between the crushed and the true length, the stock's stiffness
        beyond either; with no slack it is the plain orthotropic law."""
        from gift_bags import ipcworld as W, params as Pm
        rng = np.random.default_rng(0)
        m = 600
        C = np.tile(Pm.TOTE.membrane_C(0.), (m, 1, 1))
        c = -rng.uniform(0., .3, (m, 2))*(rng.random((m, 2)) > .3)
        phi = rng.uniform(-np.pi, np.pi, m)*(rng.random(m) > .3)
        e = rng.normal(0., .08, (m, 3))

        def energy(e_):
            r, S, _, _ = W.slack_response(C, e_, c, phi, tangent=False)
            return .5*(r*S).sum(1)
        r, S, Ct, k = W.slack_response(C, e, c, phi)
        h = 1e-7
        for i in range(3):
            d = np.zeros(3); d[i] = h
            np.testing.assert_allclose((energy(e + d) - energy(e - d))/(2*h), S[:, i], atol=1e-3*np.abs(C).max()*1e-6 + 1e-2)
            dS = (W.slack_response(C, e + d, c, phi, tangent=False)[1] - W.slack_response(C, e - d, c, phi, tangent=False)[1])/(2*h)
            np.testing.assert_allclose(dS, Ct[:, :, i], atol=1e-5*np.abs(C).max())
        self.assertTrue(((k <= 1e-12) & (k >= c - 1e-12)).all())
        self.assertTrue((energy(e) <= .5*np.einsum("ti,tij,tj->t", e, C, e) + 1e-9).all())
        one = lambda ex: W.slack_response(C[:1], np.array([[ex, 0., 0.]]), np.array([[-.2, 0.]]), np.zeros(1))
        for ex in (-.2, -.1, 0.):
            self.assertEqual(np.abs(one(ex)[1]).max(), 0.)                       # anywhere between crushed and true
        self.assertGreater(one(.01)[1][0, 0], .009*Pm.TOTE.Et_md)                # beyond the true length: the stock
        self.assertLess(one(-.25)[1][0, 0], -.04*Pm.TOTE.Et_md)                  # below the crushed length: the stock
        r0, S0, C0, _ = W.slack_response(C, e, np.zeros((m, 2)), phi)
        np.testing.assert_allclose(S0, np.einsum("tij,tj->ti", C, e), rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(C0, C, rtol=1e-9, atol=1e-6)

    def test_config_rejects_unknown_fields_and_bad_values(self):
        from gift_bags.config import Config
        Config(preset="home", task="lift-lid")
        with self.assertRaises(TypeError):
            Config(magic_glue=1.)
        with self.assertRaises(ValueError):
            Config(preset="home", task="open-bag")

    def test_catalog_and_unsupported_robots(self):
        from robots.catalog import ENVIRONMENTS
        item = ENVIRONMENTS["gift-bags"]
        self.assertEqual(item["default"], "dual-fr3")
        self.assertEqual(set(item["aliases"]), {"dual-fr3", "dual-panda", "single-fr3", "single-panda"})
        from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
        # the G1 only stands here (no balance controller: it falls when its arms move): refused under every id
        for bad in ("g1", "gift-g1", "scan-g1", "whiteboard-g1", "mobile-fr3-duo", {"instances": {}}):
            with self.assertRaises(UnsupportedConfigurationError):
                NewtonGiftBagsBackend(robot=bad, preset="home", task="lift-lid")
        with self.assertRaises(UnsupportedConfigurationError) as cm:
            NewtonGiftBagsBackend(robot="g1", preset="home", task="lift-lid")
        self.assertIn("balance controller", str(cm.exception))
        with mock.patch.dict("os.environ", {"GIFT_BAGS_G1_DEV": "1"}), self.assertRaises(UnsupportedConfigurationError):
            NewtonGiftBagsBackend(robot="g1", preset="receiving")          # no G1 layout at the receiving table
        with self.assertRaises(UnsupportedConfigurationError):
            NewtonGiftBagsBackend(robot="single-fr3", preset="counter")
        with self.assertRaises(ValueError):
            NewtonGiftBagsBackend(preset="home", task="lift-lid", cameras=("room",))   # privileged diagnostic camera

    def test_doctor_reports_gpu_memory_and_the_cpu_device_switch(self):
        """A session keeps the robot model on the GPU unless GIFT_BAGS_DEVICE=cpu; on a full card a launch dies with a
        CUDA out-of-memory error. The doctor says so beforehand."""
        from gift_bags import doctor as D
        from gift_bags.config import Config, default_device
        ok, detail = D.gpu_check({}, lambda: (300., 32000.))
        self.assertFalse(ok)
        self.assertIn("GIFT_BAGS_DEVICE=cpu", detail)
        ok, detail = D.gpu_check({}, lambda: (2000., 32000.))
        self.assertTrue(ok)
        self.assertIn("render worker", detail)                     # enough for a session, not for images
        ok, detail = D.gpu_check({"GIFT_BAGS_DEVICE": "cpu"}, lambda: (300., 32000.))
        self.assertTrue(ok)
        ok, detail = D.gpu_check({}, lambda: (20000., 32000.))
        self.assertTrue(ok)
        self.assertNotIn("render worker", detail)
        self.assertIn(default_device(), ("cuda:0", os.environ.get("GIFT_BAGS_DEVICE")))
        self.assertEqual(Config(device="cpu").device, "cpu")


    def test_physics_digest_covers_every_module_the_physics_imports(self):
        """A record's digest must change when any code its world ran changes. The digest is over
        `probes.PHYSICS_FILES`; every gift_bags module one of those files imports is in the list or is named as
        taking no part in the physics."""
        import re
        from gift_bags import probes
        gb = ROOT/"gift_bags"
        known = set(probes.PHYSICS_FILES) | set(probes.NOT_PHYSICS)
        self.assertEqual(len(set(probes.PHYSICS_FILES)), len(probes.PHYSICS_FILES))
        for f in probes.PHYSICS_FILES:
            self.assertTrue((gb/f).is_file(), f)
            src = (gb/f).read_text()
            mods = set(re.findall(r"from \.(\w+) import", src))
            for names in re.findall(r"from \. import ([\w ,]+)", src):
                mods |= {n.split(" as ")[0].strip() for n in names.split(",")}
            mods |= set(re.findall(r"from gift_bags\.(\w+) import", src))
            mods = {m for m in mods if (gb/f"{m}.py").is_file()}
            self.assertEqual({m + ".py" for m in mods} - known, set(), f)
        for f in ("dressing.py", "startcache.py", "robot_setup.py"):      # props, start-state files, rigs
            self.assertIn(f, probes.PHYSICS_FILES)
        self.assertEqual(len(probes.physics_file_hashes()), len(probes.PHYSICS_FILES))

    def test_every_rig_the_catalog_names_is_a_committed_file(self):
        """A rig file that exists only on the author's machine makes the alias unbuildable in a clean checkout."""
        import subprocess
        from robots.catalog import ENVIRONMENTS
        from gift_bags.robot_setup import G1_RIG
        rigs = sorted(set(ENVIRONMENTS["gift-bags"]["aliases"].values()) | {G1_RIG})
        for rig in rigs:
            self.assertTrue((ROOT / "robots" / "rigs" / f"{rig}.json").is_file(), rig)
        if not (ROOT / ".git").exists():
            self.skipTest("not a git checkout: the files exist, which is what an export can show")
        tracked = subprocess.run(["git", "ls-files", "--", *[f"robots/rigs/{r}.json" for r in rigs]], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(sorted(tracked), sorted(f"robots/rigs/{r}.json" for r in rigs))

    def test_g1_is_development_only_and_the_validator_names_what_a_humanoid_needs(self):
        from robots.catalog import for_environment, resolve
        from gift_bags.robot_setup import validate, resolve_setup
        from gift_bags import g1 as G1, scene as S
        with self.assertRaises(ValueError):
            for_environment("gift-bags", "g1")
        for refused in ("g1", "gift-g1", "scan-g1"):                  # scan-g1 is the same hardware under another id
            with self.assertRaises(ValueError) as cm:
                validate(refused)
            self.assertIn("balance controller", str(cm.exception))
        with mock.patch.dict("os.environ", {"GIFT_BAGS_G1_DEV": "1"}):
            self.assertEqual(resolve_setup("g1").to_dict()["instances"]["g1"]["model"], "g1@1")
            for ok in ("g1", "gift-g1", "scan-g1"):
                self.assertEqual(validate(ok), "g1")
        self.assertTrue(G1.is_g1("g1") and not G1.is_g1("dual-fr3"))
        other = resolve("gift-g1").to_dict()
        other["instances"]["g1"]["controller_profile"] = "whiteboard"
        for bad, why in (("whiteboard-g1", "g1_floor_mount"), (other, "barcode_scanning")):
            with self.assertRaises(ValueError) as cm:
                validate(bad)
            self.assertIn(why, str(cm.exception))
        with self.assertRaises(ValueError):
            S.build(__import__("gift_bags.config", fromlist=["Config"]).Config(preset="receiving", task="open-carton", robot="g1"))
        # hand postures: inside the URDF limits by construction of the mirror, closure maps the Franka half-gap scale
        self.assertEqual(G1.closure_of_half_width(.04), 0.)
        self.assertEqual(G1.hand_posture("left", 1.), {G1.joint("left", n): v for n, v in G1.HAND_CLOSED["left"].items()})
        with self.assertRaises(ValueError):
            G1.closure_of_half_width(.05)

    def test_other_gripper_is_refused_with_the_physical_reason(self):
        import copy
        from robots.catalog import resolve
        from gift_bags.robot_setup import validate, QUALIFIED
        setup = copy.deepcopy(resolve(QUALIFIED["dual-fr3"]).to_dict())
        self.assertEqual(validate(copy.deepcopy(setup)), "dual-fr3")
        hand = next(k for k, a in setup["attachments"].items() if a["parent"] == "left/flange")
        setup["attachments"][hand].update(model="wsg50@1", mount_profile="iso_50")   # the catalog cannot resolve this
        with self.assertRaises(ValueError) as cm:
            validate(setup)
        self.assertIn("two-finger parallel gripper with fingertip pads", str(cm.exception))
        self.assertIn("cannot host this robot setup", str(cm.exception))

    def test_render_idle_release_default_and_environment_option(self):
        from gym_anything_robotics.backends import newton_gift_bags as NB
        self.assertGreaterEqual(NB.IDLE_RELEASE_S, 600.)
        with mock.patch.dict("os.environ", {NB.IDLE_RELEASE_ENV: "soon"}):
            with self.assertRaises(ValueError):
                NB.NewtonGiftBagsBackend(preset="home", task="lift-lid")

    def test_every_preset_start_state_is_usable(self):
        """Not skipped: a preset whose cached start state is unsound (the tote torn while its shipping state was made)
        must fail here, in `gar doctor` and in the catalogue, not at reset."""
        from gift_bags import variants as Vr, scene as S
        av = Vr.availability(refresh=True)
        self.assertEqual(set(av), set(Vr.POLICY_PRESETS))
        for preset, st in av.items():
            self.assertTrue(st["available"], f"{preset}: {st['reason']}")
        self.assertEqual(Vr.contract()["unavailable_tasks"], {})
        ss = av["counter"].get("start_state") or {}
        if ss.get("cached") and ss.get("sound"):
            from gift_bags import bagnet as B
            st = B.shipping_state_status(S.BAG_CELLS)
            self.assertEqual(st["vertices"], st["mesh_vertices"])
            self.assertEqual(st["splits"], 0)
        with mock.patch.object(Vr, "_AVAILABILITY", {**av, "counter": dict(available=False, reason="torn (test)")}):
            with self.assertRaises(ValueError) as cm:
                Vr.resolve("counter")
            self.assertIn("unavailable", str(cm.exception))
            self.assertEqual(Vr.default_preset(), "home")
            self.assertIn("counter/open-bag", Vr.contract()["unavailable_tasks"])

    def test_start_states_for_the_current_code_are_shipped(self):
        """Every start state the current code names by its key (the tote's shipping state and handle assembly, the 18
        wrapped presents, the sealed carton) is a committed file under gift_bags/start_states, so a fresh checkout
        launches every task and seed without a build. A generator edit changes the keys and fails this test until
        `python -m gift_bags.prebuild --build --ship` has run; the catalogue lists the same table per variant."""
        import subprocess
        from gift_bags import prebuild as PB, startcache as SC, variants as Vr
        tab = PB.table()
        self.assertEqual([len(tab[k]) for k in ("counter", "home", "receiving")], [2, 18, 1])
        missing = [f"{p}: {r['name']} ({r['where']})" for p, rows in tab.items() for r in rows if r["where"] != "shipped"]
        self.assertEqual(missing, [])
        names = sorted(Path(r["path"]).name for rows in tab.values() for r in rows)
        self.assertEqual(sorted(f.name for f in SC.SHIPPED.glob("*.npz")), names)      # and nothing stale beside them
        if (ROOT/".git").exists():
            tracked = subprocess.run(["git", "ls-files", "--", "gift_bags/start_states"], cwd=ROOT, capture_output=True,
                                     text=True, check=True).stdout.split()
            self.assertEqual(sorted(Path(t).name for t in tracked if t.endswith(".npz")), names)
        av = Vr.availability(refresh=True)
        self.assertEqual(av["home"]["start_states_ready"], [18, 18])
        self.assertEqual(av["counter"]["start_states_ready"], [2, 2])
        self.assertEqual({r["where"] for r in av["home"]["start_states"]}, {"shipped"})

    def test_drawn_props_have_collision_boxes_in_both_simulators(self):
        """Everything drawn standing on a work surface or within an arm's reach has a collision box: furniture for the
        robot simulator and a static body in the contact world, placed from the constants the pictures use."""
        from gift_bags import dressing as D, scene as S
        from gift_bags.config import Config
        top = dict(center=(0., .135, S.COUNTER_Z - .02), half=(.76, .30, .02))
        counter = {p["name"]: p for p in D.physical_props("counter", S.COUNTER_Z, top)}
        for name, key in (("prop_card_terminal", "card_terminal"), ("prop_receipt_printer", "receipt_printer"),
                          ("prop_pos_stand", "pos_stand"), ("prop_ribbon_rack", "ribbon_rack")):
            self.assertEqual(tuple(counter[name]["center"][:2]), tuple(D.COUNTER_PROPS[key][:2]))
            self.assertAlmostEqual(counter[name]["center"][2] - counter[name]["half"][2], S.COUNTER_Z)   # standing on the top
        self.assertIn("prop_tissue_stack", counter)
        self.assertIn("prop_counter_cabinet", counter)
        self.assertEqual(sum(n.startswith("prop_display_shelves") for n in counter), 2)
        for preset, task, expect in (("home", "lift-lid", ("prop_other_present_0", "prop_other_present_1", "prop_chair_0",
                                                           "prop_chair_2", "prop_table_leg_nn")),
                                     ("receiving", "open-carton", ("prop_table_undershelf", "prop_steel_shelves_l",
                                                                   "prop_table_leg_pp"))):
            spec = S.BUILDERS[preset](Config(preset=preset, task=task))
            furn = {f["name"]: f for f in spec.furniture}
            bodies = {b.name: b for b in spec.bodies}
            for name in expect:
                self.assertIn(name, furn)
                self.assertIs(furn[name]["visual"], False)             # the dressing draws the prop, not this box
                self.assertTrue(furn[name].get("robot_collides", True))
                self.assertEqual(bodies[name].kind, "static")
                np.testing.assert_allclose(bodies[name].p, furn[name]["center"])
            props = [n for n in furn if n.startswith("prop_")]
            self.assertEqual([b.name for b in spec.bodies][-len(props):], props)   # last: item indices are unchanged
            # no prop box reaches into a task item at the start
            for b in spec.bodies:
                if b.kind == "static":
                    continue
                V = np.asarray(b.verts) @ np.asarray(b.R).T + np.asarray(b.p)
                for name in props:
                    lo = np.asarray(furn[name]["center"]) - furn[name]["half"]
                    hi = np.asarray(furn[name]["center"]) + furn[name]["half"]
                    self.assertFalse((V.max(0) > lo + 1e-4).all() and (V.min(0) < hi - 1e-4).all(), f"{b.name} in {name}")

    def test_doctor_declares_dependencies(self):
        from gym_anything_robotics.launch import LaunchSpec, doctor
        out = doctor(LaunchSpec("gift-bags"))
        names = [c["check"] for c in out["environment_checks"]["checks"]]
        for need in ("import ipctk", "import pypardiso", "import scipy"):
            self.assertIn(need, names)

    def test_contact_caliper(self):
        from gift_bags import ipcworld as W, params as Pm
        for stock, phi in ((Pm.TOTE, .7977), (Pm.WRAP, .6353)):
            cal = W.contact_caliper(stock)
            self.assertAlmostEqual(cal["phi"], phi, places=3)
            self.assertAlmostEqual(cal["dmin"] + cal["dhat"], stock.t, places=12)        # unloaded pitch = caliper
            self.assertAlmostEqual(cal["rest_sheet_rigid"], (1 - cal["phi"]/2)*stock.t, places=12)
            # 5 % compression at 1 MPa by construction
            d = .95*stock.t
            x, xh = d*d - cal["dmin"]**2, stock.t**2 - cal["dmin"]**2
            p = -cal["kappa"]*W._bprime(x, xh)*2*d
            self.assertAlmostEqual(p/1e6, 1., places=6)

    def test_huber_spring_is_consistent(self):
        from gift_bags.ipcworld import huber3
        rng = np.random.default_rng(1)
        for d in (np.array([1e-4, 0, 0]), rng.normal(size=3)*.01):
            E, g, H = huber3(d, 1e5, 250.)
            for k in range(3):
                e = np.zeros(3); e[k] = 1e-7
                fd = (huber3(d + e, 1e5, 250.)[0] - huber3(d - e, 1e5, 250.)[0])/2e-7
                self.assertAlmostEqual(fd, g[k], delta=1e-3*max(1., abs(g[k])))
            self.assertLessEqual(np.linalg.norm(g), 250. + 1e-9)
            self.assertTrue(np.all(np.linalg.eigvalsh(H) > 0))

    def test_flat_bag_net_is_isometric(self):
        from gift_bags import bagnet as B, ipcworld as W
        X, T, kinds, s, _ = B.bag_mesh(10)
        flat, layer = B.facet_map(X, T, kinds)
        E = W.edges_of(T)
        Lr = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        Lf = np.linalg.norm(flat[E[:, 0]] - flat[E[:, 1]], axis=1)
        self.assertLess(np.abs(Lr - Lf).max(), 1e-6)

    def test_cohesive_law_refuses_damage_after_separation(self):
        from gift_bags.ipcworld import CohesiveLaw
        from gift_bags import params as Pm
        with self.assertRaises(ValueError):            # delta0 0.5 mm > deltaf 40 um: a law that could not peel
            CohesiveLaw("first tape law", k=1.5e10, Gamma=150., sigma=150./20e-6)
        law = Pm.TAPE_ON_WRAP
        self.assertLess(law.delta0, law.deltaf)
        self.assertAlmostEqual(law.sigma, .40e6)

    def test_finger_warm_start_is_the_minimum(self):
        from gift_bags.ipcworld import finger_axis_minimum
        from scipy.optimize import minimize_scalar
        rng = np.random.default_rng(3)
        for _ in range(50):
            mf, kf, F = 10**rng.uniform(2, 4), 3e5, rng.uniform(1., 35.)
            s_pred, s_cmd = rng.normal(0., .01, 2)
            def E(s):
                u = s - s_cmd
                hub = .5*kf*u*u if abs(u) <= F/kf else F*abs(u) - .5*F*F/kf
                return .5*mf*(s - s_pred)**2 + hub
            ref = minimize_scalar(E, bracket=(-.1, .1), tol=1e-12).x
            self.assertAlmostEqual(finger_axis_minimum(mf, s_pred, kf, s_cmd, F), ref, delta=1e-7)

    def test_seeded_placement_is_identity_at_seed_0_and_within_ranges(self):
        import math
        from gift_bags import variants as V
        for preset, ranges in V.POSE_RANGES.items():
            self.assertTrue(all(v == (0., 0., 0.) for v in V.placement(preset, 0).values()))
            for seed in (1, 17, 9999):
                for item, (dx, dy, yaw) in V.placement(preset, seed).items():
                    r = ranges[item]
                    self.assertLessEqual(abs(dx), r["dx"]); self.assertLessEqual(abs(dy), r["dy"])
                    self.assertLessEqual(abs(math.degrees(yaw)), r["yaw_deg"] + 1e-9)
            self.assertNotEqual(V.placement(preset, 1), V.placement(preset, 2))

    def test_graded_tote_mesh_keeps_creases_isometric_and_has_handle_holes(self):
        from collections import Counter
        from gift_bags import bagnet as B, ipcworld as W
        X, T, kinds, s, holes = B.bag_mesh(10, graded=True, holes=True)
        flat, layer = B.facet_map(X, T, kinds)
        E = W.edges_of(T)
        self.assertLess(np.abs(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
                               - np.linalg.norm(flat[E[:, 0]] - flat[E[:, 1]], axis=1)).max(), 1e-9)
        c = Counter(B.crease_edges(X, T, s).values())
        n_diag = len(B.graded_half(B.G_BAG, 10))*4                   # every graded cell on a diagonal is a crease edge
        self.assertEqual((c["bottom_diagonal"], c["gusset_diagonal"]), (n_diag, n_diag))
        self.assertAlmostEqual(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1).min(), B.FOLD_ROW, places=6)
        self.assertEqual({f: len(v) for f, v in holes.items()}, {"front": 2, "back": 2})
        self.assertEqual(len(T), len(B.bag_mesh(10, graded=True)[1]) - 8)       # four holes of two triangles each

    def test_contact_world_source_is_pinned_to_its_physics_version(self):
        """Start states are cached by params.CONTACT_WORLD_VERSION, not by ipcworld.py's source: an edit to ipcworld.py
        must either bump the version (physics changed) or register its hash under the current version with evidence
        that results are unchanged."""
        import hashlib
        from gift_bags import params as Pm
        sha = hashlib.sha256((PACKAGES/"gift_bags"/"ipcworld.py").read_bytes()).hexdigest()[:16]
        self.assertIn(sha, Pm.CONTACT_WORLD_SHA.get(Pm.CONTACT_WORLD_VERSION, []),
                      f"gift_bags/ipcworld.py changed ({sha}): bump params.CONTACT_WORLD_VERSION or register the hash")

    def test_tape_quadrature_tiles_each_triangle(self):
        from gift_bags.tape_bonds import subdivision_points
        for m in (1, 2, 4):
            b, f = subdivision_points(m)
            self.assertEqual(len(b), m*m)
            self.assertAlmostEqual(f.sum(), 1.)
            np.testing.assert_allclose(b.sum(1), 1.)
            self.assertTrue((b > 0).all())
            np.testing.assert_allclose(b.mean(0), [1/3]*3, atol=1e-12)        # the centroids average to the centroid

    def test_rope_end_is_part_of_its_knot_and_not_in_contact_with_it(self):
        # a rope whose end node sits inside its knot: refused as an intersection unless the node is declared part of
        # the knot (Rod.attached), which is how a handle's rope is threaded through its hole (handles.author)
        import numpy as np
        from gift_bags import ipcworld as W, handles as Hd, params as Pm, scene as S
        V, F = W.box_mesh(Hd.KNOT_HALF)
        pts = np.c_[np.zeros(8), np.zeros(8), .5 + .006*np.arange(8)]

        def world(attached):
            knot = W.Body("knot_front_0", V, F, mass=Hd.KNOT_MASS, S=np.eye(3)*1e-9, kind="dynamic", p=np.array([0., 0., .5]),
                          R=np.eye(3), mu=.4, fmat="rope")
            rod = W.Rod("handle_front", pts.copy(), Hd.ROPE, fmat="rope", attached=attached)
            sheet = S.grid_sheet("bag", .02, .02, .01, Pm.TOTE, lambda X: np.c_[X[:, 0] + .2, X[:, 1], np.full(len(X), .5)])
            sheet.fmat = "laminate"
            return W.IPCWorld([sheet], [rod], [knot], mu_pairs=Pm.FRICTION, friction_strict=True)
        with self.assertRaises(ValueError):
            world(())
        w = world(((0, "knot_front_0"), (1, "knot_front_0")))
        self.assertEqual(len(w.rods[0].attached), 2)

    def test_handle_rope_passes_through_its_hole(self):
        # each rope starts at its knot's centre, inside the bag, and its next node is on the other side of the paper
        import numpy as np
        from gift_bags import bagnet as B, handles as Hd, scene as S, ipcworld as W
        st = B.shipping_state_status(S.BAG_CELLS)
        if not (st["cached"] and st["sound"]):
            self.skipTest("the tote's shipping state is not cached on this machine")
        bag, hinges, membrane, _ = B.shipping_state((0., 0., 0.), cells_per_gusset=S.BAG_CELLS, seed=0)
        support_z = float(np.asarray(bag.x0)[:, 2].min()) - W.contact_caliper(B.Pm.TOTE)["rest_sheet_rigid"]
        x, ropes, centres, welds = Hd.author(bag, support_z)
        sites = Hd.hole_sites(bag, x)
        holes = [c for c, _ in sites["front"]] + [c for c, _ in sites["back"]]
        normals = [n for _, n in sites["front"]] + [n for _, n in sites["back"]]
        for ri, vi, ki, loc in welds:
            R = ropes[ri]
            first, second = (R[0], R[1]) if vi == 0 else (R[-1], R[-2])
            np.testing.assert_allclose(first, centres[ki], atol=1e-9)        # the rope's end is in its knot
            np.testing.assert_allclose(loc, 0.)
            n = normals[ki]/np.linalg.norm(normals[ki])
            self.assertLess(((first - holes[ki]) @ n)*((second - holes[ki]) @ n), 0.)   # opposite sides of the paper
            lateral = (second - holes[ki]) - ((second - holes[ki]) @ n)*n
            self.assertLess(np.linalg.norm(lateral), .004)                    # through the hole, not beside it

    def test_knot_cannot_pass_its_hole(self):
        from gift_bags import handles as Hd
        self.assertTrue(Hd.knot_cannot_pass(.01016))
        self.assertFalse(Hd.knot_cannot_pass(.020))


    def test_receiving_friction_pairs_are_in_the_ledger(self):
        """Every material pair the receiving world can bring into contact has a ledger coefficient (the world refuses to
        build otherwise; this catches it without building)."""
        import itertools
        from gift_bags import params as Pm
        mats = ("corrugated", "packtape", "steel", "carton", "glass", "finger", "link", "phantom")
        missing = [p for p in itertools.combinations_with_replacement(mats, 2) if frozenset(p) not in Pm.FRICTION]
        self.assertEqual(missing, [])

    def test_carton_geometry_and_aged_crease(self):
        from gift_bags import carton as CT
        m = CT.masses()
        total = m["body"] + 2*m["major"] + 2*m["minor"]
        self.assertTrue(.29 < total < .35, total)                       # ~0.31 kg empty at 0.55 kg/m2
        self.assertAlmostEqual(2*CT.FLAP_D + CT.SEAM, CT.W_OUT, places=9)  # the majors meet at the seam
        tp, tr, hist = CT.aged_crease_state()
        self.assertTrue(80. < np.degrees(tp) < 90., np.degrees(tp))     # a day's hold sets most of the 90 deg fold
        self.assertLess(hist[-1][1], hist[0][1])                         # the held moment relaxes
        X2, T, tm, info = CT.tape_net()
        self.assertTrue(np.all(np.isin(np.round([CT.TAB, CT.LEG, CT.LEG + info["l_top"]], 9), np.round(info["u"], 9))))
        self.assertTrue(np.all(tm[np.asarray(X2)[T].mean(1)[:, 0] < CT.TAB] == 1))

    def test_carton_tape_bond_derivatives(self):
        """BodyBonds energy, gradient and Hessian agree with finite differences on a tape patch bonded to a dynamic box."""
        from gift_bags import ipcworld as W, params as Pm, carton as CT, scene as S
        sh = S.grid_sheet("seam_tape", .02, .012, .004, Pm.BOPP_TAPE,
                          lambda X: np.c_[X[:, 0] - .01, X[:, 1] - .006, np.full(len(X), .0102)], jitter=0.)
        box = S.box_body("plate", (.03, .03, .005), (0., 0., .005), .05, fmat="corrugated")
        w = W.IPCWorld([sh], [], [box], mu_pairs=Pm.FRICTION, friction_strict=False)
        x0 = w.x_free().copy()
        n = len(x0)
        bonds = CT.BodyBonds(np.arange(n), np.zeros(n, int), x0 - box.p, np.full(n, 1.6e-5), Pm.TAPE_ON_KRAFT)
        rng = np.random.default_rng(3)
        q = w.q.copy()
        q[:3*n] += rng.normal(0., 3e-6, 3*n)                     # every opening below delta0 (30 um): the elastic branch,
        o = w.body_dof[0]                                        # where the Newton matrix is the exact Hessian
        q[o:o+12] += rng.normal(0., 2e-6, 12)
        E, g, coo = bonds.term(w, q, None, 4e-3, True)
        self.assertLess(bonds.openings(w, q).max(), Pm.TAPE_ON_KRAFT.delta0)
        import scipy.sparse as sp
        H = sp.csr_matrix((np.concatenate(coo[2]), (np.concatenate(coo[0]), np.concatenate(coo[1]))), shape=(w.n_dof,)*2)
        d = rng.normal(0., 1., w.n_dof)
        h = 1e-8
        Ep, gp, _ = bonds.term(w, q + h*d, None, 4e-3, False)
        Em, gm, _ = bonds.term(w, q - h*d, None, 4e-3, False)
        self.assertAlmostEqual((Ep - Em)/(2*h), float(g @ d), delta=1e-6*max(1., abs(float(g @ d))))
        np.testing.assert_allclose((gp - gm)/(2*h), H @ d, rtol=1e-5, atol=1e-6*np.abs(H @ d).max())
        # softening branch (openings beyond delta0, and beyond an earlier dmax): the gradient is the envelope's traction,
        # the energy is continuous at the history point, and a full opening stores exactly Gamma per area
        law = CT.tape_law()
        b2 = CT.BodyBonds(np.arange(n), np.zeros(n, int), x0 - box.p, np.full(n, 1.6e-5), law)
        b2.dmax[:] = rng.uniform(0., .5*law.deltaf, n)
        q2 = w.q.copy()
        q2[:3*n] += rng.normal(0., .3*law.deltaf, 3*n)
        E2, g2, coo2 = b2.term(w, q2, None, 4e-3, True)
        d2 = rng.normal(0., 1., w.n_dof)
        h2 = 1e-9
        self.assertAlmostEqual((b2.term(w, q2 + h2*d2, None, 4e-3, False)[0] - b2.term(w, q2 - h2*d2, None, 4e-3, False)[0])/(2*h2),
                               float(g2 @ d2), delta=1e-5*abs(float(g2 @ d2)) + 1e-9)
        H2 = sp.csr_matrix((np.concatenate(coo2[2]), (np.concatenate(coo2[0]), np.concatenate(coo2[1]))), shape=(w.n_dof,)*2)
        self.assertGreaterEqual(np.linalg.eigvalsh(H2.toarray()).min(), -1e-6*abs(H2).max())       # Newton matrix is PSD
        b3 = CT.BodyBonds(np.arange(n), np.zeros(n, int), x0 - box.p, np.full(n, 1.6e-5), law)
        q3 = w.q.copy()
        q3[2:3*n:3] += 2*law.deltaf
        self.assertAlmostEqual(b3.term(w, q3, None, 4e-3, False)[0], law.Gamma*1.6e-5*n, delta=1e-9)
        w.q = q3
        b3.after(w, 4e-3)
        self.assertAlmostEqual(b3.dissipated, law.Gamma*1.6e-5*n, delta=1e-9)
        self.assertEqual(b3.intact_fraction, 0.)


class ConstructedCounter:
    """Constructed counter states judged by task.Evaluator without physics, over a stub contact world: the tote is
    bagnet.build()'s counter mesh (3,219 vertices, as the shipped start states) at its open rest shape, or the shipped
    start state; items are poses; contacts use the contact caliper's rest distances; nothing is checked for
    equilibrium. A state here is constructed, not executed."""

    def __init__(self):
        from gift_bags import bagnet as B, ipcworld as W, params as Pm, scene as S, task as T
        self.S, self.T = S, T
        self.cal = W.contact_caliper(Pm.TOTE)
        self.sheet = B.build((0., 0., 0.), cells_per_gusset=S.BAG_CELLS)
        self.X = np.asarray(self.sheet.X2, float)
        self.H, self.G, self.W = B.H_BAG, B.G_BAG, B.W_BAG
        z = S.COUNTER_Z
        self.mat_top = z + 2*S.MAT_HALF[2]
        self.cx, self.cy = S.MAT_XY
        self.geo = dict(mat=dict(center=(self.cx, self.cy, self.mat_top), half=S.MAT_HALF[:2]),
                        board=dict(half=S.BOARD_HALF), candle=dict(half=S.CANDLE_HALF))
        self.hb, self.hc = np.asarray(S.BOARD_HALF), np.asarray(S.CANDLE_HALF)
        self.mid0 = self.mat_top - self.cal["deflate"] + self.cal["rest_sheet_rigid"]   # paper resting on the mat
        start = np.load(ROOT/"gift_bags"/"start_states"/"flat_tote_handles_7e393930585ec35e.npz")
        self.start_x = np.asarray(start["x"], float)
        bz, _ = S._stack_z(z, *([2*self.hb[2]]*3))
        self.initial = {f"stack_bag_{k}": (start["support_p"][k + 1], start["support_R"][k + 1]) for k in range(4)}
        self.initial.update({f"board_{k}": (np.array([*S.BOARDS_XY, zc]), np.eye(3)) for k, zc in enumerate(bz)})
        self.initial["candle"] = (np.array([*S.CANDLE_XY_PURCHASE, z + S.CLEAR + self.hc[2]]), np.eye(3))
        self.mouth0 = 0.
        self.mouth0 = self.evaluator(self.start_x, {})._mouth(self.start_x)[0]    # as Evaluator.__init__ stores it
        self.open = self.place(self.X)

    def place(self, X, dz=0.):
        return X + np.array([self.cx, self.cy, self.mid0 + dz])

    def fold(self, theta_of_z):
        """The rest shape with both gussets folded inward to opening theta(z) (180 flat) along an isometric fold."""
        x, X = self.X.copy(), self.X
        a = np.radians(np.array([theta_of_z(z) for z in X[:, 2]]))/2
        gus = np.abs(np.abs(X[:, 0]) - self.W/2) < 1e-9
        face = (np.abs(np.abs(X[:, 1]) - self.G/2) < 1e-9) & ~gus
        s = self.G/2 - np.abs(X[:, 1])
        x[gus, 0] = np.sign(X[gus, 0])*(self.W/2 - s[gus]*np.cos(a[gus]))
        x[gus | face, 1] = X[gus | face, 1]*np.sin(a[gus | face])
        return x

    def board_on(self, z_paper=None, dx=0., dy=0., yaw_deg=0.):
        from scipy.spatial.transform import Rotation
        z_paper = self.mid0 if z_paper is None else z_paper
        return (np.array([self.cx + dx, self.cy + dy, z_paper + self.cal["rest_sheet_rigid"] + self.hb[2]]),
                Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix())

    def candle_on(self, board, dx=0., R=np.eye(3)):
        p = board[0]
        return np.array([p[0] + dx, p[1], p[2] + self.hb[2] + self.cal["rest_rigid_rigid"] + self.hc[2]]), R

    def evaluator(self, x, moved, task="bag-purchase", crack_mm=0.):
        T = self.T
        poses = {n: (np.array(p, float), np.array(R, float)) for n, (p, R) in self.initial.items()}
        poses.update(moved)
        names = list(poses)
        ipc = SimpleNamespace(sheets=[self.sheet], sheet_slices=[slice(0, len(x))], caliper=self.cal, reactions={},
                              stats={}, damage=np.zeros(0), bodies=[SimpleNamespace(name=n, kind="dynamic") for n in names],
                              x_free=lambda: np.asarray(x, float), body_pose=lambda i: poses[names[i]],
                              crack_length=lambda k: (1e-3*crack_mm, 0.))
        ev = T.Evaluator.__new__(T.Evaluator)
        ev.env = SimpleNamespace(config=SimpleNamespace(dt=.004, task=task))
        ev.world = SimpleNamespace(ipc=ipc, scene=SimpleNamespace(preset="counter", geometry=self.geo,
                                                                  surface_z=self.S.COUNTER_Z), sim_time=0.)
        ev.task, ev.geo = task, self.geo
        ev.body = {n: i for i, n in enumerate(names)}
        ev.initial = {n: np.array(p, float) for n, (p, _) in self.initial.items()}
        ev.initial_R = {n: np.array(R, float) for n, (_, R) in self.initial.items()}
        ev.history = T.History(.004)
        ev.crack0 = {0: 0.}
        ev._bag_setup()
        ev.mouth0 = self.mouth0
        ev.rim_gap0 = ev._rim_gap(self.start_x)              # as Evaluator.__init__ stores it
        return ev

    def evaluate(self, x, moved=None, task="bag-purchase", window_s=3., contact_rows=0, crack_mm=0.):
        """The hands-off history as env.act records it: `contact_rows` rows (from the window's start) touched."""
        ev = self.evaluator(x, moved or {}, task, crack_mm)
        base = np.asarray(x)[ev.bottom_v].mean(0)
        for k in range(int(round(window_s/.004))):
            ev.history.add(dict(t=k*.004, contact=k < contact_rows, base=base), {})
        return ev.evaluate()


class BagPurchaseVerifierTest(unittest.TestCase):
    """bag-purchase's verifier and score on constructed states (no physics, seconds); gift_bags/REPORT.pdf §7."""

    @classmethod
    def setUpClass(cls):
        cls.c = ConstructedCounter()

    def judge(self, x, moved=None, **kw):
        r = self.c.evaluate(x, moved, **kw)
        self.assertEqual(r["success"], r["score"] == 1.)          # every success scores 1, every failure below it
        self.assertTrue(0. <= r["score"] <= 1.)
        return r, [k for k, v in r["checks"].items() if not v]

    def test_valid_completions_succeed_whatever_the_route(self):
        c = self.c
        board = c.board_on()
        done = {"board_0": board, "candle": c.candle_on(board, dx=-.040)}           # candle 40 mm off-centre
        for name, x, moved in (
                ("done", c.open, done),
                ("board yawed 0.8 deg", c.open, {"board_0": c.board_on(yaw_deg=.8), "candle": c.candle_on(board, dx=-.04)}),
                ("another board", c.open, {"board_2": board, "candle": c.candle_on(board)}),
                ("two boards, candle on the top one", c.open,
                 {"board_0": board, "board_1": (board[0] + [0., 0., 2*c.hb[2] + c.cal["rest_rigid_rigid"]], board[1]),
                  "candle": c.candle_on((board[0] + [0., 0., 2*c.hb[2] + c.cal["rest_rigid_rigid"]], board[1]))}),
                ("stack nudged 25 mm", c.open,                              # a nudged stack costs nothing
                 {**done, "stack_bag_3": (c.initial["stack_bag_3"][0] + [0., .025, 0.], np.eye(3))})):
            with self.subTest(name):
                r, failed = self.judge(x, moved)
                self.assertEqual(failed, [])
                self.assertEqual(r["metrics"]["progress"], 1.)
        r, _ = self.judge(c.open, done)
        self.assertEqual(set(r["checks"]), {"hands_off_window", "mouth_open", "standing", "bottom_squared", "on_mat",
                                            "base_at_rest", "board_in_bag", "candle_in_bag", "candle_upright",
                                            "candle_on_board", "bag_not_torn"})       # gussets reported only
        self.assertLess(r["metrics"]["candle_above_board_mm"], .1)

    def test_non_solutions_fail(self):
        from scipy.spatial.transform import Rotation
        c = self.c
        board = c.board_on()
        # a board on the mat under the bag, the bag standing on it, the candle on the bag's paper
        under = (np.array([c.cx, c.cy, c.mat_top - 2*c.cal["deflate"] + c.cal["rest_rigid_rigid"] + c.hb[2]]), np.eye(3))
        z_bag = under[0][2] + c.hb[2] - c.cal["deflate"] + c.cal["rest_sheet_rigid"]
        on_board = c.place(c.X, dz=z_bag - c.mid0)
        drooped = on_board.copy()
        off = (np.abs(c.X[:, 2]) < 1e-9) & ((np.abs(c.X[:, 0]) > c.hb[0]) | (np.abs(c.X[:, 1]) > c.hb[1]))
        drooped[off, 2] = c.mid0
        paper_candle = (np.array([c.cx - .04, c.cy, z_bag + c.cal["rest_sheet_rigid"] + c.hc[2]]), np.eye(3))
        waist = lambda z: 180. - 65.*min(1., max(0., (z - .085)/.01)) if z < .110 else 115. + 65.*min(1., max(0., (z - .125)/.01))
        seat = .125 + .010*(2*np.degrees(np.arcsin(2*c.hc[1]/c.G)) - 115.)/65.
        cases = {
            "board under the bag": (on_board, {"board_0": under, "candle": paper_candle}, {"board_in_bag", "candle_on_board"}),
            "board under, bottom drooped to the mat": (drooped, {"board_0": under, "candle": paper_candle},
                                                       {"board_in_bag", "candle_on_board"}),
            "candle upside down on the board": (c.open, {"board_0": board, "candle": c.candle_on(
                board, R=Rotation.from_euler("x", 180, degrees=True).as_matrix())}, {"candle_upright"}),
            "candle upright 130 mm above the board": (c.open, {"board_0": board, "candle": (
                c.candle_on(board)[0] + [0., 0., .130], np.eye(3))}, {"candle_on_board"}),
            "candle seated on a waist of the gussets": (c.place(c.fold(waist)), {"board_0": board, "candle": (
                np.array([c.cx, c.cy, c.mid0 + seat + c.hc[2]]), np.eye(3))}, {"candle_on_board"}),
            "candle lying on its side on the board": (c.open, {"board_0": board, "candle": (
                np.array([c.cx, c.cy, board[0][2] + c.hb[2] + c.hc[0]]), Rotation.from_euler("y", 90, degrees=True).as_matrix())},
                {"candle_upright"}),
        }
        for name, (x, moved, expect) in cases.items():
            with self.subTest(name):
                r, failed = self.judge(x, moved)
                self.assertFalse(r["success"])
                self.assertTrue(expect <= set(failed), failed)
        # the bag inverted over the candle (cup over the ball): failed, and almost no credit
        inv = np.c_[c.X[:, 0], -c.X[:, 1], c.H - c.X[:, 2]] + np.array([c.cx, c.cy, c.mid0])
        r, failed = self.judge(inv, {"candle": (np.array([c.cx, c.cy, c.mat_top + c.cal["rest_rigid_rigid"] + c.hc[2]]),
                                                np.eye(3))})
        self.assertTrue({"standing", "bottom_squared", "board_in_bag", "candle_on_board"} <= set(failed), failed)
        self.assertLess(r["score"], .15)

    def test_score_reads_progress_of_the_end_state(self):
        from scipy.spatial.transform import Rotation
        c = self.c
        board = c.board_on()
        done = {"board_0": board, "candle": c.candle_on(board)}
        untouched, _ = self.judge(c.start_x)
        self.assertEqual((untouched["score"], untouched["metrics"]["progress"]), (0., 0.))
        # the unopened tote slid onto the mat is not yet any part of the end state
        flat = c.start_x - np.r_[c.start_x[:, :2].mean(0), c.start_x[:, 2].min()] + np.array([c.cx, c.cy, c.mid0])
        self.assertLess(self.judge(flat)[0]["score"], .02)
        ladder = [untouched["score"]] + [self.judge(x, m)[0]["score"] for x, m in (
            (c.place(c.fold(lambda z: 180. - 120.*min(1., max(0., (z - .02)/.03)))), {}),   # standing, mouth narrow
            (c.open, {}),                                                      # open, standing, squared
            (c.open, {"board_0": board}),                                      # + the board
            (c.open, {"board_0": board, "candle": (np.array([c.cx, c.cy, board[0][2] + c.hb[2] + c.hc[0]]),
                                                   Rotation.from_euler("y", 90, degrees=True).as_matrix())}),  # candle on its side
            (c.open, done))]                                                   # the job done
        self.assertEqual(ladder[-1], 1.)
        self.assertTrue(all(a < b for a, b in zip(ladder, ladder[1:])), ladder)
        # damage: a torn tote never scores above the same bag untorn; from 25 mm it is worth nothing
        torn = [self.judge(c.open, done, crack_mm=mm)[0] for mm in (0., 6., 11.2, 25., 79.2)]
        self.assertEqual([t["success"] for t in torn], [True, False, False, False, False])
        self.assertTrue(all(a["score"] > b["score"] for a, b in zip(torn, torn[1:-1])))
        self.assertEqual((torn[3]["score"], torn[4]["score"]), (0., 0.))
        # not released: an item still touched in the window keeps its progress but not success
        held, failed = self.judge(c.open, done, contact_rows=750)
        self.assertEqual((failed, held["score"]), (["hands_off_window"], self.c.T.FAILED_SCORE_FACTOR))
        # the opened bag knocked onto its back on the mat: only its openness counts
        back = np.c_[c.X[:, 0], c.X[:, 2] - c.H/2, -c.X[:, 1]] + np.array([c.cx, c.cy, c.mid0 + c.G/2])
        knocked, _ = self.judge(back)
        self.assertLess(knocked["score"], .15)
        self.assertEqual(knocked["metrics"]["reward"]["mouth_up"], 0.)

    def test_one_open_rule_and_the_pinch(self):
        """One rule for "open": the mouth (hull ratio), not pinched at the middle of the rim; the gusset creases are
        reported, not judged. Gussets drawn in to 110-119 deg above 130 mm (mouth 83-88 mm) with the job done pass
        wherever the fold starts; 100 deg (78 mm) fails through the mouth; a mouth pinched to 5 mm at its middle fails
        although its hull reads 1.0; a mouth folded nearly shut keeps little of "opened"."""
        c = self.c
        board = c.board_on()
        done = {"board_0": board, "candle": c.candle_on(board)}
        lin = lambda z, z0, z1, t0, t1: t0 + (t1 - t0)*min(1., max(0., (z - z0)/(z1 - z0)))
        for th in (110., 115., 119.):
            for h in (.130, .180):
                with self.subTest(theta=th, h=h):
                    r, failed = self.judge(c.place(c.fold(lambda z: lin(z, h - .02, h, 180., th))), done)
                    self.assertEqual(failed, [])
        r, failed = self.judge(c.place(c.fold(lambda z: lin(z, .11, .13, 180., 100.))), done)
        self.assertEqual(failed, ["mouth_open"])
        X = c.X.copy()
        face = (np.abs(np.abs(X[:, 1]) - c.G/2) < 1e-9) & (np.abs(np.abs(X[:, 0]) - c.W/2) > 1e-9)
        s = np.clip((X[:, 2] - .15)/(c.H - .15), 0., 1.)*np.clip(1. - np.abs(X[:, 0])/(c.W/2), 0., 1.)
        X[face, 1] *= 1. - .95*s[face]
        r, failed = self.judge(c.place(X), done)
        self.assertGreater(r["metrics"]["mouth_ratio"], .95)
        self.assertLess(r["metrics"]["face_gap_min_mm"], 10.)
        self.assertEqual(failed, ["mouth_open"])
        shut = c.place(c.fold(lambda z: lin(z, .13, .15, 180., 28.)))
        r, failed = self.judge(shut, done)
        self.assertLess(r["metrics"]["reward"]["opened"], .3)

    def test_release_window_counts_only_the_jobs_items(self):
        """A robot link resting on the untouched stock (the lower totes, the board pile) does not fail the release;
        one touching the tote, the candle or a board taken from the pile does."""
        c = self.c
        board = c.board_on()
        ev = c.evaluator(c.open, {"board_0": board, "candle": c.candle_on(board)})
        ipc = ev.world.ipc
        box = lambda h: np.array([[sx*h[0], sy*h[1], sz*h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        names = [b.name for b in ipc.bodies]
        poses = {n: ipc.body_pose(i) for i, n in enumerate(names)}
        bodies = [SimpleNamespace(name=n, kind="dynamic", verts=box(c.hc if n == "candle" else c.hb)) for n in names]
        bodies.append(SimpleNamespace(name="robot:right/fr3/fr3_link7", kind="driven", verts=box((.02, .02, .02))))
        ipc.bodies, link = bodies, len(bodies) - 1
        ipc.body_pose = lambda i: poses[names[i]] if i < len(names) else poses["robot"]
        ipc.reactions = {link: (np.array([0., 0., 2.]), np.zeros(3))}
        ev.body = {n: i for i, n in enumerate(names)}
        top = lambda p, hz: (np.array([p[0], p[1], p[2] + hz + .02]), np.eye(3))
        sp, cp = c.initial["stack_bag_3"][0], poses["candle"][0]
        rim = c.open[ev.rim_pairs[0, 0]]                                          # a front rim vertex
        for where, pose, counts in (("on the remaining stack", top(sp, .00125), False),
                                    ("on the board pile", top(c.initial["board_2"][0], c.hb[2]), False),
                                    ("on the candle", top(cp, c.hc[2]), True),
                                    ("on the tote's rim", (rim + [0., 0., .021], np.eye(3)), True)):
            with self.subTest(where):
                poses["robot"] = pose
                ev.history.rows.clear()
                ev.record()
                self.assertEqual(ev.history.rows[-1]["contact"], counts)
                self.assertTrue(ev.history.rows[-1].get("contact_any", True))
        poses["board_2"] = (c.initial["board_2"][0] + [.1, 0., 0.], np.eye(3))      # a board taken from the pile
        poses["robot"] = top(poses["board_2"][0], c.hb[2])
        ev.record()
        self.assertTrue(ev.history.rows[-1]["contact"])
        # a lower tote leaning 5 mm from the bag's face, pressed by a link 74 mm from the bag
        y = c.cy - c.G/2 - .0545
        poses["stack_bag_3"] = (np.array([c.cx, y, c.mid0 + .1]), np.eye(3))
        poses["robot"] = (np.array([c.cx, y - .04, c.mid0 + .1 + c.hb[2] + .02]), np.eye(3))
        ev.record()
        self.assertTrue(ev.history.rows[-1]["contact"])

    def test_body_pinch_and_settled_rest(self):
        """A body pinched shut below an open rim fails "open"; a bag that shifts once as the carton lands at the start
        of the window, then stands still, is at rest, one still sliding is not; a carton resting on a board just short
        of the board rule keeps that board's credit for "on"."""
        c = self.c
        board = c.board_on()
        done = {"board_0": board, "candle": c.candle_on(board)}
        th = lambda z: 180. if z < .125 else 180. - 170.*min(1., (z - .125)/.025) if z < .190 else 10. + 170.*min(1., (z - .190)/.025)
        r, failed = self.judge(c.place(c.fold(th)), done)
        self.assertEqual(failed, ["mouth_open"])
        self.assertLess(r["metrics"]["face_gap_min_mm"], 20.)
        for shift, ok in ((lambda t: min(t, .36)/.36*3.86e-3, True), (lambda t: 1e-3*t, False)):
            ev = c.evaluator(c.open, done)
            base = np.asarray(c.open)[ev.bottom_v].mean(0)
            for k in range(750):
                ev.history.add(dict(t=k*.004, contact=False, base=base + [shift(k*.004), 0., 0.]), {})
            r = ev.evaluate()
            self.assertEqual(r["checks"]["base_at_rest"], ok)
            self.assertEqual(r["success"], ok)
        raised = (board[0] + [0., 0., .0045], board[1])
        r, failed = self.judge(c.open, {"board_0": raised, "candle": c.candle_on(raised)})
        self.assertEqual(set(failed), {"board_in_bag", "candle_on_board"})
        self.assertGreater(r["metrics"]["candle_on_board_credit"], .4)
        self.assertEqual(r["metrics"]["candle_on_board_credit"], r["metrics"]["board_credit"])

    def test_open_bag_keeps_its_own_checks_and_score(self):
        """open-bag shares _eval_bag but keeps its own checks and its score (the mean of its checks)."""
        c = self.c
        r = c.evaluate(c.open, {"board_0": c.board_on(),
                                "stack_bag_3": (c.initial["stack_bag_3"][0] + [0., .025, 0.], np.eye(3))}, task="open-bag")
        self.assertEqual(list(r["checks"]), ["hands_off_window", "mouth_open", "standing", "bottom_squared", "on_mat",
                                             "gusset_left_deployed", "gusset_right_deployed", "base_at_rest",
                                             "board_in_bag", "stack_undisturbed", "bag_not_torn"])
        self.assertFalse(r["checks"]["stack_undisturbed"])
        self.assertEqual(r["score"], np.mean(list(r["checks"].values())))
        for k in ("progress", "reward", "mean_of_checks", "board_credit", "bag_tilt_signed_deg"):
            self.assertNotIn(k, r["metrics"])

    def test_final_evaluation_completes_the_hands_off_window_on_a_restored_branch(self):
        """GiftBagsEnv.final_evaluation: the window the agent did not wait out is advanced with the robot
        holding, judged and undone; not at the start state, not for other tasks; a contact in the branch stops it."""
        import threading
        from gift_bags import env as E, task as T

        class Ev:
            def __init__(self, world, touching_until):
                self.history, self.world, self.touch = T.History(.004), world, touching_until

            def record(self):
                self.history.add(dict(contact=self.world.steps <= self.touch), {})

            def evaluate(self):
                rows = list(self.history.rows)
                ok = len(rows) >= self.history.n and not any(r["contact"] for r in rows)
                return dict(success=ok, score=float(ok), checks=dict(hands_off_window=ok),
                            metrics=dict(window_rows=len(rows), at_step=self.world.steps))

        def env_for(task, touching_until=0, fault_at=None):
            env = E.GiftBagsEnv.__new__(E.GiftBagsEnv)
            env.lock, env.generation = threading.RLock(), 0
            env.config = SimpleNamespace(task=task, dt=.004)
            env.world = SimpleNamespace(fault=None, steps=0, timing={})

            def step():
                if fault_at is not None and env.world.steps + 1 == fault_at:
                    raise FloatingPointError("non-finite")
                env.world.steps += 1
            env.world.step = step
            env.evaluator = Ev(env.world, touching_until)
            env.checkpoint = lambda: (env.world.steps, list(env.evaluator.history.rows))
            env.restore = lambda s: (setattr(env.world, "steps", s[0]), env.evaluator.history.rows.clear(),
                                     env.evaluator.history.rows.extend(s[1]))

            def act(n):
                for _ in range(n):
                    env.world.step()
                    env.evaluator.record()
            env.act_ticks = act
            return env

        env = env_for("bag-purchase")
        self.assertEqual(env.final_evaluation()["metrics"]["at_step"], 0)          # the start state: not advanced
        env.act_ticks(250)                                                           # released 1 s before the end
        r = env.final_evaluation()
        self.assertTrue(r["success"])
        self.assertEqual((r["metrics"]["at_step"], r["metrics"]["settle_branch_s"]), (750, 2.))
        self.assertEqual((env.world.steps, len(env.evaluator.history.rows)), (250, 250))   # restored
        env.world.step = None                                                        # judged once per (tick, generation)
        self.assertEqual(env.final_evaluation(), r)
        env = env_for("bag-purchase", touching_until=10**9)                          # still holding an item
        env.act_ticks(100)
        r = env.final_evaluation()
        self.assertFalse(r["success"])
        self.assertEqual((r["metrics"]["settle_branch_s"], env.world.steps), (.004, 100))
        env = env_for("bag-purchase", touching_until=100)                            # let go at 0.4 s, then waited 3 s
        env.act_ticks(850)
        r = env.final_evaluation()
        self.assertEqual((r["success"], r["metrics"]["settle_branch_s"]), (True, 0.))
        env = env_for("open-bag")                                                    # other tasks: not advanced
        env.act_ticks(250)
        self.assertEqual(set(env.final_evaluation()["metrics"]), {"window_rows", "at_step"})
        env = env_for("bag-purchase", fault_at=300)
        env.act_ticks(250)
        with self.assertRaises(RuntimeError):
            env.final_evaluation()
        self.assertEqual(env.world.steps, 250)


@unittest.skipUnless(os.environ.get("GIFT_BAGS_PHYSICS_TESTS") == "1", "set GIFT_BAGS_PHYSICS_TESTS=1 (minutes, GPU)")
class GiftBagsPhysicsTest(unittest.TestCase):
    def test_restore_is_exact_and_evaluator_controls(self):
        from gift_bags.config import Config
        from gift_bags.env import GiftBagsEnv
        env = GiftBagsEnv(Config(preset="home", task="lift-lid"))
        snap = env.checkpoint()
        q = env.world.ipc.q.copy()
        env.act({}, ticks=10)
        env.restore(snap)
        self.assertTrue(np.array_equal(env.world.ipc.q, q))
        ev = env.evaluator.evaluate()
        self.assertFalse(ev["success"])
        self.assertFalse(ev["checks"]["lid_off"])            # the lid starts seated
        self.assertTrue(ev["checks"]["frame_in_base"])

    def test_sheet_rests_at_one_caliper(self):
        from gift_bags import ipcworld as W, params as Pm, scene as S
        cal = W.contact_caliper(Pm.WRAP)
        place = lambda X2: np.c_[X2[:, 0], X2[:, 1], np.full(len(X2), cal["rest_sheet_rigid"] + 2e-5)]
        sh = S.grid_sheet("wrap", .1, .08, .01, Pm.WRAP, place)
        table = S.furniture_body(dict(name="table", center=(.05, .04, -.02), half=(.2, .2, .02)))
        w = W.IPCWorld([sh], [], [table])
        for _ in range(60):
            w.step(4e-3)
        z = w.x_free()[w.sheet_slices[0], 2]         # the sheet's vertices (x_free also holds inactive spare slots)
        self.assertTrue(np.allclose(z, cal["rest_sheet_rigid"], atol=2e-6))
        self.assertEqual(len(z), len(sh.X2))

    def test_friction_table_equals_the_ledger(self):
        from gift_bags.config import Config
        from gift_bags import scene as S, params as Pm
        from gift_bags.world import World
        for preset, task in (("counter", "open-bag"), ("home", "lift-lid"), ("receiving", "open-carton")):
            cfg = Config(preset=preset, task=task)
            ipc = World(S.build(cfg), cfg).ipc
            names = [c["name"] for c in ipc._friction_classes]
            for (i, j), v in ipc._mu_table.items():
                a, b = names[i - 1], names[j - 1]
                self.assertTrue(a.startswith("mat:") and b.startswith("mat:"), (a, b))
                self.assertEqual(v, Pm.FRICTION[frozenset((a[4:], b[4:]))], (a, b))

    def test_placed_gift_box_settles_in_its_seeded_pose(self):
        import math
        from gift_bags.config import Config
        from gift_bags.env import GiftBagsEnv
        from gift_bags import variants as V
        dx, dy, yaw = V.placement("home", 7)["box"]
        env = GiftBagsEnv(Config(preset="home", task="lift-lid", seed=7))
        ev = env.get_state()["evaluation"]
        self.assertTrue(ev["checks"]["frame_in_base"] and ev["checks"]["base_not_tipped"])
        ipc = env.world.ipc
        p, R = ipc.body_pose(env.evaluator.body["gift_base"])
        self.assertAlmostEqual(math.atan2(R[1, 0], R[0, 0]), yaw, places=3)
        self.assertLess(abs(p[0] - dx) + abs(p[1] - (.17 + dy)), 1e-3)


    def _trouser(self, cell=.008, L=.096, Wd=.048, cut=.024):
        from gift_bags import ipcworld as W, params as Pm, scene as S
        sh = S.grid_sheet("wrap", L, Wd, cell, Pm.WRAP, lambda X2: np.c_[X2[:, 1] - Wd/2, X2[:, 0], np.full(len(X2), .2)], seed=1)
        w = W.IPCWorld([sh], [], [], gravity=(0., 0., 0.), contact_stock=Pm.WRAP)
        X2 = np.asarray(sh.X2)
        w.cut(0, [int(i) for i in np.argsort(X2[:, 0]) if abs(X2[i, 1] - Wd/2) < 1e-9 and X2[i, 0] <= cut + 1e-9])
        end = X2[:, 0] < 1e-9
        legs = (np.flatnonzero(end & (X2[:, 1] < Wd/2 - cell/2)), np.flatnonzero(end & (X2[:, 1] > Wd/2 + cell/2)))
        x0 = w.x_free().copy()

        def step(i):
            s = .05*(i + 1)*4e-3
            w.fixture_springs = [(legs[0], x0[legs[0]] + [0., 0., s], 5e3, 20.), (legs[1], x0[legs[1]] - [0., 0., s], 5e3, 20.)]
            w.step(4e-3)
        return w, sh, step

    def test_a_loaded_contact_below_the_friction_limit_does_not_creep_and_slides_freely_above_it(self):
        """Coulomb friction on a block on a tilted plane, at the task worlds' solver settings (friction angle 19.29
        degrees for 0.35). Below it the block comes to rest and stays (IPCWorld._stick_update; with IPC's lagged
        friction alone it creeps at eps_v tan(theta) / (2 mu): 22 um/s at 0.9 degrees). Above it the block slides
        with the ledger's coefficient (IPCWorld._contact_forces_settled; with the step-size stopping rule alone a block
        stood on a 35 degree plane at the task tolerance, and at a tenfold tighter one slid with 0.38 to 1.0 for
        0.35)."""
        import math
        from gift_bags import ipcworld as W, params as Pm, scene as S
        from gift_bags.config import Config
        cfg = Config(preset="home", task="lift-lid")
        mu, dt = .35, cfg.dt

        def slide(theta_deg, seconds, stick=True, rule=True, x0=0.):
            th = math.radians(theta_deg)
            table = S.furniture_body(dict(name="table_top", center=(0., 0., -.02), half=(1.5, .3, .02), mu=mu))
            block = S.box_body("block", (.04, .03, .02), (x0, 0., .02 + 1e-4), .25, mu=mu)
            w = W.IPCWorld([], [], [table, block], gravity=(W.G*math.sin(th), 0., -W.G*math.cos(th)),
                           contact_stock=Pm.BOX_COVER, stick_friction=stick, tol=cfg.newton_tol, newton_max=cfg.newton_max,
                           force_rtol=W.FORCE_RTOL if rule else None)
            o = w.body_dof[1]
            xs = []
            for _ in range(int(round(seconds/dt))):
                w.step(dt)
                xs.append(float(w.q[o]))
            self.assertEqual(w.stats["unconverged"], 0)
            return np.array(xs)

        def mu_effective(x, theta_deg):                            # from the acceleration over the second half
            n = len(x)
            a = 2*np.polyfit(dt*np.arange(n//2, n), x[n//2:], 2)[0]
            return math.tan(math.radians(theta_deg)) - a/(W.G*math.cos(math.radians(theta_deg)))
        self.assertIsNotNone(W.FORCE_RTOL)
        for theta in (.9, 10., 18.):                                # below the friction angle: at rest
            x = slide(theta, 2.)
            self.assertLess(abs(x[-1] - x[-126])/.5, 1e-9)          # under 0.001 um/s
            self.assertLess(x[-1] - x[0], 5e-4)                     # and it stopped within the slip of its landing
        x = slide(.9, 2., stick=False)                              # the lagged friction alone: the creep
        self.assertAlmostEqual((x[-1] - x[-126])/.5, 1e-3*math.tan(math.radians(.9))/(2*mu), delta=2e-6)
        for theta, T in ((20., .6), (25., .6), (45., .5)):          # above it: Coulomb, at 0.1 to 2 m/s
            self.assertAlmostEqual(mu_effective(slide(theta, T, x0=-1.3), theta), mu, delta=.002)
        # DIAGNOSTIC, the step-size rule alone (update under 10 um): the block does not slide on a 35 degree plane
        x = slide(35., .6, rule=False, x0=-1.3)
        self.assertLess((x[-1] - x[-51])/(50*dt), 5e-3)
        self.assertAlmostEqual(mu_effective(x, 35.), math.tan(math.radians(35.)), delta=.01)

    def test_tear_split_books_gamma_keeps_contact_valid_and_restores_exactly(self):
        """Vertex splitting at a running trouser tear: each fully opened cohesive bond has dissipated exactly Gamma l; no
        contact pair at or below dmin and no intersection after splits; a checkpoint taken before the splits restores the
        pre-split topology exactly, and re-running from it reproduces the same splits and state."""
        import ipctk
        w, sh, step = self._trouser()
        i = 0
        while i < 110:
            step(i); i += 1
        self.assertEqual([e["kind"] for e in w.tear_events if e["kind"] != "cut"], [])
        snap, i0 = w.get_state(), i
        tris0, sl0, used0 = w.tris.copy(), w.sheet_slices[0], list(w.sheet_used)
        while i < 400 and sum(e["kind"] == "vcct" for e in w.tear_events) < 2:
            step(i); i += 1
        ev = [(e["kind"], e["v"], e["copy"]) for e in w.tear_events]
        self.assertGreaterEqual(sum(e[0] == "vcct" for e in ev), 2)
        tb = w.tear_bonds
        done = tb["h"] >= tb["dc"]
        np.testing.assert_allclose(tb["diss"][done], tb["gl"][done], rtol=1e-9)
        self.assertAlmostEqual(w.tear_dissipated, float(tb["diss"].sum()), places=12)
        self.assertEqual(len(sh.X2), w.sheet_slices[0].stop - w.sheet_slices[0].start)
        self.assertEqual(len(np.asarray(sh.x0)), len(sh.X2))
        xc = np.asfortranarray(w.x_coll())
        self.assertFalse(ipctk.has_intersections(w.mesh, xc))
        C = w.csets[0].collisions(xc)
        if len(C):
            self.assertGreater(np.sqrt(C.compute_minimum_distance(w.mesh, xc)), w.dmin)
        q1, tris1, diss1 = w.q.copy(), w.tris.copy(), w.tear_dissipated
        w.set_state(snap)
        self.assertTrue(np.array_equal(w.tris, tris0))
        self.assertEqual((w.sheet_slices[0], w.sheet_used), (sl0, used0))
        self.assertEqual(len(sh.X2), sl0.stop - sl0.start)
        for j in range(i0, i):
            step(j)
        self.assertEqual([(e["kind"], e["v"], e["copy"]) for e in w.tear_events], ev)
        self.assertTrue(np.array_equal(w.tris, tris1))
        np.testing.assert_allclose(w.q, q1, atol=1e-9)
        self.assertAlmostEqual(w.tear_dissipated, diss1, places=9)

    def test_vertex_joining_several_fans_separates(self):
        """A vertex whose live triangles form four separate fans (every other triangle around it fully torn) splits into
        four copies in one step."""
        from gift_bags import ipcworld as W, params as Pm, scene as S
        sh = S.grid_sheet("wrap", .048, .048, .008, Pm.WRAP, lambda X2: np.c_[X2[:, 0], X2[:, 1], np.full(len(X2), .2)], seed=1)
        w = W.IPCWorld([sh], [], [], gravity=(0., 0., 0.), contact_stock=Pm.WRAP)
        X2, T = np.asarray(sh.X2), np.asarray(sh.tris)
        v = int(np.argmin(np.linalg.norm(X2 - [.024, .024], axis=1)))
        fan = [t for t in range(len(T)) if v in T[t]]
        order = [fan[i] for i in np.argsort([np.arctan2(*(X2[T[t]].mean(0) - X2[v])[::-1]) for t in fan])]
        w.damage[order[::2]] = 1.
        w._refresh_boundary(); w._make_contact_meshes()
        self.assertEqual(len(w._fan(v)), 4)
        w.step(4e-3)
        self.assertEqual([e["kind"] for e in w.tear_events], ["bowtie"]*3)
        self.assertEqual(len(w._fan(v)), 1)

    def test_bag_tear_check_fails_on_a_cut_and_passes_without(self):
        """The open-bag / bag-purchase tear check reads the world's crack length (split faces + crack band): an uncut tote
        stock passes, the same stock cut 24 mm fails (tolerance task.TEAR_TOL_MM)."""
        from types import SimpleNamespace
        from gift_bags import ipcworld as W, params as Pm, scene as S, task as T
        for cut, ok in ((0., True), (.024, False)):
            sh = S.grid_sheet("bag", .064, .048, .008, Pm.TOTE, lambda X2: np.c_[X2[:, 0], X2[:, 1], np.full(len(X2), .2)], seed=1)
            w = W.IPCWorld([sh], [], [], gravity=(0., 0., 0.), contact_stock=Pm.TOTE)
            ev = T.Evaluator.__new__(T.Evaluator)
            ev.world, ev.crack0 = SimpleNamespace(ipc=w), {0: 0.}
            if cut:
                X2 = np.asarray(sh.X2)
                w.cut(0, [int(i) for i in np.argsort(X2[:, 0]) if abs(X2[i, 1] - .024) < 1e-9 and X2[i, 0] <= cut + 1e-9])
            checks, metrics = {}, {}
            ev._eval_tear(checks, metrics, 0, "bag", T.TEAR_TOL_MM)
            self.assertEqual(checks["bag_not_torn"], ok)
            self.assertAlmostEqual(metrics["bag_new_crack_length_mm"], 1e3*cut, places=6)

    def test_a_split_whose_bond_never_opened_is_not_counted_as_torn(self):
        """Crack length weights each split edge by the dissipated fraction of its cohesive bond's fracture energy: a
        split inserted at an edge whose bond then never opens counts 0 mm, half opened half its edge, fully opened all
        of it."""
        from types import SimpleNamespace
        from gift_bags import ipcworld as W, params as Pm, scene as S, task as T
        sh = S.grid_sheet("bag", .064, .048, .008, Pm.TOTE, lambda X2: np.c_[X2[:, 0], X2[:, 1], np.full(len(X2), .2)], seed=1)
        w = W.IPCWorld([sh], [], [], gravity=(0., 0., 0.), contact_stock=Pm.TOTE)
        X2 = np.asarray(sh.X2)
        v = int(np.argmin(np.linalg.norm(X2 - [0., .024], axis=1)))        # on the free edge x = 0
        tip = int(np.argmin(np.linalg.norm(X2 - [.008, .024], axis=1)))    # its neighbour inside the sheet
        chain = w._fan(v)[0]
        k = next(i for i in range(1, len(chain)) if chain[i - 1][2] == tip)
        edge = float(np.linalg.norm(X2[v] - X2[tip]))
        gl = Pm.TOTE.Gamma*.5*edge
        s_ = w._split(v, [c[0] for c in chain[:k]], dict(T0=10., gl=gl, ell=.5*edge, psi=1.), dict(kind="init", tip=tip))
        self.assertIsNotNone(s_)
        w._after_topology()
        ev = T.Evaluator.__new__(T.Evaluator)
        ev.world, ev.crack0 = SimpleNamespace(ipc=w), {0: 0.}
        for frac, torn_mm in ((0., 0.), (.5, 500.*edge), (1., 1e3*edge)):
            w.tear_bonds["diss"][0] = frac*gl
            checks, metrics = {}, {}
            ev._eval_tear(checks, metrics, 0, "bag", T.TEAR_TOL_MM)
            self.assertAlmostEqual(metrics["bag_new_crack_length_mm"], torn_mm, places=6)
            self.assertEqual(checks["bag_not_torn"], torn_mm <= T.TEAR_TOL_MM)


    def test_pulling_the_leaves_of_a_closed_fold_apart_is_not_a_trouser_pull(self):
        """Tear start at a free edge: the trouser part of the pull is taken along the mean of the two sides' normals,
        not normalised: 1 on a flat sheet, 0 across a fold closed flat on itself (pulling its leaves apart opens the
        fold; in the unfolded sheet both flanks move to the same side). Normalised, the direction across a closed fold
        would be arbitrary. A sheet folded flat, its leaves pulled 8 mm apart at the fold's end with up to 3 N a side,
        opens without a split."""
        from gift_bags import ipcworld as W, params as Pm, scene as S
        m, L, Wd, h, gap = Pm.WRAP, .032, .032, .004, 4e-4

        def world(folded):
            def place(X2):
                if not folded:
                    return np.c_[X2[:, 0], X2[:, 1], np.full(len(X2), .1)]
                up = X2[:, 1] > Wd/2 + 1e-9                               # the upper leaf lies back over the lower one
                y = np.where(up, Wd - X2[:, 1], X2[:, 1])
                z = .1 + np.where(up, gap, np.where(np.abs(X2[:, 1] - Wd/2) < 1e-9, gap/2, 0.))
                return np.c_[X2[:, 0], y, z]
            sh = S.grid_sheet("wrap", L, Wd, h, m, place, seed=1)
            return W.IPCWorld([sh], [], [], gravity=(0., 0., 0.), contact_stock=m), np.asarray(sh.X2)

        def across_the_line(w, X2):
            """The two triangles on either side of the line y = Wd/2 at the free edge x = 0, and the edge's vertices."""
            v = int(np.argmin(np.linalg.norm(X2 - [0., Wd/2], axis=1)))
            chain = w._fan(v)[0]
            k = next(i for i in range(1, len(chain)) if abs(X2[chain[i - 1][2], 1] - Wd/2) < 1e-9)
            return chain[k - 1][0], chain[k][0]
        wf, X2 = world(folded=False)
        self.assertAlmostEqual(float(np.linalg.norm(wf._mean_normal(*across_the_line(wf, X2), wf.x_free()))), 1., places=6)
        w, X2 = world(folded=True)
        t1, t2 = across_the_line(w, X2)
        self.assertLess(float(np.linalg.norm(w._mean_normal(t1, t2, w.x_free()))), .12)   # the closed fold (0.4 mm apart)
        tv, nn, mm = w._edge_frame(int(np.argmin(np.linalg.norm(X2 - [0., Wd/2], axis=1))),
                                   int(np.argmin(np.linalg.norm(X2 - [h, Wd/2], axis=1))), t1, t2, w.x_free())
        self.assertAlmostEqual(float(np.linalg.norm(nn)), 1., places=6)   # the crack-tip frame stays orthonormal (VCCT)
        n = len(X2)
        a = int(np.argmin(np.linalg.norm(X2 - [0., Wd/2 - h], axis=1)))
        b = int(np.argmin(np.linalg.norm(X2 - [0., Wd/2 + h], axis=1)))
        far = np.flatnonzero(X2[:, 0] > L - 1e-9)
        x0 = w.x_free()[:n].copy()
        for i in range(75):
            d = min(1., (i + 1)/50.)*.004
            tg = x0.copy()
            tg[a, 2] -= d; tg[b, 2] += d
            w.fixture_springs = [(np.array([a, b]), tg[[a, b]], 2e3, 3.), (far, x0[far], 2e3, 5.)]
            w.step(4e-3)
        self.assertEqual([e["kind"] for e in w.tear_events], [])
        self.assertGreater(float(w.x_free()[b, 2] - w.x_free()[a, 2]), .006)       # the fold did open

    def test_joint_target_command_is_a_timed_motion_and_is_refused_when_too_fast(self):
        """A named joint target through the public action path: a far target with a short duration is refused with the
        minimum duration in the message and nothing moves; the same target over that duration runs without a velocity
        reflex."""
        import math, tempfile
        from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
        from gym_anything_robotics.actions import NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        b = NewtonGiftBagsBackend(robot="dual-fr3", preset="home", task="lift-lid", cameras=(), recording=False,
                                  output_dir=Path(tempfile.mkdtemp(prefix="gb_jt_"))/"session")
        try:
            rig = b.env.world.rig
            j = next(n for n in rig.joints if n.startswith("right/") and "finger" not in n)
            q0 = float(rig.targets[rig.joints[j]["q"]])
            steps0 = b.env.world.steps
            with self.assertRaises(ActionRejectedError) as ctx:
                b.act(NamedJointTarget({j: q0 + 1.0}, 5))
            self.assertIn("rad/s", str(ctx.exception))
            self.assertIn("ticks", str(ctx.exception))
            self.assertEqual(b.env.world.steps, steps0)                       # nothing executed
            self.assertEqual(float(rig.targets[rig.joints[j]["q"]]), q0)
            need = int(math.ceil(1.875*1.0/rig.joints[j]["velocity_limit"]/b.config.dt)) + 2
            n0 = len(rig.reflexes)
            info = b.act(NamedJointTarget({j: q0 + 1.0}, need)).info
            self.assertEqual(info["executed_ticks"], need)
            self.assertEqual(len(rig.reflexes), n0)
            self.assertLess(abs(float(rig.state.joint_q.numpy()[rig.joints[j]["q"]]) - (q0 + 1.0)), .05)
            self.assertEqual(b.act(NamedJointTarget({}, 3)).info["executed_ticks"], 3)   # an empty command holds
        finally:
            b.close()

    def test_cartesian_path_refuses_branch_changes(self):
        from gift_bags.config import Config
        from gift_bags.env import GiftBagsEnv
        from gift_bags.kinematics import ArmKinematics, tool_quat
        env = GiftBagsEnv(Config(preset="home", task="lift-lid"))
        ik = ArmKinematics(env.world.rig)
        q0 = ik.joints_now("left")
        p = ik.forward("left")[:3, 3]
        down = tool_quat(z_axis=(0, 0, -1), y_axis=(1, 0, 0))
        path = ik.cartesian_path("left", q0, [((p[0], p[1], p[2] - .03), down)])
        steps = np.array([[j[n] for n in ik.arms["left"][1]] for j, _ in path])
        self.assertLess(np.abs(np.diff(steps, axis=0)).max(), .35)
        with self.assertRaises(ValueError):                    # a pose behind the robot base is not reachable
            ik.cartesian_path("left", q0, [((p[0], p[1] + 1.5, p[2]), down)])

    def test_g1_world_builds_and_stands(self):
        """The floating G1 (no weld, no support) in the home preset: every link is a driven obstacle, the Dex3 finger
        links carry the dex3_finger friction class from the ledger, the friction table equals the merged ledger, and the
        robot stands 1 s after the reset with its pelvis within 5 mm of its settled height and under 2 deg of tilt."""
        from gift_bags.config import Config
        from gift_bags.env import GiftBagsEnv
        from gift_bags import g1 as G1, params as Pm
        with mock.patch.dict("os.environ", {"GIFT_BAGS_G1_DEV": "1"}):     # unqualified: development only
            env = GiftBagsEnv(Config(preset="home", task="lift-lid", robot="g1", rigid_dt=G1.RIGID_DT_MAX, substeps=G1.SUBSTEPS))
        w = env.world
        robot = [b for b in w.ipc.bodies if b.name.startswith("robot:")]
        self.assertEqual(len(robot), len(w.rig.link_collision_shapes()))
        self.assertEqual(sum(b.fmat == "dex3_finger" for b in robot), 14)
        ledger = {**Pm.FRICTION, **Pm.FRICTION_DEX3}
        names = [c["name"] for c in w.ipc._friction_classes]
        for (i, j), v in w.ipc._mu_table.items():
            self.assertEqual(v, ledger[frozenset((names[i - 1][4:], names[j - 1][4:]))])
        p0, _ = G1.pelvis_state(w.rig)
        env.act({}, ticks=int(round(1./env.config.dt)))
        p1, tilt = G1.pelvis_state(w.rig)
        self.assertLess(abs(p1[2] - p0[2]), 5e-3)
        self.assertLess(tilt, 2.)
        with self.assertRaises(ValueError):                    # the G1 is qualified at 0.5 ms only (g1.RIGID_DT_MAX)
            GiftBagsEnv(Config(preset="home", task="lift-lid", robot="g1"))

    def test_sealed_carton_builds_and_rests(self):
        """Receiving preset: the cached sealed carton builds in the full world, stays put for 0.2 s with the tape bonds
        intact and the flaps shut, restores exactly, and the evaluator reports the untouched carton as not opened."""
        from gift_bags.config import Config
        from gift_bags.env import GiftBagsEnv
        from gift_bags import carton as CT
        env = GiftBagsEnv(Config(preset="receiving", task="open-carton"))
        ipc = env.world.ipc
        snap = env.checkpoint()
        q = ipc.q.copy()
        items = [i for i, b in enumerate(ipc.bodies) if b.kind == "dynamic"]
        p0 = {i: ipc.body_pose(i)[0].copy() for i in items}
        env.act({}, ticks=50)
        self.assertLess(max(np.linalg.norm(ipc.body_pose(i)[0] - p0[i]) for i in items), 2e-4)
        self.assertLess(max(abs(a) for a in CT.flap_angles(ipc).values()), 1.)
        self.assertEqual(ipc.carton_bonds.intact_fraction, 1.)
        ev = env.get_state()["evaluation"]
        self.assertFalse(ev["success"])
        self.assertFalse(ev["checks"]["seam_open"])
        self.assertFalse(any(ev["checks"][f"{f}_open"] for f in CT.FLAPS))
        self.assertTrue(ev["checks"]["contents_in_place"] and ev["checks"]["carton_upright"])
        env.restore(snap)
        self.assertTrue(np.array_equal(ipc.q, q))
        self.assertTrue(np.all(ipc.carton_bonds.D == snap["ipc"]["bonds"][0]["D"]))


class GiftBagsVariationTest(unittest.TestCase):
    """The home preset's seeded present (variants.HOME_AXES): box size, lid clearance, wrap stock. No stepping."""

    def test_variation_seed_0_is_the_authored_present(self):
        from gift_bags import variants as V, giftbox as GB, wrapmap as WM, scene as S, params as Pm
        from gift_bags.config import Config
        self.assertEqual(V.home_variant(0), dict(box=GB.AUTHORED, clearance=GB.CLEARANCE, wrap_stock=WM.DEFAULT_STOCK))
        self.assertEqual(GB.box_spec(), GB.BoxSpec(GB.AUTHORED, *GB.BOX_HALF, GB.BOX_H, GB.LID_H, GB.WALL, GB.CLEARANCE,
                                                   GB.BASE_MASS, GB.LID_MASS))
        self.assertIs(Pm.WRAP_STOCKS[WM.DEFAULT_STOCK][0], Pm.WRAP)
        self.assertIs(Pm.WRAP_STOCKS[WM.DEFAULT_STOCK][1], Pm.WRAP_CREASE)
        hx, hy = GB.BOX_HALF
        self.assertEqual(WM.cuboid(), (2*(hx + GB.CLEARANCE + GB.WALL), hy + GB.CLEARANCE + GB.WALL, GB.BOX_H + GB.WALL + 5e-5))
        key = WM._cache_key(S.TABLE_Z, S.BOX_XY, "girth", 0, .008)
        self.assertEqual(key, WM._cache_key(S.TABLE_Z, S.BOX_XY, "girth", 0, .008, GB.box_spec(GB.AUTHORED, GB.CLEARANCE),
                                            WM.DEFAULT_STOCK))
        self.assertNotEqual(key, WM._cache_key(S.TABLE_Z, S.BOX_XY, "girth", 0, .008, None, "consumer_53gsm"))
        spec = S.build(Config(preset="home", task="lift-lid", seed=0))
        g = spec.geometry
        self.assertEqual(g["frame"], dict(half=S.FRAME_HALF, mass=S.FRAME_MASS))
        self.assertEqual((g["box"]["hx"], g["box"]["hy"], g["box"]["base_h"], g["box"]["clearance"]),
                         (hx, hy, GB.BOX_H, GB.CLEARANCE))
        masses = {b.name: b.mass for b in spec.bodies}
        self.assertEqual((masses["gift_base"], masses["gift_lid"], masses["photo_frame"]),
                         (GB.BASE_MASS, GB.LID_MASS, S.FRAME_MASS))

    def test_variation_seeds_draw_within_the_axes_and_keep_placement(self):
        import math
        from gift_bags import variants as V, params as Pm
        A = V.HOME_AXES
        self.assertEqual(set(A["box"][1:]), set(Pm.GIFT_BOXES))
        self.assertEqual(tuple(A["clearance_m"]), Pm.LID_CLEARANCES)
        self.assertEqual(set(A["wrap_stock"]), set(Pm.WRAP_STOCKS))
        seen = set()
        for seed in range(1, 300):
            v = V.home_variant(seed)
            self.assertEqual(v, V.home_variant(seed))                         # deterministic
            self.assertIn(v["box"], A["box"]); self.assertIn(v["clearance"], A["clearance_m"])
            self.assertIn(v["wrap_stock"], A["wrap_stock"])
            seen.add((v["box"], v["clearance"], v["wrap_stock"]))
        self.assertEqual(len(seen), math.prod(len(x) for x in A.values()))  # every combination is reachable
        self.assertTrue(.5e-3 <= min(A["clearance_m"]) and max(A["clearance_m"]) <= 1.0e-3)
        # the variation draws from its own stream and leaves every seed's placement as it is (seed 7, test above)
        rng = np.random.default_rng(7)
        u = rng.uniform(-1., 1., 3)
        r = V.POSE_RANGES["home"]["box"]
        self.assertEqual(V.placement("home", 7)["box"], (float(u[0]*r["dx"]), float(u[1]*r["dy"]),
                                                         float(math.radians(u[2]*r["yaw_deg"]))))

    def test_variation_wrap_net_obeys_the_folding_rules_for_every_box(self):
        from gift_bags import variants as V, giftbox as GB, wrapmap as WM, ipcworld as W, scene as S
        for name in V.HOME_AXES["box"]:
            for c in (min(V.HOME_AXES["clearance_m"]), max(V.HOME_AXES["clearance_m"])):
                box = GB.box_spec(name, c)
                Lx, b, Hc = WM.cuboid(box)
                rules = WM.net_rules(Lx, b, Hc)
                self.assertTrue(all(rules.values()), (name, c, rules))
                sheet, geo = WM.build(np.zeros(3), box=box, stock="consumer_53gsm")
                X2 = np.asarray(sheet.X2)
                # the net: length + two end flaps by the girth + the 40 mm seam overlap
                np.testing.assert_allclose(np.ptp(X2, axis=0), [Lx + 2*WM.END_FLAP, 2*(2*b + Hc) + WM.SEAM_OVERLAP],
                                           atol=1e-9)
                self.assertAlmostEqual(geo["y0"], -WM.SEAM_OVERLAP/2)
                E = W.edges_of(np.asarray(sheet.tris))
                Lr = np.linalg.norm(X2[E[:, 0]] - X2[E[:, 1]], axis=1)
                Lf = np.linalg.norm(sheet.x0[E[:, 0]] - sheet.x0[E[:, 1]], axis=1)
                self.assertLess(float(np.max(np.abs(Lf - Lr)/Lr)), .05)       # the authored net's fold map: 4.5 %
                self.assertEqual(len(WM.tape_sheets(sheet.x0, sheet.tris, geo)), 3)   # every tape vertex has paper beneath
                fh, _ = S._home_frame(name)
                self.assertTrue(fh[0] < box.hx - box.wall and fh[1] < box.hy - box.wall and 2*fh[2] < box.h - box.wall)
        with self.assertRaises(ValueError):                                    # a deep box breaks the end-flap rule
            WM.wrap_mesh(box=GB.BoxSpec("deep", .09, .065, .10, .03, 1.5e-3, .5e-3, .1, .1))

    def test_variation_box_masses_and_consumer_stock(self):
        from gift_bags import giftbox as GB, params as Pm, ipcworld as W
        hx, hy, h, w, c = *GB.BOX_HALF, GB.BOX_H, GB.WALL, GB.CLEARANCE
        areal = Pm.GREYBOARD_RHO*w + Pm.BOX_COVER_RHO_A                      # the rule box_spec applies to catalogue boxes
        self.assertAlmostEqual(areal*(4*hx*hy + 4*(hx + hy)*h), GB.BASE_MASS, delta=.03*GB.BASE_MASS)
        lhx, lhy = hx + c + w, hy + c + w
        self.assertAlmostEqual(areal*(4*lhx*lhy + 4*(lhx + lhy)*GB.LID_H), GB.LID_MASS, delta=.06*GB.LID_MASS)
        big = GB.box_spec("uline_setup_8.5x11x2", .5e-3)
        self.assertAlmostEqual(2*(big.hx - big.wall), 11*Pm.IN, places=9)
        self.assertAlmostEqual(big.wall, .032*Pm.IN, places=12)
        C, A = Pm.WRAP_CONSUMER, Pm.WRAP
        self.assertGreater(C.t, Pm.MAGIC_TAPE.t)                               # the paper, not the tape, sets the caliper
        self.assertAlmostEqual(C.t, 58.3e-6, places=12)
        for k in ("X_md", "X_cd", "Gamma", "Gamma_I"):
            self.assertAlmostEqual(getattr(C, k)/getattr(A, k), .053/.081, places=12)
        self.assertLess(Pm.WRAP_CONSUMER_CREASE.k, Pm.WRAP_CREASE.k)
        self.assertEqual(W.contact_caliper(C)["rest_sheet_sheet"], C.t)

    def test_variation_contract_lists_the_axes(self):
        from gift_bags import variants as V
        c = V.contract()
        self.assertEqual(c["home_axes"], {k: list(v) for k, v in V.HOME_AXES.items()})
        json.dumps(c)
        for word in ("box", "clearance", "wrap stock"):
            self.assertIn(word, c["qualification"])
        self.assertEqual(c["selectors"], ["preset", "task", "seed"])


if __name__ == "__main__":
    unittest.main()

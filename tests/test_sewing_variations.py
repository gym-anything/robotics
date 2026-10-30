"""Checks of sewing's variation catalogue, physics trace, checkpoints and real rigid-only builds:

- the variation catalogue sewing-sample-room-v1 (sewing/variations.py: task x seed, its frozen identity, selection and
  refusals) and its in-session switching through the backend and GAR's HTTP routes (/v1/variations,
  /v1/variations/select, /v1/variations/start, reset with options {"variation_id": N});
- the per-action physics trace (sewing/trace.py) through the backend hooks and /v1/history/{id}/trace;
- the checkpoints' stated scope;
- a cheap real build per deployment: the backend over the real world built rigid-only (Config(cloth=False): the
  robots, table and machine casting, no cloth, no machine model, no start state), for the FR3 pair, the single FR3,
  the Panda pair and the G1, checked for the contract facts test_sewing_robots.py checks over its stand-in world, and
  the stand-in world's shape checked against the real one.

The real builds run on the CPU (the rigid model and MuJoCo are on the host; nothing touches the GPU without cloth) in
seconds; a deployment whose locked robot assets are not installed is skipped (resolved as sewing/doctor.py does). The
physics class at the end (a cloth session switching variations) is skipped unless GAR_SEWING_PHYSICS_TESTS=1, the
repository's convention for GPU tests.
"""

import json

import numpy as np
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
import weakref

from gym_anything_robotics.errors import ActionRejectedError, BackendProtocolError, UnsupportedActionError
from tests.packages import PACKAGES

ROOT = PACKAGES
# the catalogue's identity: changing what an entry holds (a task's setup, a seed's layout) needs a new VERSION
FROZEN_DIGEST = "a6b9ddbddf685d2d386c70c9f35736fbe3aad763bd03126dc1df37dde00a71d1"


def _missing_assets(robot):
    """None if every locked asset the deployment loads resolves (sewing/doctor.py's check), else the reason."""
    from sewing.doctor import MODELS
    from robots.assets import resolve
    for name in MODELS[robot]:
        try:
            resolve(name)
        except Exception as exc:
            return f"{robot}: {name} assets not installed ({type(exc).__name__}: {exc}); see `gar doctor sewing`"
    return None


def _rigid_backend(robot="single-fr3", task="plain-seam", seed=0, cameras=(), **options):
    from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
    out = Path(tempfile.mkdtemp(prefix="sewvar-"))/"session"
    return NewtonSewingBackend(robot=robot, task=task, seed=seed, cameras=cameras, config={"cloth": False},
                               output_dir=str(out), **options)


class CatalogueTest(unittest.TestCase):
    """sewing/variations.py alone (no world)."""


    def test_every_entry_is_the_episode_a_fresh_session_selects(self):
        from sewing import variations as VR
        from sewing.task import INSTRUCTIONS
        from sewing.variants import resolve, task_resolve
        import hashlib
        for i in range(VR.COUNT):
            e = VR.specification(i)
            with self.subTest(variation=i):
                t = task_resolve({"task": e["task"], "seed": e["seed"]})
                self.assertEqual(e["task_id"], t["task_id"])          # GAR's task catalogue names the same episode
                self.assertEqual(VR.config_values(e), resolve(e["preset"], e["task"], e["seed"]))
                self.assertEqual(e["config"]["start"], e["start"])
                self.assertIsNone(e["split"])
                self.assertIn(e["task"], INSTRUCTIONS)
                body = {k: v for k, v in e.items() if k != "sha256"}
                self.assertEqual(e["sha256"], hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"))
                                                             .encode()).hexdigest())
        self.assertIsNot(VR.specification(3), VR.specification(3))     # callers own their copy

    def test_selection_is_exact_and_never_relaxed(self):
        from sewing import variations as VR
        self.assertEqual(VR.select_request({"variation_id": 33})["task_id"], "sewing/v1/sample-room/pivot/seed-3")
        self.assertEqual(VR.select_request({"task": "pivot", "seed": 3})["variation_id"], 33)
        self.assertEqual(VR.select_request({"task": "A5", "seed": 3, "preset": "sample-room"})["variation_id"], 33)
        self.assertEqual(VR.select_request({"variation_id": 33, "task": "pivot", "seed": 3})["variation_id"], 33)
        for bad in ({}, {"variation_id": 90}, {"variation_id": -1}, {"variation_id": True}, {"variation_id": 3.0},
                    {"variation_id": "3"}, {"variation_id": 33, "task": "curve"}, {"variation_id": 33, "seed": 4},
                    {"task": "pivot"}, {"seed": 3}, {"task": "hem", "seed": 0}, {"task": "pivot", "seed": 10},
                    {"task": "pivot", "seed": 1, "preset": "home"}, {"task": "pivot", "seed": 1, "split": "test"},
                    [], "33"):
            with self.subTest(request=bad), self.assertRaises(ValueError):
                VR.select_request(bad)

    def test_contract_states_the_switch_and_its_cost(self):
        from sewing import variations as VR
        c = VR.contract({"x": 1}, checkpoints="scope")
        self.assertTrue(c["supported"])
        self.assertEqual((c["count"], c["integer_id_range"]), (VR.COUNT, [0, VR.COUNT - 1]))
        self.assertEqual(c["selectors"], ["variation_id", "preset", "task", "seed", "contract"])
        self.assertEqual((c["current"], c["checkpoints"], c["splits"]), ({"x": 1}, "scope", []))
        self.assertFalse(c["preview_mutates"])
        self.assertTrue(c["start_replaces_episode"])
        self.assertIn("start state", c["note"])
        self.assertIn("not yet cached", c["note"])
        json.dumps(c)


class _RigidCase(unittest.TestCase):
    ROBOT = "single-fr3"

    def setUp(self):
        reason = _missing_assets(self.ROBOT)
        if reason:
            self.skipTest(reason)


class BackendVariationTest(_RigidCase):
    """The backend's switch over real rigid-only worlds (seconds each)."""

    def test_contract_and_preview_change_nothing(self):
        b = _rigid_backend(task="plain-seam", seed=0)
        try:
            env = b.env
            c = b.variation_contract()
            self.assertEqual(c["current"]["variation"]["variation_id"], 10)
            self.assertEqual((c["current"]["task"], c["current"]["seed"]), ("plain-seam", 0))
            self.assertEqual(c["current"]["config_overrides"], ["cloth"])
            from gym_anything_robotics.backends.newton_sewing import CHECKPOINT_SCOPE
            self.assertEqual(c["checkpoints"], CHECKPOINT_SCOPE)
            sel = b.select_variation({"task": "pivot", "seed": 3})
            self.assertEqual(sel["reset_request"], {"options": {"variation_id": 33}})
            self.assertEqual((sel["task"], sel["variation"]["seed"]), ("pivot", 3))
            from sewing.task import INSTRUCTIONS
            self.assertEqual(sel["description"], INSTRUCTIONS["pivot"])
            self.assertIs(b.env, env)
            self.assertEqual(b.config.task, "plain-seam")
            self.assertIn("(variation 10)", b.capabilities.notes["task"])
        finally:
            b.close()

    def test_reset_with_a_variation_switches_as_a_fresh_session_would(self):
        from dataclasses import asdict
        from sewing.env import start_path
        from sewing.task import INSTRUCTIONS
        b = _rigid_backend(task="plain-seam", seed=0)
        try:
            b.save_checkpoint(label="c0")
            old_env, old_world = weakref.ref(b.env), weakref.ref(b.env.world)
            gen = b.env.generation
            state, info = b.reset(options={"variation_id": 33})
            # the replaced world is released by the switch itself (no collection here; with the engine imported off
            # the session's stack, the session's first world too)
            self.assertIsNone(old_env())
            self.assertIsNone(old_world())
            self.assertEqual((info["continuity"], info["variation"]["variation_id"]), ("new_variation", 33))
            self.assertIn("build_s", info["start_state"])
            self.assertEqual((b.config.task, b.config.seed, b.env.generation), ("pivot", 3, gen + 1))
            self.assertEqual((state["config"]["task"], state["config"]["seed"], state["tick"]), ("pivot", 3, 0))
            self.assertEqual(b.list_checkpoints(), ())                  # dropped, as the scope states
            with self.assertRaises(KeyError):
                b.restore_checkpoint("c0")
            self.assertIn("(variation 33)", b.capabilities.notes["task"])
            self.assertEqual(b.capabilities.notes["instruction"], INSTRUCTIONS["pivot"])
            self.assertEqual(b.variation_contract()["current"]["variation"]["variation_id"], 33)
            fresh = _rigid_backend(task="pivot", seed=3)
            try:
                self.assertEqual(asdict(b.config), asdict(fresh.config))
                self.assertEqual(start_path(b.config), start_path(fresh.config))   # the same cached start state
            finally:
                fresh.close()
            # a plain reset returns to the switched variation's start
            state, info = b.reset()
            self.assertEqual((state["config"]["task"], info["continuity"]), ("pivot", "reset"))
        finally:
            b.close()

    def test_refusals_change_nothing(self):
        b = _rigid_backend(task="plain-seam", seed=0)
        try:
            env, config = b.env, b.config
            for kwargs in (dict(options={"variation_id": 90}), dict(options={"variation_id": "3"}),
                           dict(options={"variation_id": 3}, seed=1), dict(options={"variation_id": 3}, seed=0),
                           dict(options={"variation_id": 3, "task": "pivot"}), dict(options={"task": "pivot"}),
                           dict(seed=4)):
                with self.subTest(**kwargs), self.assertRaises(ActionRejectedError):
                    b.reset(**kwargs)
                self.assertIs(b.env, env)
                self.assertIs(b.config, config)
            with patch("sewing.task_set_env.make_environment", side_effect=RuntimeError("no GPU")):
                with self.assertRaises(BackendProtocolError) as caught:
                    b.reset(options={"variation_id": 3})
            self.assertIn("previous episode is unchanged", str(caught.exception))
            self.assertIs(b.env, env)
            self.assertEqual(b.variation["variation_id"], 10)
        finally:
            b.close()

    def test_checkpoint_scope_is_stated(self):
        from gym_anything_robotics.backends.newton_sewing import CHECKPOINT_SCOPE
        for words in ("In memory", "server restart", "variation switch", "nothing is written to disk"):
            self.assertIn(words, CHECKPOINT_SCOPE)
        b = _rigid_backend()
        try:
            self.assertEqual(b.capabilities.notes["checkpoints"], CHECKPOINT_SCOPE)
            saved = b.save_checkpoint(label="a")
            self.assertEqual((saved["storage"], saved["scope"], saved["variation_id"]),
                             ("in_memory_session", CHECKPOINT_SCOPE, 10))
            _, info = b.restore_checkpoint("a")
            self.assertEqual((info["storage"], info["scope"]), ("in_memory_session", CHECKPOINT_SCOPE))
        finally:
            b.close()


class TraceTest(_RigidCase):
    """sewing/trace.py: one sample per tick through the backend hooks, and the machine's summary and events."""

    def _joint(self, b):
        j = next(n for n in b.env.world.joints if n.endswith("joint4"))
        return j, b.env.get_state()["joints"][j]["position"]

    def test_one_sample_per_tick_through_the_backend(self):
        from gym_anything_robotics.actions import NamedJointTarget
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        b = _rigid_backend()
        try:
            j, q = self._joint(b)
            ref = b.begin_action_trace({"action_id": "a1", "request_id": "r1"})
            self.assertTrue(ref["available"])
            b.act(NamedJointTarget(targets_si={j: q + 0.05}, ticks=40))
            s = b.end_action_trace()
            self.assertTrue(s["complete"])
            self.assertEqual((s["retained_samples"], s["expected_samples_through_last_tick"]), (40, 40))
            meta = s["metadata"]["data"]                       # the header as recorded
            self.assertEqual((meta["task"], meta["robot"], meta["seed"]), ("plain-seam", "single-fr3", 0))
            self.assertEqual(meta["joint_names"], list(b.env.world.joints))
            store = PhysicsTraceStore(Path(ref["store_path"]), readonly=True)
            try:
                items = store.page("a1", kind="sample", limit=128)["items"]
                events = [e["data"] for e in store.page("a1", kind="event", limit=128)["items"]]
            finally:
                store.close()
            self.assertEqual([i["tick"] for i in items[:3]], [1, 2, 3])
            d = items[-1]["data"]
            n = len(meta["joint_names"])
            self.assertEqual({k: len(v) for k, v in d["joints"].items()},
                             {"q": n, "qd": n, "target": n, "servo_target": n})
            self.assertAlmostEqual(d["joints"]["target"][meta["joint_names"].index(j)], q + 0.05, places=5)
            self.assertEqual(len(d["robot"]["external_torque_nm"]["right"]), 7)
            self.assertIsNone(d["machine"])                     # rigid-only: no machine, no cloth
            self.assertIsNone(d["robot"]["cloth_force_n"])
            self.assertIn("rigid_s", d["wall"])
            cost = [e for e in events if e["type"] == "trace_cost"]
            self.assertEqual((len(cost), cost[0]["ticks"], cost[0]["capture_errors"]), (1, 40, 0))
            # the switch rebinds the trace to the new world
            b.reset(options={"variation_id": 31})
            self.assertIs(b.trace.env, b.env)
            b.begin_action_trace({"action_id": "a2", "request_id": "r2"})
            b.act(NamedJointTarget(targets_si={j: q}, ticks=5))
            s = b.end_action_trace()
            self.assertEqual((s["retained_samples"], s["metadata"]["data"]["task"], s["metadata"]["data"]["seed"]),
                             (5, "pivot", 1))
        finally:
            b.close()

    def test_disabled_trace_and_option_checks(self):
        b = _rigid_backend(trace=False)
        try:
            self.assertEqual(b.begin_action_trace({"action_id": "x", "request_id": "y"})["available"], False)
            self.assertIsNone(b.end_action_trace())
            self.assertIn("disabled", b.capabilities.notes["trace"])
        finally:
            b.close()
        for bad in (dict(trace="yes"), dict(trace_bytes=1024), dict(trace_bytes=True)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                _rigid_backend(**bad)

    def test_machine_summary_and_events_on_a_real_machine(self):
        # the machine's state and ledger as the sampler reads them after a tick, on a real Machine (built without the
        # cloth; its records appended as the machine appends them), without stepping any physics
        from sewing.env import SewingEnv
        from sewing.machine import Hole, Machine, MachineSettings
        from sewing.trace import Trace
        from sewing.world import Config
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        env = SewingEnv(Config(cloth=False, robot="single-fr3"))
        path = Path(tempfile.mkdtemp(prefix="sewtrace-"))/"t.sqlite3"
        tr = Trace(env, path)
        M = env.world.machine = Machine(MachineSettings())
        try:
            tr.begin({"action_id": "m", "request_id": "m"})
            env.steps = 1
            M.command(speed_spm=600.0)
            env.world.events.append(dict(tick=0, machine_io={"speed_spm": 600.0}))
            M.holes.append(Hole(ply=0, face=7, bary=(.2, .3, .5), uv=(.19, .41), tick=1, theta=3.0, cycle=0, z0=1e-3))
            M.stitches.append(dict(cycle=0, tick=1, formed=True, reason=None, reverse=False, holes=[0], needle="ok"))
            M.threads.append(dict(con=0, kind="needle", cycle=0, rest=1e-3, broken=False, tension=.31))
            M.faults.append(dict(tick=1, fault="thread_break", kind="bobbin", cycle=0, tension_N=9.0))
            tr._on_tick()
            env.steps = 2
            M.stitches.append(dict(cycle=1, tick=2, formed=False, reason="missed_loop", reverse=False, holes=[],
                                   needle="ok"))
            M.log.append(dict(event="trim", cycle=1))
            M.presser = "lift1"
            tr._on_tick()
            s = tr.finish()
            self.assertEqual((s["retained_samples"], s["complete"]), (2, True))
            store = PhysicsTraceStore(path, readonly=True)
            try:
                samples = [i["data"] for i in store.page("m", kind="sample")["items"]]
                events = [e["data"] for e in store.page("m", kind="event", limit=128)["items"]]
            finally:
                store.close()
            m1, m2 = samples[0]["machine"], samples[1]["machine"]
            self.assertEqual((m1["mode"], m1["speed_cmd_spm"], m1["holes"], m1["stitches"], m1["formed"]),
                             ("run", 600.0, 1, 1, 1))
            self.assertEqual(m1["threads"], {"count": 1, "broken": 0, "max_tension_n": 0.31})
            self.assertEqual((m2["stitches"], m2["formed"], m2["presser"]), (2, 1, "lift1"))
            self.assertEqual(set(m1), {"shaft_angle_rad", "speed_spm", "speed_cmd_spm", "mode", "needle_tip_z_m",
                                       "needle", "needle_state", "presser", "lifter_z_m", "foot_z_m", "foot_load_n",
                                       "reverse_switch", "reverse_feed", "cycle", "holes", "engaged", "stitches",
                                       "formed", "interlocks", "threads", "tether", "needle_deflection_m"})
            kinds = [e["type"] for e in events]
            for k in ("machine_io", "puncture", "stitch", "machine_fault", "machine_mode", "trim", "presser",
                      "trace_cost"):
                self.assertIn(k, kinds)
            stitches = [e for e in events if e["type"] == "stitch"]
            self.assertEqual([(e["formed"], e["reason"]) for e in stitches], [(True, None), (False, "missed_loop")])
            self.assertEqual([e for e in events if e["type"] == "machine_mode"][0]["to"], "run")
            self.assertEqual([e for e in events if e["type"] == "machine_fault"][0]["fault"], "thread_break")
        finally:
            env.world.machine = None
            tr.close()


class HTTPVariationTest(_RigidCase):
    """GAR's HTTP routes over the real backend (rigid-only world): catalogue, preview, start, refusal, trace."""

    def setUp(self):
        super().setUp()
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.http_server import GARHTTPServer, GARHTTPService
        self.backend = _rigid_backend(task="plain-seam", seed=0)
        self.env = RoboticsEnv(self.backend)
        self.server = GARHTTPServer(("127.0.0.1", 0), GARHTTPService(self.env), api_token="test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.root = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.env.close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.root + path, method="POST" if body is not None else "GET",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as f:
            return json.loads(f.read())

    def test_catalogue_preview_start_refusal_and_trace(self):
        v = self.call("/v1/variations")["variations"]
        self.assertTrue(v["supported"])
        self.assertEqual(v["current"]["variation"]["task_id"], "sewing/v1/sample-room/plain-seam/seed-0")
        preview = self.call("/v1/variations/select", {"task": "curve", "seed": 2})
        self.assertFalse(preview["committed"])
        self.assertEqual(preview["reset_request"], {"options": {"variation_id": 42}})
        self.assertEqual(self.backend.config.task, "plain-seam")
        started = self.call("/v1/variations/start", {"task": "curve", "seed": 2})
        self.assertTrue(started["committed"])
        self.assertEqual((started["info"]["continuity"], started["variation_id"]), ("new_variation", 42))
        self.assertEqual((self.backend.config.task, self.backend.config.seed), ("curve", 2))
        self.assertEqual(self.call("/v1/variations")["variations"]["current"]["variation"]["variation_id"], 42)
        # a refused start is a 400 that changes nothing and leaves the session ready for the next command
        for bad in ({"variation_id": 90}, {"task": "curve"}, {"task": "curve", "seed": 2, "split": "test"}):
            with self.subTest(request=bad), self.assertRaises(urllib.error.HTTPError) as caught:
                self.call("/v1/variations/start", bad)
            self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.backend.variation["variation_id"], 42)
        j = next(n for n in self.backend.env.world.joints if n.endswith("joint4"))
        q = self.backend.env.get_state()["joints"][j]["position"]
        resp = self.call("/v1/action", {"type": "named_joint_target", "targets_si": {j: q + .02}, "ticks": 12})
        self.assertTrue(resp["committed"])
        rec = [r for r in self.call(f"/v1/history?request_id={resp['request_id']}")["records"]
               if r["kind"] == "action"][0]
        page = self.call(f"/v1/history/{rec['id']}/trace?limit=16")
        self.assertTrue(page["available"])
        self.assertTrue(page["summary"]["complete"])
        self.assertEqual(page["summary"]["retained_samples"], 12)
        self.assertEqual(page["summary"]["metadata"]["data"]["task"], "curve")
        self.assertEqual(len(page["items"]), 12)
        events = self.call(f"/v1/history/{rec['id']}/trace/events")
        self.assertIn("trace_cost", [e["data"]["type"] for e in events["items"]])


class _HTTPCase(_RigidCase):
    """GAR's HTTP service over the real backend (rigid-only world), with an owner key and a policy key."""
    TASK = "plain-seam"

    def setUp(self):
        super().setUp()
        from gym_anything_robotics.access import APIKeys
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.http_server import GARHTTPServer, GARHTTPService
        self.backend = _rigid_backend(robot=self.ROBOT, task=self.TASK, seed=0)
        self.env = RoboticsEnv(self.backend)
        self.server = GARHTTPServer(("127.0.0.1", 0), GARHTTPService(self.env), api_token="test-token")
        self.server.keys = APIKeys("test-token", "policy-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.root = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.env.close()

    def call(self, path, body=None, token="test-token"):
        """(HTTP status, decoded body) of a request with the owner's key, or the policy's."""
        req = urllib.request.Request(self.root + path, method="POST" if body is not None else "GET",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as f:
                return f.status, json.loads(f.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def state(self):
        return self.call("/v1/state")[1]["state"]

    HOLD = {"type": "named_joint_target", "targets_si": {}, "ticks": 1}     # one tick, the joint commands held


class G1WaistArmHTTPTest(_HTTPCase):
    """The G1's capabilities offer Cartesian commands on 'left+waist' and 'right+waist' (the hand's IK with the
    three waist joints), and the API passes both names on to the backend. Sent here through the server's own
    decoding, with the owner's key and the policy's."""
    ROBOT = "g1"

    def test_a_waist_leaning_cartesian_command_is_accepted_and_moves_the_waist(self):
        self.assertEqual(self.backend.extra_arm_names, ("left+waist", "right+waist"))
        for token in ("test-token", "policy-token"):
            caps = self.call("/v1/capabilities", token=token)[1]["capabilities"]
            arms = (caps["hardware"]["arms"] if token == "policy-token" else caps["notes"]["arms"].split(","))
            self.assertEqual(list(arms), ["left", "right", "left+waist", "right+waist"])
            for arm in arms:
                with self.subTest(key=token, arm=arm):
                    before = self.state()
                    code, r = self.call("/v1/action", {"type": "cartesian_delta", "arm": arm,
                                                       "delta_m": [0.0, 0.01, -0.01], "duration_s": 0.005}, token)
                    self.assertEqual(code, 200, r)
                    self.assertTrue(r["committed"])
                    after = self.state()
                    self.assertEqual(after["tick"], before["tick"] + 10)
                    moved = {n for n, j in after["joints"].items()
                             if abs(j["target"] - before["joints"][n]["target"]) > 1e-9}
                    waist = {n for n in moved if "waist" in n}
                    # the arm's seven joints, and with '+waist' the three waist joints as well
                    self.assertEqual(len(waist), 3 if arm.endswith("+waist") else 0, moved)
                    self.assertEqual(len(moved - waist), 7, moved)
                    self.assertTrue(all(arm.split("+")[0] in n for n in moved - waist), moved)
        # a pose command too, and a name the deployment does not have is still refused with nothing executed
        tick = self.state()["tick"]
        pose = self.backend.env.world.kin.forward("right+waist")[:3, 3]
        code, r = self.call("/v1/action", {"type": "cartesian_pose", "arm": "right+waist",
                                           "position_m": [float(pose[0]), float(pose[1]), float(pose[2]) + 0.005],
                                           "duration_s": 0.005})
        self.assertEqual((code, r.get("committed")), (200, True), r)
        for token in ("test-token", "policy-token"):
            code, r = self.call("/v1/action", {"type": "cartesian_delta", "arm": "torso", "delta_m": [0, 0, 0.01]},
                                token)
            self.assertEqual(code, 400, r)
        self.assertEqual(self.state()["tick"], tick + 10)
        self.assertTrue(self.call("/v1/health")[1]["ready"])


class ArmNamesAreTheBackendsOwnTest(_HTTPCase):
    """The waist names are the G1's alone: an arm deployment's session refuses them at decoding."""

    def test_an_arm_rig_refuses_the_waist_names(self):
        self.assertEqual(self.backend.extra_arm_names, ())
        tick = self.state()["tick"]
        for token in ("test-token", "policy-token"):
            code, r = self.call("/v1/action", {"type": "cartesian_delta", "arm": "right+waist",
                                               "delta_m": [0, 0, 0.01]}, token)
            self.assertEqual(code, 400, r)
        self.assertEqual(self.state()["tick"], tick)
        code, r = self.call("/v1/action", self.HOLD)
        self.assertEqual(code, 200, r)


class BatchAndStateTest(_HTTPCase):
    """Through the server: a batch with a refused command executes nothing and leaves the session ready; the
    environment's state is built once per command."""

    def joint(self):
        j = next(n for n in self.backend.env.world.joints if n.endswith("joint4"))
        return j, self.state()["joints"][j]["position"]

    def test_a_batch_is_checked_whole_before_its_first_command(self):
        j, q = self.joint()
        good = {"type": "named_joint_target", "targets_si": {j: q + 0.01}, "ticks": 4}
        ticks_2s = int(round(2.0/self.backend.config.dt))
        refused = ({"type": "tool_io", "arm": "right", "outputs": {"speed_spm": 9000.0}},
                   {"type": "tool_io", "arm": "right", "outputs": {"knife": True}},
                   {"type": "tool_io", "arm": "left", "outputs": {"speed_spm": 600.0}},          # no left arm here
                   {"type": "tool_io", "arm": "right", "outputs": {"speed_spm": 0.0}, "ticks": ticks_2s + 1},
                   {"type": "named_joint_target", "targets_si": {j: 99.0}, "ticks": 1},
                   {"type": "named_joint_target", "targets_si": {"right/no_such_joint": 0.0}, "ticks": 1},
                   {"type": "gripper", "arm": "right", "inward_effort_n": 500.0},
                   {"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, 0.001], "duration_s": 0.00031},
                   {"type": "raw", "request": {"op": "joint_impedance", "arm": "right",
                                               "stiffness_nm_per_rad": 99999.0}})
        for bad in refused:
            with self.subTest(second=json.dumps(bad)[:70]):
                before = self.state()
                code, r = self.call("/v1/action", {"actions": [good, bad]})
                self.assertIn(code, (400, 422), r)
                after = self.state()
                self.assertEqual(after["tick"], before["tick"])                      # the first command did not run
                self.assertEqual(after["joints"][j]["target"], before["joints"][j]["target"])
                health = self.call("/v1/health")[1]
                self.assertTrue(health["ready"] and health["runtime_ok"], health)
                code, r = self.call("/v1/action", self.HOLD)
                self.assertEqual(code, 200, r)                                       # and the session takes commands
        # the policy's key too
        tick = self.state()["tick"]
        code, r = self.call("/v1/action", {"actions": [good, refused[0]]}, "policy-token")
        self.assertIn(code, (400, 422), r)
        self.assertEqual(self.state()["tick"], tick)
        # a batch of valid commands runs whole
        code, r = self.call("/v1/action", {"actions": [good, {**self.HOLD, "ticks": 2}]})
        self.assertEqual((code, r.get("committed")), (200, True), r)
        self.assertEqual(self.state()["tick"], tick + 6)
        self.assertAlmostEqual(self.state()["joints"][j]["target"], q + 0.01, places=6)

    def test_validation_does_not_judge_reach_from_the_pose_before_the_batch(self):
        # the IK starts from the joints the arm has when the command's turn comes: validate_action leaves it out
        from gym_anything_robotics.actions import CartesianDelta, CartesianPose
        b = self.backend
        with patch.object(b.env.world.kin, "solve", side_effect=AssertionError("IK consulted")) as solve:
            b.validate_action(CartesianDelta(arm="right", delta_m=(0.0, 0.0, 5.0), duration_s=0.01))
            b.validate_action(CartesianPose(arm="right", position_m=(9.0, 9.0, 9.0), duration_s=0.01))
            with self.assertRaises(ActionRejectedError):
                b.validate_action(CartesianDelta(arm="right", delta_m=(0.0, 0.0, 0.01), duration_s=2.5))
            with self.assertRaises(UnsupportedActionError):
                b.validate_action(CartesianDelta(arm="left", delta_m=(0.0, 0.0, 0.01)))
            solve.assert_not_called()
        # the same unreachable target alone is refused by the command itself, before any tick
        tick = self.state()["tick"]
        code, r = self.call("/v1/action", {"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, 5.0]})
        self.assertEqual(code, 400, r)
        self.assertIn("IK unreachable", json.dumps(r))
        self.assertEqual(self.state()["tick"], tick)

    def test_the_state_is_built_once_per_command_and_never_served_stale(self):
        b = self.backend
        j, q = self.joint()
        with patch.object(b.env, "get_state", wraps=b.env.get_state) as built:
            a = self.state()
            self.assertEqual(self.state(), a)                               # the same tick: the kept state
            self.assertEqual(self.call("/v1/state", token="policy-token")[0], 403)
            self.assertLessEqual(built.call_count, 1)
            built.reset_mock()
            code, r = self.call("/v1/action", {"type": "named_joint_target", "targets_si": {j: q + 0.01}, "ticks": 3})
            self.assertEqual(code, 200, r)
            # GAR read the state before and after the command (its history) and for the result: one build
            self.assertEqual(built.call_count, 1)
            c = self.state()
            self.assertEqual(built.call_count, 1)
            self.assertEqual(c["tick"], a["tick"] + 3)
            self.assertAlmostEqual(c["joints"][j]["target"], q + 0.01, places=6)
            # a caller's copy is its own
            kept, mine = b.get_state(), b.get_state()
            mine["joints"][j]["target"] = 123.0
            mine["evaluation"]["score"] = 1.0
            self.assertEqual(b.get_state(), kept)
            self.assertEqual(built.call_count, 1)
            # every way the world changes drops the kept state: a command, a checkpoint restore, a reset, a
            # variation switch; and a tick of the world behind the backend's back changes the key
            b.save_checkpoint(label="k")
            code, r = self.call("/v1/action", {"type": "named_joint_target", "targets_si": {j: q + 0.02}, "ticks": 2})
            self.assertEqual(code, 200, r)
            self.assertAlmostEqual(self.state()["joints"][j]["target"], q + 0.02, places=6)
            state, info = b.restore_checkpoint("k")
            self.assertEqual(state["tick"], c["tick"])
            self.assertAlmostEqual(state["joints"][j]["target"], q + 0.01, places=6)
            self.assertEqual(b.get_state()["generation"], state["generation"])
            self.assertAlmostEqual(self.state()["joints"][j]["target"], q + 0.01, places=6)
            state, info = b.reset()
            self.assertEqual((state["tick"], self.state()["tick"], self.state()["generation"]),
                             (0, 0, state["generation"]))
            b.env.world.step()                                              # outside the backend: the world's own tick
            self.assertEqual(b._state_cache["key"][:3], (id(b.env), b.env.generation, 0))
            self.assertNotEqual(b._state_key(), b._state_cache["key"])
        state, info = b.reset(options={"variation_id": 33})
        self.assertEqual((state["config"]["task"], b.get_state()["config"]["task"]), ("pivot", "pivot"))


class RigidBuildContractTest(unittest.TestCase):
    """Each deployment's real world, built rigid-only, against the contract the stand-in test checks
    (test_sewing_robots.DeploymentContractTest.EXPECTED) and against the stand-in's shape."""

    def _check(self, robot):
        reason = _missing_assets(robot)
        if reason:
            self.skipTest(reason)
        from tests.test_sewing_robots import DeploymentContractTest, _stub_world
        arms, gripper, cams = DeploymentContractTest.EXPECTED[robot]
        b = _rigid_backend(robot=robot, cameras=None)
        try:
            w = b.env.world
            self.assertIsNone(w.cloth)
            self.assertEqual(b.arms, arms)
            self.assertEqual("gripper" in b.kinds, gripper)
            self.assertTrue({"named_joint_target", "cartesian_pose", "cartesian_delta", "tool_io"} <= b.kinds)
            self.assertEqual((b.policy_cameras, b.cameras), (cams, cams))
            pa = b.policy_access()
            caps = pa.capabilities()
            self.assertEqual([c.name for c in pa.cameras], list(cams))
            self.assertEqual(caps["hardware"]["arms"], list(arms))
            tau = [f for f in caps["proprioception_fields"] if f.endswith(".external_torque")]
            if robot == "g1":
                self.assertEqual(tau, [])
                self.assertIn("Dex3-1 hands", b.capabilities.notes["gripper"])
                self.assertIn("no balance controller", b.capabilities.notes["control"])
                self.assertIsNone(b.finger_effort_limit)
            else:
                self.assertEqual(len(tau), 7*len(arms))
                self.assertIn("Franka Hand per arm", b.capabilities.notes["gripper"])
                self.assertIn("libfranka", b.capabilities.notes["control"])
                self.assertGreater(b.finger_effort_limit, 0.)
            # every joint the policy is told about is a joint of the built rig, and every arm the IK serves exists
            self.assertEqual(set(caps["hardware"]["joints"]), set(w.joints))
            for arm in arms:
                b._arm(arm)
            # the stand-in world of test_sewing_robots.py is shaped as this deployment builds it
            stub = _stub_world(robot)
            self.assertEqual(set(stub.rig.instances), set(w.rig.instances))
            self.assertEqual(stub.hand_arms, w.hand_arms)
            self.assertEqual(stub.g1, w.g1)
            self.assertEqual({a: len(d) for a, d in stub._arm_mj_dofs.items()},
                             {a: len(d) for a, d in w._arm_mj_dofs.items()})
            for a in w.hand_arms:
                self.assertEqual(len(stub.gripper_joints(a)), len(w.gripper_joints(a)))
                for name in w.gripper_joints(a):
                    self.assertIn(name, w.joints)
        finally:
            b.close()

    def test_dual_fr3(self):
        self._check("dual-fr3")

    def test_single_fr3(self):
        self._check("single-fr3")

    def test_dual_panda(self):
        self._check("dual-panda")

    def test_g1(self):
        self._check("g1")


@unittest.skipUnless(os.environ.get("GAR_SEWING_PHYSICS_TESTS") == "1", "set GAR_SEWING_PHYSICS_TESTS=1 (GPU)")
class PhysicsSwitchTest(unittest.TestCase):
    """A cloth session (FR3 pair, no cameras) switching pivot/seed-0 -> plain-seam/seed-0 with the machine sewing: the
    trace carries the machine's state and its stitches, the replaced world is released, checkpoints are dropped. The
    two start states are prepared on first use after a physics change (minutes each), loaded from the cache after."""

    def test_switch_and_trace_with_the_cloth(self):
        from gym_anything_robotics.actions import ToolIO
        from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        out = Path(tempfile.mkdtemp(prefix="sewvar-"))/"session"
        b = NewtonSewingBackend(robot="dual-fr3", task="pivot", seed=0, cameras=(), output_dir=str(out))
        try:
            ref = b.begin_action_trace({"action_id": "p1", "request_id": "p1"})
            b.act(ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=200))
            s = b.end_action_trace()
            self.assertTrue(s["complete"])
            self.assertEqual(s["retained_samples"], 200)
            store = PhysicsTraceStore(Path(ref["store_path"]), readonly=True)
            try:
                last = store.page("p1", kind="sample", start_tick=200)["items"][0]["data"]
                kinds = {e["data"]["type"] for e in store.page("p1", kind="event", limit=128)["items"]}
            finally:
                store.close()
            self.assertEqual(last["machine"]["mode"], "run")
            self.assertGreater(last["machine"]["foot_load_n"], 0.)
            self.assertEqual(set(last["robot"]["cloth_force_n"]), {"left", "right"})
            self.assertTrue({"machine_io", "machine_mode", "puncture"} <= kinds, kinds)
            b.save_checkpoint(label="c")
            world = weakref.ref(b.env.world)
            _, info = b.reset(options={"variation_id": 10})
            self.assertIsNone(world())
            self.assertEqual((b.config.task, b.list_checkpoints()), ("plain-seam", ()))
            self.assertEqual(info["continuity"], "new_variation")
            b.begin_action_trace({"action_id": "p2", "request_id": "p2"})
            b.act(ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=20))
            s = b.end_action_trace()
            self.assertEqual((s["retained_samples"], s["metadata"]["data"]["task"]), (20, "plain-seam"))
        finally:
            b.close()


if __name__ == "__main__":
    unittest.main()


class JointImpedanceTest(unittest.TestCase):
    """The arms' joint impedance: offered to the FR3 and Panda as the raw operation joint_impedance, bounded as
    libfranka's Robot::setJointImpedance is, refused before any tick when out of range, and acting on the world: a
    joint load deflects the arm further at a lower stiffness."""


    def test_a_lower_stiffness_lets_a_load_deflect_the_arm_further(self):
        from gym_anything_robotics.actions import RawAction
        b = _rigid_backend("single-fr3")
        w = b.env.world
        arm = w.arms[0]
        body = [i for i, lab in enumerate(w.model.body_label) if lab.startswith(arm + "/") and lab.endswith("link7")][0]
        q = [w.joints[n]["q"] for n, _ in w._arm_mj_dofs[arm]]

        def deflection():
            for _ in range(200):
                w.step()
            q0 = w.state.joint_q.numpy()[q].copy()
            w.body_wrench[body, :] = 0.0
            w.body_wrench[body, 2] = -5.0                  # 5 N down on link 7
            for _ in range(1000):
                w.step()
            d = np.abs(w.state.joint_q.numpy()[q] - q0).max()
            w.body_wrench[body, :] = 0.0
            for _ in range(1000):
                w.step()
            return d
        stiff = deflection()
        b.act(RawAction({"op": "joint_impedance", "arm": arm, "stiffness_nm_per_rad": 50}))
        soft = deflection()
        self.assertGreater(soft, 5*stiff)

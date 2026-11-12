"""lockkey-tasks-v2: catalogue, starts, settle, scoring and backend wiring.

Pure Python/NumPy/SciPy: no Newton step, Warp kernel, renderer or simulator. The Panda description assets are not
installed in a plain checkout, so the joint origins of franka_description's panda_arm_hand.urdf stand in for them;
they reproduce every recorded IK witness below to < 0.3 mm. The native LockKeyEnv is replaced by an analytical
joint-to-pose double that runs the unchanged `_evaluate`/`_after_physics_step` source. Constructed states do not
demonstrate physical solvability.
"""

import ast
import json
import math
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from gym_anything_robotics import tasks
from pencap.newton15.kinematics import Kinematics as PandaDescription
from lockkey import lock_specs as LK
from lockkey.newton15.config import Config
from lockkey.newton15.evaluation import UnlockProgress, OPEN_LATERAL
from lockkey.newton15.kinematics import Kinematics
from lockkey.newton15.shackle import LONG_AXIS, geometry_state
from lockkey.task_set import catalogue_v2 as v2
from lockkey.task_set import presets as presets_v1
from lockkey.task_set import scoring as scoring_v1
from lockkey.task_set.catalogue import (DEFAULT_TASK, INSTRUCTIONS, TASKS, TASK_SET_ORDER, UNLOCK_CONTRACT,
                                        CONTRACT as V1, task_resolve, task_setup)
from lockkey.task_set.presets import mid_cycle, snapshot
from lockkey.task_set.presets_v2 import apply_start, placements
from lockkey.task_set.scoring_v2 import TaskProgress, stand_fit
from lockkey.task_set.settle_v2 import settle

# Which starts are recorded mid-cycle or settled, and return-key's spot on the mat: the tasks' setup.py.
MID_CYCLE = tuple(t for t in TASKS if mid_cycle(t))
SETTLED = tuple(t for t in TASKS if getattr(task_setup(t), "SETTLED", False))
MAT_KEY_XY = task_setup("return-key").MAT_KEY_XY

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "benchmarks/roboworld/environments"
V2 = v2.CONTRACT
ADDITIONS = TASKS[1:]

# franka_description panda_arm_hand.urdf joint origins (xyz, rpy): panda_joint1..8, panda_hand_joint.
P = math.pi/2
PANDA_URDF = (((0., 0., .333), (0., 0., 0.)), ((0., 0., 0.), (-P, 0., 0.)), ((0., -.316, 0.), (P, 0., 0.)),
              ((.0825, 0., 0.), (P, 0., 0.)), ((-.0825, .384, 0.), (-P, 0., 0.)), ((0., 0., 0.), (P, 0., 0.)),
              ((.088, 0., 0.), (P, 0., 0.)), ((0., 0., .107), (0., 0., 0.)), ((0., 0., 0.), (0., 0., -math.pi/4)))


def panda_chain(self):
    self.chain = []
    for xyz, rpy in PANDA_URDF:
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = Rotation.from_euler("xyz", rpy).as_matrix(), xyz
        self.chain.append(t)


def setUpModule():
    global panda_patch
    panda_patch = patch.object(PandaDescription, "__init__", panda_chain)
    panda_patch.start()


def tearDownModule():
    panda_patch.stop()


# Bodies of the analytical test doubles. Shapes: key blade, key bow, wrong blade, shackle leg, fingers, shell.
LABELS = ["key", "wrong_key", "shackle", "panda/panda_link7", "panda/panda_leftfinger", "panda/panda_rightfinger"]
KEY, WRONG, SHACKLE, HAND, LEFT, RIGHT = range(6)
SHAPE_LABELS = ["key_blade_0", "key_bow", "wrong_blade_0", "shackle_leg", "panda/l", "panda/r", "shell"]
SHAPE_BODY = [KEY, KEY, WRONG, SHACKLE, LEFT, RIGHT, -1]
BOTH = ((4, 0), (5, 0))
NEAR_STAND, OTHER_STAND = np.array([.16, -.16, .805]), np.array([.16, .16, .805])
ORIGIN = np.array([.10, 0., .85])


class Array:
    def __init__(self, value):
        self.value = np.array(value)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value[:] = value


def contacts(pairs=()):
    return NS(rigid_contact_count=Array([len(pairs)]),
              rigid_contact_shape0=Array(np.array([a for a, _ in pairs], dtype=int)),
              rigid_contact_shape1=Array(np.array([b for _, b in pairs], dtype=int)))


def start():
    q = np.zeros((6, 7))
    q[:, 6] = 1.
    q[KEY, :3], q[WRONG, :3], q[SHACKLE, :3] = NEAR_STAND, OTHER_STAND, ORIGIN
    q[HAND, :3], q[LEFT, :3], q[RIGHT, :3] = (.45, 0., 1.25), (.45, .04, 1.15), (.45, -.04, 1.15)
    return q


def shackle_pose(swivel_deg, rise):
    r = Rotation.from_euler("z", swivel_deg, degrees=True)
    return np.r_[ORIGIN+LONG_AXIS-r.apply(LONG_AXIS)+[0., 0., rise], r.as_quat()]


def key_at(q, xyz, roll_deg=0.):
    q = q.copy()
    q[KEY] = np.r_[xyz, Rotation.from_euler("x", roll_deg, degrees=True).as_quat()]
    return q


def with_hand(q, tcp, gap):
    q = q.copy()
    q[HAND, :3] = np.asarray(tcp)+[0., 0., .1]
    q[LEFT, :3], q[RIGHT, :3] = np.asarray(tcp)+[0., gap, 0.], np.asarray(tcp)-[0., gap, 0.]
    return q


def unlocked(swivel):
    """The near key seated and turned 37 deg; shackle risen its full travel and swung `swivel` degrees."""
    angle = 37.
    q = key_at(start(), (.089, 0., .85), angle)
    q[SHACKLE] = shackle_pose(swivel, Config().shackle_travel)
    return q, angle


class FakeEnv:
    def __init__(self, q, released=False):
        self.config = Config()
        self.bodies = {n: i for i, n in enumerate(LABELS) if not n.startswith("panda")}
        self.builder = NS(body_label=LABELS, shape_body=SHAPE_BODY, shape_label=SHAPE_LABELS)
        self.state = NS(body_q=Array(q))
        self.released, self.start_preset = released, {}
        self.set(q)

    def set(self, q, pairs=()):
        self.state.body_q.assign(q)
        self.contacts = contacts(pairs)


def task_fixture(task):
    q = start()
    stands = {"key": q[KEY].copy(), "wrong_key": q[WRONG].copy()}
    for name, pose in placements(task).items():
        q[KEY if name == "key" else WRONG] = pose
    angle = 0.
    if task in MID_CYCLE:
        saved = snapshot(task)
        q[SHACKLE] = saved["body_poses"]["shackle"]
        angle = math.degrees(saved["joint_coordinates"]["plug"])
    if task == "which-key":
        stands = {"key": q[KEY].copy(), "wrong_key": q[WRONG].copy()}
    env = FakeEnv(q, released=task in ("swing-open", "remove-key"))
    env.start_preset = {"task": task}
    p = UnlockProgress(env) if task == DEFAULT_TASK else TaskProgress(env, task, stands)
    return env, p, p.update(env, angle)


def update(env, progress, q, angle=0., pairs=()):
    env.set(q, pairs)
    result = progress.update(env, angle)
    if not env.released and angle > 30.:
        env.released = True
        result = progress.update(env, angle)
    return result


class CatalogueTests(unittest.TestCase):
    def test_v2_selections_resolve_with_contract_in_task_id_options_and_selection(self):
        for task in ADDITIONS:
            with self.subTest(task=task):
                spec = tasks.resolve("lockkey", {"task": task, "contract": V2})
                self.assertEqual(spec.task_id, f"lockkey/{task}/{V2}")
                self.assertEqual(spec.description, INSTRUCTIONS[task])
                self.assertEqual(spec.launch.backend_options, {"task": task, "contract": V2})
                self.assertEqual(spec.selection, {"task": task, "contract": V2})
                self.assertIsNone(spec.split)
                self.assertEqual(tasks.TaskSpec.from_dict(spec.to_dict()).to_dict(), spec.to_dict())

    def test_released_selections_without_contract_are_unchanged(self):
        self.assertEqual(task_resolve({}), dict(task_id="lockkey/unlock", description=INSTRUCTIONS["unlock"],
                                                backend_options={}, selection={}, split=None))
        self.assertEqual(task_resolve({"task": "unlock"}), task_resolve({}))
        for task in ADDITIONS:
            self.assertEqual(task_resolve({"task": task}), dict(
                task_id=f"lockkey/{task}", description=INSTRUCTIONS[task], backend_options={"task": task},
                selection={"task": task}, split=None))
        with self.assertRaisesRegex(ValueError, "only the task selector"):
            task_resolve({"task": "swing-open", "seed": 0})

    def test_contract_selector_rejects_unknown_contracts_unlock_and_extra_selectors(self):
        for selection in ({"task": "swing-open", "contract": V1}, {"task": "swing-open", "contract": None},
                          {"task": "swing-open", "contract": "lockkey-tasks8"},
                          {"task": "unlock", "contract": V2}, {"contract": V2},
                          {"task": "typo", "contract": V2}, {"task": "swing-open", "contract": V2, "seed": 0}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                tasks.resolve("lockkey", selection)

    def test_benchmark_lists_v1_unchanged_and_the_eight_v2_selections(self):
        raw = tasks.catalog("lockkey")
        self.assertEqual(raw["selectors"], ["task", "contract"])
        self.assertEqual(raw["selector_choices"]["contract"], [V2])
        self.assertEqual(raw["resolver"], "lockkey.task_set.catalogue:task_resolve")
        released = raw["task_set_v1"]["tasks"]
        self.assertEqual([e["selection"] for e in released], [{"task": t} for t in TASK_SET_ORDER])
        for entry in released:
            spec = tasks.resolve("lockkey", entry["selection"])
            self.assertEqual(spec.task_id, entry["task_id"])
            self.assertEqual(entry["contract"], UNLOCK_CONTRACT if entry["selection"]["task"] == "unlock" else V1)
        test_split = json.loads((ROOT / "benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        audited = next(t["selection"] for t in test_split if t["environment_id"] == "lockkey")
        selections = raw["task_set"]["selections"]
        self.assertEqual(selections, [audited if t == "unlock" else {"task": t, "contract": V2}
                                      for t in TASK_SET_ORDER])
        self.assertEqual(audited, {})
        self.assertTrue(raw["task_set"]["note"])
        for task, selection in zip(TASK_SET_ORDER, selections):
            spec = tasks.resolve("lockkey", selection)
            self.assertEqual(spec.task_id, "lockkey/unlock" if task == "unlock" else f"lockkey/{task}/{V2}")
            self.assertEqual(spec.description, INSTRUCTIONS[task])

    def test_v2_modules_live_in_the_released_task_set_package(self):
        for name in ("catalogue_v2", "presets_v2", "scoring_v2", "settle_v2", "adapter_v2"):
            self.assertTrue((ENVIRONMENTS / f"lockkey/task_set/{name}.py").is_file(), name)
        # The mid-cycle snapshots are the released ones; v2 adds no new start data.
        for task in MID_CYCLE:
            np.testing.assert_array_equal(placements(task)["key"], presets_v1.placements(task)["key"])


class ScoreTests(unittest.TestCase):
    def test_all_eight_untouched_starts_are_zero_and_unsuccessful(self):
        for task in TASKS:
            with self.subTest(task=task):
                env, p, out = task_fixture(task)
                self.assertEqual((out["success"], out["score"], out["failure"]), (False, 0., False))
                self.assertFalse(out["unrecoverable"])
                self.assertEqual(out["contract"], UNLOCK_CONTRACT if task == "unlock" else V2)
                if task != "unlock":
                    self.assertEqual(out["task_id"], f"lockkey/{task}/{V2}")

    def test_all_eight_have_constructed_success_and_current_state_failure(self):
        for task in TASKS:
            with self.subTest(task=task):
                env, p, _ = task_fixture(task)
                original_wrong = p.start["wrong_key"].copy()
                q, angle = unlocked(37.)
                q[WRONG] = original_wrong
                if task not in ("insert-key", "return-key"):
                    update(env, p, q, angle)
                if task in ("return-key", "remove-key", "unlock-and-stow"):
                    q[KEY] = p.start["key"]
                out = update(env, p, q, angle)
                self.assertTrue(out["success"], out)
                self.assertEqual(out["score"], 1.)
                q[WRONG, 1] += .05
                out = update(env, p, q, angle)
                self.assertFalse(out["success"])
                self.assertAlmostEqual(out["score"], .9*out["progress"])
                self.assertLessEqual(out["score"], .9)

    def test_seated_but_held_fails_insert_then_release_passes(self):
        env, p, _ = task_fixture("insert-key")
        q = with_hand(key_at(start(), (.089, 0., .85)), (.127, 0., .86), .005)
        out = update(env, p, q, pairs=BOTH)
        self.assertEqual((out["success"], out["score"]), (False, .9))
        self.assertTrue(update(env, p, q)["success"])

    def test_wrong_key_release_cannot_be_repaired_by_later_seating(self):
        for task in ("turn-to-open", "which-key", "unlock-and-stow"):
            env, p, _ = task_fixture(task)
            q, a = unlocked(37.)
            q[KEY] = p.start["key"]
            out = update(env, p, q, a)
            self.assertEqual((out["score"], out["success"], out["unrecoverable"]), (0., False, True))
            q, a = unlocked(37.)
            self.assertEqual(update(env, p, q, a)["score"], 0.)

    def test_same_end_state_different_routes_and_swing_directions(self):
        for task in ("swing-open", "turn-to-open", "which-key", "unlock-and-stow"):
            for swing in (-90., -37., 37., 90.):
                env, p, _ = task_fixture(task)
                q, a = unlocked(swing)
                q[WRONG] = p.start["wrong_key"]
                out = update(env, p, q, a)
                if task == "unlock-and-stow":
                    q[KEY] = p.start["key"]
                    out = update(env, p, q, 0.)
                self.assertTrue(out["success"], (task, swing, out))

    def test_return_progress_is_continuous_to_the_stand(self):
        env, p, _ = task_fixture("return-key")
        target = p.start["key"]
        values = [stand_fit(np.r_[target[:3]+[0., 0., d], target[3:]], target) for d in np.linspace(.17, 0., 100)]
        self.assertEqual((values[0], values[-1]), (0., 1.))
        self.assertLess(max(np.diff(values)), .02)


class ZeroPointTests(unittest.TestCase):
    """Idle drift of the settled start earns nothing; motion past the idle margin does."""

    def test_swing_open_offset_is_measured_from_the_settled_shackle(self):
        offset = lambda q: geometry_state(q[SHACKLE], p.origin)["shackle_short_leg_lateral_offset_m"]
        env, p, out = task_fixture("swing-open")
        lateral = offset(env.state.body_q.numpy())
        self.assertGreater(lateral, .0008)
        self.assertAlmostEqual(p.swing_zero, max(.001, lateral+.0005))
        for swivel in (3.2, 3.6, 5.):
            q, a = unlocked(swivel)
            q[WRONG] = p.start["wrong_key"]
            out = update(env, p, q, a)
            self.assertEqual(out["score"] > 0., offset(q) > lateral+.0005, (swivel, offset(q)))
        # 3.2 deg of swivel stays inside the margin past the settled start (no credit); 5 deg does not.
        self.assertTrue(.001 < offset(unlocked(3.2)[0]) < p.swing_zero < offset(q))

    def test_remove_key_withdrawal_and_turn_to_open_plug_have_idle_margins(self):
        env, p, out = task_fixture("remove-key")
        q = env.state.body_q.numpy()
        start_x = q[KEY, 0]
        for dx, scored in ((.0004, False), (.002, True)):
            q[KEY, 0] = start_x+dx
            self.assertEqual(update(env, p, q)["score"] > 0., scored, dx)
        env, p, out = task_fixture("turn-to-open")
        self.assertAlmostEqual(p.turn_zero, max(0., math.degrees(snapshot("turn-to-open")["joint_coordinates"]["plug"]))+.5)
        q = env.state.body_q.numpy()
        self.assertEqual(update(env, p, q, .45)["score"], 0.)
        self.assertGreater(update(env, p, q, 2.)["score"], 0.)
        # Full turn credit is still reached at the release line, so the chain stays continuous there.
        self.assertEqual(update(env, p, q, 30.)["components"]["turn"], 1.)

    def test_return_key_roll_toward_upright_inside_the_margin_earns_nothing(self):
        env, p, out = task_fixture("return-key")
        self.assertEqual(p.tilt_zero, 85.)
        q = env.state.body_q.numpy()
        start_rotation = Rotation.from_quat(q[KEY, 3:])
        for roll, scored in ((-4., False), (-10., True)):
            q[KEY, 3:] = (start_rotation*Rotation.from_euler("x", roll, degrees=True)).as_quat()
            self.assertEqual(update(env, p, q)["score"] > 0., scored, roll)

    def test_zero_points_are_reported_and_untouched_catalogue_tasks_keep_released_zeros(self):
        for task in ("insert-key", "which-key", "unlock-and-stow"):
            env, p, out = task_fixture(task)
            self.assertEqual(out["zero_points"], {"shackle_offset_m": .001, "plug_deg": 0.,
                                                  "key_withdrawal_x_m": .091, "key_tilt_deg": 90.})

    def test_v1_scoring_is_the_released_one(self):
        env, p, out = task_fixture("swing-open")
        released = scoring_v1.TaskProgress(env, "swing-open", dict(p.start)).update(env, 57.7)
        self.assertEqual(released["contract"], V1)
        self.assertEqual(released["task_id"], "lockkey/swing-open")
        self.assertIn("lock_progress", released)
        self.assertNotIn("zero_points", released)


class DiagnosticFieldTests(unittest.TestCase):
    def test_unlock_chain_fields_are_renamed_on_new_tasks_only(self):
        env, p, out = task_fixture("unlock")
        self.assertIn("lock_progress", out)
        self.assertIn("stage", out)
        for task in ADDITIONS:
            env, p, out = task_fixture(task)
            self.assertNotIn("lock_progress", out)
            self.assertNotIn("stage", out)
            reading = out["unlock_chain_reading"]
            self.assertEqual(reading["contract"], UNLOCK_CONTRACT)
            self.assertIn("not this task's progress", reading["note"])
        env, p, out = task_fixture("swing-open")
        self.assertEqual((out["unlock_chain_reading"]["stage"], out["unlock_chain_reading"]["progress"]),
                         ("released", .8))
        self.assertEqual(out["progress"], 0.)


class PlacementTests(unittest.TestCase):
    def test_flat_key_footprint_fits_mat_clears_stand_and_rests_on_blade_and_bow(self):
        c = Config()
        pose = placements("return-key")["key"]
        np.testing.assert_allclose(pose[:2], MAT_KEY_XY)
        # Key-frame extents of the actual collision shapes (blade prisms, bow/neck hull), read from the built
        # model in the fix round; the blade is 2.0 mm thick, the bow 5.0 mm.
        boxes = {"blade": ([-.00471, -.001, -.00638], [.02989, .001, .00221]),
                 "bow": ([.02689, -.0025, -.00312], [.04889, .0025, .02088])}
        r = Rotation.from_quat(pose[3:])
        low, xyz = {}, []
        for name, (lo, hi) in boxes.items():
            corners = np.array([pose[:3]+r.apply([x, y, z]) for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            low[name] = corners[:, 2].min()
            xyz.append(corners)
        xyz = np.vstack(xyz)
        mat = c.table_height+.0016
        self.assertTrue(np.all(np.abs(xyz[:, 1]) <= .245))
        self.assertLess(xyz[:, 1].max(), -.175)
        # Well inside the mat's x=.27 edge (spurious bow/mat contacts there pushed the key off) and clear of the chest.
        self.assertLess(xyz[:, 0].max(), .27-.07)
        self.assertGreater(xyz[:, 0].min(), .048+.09)
        # Pitched tip-down so that both the blade tip and the bow side touch the mat: nothing falls at the start.
        for name in boxes:
            self.assertGreaterEqual(low[name], mat-2e-5, name)
            self.assertLess(low[name], mat+5e-5, name)
        # v1 keeps its released level pose at x=.21.
        np.testing.assert_allclose(presets_v1.placements("return-key")["key"][:3], [.21, -.20, c.table_height+.0041])

    def test_moved_return_key_is_within_reach_by_offline_ik_witness(self):
        c, k = Config(), Kinematics()
        bow = np.array([c.face_x+c.insertion_travel+.012, 0., c.spine_z+LK.BLADE_HEIGHT/2+.012])
        pose = placements("return-key")["key"]
        target = pose[:3]+Rotation.from_quat(pose[3:]).apply(bow)
        shoulder = np.array([.54, 0., 1.193])
        self.assertLess(np.linalg.norm(target-shoulder), .61)
        self.assertGreater(target[2], .43)
        # Rounded scipy least_squares witnesses (fix round); the released starts' witnesses check the stand-in chain.
        witnesses = {"return-key": [.5505, .7953, .0436, -2.4516, .2892, 3.251, 1.0789],
                     "swing-open": [.0209, .4502, -.0003, -2.6056, .0025, 3.0521, .8035],
                     "near_stand": [.4497, .5686, .0219, -2.6533, .1545, 3.2203, 1.0996]}
        goal_rotation = Rotation.from_euler("y", 180., degrees=True)
        for name, joints in witnesses.items():
            if name == "near_stand":
                point = NEAR_STAND+bow
            else:
                p = placements(name)["key"]
                point = p[:3]+Rotation.from_quat(p[3:]).apply(bow)
            tcp = k.forward(joints)
            self.assertLess(np.linalg.norm(tcp[:3, 3]-point), .001, name)
            self.assertLess((goal_rotation.inv()*Rotation.from_matrix(tcp[:3, :3])).magnitude(), .02, name)


class State:
    def __init__(self, poses, q):
        self.body_q, self.body_qd = Array(poses), Array(np.zeros((len(poses), 6)))
        self.joint_q, self.joint_qd = Array(q), Array(np.zeros_like(q))

    def assign(self, other):
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            getattr(self, name).assign(getattr(other, name).numpy())


class StubNativeEnv:
    """Analytical joint-to-pose test double; cannot advance time or solve contacts."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        labels = LABELS + ["plug", "lid"] + [f"{p}_{i}" for p in ("keypin", "driver") for i in range(5)]
        self.bodies = {n: i for i, n in enumerate(labels) if not n.startswith("panda")}
        names = ["plug"]+[f"{p}_{i}" for p in ("keypin", "driver") for i in range(5)]+["shackle", "lid"]
        self.mechanism_joints = {name: i for i, name in enumerate(names)}
        starts = list(range(12))+[13, 14, 21, 28]
        children = [self.bodies[n] for n in names]+[self.bodies["key"], self.bodies["wrong_key"]]
        self.builder = NS(body_label=labels, shape_body=SHAPE_BODY, shape_label=SHAPE_LABELS,
                          joint_q_start=starts, joint_qd_start=starts)
        self.model = NS(joint_child=Array(children), joint_q_start=Array(starts), joint_qd_start=Array(starts),
                        joint_effort_limit=Array(np.full(28, 20.)), joint_limit_upper=Array(np.zeros(28)))
        q = np.zeros(28)
        q[14:21], q[21:28] = start()[KEY], start()[WRONG]
        poses = np.zeros((len(labels), 7))
        poses[:, 6] = 1.
        poses[:6] = start()
        self.state, self.next_state, self.initial = [State(poses, q) for _ in range(3)]
        self.control = NS(joint_target_q=Array(q.copy()))
        self.initial_targets = q.copy()
        self.targets = q.copy()
        self.contacts = contacts()
        self.newton = NS(eval_fk=self.fk, ModelFlags=NS(JOINT_DOF_PROPERTIES=1))
        self.solver = NS(_update_mjc_data=Mock(), _mujoco=NS(mj_forward=Mock()),
                         mj_data=object(), mj_model=object(), notify_model_changed=Mock())
        self.tick, self.generation, self.released, self.trace_callback = 0, 0, False, None
        self.joints, self.fault = {}, None
        self.fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.initial.assign(self.state)
        self.progress = UnlockProgress(self)
        self.evaluation = self._evaluate()

    def fk(self, model, joint_q, joint_qd, state):
        c, q, b = self.config, joint_q.numpy(), self.bodies
        poses = state.body_q.numpy()
        origin = np.array([.1, 0., c.table_height+.1])
        r = Rotation.from_euler("x", q[0])
        poses[b["plug"]] = np.r_[origin, r.as_quat()]
        for i, depth in enumerate(c.bitting):
            length = LK.KEY_PIN_LEN[int(depth)]
            kpz = c.keyway_top+length/2+.0003
            local = [c.chambers_x[i], 0., kpz+q[1+i]]
            poses[b[f"keypin_{i}"]] = np.r_[origin+r.apply(local), r.as_quat()]
            poses[b[f"driver_{i}"]] = np.r_[origin+[c.chambers_x[i], 0., kpz+length/2+LK.DRIVER_LEN/2+.0002+q[6+i]],
                                            0., 0., 0., 1.]
        r = Rotation.from_euler("z", q[12])
        poses[b["shackle"]] = np.r_[origin+LONG_AXIS-r.apply(LONG_AXIS)+[0., 0., q[11]], r.as_quat()]
        poses[b["key"]], poses[b["wrong_key"]] = q[14:21], q[21:28]
        state.body_q.assign(poses)

    def get_state(self):
        return {"evaluation": self.evaluation, "generation": self.generation}

    def reset(self):
        generation = self.generation+1
        callback = self.trace_callback
        StubNativeEnv.__init__(self, self.config)
        self.generation, self.trace_callback = generation, callback
        return self.get_state()

    def _step(self, *args, **kwargs):
        raise AssertionError("A stubbed test must never step the simulator")

    def close(self):
        pass


# Execute only these two methods from the unchanged native source. Neither steps physics; together they are the
# release event the task adapters use.
NATIVE = ENVIRONMENTS / "lockkey/newton15/env.py"
tree = ast.parse(NATIVE.read_text())
native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LockKeyEnv")
for method in (n for n in native.body if isinstance(n, ast.FunctionDef) and n.name in ("_evaluate", "_after_physics_step")):
    namespace = {"np": np, "__name__": "lockkey.newton15.env", "__package__": "lockkey.newton15"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(NATIVE), "exec"), namespace)
    setattr(StubNativeEnv, method.name, namespace[method.name])


class SteppingStub(StubNativeEnv):
    """A settle in which nothing moves: native ticks that leave every pose and speed as they are."""
    def __init__(self, config):
        super().__init__(config)
        self.steps = 0

    def _step(self, *args, **kwargs):
        self.tick += 1
        self.steps += 1
        self._after_physics_step()


def settled_stub(task, **changes):
    env = SteppingStub(Config())
    stands, env.start_preset = apply_start(env, task)
    env.progress = TaskProgress(env, task, stands)
    env.evaluation = env._evaluate()
    for name, value in changes.items():
        setattr(env, name, value)
    return env


def contact(env, shape0, shape1, normal, offset):
    poses = env.state.body_q.numpy()
    env.contacts = NS(rigid_contact_count=Array([1]), rigid_contact_shape0=Array([shape0]),
                      rigid_contact_shape1=Array([shape1]), rigid_contact_normal=Array([normal]),
                      rigid_contact_point0=Array([poses[KEY, :3]+offset]),
                      rigid_contact_point1=Array([[0., 0., 0.]]))


class SettleTests(unittest.TestCase):
    def test_constructed_starts_settle_hands_off_and_restart_at_tick_zero(self):
        self.assertEqual(set(SETTLED), set(MID_CYCLE) | {"return-key"})
        for task in SETTLED:
            with self.subTest(task=task):
                env = settled_stub(task)
                callback = Mock()
                env.trace_callback = callback
                env.targets[:] += .1                      # the robot is commanded back to HOME for the settle
                record = settle(env, task)
                self.assertEqual((env.steps, env.tick, record["ticks"]), (960, 0, 960))
                np.testing.assert_allclose(env.targets, env.initial_targets)
                callback.assert_not_called()
                self.assertIs(env.trace_callback, callback)
                self.assertEqual(record["key_moved_m"], 0.)
                self.assertEqual(record["released"], task in ("swing-open", "remove-key"))
        for task in ("insert-key", "which-key", "unlock-and-stow"):
            env = settled_stub(task)
            self.assertIsNone(settle(env, task))
            self.assertEqual(env.steps, 0)

    def test_invalid_settled_starts_raise(self):
        env = settled_stub("turn-to-open", released=True)
        with self.assertRaisesRegex(RuntimeError, "latch released"):
            settle(env, "turn-to-open")
        env = settled_stub("swing-open")
        qd = env.state.body_qd.numpy()
        qd[KEY, 0] = .1
        env.state.body_qd.assign(qd)
        with self.assertRaisesRegex(RuntimeError, "key still moving at 0.100 m/s"):
            settle(env, "swing-open")
        env = settled_stub("remove-key")
        q = env.state.joint_q.numpy()
        q[14] += .01                                      # key 10 mm out of its seat
        env.state.joint_q.assign(q)
        env.fk(env.model, env.state.joint_q, env.state.joint_qd, env.state)
        with self.assertRaisesRegex(RuntimeError, "no longer seated"):
            settle(env, "remove-key")
        for depth, raises in ((.0002, False), (.0012, True)):
            env = settled_stub("return-key")
            contact(env, 6, 0, [0., 0., 1.], [0., 0., depth])   # key blade inside the fixed shell along +z
            if raises:
                with self.assertRaisesRegex(RuntimeError, "penetrates 1.200 mm"):
                    settle(env, "return-key")
            else:
                # Beyond the design's 0.1 mm bar: recorded, not served as an error (settle_v2 docstring).
                record = settle(env, "return-key")
                self.assertAlmostEqual(record["max_penetration_m"], depth)
                self.assertFalse(record["spec"]["penetration_le_0_1_mm"])
                self.assertTrue(record["spec"]["key_at_rest"])
        env = settled_stub("return-key")
        q = env.state.body_q.numpy()
        q[KEY, 0] += .006
        env.state.body_q.assign(q)
        with self.assertRaisesRegex(RuntimeError, "key left its place on the mat"):
            settle(env, "return-key")
        env = settled_stub("return-key", tick=5)
        with self.assertRaisesRegex(RuntimeError, "newly placed"):
            settle(env, "return-key")

    def test_seated_bow_in_the_shell_is_bounded_by_the_carried_overlap(self):
        for depth, ok in ((.0003, True), (.0005, False)):
            env = settled_stub("turn-to-open")
            contact(env, 6, 1, [1., 0., 0.], [depth, 0., 0.])
            if ok:
                record = settle(env, "turn-to-open")
                self.assertAlmostEqual(record["shell_bow_overlap_m"], depth)
                self.assertEqual(record["max_penetration_m"], 0.)
            else:
                with self.assertRaisesRegex(RuntimeError, "into the shell collider"):
                    settle(env, "turn-to-open")


class BackendTests(unittest.TestCase):
    def backends(self, stack, **options):
        """Construct every task through the GAR backend on the analytical double; yields (task, backend)."""
        from gym_anything_robotics.backends.newton_lockkey import NewtonLockKeyBackend
        env_module = ModuleType("lockkey.newton15.env")
        env_module.LockKeyEnv = StubNativeEnv
        catalogue_module = ModuleType("lockkey.newton15.catalog")
        catalogue_module.Catalog = lambda env: NS()
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="lockkey-v2-tests-"))
        stack.enter_context(patch.dict(sys.modules, {
            "lockkey.newton15.env": env_module, "lockkey.newton15.catalog": catalogue_module}))
        for name in ("lockkey.task_set.adapter", "lockkey.task_set.adapter_v2"):
            sys.modules.pop(name, None)
        for task in TASKS:
            selection = options if task != "unlock" else {}
            yield task, NewtonLockKeyBackend(task=task, cameras=(), recording=False,
                                             output_dir=Path(tmp)/task, **selection)

    def test_v2_contract_builds_settled_starts_through_the_backend_and_on_reset(self):
        settled = []

        def fake_settle(env, task):
            settled.append(task)
            return {"ticks": 960} if task in SETTLED else None
        with ExitStack() as stack:
            stack.enter_context(patch("lockkey.task_set.settle_v2.settle", fake_settle))
            for task, backend in self.backends(stack, contract=V2):
                with self.subTest(task=task):
                    env = backend.env
                    self.assertEqual(backend.policy_access().instruction().split(" Operate")[0], INSTRUCTIONS[task])
                    self.assertEqual((env.tick, env.evaluation["score"], env.evaluation["success"]), (0, 0., False))
                    self.assertEqual(env.released, task in ("swing-open", "remove-key"))
                    self.assertEqual(type(env) is StubNativeEnv, task == "unlock")
                    note = backend.capabilities.notes["qualification"]
                    if task == "unlock":
                        self.assertEqual(env.evaluation["contract"], UNLOCK_CONTRACT)
                        self.assertNotIn(V2, note)
                    else:
                        self.assertEqual(type(env).__name__, "TaskLockKeyEnvV2")
                        self.assertEqual((env.evaluation["contract"], env.evaluation["task_id"]),
                                         (V2, f"lockkey/{task}/{V2}"))
                        self.assertIn(V2, note)
                        self.assertEqual(env.start_preset.get("settle"), {"ticks": 960} if task in SETTLED else None)
                        self.assertEqual(env.start_preset["settle_checked"], task in SETTLED)
                    if task == "return-key":
                        np.testing.assert_allclose(env.state.body_q.numpy()[KEY], placements("return-key")["key"])
                    start_poses = env.state.body_q.numpy()
                    q = env.state.joint_q.numpy()
                    q[14] += .1
                    env.state.joint_q.assign(q)
                    callback = Mock()
                    env.trace_callback = callback
                    out, info = backend.reset()
                    np.testing.assert_allclose(env.state.body_q.numpy(), start_poses)
                    self.assertEqual((out["evaluation"]["score"], out["generation"]), (0., 1))
                    callback.assert_not_called()
                    self.assertEqual(info, {"continuity": "reset"})
                    backend.close()
        # Every v2 task is prepared at construction and again on reset; unlock never.
        self.assertEqual(settled, [t for t in ADDITIONS for _ in range(2)])

    def test_without_contract_the_backend_builds_the_released_v1_tasks(self):
        with ExitStack() as stack:
            settle_v2 = stack.enter_context(patch("lockkey.task_set.settle_v2.settle"))
            for task, backend in self.backends(stack):
                with self.subTest(task=task):
                    env = backend.env
                    self.assertEqual(type(env) is StubNativeEnv, task == "unlock")
                    self.assertEqual(env.evaluation["score"], 0.)
                    if task != "unlock":
                        self.assertEqual(type(env).__name__, "TaskLockKeyEnv")
                        self.assertEqual((env.evaluation["contract"], env.evaluation["task_id"]),
                                         (V1, f"lockkey/{task}"))
                        self.assertNotIn("settle", env.start_preset)
                        self.assertIn("lock_progress", env.evaluation)
                        self.assertEqual(backend.capabilities.notes["qualification"],
                                         "Task addition checked in constructed states only; settling, contacts and "
                                         "rendering are not yet checked in the simulator")
                    if task == "return-key":
                        np.testing.assert_allclose(env.state.body_q.numpy()[KEY],
                                                   presets_v1.placements("return-key")["key"])
                    backend.close()
            settle_v2.assert_not_called()

    def test_contract_is_validated_before_anything_is_built(self):
        from gym_anything_robotics.backends.newton_lockkey import NewtonLockKeyBackend
        with tempfile.TemporaryDirectory(prefix="lockkey-v2-tests-") as tmp:
            for task, contract in (("unlock", V2), ("swing-open", V1), ("swing-open", "typo")):
                with self.subTest(task=task, contract=contract), self.assertRaises(ValueError):
                    NewtonLockKeyBackend(task=task, contract=contract, cameras=(), recording=False,
                                         output_dir=Path(tmp)/"never")
            self.assertFalse((Path(tmp)/"never").exists())

    def test_v2_midcycle_config_mismatch_and_late_placement_fail(self):
        env = StubNativeEnv(Config(bitting="12345"))
        with self.assertRaisesRegex(ValueError, "reference mechanism"):
            apply_start(env, "swing-open")
        env = StubNativeEnv(Config())
        env.tick = 1
        with self.assertRaisesRegex(RuntimeError, "reset"):
            apply_start(env, "which-key")


if __name__ == "__main__":
    unittest.main()

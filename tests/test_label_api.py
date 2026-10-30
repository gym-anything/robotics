"""label-application API contract on a fake label world (no Newton, no GPU, seconds).

History trace: the label's per-command trace is in GAR's PhysicsTraceStore format and its reference carries
`store_path`, so `GET /v1/history/{id}/trace` and `/trace/events` (what `gar export --include-traces` reads) serve it.
A reference without `store_path` would make every trace request, and with it the evaluation's history export, fail
with HTTP 400 'store_path'.

Episode end: after a tear ends the episode, every command is refused before it runs (committed: false), and a policy
caller is told why. Without the refusal, later commands would commit and move the robot with no end-of-episode
signal.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import numpy as np

from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.http_server import GARHTTPServer, GARHTTPService
from tests.test_http_server import HTTPFakeBackend


def fake_label_env():
    """The attributes label_application.trace.Trace reads from LabelEnv / World / LabelSim."""
    surface = lambda name: SimpleNamespace(name=name)
    label = SimpleNamespace(
        colliders=[SimpleNamespace(name="carton", role="item", body=1, surface=surface("kraft_carton")),
                   SimpleNamespace(name="right_fr3_rightfinger_pad_118", role="pad", body=2, surface=surface("pad"))],
        bond=np.array([0, 0, 0, -1, -1]), v=np.zeros((5, 3)), G=np.zeros(5), front_speed=np.zeros(5),
        reactions={"carton": (np.array([0., 0., -1.]), np.zeros(3), 1)}, torn=False, pockets=[], stats={"steps": 0})
    arrays = SimpleNamespace(numpy=lambda: np.zeros(3))
    world = SimpleNamespace(
        steps=0, tick=0, label=label, label_frozen=None, reflexes=[], targets=np.zeros(3),
        model=SimpleNamespace(body_label=["world", "carton", "right/fr3/fr3_rightfinger"]),
        scene=SimpleNamespace(items={"carton": {"body": 1}}), joints={"right/fr3_joint1": {"q": 0, "dof": 0}},
        state=SimpleNamespace(joint_q=arrays, joint_qd=arrays), last_pad_loads={2: 1.5})
    env = SimpleNamespace(world=world, generation=0, trace_callback=None,
                          config=SimpleNamespace(label_dt=.002, label_substeps=2, rigid_dt=.001))
    return env


def step(env, ticks=1):
    for _ in range(ticks):
        env.world.steps += 1
        env.world.tick += 2
        if env.trace_callback:
            env.trace_callback(env)


class LabelTraceBackend(HTTPFakeBackend):
    """GAR's fake HTTP backend with the label's real Trace on a fake label world. One action = 4 label steps: the
    pad bonds a node at the 2nd, the label tears at the 3rd (and the trace records both as events)."""

    def __init__(self, root):
        super().__init__(root)
        from label_application.trace import Trace
        self.label_env = fake_label_env()
        self.trace = Trace(self.label_env, root/"physics.sqlite3")

    def begin_action_trace(self, context):
        return self.trace.begin(context)

    def end_action_trace(self):
        return self.trace.finish()

    def act(self, action):
        e = self.label_env
        step(e)
        e.world.label.bond[3] = 1
        step(e)
        e.world.label.torn = True
        e.world.label_frozen = dict(tick=e.world.steps + 1, time_s=(e.world.steps + 1)*.002, failed_triangles=5,
                                    reason="label_torn")
        step(e, 2)
        return super().act(action)

    def close(self):
        self.trace.close()
        super().close()


class LabelServerTest(unittest.TestCase):
    backend_class = LabelTraceBackend

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.backend = self.backend_class(Path(self.tmp.name))
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
        self.tmp.cleanup()

    def get(self, path, token="test-token", body=None):
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        req = urllib.request.Request(self.root + path, data=data, headers=headers,
                                     method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())


class LabelTraceHTTPTest(LabelServerTest):
    def test_history_trace_routes_read_the_label_trace(self):
        act = self.get("/v1/action", body={"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]})
        rec = self.get("/v1/history?request_id=" + act["request_id"])["records"][0]
        page = self.get(f"/v1/history/{rec['id']}/trace")      # HTTP 400 'store_path' without store_path
        ref = self.get("/v1/history/" + rec["id"])["record"]["trace"]
        self.assertTrue(ref["available"])
        self.assertEqual(ref["id"], rec["id"])
        self.assertEqual(Path(ref["store_path"]), (Path(self.tmp.name)/"physics.sqlite3").resolve())
        self.assertTrue(ref["finalization"]["complete"])
        self.assertTrue(page["available"])
        self.assertTrue(page["summary"]["complete"], page["summary"])
        self.assertEqual(page["summary"]["dt"], .002)
        self.assertEqual([it["tick"] for it in page["items"]], [1, 2, 3, 4])      # every label step, none missing
        first, second = page["items"][0]["data"], page["items"][1]["data"]
        self.assertEqual(first["bonded"], {"carton": 3})
        self.assertEqual(second["bonded"], {"carton": 3, "right_fr3_rightfinger_pad_118": 1})
        self.assertEqual(first["pad_loads_n"], {"right/fr3/fr3_rightfinger": 1.5})
        self.assertEqual(first["reactions"], {"carton": [0., 0., -1.]})
        self.assertEqual((first["rigid_tick"], first["terminated"], page["items"][2]["data"]["terminated"]),
                         (2, False, True))
        events = self.get(f"/v1/history/{rec['id']}/trace/events")["items"]
        self.assertEqual([(e["data"]["type"], e["tick"]) for e in events], [("bond_onset", 2), ("label_torn", 3)])
        self.assertEqual(events[0]["data"]["collider"], "right_fr3_rightfinger_pad_118")
        self.assertEqual(events[1]["data"]["reason"], "label_torn")
        # a second command gets its own trace; nothing of the first leaks into it
        act = self.get("/v1/action", body={"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]})
        rec2 = self.get("/v1/history?request_id=" + act["request_id"])["records"][0]
        page2 = self.get(f"/v1/history/{rec2['id']}/trace")
        self.assertEqual([it["tick"] for it in page2["items"]], [5, 6, 7, 8])
        self.assertTrue(page2["summary"]["complete"])

    def test_trace_rejects_overlapping_commands(self):
        tr = self.backend.trace
        tr.begin({"action_id": "a", "request_id": "r"})
        with self.assertRaises(RuntimeError):
            tr.begin({"action_id": "b", "request_id": "r"})
        self.assertEqual(tr.finish()["id"], "a")
        self.assertIsNone(tr.finish())


TORN = dict(tick=3031, time_s=6.062, failed_triangles=5, reason="label_torn")
REASON = "episode over: the label tore at tick 3031 (t = 6.062 s), which ends the episode; no further command runs"


def label_backend(frozen):
    """The real backend's command path on a fake world (no Newton): what _act reads before and after env.act."""
    from gym_anything_robotics.backends.newton_label_application import NewtonLabelApplicationBackend
    be = object.__new__(NewtonLabelApplicationBackend)
    world = SimpleNamespace(label_frozen=frozen, steps=3031, targets=np.zeros(3), grip_force={}, grasp_latched={},
                            reflexes=[], gripper_joints=lambda arm: (f"{arm}/fr3_finger_joint1", f"{arm}/fr3_finger_joint2"))
    world.set_grip_force = mock.Mock()
    state = dict(evaluation=dict(score=.1), termination=None, label_solver={}, terminated=False)
    be.env = SimpleNamespace(lock=threading.RLock(), world=world, act=mock.Mock(return_value=state))
    be.arms, be.cameras, be.config = ("left", "right"), (), SimpleNamespace(label_dt=.002)
    return be


class LabelEpisodeEndTest(unittest.TestCase):
    def test_commands_after_a_tear_are_refused_before_they_run(self):
        from gym_anything_robotics.actions import CartesianDelta, Gripper, NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        be = label_backend(dict(TORN))
        for action in (Gripper(arm="right", inward_effort_n=20., duration_s=.1),
                       CartesianDelta(arm="right", delta_m=(0., .1, .12), duration_s=.6),
                       NamedJointTarget({"right/fr3_joint1": .1}, ticks=10)):
            for call in (be.act, be.act_policy):
                with self.assertRaises(ActionRejectedError) as cm:
                    call(action)
                self.assertEqual(cm.exception.policy_reason, REASON)
        be.env.act.assert_not_called()                          # no tick ran
        be.env.world.set_grip_force.assert_not_called()         # no target or force was written
        self.assertEqual(be.env.world.steps, 3031)

    def test_commands_before_a_tear_run(self):
        from gym_anything_robotics.actions import Gripper
        be = label_backend(None)
        out = be.act(Gripper(arm="right", half_width_m=.02, duration_s=.1))
        be.env.act.assert_called_once()
        self.assertEqual(be.env.act.call_args.kwargs["ticks"], 50)
        self.assertFalse(out.terminated)


class EndedEpisodeBackend(HTTPFakeBackend):
    """GAR's fake HTTP backend whose commands go through the label backend after a tear, with a policy profile."""

    def __init__(self, root):
        super().__init__(root)
        from gym_anything_robotics.policy_access import PolicyAccess, ROBOT_ACTIONS
        self.label = label_backend(dict(TORN))
        self.act = self.act_policy = self.label.act
        self.profile = PolicyAccess(robot="yam", cameras=self.capabilities.camera_specs,
                                    proprioception={("tick",): None}, action_types=ROBOT_ACTIONS,
                                    instruction=lambda: "Smooth down the lifted corner of this shipping label.",
                                    hardware={})

    def policy_access(self):
        return self.profile


class LabelEpisodeEndHTTPTest(LabelServerTest):
    backend_class = EndedEpisodeBackend

    def setUp(self):
        super().setUp()
        from gym_anything_robotics.access import APIKeys
        self.server.keys = APIKeys("test-token", "policy-token")

    def refused(self, token):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/v1/action", token=token,
                     body={"type": "cartesian_delta", "arm": "right", "delta_m": [0, .1, .12], "duration_s": .6})
        self.assertEqual(cm.exception.code, 400)
        return json.loads(cm.exception.read())

    def test_policy_is_told_the_episode_ended(self):
        body = self.refused("policy-token")
        self.assertEqual((body["ok"], body["committed"], body["error_type"]), (False, False, "ActionRejectedError"))
        self.assertTrue(body["error"].endswith("Reason: " + REASON), body["error"])
        self.assertEqual(self.refused("test-token")["error"], REASON)
        self.backend.label.env.act.assert_not_called()
        self.assertTrue(self.get("/v1/health", token="test-token")["ok"])      # a refusal, not a session fault


if __name__ == "__main__":
    unittest.main()

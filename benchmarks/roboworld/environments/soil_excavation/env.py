"""Persistent soil-excavation environment: robot-only commands on exact 0.5 ms ticks, explicit faults, checkpoints.

One command = joint targets (arms, Spine, fingers), an optional base twist and grip-force changes, held for N ticks
(classroom_arrangement's command model, unchanged). Soil moves only through the tools', inclusions' and robot's
contacts with it (the MPM coupling); assisted fixtures (probes, demos) attach to world.fixture and are never reachable
from the policy interface.

Start state (painting's rule, extended to two-handed tools): the task's tool is already in a valid grasp -- each hand's
TCP at an IK solution for its grip on the tool placed at the task's start pose, the fingers closed in force mode -- so
the task is the digging, not acquiring a 1.9 kg spade from the walkway. Pre-dug starts (a set root ball, a braced post,
an open pipe trench) relocate the excavated particles loose onto the tarp (world.build_particles). The reset settle runs
the soil alone with the bodies held, then the coupled world, until every free body is slower than 2 mm/s (classroom
rule) and the soil's fastest particle slower than 5 mm/s.
"""

import copy
import math
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from .config import Config
from .world import SoilWorld

# TCP frame: z approach, y finger travel (classroom). Grip frames below give the TCP pose in the TOOL frame.
# A shaft along tool z grasped from the tool's -x side (behind the blade's front face), fingers closing across tool y,
# TCP x up the shaft:
def shaft_grip(z, x_off=0.0, y_off=0.0, approach=(1., 0., 0.), up=(0., 0., 1.)):
    a = np.asarray(approach, float); a /= np.linalg.norm(a)
    u = np.asarray(up, float); u = u - a*float(u @ a); u /= np.linalg.norm(u)
    R = np.stack([u, np.cross(a, u), a], 1)          # columns: x_tcp (up the shaft), y_tcp, z_tcp (approach)
    return dict(R=R, p=np.array([x_off, y_off, z]))


# grips per arm (TCP pose in the tool frame); alternatives are tried in order with the Spine heights of _place_held
def _fan(z_list, sign, angles=(25., 45., 0., 60.), x_off=0.0):
    """Shaft grips approaching from behind the shaft, swung toward the arm's own side by each angle (the right arm
    from the robot's right-rear: tool -y): two hands on one vertical shaft from straight behind put the elbows into
    each other (left link5 against right link4)."""
    out = []
    for a in angles:
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        for z in z_list:
            out.append(shaft_grip(z, x_off=x_off, approach=(c, sign*s, 0.)))   # hand at (-c, -sign s): the arm's own side
    return out


TS_SHAFT_X = -math.sin(math.radians(P.TRENCH_SHOVEL["lift_deg"]))*P.TRENCH_SHOVEL["head"][1]
TOOL_START = {
    "spade": dict(height=0.03, grips=dict(right=_fan((0.52, 0.58), 1.), left=_fan((0.80, 0.74), -1.))),
    # the trenching shovel's handle rises from the socket at the top of the blade, which leans back by the lift angle:
    # the shaft is 90 mm behind the edge (grips at x = 0 put both hands 5-6 cm into the handle: a 8 kN start impulse)
    "trench_shovel": dict(height=0.03, grips=dict(right=_fan((0.80, 0.72), 1., x_off=TS_SHAFT_X),
                                                  left=_fan((1.10, 1.02), -1., x_off=TS_SHAFT_X))),
    # the trowel's hollow faces the robot (tool x = -forward): the hand comes from the robot's side, swung outward
    "trowel": dict(height=0.04, grips=dict(right=[shaft_grip(z, approach=(-math.cos(math.radians(a)), -math.sin(math.radians(a)), 0.))
                                                  for a in (0., 25., 45., 60.) for z in (0.25, 0.22)],
                                           left=[shaft_grip(z, approach=(-math.cos(math.radians(a)), math.sin(math.radians(a)), 0.))
                                                 for a in (0., 25., 45., 60.) for z in (0.25, 0.22)])),
    # the 2x4 is gripped across its 38 mm side with the TCP 14 mm out of its 89 mm depth toward the hand (centred, the
    # hand's body sat 7 mm into the board's face: a 300 N start contact, backfill-pipe stills)
    "tamper": dict(height=0.05, grips=dict(left=[shaft_grip(0.80, y_off=0.014, approach=(0., -1., 0.)),
                                                 shaft_grip(0.70, y_off=0.014, approach=(0., -1., 0.))],
                                           right=[shaft_grip(0.80, y_off=0.014, approach=(0., -1., 0.))])),
    "mattock": dict(height=0.10, grips=dict(right=_fan((0.09, 0.14), 1.), left=_fan((0.09, 0.14), -1.))),
    "digger": dict(height=0.03, grips=dict(right=[shaft_grip(0.95, x_off=0.062, approach=(0., 1., 0.)),
                                                  shaft_grip(0.85, x_off=0.058, approach=(0., 1., 0.))],
                                           left=[shaft_grip(0.95, x_off=-0.062, approach=(0., 1., 0.)),
                                                 shaft_grip(0.85, x_off=-0.058, approach=(0., 1., 0.))])),
    "drill": dict(height=0.03, grips=dict(right=[dict(R=np.stack([np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                                                                np.array([0., 1., 0.])], 1), p=np.array([-0.14, 0., -0.005]))],
                                          left=[shaft_grip(0.13, approach=(1., 0., 0.), up=(0., 1., 0.)) | dict(
                                              p=np.array([0., 0.13, 0.04]))])),
}
SECOND = {"set-post": ("tamper", "left"), "backfill-pipe": ("tamper", "left")}


class SoilEnv:
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True, device=None):
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        self.spec = spec or L.setup(c.task, c.seed, moisture=c.moisture, topsoil=c.topsoil, stones=c.stones)
        self.spec.setdefault("glides", "nylon")
        from .robot_setup import check_task
        check_task(c.robot, c.task)
        self.world = w = SoilWorld(c, self.spec, device=device or c.device)
        from classroom_arrangement.kinematics import ArmKinematics
        self.ik = ArmKinematics(w)
        self.generation = 0
        self.hold_record = {}
        self._place_held()
        if self.spec.get("braces"):
            w.set_braces(self.spec["braces"])
        if settle:
            self._settle()
        from .task import Evaluator
        self.evaluator = Evaluator(self)
        self.initial = self.checkpoint()

    # ------------------------------------------------------------------ start state
    def _set_joint(self, name, value):
        w = self.world
        j = w.joints[name]["q"]
        q = w.state.joint_q.numpy(); q[j] = value; w.state.joint_q.assign(q)
        t = w.control.joint_target_q.numpy(); t[j] = value; w.control.joint_target_q.assign(t)
        w.targets[j] = value

    def _fk(self):
        w = self.world
        w.newton.eval_fk(w.model, w.state.joint_q, w.state.joint_qd, w.state)
        w.next_state.assign(w.state)
        w.sync_solver()

    def _place_body(self, name, p, R, at_rest=False):
        w = self.world
        b = w.scene.bodies[name]
        jq = w.model.joint_q_start.numpy()
        jchild = w.model.joint_child.numpy()
        jt = w.model.joint_type.numpy()
        import newton
        jfree = [j for j in range(len(jchild)) if jchild[j] == b and int(jt[j]) == int(newton.JointType.FREE)][0]
        q = w.state.joint_q.numpy()
        s = int(jq[jfree])
        q[s:s + 3] = p
        q[s + 3:s + 7] = Rotation.from_matrix(R).as_quat()
        w.state.joint_q.assign(q)
        qd = w.state.joint_qd.numpy(); qd[:] = np.where(np.isfinite(qd), qd, 0.0)
        if at_rest:
            sd = int(w.model.joint_qd_start.numpy()[jfree])
            qd[sd:sd + 6] = 0.0
        w.state.joint_qd.assign(qd)

    def tool_start_pose(self, tool):
        """World pose (p, R) of a held tool at the start: at the task's target, a few cm above the surface."""
        sp = self.spec
        rx, ry, yaw = sp["robot"]
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.])
        side = np.array([-fwd[1], fwd[0], 0.])          # robot's left
        target = sp.get("target") or (sp.get("targets") or [None])[0]
        if target is None and sp.get("target_line"):
            # the point of the line in front of the robot (the line's first end was 0.75 m to its side: no start grasp)
            (x0, y0), (x1, y1) = sp["target_line"]
            t = np.clip(((rx - x0)*(x1 - x0) + (ry - y0)*(y1 - y0))/max((x1 - x0)**2 + (y1 - y0)**2, 1e-9), 0., 1.)
            target = (x0 + t*(x1 - x0), y0 + t*(y1 - y0))
        if target is None and sp.get("target_square"):
            (x0, y0), (x1, y1) = sp["target_square"]
            target = (0.5*(x0 + x1), 0.5*(y0 + y1))
        if target is None and sp.get("pipe"):
            target = (rx, sp["pipe"]["y"])           # over the irrigation lateral, in front of the robot
        if target is None:
            b = sp["block"]
            target = (0.5*(b["lo"][0] + b["hi"][0]), b["lo"][1] + 0.35*(b["hi"][1] - b["lo"][1]))
        target = np.array([target[0], target[1], L.GRADE])
        up = np.array([0., 0., 1.])
        h = TOOL_START.get(tool, {}).get("height", 0.03)
        off = np.asarray(self.spec.get("start_offset", {}).get(tool, (0., 0., 0.)), float)
        if tool in ("spade", "trench_shovel"):
            # shaft vertical; blade front (+x tool) facing away from the robot; tool y = robot's left
            R = np.stack([fwd, side, up], 1)
            p = target + up*h
        elif tool == "tamper":
            # the 2x4 upright, its 38 mm side (tool x) across the fingers, beside the trowel work on the left
            R = np.stack([side, -fwd, up], 1)
            p = target + up*h + 0.30*side - 0.05*fwd
        elif tool == "trowel":
            # tip down, handle up, the scoop's hollow (+x tool) toward the robot; over a pre-dug hole or trench the
            # trowel waits at its near edge, beside the post / root ball / pipe (at the target itself it intersected
            # the post: no start grasp for set-post, w_benches first run)
            R = np.stack([-fwd, -side, up], 1)
            pd = sp.get("predug")
            if pd and pd.get("kind") in ("post_hole", "planting_hole"):
                target = target - fwd*(pd["radius"] + 0.06)
            elif pd and pd.get("kind") == "pipe_trench":
                target = target - fwd*(pd["width"]/2 + 0.06)
            p = target + up*h
        elif tool == "mattock":
            # carried head down, handle up, pick toward the work, its head h above the target: held level by one hand
            # 0.3 m from a 0.68 kg head the mattock turned in the pinch (2.2 N m against ~1.5 N m) and struck the
            # hardpan during the settle (first stills: 21 m/s guard)
            R = np.stack([fwd, -side, -up], 1)
            p = target + up*(h + P.MATTOCK["handle"] + 0.06)
        elif tool == "digger":
            # pin axis along the robot's forward direction, half a on the robot's right, blade tips 3 cm above grade
            R = np.stack([-side, fwd, up], 1)
            p = target + up*(P.DIGGER["pin_above_tip"] + h)
        elif tool == "drill":
            # motor axis down, grip and battery toward the robot, side handle on the robot's left
            R = np.stack([-up, side, fwd], 1)
            tip = 0.075 + P.AUGER["length"] + 0.02
            p = target + up*(tip + h) - fwd*0.04
        else:
            raise ValueError(tool)
        return p + off, R

    GRIP_HALF = dict(spade=P.SPADE["shaft_r"], trench_shovel=P.TRENCH_SHOVEL["shaft_r"], trowel=P.TROWEL["grip_r"],
                     tamper=P.TAMPER["section"][0]/2, mattock=0.016, digger=0.017, drill=0.022)

    def _grip_half_width(self, tool):
        return self.GRIP_HALF[tool]

    def _place_held(self):
        """Put the held tools into the hands (IK for each grip), close the fingers in force mode."""
        w, sp = self.world, self.spec
        held = sp.get("held") or {}
        if not held:
            return
        from classroom_arrangement.planner import ArmPlanner
        planner = ArmPlanner(w, self.ik, margin=0.02)       # with 5 mm a sagging link4 touches the base hull
        spine = [n for n in w.joints if "spine_vertical" in n][0]
        for tool_slot, tool in held.items():
            arms = list(w.arms) if tool_slot == "both" else [tool_slot if tool_slot in w.arms else w.arms[0]]
            p_tool, R_tool = self.tool_start_pose(tool)
            done = None
            # the tool (and an articulated tool's other bodies) may touch the hands; nothing else may
            tbodies = {w._mj_body[w.scene.bodies[n]] for n in (tool, f"{tool}_a", f"{tool}_b", "auger", "drill_trigger")
                       if n in w.scene.bodies}
            q_save = w.state.joint_q.numpy().copy()
            self._place_body(tool, p_tool, R_tool)
            self._fk()
            for spine_h in (0.30, 0.15, 0.45, 0.0, 0.60, 0.80):
                w.state.joint_q.assign(q_save)
                self._place_body(tool, p_tool, R_tool)
                self._set_joint(spine, spine_h)
                self._fk()
                sols = {}
                for arm in arms:                 # sequential: the second arm is checked against the first's pose
                    names = self.ik.arms[arm][1]
                    for g in TOOL_START[tool]["grips"][arm]:
                        Rtcp = R_tool @ g["R"]
                        ptcp = p_tool + R_tool @ g["p"]
                        for q1 in self.ik.solutions(arm, ptcp, Rotation.from_matrix(Rtcp).as_quat(), seeds=32, tol_rad=0.035):
                            if not self.ik.self_contact(arm, q1) and not planner.contacts(arm, q1, allow=tbodies):
                                sols[arm] = (dict(zip(names, map(float, q1))), ptcp, Rtcp)
                                break
                        if arm in sols:
                            for n, v in sols[arm][0].items():
                                self._set_joint(n, v)
                            self._fk()
                            break
                    if arm not in sols:
                        break
                if len(sols) == len(arms):
                    done = (spine_h, sols)
                    break
            if done is None:
                raise RuntimeError(f"no collision-free start grasp of the {tool} for {arms}")
            spine_h, sols = done
            self._set_joint(spine, spine_h)
            for arm, (sol, _, _) in sols.items():
                for n, v in sol.items():
                    self._set_joint(n, v)
            self._fk()
            self._place_body(tool, p_tool, R_tool)
            for arm in arms:
                half = self._grip_half_width(tool)
                for n in w.gripper_joints(arm):
                    # the drill's pistol grip rests in an open hand: fingers placed at the grip width pushed the
                    # trigger 11 mm in at tick 0, past its stop, and it stayed in (D9 v2-v4 ran the motor throughout)
                    self._set_joint(n, 0.04 if (tool == "drill" and arm == "right") else min(0.04, half + 0.0008))
            self._fk()
            for arm, (sol, ptcp, Rtcp) in sols.items():
                # 60 N per finger (120 N pinch): the slip load along a shaft is 2 x 60 x mu_pad 0.8 = 96 N per hand,
                # near the 100 N hand-force protective stop (contact_x5)
                if tool == "drill" and arm == "right":
                    # the pistol grip rests in a relaxed hand (position mode at the grip width): squeezing it in force
                    # mode closes a pad on the trigger (8 N preload, 20 N at full travel) and runs the motor
                    # fingers fully open around the grip, clear of the trigger (D9's first runs ran the motor from the
                    # start: at 14 mm clearance a pad still pressed the trigger, whose spring holds it 34 mm out)
                    w.set_grip_force(arm, 0.0)
                    for n in w.gripper_joints(arm):
                        w.targets[w.joints[n]["q"]] = 0.04
                else:
                    w.set_grip_force(arm, 60.0)
                    for n in w.gripper_joints(arm):
                        w.targets[w.joints[n]["q"]] = 0.0
                self.hold_record[arm] = dict(tool=tool, tcp=ptcp.tolist(), spine=spine_h)
        w.sync_solver()

    def _settle(self):
        """Soil alone with the bodies held (SOIL_SETTLE_S), then the coupled world: >= 1 s, until every free body is
        slower than 2 mm/s through a 0.1 s window (classroom rule), at most 5 s."""
        w, s = self.world, CP.SETTLE
        w.settle_soil()
        w.step()
        self.start_contacts = self._contact_audit()
        deep = [c for c in self.start_contacts if c["depth_m"] > 0.008]
        if deep:
            # a start grasp or placement that leaves bodies interpenetrating is a construction error, not a settle
            # (the trenching shovel's first grips put both hands 5-6 cm inside its handle: an 8 kN impulse at tick 0)
            raise RuntimeError(f"start state interpenetrates at tick 1: {deep[:3]}")
        seconds, fastest = self._settle_coupled(s["min_s"])
        self.settle_record = dict(seconds=round(seconds, 3), max_speed_m_s=round(fastest, 6),
                                  soil_max_speed_m_s=round(float(w.soil.max_speed), 6), start_contacts=self.start_contacts[:6])
        if self.spec.get("braces"):
            self.settle_record["braces_fastened_plumb"] = self._fasten_braces_plumb()
        if w.reflexes:
            raise RuntimeError(f"robot protective stop during the reset settle: {w.reflexes[:2]}")
        w.tick = 0
        w.base.odom[:] = 0.
        w.timing = dict(physics_s=0., control_s=0., soil_s=0.)

    def _settle_coupled(self, min_s):
        """The coupled world in 0.1 s windows for >= min_s, until every free body is slower than 2 mm/s through a window
        (classroom rule), at most SETTLE max_s. Returns (seconds, fastest speed in the last window)."""
        w, s = self.world, CP.SETTLE
        bodies = np.array(sorted(set(w._objects) | {w.scene.robot_body["base_link"]}), int)
        chunk, every = int(round(s["window_s"]/CP.DT)), int(round(0.01/CP.DT))
        n = 0
        while True:
            fastest = 0.
            for k in range(chunk):
                w.step()
                if (k + 1) % every == 0:
                    qd = w.state.body_qd.numpy()
                    fastest = max(fastest, float(np.linalg.norm(qd[bodies, :3], axis=1).max()))
            n += chunk
            if n*CP.DT >= min_s - 1e-9 and (fastest < s["speed_m_s"] or n*CP.DT >= s["max_s"] - 1e-9):
                break
        return n*CP.DT, fastest

    BRACE_RESETTLE_S = 0.25

    def _fasten_braces_plumb(self):
        """A braced post is plumbed where it seats and only then fastened (This Old House, QUIKRETE: set the post, plumb
        it on two faces with a level, then fasten the braces). Braces attached at the post's placement pose, before the
        reset settle seats it on the loose gravel base, would tip it 1.0 deg as it sinks 16.3 mm. Here the seated post is stood plumb about its base centre
        (its body origin; yaw kept, the seat where the settle left it), the braces are fastened at that pose (stiffness,
        damping, attachment height and stake positions unchanged; only their fastened length follows the plumb post)
        and the coupled world settles again. Construction order only: no object's physics changes."""
        from .task import _tilt_deg
        w = self.world
        b = w.scene.bodies["post"]
        p, R = w.body_pose(b)
        before = _tilt_deg(R)
        yaw = math.atan2(R[1, 0], R[0, 0])
        self._place_body("post", p, Rotation.from_euler("z", yaw).as_matrix(), at_rest=True)
        self._fk()
        w.set_braces(self.spec["braces"])
        seconds, fastest = self._settle_coupled(self.BRACE_RESETTLE_S)
        p1, R1 = w.body_pose(b)
        x0, y0, z0, _ = self.spec["items"]["post"]["pose"]
        return dict(tilt_seated_deg=round(before, 4), tilt_after_deg=round(_tilt_deg(R1), 4),
                    seat_below_placement_m=round(float(z0 - p1[2]), 4), resettle_s=round(seconds, 3),
                    resettle_max_speed_m_s=round(fastest, 6))

    def _contact_audit(self):
        """MuJoCo contacts deeper than 3 mm (body pair, depth, normal force) in the current state."""
        import mujoco
        w = self.world
        m, d = w.mjm, w.mjd
        out = {}
        for i in range(d.ncon):
            dist = float(d.contact.dist[i])
            if dist > -0.003:
                continue
            g1, g2 = d.contact.geom[i]
            names = tuple(sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?" for g in (g1, g2)))
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            prev = out.get(names)
            if prev is None or -dist > prev["depth_m"]:
                out[names] = dict(bodies=list(names), depth_m=round(-dist, 4), force_n=round(float(f[0]), 1))
        return sorted(out.values(), key=lambda c: -c["depth_m"])

    # ------------------------------------------------------------------ commands
    def act(self, targets=None, *, twist=None, ticks=1, grip=None, state=True):
        w = self.world
        if type(ticks) is not int or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if twist is not None and any(abs(v) > 0 for v in twist) and w.base_held():
                raise ValueError(f"Base command refused: an arm protective stop holds the base for {CP.REFLEX_RECOVERY_S} s")
            for arm, f in (grip or {}).items():
                w.set_grip_force(arm, f)
            if targets:
                w.set_targets(dict(targets))
            if twist is not None:
                w.base.set_twist(*twist)
            for _ in range(ticks):
                w.step()
            return self.get_state() if state else None

    # ------------------------------------------------------------------ state
    def joints_state(self):
        w = self.world
        q, qd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
        return {n: {"position": float(q[s["q"]]), "velocity": float(qd[s["dof"]])} for n, s in w.joints.items()}

    def robot_status(self):
        w = self.world
        b = w.base
        return {"base": {"odometry": [round(float(v), 6) for v in b.odom], "twist_measured": [round(float(v), 6) for v in b.odom_twist],
                         "twist_commanded": [round(float(v), 6) for v in b.cmd]},
                "arms": {a: {"stopped_s": round(max(0., float(w._reflex_until.get(a, 0.)) - w.sim_time), 4),
                             "external_torque_nm": [round(float(v), 3) for v in w._tau_f[a]],
                             "external_wrench_hand": [round(float(v), 3) for v in w._wrench_f[a]],
                             "grip_force_n": w.grip_force[a]} for a in w.arms},
                "refused_targets": int(w.refused_targets),
                "base_sensors": self.base_sensors()}

    def base_sensors(self):
        key = (int(self.world.tick), self.generation)
        if getattr(self, "_sensor_key", None) != key:
            w = self.world
            imu = getattr(w, "imu", None) or dict(angular_velocity_rad_s=np.zeros(3), specific_force_m_s2=np.array([0., 0., CP.GRAVITY]))
            scans = w.lidar_scan()
            self._sensor_cache = {
                "imu": {k: [round(float(x), 5) for x in v] for k, v in imu.items()},
                "lidar": {k: dict(angle_min_deg=float(s["angle_min_deg"]), angle_step_deg=float(s["angle_step_deg"]),
                                  ranges_m=[float(x) for x in s["ranges_m"]]) for k, s in scans.items()}}
            self._sensor_key = key
        return self._sensor_cache

    def tools_state(self):
        """Privileged: tool and inclusion poses, soil wrench per coupled body, drill state (owner view)."""
        w = self.world
        out = {}
        rows = {n: k for k, n in enumerate(w.coupled_names)}
        for n, it in w.scene.items.items():
            p, R = w.body_pose(it["body"])
            d = dict(kind=it["kind"], position=[round(float(x), 5) for x in p],
                     quat_xyzw=[round(float(x), 6) for x in Rotation.from_matrix(R).as_quat()])
            if n in rows:
                d["soil_force_n"] = [round(float(x), 2) for x in w.soil_wrench[rows[n], :3]]
            out[n] = d
        if w.drill_state is not None:
            out["drill"]["motor"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.drill_state.items()}
        return out

    def get_state(self):
        w = self.world
        return {"tick": int(w.tick), "time_s": round(w.sim_time, 6), "generation": self.generation, "fault": w.fault,
                "finite": w.fault is None, "ready": True, "joints": self.joints_state(), "robot_status": self.robot_status(),
                "base_pose": [round(v, 5) for v in w.base_pose()], "tools": self.tools_state(), "reflexes": len(w.reflexes),
                "strikes": len(w.strikes), "soil": dict(steps=int(w.soil.steps), max_speed_m_s=round(float(getattr(w.soil, "max_speed", 0.)), 4)),
                "evaluation": self.evaluation()}

    EVAL_EVERY = 1000          # ticks (0.5 s): the progress read of 0.6 M particles costs ~0.3 s; verify is always fresh

    def evaluation(self, fresh=False):
        """Advisory progress (the verdict is evaluator.evaluate(), GAR verify). Recomputed at most every EVAL_EVERY ticks
        of simulated time unless fresh, and always after a reset/restore (generation change)."""
        if not hasattr(self, "evaluator"):
            return None
        tick, gen = int(self.world.tick), self.generation
        last = getattr(self, "_eval_key", None)
        if fresh or last is None or last[1] != gen or tick < last[0] or tick - last[0] >= self.EVAL_EVERY:
            self._eval_cache, self._eval_key = self.evaluator.progress(), (tick, gen)
            self._eval_cache = dict(self._eval_cache, computed_at_tick=tick)
        return self._eval_cache

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        return dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(),
                    control_qd=w.control.joint_target_qd.numpy().copy(), tick=w.tick,
                    grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), grip_ref=dict(w._grip_ref),
                    reflex_until=dict(w._reflex_until), reflex_fired=dict(w._reflex_fired), tau_f=copy.deepcopy(w._tau_f),
                    vref=w._vref.copy(), tau_sf=copy.deepcopy(w._tau_sf), wrench_f=copy.deepcopy(w._wrench_f),
                    base=w.base.state(), refused=w.refused_targets, n_reflexes=len(w.reflexes),
                    mj_qpos=w.mjd.qpos.copy(), mj_qvel=w.mjd.qvel.copy(), mj_act=w.mjd.act.copy() if w.mjm.na else None,
                    mj_warm=w.mjd.qacc_warmstart.copy(), soil=w.soil.snapshot(), soil_wrench=w.soil_wrench.copy(),
                    soil_kin=None if getattr(w, "soil_kinematic", None) is None else w.soil_kinematic.copy(),
                    qd_k=None if getattr(w, "_qd_k", None) is None else w._qd_k.copy(),
                    strikes=list(w.strikes), braces=copy.deepcopy(w.braces), drill=copy.deepcopy(w.drill_state),
                    variation=dict(task=self.spec["task"], seed=self.spec["seed"]))

    def restore(self, snap):
        w = self.world
        self._eval_key = None
        with self.lock:
            for a, v in snap["rigid"].items():
                getattr(w.state, a).assign(v)
                getattr(w.next_state, a).assign(v)
            w.targets = snap["targets"].copy()
            w.control.joint_target_q.assign(snap["control"])
            w.control.joint_target_qd.assign(snap["control_qd"])
            w.tick = snap["tick"]
            w.grip_force, w.grasp_latched, w._grip_ref = dict(snap["grip_force"]), dict(snap["grasp_latched"]), dict(snap["grip_ref"])
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w._tau_f = copy.deepcopy(snap["tau_f"])
            w._vref = snap["vref"].copy()
            w._tau_sf = copy.deepcopy(snap["tau_sf"]); w._wrench_f = copy.deepcopy(snap["wrench_f"])
            w.base.restore(snap["base"])
            w.refused_targets = snap["refused"]
            del w.reflexes[snap["n_reflexes"]:]
            w.fault = None
            w.sync_solver()
            w.mjd.qpos[:] = snap["mj_qpos"]
            w.mjd.qvel[:] = snap["mj_qvel"]
            w.mjd.qacc_warmstart[:] = snap["mj_warm"]
            if snap["mj_act"] is not None:
                w.mjd.act[:] = snap["mj_act"]
            w._mujoco.mj_forward(w.mjm, w.mjd)
            w.soil.restore(snap["soil"])
            w.soil_wrench = snap["soil_wrench"].copy()
            w._F_relax = snap["soil_wrench"].copy()          # the relaxed wrench is the applied one
            w.soil_kinematic = None if snap.get("soil_kin") is None else snap["soil_kin"].copy()
            w._qd_k = None if snap.get("qd_k") is None else snap["qd_k"].copy()
            if w.soil_kinematic is not None:
                w.soil.set_masses(np.where(w.soil_kinematic, 0.0, w.coupled_mass))
            w.strikes = list(snap["strikes"])
            w.braces = copy.deepcopy(snap["braces"])
            w.drill_state = copy.deepcopy(snap["drill"])

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()

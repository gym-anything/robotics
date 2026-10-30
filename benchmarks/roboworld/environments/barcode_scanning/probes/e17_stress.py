"""E17 -- stability under LEGAL actions: what a policy (or a hack agent) can send through the normal interface.

Every command below is a legal action: joint targets inside the joint limits (the servos slew them at the datasheet
joint speeds), grip force inside [0, the finger's effort limit] (35 N per finger on the FR3, 70 N for the hand), Cartesian targets
through the same IK the HTTP interface uses.
No phantom forces. A robot failing the task, dropping the scanner or knocking things over is a legitimate outcome;
NaN/Inf, explosions (implausible speeds), deep interpenetration, joints past their limits or backend faults are not.

  1. random: from the settled scene (restored for every case), every arm and finger joint of every arm is sent to a
     uniformly random target inside its limits with a random grip force, redrawn every 0.5 s for `--draws` draws (an
     arm in a protective stop keeps being sent targets, which it refuses for its 1 s hold); `--random` cases per preset,
     a different random stream per preset.
  1b. push: the left hand driven at full speed 15 cm sideways through the stock item nearest to it and held there,
     its target inside the item, for 2 s (an arm stopped against an item is a state any policy reaches).
Every case ends with SETTLE_S hands-off (grip forces released except in the held cases, where the hand keeps its grip)
and is judged on it: a non-robot body whose positions in the last second span more than SETTLE_EXTENT_MM without
shrinking (at least 80 % of the span in the second before) is a problem (a bottle against a
stopped arm slid back and forth at 1.75 mm/s for 30 s). The held cases also report how far the scanner moved in the
hand (relative to the hand's TCP) and turned, after 3 s and at the end of a longer HELD_SETTLE_S, with the slide rate
over its last 5 s (after 3 s a whipped scanner was still sliding, and kept sliding for 30 s).
  2. directed abuse with the scanner held (the right hand takes it by the head, the scripted E7 grasp; the state is
     checkpointed and every case starts from there):
       whip        the hand's full grip (35 N per finger), driven +-0.15 m sideways six times at full joint speed
       slam        the held scanner driven down at full speed to 5 cm below the deck/counter surface
       grip_toggle grip force toggled 0 / 35 N per finger every 20 ms for 2 s while the hand rises
       jam_cradle  the held scanner driven sideways through the cradle's cup
  3. directed abuse without the scanner: the left hand driven at full speed 4 cm into the keypad, then onto the
     nearest stock item and squeezed at the full grip (35 N per finger).
Per case, per tick: finiteness; the largest linear and angular speed of every non-robot body (and of robot links,
reported apart); the deepest penetration a geom pair holds for 5 consecutive ticks (sustained; judged) and the deepest
single-tick contact depth with the count of isolated spikes over 5 mm (MuJoCo box-box artifacts on grazing edges;
reported); the largest joint-limit excursion; the
firmware/sensor events; velocity and collision reflexes; faults. Frames the firmware acquires use a dark stub image (this probe
is physics only).

  python -m barcode_scanning.probes.e17_stress --out barcode_scanning/evidence/e17_stress [--presets stockroom,counter,library]
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np

RECORD = {"random_0": "random joint targets and grip forces, twice, then hands off",
          "held_whip": "the held scanner whipped +-0.15 m six times at full joint speed, 35 N per finger",
          "held_slam": "the held scanner driven down at full speed to 5 cm below the surface",
          "keypad_and_item": "a fingertip driven 4 cm into the keypad, then onto the nearest item and squeezed at 35 N per finger",
          "push_stock": "the left hand driven through the nearest item and held pressing it for 2 s, then hands off",
          "held_jam_cradle": "the held scanner driven sideways through the cradle's cup at full speed, then hands off"}
LIN_LIMIT, ANG_LIMIT, PEN_LIMIT_MM, LIMIT_EXCURSION_RAD = 5.0, 200.0, 5.0, 0.01   # judged thresholds
SETTLE_S, SETTLE_EXTENT_MM = 3.0, 0.1
HELD_SETTLE_S = 15.0                    # the held cases: the hand keeps its grip; slip judged after 3 s and at the end
SLIDE_LIMIT_MM_S = 0.01                 # judged: a scanner held still in a closed hand sliding faster than this at the end


class DarkRenderer:
    """Stands in for the render worker: the imager sees a dark frame (no symbol), so the firmware runs unchanged."""
    last_timings = {}

    def scan(self, subframes, tick, lights, visibility=None, samples=None):
        from barcode_scanning import params as P
        return np.zeros((P.HEIGHT_PX, P.WIDTH_PX, 3), np.float32)


class Tracker:
    def __init__(self, env):
        self.env, self.w = env, env.world
        w = self.w
        self.robot = set(int(b) for b in w.scene.robot_bodies)
        self.labels = list(w.model.body_label)
        self.lower = np.array([s["lower"] for s in w.joints.values()]); self.upper = np.array([s["upper"] for s in w.joints.values()])
        self.qi = np.array([s["q"] for s in w.joints.values()])
        self.recorder = None
        self.reset()

    def reset(self):
        self.lin, self.ang, self.rlin, self.rang = {}, {}, 0., 0.
        self.pen_mm, self.pen_pair, self.excursion, self.finite = 0., None, 0., True
        self.window, self.sustained_mm, self.sustained_pair, self.spikes = {}, 0., None, 0
        self.settle_tick, self.settle_speed, self.settle_pos = None, [], []   # per tick of the settle phase: every body's speed, position

    def __call__(self, env):
        w = self.w
        if self.recorder is not None and w.tick % max(1, int(round(.02/w.config.rigid_dt))) == 0:   # 20 ms
            self.recorder.snapshot()
        qd = w.state.body_qd.numpy()
        if not np.isfinite(qd).all():
            self.finite = False
            return
        v = np.linalg.norm(qd[:, :3], axis=1); om = np.linalg.norm(qd[:, 3:6], axis=1)
        if self.settle_tick is not None and w.tick >= self.settle_tick:
            self.settle_speed.append(v.copy())
            self.settle_pos.append(w.state.body_q.numpy()[:, :3].copy())
        for b in range(len(v)):
            if b in self.robot:
                self.rlin, self.rang = max(self.rlin, float(v[b])), max(self.rang, float(om[b]))
            else:
                n = self.labels[b]
                if v[b] > self.lin.get(n, 0.):
                    self.lin[n] = float(v[b])
                if om[b] > self.ang.get(n, 0.):
                    self.ang[n] = float(om[b])
        d = w.solver.mj_data
        import mujoco
        m = w.solver.mj_model
        # per geom pair, the deepest contact this tick; a penetration counts as SUSTAINED when the pair stays at least
        # that deep for 5 consecutive ticks. MuJoCo's box-box routine now and then returns one bogus deep point on
        # grazing edges for a single tick (a finger 23 mm off a post's corner reported -33 mm, E17): those are counted
        # apart as spikes
        deep = {}
        for k in range(d.ncon):
            c = d.contact[k]
            key = (int(c.geom[0]), int(c.geom[1]))
            deep[key] = min(deep.get(key, 0.), float(c.dist))
        window = {}
        for key, dist in deep.items():
            hist = (self.window.get(key, []) + [-dist*1e3])[-5:]
            window[key] = hist
            if -dist*1e3 > self.pen_mm:
                self.pen_mm = -dist*1e3
                self.pen_pair = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in key]
            if len(hist) == 5 and min(hist) > self.sustained_mm:
                self.sustained_mm = min(hist)
                self.sustained_pair = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in key]
            prev = self.window.get(key, [0.])[-1]
            if -dist*1e3 > 5. and prev < 2.:
                self.spikes += 1
        self.window = window
        q = w.state.joint_q.numpy()[self.qi]
        self.excursion = max(self.excursion, float(np.max(np.maximum(self.lower - q, q - self.upper))))

    def settle(self):
        """The settle phase, per non-robot body: its largest speed in the last 0.5 s, the path it travelled (integral of
        its speed), its net displacement and the extent of its positions in the last second (and the extent in the second
        before). Speed and path read the solver's micrometre contact jitter (a 10 g lip balm wedged in the basket after
        the keypad case vibrated within 21 um at ~50 Hz: 1.9 mm of path per second, 0.007 mm net; resting bodies
        peak at 0.06-0.13 mm/s), so a body is judged still moving when its positions in the last second span more than
        SETTLE_EXTENT_MM and did not shrink below 80 % of the second before's (the sliding bottle spanned 0.6 mm)."""
        if not self.settle_speed:
            return None
        S = np.array(self.settle_speed); X = np.array(self.settle_pos)
        dt = self.w.config.rigid_dt
        n1, n05 = int(round(1./dt)), int(round(.5/dt))
        free = [b for b in range(S.shape[1]) if b not in self.robot]
        peak = S[-n05:, free].max(0)
        path_last = S[-n1:, free].sum(0)*dt*1e3
        path_before = S[-2*n1:-n1, free].sum(0)*dt*1e3
        net_last = np.linalg.norm(X[-1, free] - X[-n1, free], axis=1)*1e3
        ext_last = np.linalg.norm(X[-n1:, free].max(0) - X[-n1:, free].min(0), axis=1)*1e3
        ext_before = np.linalg.norm(X[-2*n1:-n1, free].max(0) - X[-2*n1:-n1, free].min(0), axis=1)*1e3
        order = np.argsort(-ext_last)[:3]
        moving = [dict(body=self.labels[free[i]], peak_last_05s_mm_s=round(float(peak[i])*1e3, 4),
                       path_last_1s_mm=round(float(path_last[i]), 4), path_second_before_mm=round(float(path_before[i]), 4),
                       net_last_1s_mm=round(float(net_last[i]), 4), extent_last_1s_mm=round(float(ext_last[i]), 4),
                       extent_second_before_mm=round(float(ext_before[i]), 4)) for i in order]
        stuck = [m for m in moving if m["extent_last_1s_mm"] > SETTLE_EXTENT_MM and m["extent_last_1s_mm"] >= .8*m["extent_second_before_mm"]]
        return dict(settle_s=round(len(S)*dt, 3), most_moving_last_1s=moving, not_settling=stuck)

    def summary(self):
        top = sorted(self.lin.items(), key=lambda kv: -kv[1])[:4]
        topa = sorted(self.ang.items(), key=lambda kv: -kv[1])[:4]
        return dict(finite=self.finite, max_body_speed_m_s=dict(top), max_body_angular_rad_s=dict(topa), settle=self.settle(),
                    max_robot_link_speed_m_s=self.rlin, max_robot_link_angular_rad_s=self.rang,
                    deepest_penetration_mm=self.pen_mm, deepest_pair=self.pen_pair, sustained_penetration_mm=self.sustained_mm,
                    sustained_pair=self.sustained_pair, contact_spikes_over_5mm=self.spikes, joint_limit_excursion_rad=self.excursion)


def verdict(s, fault):
    bad = []
    if not s["finite"]:
        bad.append("non-finite state")
    if fault:
        bad.append(f"fault: {fault}")
    if s["max_body_speed_m_s"] and max(s["max_body_speed_m_s"].values()) > LIN_LIMIT:
        bad.append("body speed > %.0f m/s" % LIN_LIMIT)
    if s["max_body_angular_rad_s"] and max(s["max_body_angular_rad_s"].values()) > ANG_LIMIT:
        bad.append("body angular speed > %.0f rad/s" % ANG_LIMIT)
    if s["sustained_penetration_mm"] > PEN_LIMIT_MM:
        bad.append("sustained penetration > %.0f mm" % PEN_LIMIT_MM)
    if s["joint_limit_excursion_rad"] > LIMIT_EXCURSION_RAD:
        bad.append("joint past its limit by > %.0f mrad" % (LIMIT_EXCURSION_RAD*1e3))
    if s.get("settle") and s["settle"]["not_settling"]:
        bad.append("still moving after the settle phase without slowing: " +
                   ", ".join(f"{m['body']} spanning {m['extent_last_1s_mm']:.2f} mm in the last second (net {m['net_last_1s_mm']:.2f} mm)"
                             for m in s["settle"]["not_settling"]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--presets", default="stockroom,counter,library")
    ap.add_argument("--random", type=int, default=12)
    ap.add_argument("--draws", type=int, default=8, help="random draws per random case, 0.5 s each")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--timeconst", type=float, default=None, help="probe-only override of Config.contact_timeconst")
    ap.add_argument("--solimp", default=None, help="probe-only override of Config.contact_solimp: d0,dmax,width")
    ap.add_argument("--only-random", action="store_true")
    ap.add_argument("--cases", default="", help="comma-separated case names to run (diagnosis; default: every case)")
    a = ap.parse_args()
    from barcode_scanning.config import Config, TASKS
    from barcode_scanning.env import ScanEnv
    from barcode_scanning.scripted import Operator
    from barcode_scanning.probes.e7_hold import grasp_frame
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from barcode_scanning.manifest import launch_stamp
    report = dict(code_at_launch=launch_stamp(), experiment="E17 legal-action stress", thresholds=dict(body_speed_m_s=LIN_LIMIT, body_angular_rad_s=ANG_LIMIT,
                  penetration_mm=PEN_LIMIT_MM, joint_limit_excursion_rad=LIMIT_EXCURSION_RAD), presets={})
    import zlib
    for preset in a.presets.split(","):
        rng = np.random.default_rng([a.seed, zlib.crc32(preset.encode())])          # a different stream per preset
        t_build = time.time()
        over = {}
        if a.timeconst is not None:
            over["contact_timeconst"] = a.timeconst
        if a.solimp:
            over["contact_solimp"] = tuple(float(x) for x in a.solimp.split(","))
        env = ScanEnv(Config(preset=preset, task=TASKS[preset][0], **over), output_dir=out/f"{preset}_session")
        env.render_meta = {"scanner": {"leds": [], "aim": "aim", "led_on": "led_on"}}
        env.renderer_factory = lambda: DarkRenderer()
        w = env.world
        tr = Tracker(env)
        env.trace_callback = tr
        base = env.checkpoint()
        cases = []

        def in_hand(arm_):
            """The scanner's pose in the hand's TCP frame (position m, rotation)."""
            from scipy.spatial.transform import Rotation as Rot
            T = Operator(env, record_every=10**9).k.forward(arm_)
            ps, Rs = w.body_pose("scanner")
            Rt = Rot.from_matrix(T[:3, :3])
            return Rt.inv().apply(ps - T[:3, 3]), Rt.inv()*Rot.from_matrix(Rs)

        def run_case(name, fn, start, held_by=None):
            if a.cases and name not in a.cases.split(","):
                return
            env.restore(start)
            w.fault = None
            tr.reset()
            n_ref, n_frames, t0, s0 = len(w.reflexes), len(env.frames), time.time(), w.sim_time
            n_refused = w.refused_targets
            err = None
            h0 = in_hand(held_by) if held_by else None
            if name in RECORD:                                # 20 ms snapshots for a video of this case
                tr.recorder = Operator(env, record_every=10**9)
                tr.recorder.phase = f"LEGAL ACTIONS (stress, no phantom forces): {RECORD[name]}"
            try:
                fn()
            except Exception as exc:                          # a refused command is recorded, not hidden
                err = f"{type(exc).__name__}: {exc}"
            # settle: hands off (the held cases keep their grip), judged on the bodies still moving
            if not held_by:
                for arm_ in w.rig.instances:
                    w.set_grip_force(arm_, 0.)
            if tr.recorder is not None:
                tr.recorder.phase = (f"JUDGED SETTLE {HELD_SETTLE_S:g} s: the hand keeps its grip and its last target" if held_by
                                     else f"JUDGED SETTLE {SETTLE_S:g} s: hands off, grips released")
            tr.settle_tick = w.tick
            ticks = lambda s: int(round(s/w.config.rigid_dt))
            slip = None
            if held_by:
                env.act({}, ticks=ticks(SETTLE_S))
                p3, r3 = in_hand(held_by)
                env.act({}, ticks=ticks(HELD_SETTLE_S - SETTLE_S - 5.))
                p10, _ = in_hand(held_by)
                env.act({}, ticks=ticks(5.))
                p1, r1 = in_hand(held_by)
                moved = lambda p: round(float(np.linalg.norm(p - h0[0]))*1e3, 2)
                slip = dict(scanner_moved_in_hand_mm=moved(p1),
                            scanner_turned_in_hand_deg=round(float(np.degrees((h0[1].inv()*r1).magnitude())), 2),
                            moved_in_hand_after_3s_mm=moved(p3),
                            turned_in_hand_after_3s_deg=round(float(np.degrees((h0[1].inv()*r3).magnitude())), 2),
                            slide_last_5s_mm_s=round(float(np.linalg.norm(p1 - p10))*1e3/5., 4),
                            still_in_hand=bool(np.linalg.norm(p1) < .15))
            else:
                env.act({}, ticks=ticks(SETTLE_S))
            if tr.recorder is not None:
                tr.recorder.save(out/"raw", f"e17_{preset}_{name}", extra=dict(case=name))
                tr.recorder = None
            s = tr.summary()
            rec = dict(case=name, **s, error=err, fault=w.fault, reflexes=len(w.reflexes) - n_ref, frames=len(env.frames) - n_frames,
                       refused_targets=w.refused_targets - n_refused, in_hand=slip,
                       scanner_seated=bool(w.seated), trigger_pressed=bool(w.trigger.state), sim_s=w.sim_time - s0, wall_s=time.time() - t0)
            rec["problems"] = verdict(s, w.fault) + ([f"error: {err}"] if err and "Invalid" not in err else [])
            if slip and slip["still_in_hand"] and slip["slide_last_5s_mm_s"] > SLIDE_LIMIT_MM_S:
                rec["problems"].append(f"held scanner still sliding in the hand ({slip['slide_last_5s_mm_s']:.3f} mm/s over the "
                                       f"last 5 s of a {HELD_SETTLE_S:g} s hold)")
            cases.append(rec)
            print(preset, json.dumps({k: rec[k] for k in ("case", "finite", "sustained_penetration_mm", "sustained_pair", "deepest_penetration_mm", "contact_spikes_over_5mm", "max_body_speed_m_s", "joint_limit_excursion_rad",
                                                         "problems", "reflexes", "error", "in_hand")}), flush=True)

        joints = w.joints
        for k in range(a.random):
            def rnd():
                for _ in range(a.draws):
                    tg = {n: float(rng.uniform(s["lower"], s["upper"])) for n, s in joints.items()}
                    for arm in w.rig.instances:
                        lim = float(joints[w.gripper_joints(arm)[0]]["effort_limit"])
                        w.set_grip_force(arm, float(rng.uniform(0, lim)))
                    env.act(tg, ticks=int(round(.5/w.config.rigid_dt)))
            run_case(f"random_{k}", rnd, base)

        def push():
            # the hand driven sideways at full speed through the nearest stock item and held with its target inside it
            o = Operator(env, record_every=10**9)
            pusher = next(x for x in w.rig.instances if x.startswith("left")) if len(w.rig.instances) > 1 else list(w.rig.instances)[0]
            p, R = o.tcp(pusher)
            item = min((it for it in w.scene.items.values() if it.get("sku")),
                       key=lambda it: np.linalg.norm(w.body_pose(it["body"])[0] - p))
            pi, _ = w.body_pose(item["body"])
            side = np.array([pi[0] - p[0], pi[1] - p[1], 0.]); side /= max(np.linalg.norm(side), 1e-9)
            o.move_joints(pusher, pi - .12*side + (0, 0, .03), R.as_quat(), seconds=1.2)
            o.move(pusher, pi + .03*side + (0, 0, .03), R.as_quat(), seconds=.25)
            o.wait(2.0)
        run_case("push_stock", push, base)

        if a.only_random:
            report["presets"][preset] = dict(cases=cases, problems=sum(bool(c["problems"]) for c in cases), n=len(cases), overrides=over)
            (out/"e17_stress.json").write_text(json.dumps(report, indent=1, default=float))
            continue
        # --- the right hand holds the scanner (scripted head grasp), checkpoint, abuse
        arm = next(x for x in w.rig.instances if x.startswith("right")) if len(w.rig.instances) > 1 else list(w.rig.instances)[0]
        env.restore(base)
        op = Operator(env, record_every=10**9)
        p_g, q_g, approach, axis = grasp_frame(env, "head", .030)
        op.open_hand(arm)
        op.move_joints(arm, p_g - .10*approach, q_g, seconds=2.0)
        op.move(arm, p_g, q_g, seconds=1.2)
        op.grasp(arm, 17.5, seconds=1.0)
        op.move(arm, p_g + .10*axis, q_g, seconds=1.2)
        held = env.checkpoint()
        lim = float(joints[w.gripper_joints(arm)[0]]["effort_limit"])

        def whip():
            o = Operator(env, record_every=10**9)
            w.set_grip_force(arm, lim)
            p, R = o.tcp(arm)
            side = np.cross(approach, (0, 0, 1.)); side /= np.linalg.norm(side)
            for i in range(6):
                o.move(arm, p + (.15 if i % 2 == 0 else -.15)*side, R.as_quat(), seconds=.25)
            o.wait(.5)

        def slam():
            o = Operator(env, record_every=10**9)
            p, R = o.tcp(arm)
            ps, _ = w.body_pose("scanner")
            z_surface = float(w.scene.cradle["seat"][2]) - .03
            o.move(arm, np.array([p[0], p[1], z_surface - .05 + (p[2] - ps[2])]), R.as_quat(), seconds=.3)
            o.wait(.8)

        def grip_toggle():
            o = Operator(env, record_every=10**9)
            p, R = o.tcp(arm)
            # (a hand moved 20 mm in one 10 ms waypoint every 10 toggles makes the operator refuse the 0.28 rad
            # jump)
            for i in range(100):
                w.set_grip_force(arm, lim if i % 2 else 0.)
                env.act({}, ticks=int(round(.02/w.config.rigid_dt)))
                if i % 10 == 9:
                    o.move(arm, p + (0, 0, .004*(i//10 + 1)), R.as_quat(), seconds=.1)
            o.wait(.5)

        def jam_cradle():
            o = Operator(env, record_every=10**9)
            p, R = o.tcp(arm)
            seat = np.asarray(w.scene.cradle["seat"], float)
            # joint-space moves with IK restarts (straight lines at the edge of the workspace were refused 3.2 mm out
            # of tolerance and the case never ran); the jam itself is the sideways drive through the cup
            o.move_joints(arm, seat + (0, 0, .05) + (p - w.body_pose("scanner")[0]) + .06*np.cross(axis, (0, 0, 1.)), R.as_quat(), seconds=.6)
            o.move_joints(arm, seat + (0, 0, .02) + (p - w.body_pose("scanner")[0]) - .06*np.cross(axis, (0, 0, 1.)), R.as_quat(), seconds=.4)
            o.wait(.8)

        for name, fn in (("whip", whip), ("slam", slam), ("grip_toggle", grip_toggle), ("jam_cradle", jam_cradle)):
            run_case(f"held_{name}", fn, held, held_by=arm)

        other = next((x for x in w.rig.instances if x != arm), arm)

        def keypad_and_item():
            o = Operator(env, record_every=10**9)
            p, R = o.tcp(other)
            kp = w.scene.fixtures.get("keypad", {}).get("center")
            if kp is not None:
                o.move_joints(other, np.asarray(kp, float) + (0, 0, .06), R.as_quat(), seconds=1.0)
                o.move(other, np.asarray(kp, float) - (0, 0, .04), R.as_quat(), seconds=.2)
                o.wait(.4)
            item = min((it for it in w.scene.items.values() if it.get("sku")),
                       key=lambda it: np.linalg.norm(w.body_pose(it["body"])[0] - p))
            pi, _ = w.body_pose(item["body"])
            o.open_hand(other)
            o.move_joints(other, pi + (0, 0, .12), R.as_quat(), seconds=1.2)
            o.move(other, pi + (0, 0, .01), R.as_quat(), seconds=.3)
            w.set_grip_force(other, lim)
            o.set_targets({j: 0. for j in w.gripper_joints(other)})
            o.wait(.8)
            o.move(other, pi + (0, 0, .15), R.as_quat(), seconds=.3)
            o.wait(.8)

        run_case("keypad_and_item", keypad_and_item, base)
        report["presets"][preset] = dict(build_s=time.time() - t_build, cases=cases,
                                         problems=sum(bool(c["problems"]) for c in cases), n=len(cases))
        (out/"e17_stress.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({p: (v["problems"], v["n"]) for p, v in report["presets"].items()}))


if __name__ == "__main__":
    main()

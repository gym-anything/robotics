"""Persistent barcode-scanning environment: robot-only actions, exact ticks, explicit faults.

One action = N physics ticks (1 ms). During the ticks the world records, per tick, the inputs of the
scanner firmware (trigger switch, cradle seat sensor, scanner motion, presentation-zone change) and the body
poses at every tick where a frame could be due. After the ticks the firmware is replayed over that record;
at each frame instant the imager frame is rendered from the recorded poses (render.py), formed by the sensor
model (optics.py) and decoded (decoder.py); a decode drives the firmware (a read ends the handheld session,
so later frames of that trigger pull are never taken) and the host app. Decoding never changes physics, so
this ordering reproduces the in-loop behaviour exactly.
"""

import copy
import math
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import params as P
from .config import Config
from .world import World
from .device import Firmware, HostApp
from .optics import Imager
from . import decoder as D
from .catalog import SKUS


class ScanEnv:
    def __init__(self, config=None, *, output_dir=None):
        self.config = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir or Path(__file__).resolve().parent/"output"/"session")
        self.output.mkdir(parents=True, exist_ok=True)
        self.world = World(self.config)
        self.generation = 0
        self._settle()
        self.formats = D.FORMATS["library" if self.config.preset == "library" else
                                 "retail" if self.config.preset == "counter" else "inventory"]
        self._new_devices()
        self.initial = self.checkpoint()
        self.renderer_factory = None        # set by the backend: returns a Renderer for this operation
        self.render_meta = None             # set with the renderer (render.export metadata)
        self.frame_hook = None              # probes: called with every processed frame (diagnostics only)
        self.trace_callback = None          # privileged trace (trace.py), called every tick
        from .task import Evaluator
        self.evaluator = Evaluator(self)

    def _settle(self):
        """Gravity settle before tick 0: at least `settle_s`, then until every free body (stock, containers, scanner)
        has stayed slower than `settle_speed` through the whole last 0.1 s (sampled every 10 ms), at most
        `settle_max_s`. The time used is recorded. (An instantaneous test stopped while a lying can was momentarily
        slow; it then rolled at 38 mm/s into the task, counter preset at 0.5 ms.)"""
        c = self.config
        free = sorted({it["body"] for it in self.world.scene.items.values()} | {self.world.scene.bodies["scanner"]})
        # the G1 stands on its servos and sways for about a second after the drop onto its feet; its shoulder links stand
        # inside the docked scanner's presentation zone at the counter and the library desk and woke it at 4-6 mm/s for
        # the first 1-2 s of every session: the settle also waits for the robot's links to be slower than the
        # presentation wake speed
        robot = sorted(self.world.scene.robot_bodies) if self.world.g1 else []
        chunk = int(round(.1/c.rigid_dt))
        every = max(1, int(round(.01/c.rigid_dt)))
        n = 0
        while True:
            fastest = fastest_robot = 0.
            for k in range(chunk):
                self.world.step()
                if (k + 1) % every == 0:
                    qd = self.world.state.body_qd.numpy()
                    fastest = max(fastest, float(np.linalg.norm(qd[free, :3], axis=1).max()))
                    if robot:
                        fastest_robot = max(fastest_robot, float(np.linalg.norm(qd[robot, :3], axis=1).max()))
            n += chunk
            still = fastest < c.settle_speed and fastest_robot < P.PRESENTATION_WAKE_SPEED_M_S
            if n*c.rigid_dt >= c.settle_s - 1e-9 and (still or n*c.rigid_dt >= c.settle_max_s - 1e-9):
                break
        qd = self.world.state.body_qd.numpy()
        self.settle_record = dict(seconds=round(n*c.rigid_dt, 3),
                                  max_speed_m_s=float(np.linalg.norm(qd[free, :3], axis=1).max()))
        if robot:
            self.settle_record["robot_max_speed_m_s"] = float(np.linalg.norm(qd[robot, :3], axis=1).max())
        if self.world.g1:
            # a humanoid that fell during the settle is a broken reset, not a task state (a start pose with the hands
            # inside the hip links threw the arms apart and felled the G1 before tick 0)
            from . import g1 as G1
            z, tilt = G1.pelvis_state(self.world)
            self.settle_record.update(g1_pelvis_z_m=round(z, 4), g1_tilt_deg=round(tilt, 3))
            if tilt > 10. or z < G1.PELVIS_Z - .1:
                raise RuntimeError(f"G1 fell during the reset settle (pelvis {z:.3f} m, tilt {tilt:.1f} deg)")
        self.world.tick = 0
        self.world.trigger.events.clear()
        for s in self.world.key_switch.values():
            s.events.clear()

    def _new_devices(self):
        sc = self.world.scene
        names = {}
        from .catalog import sku_gtin14
        for s in SKUS.values():
            names[sku_gtin14(s)] = s.name
        for it in sc.items.values():
            if it.get("library_id"):
                names[it["library_id"]] = SKUS[it["sku"]].name
        self.firmware = Firmware(seated=self.world.seated)
        self.host = HostApp(self.config.task, names, sc.slots.keys())
        self.imager = Imager("handheld", rng=np.random.default_rng(1000 + self.config.seed))
        self.frames = []            # processed frame records (metadata)
        self._new_history()
        self.pending = []           # recorded ticks awaiting firmware replay
        self._key_seen = {k: len(s.events) for k, s in self.world.key_switch.items()}

    def _new_history(self):
        """What the evaluator needs from the whole episode, not only its end: the longest steady hold of the scanner
        out of its cradle (speed < 20 mm/s and < 0.25 rad/s, held: a finger or hand pressing it with more than 0.2 N on
        every tick of the run), and each free body's highest rise above its start."""
        w = self.world
        q = w.state.body_q.numpy()
        self._free = np.array(sorted({it["body"] for it in w.scene.items.values()}), int)
        self.history = dict(steady_run_s=0., max_steady_out_s=0., max_rise_m={int(b): 0. for b in self._free},
                            start_z={int(b): float(q[b, 2]) for b in self._free}, left_cradle=False)

    def _track(self, q, qd):
        h, w = self.history, self.world
        sb = w.scene.bodies["scanner"]
        steady = float(np.linalg.norm(qd[sb, :3])) < .020 and float(np.linalg.norm(qd[sb, 3:6])) < .25
        if not w.seated:
            h["left_cradle"] = True
        # a scanner lying still on a tray is not held: the run counts only while a finger or hand presses it (checked
        # only when it is out and still, to keep ticks cheap)
        held = (not w.seated and steady) and self.evaluator._scanner_held()
        h["steady_run_s"] = h["steady_run_s"] + w.config.rigid_dt if held else 0.
        h["max_steady_out_s"] = max(h["max_steady_out_s"], h["steady_run_s"])
        if w.tick % max(1, int(round(.01/w.config.rigid_dt))) == 0:          # every 10 ms
            rise = q[self._free, 2] - np.array([h["start_z"][int(b)] for b in self._free])
            for b, r in zip(self._free, rise):
                if r > h["max_rise_m"][int(b)]:
                    h["max_rise_m"][int(b)] = float(r)

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        return dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(), tick=w.tick,
                    grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), grip_ref=dict(w._grip_ref),
                    reflex_until=dict(w._reflex_until), reflex_fired=dict(w._reflex_fired),
                    # the protective-stop record and the refused-target count belong to the branch being saved: a
                    # restore must not keep the abandoned branch's stops
                    reflexes=copy.deepcopy(w.reflexes), refused_targets=int(w.refused_targets),
                    ramp=copy.deepcopy(getattr(w, "_ramp", None)),
                    trigger=copy.deepcopy(w.trigger.__dict__), keys={k: copy.deepcopy(s.__dict__) for k, s in w.key_switch.items()},
                    seated=(w.seated, w._seat_since, w._cup_ok),
                    firmware=copy.deepcopy(self.firmware.__dict__) if hasattr(self, "firmware") else None,
                    host=copy.deepcopy(self.host.__dict__) if hasattr(self, "host") else None,
                    imager=(self.imager.t_e, self.imager.gain, copy.deepcopy(self.imager.rng.bit_generator.state))
                    if hasattr(self, "imager") else None,
                    frames=len(self.frames) if hasattr(self, "frames") else 0,
                    history=copy.deepcopy(self.history) if hasattr(self, "history") else None)

    def restore(self, snap):
        w = self.world
        for a, v in snap["rigid"].items():
            getattr(w.state, a).assign(v)
            getattr(w.next_state, a).assign(v)
        w.targets = snap["targets"].copy()
        w.control.joint_target_q.assign(snap["control"])
        w.tick = snap["tick"]
        w.grip_force, w.grasp_latched = dict(snap["grip_force"]), dict(snap["grasp_latched"])
        w._grip_ref = dict(snap.get("grip_ref", {}))
        w._reflex_until = dict(snap.get("reflex_until", {}))
        w._reflex_fired = dict(snap.get("reflex_fired", {}))
        if "reflexes" in snap:
            w.reflexes = copy.deepcopy(snap["reflexes"])
            w.refused_targets = int(snap["refused_targets"])
        w._ramp = copy.deepcopy(snap.get("ramp"))
        w.trigger.__dict__.update(copy.deepcopy(snap["trigger"]))
        for k, d in snap["keys"].items():
            w.key_switch[k].__dict__.update(copy.deepcopy(d))
        w.seated, w._seat_since, w._cup_ok = snap["seated"]
        w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
        w.solver._mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
        w.fault = None
        self._new_devices()
        if snap.get("firmware"):
            self.firmware.__dict__.update(copy.deepcopy(snap["firmware"]))
            self.host.__dict__.update(copy.deepcopy(snap["host"]))
            self.imager.t_e, self.imager.gain = snap["imager"][:2]
            if len(snap["imager"]) > 2:
                self.imager.rng.bit_generator.state = copy.deepcopy(snap["imager"][2])     # same sensor noise after restore
        self._key_seen = {k: len(s.events) for k, s in w.key_switch.items()}
        if snap.get("history"):
            self.history = copy.deepcopy(snap["history"])

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            from .task import Evaluator
            self.evaluator = Evaluator(self)
            return self.get_state()

    # ------------------------------------------------------------------ stepping
    @property
    def joints(self):
        return self.world.joints

    def _zone_change(self, q, qd):
        """Presentation-zone change (authored stand-in for image-change wake-up): a body other than
        the scanner moving faster than 5 mm/s within 0.4 m ahead of the window, inside the field of view."""
        w = self.world
        sb = w.scene.bodies["scanner"]
        p = q[sb, :3]
        R = Rotation.from_quat(q[sb, 3:]).as_matrix()
        rel = (q[:, :3] - p) @ R                          # scanner frame
        x = rel[:, 0]
        tan_h, tan_v = math.tan(math.radians(P.HFOV_DEG/2)), math.tan(math.radians(P.VFOV_DEG/2))
        inside = (x > .005) & (x < P.PRESENTATION_ZONE_M) & (np.abs(rel[:, 1]) < x*tan_h + .05) & (np.abs(rel[:, 2]) < x*tan_v + .05)
        inside[sb] = False
        inside[w.scene.bodies["scanner_trigger"]] = False
        # Newton body_qd = (linear COM velocity, angular velocity), world frame; a point 50 mm from the COM
        speed = np.linalg.norm(qd[:, :3], axis=1) + .05*np.linalg.norm(qd[:, 3:6], axis=1)
        return bool(np.any(inside & (speed > P.PRESENTATION_WAKE_SPEED_M_S)))

    def act(self, targets, *, ticks, guard=None, ramp=False):
        """Run `ticks` physics ticks with new joint targets. `ramp` (G1 Cartesian commands): the targets are reached
        along a quintic ramp over the whole action instead of at once (World.set_targets)."""
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            w = self.world
            if w.fault:
                raise RuntimeError(w.fault)
            if ramp:
                w.set_targets(targets, ramp_ticks=ticks)
            else:
                w.set_targets(targets)
            rec = []
            for _ in range(ticks):
                w.step()
                q = w.state.body_q.numpy()
                qd = w.state.body_qd.numpy()
                sb = w.scene.bodies["scanner"]
                moving = float(np.linalg.norm(qd[sb, :3])) > P.MOTION_WAKE_LINEAR_M_S or \
                    float(np.linalg.norm(qd[sb, 3:6])) > P.MOTION_WAKE_ANGULAR_RAD_S
                zone = w.seated and (w.tick % max(1, int(round(.01/w.config.rigid_dt))) == 0) and self._zone_change(q, qd)
                could = w.trigger.state or w.seated
                rec.append(dict(tick=w.tick, t=w.sim_time, trigger=w.trigger.state, seated=w.seated, moving=moving, zone=zone,
                                q=q.copy() if could else None))
                self._track(q, qd)
                for k, s in w.key_switch.items():
                    while self._key_seen[k] < len(s.events):
                        t_ev, closed = s.events[self._key_seen[k]]
                        self._key_seen[k] += 1
                        if closed:
                            self.host.on_key(k, t_ev)
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            self._replay(rec)
            return self.get_state()

    # ------------------------------------------------------------------ firmware replay and frames
    def _replay(self, rec):
        fw = self.firmware
        for i, r in enumerate(rec):
            due = fw.tick(r["t"], r["trigger"], r["seated"], r["moving"], r["zone"])
            if not due:
                continue
            docked = fw.state == "DOCKED"
            self.imager.mode = "handsfree" if docked else "handheld"
            self._frame(r, rec[i + 1:i + 3])

    def _frame(self, r, after):
        """Acquire, render, expose and decode one imager frame at tick record r."""
        fw, w = self.firmware, self.world
        if self.renderer_factory is None or self.render_meta is None:
            raise RuntimeError("No renderer attached: scanner frames need the render worker (the firmware began acquiring)")
        sb = w.scene.bodies["scanner"]
        t_e = self.imager.t_e
        # motion blur over the exposure: the fastest image-space motion of any label or body in view relative to the
        # imager (the scanner moving, or an item swiped past a docked scanner), from the recorded 1 kHz poses
        poses = [r["q"]]
        q_next = next((a["q"] for a in after if a["q"] is not None), None)
        blur_px = 0.
        if q_next is not None:
            dt_tick = self.config.rigid_dt
            blur_px = self._image_speed(r["q"], q_next, dt_tick)*t_e/self._ifov()
            K = int(min(16, max(1, math.ceil(blur_px/.5))))
            if K > 1:
                poses = [_interp(r["q"], q_next, (t_e/dt_tick)*k/(K - 1)) for k in range(K)]
        lights = {p: fw.led_intensity(r["t"]) for p in self._meta("leds")}
        lights[self._meta("aim")] = 0.                      # aim off during exposure (SE4710 default)
        renderer = self.renderer_factory()
        spp = self._scan_samples()
        rad = renderer.scan(poses, r["tick"], lights, samples=spp)
        frame, rec_sensor = self.imager.expose(self.imager.mono(rad))
        found = D.read_all(frame, self.formats)
        chosen = D.choose(found, P.WIDTH_PX, P.HEIGHT_PX)
        rep = fw.on_frame_result(r["t"], chosen, found)
        if fw.state == "DOCKED":
            fw.note_seen(r["t"], found)
        attribution = None
        if rep is not None:
            attribution = self._attribute(rep, r["q"])
            rep["attribution"] = attribution
            if attribution.get("ghost"):
                # a decode that no printed label on its ray can explain is a backend fault, never an outcome; a printed
                # label decoded as another number (attribution["misread"], for example in glare) is what a real
                # imager does now and then, so it reaches the host app like any read and the task's checks judge it
                w.fault = f"GhostRead at tick {r['tick']}: {rep['format']} {rep['text']!r}: {attribution.get('reason')}"
            self.host.on_scan(rep)
        meta = dict(tick=r["tick"], t=r["t"], mode=self.imager.mode, subframes=len(poses), blur_px=blur_px, spp=spp, **rec_sensor,
                    found=[{k: f[k] for k in ("format", "text")} for f in found], chosen=chosen and chosen["text"],
                    reported=rep is not None, attribution=attribution, render_s=getattr(renderer, "last_timings", {}).get("sub0_s"))
        self.frames.append(meta)
        if self.frame_hook:
            self.frame_hook(self, frame, rad, meta)

    def _image_speed(self, q0, q1, dt):
        """Largest angular speed (rad/s) in the imager's view of any registered label centre or body origin within
        1 m inside the field of view, relative to the imager, between two recorded poses dt apart."""
        w = self.world
        sb = w.scene.bodies["scanner"]
        pupil = np.asarray(w.scene.scanner["pupil"])
        def points(q):
            pts = []
            for L in w.scene.labels:
                b = L["body"]
                if b >= 0:
                    pts.append(q[b, :3] + Rotation.from_quat(q[b, 3:]).apply(L["center"]))
                else:
                    pts.append(np.asarray(L["center"], float))
            for it in w.scene.items.values():
                pts.append(q[it["body"], :3])
            P_ = np.asarray(pts)
            Rs = Rotation.from_quat(q[sb, 3:])
            return Rs.inv().apply(P_ - (q[sb, :3] + Rs.apply(pupil)))
        a, b = points(q0), points(q1)
        da, db = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        th, tv = math.tan(math.radians(P.HFOV_DEG/2)), math.tan(math.radians(P.VFOV_DEG/2))
        view = (a[:, 0] > .02) & (da < 1.) & (np.abs(a[:, 1]) < a[:, 0]*th) & (np.abs(a[:, 2]) < a[:, 0]*tv)
        if not view.any():
            return 0.
        cosang = np.clip(np.sum(a*b, axis=1)/np.maximum(da*db, 1e-9), -1., 1.)
        return float(np.max(np.arccos(cosang)[view]))/dt

    def _ifov(self):
        return 2*math.tan(math.radians(P.HFOV_DEG/2))/P.WIDTH_PX

    def _scan_samples(self):
        """Samples per pixel for the next scanner frame: the render's Monte Carlo noise at a third of the sensor's relative
        noise at the gain the auto-exposure set for this frame (params.RENDER_*)."""
        sensor = math.sqrt(255.*self.imager.gain/(P.AE_TARGET_DN*P.FULL_WELL_E))
        n, mc = P.RENDER_MC_REF
        lo, hi = P.RENDER_SPP_RANGE
        return int(min(hi, max(lo, math.ceil(n*mc/(P.RENDER_NOISE_FRACTION*sensor)))))

    def _meta(self, key):
        return self.render_meta["scanner"][key]

    # ------------------------------------------------------------------ attribution (privileged, verifier)
    def _attribute(self, rep, q):
        """Which printed label produced a read (privileged, verifier): back-project the decoded symbol's centre from the
        entrance pupil and intersect EVERY registered label patch (flat or cylindrical) facing the imager, at the
        frame's recorded poses; the nearest hit inside a patch is the physical source. Its payload is then compared
        with the decode. No label on the ray: a ghost read, a backend fault. A label whose payload
        differs from the decode: a misread of that label (a real imager's occasional wrong decode), reported to the host
        app like any read; the evaluator counts it (metrics["misreads"]) and the task's own checks judge its effect."""
        w = self.world
        sb = w.scene.bodies["scanner"]
        Rs = Rotation.from_quat(q[sb, 3:]).as_matrix()
        pupil = q[sb, :3] + Rs @ np.asarray(w.scene.scanner["pupil"])
        quad = np.asarray(rep["quad_px"], float)
        u, v = quad.mean(0)
        f = (P.WIDTH_PX/2)/math.tan(math.radians(P.HFOV_DEG/2))
        d_cam = np.array([1., -(u - P.WIDTH_PX/2)/f, -(v - P.HEIGHT_PX/2)/f])     # scanner frame: +x out, +y left, +z up
        d = Rs @ (d_cam/np.linalg.norm(d_cam))
        best = None
        margin = .003
        for k, L in enumerate(w.scene.labels):
            if L["body"] >= 0:
                Rb = Rotation.from_quat(q[L["body"], 3:]).as_matrix(); pb = q[L["body"], :3]
            else:
                Rb, pb = np.eye(3), np.zeros(3)
            c = pb + Rb @ np.asarray(L["center"]); n = Rb @ np.asarray(L["normal"]); up = Rb @ np.asarray(L["up"])
            if float(d @ n) >= 0:                              # printed side faces away from the imager
                continue
            if L.get("radius"):
                t = _ray_cylinder(pupil, d, c - n*L["radius"], Rb @ np.array([0, 0, 1.]), L["radius"])
            else:
                t = float((c - pupil) @ n)/float(d @ n)
            if t is None or t <= 0:
                continue
            rel = pupil + t*d - c
            right = np.cross(up, n)
            if abs(rel @ right) > L["size"][0]/2 + margin or abs(rel @ up) > L["size"][1]/2 + margin:
                continue
            if best is None or t < best[0]:
                best = (t, k)
        want = D.normalise(rep["format"], rep["text"])
        if best is None:
            return dict(label=None, ghost=True, reason="no registered label on the ray through the decoded symbol")
        L = w.scene.labels[best[1]]
        match = D.expected(L["symbol"].kind, L["symbol"].data) == want
        out = dict(label=L["name"], kind=L["kind"], item=L.get("item"), distance_m=best[0], ghost=False, misread=not match)
        if not match:
            out["reason"] = f"label payload {L['symbol'].data!r} differs from the decode {rep['text']!r}"
        return out

    # ------------------------------------------------------------------ state
    def get_state(self):
        with self.lock:
            w = self.world
            jq, jv = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
            q = w.state.body_q.numpy()
            ev = self.evaluator.evaluate() if not w.fault else dict(success=False, score=0., checks={}, metrics={})
            from .task import instruction
            return {
                "tick": w.tick, "generation": self.generation, "time_s": w.sim_time, "ready": w.fault is None,
                "fault": w.fault, "instruction": instruction(self.config, getattr(w.scene, "task_meta", None)),
                "config": dict(preset=self.config.preset, task=self.config.task, seed=self.config.seed, robot=self.config.robot),
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]), "target": float(w.targets[s["q"]])}
                           for n, s in w.joints.items()},
                "scanner": {"trigger_travel_mm": w.trigger_travel_m()*1e3, "trigger_switch": w.trigger.state,
                            "seated": w.seated, "firmware_state": self.firmware.state, "exposure_s": self.imager.t_e,
                            "gain": self.imager.gain, "frames": len(self.frames), "reads": len(self.firmware.reads)},
                "keys_pressed": [k for k, s in w.key_switch.items() if s.state],
                "host": {"screen": self.host.screen, "record": self.host.record},
                "objects": {n: {"position_m": q[i["body"], :3].tolist(), "quaternion_xyzw": q[i["body"], 3:].tolist()}
                            for n, i in w.scene.items.items()},
                "reflexes": len(w.reflexes),
                "refused_targets": w.refused_targets,
                "reset_settle": getattr(self, "settle_record", None),
                "evaluation": ev,
            }


def _interp(q0, q1, a):
    out = q0.copy()
    out[:, :3] = (1 - a)*q0[:, :3] + a*q1[:, :3]
    r0, r1 = q0[:, 3:], q1[:, 3:]
    dot = np.sum(r0*r1, axis=1, keepdims=True)
    r1 = np.where(dot < 0, -r1, r1)
    r = (1 - a)*r0 + a*r1
    out[:, 3:] = r/np.linalg.norm(r, axis=1, keepdims=True)
    return out


def _ray_cylinder(o, d, c, axis, r):
    axis = axis/np.linalg.norm(axis)
    oc = o - c
    dp = d - (d @ axis)*axis
    op = oc - (oc @ axis)*axis
    a = dp @ dp; b = 2*(dp @ op); cc = op @ op - r*r
    disc = b*b - 4*a*cc
    if a < 1e-12 or disc < 0:
        return None
    s = math.sqrt(disc)
    ts = [t for t in ((-b - s)/(2*a), (-b + s)/(2*a)) if t > 0]
    return min(ts) if ts else None

"""Percussion environment: world + instruments + acoustics + task layout + evaluator; state, actions, checkpoints.

reset(): build the therapy-room scene for (preset, task, seed), let the instruments and tools settle onto their felt,
pins and saddles (rigid, settle_s), clear the instruments' vibration and the microphones, and hand over the scene at
t = 0 with everything silent.
"""

import copy
import threading
import time
import numpy as np

from . import params as P
from .config import Config


class PercussionEnv:
    def __init__(self, config=None, *, output_dir=None):
        self.config = config or Config()
        self.output_dir = output_dir
        self.lock = threading.RLock()
        self.generation = 0
        self.trace_callback = None
        self.render_meta = None
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self):
        from .world import World
        from .task import Evaluator
        t0 = time.perf_counter()
        self.world = World(self.config)
        self.layout = self.world.scene.layout
        self.timing = dict(build_s=time.perf_counter() - t0)
        self._settle()
        self.evaluator = Evaluator(self)
        self._still_since = 0.
        self._last_targets = self.world.targets.copy()
        self.cue = dict(on=False, at=None)                  # station cue light / therapist's foot switch
        self.stimulus = None                                 # loudspeaker stimulus
        self._schedule_task_events()
        self._initial = self.checkpoint()

    def _settle(self):
        """Rigid settle with the instruments' vibration discarded (the scene is handed over silent at t = 0)."""
        w = self.world
        t0 = time.perf_counter()
        q0 = w.state.body_q.numpy().copy()
        for _ in range(int(round(self.config.settle_s/P.RIGID_DT_S))):
            w.step()
        # an authored layout settles by millimetres (felt, saddles); anything that moved further was spawned
        # interpenetrating (a layout bug, e.g. a resonator box over the stick tray's rim was shot off the table)
        q1 = w.state.body_q.numpy()
        d = np.linalg.norm(q1[w._objects, :3] - q0[w._objects, :3], axis=1)
        k = int(np.argmax(d)) if len(d) else 0
        self.settle_report = dict(max_settle_mm=round(1e3*float(d.max()), 3) if len(d) else 0.0,
                                  body=w.model.body_label[int(w._objects[k])] if len(d) else None)
        if len(d) and d.max() > 0.005:
            raise RuntimeError(f"Layout settle moved {self.settle_report['body']} by {self.settle_report['max_settle_mm']} mm "
                               f"(spawned interpenetrating) for {self.config.preset}/{self.config.task} seed {self.config.seed}")
        if w.g1:
            from . import g1 as G1
            z, tilt = G1.pelvis_state(w)
            self.settle_report.update(pelvis_z_m=round(z, 4), pelvis_tilt_deg=round(tilt, 3))
            if tilt > 10. or z < 0.6:
                raise RuntimeError(f"G1 is down after the settle (pelvis {z:.3f} m, tilt {tilt:.1f} deg)")
        w.tick = 0
        for st in list(w.structs.values()) + list(w.stick_modes.values()):
            st.z[:] = 0
        w.contact_log.clear()
        w.pairs.clear()
        w.reflexes.clear()
        w.noslip_fallbacks.clear()
        w.window_events.update(late_open=0, max_open_penetration_m=0.0)
        for k in w.timing:
            w.timing[k] = 0 if isinstance(w.timing[k], int) else 0.
        if w.acoustics is not None:
            from .acoustics import Acoustics
            w.acoustics = Acoustics(dict(w.scene.mics), seed=self.config.seed,
                                    noise=self.config.room_noise)
            w._update_radiation(force=True)
        self.timing["settle_s"] = time.perf_counter() - t0

    def _schedule_task_events(self):
        g = self.layout["goal"]
        if self.config.task == "echo" and self.world.acoustics is not None:
            from .task import echo_stimulus
            sig, onsets = echo_stimulus(g)
            sp = self.world.scene.fixtures["speaker"]["position"]
            start = int(round(g["start_s"]*P.AUDIO_FS))
            self.world.acoustics.add_point_source(np.asarray(sp, float), sig, start)
            self.stimulus = dict(onsets_s=[g["start_s"] + t for t in onsets], start_s=g["start_s"])
        if self.config.task == "strike-and-stop":
            self.cue = dict(on=False, at=None, delay=float(g["cue_delay_s"]), surface=f"rbar_{g['bar']}")

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        import mujoco
        w = self.world
        snap_mj = mujoco.MjData(w.mjm)
        mujoco.mj_copyData(snap_mj, w.mjm, w.mjd)
        snap = dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    mj=snap_mj, targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(), tick=w.tick,
                    ref=copy.deepcopy(w.ref), grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched),
                    grip_ref=dict(w._grip_ref), reflex_until=dict(w._reflex_until), reflex_fired=dict(w._reflex_fired),
                    reflexes=copy.deepcopy(w.reflexes), refused=w.refused_targets, tau_ext=copy.deepcopy(w._tau_ext),
                    structs={k: (v.state(), getattr(v, "removed", ())) for k, v in w.structs.items()},
                    sticks={k: v.state() for k, v in w.stick_modes.items()},
                    pairs=copy.deepcopy({k: {s: getattr(p, s) for s in ("F_prev", "seg", "since", "t", "last_p", "last_n", "anchor",
                                                                                 "d_off", "zo")}
                                         for k, p in w.pairs.items()}),
                    contact_log=copy.deepcopy(w.contact_log), ring_ref=copy.deepcopy(w.ring_ref), still_since=self._still_since,
                    acoustics=w.acoustics.state() if w.acoustics is not None else None,
                    cue=dict(self.cue), window_events=dict(w.window_events), noslip_fallbacks=list(w.noslip_fallbacks),
                    rigid_impacts=copy.deepcopy(w.rigid_impacts),
                    evaluator=self.evaluator.snapshot() if hasattr(self, "evaluator") else None)
        return snap

    def restore(self, snap):
        import mujoco
        from . import modal as MD
        from .world import Pair
        with self.lock:
            w = self.world
            for a, v in snap["rigid"].items():
                getattr(w.state, a).assign(v)
                getattr(w.next_state, a).assign(v)
            mujoco.mj_copyData(w.mjd, w.mjm, snap["mj"])
            # wake every tree: MuJoCo's sleeping-island cycles (tree_asleep >= 0 links the trees of one sleeping island)
            # must match the restored contacts; awake trees fall asleep again after mjMINAWAKE quiet steps
            w.wake_all()
            w.targets = snap["targets"].copy()
            w.control.joint_target_q.assign(snap["control"])
            w.tick = snap["tick"]
            w.ref = copy.deepcopy(snap["ref"])
            w.grip_force, w.grasp_latched, w._grip_ref = dict(snap["grip_force"]), dict(snap["grasp_latched"]), dict(snap["grip_ref"])
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w.reflexes, w.refused_targets = copy.deepcopy(snap["reflexes"]), snap["refused"]
            w._tau_ext = copy.deepcopy(snap["tau_ext"])
            for k, (z, removed) in snap["structs"].items():
                st = w.structs[k]
                if k == "xylophone" and tuple(getattr(st, "removed", ())) != tuple(removed):
                    st = MD.build_xylophone(w.scene.layout["xylophone"]["notes"], with_resonators=w.config.resonators,
                                            removed=tuple(removed))
                    st.removed = tuple(removed)
                    w.structs[k] = st
                st.set_state(z)
            for k, z in snap["sticks"].items():
                w.stick_modes[k].set_state(z)
            w.pairs = {}
            for k, d in snap["pairs"].items():
                p = Pair(*k)
                for s, v in d.items():
                    setattr(p, s, copy.deepcopy(v))
                w.pairs[k] = p
            w.contact_log = copy.deepcopy(snap["contact_log"])
            w.ring_ref = copy.deepcopy(snap.get("ring_ref", {}))
            w.window_events = dict(snap["window_events"])
            w.noslip_fallbacks = list(snap.get("noslip_fallbacks", []))
            w.rigid_impacts = copy.deepcopy(snap.get("rigid_impacts", []))
            w._click_pairs, w._cvel_prev = set(), None
            if snap["acoustics"] is not None:
                w.acoustics.set_state(snap["acoustics"])
            w.fault = None
            w._update_radiation(force=True)
            self._still_since = snap["still_since"]
            self.cue = dict(snap["cue"])
            if snap.get("evaluator") is not None:
                self.evaluator.load(snap["evaluator"])

    def reset(self):
        with self.lock:
            self.restore(self._initial)
            self.generation += 1
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.joints

    def act(self, targets, *, ticks, guard=None, state=True):
        """Run `ticks` world ticks (1 ms each; audio-rate windows inside) with new joint targets. state=False skips
        building the returned state (scripted operators streaming 1 ms targets)."""
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            w = self.world
            if w.fault:
                raise RuntimeError(w.fault)
            w.set_targets(targets)
            for _ in range(ticks):
                w.step()
                self._track_still()
                self.evaluator.rim_check()
                if self.config.task == "strike-and-stop":
                    self.evaluator.sample_vibration()
                elif self.config.task == "melody":
                    self.evaluator.sample_ring()
                if w.tick % 10 == 0:
                    self.evaluator.monitor()
                if self.cue.get("delay") is not None and self.cue.get("at") is None:
                    # the first stroke on the bar: a finished contact that put > 10 uJ into it (a resting or hovering
                    # fingertip adds ~0.1 uJ, a mallet stroke mJ); the cue's delay (0.20-0.35 s) exceeds any stroke's contact
                    first = next((s_ for s_ in w.contact_log if s_["surface"] == self.cue["surface"] and
                                  s_.get("E_after") is not None and s_["E_after"] - s_["E_before"] > 1e-5), None)
                    if first is not None:
                        self.cue["at"] = first["start"]/P.AUDIO_FS + self.cue["delay"]
                if self.cue.get("at") is not None and not self.cue["on"] and w.sim_time >= self.cue["at"]:
                    self.cue["on"] = True
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state() if state else None

    def _track_still(self):
        w = self.world
        qd = w.state.joint_qd.numpy()
        moving = any(abs(float(qd[s["dof"]])) > 0.01 for n, s in w.joints.items() if "finger" not in n)
        changed = not np.array_equal(w.targets, self._last_targets)
        if moving or changed:
            self._still_since = w.sim_time
        self._last_targets = w.targets.copy()

    def robot_still_s(self):
        return self.world.sim_time - self._still_since

    # ------------------------------------------------------------------ sensing
    def mic_window(self, mic, seconds):
        """The last `seconds` of a microphone's pressure signal (Pa), zero-padded before t = 0."""
        A = self.world.acoustics
        n = int(round(seconds*P.AUDIO_FS))
        if A is None:
            return np.zeros(n)
        x = A.signal(mic, max(0, A.n - n), A.n)
        return np.r_[np.zeros(n - len(x)), x]

    def levels(self, mic, seconds=2.0, frame_s=0.01):
        from .acoustics import leq_frames
        x = self.mic_window(mic, seconds + 0.2)
        L = leq_frames(x, int(round(frame_s*P.AUDIO_FS)))
        return L[-int(round(seconds/frame_s)):]

    # ------------------------------------------------------------------ state
    def item_poses(self):
        w = self.world
        out = {}
        for name, it in w.scene.items.items():
            if "body" not in it:
                continue
            p, R = w.body_pose(it["body"])
            out[name] = dict(kind=it["kind"], position_m=[round(float(x), 5) for x in p],
                             tilt_deg=round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 3))
        return out

    def get_state(self):
        with self.lock:
            w = self.world
            q = w.state.joint_q.numpy()
            qd = w.state.joint_qd.numpy()
            joints = {n: {"position": float(q[s["q"]]), "velocity": float(qd[s["dof"]])} for n, s in w.joints.items()}
            bq = w.state.body_q.numpy()
            finite = bool(np.isfinite(bq).all() and np.isfinite(q).all())
            ev = None
            if not w.fault and finite:
                try:
                    ev = self.evaluator.evaluate()
                except Exception as exc:              # never let a scoring bug hide the physical state
                    ev = dict(success=False, score=0., error=f"{type(exc).__name__}: {exc}")
            energy = {k: float(v.energy()) for k, v in w.structs.items()}
            return dict(tick=int(w.tick), time_s=round(w.sim_time, 6), generation=self.generation, joints=joints,
                        items=self.item_poses(), ready=True, finite=finite, fault=w.fault,
                        instruments=dict(vibration_energy_J=energy, strikes=len(w.contact_log),
                                         window_ticks=int(w.timing["window_ticks"])),
                        cue=dict(self.cue), evaluation=ev or dict(success=False, score=0.))

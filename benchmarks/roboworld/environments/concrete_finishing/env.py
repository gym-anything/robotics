"""Persistent concrete-finishing environment: robot-only commands, exact 0.5 ms ticks, a material clock, explicit faults,
checkpoints.

One command = joint targets (arms, spine, fingers), an optional base twist and grip-force changes, held for N ticks
(classroom_arrangement.env.ClassroomEnv, reused), or a `wait` that advances the material clock while the robot holds
still. Nothing here moves a tool or the paste except through the robot's contacts; assisted fixtures
(probes) attach to world.fixture / world.ext_wrench and are never reachable from the backend's policy interface.
"""

import copy
import math
import threading
from pathlib import Path
import numpy as np
from classroom_arrangement.env import ClassroomEnv
from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from .config import Config
from .world import FinishWorld

REST_WINDOW_S = 0.5          # a wait needs every free body slower than REST_SPEED over this long ...
RESEAT_S = 0.2               # ... and is followed by this much full stepping (bodies re-seat on the changed surfaces)
WAIT_CONTACT_N = 0.05        # same contact/no-contact tolerance used by Coupler.after_step for a sole on a rigid base


class FinishEnv(ClassroomEnv):
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True):
        """`spec` (probes only): a layout.setup result edited by an experiment; the public backend always builds it
        from the configuration."""
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        self.spec = spec or L.setup(c.task, c.seed, variant=c.variant, **c.site())
        self.world = FinishWorld(c, self.spec)
        self.generation = 0
        self.waited_s = 0.0
        if settle:
            self._settle()
        else:
            self.settle_record = dict(seconds=0.0, skipped=True)
        self._new_history()
        from .task import Evaluator
        self.evaluator = Evaluator(self)
        self.initial = self.checkpoint()

    # ------------------------------------------------------------------ reset settle
    def _settle(self):
        """The rigid world's settle (at least 1 s, then until every free body is slower than 2 mm/s for 0.1 s, at
        most 5 s). The material clock runs with it, then is put back to the start state's age: the settle is the
        robot's and the tools' coming to rest, not working time."""
        w = self.world
        snaps = {n: r.snapshot() for n, r in w.regions.items()}
        super()._settle()
        for n, r in w.regions.items():
            changed = float(np.abs(r.h - _h(snaps[n])).max())
            if changed > 1e-6:
                raise RuntimeError(f"the reset settle changed region {n} by {changed*1e3:.3f} mm: something touched it")
            r.restore(snaps[n])
        w.timing = dict(physics_s=0., control_s=0., paste_s=0.)

    # ------------------------------------------------------------------ history
    def _new_history(self):
        super()._new_history()
        w = self.world
        extra = [n for n in w.scene.bodies if n not in self.watch]
        self.watch = self.watch + extra
        self._idx = np.array([w.scene.bodies[n] for n in self.watch], int)
        self.samples.clear()
        q = w.state.body_q.numpy()
        self.start_q = q[self._idx].copy()
        self.history = dict(max_tilt={n: 0. for n in self.watch}, reflexes=0, peak_wheel_slip=0., waits=[],
                            peak_plate_force={c.name: 0.0 for c in w.contactors}, form_peak_n=0.0)
        self._sample()

    def _sample(self):
        super()._sample()
        w = self.world
        peak = self.history.get("peak_plate_force")
        if peak is None:
            return
        for c in w.contactors:
            f = w.plate_force.get(c.name, 0.0)
            if f > peak[c.name]:
                self.history["peak_plate_force"][c.name] = float(f)

    # ------------------------------------------------------------------ the wait command
    def wait(self, seconds):
        """Advance the material clock by `seconds` (1 s steps) while the robot holds still: refused, before anything
        happens, unless every tool and the robot's base and arms have been slower than REST_SPEED for the last
        REST_WINDOW_S (the history samples every 50 ms), no arm is in a protective stop, and no tool is loaded
        against a material region. Afterwards RESEAT_S of full
        ticks re-seat every resting body on the changed surfaces."""
        w = self.world
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) \
                or not 0 < seconds <= P.WAIT_MAX:
            raise ValueError(f"wait: seconds must be a number in (0, {P.WAIT_MAX:g}]")
        with self.lock:
            reason = self.wait_refusal()
            if reason:
                raise ValueError(f"wait refused, nothing executed: {reason}")
            t0 = w.sim_time
            done = w.advance_material(float(seconds))
            self.waited_s += done
            self.history["waits"].append(dict(at_s=round(t0, 4), seconds=round(done, 3)))
            every = int(round(0.05/CP.DT))
            for _ in range(int(round(RESEAT_S/CP.DT))):
                w.step()
                if w.tick % every == 0:
                    self._sample()
            return self.get_state()

    def wait_refusal(self):
        w = self.world
        if getattr(getattr(w, "coupler", None), "falling", ()):
            return "material is falling; advance ordinary rigid ticks before accelerated wait"
        # Stillness does not mean unloaded: skipping rigid/contact ticks under a
        # pressed tool would freeze its support while the paste ages underneath.
        # Include filtered plate load to cover a single chatter tick without contact,
        # and sole/base contact for knives spanning a thin coating on a rigid board.
        loaded = [c.name for c in w.contactors if c.region is not None and
                  max(float(w.plate_force.get(c.name, 0.)), float(c.F),
                      float(getattr(w, "sole_force", {}).get(c.name, 0.))) > WAIT_CONTACT_N]
        if loaded:
            return f"tool still loaded against a material region: {', '.join(loaded[:4])}; lift it clear before accelerated wait"
        if any(w.sim_time < u for u in w._reflex_until.values()):
            return "an arm is in its protective stop"
        n = int(round(REST_WINDOW_S/0.05)) + 1
        if len(self.samples) < n:
            return f"the world has not been watched at rest for {REST_WINDOW_S} s yet (step it first)"
        recent = list(self.samples)[-n:]
        if (recent[-1][0] - recent[0][0])*CP.DT < REST_WINDOW_S - 1e-9:
            return f"the world has not been watched at rest for {REST_WINDOW_S} s yet (step it first)"
        qs = np.stack([q for _, q in recent])
        span = (recent[-1][0] - recent[0][0])*CP.DT
        moved = np.linalg.norm(qs[:, :, :3] - qs[-1][None, :, :3], axis=2).max(axis=0)
        fast = [nm for nm, d in zip(self.watch, moved) if d/span > P.REST_SPEED]
        if fast:
            return f"still moving: {', '.join(fast[:4])}"
        qd = w.state.joint_qd.numpy()
        jfast = [n for n, s in w.joints.items() if abs(float(qd[s["dof"]])) > 5e-3 and "finger" not in n]
        if jfast:
            return f"robot joints moving: {', '.join(j.split('/')[-1] for j in jfast[:4])}"
        cur = w.control.joint_target_q.numpy()
        # fingers excepted, as in the motion check above: a Franka finger closed by force on a tool, or on nothing,
        # never reaches its commanded position, and a robot holding a tool could never wait
        pend = [n for n in w._slew_names if "finger" not in n
                and abs(float(w.targets[w.joints[n]['q']] - cur[w.joints[n]['q']])) > 1e-6]
        if pend:
            return f"joint targets still being reached: {', '.join(j.split('/')[-1] for j in pend[:4])}"
        if any(abs(v) > 0 for v in w.base.cmd):
            return "the base has a twist command"
        return None

    # ------------------------------------------------------------------ state
    def get_state(self):
        s = super().get_state()
        w = self.world
        q = w.state.body_q.numpy()
        for n, i in zip(self.watch, self._idx):
            s["items"][n]["z"] = round(float(q[i, 2]), 5)
        s["material_clock_s"] = round(self.material_clock(), 3)
        s["waited_s"] = round(self.waited_s, 3)
        s["regions"] = {n: r.summary() for n, r in w.regions.items()}
        s["contact"] = {c.name: dict(region=None if c.region is None else c.region.name,
                                     plate_force_n=round(float(w.plate_force.get(c.name, 0.0)), 3),
                                     roll_ml=round(c.roll*1e6, 3)) for c in w.contactors if c.region is not None
                        or w.plate_force.get(c.name, 0.0) > 0 or c.roll > 0}
        return s

    def material_clock(self):
        regs = list(self.world.regions.values())
        return float(regs[0].clock) if regs else self.world.sim_time

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        snap = super().checkpoint()
        snap["paste"] = self.world.snapshot()
        snap["waited_s"] = self.waited_s
        return snap

    def restore(self, snap):
        with self.lock:
            super().restore(snap)
            self.world.restore_snapshot(snap["paste"])
            self.waited_s = snap["waited_s"]


def _h(snap):
    f = snap["fields"]
    return f["b"] + f["tb"] + f["ts"] + f["bl"]

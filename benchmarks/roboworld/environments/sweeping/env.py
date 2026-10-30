"""Persistent sweeping environment: robot-only commands, exact 0.5 ms ticks, explicit faults, checkpoints.

One command = joint targets (arms, spine, fingers), an optional base twist and grip-force changes, held for N ticks
(classroom_arrangement.env.ClassroomEnv, reused). Nothing here moves a tool, a container, furniture or debris except
through the robot's contacts; assisted fixtures (probes) attach to world.fixture and are never reachable from the
backend's policy interface.
"""

import threading
from pathlib import Path
import numpy as np
from classroom_arrangement.env import ClassroomEnv
from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from .config import Config
from .world import SweepWorld


class SweepEnv(ClassroomEnv):
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True):
        """`spec` (probes only): a layout.setup result edited by an experiment; the public backend always builds it
        from the configuration."""
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        self.spec = spec or L.setup(c.task, c.seed, variant=c.variant, glides=c.glides, amount=c.amount)
        self.world = SweepWorld(c, self.spec)
        self.generation = 0
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
        most 5 s), then until every loose piece sleeps (at most 8 s more)."""
        w = self.world
        super()._settle()
        w.timing["sweep_s"] = 0.0
        seconds, slept = w.settle_debris(8.0)
        self.settle_record.update(debris_seconds=round(seconds, 3), debris_all_asleep=bool(slept), debris_awake=int(w.awake))
        if w.reflexes:
            raise RuntimeError(f"robot protective stop during the reset settle: {w.reflexes[:2]}")
        w.tick = 0
        w.base.odom[:] = 0.
        w.timing = dict(physics_s=0., control_s=0., sweep_s=0.)
        w.sweeper.skipped_ticks = 0
        w.sweeper.substeps_run = 0

    # ------------------------------------------------------------------ history
    def _new_history(self):
        super()._new_history()
        w = self.world
        # tools and containers are watched like furniture (rest window, tilt history)
        extra = [n for n in w.scene.bodies if n not in self.watch]
        self.watch = self.watch + extra
        self._idx = np.array([w.scene.bodies[n] for n in self.watch], int)
        self.samples.clear()
        q = w.state.body_q.numpy()
        self.start_q = q[self._idx].copy()
        self.history = dict(max_tilt={n: 0. for n in self.watch}, reflexes=0, peak_wheel_slip=0.)
        self.start_debris = w.debris()
        self._sample()

    # ------------------------------------------------------------------ state
    def get_state(self):
        s = super().get_state()
        w = self.world
        q = w.state.body_q.numpy()
        for n, i in zip(self.watch, self._idx):
            s["items"][n]["z"] = round(float(q[i, 2]), 5)
        s["debris"] = dict(pieces=int(w.sweeper.B), awake=int(w.awake), mass_g=round(1e3*float(w.sweeper.body_mass.sum()), 3))
        s["tools_active"] = {n: bool(a) for n, a in zip(w.tool_names, w.tool_active)}
        s["fine_dust_g"] = None if w.dust is None else w.dust.summary()
        return s

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        snap = super().checkpoint()
        snap["sweep"] = self.world.snapshot()
        return snap

    def restore(self, snap):
        with self.lock:
            super().restore(snap)
            self.world.restore_snapshot(snap["sweep"])

"""Persistent gift-bags environment: robot-only actions, exact ticks, explicit faults.

Policy-visible controls are joint position targets for the arms and the Franka Hand (position or force-mode
grasp). One environment tick = one contact-world step = `substeps` robot ticks (4 ms by default). ASSISTED
fixtures act on World/IPCWorld directly and are labelled; nothing in this class can reach them.
"""

from dataclasses import asdict
import copy
import threading
from pathlib import Path

import numpy as np

from .config import Config
from . import scene as S
from .world import World
from .task import Evaluator, INSTRUCTIONS, SETTLE_TASKS

# World diagnostics a step updates outside the checkpoint (wall-time totals, reaction/impulse peaks, the last item
# velocities): put back after final_evaluation's settle branch so that the branch leaves no trace in the episode
WORLD_LOGS = ("timing", "barrier_peaks", "item_impulse_peaks", "_item_v", "unconverged_pending", "reaction_peaks",
              "_unconv_seen")


class GiftBagsEnv:
    def __init__(self, config=None, *, output_dir=None, assisted_scene_hook=None):
        """`assisted_scene_hook(spec)`: ASSISTED demonstrations only (labelled phantom fixtures added to the
        scene before the world is built); the GAR backend never passes one."""
        self.config = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        spec = S.build(self.config)
        if assisted_scene_hook is not None:
            assisted_scene_hook(spec)
        self.assisted = assisted_scene_hook is not None
        self.world = World(spec, self.config)
        self.generation = 0
        self.trace_callback = None
        self._settle()
        self.initial = self.checkpoint()
        self.evaluator = Evaluator(self)

    # ------------------------------------------------------------------ lifecycle
    def _settle(self):
        """Items settle under gravity before the task starts while the robots hold their ready pose; the
        clock then restarts at zero (the settle is part of building the start state)."""
        n = int(round(self.config.settle_s/self.config.dt))
        for _ in range(n):
            self.world.step()
        self.world.steps = 0
        self.world.rig.tick = 0
        self.world.ipc.time = 0.
        self.world.timing = dict(robot_s=0., contact_s=0., couple_s=0.)
        self.settle_stats = dict(steps=n, unconverged=self.world.ipc.stats["unconverged"],
                                 newton=self.world.ipc.stats["newton"])

    def checkpoint(self):
        w = self.world
        rig = w.rig
        return dict(
            rigid={a: getattr(rig.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
            targets=rig.targets.copy(), control=rig.control.joint_target_q.numpy().copy(), tick=rig.tick,
            steps=w.steps, wrench=rig.body_wrench.copy(), hand=copy.deepcopy(rig.hand), reflexes=list(rig.reflexes),
            ipc=w.ipc.get_state(), filtered=copy.deepcopy(getattr(w, "filtered", {})),
            history=copy.deepcopy((list(self.evaluator.history.rows), self.evaluator.history.peak))
            if hasattr(self, "evaluator") else None)

    def restore(self, snap):
        w = self.world
        rig = w.rig
        for a, v in snap["rigid"].items():
            getattr(rig.state, a).assign(v)
            getattr(rig.next_state, a).assign(v)
        rig.targets = snap["targets"].copy()
        rig.control.joint_target_q.assign(snap["control"])
        rig.tick = snap["tick"]
        w.steps = snap["steps"]
        rig.body_wrench = snap["wrench"].copy()
        rig.hand = copy.deepcopy(snap["hand"])
        rig.reflexes = list(snap["reflexes"])
        rig.solver._update_mjc_data(rig.solver.mj_data, rig.model, rig.state)
        rig.solver._mujoco.mj_forward(rig.solver.mj_model, rig.solver.mj_data)
        w.ipc.set_state(snap["ipc"])
        w.filtered = copy.deepcopy(snap["filtered"])
        w._pred = {}
        w.fault = None
        if snap.get("history") is not None and hasattr(self, "evaluator"):
            rows, peak = copy.deepcopy(snap["history"])
            self.evaluator.history.rows.clear()
            self.evaluator.history.rows.extend(rows)
            self.evaluator.history.peak = peak

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            self.evaluator = Evaluator(self)
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.rig.joints

    def set_grip(self, arm, *, width=None, force=None):
        with self.lock:
            self.world.rig.set_grip(arm, width=width, force=force)

    def act(self, targets, *, ticks, guard=None, schedule=None):
        """Set `targets` and run `ticks` contact steps; `schedule` (one joint-target dict per step) moves the targets
        step by step (a timed trajectory), `guard(env)` True stops after that step."""
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        if schedule is not None and len(schedule) != ticks:
            raise ValueError("schedule must give one target set per step")
        with self.lock:
            if self.world.fault:
                raise RuntimeError(self.world.fault)
            self.world.rig.set_targets(targets)
            for k in range(ticks):
                if schedule is not None:
                    self.world.rig.set_targets(schedule[k])
                self.world.step()
                self.evaluator.record()
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state()

    # ------------------------------------------------------------------ evaluation
    def final_evaluation(self):
        """The evaluation that judges an episode (the GAR backend's evaluate(): GET /v1/evaluation, `gar verify`, the
        benchmark's before/after calls). For a task in task.SETTLE_TASKS the hands-off window is completed here instead
        of being left to the agent, which its instruction ("Release everything when you are done") does not ask to wait:
        when the recorded history does not already end with HANDS_OFF_S free of robot
        contact, the world is advanced on a branch for the rest of that time, the robot holding its last command and
        nothing else acting (stepped as act() steps it, without the action trace), the evaluator judges the state it
        reaches, and the world, the robot and the history are restored exactly (checkpoint/restore). A robot contact in
        the branch fails the window whatever follows, so the branch stops at it and the state is judged there. Not
        advanced: other tasks (unchanged: the agent supplies the window), a faulted world, and the start state (no step
        since the build's settle or a reset; its window is empty and nothing has been released). A fault inside the
        branch raises RuntimeError after the restore: it is not a task result. Each (tick, generation) is judged once;
        a repeated call returns a copy of that result."""
        with self.lock:
            ev, w = self.evaluator, self.world
            if w.fault or w.steps == 0 or self.config.task not in SETTLE_TASKS:
                return ev.evaluate()
            key = (w.steps, self.generation)
            done = getattr(self, "_final", None)
            if done is not None and done[0] == key:
                return copy.deepcopy(done[1])
            free = 0
            for r in reversed(ev.history.rows):
                if r["contact"]:
                    break
                free += 1
            need, advanced = max(0, ev.history.n - free), 0
            if need == 0:
                out = ev.evaluate()
            else:
                snap = self.checkpoint()
                logs = {k: copy.deepcopy(getattr(w, k)) for k in WORLD_LOGS if hasattr(w, k)}
                fault = None
                try:
                    for _ in range(need):
                        w.step()
                        ev.record()
                        advanced += 1
                        if ev.history.rows[-1]["contact"]:
                            break
                    out = ev.evaluate()
                except Exception as exc:
                    fault = f"{type(exc).__name__}: {exc}"
                finally:
                    self.restore(snap)
                    for k in WORLD_LOGS:
                        if k in logs:
                            setattr(w, k, logs[k])
                        elif hasattr(w, k):
                            delattr(w, k)
                if fault is not None:
                    raise RuntimeError(f"the settle branch of the evaluation faulted after {advanced} of {need} steps "
                                       f"({fault}); the episode's state is restored; not a task result")
            # a verdict that ran a branch is a pure function of the episode's state only on one solver thread; say how it
            # was computed
            from .probes import thread_stamp
            th = thread_stamp()
            out["metrics"].update(settle_branch_s=advanced*self.config.dt, settle_branch_planned_s=need*self.config.dt,
                                  solver_threads=th.get("solver_threads"), bit_reproducible=th.get("bit_reproducible"))
            self._final = (key, copy.deepcopy(out))
            return out

    # ------------------------------------------------------------------ state
    def get_state(self, evaluate=True):
        with self.lock:
            w = self.world
            rig = w.rig
            jq, jv = rig.state.joint_q.numpy(), rig.state.joint_qd.numpy()
            ipc = w.ipc
            ev = (self.evaluator.evaluate() if (evaluate and not w.fault) else
                  dict(success=False, score=0., checks={}, metrics={}))
            from scipy.spatial.transform import Rotation
            objects = {}
            for i, b in enumerate(ipc.bodies):
                if b.kind == "dynamic":
                    p, R = ipc.body_pose(i)
                    objects[b.name] = {"position_m": p.tolist(), "quaternion_xyzw": Rotation.from_matrix(R).as_quat().tolist()}
            return {
                "tick": w.steps, "generation": self.generation, "time_s": w.steps*self.config.dt,
                "ready": w.fault is None, "fault": w.fault, "config": asdict(self.config),
                "instruction": INSTRUCTIONS[self.config.task],
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                               "target": float(rig.targets[s["q"]])} for n, s in rig.joints.items()},
                "grippers": {a: {"mode": h["mode"], "force_n_per_finger": h["force"], "half_width_target_m": h["width"]}
                             for a, h in rig.hand.items()},
                "objects": objects,
                "sheets": {s.name: {"vertices": int(sl.stop - sl.start)} for s, sl in zip(ipc.sheets, ipc.sheet_slices)},
                "velocity_reflexes": len(rig.reflexes),
                "contact_solver": {k: ipc.stats.get(k) for k in ("steps", "newton", "unconverged", "line_search_failures",
                                                                 "hessian_failures", "max_iters")},
                "terminated": False,
                "termination": None,
                "evaluation": ev,
            }

    def close(self):
        pass

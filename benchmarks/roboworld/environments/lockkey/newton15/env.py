"""Persistent environment with robot-only actions and explicit fault recovery."""

from dataclasses import asdict
import math
import threading
import numpy as np
from .assembly import build
from .config import Config
from .dynamics import Dynamics


class LockKeyEnv(Dynamics):
    def __init__(self, config=None):
        c = config or Config()
        self._initialize(c, build(c))

    def _initialize(self, c, assembly):
        builder, self.bodies, self.joints, self.mechanism_joints, self.materials = assembly
        super().__init__(builder, c)
        self.lock = threading.RLock()
        self.initial = self.model.state()
        self.initial.assign(self.state)
        self.initial_targets = self.control.joint_target_q.numpy().copy()
        self.targets = self.initial_targets.copy()
        self.renderer, self.trace_callback = None, None
        self.generation, self.released = 0, False
        from .evaluation import UnlockProgress
        self.progress = UnlockProgress(self)
        self.evaluation = self._evaluate()

    def _evaluate(self):
        q = self.state.joint_q.numpy()
        address = lambda n: self.builder.joint_q_start[self.mechanism_joints[n]]
        angle = float(np.rad2deg(q[address("plug")]))
        lid = -float(np.rad2deg(q[address("lid")]))
        rise = float(q[address("shackle")])
        swivel = float(np.rad2deg(q[address("shackle")+1])) if self.config.shackle_swivel else 0.
        from .shackle import geometry_state
        geometry = geometry_state(self.state.body_q.numpy()[self.bodies["shackle"]],
            (.10, 0., self.config.table_height+.10))
        progress = self.progress.update(self, angle)
        return {**progress, "plug_angle_deg": angle,
            "lid_open_deg": lid, "shackle_rise_m": rise, "release_triggered": self.released,
            "shackle_swivel_deg": swivel, **geometry,
            "mechanism_unlocked": self.released and geometry["shackle_short_leg_clearance_m"] > 0.}

    def _step(self, fixture_wrenches=None):
        current = self.control.joint_target_q.numpy()
        for spec in self.joints.values():
            j = spec["q"]
            slew = spec["velocity_limit"]*self.config.dt
            current[j] += np.clip(self.targets[j]-current[j], -slew, slew)
        self.control.joint_target_q.assign(current)
        if fixture_wrenches is not None:
            raise ValueError("Use a labelled component fixture, not a hidden environment force action")
        super().step()
        self._after_physics_step()

    def _after_physics_step(self):
        """Same release model and recording hook for policy and labelled fixtures."""
        self.evaluation = self._evaluate()
        if not self.released and self.evaluation["plug_angle_deg"] > 30.:
            self.released = True
            j = self.mechanism_joints["shackle"]
            q, v = self.builder.joint_q_start[j], self.builder.joint_qd_start[j]
            target = self.control.joint_target_q.numpy()
            target[q] = self.config.shackle_travel
            self.control.joint_target_q.assign(target)
            limits = self.model.joint_effort_limit.numpy()
            limits[v] = 6.
            self.model.joint_effort_limit.assign(limits)
            if self.config.mechanical_latch:
                travel = self.model.joint_limit_upper.numpy()
                travel[v] = self.config.shackle_travel
                self.model.joint_limit_upper.assign(travel)
            self.solver.notify_model_changed(self.newton.ModelFlags.JOINT_DOF_PROPERTIES)
            self.evaluation = self._evaluate()
        if self.trace_callback:
            try:
                self.trace_callback(self)
            except Exception as exc:
                self.fault = f"Recording failed after tick {self.tick}: {exc}"
                raise

    def act(self, targets, *, ticks, guard=None):
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            pending = self.targets.copy()
            for name, value in targets.items():
                if name not in self.joints:
                    raise ValueError(f"Not a robot actuator: {name}")
                spec = self.joints[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not spec["lower"] <= value <= spec["upper"]:
                    raise ValueError(f"Invalid joint target: {name}")
                pending[spec["q"]] = value
            self.targets = pending
            for _ in range(ticks):
                self._step()
                if guard and guard(self):
                    break
            return self.get_state()

    def reset(self):
        with self.lock:
            # Recreate only native dynamics state, not geometry, assets or renderer.
            Dynamics.__init__(self, self.builder, self.config)
            self.targets = self.initial_targets.copy()
            self.control.joint_target_q.assign(self.targets)
            self.generation += 1
            self.released = False
            from .evaluation import UnlockProgress
            self.progress = UnlockProgress(self)
            self.evaluation = self._evaluate()
            return self.get_state()

    def get_state(self):
        with self.lock:
            q, v = self.state.body_q.numpy(), self.state.body_qd.numpy()
            jq, jv = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
            return {"tick": self.tick, "generation": self.generation, "time_s": self.tick*self.config.dt,
                "ready": self.fault is None, "fault": self.fault, "config": asdict(self.config),
                "objects": {n: {"position_m": q[i, :3].tolist(), "quaternion_xyzw": q[i, 3:].tolist(),
                    "velocity_linear_angular": v[i].tolist()} for n, i in self.bodies.items()},
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                    "target": float(self.targets[s["q"]])} for n, s in self.joints.items()},
                "contacts": int(self.contacts.rigid_contact_count.numpy()[0]), "evaluation": self.evaluation}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

"""Persistent physical scene. Task objects are never controlled by normal actions."""

from dataclasses import asdict
import math
import threading
import numpy as np
from .build import build
from .config import Config
from .runtime import require_engine
from .solver import create, make_solver
from .state_validation import validate_state


class PenCapEnv:
    def __init__(self, config=None):
        self.config = config or Config()
        self.newton, self.wp = require_engine()
        self.lock = threading.RLock()
        with self.wp.ScopedDevice(self.config.device):
            self.builder, self.bodies, self.joints, self.materials = build(self.config)
            (self.model, self.pipeline, self.solver, self.state, self.next_state,
             self.control, self.contacts) = create(self.builder, self.config, self.joints)
        self.initial = self.model.state()
        self.initial.assign(self.state)
        self.initial_targets = self.control.joint_target_q.numpy().copy()
        self.targets = self.initial_targets.copy()
        self.tick, self.fault, self.renderer, self.trace_callback = 0, None, None, None
        self.generation = 0
        from .task import Evaluator
        self.evaluator = Evaluator()
        self.evaluation = self.evaluator.sample(self)

    def reset(self):
        with self.lock:
            self.state.assign(self.initial)
            self.next_state.assign(self.initial)
            self.targets = self.initial_targets.copy()
            self.control.joint_target_q.assign(self.targets)
            self.pipeline.native.reset_contact_matching()
            # Built-in solver state, including warm starts, belongs to reset.
            self.solver = make_solver(self.model, self.config)
            self.contacts = self.pipeline.contacts()
            self.tick, self.fault = 0, None
            self.generation += 1
            from .task import Evaluator
            self.evaluator = Evaluator()
            self.evaluation = self.evaluator.sample(self)
            return self.get_state()

    def act(self, targets, *, ticks, guard=None):
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if self.fault:
                raise RuntimeError(f"Environment faulted; explicit reset required: {self.fault}")
            pending = self.targets.copy()
            for name, value in targets.items():
                if name not in self.joints:
                    raise ValueError(f"Unknown robot joint: {name}")
                spec = self.joints[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not spec["lower"] <= value <= spec["upper"]:
                    raise ValueError(f"Joint target outside physical limits: {name}")
                pending[spec["q"]] = value
            self.targets = pending
            for _ in range(ticks):
                self._step()
                if guard and guard(self):
                    break
        return self.get_state()

    def _step(self, fixture_wrenches=None):
        self.state.clear_forces()
        current = self.control.joint_target_q.numpy()
        for spec in self.joints.values():
            # Physical actuator slew, not a changed requested target or duration.
            j = spec["q"]
            delta = spec["velocity_limit"]*self.config.dt
            current[j] += np.clip(self.targets[j]-current[j], -delta, delta)
        self.control.joint_target_q.assign(current)
        if fixture_wrenches is not None:
            self.state.body_f.assign(fixture_wrenches)
        try:
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, self.config.dt)
            validate_state(self.model, self.next_state, self.tick+1)
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self.evaluation = self.evaluator.sample(self)
            if self.trace_callback:
                self.trace_callback(self)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def get_state(self):
        with self.lock:
            q, v = self.state.body_q.numpy(), self.state.body_qd.numpy()
            jq, jv = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
            return {"tick": self.tick, "generation": self.generation, "time_s": self.tick*self.config.dt,
                "ready": self.fault is None, "fault": self.fault, "config": asdict(self.config),
                "numerics": {"integrator": {0:"euler", 1:"rk4", 2:"implicit", 3:"implicitfast"}[
                    int(self.solver.mj_model.opt.integrator)],
                    "rne_bias_derivative_sign_corrected": type(self.solver).__name__ == "CorrectedImplicitSolver"},
                "objects": {name: {"position_m": q[i, :3].tolist(), "quaternion_xyzw": q[i, 3:].tolist(),
                    "velocity_linear_angular": v[i].tolist()} for name, i in self.bodies.items()},
                "joints": {name: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                    "target": float(self.targets[s["q"]])} for name, s in self.joints.items()},
                "contacts": int(self.contacts.rigid_contact_count.numpy()[0]),
                "evaluation": self.evaluation}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

"""Persistent step-based environment; only robot actuators are policy writable."""
from collections import deque
from dataclasses import asdict
import math
import threading
import numpy as np
from .config import Config, GARMENTS
from .simulation import Simulation

MOTION_STRIDE_TICKS = 24


class LaundryEnv:
    def __init__(self, config=None, *, start=None):
        self.config = config or Config()
        self.lock = threading.RLock()
        # start=None: the task's start (an at-rest lay where one is recorded); "catalogue": the P13 lay.
        self.sim = Simulation(self.config, start=start)
        self.tick, self.generation, self.fault = 0, 0, None
        self.trace_callback, self.renderer = None, None
        from .readback import Readback
        self.readback = Readback()
        self.step_arrays = None
        self._bind()
        from ..garment_task import SemanticGarmentIndex, GarmentTaskEvaluator, GARMENT_TASK_SPECS, TabletopFrame
        matches = [s for (c, _), s in GARMENT_TASK_SPECS.items() if c == GARMENTS[self.config.garment]]
        if len(matches) != 1:
            raise ValueError("Each garment must select exactly one documented task")
        self.task = GarmentTaskEvaluator(matches[0], SemanticGarmentIndex(
            self.sim.construction, self.sim.pattern_mesh, self.sim.topology),
            reference_positions_m=self.sim.reset_positions)
        self.tabletop = TabletopFrame(self.config.table_top, self.config.table_half_x, self.config.table_half_y)
        # Recent particle positions, kept only for tasks that define rest over a time window.
        self.motion = None if self.task.spec.rest_window_s is None else deque()
        self._record_motion(reset=True)

    def _record_motion(self, *, reset=False):
        if self.motion is None:
            return
        if reset:
            self.motion.clear()
        elif self.tick % MOTION_STRIDE_TICKS:
            return
        positions = self.state.particle_q.numpy() if reset else self.step_arrays["particle_q"]
        self.motion.append((self.tick, np.array(positions, dtype=np.float64)))
        window = int(math.ceil(self.task.spec.rest_window_s/self.config.dt))
        while len(self.motion) > 1 and self.motion[1][0] <= self.tick-window:
            self.motion.popleft()

    def _bind(self):
        self.newton = self.sim.newton
        for name in ("builder", "model", "pipeline", "contacts", "control", "joints", "materials"):
            setattr(self, name, getattr(self.sim, name))
        self.solver = self.sim.rigid_solver

    @property
    def state(self):
        return self.sim.state_0

    @property
    def targets(self):
        return self.sim.targets

    def reset(self):
        with self.lock:
            if self.trace_callback:
                raise RuntimeError("Cannot reset inside an active recorded action")
            self.sim.reset()
            self._bind()
            self.tick, self.fault = 0, None
            self.step_arrays = None
            self._record_motion(reset=True)
            self.generation += 1
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
                    raise ValueError(f"Unknown robot joint {name}")
                s = self.joints[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not s["lower"] <= value <= s["upper"]:
                    raise ValueError(f"Invalid target for {name}; no physics executed")
                pending[s["q"]] = value
            self.sim.targets = pending
            for _ in range(ticks):
                try:
                    self.sim.step()
                    self.tick = self.sim.step_count
                    self.step_arrays = self.readback.read({key:getattr(self.state, key)
                        for key in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")})
                    for key, value in self.step_arrays.items():
                        if not np.isfinite(value).all():
                            raise FloatingPointError(f"Non-finite {key} at tick {self.tick}")
                    self._record_motion()
                    if self.trace_callback:
                        self.trace_callback(self)
                except Exception as exc:
                    self.fault = f"{type(exc).__name__}: {exc}"
                    raise
                if guard and guard(self):
                    break
            return self.get_state()

    def evaluate(self):
        from ..garment_assembly import _strain_metrics
        x, v = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        top = self.sim.topology
        strain = _strain_metrics(x, top.physical_triangles, top.panel_material_coordinates_m, top.panel_triangles)
        n = min(int(self.contacts.soft_contact_count.numpy()[0]), self.contacts.soft_contact_max)
        shapes = self.contacts.soft_contact_shape.numpy()[:n]
        dynamic = [i for i in shapes if self.builder.shape_body[int(i)] >= 0]
        clear = not dynamic
        recent = {}
        if self.motion is not None:
            recent = dict(recent_positions_m=np.stack([p for _, p in self.motion]),
                          recent_span_s=(self.tick-self.motion[0][0])*self.config.dt)
        result = self.task.evaluate(x, v, maximum_structural_strain=max(abs(strain.minimum_principal_stretch-1),
            abs(strain.maximum_principal_stretch-1)), nonlocal_intersection_count=self.sim.self_intersection_count(),
            runtime_pose_writes=0, external_particle_force_applications=0, released=clear, fixtures_clear=clear,
            tabletop=self.tabletop, principal_stretch_range=(strain.minimum_principal_stretch,
            strain.maximum_principal_stretch), **recent)
        value = {
            "success": result.success, "score": result.potential, "gates": dict(result.gates),
            "failed_gates": list(result.failed_gates), "goal_scores": dict(result.goal_scores)}
        if result.progress is not None:
            value["progress"] = result.progress
        return value

    def get_state(self):
        with self.lock:
            if self.fault:
                raise RuntimeError(f"Environment faulted at tick {self.tick}: {self.fault}")
            q, v = self.state.body_q.numpy(), self.state.body_qd.numpy()
            jq, jv = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
            return {"tick": self.tick, "generation": self.generation, "time_s": self.tick*self.config.dt,
                "ready": True, "fault": None, "config": asdict(self.config),
                "garment": {"id": self.sim.construction.id, "positions_m": self.state.particle_q.numpy().tolist(),
                    "velocities_m_s": self.state.particle_qd.numpy().tolist(), "mass_kg": self.sim.inputs.total_mass_kg},
                "objects": {name: {"position_m": q[i, :3].tolist(), "quaternion_xyzw": q[i, 3:].tolist(),
                    "velocity_linear_angular": v[i].tolist()} for i, name in enumerate(self.builder.body_label)},
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                    "target": float(self.targets[s["q"]])} for n, s in self.joints.items()},
                "contacts": {"rigid": int(self.contacts.rigid_contact_count.numpy()[0]),
                    "cloth": int(self.contacts.soft_contact_count.numpy()[0])},
                "coupling": {k:v.tolist() if isinstance(v, np.ndarray) else v
                    for k,v in self.sim.coupling_snapshot().items()},
                "evaluation": self.evaluate()}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

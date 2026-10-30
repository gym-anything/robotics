"""Persistent, explicitly stepped Newton 1.5 simulation. No background clock."""

import copy
from pathlib import Path
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from .config import Config
from .geometry import classroom, tools
from .ink import InkSurface
from .runtime import require_newton


class WhiteboardEnv:
    def __init__(self, config=None, *, fixture_mode=False, task=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass either Config or individual parameters, not both")
        self.config = config or Config(**kwargs)
        self.fixture_mode = bool(fixture_mode)
        from .storage import mechanics_fingerprint

        self._mechanics_fingerprint = mechanics_fingerprint(self)
        from .tasks import Task

        self.task = task or Task()
        self.task.validate_board(self.config.board)
        self._lock = threading.RLock()
        self.newton, self.wp = require_newton()
        n, wp = self.newton, self.wp
        wp.set_device(self.config.device)
        self.objects = classroom(self.config) + tools(self.config)
        builder = n.ModelBuilder(up_axis=n.Axis.Z)
        n.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.ke = 20000
        builder.default_shape_cfg.kd = 10
        builder.default_shape_cfg.gap = 0.0002
        self.body_ids, self.shape_ids = {}, {}
        self.robot_joints = {}
        for obj in self.objects:
            body = -1
            if obj.mass:
                body = builder.add_body(
                    xform=wp.transform(obj.position, obj.rotation), label=obj.name
                )
                self.body_ids[obj.name] = body
            r = Rotation.from_quat(obj.rotation)
            for part in obj.parts:
                pos, quat = part.position, part.rotation
                if body < 0:
                    pos = np.asarray(obj.position) + r.apply(pos)
                    quat = (r * Rotation.from_quat(quat)).as_quat()
                cfg = builder.ShapeConfig(
                    density=1000,
                    mu=part.friction,
                    ke=20000,
                    kd=10,
                    has_shape_collision=part.collision,
                    has_particle_collision=False,
                    margin=0.0,
                    gap=0.0002,
                )
                label = f"{obj.name}/{part.name}"
                transform = wp.transform(wp.vec3(*pos), wp.quat(*quat))
                if part.kind == "box":
                    x, y, z = np.asarray(part.size) / 2
                    shape = builder.add_shape_box(
                        body, xform=transform, hx=x, hy=y, hz=z, cfg=cfg, label=label
                    )
                elif part.kind == "mesh":
                    source = n.Mesh(
                        vertices=part.vertices.astype(np.float32),
                        indices=part.faces.flatten().astype(np.int32),
                    )
                    shape = builder.add_shape_mesh(
                        body, xform=transform, mesh=source, cfg=cfg, label=label
                    )
                else:
                    radius, length = part.size
                    shape = builder.add_shape_cylinder(
                        body,
                        xform=transform,
                        radius=radius,
                        half_height=length / 2,
                        cfg=cfg,
                        label=label,
                    )
                self.shape_ids[label] = shape
            if body >= 0:
                # Preserve geometry-derived inertia distribution while matching selected total mass.
                factor = obj.mass / builder.body_mass[body]
                builder.body_mass[body] = obj.mass
                builder.body_inertia[body] = wp.mat33(
                    np.asarray(builder.body_inertia[body]).reshape(3, 3) * factor
                )
        # The compliant tip/pad represents compression against the coated board
        # assembly. Its undeformed collision proxy must not ALSO hit the MDF
        # 1 mm behind the coating and produce a second normal/friction wrench.
        for board_part in ("porcelain", "mdf_backer"):
            board_shape = self.shape_ids["board/" + board_part]
            for name in self.ink_colors:
                builder.shape_collision_filter_pairs.append(
                    (board_shape, self.shape_ids[f"marker_{name}/nib"])
                )
            builder.shape_collision_filter_pairs.append((board_shape, self.shape_ids["eraser/pad"]))
        # Cap-wall/barrel contact is owned by resolved elastic shell material.
        # Keep the cap end, fingers, other objects and every unrelated pair.
        for color in self.ink_colors:
            for wall in range(12):
                for part in ("barrel", "collar", "tail", "nib"):
                    builder.shape_collision_filter_pairs.append(
                        (
                            self.shape_ids[f"cap_{color}/wall_{wall}"],
                            self.shape_ids[f"marker_{color}/{part}"],
                        )
                    )
        first_robot_shape = builder.shape_count
        self.robot_joints = self._add_articulation(builder)
        self.robot_shapes = range(first_robot_shape, builder.shape_count)
        self.builder = builder
        self.model = builder.finalize(device=self.config.device)
        from .rigid_contact import (
            configure_rigid_contact,
            configure_robot_contact,
            configure_surface_masks,
        )

        configure_rigid_contact(self.model)
        configure_robot_contact(self.model, self.robot_shapes, self.config.physics_dt)
        configure_surface_masks(self.model, self.shape_ids)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        n.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.com = self.model.body_com.numpy()
        from .solver import WhiteboardSolver

        self.solver = WhiteboardSolver(
            self.model,
            use_mujoco_cpu=self.config.device == "cpu",
            iterations=60,
            ls_iterations=20,
            integrator="implicitfast",
            jacobian=self.config.jacobian,
            solver=self.config.constraint_solver,
            cone="elliptic",
            njmax=6000,
            nconmax=1500,
            use_mujoco_contacts=True,
            enable_multiccd=True,
            update_data_interval=0,
            deterministic=wp.DeterministicMode.RUN_TO_RUN
            if self.config.deterministic
            else wp.DeterministicMode.NOT_GUARANTEED,
        )
        self.solver.mj_model.opt.noslip_iterations = self.config.noslip_iterations
        if not self.solver.use_mujoco_cpu:
            import mujoco

            # MuJoCo-Warp 3.11 launches tactile preprocessing even with zero
            # sensors. That unused kernel mixes atomic-max/add and cannot be
            # compiled in Warp 1.16 deterministic mode. Skip only the EMPTY
            # native sensor pipeline, never dynamics/contact or camera capture.
            if self.solver.mj_model.nsensor != 0:
                raise RuntimeError(
                    "GPU native sensor pipeline needs its own deterministic verification"
                )
            self.solver.mjw_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        from .cpu_collision import refit_cpu_bvh

        self.collision_refit = refit_cpu_bvh(self.solver.mj_model)
        self._needs_sync = True
        self.tick, self.action_id, self.healthy = 0, 0, True
        self.ink = InkSurface(
            self.config.board,
            self.config.texture_width,
            self.config.texture_height,
            self.config.materials,
        )
        self._previous_contacts = {}
        self.deformations = {}
        self._pad_contact_state = None
        self.last_contacts = []
        self._renderer = None
        from .history import Trace

        self.trace = Trace()
        from .energy import Ledger
        from .contact import tool_contacts

        self.ledger = Ledger(self)
        self.last_audit = None
        tool_contacts(self, self.state_0.body_q.numpy(), self.state_0.body_qd.numpy())
        self._initial = self.snapshot()

    @property
    def ink_colors(self):
        from .config import COLORS

        return tuple(COLORS)

    def _add_articulation(self, builder):
        """Normal G1 assembly; labelled component fixtures may override it."""
        if not self.config.robot:
            return {}
        from .robot import add_robot

        return add_robot(builder, self.wp, self.config.layout)

    def _apply_ink(self, contacts):
        dt = self.config.physics_dt
        for name, color, patch in contacts:
            old = self._previous_contacts.get(name)
            if patch is None:
                self._previous_contacts.pop(name, None)
                continue
            p = patch["center"]
            distance = 0 if old is None else float(np.linalg.norm(p - old))
            # Follow actual continuous contact, with spatial integration within a physics tick.
            texel = min(
                self.config.board.writing_width / self.config.texture_width,
                self.config.board.writing_height / self.config.texture_height,
            )
            count = max(1, int(np.ceil(distance / (texel * 0.5))))
            if count > 4096:
                raise FloatingPointError("Tool moved an unphysical distance in one physics tick")
            for i in range(1, count + 1):
                point = p if old is None else old + (p - old) * (i / count)
                if color:
                    self.ink.deposit(
                        color,
                        point,
                        patch["axes"],
                        patch["load"],
                        dt / count,
                        self.tick * dt,
                    )
                else:
                    self.ink.erase(
                        point,
                        patch["axes"],
                        patch["load"],
                        distance / count,
                        self.tick * dt,
                    )
            self._previous_contacts[name] = p.copy()

    def step(self, ticks=1, *, _external_forces=None, _return_state=True):
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        if _external_forces is not None and not self.fixture_mode:
            raise PermissionError("External test loads are fixture-only")
        from .contact import tool_contacts

        with self._lock:
            if not self.healthy:
                raise RuntimeError(
                    "Simulation failed; inspect failure and explicitly reset/restore"
                )
            if self.tick + ticks > self.config.episode_ticks:
                raise ValueError("Action would exceed episode_ticks; no partial/clamped action")
            try:
                for _ in range(ticks):
                    if self.trace.store is not None:
                        self.trace.store.preflight()
                    audit_before = self.ledger.mechanical(self)
                    audit_sync = self._needs_sync
                    q, qd = self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()
                    forces, contacts = tool_contacts(self, q, qd, _external_forces)
                    if _external_forces is not None:
                        forces += _external_forces
                    self.state_0.clear_forces()
                    self.state_0.body_f.assign(forces)
                    if self.solver.use_mujoco_cpu:
                        warnings_before = np.asarray(self.solver.mj_data.warning.number).copy()
                    # Native MuJoCo retains its double-precision integration state.
                    # Do not round-trip through float32 Newton outputs on every tick.
                    # Explicit fixture/reset operations request one resynchronization.
                    self.solver.update_data_interval = 1 if self._needs_sync else 0
                    self.solver.step(
                        self.state_0,
                        self.state_1,
                        self.control,
                        None,
                        self.config.physics_dt,
                    )
                    self._needs_sync = False
                    self.solver.update_data_interval = 0
                    self.state_0, self.state_1 = self.state_1, self.state_0
                    self.tick += 1
                    if self.solver.use_mujoco_cpu:
                        # MuJoCo may reset a diverged state internally to finite values.
                        # Treat bad position/velocity/acceleration warnings as failure,
                        # not an apparently healthy, silently restarted simulation.
                        warnings_after = np.asarray(self.solver.mj_data.warning.number)
                        if np.any(warnings_after > warnings_before):
                            import mujoco

                            names = [
                                mujoco.mjtWarning(int(i)).name
                                for i in np.flatnonzero(warnings_after > warnings_before)
                            ]
                            raise FloatingPointError(
                                f"MuJoCo warning at tick {self.tick}: {', '.join(names)}"
                            )
                    else:
                        from .gpu_runtime import readback

                        readback(self.solver)
                    if (
                        not np.isfinite(self.state_0.body_q.numpy()).all()
                        or not np.isfinite(self.state_0.body_qd.numpy()).all()
                    ):
                        raise FloatingPointError(f"Non-finite rigid state at tick {self.tick}")
                    self._apply_ink(contacts)
                    self.last_contacts = [
                        {
                            "tool": name,
                            "load_n": float(p["load"]),
                            "compression_m": float(p["compression"]),
                        }
                        for name, _, p in contacts
                        if p
                    ]
                    self.last_audit = self.ledger.finish(self, audit_before, forces, audit_sync)
                    self.trace.record(self)
            except Exception as exc:
                self.healthy = False
                try:
                    self.trace.event(
                        "physics_failure", self.tick, error=type(exc).__name__, detail=str(exc)
                    )
                except Exception:
                    pass  # Recording failure must not replace the original exception.
                raise
            return self.get_state() if _return_state else None

    def act(self, targets, *, ticks):
        """Joint position targets in radians; reject unknown/out-of-range values before advancing."""
        with self._lock:
            if not isinstance(targets, dict):
                raise ValueError("targets must be an object mapping joint names to radians")
            if not self.healthy:
                raise RuntimeError("Simulation failed; reset or restore before actions")
            # CPU numpy() aliases native memory. Validation must use a detached
            # candidate, otherwise a later invalid field changes live targets.
            proposed = self.control.joint_target_q.numpy().copy()
            for name, value in targets.items():
                if name not in self.robot_joints:
                    raise ValueError(f"Unknown joint {name}")
                j = self.robot_joints[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or not j["lower"] <= value <= j["upper"]
                ):
                    raise ValueError(f"{name}: target outside physical joint limits")
                proposed[j["q"]] = value
            if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
                raise ValueError("ticks must be a positive integer")
            if self.tick + ticks > self.config.episode_ticks:
                raise ValueError("Action would exceed episode_ticks; no partial/clamped action")
            self.control.joint_target_q.assign(proposed)
            self.action_id += 1
            return self.step(ticks)

    def get_state(self):
        """Privileged diagnostics. Calling this never advances physics."""
        with self._lock:
            q, qd = self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()
            return {
                "tick": self.tick,
                "time_s": self.tick * self.config.physics_dt,
                "action_id": self.action_id,
                "ready": self.healthy,
                "newton_version": "1.5.0",
                "fixture_mode": self.fixture_mode,
                "episode_ticks": self.config.episode_ticks,
                "truncated": self.tick >= self.config.episode_ticks,
                "objects": {name: q[i].tolist() for name, i in self.body_ids.items()},
                "bodies": {
                    name: {"pose_xyzw": q[i].tolist(), "velocity": qd[i].tolist()}
                    for i, name in enumerate(self.model.body_label)
                },
                "joints": self.proprioception()["joints"],
                "ink": self.ink.metadata(),
                "tool_contacts": copy.deepcopy(self.last_contacts),
                "physics_audit": copy.deepcopy(self.last_audit),
                "deformations": {
                    name: {
                        key: value.tolist()
                        if isinstance(value, np.ndarray)
                        else copy.deepcopy(value)
                        for key, value in record.items()
                    }
                    for name, record in self.deformations.items()
                },
            }

    def proprioception(self):
        with self._lock:
            q, qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
            return {
                "tick": self.tick,
                "joints": {
                    name: {
                        "position_rad": float(q[j["q"]]),
                        "velocity_rad_s": float(qd[j["dof"]]),
                    }
                    for name, j in self.robot_joints.items()
                },
            }

    def evaluate(self):
        with self._lock:
            return self.task.evaluate(self.ink)

    def fixture_pose(self, name, position, rotation=(0, 0, 0, 1), velocity=(0, 0, 0, 0, 0, 0)):
        """Explicit test intervention, never a policy action. Clears stroke continuity."""
        if not self.fixture_mode:
            raise PermissionError("Direct positioning requires fixture_mode=True")
        with self._lock:
            if any(
                np.asarray(v).shape != (size,) or not np.isfinite(v).all()
                for v, size in ((position, 3), (rotation, 4), (velocity, 6))
            ):
                raise ValueError(
                    "Fixture pose/velocity must have finite components and correct shape"
                )
            if abs(np.linalg.norm(rotation) - 1) > 1e-6:
                raise ValueError("Fixture quaternion must be normalized")
            i = self.body_ids[name]
            joint = next(j for j, b in enumerate(self.builder.joint_child) if b == i)
            qs, ds = (
                self.builder.joint_q_start[joint],
                self.builder.joint_qd_start[joint],
            )
            q, qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
            q[qs : qs + 7] = [*position, *rotation]
            qd[ds : ds + 6] = velocity
            self.state_0.joint_q.assign(q)
            self.state_0.joint_qd.assign(qd)
            self.newton.eval_fk(
                self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
            )
            if self.solver.use_mujoco_cpu:
                self.solver.mj_data.qacc_warmstart[:] = 0
                self.solver.mj_data.qacc[:] = 0
            self._previous_contacts.pop(name, None)
            if name == "eraser":
                self._pad_contact_state = None
            self._needs_sync = True

    def snapshot(self):
        with self._lock:
            data = {
                name: getattr(self.state_0, name).numpy().copy()
                for name in ("body_q", "body_qd", "joint_q", "joint_qd")
            }
            data.update(
                tick=self.tick,
                action_id=self.action_id,
                ink=copy.deepcopy(self.ink),
                needs_sync=self._needs_sync,
                target=self.control.joint_target_q.numpy().copy(),
                previous=copy.deepcopy(self._previous_contacts),
                contacts=copy.deepcopy(self.last_contacts),
                pad_contact_state=copy.deepcopy(self._pad_contact_state),
                deformations=copy.deepcopy(self.deformations),
                physics_audit=copy.deepcopy(self.last_audit),
            )
            if self.solver.use_mujoco_cpu:
                import mujoco

                state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                integration = np.zeros(mujoco.mj_stateSize(self.solver.mj_model, state_spec))
                mujoco.mj_getState(
                    self.solver.mj_model, self.solver.mj_data, integration, state_spec
                )
                data["solver_integration"] = integration
                data["solver_step"] = self.solver._step
            else:
                from .gpu_runtime import snapshot

                data["gpu_integration"] = snapshot(self.solver)
                data["solver_step"] = self.solver._step
            return data

    def restore(self, snapshot):
        """Explicit branch: restores physics and pigment state; not continuous rollout evidence."""
        with self._lock:
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                getattr(self.state_0, name).assign(snapshot[name])
                getattr(self.state_1, name).assign(snapshot[name])
            self.control.joint_target_q.assign(snapshot["target"])
            self.tick, self.action_id = snapshot["tick"], snapshot["action_id"]
            self._needs_sync = snapshot["needs_sync"]
            self.ink = copy.deepcopy(snapshot["ink"])
            self._previous_contacts = copy.deepcopy(snapshot["previous"])
            self.last_contacts = copy.deepcopy(snapshot["contacts"])
            self._pad_contact_state = copy.deepcopy(snapshot["pad_contact_state"])
            self.deformations = copy.deepcopy(snapshot["deformations"])
            self.last_audit = copy.deepcopy(snapshot["physics_audit"])
            if self.solver.use_mujoco_cpu:
                import mujoco

                mujoco.mj_resetData(self.solver.mj_model, self.solver.mj_data)
                mujoco.mj_setState(
                    self.solver.mj_model,
                    self.solver.mj_data,
                    snapshot["solver_integration"],
                    mujoco.mjtState.mjSTATE_INTEGRATION,
                )
                self.solver._step = snapshot["solver_step"]
            else:
                from .gpu_runtime import restore

                restore(self.solver, snapshot["gpu_integration"])
                self.solver._step = snapshot["solver_step"]
            self.healthy = True
            self.trace.event("restore_discontinuity", self.tick)
            return self.get_state()

    def reset(self):
        return self.restore(self._initial)

    def export(self, path):
        """Immutable render package: same current Newton poses and board pigment."""
        from .usd_scene import export_scene

        with self._lock:
            return export_scene(self, Path(path))

    def open_renderer(self, **kwargs):
        from .render_client import LiveRenderer

        with self._lock:
            if self._renderer is not None:
                raise RuntimeError("A renderer is already open")
            self._renderer = LiveRenderer(self, **kwargs)
            return self._renderer

    def obs(self, cameras=("g1_rgb",)):
        if self._renderer is None:
            raise RuntimeError("Call env.open_renderer() once before camera observations")
        return self._renderer.observe(cameras)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

"""Preserved Style3D cloth plus a force-reactive articulated robot workcell."""
import numpy as np
from ..newton_style3d_garment import Style3DGarmentSimulation, Style3DSolverConfig
from ..newton_style3d_ipc import IPCStyle3DSelfContactConfig
from ..newton_style3d_table_ccd import TablePlaneCCDConfig
from ..specs import MATERIAL_PROFILES
from .runtime import require_engine


class Simulation(Style3DGarmentSimulation):
    def __init__(self, c, start=None):
        from .garment import load, rest_start
        if start not in (None, "catalogue"):
            raise ValueError("start must be None (the task's start) or 'catalogue' (the P13 lay)")
        self.workcell = c
        require_engine()
        import ipctk
        # IPC uses a process-wide TBB worker cap. This changes scheduling, not
        # collision candidates, tolerances, iteration counts or cloth resolution.
        # A microbenchmark improved with four workers, but the complete camera
        # workload did not. Keep the library default unless explicitly selected.
        if c.ipc_threads is not None:
            ipctk.set_num_threads(c.ipc_threads)
        from .readback import Readback
        self.contact_readback = Readback()
        construction, pattern, topology, positions = load(c)
        settled = None if start == "catalogue" else rest_start(c, topology)
        self.start_kind = "catalogue_p13" if settled is None else "at_rest"
        if settled is not None:
            # Already in world coordinates and resting on this tabletop: no lift.
            self.reset_positions = settled.copy()
        else:
            self.reset_positions = positions.copy()
            # P13 shape origins differ (some pocket/hem vertices are below zero).
            # A rigid Z translation places the lowest physical surface above the
            # table; no sewing, rest geometry or within-garment relationship changes.
            lowest = float(np.min(positions[:, 2]-topology.physical_vertex_radius_m))
            self.reset_positions[:, 2] += c.table_top+c.initial_table_clearance-lowest
        # Preserve the P12 material/topology and 20/60/self-contact/CCD settings.
        # Robot coupling's timestep/contact damping upgrade is explicit in c.
        material = MATERIAL_PROFILES[topology.material_profile_ids[0]]
        config = Style3DSolverConfig(device=c.device, time_step_s=c.dt,
            nonlinear_iterations=c.nonlinear_iterations, linear_iterations=c.linear_iterations,
            gravity_m_s2=9.80665, enable_self_contact=True, enable_table=True,
            initial_translation_m=(0., 0., c.table_top+.01), table_top_z_m=c.table_top,
            table_half_extent_x_m=c.table_half_x, table_half_extent_y_m=c.table_half_y,
            table_thickness_m=.04, table_friction_mu=material.table_kinetic_friction,
            contact_damping_n_s_m=c.contact_damping,
            enable_membrane_rate_damping=True, enable_bending_rate_damping=True,
            enable_air_drag=True, ipc_self_contact=IPCStyle3DSelfContactConfig(local_contact_projection=c.local_contact_projection),
            table_plane_ccd=TablePlaneCCDConfig())
        super().__init__(construction, pattern, topology, config=config)
        self.initial = self.model.state()
        self.initial.assign(self.state_0)
        self.initial_targets = self.control.joint_target_q.numpy().copy()
        self.targets = self.initial_targets.copy()

    def read_contact_arrays(self, **arrays):
        return self.contact_readback.read(arrays)

    def _initialize_state(self):
        # Rest geometry has already initialized the solver. Integrity guards
        # now inspect the selected P13 reset, not a temporary assembly pose.
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(self.reset_positions.astype(np.float32))
            state.particle_qd.zero_()

    def _extend_builder(self, builder):
        from .robot import add_robot
        from .scene import build
        self.newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.request_state_attributes("mujoco:qfrc_actuator")
        self.joints = {}
        for arm in ("left", "right"):
            self.joints.update(add_robot(builder, arm, self.workcell))
        self.materials = build(builder, self.workcell, self.newton, self.wp)
        self.materials[self.table_shape] = "wood"
        self.builder = builder

    def _after_build(self):
        from .coupling import ReactionCollision
        from .graph import CapturedClothStep
        from .graph import CapturedCollision
        if isinstance(self.pipeline, CapturedCollision):
            self.pipeline = self.pipeline.native
        if self.workcell.capture_collision:
            self.pipeline = CapturedCollision(self.pipeline)
        if isinstance(self.solver, CapturedClothStep):
            self.solver = self.solver.native
        self.model.request_contact_attributes("force")
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        native = self.solver.collision
        if isinstance(native, ReactionCollision):
            native = native.native
        native.integrate_with_external_rigid_solver = True
        self.solver.collision = ReactionCollision(native, self)
        from .graph import CapturedPCG
        if not isinstance(self.solver.linear_solver, CapturedPCG):
            self.solver.linear_solver = CapturedPCG(self.solver.linear_solver)
        else:
            self.solver.linear_solver.graphs.clear()
        self.rigid_solver = self.newton.solvers.SolverMuJoCo(self.model,
            use_mujoco_contacts=False, integrator="implicitfast", cone="elliptic", solver="newton",
            iterations=self.workcell.rigid_iterations, ls_iterations=20,
            nconmax=self.workcell.contact_capacity, njmax=self.workcell.contact_capacity*3)
        if self.workcell.capture_rigid:
            from .graph import CapturedRigidStep
            self.rigid_solver = CapturedRigidStep(self.rigid_solver)
        self.applied_wrenches = self.wp.zeros(self.model.body_count, dtype=self.wp.spatial_vector,
                                            device=self.config.device)
        if self.workcell.capture_cloth:
            self.solver = CapturedClothStep(self.solver)

    def _prepare_external_shapes(self, dt):
        # Native Style3D uses previous/current rigid transforms to evaluate slip.
        # Advance physical robot joints, not authored link poses. Previous cloth
        # reaction is the explicitly documented one-tick partition delay.
        self.applied_wrenches.assign(self.solver.collision.wrenches)
        self.state_0.body_f.assign(self.applied_wrenches)
        current = self.control.joint_target_q.numpy()
        for spec in self.joints.values():
            j = spec["q"]
            limit = spec["velocity_limit"]*dt
            current[j] += np.clip(self.targets[j]-current[j], -limit, limit)
        self.control.joint_target_q.assign(current)
        self.pipeline.collide(self.state_0, self.contacts)
        self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)

    def coupling_snapshot(self):
        return dict(method="native_style3d_contact_lagged_reaction", delay_ticks=1,
            applied_body_wrenches=self.applied_wrenches.numpy(),
            next_tick_body_wrenches=self.solver.collision.wrenches.numpy())

    def reset(self):
        self.state_0.assign(self.initial)
        self.state_1.assign(self.initial)
        self.control.joint_target_q.assign(self.initial_targets)
        self.targets = self.initial_targets.copy()
        self.pipeline.reset_contact_matching()
        self.contacts = self.pipeline.contacts()
        self._after_build()
        self.solver.rebuild_bvh(self.state_0)
        from ..newton_style3d_ipc import IPCStyle3DSelfContactController
        from ..newton_style3d_table_ccd import TablePlaneCCDController
        self._ipc_self_contact_controller = IPCStyle3DSelfContactController(self, self.config.ipc_self_contact)
        self._table_plane_ccd_controller = TablePlaneCCDController(self, self.config.table_plane_ccd)
        self.step_count = 0

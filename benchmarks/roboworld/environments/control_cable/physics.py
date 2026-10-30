"""Native Newton 1.5 two-way rigid/cable dynamics, initially a coupling fixture.

No task-success flags, auto-grasps, hidden kinematic object motion or policy
phantom-force action exists here. The isolated fixture is not the final scene.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import math
import time
import xml.etree.ElementTree as ET
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization
from .config import Config

HOME = (0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.79, 0.04, 0.04)
FR3_MAX_JOINT_ACCELERATION_RAD_S2 = 10.0
FR3_MAX_JOINT_JERK_RAD_S3 = 5000.0
FR3_VELOCITY_FRACTION = 0.95


class FR3Trajectory:
    """Jerk-limited FR3 desired motion, including the velocity servo reference."""

    def __init__(self, joints, state, dt):
        self.dt = dt
        self.arms = {}
        for arm in ("left", "right"):
            names = [f"{arm}/fr3_joint{i}" for i in range(1, 8)]
            if not all(name in joints for name in names):
                continue
            q = np.array([joints[name]["q"] for name in names], dtype=int)
            dof = np.array([joints[name]["dof"] for name in names], dtype=int)
            inp, out = InputParameter(7), OutputParameter(7)
            inp.current_position = [float(value) for value in state.joint_q.numpy()[q]]
            inp.current_velocity = [float(value) for value in state.joint_qd.numpy()[dof]]
            inp.current_acceleration = [0.0] * 7
            inp.target_position = list(inp.current_position)
            inp.target_velocity = [0.0] * 7
            inp.target_acceleration = [0.0] * 7
            inp.max_velocity = [
                FR3_VELOCITY_FRACTION * float(joints[name]["velocity_limit"]) for name in names
            ]
            inp.max_acceleration = [FR3_MAX_JOINT_ACCELERATION_RAD_S2] * 7
            inp.max_jerk = [FR3_MAX_JOINT_JERK_RAD_S3] * 7
            inp.synchronization = Synchronization.Time
            self.arms[arm] = dict(
                q=q, dof=dof, inp=inp, out=out, generator=Ruckig(7, self.dt)
            )
        self.fingers = np.array(
            [spec["q"] for name, spec in joints.items() if "finger" in name], dtype=int
        )
        self.finger_dofs = np.array(
            [spec["dof"] for name, spec in joints.items() if "finger" in name], dtype=int
        )
        self.finger_slew = np.array(
            [spec["velocity_limit"] * dt for name, spec in joints.items() if "finger" in name]
        )

    def update(self, targets, control):
        positions = control.joint_target_q.numpy()
        velocities = control.joint_target_qd.numpy()
        for group in self.arms.values():
            inp, out = group["inp"], group["out"]
            inp.target_position = [float(value) for value in targets[group["q"]]]
            result = group["generator"].update(inp, out)
            if result not in (Result.Working, Result.Finished):
                raise RuntimeError(f"FR3 motion generator: {result}")
            out.pass_to_input(inp)
            positions[group["q"]] = inp.current_position
            velocities[group["dof"]] = inp.current_velocity
        if len(self.fingers):
            delta = np.clip(
                targets[self.fingers] - positions[self.fingers],
                -self.finger_slew,
                self.finger_slew,
            )
            positions[self.fingers] += delta
            velocities[self.finger_dofs] = delta / self.dt
        control.joint_target_q.assign(positions)
        control.joint_target_qd.assign(velocities)

    def stop_all(self, state, targets, control):
        positions = control.joint_target_q.numpy()
        velocities = control.joint_target_qd.numpy()
        state_q = state.joint_q.numpy()
        for group in self.arms.values():
            q, dof, inp = group["q"], group["dof"], group["inp"]
            hold = [float(value) for value in state_q[q]]
            inp.current_position = hold
            inp.current_velocity = [0.0] * 7
            inp.current_acceleration = [0.0] * 7
            inp.target_position = hold
            inp.target_velocity = [0.0] * 7
            inp.target_acceleration = [0.0] * 7
            targets[q] = state_q[q]
            positions[q] = state_q[q]
            velocities[dof] = 0.0
        control.joint_target_q.assign(positions)
        control.joint_target_qd.assign(velocities)

def filter_rigid_compound_self_contacts(builder, shape_ids):
    """A collapsed rigid assembly has no internal collision interfaces."""
    by_body = {}
    for shape in shape_ids:
        body = builder.shape_body[shape]
        if body >= 0:
            by_body.setdefault(body, []).append(shape)
    existing = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
    added = []
    for shapes in by_body.values():
        for offset, first in enumerate(shapes):
            for second in shapes[offset + 1:]:
                pair = tuple(sorted((first, second)))
                if pair not in existing:
                    builder.shape_collision_filter_pairs.append(pair)
                    existing.add(pair)
                    added.append(pair)
    return added

def audit_rigid_tree(assembly):
    """Reject dynamic dependencies between parallel native FK articulations."""
    b=assembly.builder;owned=set(assembly.rigid_joints)
    incoming={b.joint_child[j]:j for j in owned if b.joint_articulation[j]>=0}
    for j in sorted(owned):
        art=b.joint_articulation[j];parent=b.joint_parent[j]
        if art<0 or parent<0:continue
        parent_joint=incoming.get(parent)
        if parent_joint is None:
            if b.body_mass[parent]>0:
                raise ValueError("Dynamic rigid parent lacks an owned root: "+b.joint_label[j])
            continue
        if b.joint_articulation[parent_joint]!=art or parent_joint>=j:
            raise ValueError("Rigid tree must be parent-first in one articulation: "+
                b.joint_label[parent_joint]+" -> "+b.joint_label[j])
    return dict(checked_rigid_joints=len(owned),parallel_dynamic_dependencies=0)

class Assembly:
    def __init__(self, config: Config, *, robots=True, attachment="native"):
        if attachment not in ("admm", "native", "none"):
            raise ValueError("Unknown diagnostic attachment arrangement")
        self.attachment = attachment
        if importlib.metadata.version("newton") != "1.5.0":
            raise RuntimeError("This environment requires Newton exactly 1.5.0")
        newton.use_coord_layout_targets = True
        self.config = config
        b = self.builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        SolverMuJoCo.register_custom_attributes(b)
        SolverVBD.register_custom_attributes(b)
        b.rigid_gap = config.contact_gap_m
        b.default_shape_cfg = self.shape_config()
        self.rigid_bodies, self.rigid_joints = [], []
        self.cable_bodies, self.cable_joints = [], []
        self.robot_joints, self.objects, self.materials = {}, {}, {}
        self.sources = {}
        self.add_world(robots)
        self.add_cable_fixture()
        filtered=filter_rigid_compound_self_contacts(b,range(b.shape_count))
        self.sources["rigid_compound_collision_filter"]={
            "physical_rule":"shapes in one rigid body have no relative collision",
            "pairs":len(filtered)}
        b.color()
        if config.rigid_force_space:
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE
            b.custom_attributes["mujoco:solref_mode"].values={i:SOLREF_MODE_FORCE_SPACE for i in range(b.shape_count)}
        self.sources["rigid_contact_material_mode"]=(
            "authored force-space ke/kd; native inverse-weight impedance approximation"
            if config.rigid_force_space else "native MJCF_DEFAULT solref; authored ke/kd not automatically mapped")

    def add_world(self,robots):
        b,config=self.builder,self.config
        if robots:
            self.add_robot("left", (0., -.43, config.table_height_m))
            self.add_robot("right", (0., .43, config.table_height_m))
        z = config.table_height_m
        self.box("worktop", (.40, 0, z-.025), (.60, .85, .025), material="laminate")
        for x in (-.12, .92):
            for y in (-.77, .77):
                self.box(f"table_leg_{x}_{y}", (x,y,(z-.05)/2),
                         (.025,.025,(z-.05)/2), material="aluminum")
        b.add_ground_plane(cfg=self.shape_config(), label="floor")
        self.materials[b.shape_count-1] = "concrete"

    def shape_config(self, **kwargs):
        c = self.config
        values = dict(ke=c.contact_stiffness_n_m, kd=c.contact_damping_ns_m,
                      mu=.45, gap=c.contact_gap_m, margin=0., density=1200.,
                      mu_torsional=0., mu_rolling=0.)
        values.update(kwargs)
        return newton.ModelBuilder.ShapeConfig(**values)

    def box(self, name, position, half, *, body=-1, material="steel", rotation=None, cfg=None):
        i = self.builder.add_shape_box(body,
            xform=wp.transform(position, rotation or wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2],
            cfg=cfg or self.shape_config(), label=name)
        self.materials[i] = material
        return i

    def add_robot(self, name, position):
        from robots.backends.newton import add_robot as build, mount
        first=self.builder.joint_count
        handle=build(self.builder,'fr3','control_cable',instance=name,placement=mount(position),
            shape_config=self.shape_config(),parameters={'finger_effort_limit':self.config.gripper_effort_per_finger_n})
        self.sources['fr3_urdf']=handle.source
        self.robot_joints.update(handle.metadata['legacy_addresses']);self.robot_bases[name]=position
        self.rigid_bodies.extend(handle.body_ids);self.rigid_joints.extend(range(first,self.builder.joint_count))
        for i in handle.shape_ids:self.materials[i]='robot_white'

    def add_cable_fixture(self):
        """Product-scale straight cable on the table, free to sag/slip under force."""
        b, s, c = self.builder, self.config.cable, self.config
        points = np.column_stack((np.full(s.segments+1, .42),
                    np.linspace(-s.length_m/2, s.length_m/2, s.segments+1),
                    np.full(s.segments+1, c.table_height_m+.010)))
        cfg = self.shape_config(density=1000., mu=s.friction)
        first_shape = b.shape_count
        bodies, joints = b.add_rod(points.tolist(), radius=s.diameter_m/2,
            cfg=cfg, body_frame_origin="com", label=s.name+"_cable",
            stretch_stiffness=s.axial_rigidity_n/s.segment_m,
            shear_stiffness=s.axial_rigidity_n/s.segment_m,
            bend_stiffness=s.bending_rigidity_nm2/s.segment_m,
            twist_stiffness=s.torsional_rigidity_nm2/s.segment_m,
            stretch_damping=.05, shear_damping=.05,
            bend_damping=2e-5/s.segment_m, twist_damping=1e-5/s.segment_m)
        # add_rod uses overlapping capsule ends. Preserve product line mass rather
        # than counting that geometric overlap as extra jacket material.
        target_mass = s.line_mass_kg_m*s.segment_m
        for body in bodies:
            scale = target_mass/b.body_mass[body]
            b.body_mass[body] = target_mass
            b.body_inertia[body] = b.body_inertia[body]*scale
        self.cable_bodies.extend(bodies)
        self.cable_joints.extend(joints)
        self.objects["sensor_cable"] = bodies
        for i in range(first_shape, b.shape_count):
            self.materials[i] = "cable_black"
        # A simple end fitting is exclusively an early coupling fixture, not a
        # claimed M12 socket or thread model. Native mode owns it with the cable.
        rotation = wp.quat_from_axis_angle(wp.vec3(1,0,0), -math.pi/2)
        position = points[0]-np.array([0,.020,0])
        js = b.joint_count
        connector = b.add_body(xform=wp.transform(position, rotation), label="coupling_fixture_end")
        self.rigid_bodies.append(connector)
        self.rigid_joints.extend(range(js, b.joint_count))
        shape = b.add_shape_cylinder(connector, radius=.007, half_height=.020,
                                    cfg=self.shape_config(density=1600.),
                                    label="fixture_end_fitting")
        self.materials[shape] = "nickel"
        mass = .018
        b.body_inertia[connector] = b.body_inertia[connector]*(mass/b.body_mass[connector])
        b.body_mass[connector] = mass
        self.objects["end_fitting"] = connector
        # Cross-owner FIXED row belongs to ADMM, not either inner solver.
        if self.attachment != "none":
            joint = b.add_joint_fixed(connector, bodies[0],
                parent_xform=wp.transform((0,0,.020), wp.quat_identity()),
                child_xform=wp.transform((0,0,-s.segment_m/2), wp.quat_identity()),
                collision_filter_parent=True, label="molded_strain_relief")
            b.add_articulation([joint], label="coupling_interface")
            if self.attachment == "native":
                self.rigid_bodies.remove(connector)
                self.cable_bodies.append(connector)
                owned = list(range(js, joint))
                self.rigid_joints = [j for j in self.rigid_joints if j not in owned]
                self.cable_joints.extend([*owned, joint])

class Physics:
    def __init__(self, assembly: Assembly, *, passive_only=False, record_coupling_forces=True):
        c = self.config = assembly.config
        self.assembly, self.builder = assembly, assembly.builder
        assembly.sources["rigid_tree_audit"]=audit_rigid_tree(assembly)
        with wp.ScopedDevice(c.device):
            self.model = self.builder.finalize(device=c.device)
            from .inertia_compat import preserve_small_inertias
            assembly.sources["inertia_compatibility"] = preserve_small_inertias(self.builder,self.model)
            from .passive_diagnostic import SingleVBD,cable_solver
            if c.friction_static_tangent_fix:
                from .friction_compat import install
                assembly.sources["friction_compatibility"]=install()
            from .force_observer import RecordedADMM
            coupled_type=RecordedADMM if record_coupling_forces else SolverCoupledADMM
            self.solver = SingleVBD(self.model,c) if passive_only else coupled_type(self.model, entries=[
                SolverCoupled.Entry(name="rigid",
                    solver=lambda view: SolverMuJoCo(view, solver="newton",
                        integrator="implicitfast", iterations=c.mujoco_iterations,
                        cone=c.rigid_contact_cone,impratio=1.,
                        ls_iterations=25, use_mujoco_contacts=False,
                        nconmax=c.contact_capacity, njmax=c.contact_capacity*3),
                    bodies=assembly.rigid_bodies, joints=assembly.rigid_joints),
                SolverCoupled.Entry(name="cables",
                    solver=lambda view: cable_solver(view,c),
                    bodies=assembly.cable_bodies, joints=assembly.cable_joints)],
                coupling=SolverCoupledADMM.Config(iterations=c.coupling_iterations,
                    rho=c.coupling_rho, gamma=.001, baumgarte=.5,
                    rigid_contact_matching=c.coupling_contact_matching,
                    contact_pairs=[SolverCoupledADMM.ContactPair(source="rigid", destination="cables")]))
            self.pipeline = newton.CollisionPipeline(self.model,
                rigid_contact_max=c.contact_capacity, broad_phase="sap",
                reduce_contacts=True)
            self.contacts = self.pipeline.contacts()
            self.solver.prepare_contacts(self.contacts)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
            self.next_state.assign(self.state)
        self.tick, self.fault = 0, None
        self._step_read_cache = None
        from .host_reads import HostReads
        self._host_reads = HostReads()
        self.targets = self.control.joint_target_q.numpy().copy() if self.control.joint_target_q is not None else np.empty(0,np.float32)
        self.fr3_trajectory = FR3Trajectory(
            self.assembly.robot_joints, self.state, self.config.dt
        )
        assembly.sources["fr3_motion_generator"] = {
            "velocity_fraction": FR3_VELOCITY_FRACTION,
            "acceleration_limit_rad_s2": FR3_MAX_JOINT_ACCELERATION_RAD_S2,
            "jerk_limit_rad_s3": FR3_MAX_JOINT_JERK_RAD_S3,
            "source": "FR3 control-parameter table retained in drilling/params.py",
        }
        self.graphs = {}

    def read_array(self,array):
        """Reuse only within the completed-step observation scope, never across ticks.

        Outside that scope always read the live array. Diagnostic pose assignment,
        reset, renderer access and faults cannot inherit a stale cached reading.
        Callers must copy before editing a returned observation.
        """
        cache=self._step_read_cache
        if cache is None:return array.numpy()
        key=id(array)
        if key not in cache:cache.update(self._host_reads.capture([array]))
        return cache[key]

    def prefetch_arrays(self,arrays):
        cache=self._step_read_cache
        if cache is not None:
            missing=[a for a in arrays if a is not None and id(a) not in cache]
            if missing:cache.update(self._host_reads.capture(missing))

    def _integrate(self, state, next_state):
        self.pipeline.collide(state, self.contacts)
        self.solver.step(state, next_state, self.control, self.contacts, self.config.dt)
        newton.eval_ik(self.model, next_state, next_state.joint_q, next_state.joint_qd)

    def enable_graphs(self):
        """Capture each ping-pong direction, without advancing a physics tick."""
        if not self.model.device.is_cuda:
            raise ValueError("CUDA graph execution needs a CUDA device")
        initial = self.state.body_q.numpy().copy()
        for a,b in ((self.state,self.next_state),(self.next_state,self.state)):
            with wp.ScopedDevice(self.model.device):
                with wp.ScopedCapture() as capture:
                    self._integrate(a,b)
            self.graphs[id(a)] = capture.graph
        if not np.array_equal(initial,self.state.body_q.numpy()):
            raise RuntimeError("Graph capture unexpectedly changed the physical pose")

    def step(self, *, fixture_body_forces=None):
        if self.fault:
            raise RuntimeError(self.fault)
        self.state.clear_forces()
        if fixture_body_forces is not None:
            # Only the explicitly labelled diagnostic harness calls this.
            force = np.asarray(fixture_body_forces, np.float32)
            if force.shape != (self.model.body_count, 6) or not np.isfinite(force).all():
                raise ValueError("Invalid diagnostic wrench array")
            self.state.body_f.assign(force)
        if self.control.joint_target_q is not None:
            self.fr3_trajectory.update(self.targets, self.control)
        try:
            if self.graphs:
                wp.capture_launch(self.graphs[id(self.state)])
            else:
                self._integrate(self.state,self.next_state)
            vbd=self.solver.solver("cables")
            completed=self._host_reads.capture([getattr(self.next_state,n) for n in
                ("body_q","body_qd","joint_q","joint_qd")]+
                [vbd.body_body_contact_overflow_max,self.contacts.rigid_contact_count]+
                ([self.control.joint_target_q] if self.control.joint_target_q is not None else []))
            overflow=int(completed[id(vbd.body_body_contact_overflow_max)][0])
            if overflow>self.config.body_contact_capacity:
                raise RuntimeError(f"Per-body contact capacity exceeded: {overflow} > {self.config.body_contact_capacity}")
            total=int(completed[id(self.contacts.rigid_contact_count)][0])
            if total>=self.config.contact_capacity:
                raise RuntimeError(f"Global contact capacity reached: {total}")
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                array=getattr(self.next_state,name)
                if array is None:continue
                if not np.isfinite(completed[id(array)]).all():
                    raise FloatingPointError(f"Non-finite {name}, attempted tick {self.tick+1}")
            # Finite is not sufficient: abort on physically catastrophic divergence.
            rotations=completed[id(self.next_state.body_q)][:,3:]
            norm_error=np.abs(np.linalg.norm(rotations,axis=1)-1.)
            if norm_error.max(initial=0.)>5e-5:
                body=int(np.argmax(norm_error))
                raise FloatingPointError(
                    f"Non-unit rotation for {self.model.body_label[body]} at attempted tick {self.tick+1}: "
                    f"norm error {norm_error[body]}")
            pos = completed[id(self.next_state.body_q)][:,:3]
            if np.max(np.abs(pos)) > 10:
                body = int(np.argmax(np.max(np.abs(pos), axis=1)))
                raise FloatingPointError(
                    f"{self.model.body_label[body]} escaped the 10 m fixture bound "
                    f"at tick {self.tick+1}: {pos[body].tolist()}")
            self.state, self.next_state = self.next_state, self.state
            self._completed_host_state=completed
            self.tick += 1
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

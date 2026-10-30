"""Newton 1.5 rigid dynamics plus the retained constitutive football models.

One public tick is one native physics step. Custom models apply wrenches, never
outgoing ball state. Robot, goal frame, and non-shoe ball contacts remain native.
"""

from dataclasses import asdict
import threading
import numpy as np
from football.football_specs import BallSpec, FieldSpec
from football.aerodynamics import AerodynamicModel, BallKinematics, WakeState
from football.atmosphere import evaluate_atmosphere
from football.foot_ball_coupler import FootBallCoupler
from football.pitch_ball_coupler import PitchBallCoupler
from football.net_ball_coupler import NetBallCoupler
from .task import link_group
from .drills import make_verifier
from .variants import layout as task_layout
from .config import Config
from .rigid_view import RigidView
from .robot import add_robot, Robot
from .actuators import G1Actuators
from .nets import build_nets


class FootballWorld:
    def __init__(self, config=None):
        import newton as n
        import warp as wp
        import mujoco

        if n.__version__ != "1.5.0":
            raise RuntimeError(f"Newton 1.5.0 required, got {n.__version__}")
        n.use_coord_layout_targets = True
        self.config = config or Config()
        self.lock = threading.RLock()
        self.wp, self.n = wp, n
        self.ball, self.field = BallSpec(), FieldSpec()
        wp.set_device(self.config.device)
        b = n.ModelBuilder(up_axis=n.Axis.Z, gravity=-9.80665)
        n.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.mu = 0.62
        b.default_shape_cfg.ke, b.default_shape_cfg.kd = 20000.0, 100.0
        self.pitch_shape = b.add_shape_box(
            -1,
            hx=18.5,
            hy=12.0,
            hz=0.10,
            xform=wp.transform((-15.5, 0, -0.10), wp.quat_identity()),
            label="pitch",
        )
        for y in (-3.72, 3.72):
            b.add_shape_cylinder(
                -1,
                radius=0.06,
                half_height=1.22,
                xform=wp.transform((0, y, 1.22), wp.quat_identity()),
                label=f"goal_post_{y}",
            )
        b.add_shape_cylinder(
            -1,
            radius=0.06,
            half_height=3.78,
            xform=wp.transform(
                (0, 0, 2.50), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)
            ),
            label="crossbar",
        )
        if self.config.net_layout == "enclosure":
            # Fixed training goal: anchored tubular rear stays/ground rails.
            # Net boundary nodes represent ties to these supports, not floating
            # force sources. This is not a freely overturning portable goal.
            def rail(label, a, c):
                a, c = np.asarray(a), np.asarray(c)
                delta = c - a
                length = float(np.linalg.norm(delta))
                direction = delta / length
                axis = np.cross((0.0, 0.0, 1.0), direction)
                rotation = (
                    wp.quat_identity()
                    if np.linalg.norm(axis) < 1e-10
                    else wp.quat_from_axis_angle(
                        wp.vec3(*(axis / np.linalg.norm(axis))),
                        float(np.arccos(direction[2])),
                    )
                )
                b.add_shape_cylinder(
                    -1,
                    radius=0.025,
                    half_height=length / 2,
                    xform=wp.transform(tuple((a + c) / 2), rotation),
                    label=label,
                )

            for side in (-3.66, 3.66):
                rail(f"rear_stay_{side}", (2.1, side, 0.03), (0.82, side, 2.44))
                rail(f"ground_rail_{side}", (0.0, side, 0.03), (2.1, side, 0.03))
                rail(f"roof_rail_{side}", (0.0, side, 2.44), (0.82, side, 2.44))
            rail("rear_ground_rail", (2.1, -3.66, 0.03), (2.1, 3.66, 0.03))
            rail("rear_roof_rail", (0.82, -3.66, 2.44), (0.82, 3.66, 2.44))
        self.ball_id = b.add_body(
            xform=wp.transform(self.config.ball_position, wp.quat_identity()),
            label="ball",
        )
        self.ball_shape = b.add_shape_sphere(
            self.ball_id, radius=self.ball.radius_m, label="ball/shell"
        )
        b.body_mass[self.ball_id] = self.ball.mass_kg
        b.body_inertia[self.ball_id] = wp.mat33(
            np.eye(3) * (2 / 3 * self.ball.mass_kg * self.ball.radius_m**2)
        )
        b.body_qd[self.ball_id] = wp.spatial_vector(
            *self.config.ball_velocity, *self.config.ball_omega
        )
        # add_body creates a free joint. eval_fk reads joint_qd, not body_qd;
        # initialize both representations before the first engine step.
        ball_joint = b.joint_label.index("ball_free_joint")
        start = b.joint_qd_start[ball_joint]
        b.joint_qd[start : start + 6] = [
            *self.config.ball_velocity,
            *self.config.ball_omega,
        ]
        b.shape_collision_filter_pairs.append((self.pitch_shape, self.ball_shape))
        self.joints, self.actuators = {}, None
        if self.config.robot:
            self.joints, velocity_limits = add_robot(b, wp, self.config)
            self.actuators = G1Actuators(b, self.joints, velocity_limits)
        self.body_ids = {name.split("/")[-1]: i for i, name in enumerate(b.body_label)}
        if self.config.robot:
            foot_id = self.body_ids["right_ankle_roll_link"]
            for i, body in enumerate(b.shape_body):
                if body == foot_id:
                    b.shape_collision_filter_pairs.append((self.ball_shape, i))
        # Task props (variants.py): static boxes on the world body with the frame's default shape settings, added
        # after every existing shape so existing shape indices are unchanged; goal has none.
        self.layout = task_layout(self.config.task, self.config.seed)
        self.prop_materials = {}
        for prop in self.layout["props"]:
            label = "prop/" + prop["name"]
            hx, hy, hz = (float(v) for v in prop["half_extents_m"])
            b.add_shape_box(
                -1,
                hx=hx,
                hy=hy,
                hz=hz,
                xform=wp.transform(
                    tuple(float(v) for v in prop["position_m"]),
                    wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(prop["yaw_rad"])),
                ),
                label=label,
            )
            self.prop_materials[label] = prop["material"]
        self.builder = b
        self.model = b.finalize(device=self.config.device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        n.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.control = self.model.control()
        self.targets = self.control.joint_target_q.numpy().copy()
        self.solver = n.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            integrator="implicit",
            solver="newton",
            cone="elliptic",
            iterations=self.config.iterations,
            ls_iterations=20,
            njmax=6000,
            nconmax=1500,
            use_mujoco_contacts=True,
            update_data_interval=1,
            enable_multiccd=True,
        )
        from .collision_ownership import enforce_ball_pairs

        self.collision_ownership = enforce_ball_pairs(self)
        if self.actuators:
            self.actuators.bind(self)
        # Native geoms of robot links, for the task's kick and handball rules (task.py rule 3).
        self.robot_geom_links = {}
        if self.config.robot:
            shapes = self.solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
            for geom, shape in enumerate(shapes):
                body = b.shape_body[int(shape)]
                if body >= 0 and body != self.ball_id:
                    name = b.body_label[body].split("/")[-1]
                    link_group(name)
                    self.robot_geom_links[geom] = name
        mapping = self.solver.mjc_body_to_newton.numpy().reshape(-1)
        self.mj_body_ids = {int(i): j for j, i in enumerate(mapping) if i >= 0}
        self.mass_query_data = mujoco.MjData(self.solver.mj_model)
        self.com = self.model.body_com.numpy()
        self.wrenches = np.zeros((self.model.body_count, 6), np.float32)
        self.views = {name: RigidView(self, i) for name, i in self.body_ids.items()}
        self.tick, self.fault, self.generation = 0, None, 0
        self.trace_callback = None
        self.refresh()
        self.robot = Robot(self) if self.config.robot else None
        self.foot = None
        if self.robot:
            foot_id = self.body_ids["right_ankle_roll_link"]
            self.foot = FootBallCoupler(
                ball_view=self.views["ball"],
                foot_view=self.views["right_ankle_roll_link"],
                foot_center_of_mass_local_m=tuple(float(x) for x in self.com[foot_id]),
                contact_substep_hz=self.config.contact_hz,
                foot_inverse_mass_provider=self.robot.inverse_mass,
            )
        self.pitch = PitchBallCoupler(
            ball_view=self.views["ball"], contact_substep_hz=self.config.contact_hz
        )
        self.nets, self.net_initialization = build_nets(self.config.net_layout)
        self.net_couplers = {
            name: NetBallCoupler(
                ball_view=self.views["ball"],
                net=net,
                contact_substep_hz=self.config.net_hz,
                recurrent_contact=self.config.net_layout == "enclosure",
            )
            for name, net in self.nets.items()
        }
        self.aero, self.air, self.wake = (
            AerodynamicModel(),
            evaluate_atmosphere(),
            WakeState(),
        )
        # The task's verifier: goal's PitchAwareGoalDetector(require_kick=robot) for goal and free-kick (drills.py).
        self.goal = make_verifier(self.config, self.layout)
        self.verifier_uses_robot_pose = bool(getattr(self.goal, "uses_robot_pose", False))
        if self.verifier_uses_robot_pose:
            self.goal.robot = self.robot_pose()
        self.goal.update(
            self.time,
            tuple(map(float, self.poses[self.ball_id, :3])),
            tuple(map(float, self.velocities[self.ball_id, :3])),
        )
        self.last = {}

    @property
    def time(self):
        return self.tick * self.config.dt

    def refresh(self):
        self.poses = self.state_0.body_q.numpy()
        self.velocities = self.state_0.body_qd.numpy()
        self.q, self.qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()

    def set_joint_targets(self, targets):
        checked = {}
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f"Unknown joint {name}")
            a = self.joints[name]
            if not np.isfinite(value) or not a["lower"] <= value <= a["upper"]:
                raise ValueError(
                    f"{name} target {value} outside [{a['lower']}, {a['upper']}]"
                )
            checked[a["q"]] = value
        for index, value in checked.items():
            self.targets[index] = value
        self.control.joint_target_q.assign(self.targets)

    def step(self, count=1, *, before_step=None, after_step=None, fixture_wrench=None):
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        with self.lock:
            if self.fault:
                raise RuntimeError(f"Physics stopped: {self.fault}")
            for _ in range(count):
                try:
                    if before_step:
                        before_step(self)
                    self._step(fixture_wrench=fixture_wrench)
                    if self.trace_callback:
                        self.trace_callback(self)
                    if after_step:
                        after_step(self)
                except Exception as exc:
                    self.fault = f"{type(exc).__name__}: {exc} at tick {self.tick}"
                    raise

    def _step(self, *, fixture_wrench=None):
        self.wrenches.fill(0)
        self.last = {}
        if self.foot:
            self.last["foot"] = self.foot.step(self.time, self.config.dt)
        view = self.views["ball"]
        p, quat = view.get_world_pose()
        kin = BallKinematics(
            tuple(p),
            tuple(view.get_linear_velocity()),
            tuple(view.get_angular_velocity()),
            tuple(quat),
        )
        aero = self.aero.evaluate(kin, self.air, self.wake, self.config.dt)
        self.wake = aero.wake
        view.apply_forces_and_torques_at_pos(
            forces=aero.applied_force_without_engine_gravity_n,
            torques=aero.aerodynamic_torque_n_m,
            is_global=True,
        )
        self.last["aero"] = asdict(aero)
        for name, coupler in self.net_couplers.items():
            self.last["net_" + name] = coupler.step(self.time, self.config.dt)
        external = self.wrenches[self.ball_id].copy()
        self.last["pitch"] = self.pitch.step(
            self.time,
            self.config.dt,
            external_force_on_ball_world_n=external[:3],
            external_torque_on_ball_world_n_m=external[3:],
        )
        if fixture_wrench is not None:
            applied = np.asarray(fixture_wrench(self), dtype=np.float32)
            if applied.shape != self.wrenches.shape or not np.isfinite(applied).all():
                raise ValueError("Fixture must return finite per-body COM wrenches")
            self.wrenches += applied
            self.last["labelled_fixture_wrenches"] = applied.tolist()
        if self.actuators:
            self.actuators.apply(self)
        self.state_0.clear_forces()
        self.state_0.body_f.assign(self.wrenches)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.config.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.tick += 1
        self.refresh()
        for name, array in (
            ("body_q", self.poses),
            ("body_qd", self.velocities),
            ("joint_q", self.q),
            ("joint_qd", self.qd),
        ):
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Non-finite {name}")
        warnings = self.solver.mj_data.warning.number
        if np.any(warnings):
            raise RuntimeError(f"Native MuJoCo warnings: {warnings.tolist()}")
        if self.verifier_uses_robot_pose:
            self.goal.robot = self.robot_pose()
        self.goal.update_supported(
            self.time,
            tuple(map(float, self.poses[self.ball_id, :3])),
            tuple(map(float, self.velocities[self.ball_id, :3])),
            normal_force_n=self.last["pitch"]["normal_force_n"],
            compression_m=self.last["pitch"]["compression_m"],
            robot_touches=self.robot_ball_touches(),
        )

    def robot_pose(self):
        """Pelvis position, the z component of the pelvis up axis and the midpoint of the two ankle-roll links, end
        of tick, world frame: the robot state the trap verifier judges (drills.Trap). None without a robot."""
        if not self.config.robot:
            return None
        pelvis = self.poses[self.body_ids["pelvis"]]
        qx, qy = float(pelvis[3]), float(pelvis[4])  # body_q quaternion is XYZW
        left = self.poses[self.body_ids["left_ankle_roll_link"], :3]
        right = self.poses[self.body_ids["right_ankle_roll_link"], :3]
        return {
            "pelvis_m": tuple(float(x) for x in pelvis[:3]),
            "pelvis_up_z": 1.0 - 2.0 * (qx * qx + qy * qy),
            "feet_midpoint_m": tuple(0.5 * float(a + b) for a, b in zip(left, right)),
        }

    def robot_ball_touches(self):
        """Robot links that pushed on the ball during the last tick: the right shoe's constitutive contact, and
        active native contacts (efc_address >= 0, as trace.py) whose normal force is positive."""
        if not self.config.robot:
            return ()
        import mujoco

        links = set()
        if self.last.get("foot", {}).get("normal_force_n", 0.0) > 0.0:
            links.add("right_ankle_roll_link")
        m, d = self.solver.mj_model, self.solver.mj_data
        if d.ncon:
            ball = self.collision_ownership["ball_geom_id"]
            g1, g2 = d.contact.geom1, d.contact.geom2
            force = np.zeros(6)
            for i in np.flatnonzero((d.contact.efc_address >= 0) & ((g1 == ball) | (g2 == ball))):
                link = self.robot_geom_links.get(int(g2[i] if g1[i] == ball else g1[i]))
                if link is not None and link not in links:
                    mujoco.mj_contactForce(m, d, int(i), force)
                    if force[0] > 0.0:
                        links.add(link)
        return tuple(sorted(links))

    def state(self):
        return {
            "tick": self.tick,
            "sim_time_s": self.time,
            "fault": self.fault,
            "generation": self.generation,
            "engine": "newton-1.5.0",
            "device": self.config.device,
            "collision_ownership": self.collision_ownership,
            "bodies": {
                name: {
                    "position_m": self.poses[i, :3].tolist(),
                    "orientation_xyzw": self.poses[i, 3:].tolist(),
                    "linear_velocity_m_s": self.velocities[i, :3].tolist(),
                    "angular_velocity_rad_s": self.velocities[i, 3:].tolist(),
                }
                for name, i in self.body_ids.items()
            },
            "joints": {
                name: {
                    "position_rad": float(self.q[a["q"]]),
                    "velocity_rad_s": float(self.qd[a["dof"]]),
                    "target_rad": float(self.targets[a["q"]]),
                }
                for name, a in self.joints.items()
            },
            "goal_crossing": None
            if self.goal.crossing is None
            else asdict(self.goal.crossing),
        }

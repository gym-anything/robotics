"""Read-only G1 kinematic previews followed exclusively by physical joint servos.

No object pose writes. Preview checks a discretized joint path, not a guarantee
against dynamic collisions; requested guards also run during execution.
"""

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def exact_ticks(duration_s, dt):
    value = duration_s / dt
    if not math.isfinite(value) or value <= 0 or abs(value - round(value)) > 1e-8:
        raise ValueError("Duration must be an exact positive multiple of physics_dt; no rounding")
    return int(round(value))


class HandIK:
    """Seven joint chain; TCP is 80 mm along the wrist-yaw/palm +X axis."""

    tcp_local_m = np.array([0.08, 0.0, 0.0])

    def __init__(self, env):
        import mujoco

        self.env, self.mj = env, mujoco
        self.model = env.solver.mj_model
        self.data = mujoco.MjData(self.model)
        self.body_mapping = env.solver.mjc_body_to_newton.numpy()[0]

    def sync(self):
        # Mapping is provided by pinned Newton; this only writes PRIVATE preview data.
        self.env.solver._update_mjc_data(self.data, self.env.model, self.env.state_0)
        self.mj.mj_forward(self.model, self.data)

    def chain(self, arm):
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        names = [
            name
            for name in self.env.robot_joints
            if name.startswith(arm + "_")
            and any(token in name for token in ("shoulder_", "elbow_", "wrist_"))
        ]
        if len(names) != 7:
            raise RuntimeError("Expected the pinned seven-joint G1 arm")
        labels = {
            label.split("/")[-1]: label.replace("/", "_") for label in self.env.model.joint_label
        }
        ids = [self.model.joint(labels[name]).id for name in names]
        body = next(
            i for i in range(self.model.nbody) if self.body_name(i) == arm + "_wrist_yaw_link"
        )
        return names, np.asarray(self.model.jnt_qposadr[ids]), body

    def body_name(self, i):
        index = int(self.body_mapping[i])
        return self.env.model.body_label[index].split("/")[-1] if index >= 0 else "world"

    def pose(self, arm, *, sync=True):
        if sync:
            self.sync()
        _, _, body = self.chain(arm)
        rotation = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
        position = self.data.xpos[body] + rotation.apply(self.tcp_local_m)
        quat = rotation.as_quat()
        return {
            "position_m": position.tolist(),
            "orientation_wxyz": quat[[3, 0, 1, 2]].tolist(),
            "tcp_local_m": self.tcp_local_m.tolist(),
            "frame": arm + "_wrist_yaw_link",
        }

    def contacts(self, data, arm, safety):
        result = []
        for contact in data.contact[: data.ncon]:
            if contact.dist >= 0:
                continue
            names = [self.body_name(int(self.model.geom_bodyid[g])) for g in contact.geom]
            is_arm = [
                any(t in name for t in ("shoulder_", "elbow_", "wrist_", "hand_")) for name in names
            ]
            moving = any(a and n.startswith(arm + "_") for a, n in zip(is_arm, names))
            both_opposite = (
                all(is_arm)
                and any(n.startswith("left_") for n in names)
                and any(n.startswith("right_") for n in names)
            )
            robot = all(any(t in n for t in ("link", "pelvis")) for n in names)
            if moving and (
                safety.stop_on_contact
                or safety.stop_on_arm_arm_contact
                and both_opposite
                or safety.stop_on_robot_contact
                and robot
            ):
                result.append({"bodies": names, "distance_m": float(contact.dist)})
        return result

    def plan(self, arm, position, orientation_wxyz=None, *, safety=None):
        from gym_anything_robotics import MotionSafety

        safety = safety or MotionSafety()
        if safety.stop_on_grip_loss is not None:
            raise ValueError("G1 IK does not implement grip-loss guards")
        position = np.asarray(position, float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Target must contain three finite meter coordinates")
        self.sync()
        names, addresses, body = self.chain(arm)
        start = self.data.qpos[addresses].copy()
        base_qpos = self.data.qpos.copy()
        target = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
        if orientation_wxyz is not None:
            quat = np.asarray(orientation_wxyz, float)
            if (
                quat.shape != (4,)
                or not np.isfinite(quat).all()
                or abs(np.linalg.norm(quat) - 1) > 1e-6
            ):
                raise ValueError("Orientation must be a unit wxyz quaternion")
            target = Rotation.from_quat(quat[[1, 2, 3, 0]])

        def errors(q):
            self.data.qpos[addresses] = q
            self.mj.mj_kinematics(self.model, self.data)
            rotation = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3))
            point = self.data.xpos[body] + rotation.apply(self.tcp_local_m)
            return np.r_[point - position, 0.15 * (target * rotation.inv()).as_rotvec()]

        bounds = (
            [self.env.robot_joints[n]["lower"] for n in names],
            [self.env.robot_joints[n]["upper"] for n in names],
        )
        # A physical joint can penetrate a soft stop. Start optimization in its legal domain;
        # this is an internal candidate, never a clamp of the requested target.
        initial = np.clip(start, bounds[0], bounds[1])
        solve = least_squares(
            errors, initial, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=150
        )
        residual = errors(solve.x)
        pos_error, angle_error = (
            float(np.linalg.norm(residual[:3])),
            float(np.linalg.norm(residual[3:]) / 0.15),
        )
        count = max(1, int(np.ceil(np.max(np.abs(solve.x - start)) / 0.03)))
        collision = []
        for t in np.linspace(0, 1, count + 1):
            self.data.qpos[:] = base_qpos
            self.data.qpos[addresses] = start + t * (solve.x - start)
            self.mj.mj_forward(self.model, self.data)
            collision = self.contacts(self.data, arm, safety)
            if collision:
                break
        return {
            "reachable": pos_error <= 0.002 and angle_error <= 0.03 and not collision,
            "position_error_m": pos_error,
            "orientation_error_rad": angle_error,
            "targets_rad": dict(zip(names, solve.x.tolist())),
            "target_m": position.tolist(),
            "target_orientation_wxyz": target.as_quat()[[3, 0, 1, 2]].tolist(),
            "start_rad": dict(zip(names, start.tolist())),
            "path_samples": count + 1,
            "collisions": collision,
            "preview": "Kinematics only; actual dynamics guarded separately",
        }


def execute_cartesian(env, action):
    from gym_anything_robotics import CartesianDelta

    if action.gripper_half_target_m is not None:
        raise ValueError(
            "Dex3 has articulated fingers, not a parallel half-width gripper; command named finger joints"
        )
    ticks = exact_ticks(action.duration_s, env.config.physics_dt)
    if env.tick + ticks > env.config.episode_ticks:
        raise ValueError("Action exceeds episode horizon")
    ik = HandIK(env)
    start_pose = ik.pose(action.arm)
    target = (
        (np.asarray(start_pose["position_m"]) + action.delta_m).tolist()
        if isinstance(action, CartesianDelta)
        else action.position_m
    )
    orientation = None if isinstance(action, CartesianDelta) else action.orientation_wxyz
    plan = ik.plan(action.arm, target, orientation, safety=action.safety)
    if not plan["reachable"]:
        return {"ticks_executed": 0, "stop_reason": "ik_or_preview_collision", "plan": plan}
    start_tick = env.tick
    start_q = plan["start_rad"]
    stop = None
    for i in range(1, ticks + 1):
        # Quintic ramp bounds endpoint velocity/acceleration, without changing duration.
        t = i / ticks
        blend = 10 * t**3 - 15 * t**4 + 6 * t**5
        env.act(
            {n: start_q[n] + blend * (v - start_q[n]) for n, v in plan["targets_rad"].items()},
            ticks=1,
        )
        ik.sync()
        if ik.contacts(ik.data, action.arm, action.safety):
            stop = "requested_contact_guard"
            # Hold measured joint positions; this is not a state teleport.
            control = env.control.joint_target_q.numpy()
            measured = env.state_0.joint_q.numpy()
            for name in plan["targets_rad"]:
                j = env.robot_joints[name]["q"]
                control[j] = measured[j]
            env.control.joint_target_q.assign(control)
            break
    final = ik.pose(action.arm)
    return {
        "ticks_executed": env.tick - start_tick,
        "stop_reason": stop,
        "plan": plan,
        "final_tcp": final,
        "actual_position_error_m": float(np.linalg.norm(np.asarray(final["position_m"]) - target)),
    }

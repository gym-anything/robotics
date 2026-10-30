"""Powered pallet roller conveyor with photo-eyes (warehouse production edge).

Rollers: 2.5 in steel tubes on 0.10 m centres across the conveyor, each a body on a revolute joint, chain-driven by a
gear motor: a velocity servo on every roller (torque capped at CONVEYOR roller_torque) toward the surface speed. The
pallet rides on its bottom deck boards by friction (native contact). Control as on real infeed conveyors: the entry
photo-eye starts the drive when a load is set on the rollers, the end photo-eye stops it when the load reaches the end
stop. Photo-eyes are MuJoCo rays across the conveyor just above the roller tops; they see any load geometry.
In tasks that do not use the conveyor, the rollers are static colliders (same geometry, no drive).
"""

import math

import numpy as np

from . import params as P

ROLLER_R = P.CONVEYOR["roller_d"] / 2
ROLLER_PITCH = P.CONVEYOR["pitch"]
ROLLER_MASS = P.CONVEYOR["roller_mass"]
SPEED = P.CONVEYOR["speed"]                   # m/s surface speed (pallet conveyors 30-60 fpm, estimate)
ROLLER_TORQUE = P.CONVEYOR["roller_torque"]           # N m per roller from the chain drive (estimate)


def add_conveyor(scene, conv, powered):
    wp, b = scene.wp, scene.b
    x0, x1 = conv["x"]
    y0, y1 = conv["y"]
    top = conv["top"]
    n = int(round((x1 - x0) / ROLLER_PITCH))
    yc, hw = (y0 + y1) / 2, (y1 - y0) / 2
    q_y = (math.sin(-math.pi / 4), 0.0, 0.0, math.cos(-math.pi / 4))          # cylinder axis z -> y
    cfg = scene.cfg(mu=P.MU_CONVEYOR_ROLLER)
    info = dict(rollers=[], joints=[], powered=powered, conv=conv, n=n)
    static_body = None if powered else scene.static_body("conveyor_rollers", (sum(conv["x"]) / 2, yc, top - ROLLER_R))
    origin = np.array([sum(conv["x"]) / 2, yc, top - ROLLER_R])
    for i in range(n):
        x = x0 + (i + 0.5) * ROLLER_PITCH
        c = np.array([x, yc, top - ROLLER_R])
        if powered:
            # the roller spins about y (its joint axis): spin inertia m r^2 x 0.8 (a thick-walled tube) about y, a rod's
            # m (3 r^2 + L^2) / 12 about x and z. They had been swapped (spin inertia 0.78 kg m^2, ~150 times too large,
            # and a tensor that failed the triangle inequality: Newton "corrected" all 62 rollers at build)
            I_spin = ROLLER_MASS * ROLLER_R ** 2 * P.CONVEYOR_ROLLER_JOINT["spin_factor"]
            I_t = ROLLER_MASS * (3 * ROLLER_R ** 2 + (2 * hw) ** 2) / 12
            body = b.add_link(xform=wp.transform(tuple(c), wp.quat_identity()), mass=ROLLER_MASS,
                              inertia=wp.mat33(I_t, 0, 0, 0, I_spin, 0, 0, 0, I_t), lock_inertia=True,
                              label=f"conveyor/roller{i:02d}")
            j = b.add_joint_revolute(-1, body, parent_xform=wp.transform(tuple(c), wp.quat_identity()),
                                     child_xform=wp.transform_identity(), axis=(0.0, 1.0, 0.0), label=f"conveyor/roller{i:02d}",
                                     armature=P.CONVEYOR_ROLLER_JOINT["armature"],
                                     friction=P.CONVEYOR_ROLLER_JOINT["bearing_friction"])
            b.add_articulation([j], label=f"conveyor/roller{i:02d}")
            s = b.add_shape_cylinder(body, xform=wp.transform((0.0, 0.0, 0.0), wp.quat(*q_y)), radius=ROLLER_R,
                                     half_height=hw - 0.01, cfg=cfg, label=f"conveyor:roller{i:02d}")
            info["rollers"].append(body); info["joints"].append(j)
        else:
            s = b.add_shape_cylinder(static_body, xform=wp.transform(tuple(c - origin), wp.quat(*q_y)), radius=ROLLER_R,
                                     half_height=hw - 0.01, cfg=cfg, label=f"conveyor:roller{i:02d}")
        scene.shape_group[s] = "obj"
        scene.materials[s] = "steel_roller"
        scene.statics.setdefault("conveyor_rollers", []).append(s)
    scene.machines["conveyor"] = info
    return info


class Conveyor:
    def __init__(self, world):
        self.w = world
        self.info = world.scene.machines["conveyor"]
        qds = world.model.joint_qd_start.numpy()
        self.dofs = np.array([int(qds[j]) for j in self.info["joints"]], int)
        self.running = False
        self.eye_entry = self.eye_end = False
        self.end_events = []
        self.load_geoms = None
        self.energy = 0.0

    def _eye(self, x):
        import mujoco
        m, d = self.w.solver.mj_model, self.w.solver.mj_data
        conv = self.info["conv"]
        p = np.array([x, conv["y"][0] - 0.12, conv["top"] + 0.05])
        vec = np.array([0.0, 1.0, 0.0])
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, p, vec, None, 1, -1, geomid)
        if dist < 0 or dist > (conv["y"][1] - conv["y"][0]) + 0.24:
            return False
        lab = self.w.mj_geom_label(int(geomid[0]))
        return not lab.startswith(("conv_", "conveyor"))

    def step(self, world, jf):
        conv = self.info["conv"]
        if world.tick % 10 == 0:
            self.eye_entry = self._eye(conv["x"][0] + 0.35)
            end = self._eye(conv["eye_x"])
            if end and not self.eye_end:
                self.end_events.append(world.sim_time)
            self.eye_end = end
            if self.eye_entry and not self.eye_end:
                self.running = True
            if self.eye_end:
                self.running = False
        if not self.info["powered"]:
            return
        qd = world.state.joint_qd.numpy()
        w_target = (SPEED / ROLLER_R) if self.running else 0.0
        # rotation about +y moves a roller's top toward +x (omega x r = (w r, 0, 0) for r = (0, 0, R))
        w = qd[self.dofs]
        tau = np.clip(P.CONVEYOR_ROLLER_JOINT["speed_gain"] * (w_target - w), -ROLLER_TORQUE, ROLLER_TORQUE)
        jf[self.dofs] += tau
        self.energy += float(np.sum(np.abs(tau * w))) * world.config.dt

    def checkpoint(self):
        return dict(running=self.running, entry=self.eye_entry, end=self.eye_end, events=list(self.end_events))

    def restore(self, c):
        self.running, self.eye_entry, self.eye_end, self.end_events = c["running"], c["entry"], c["end"], list(c["events"])

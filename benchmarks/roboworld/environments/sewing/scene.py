"""Rigid scene of the sewing cell (Newton ModelBuilder -> MuJoCo on the CPU): floor, the K-leg table with the
machine bed flush in its top, the machine head (pillar, arm, head, needle and presser bars, handwheel) as static
colliders, the robot pedestal, and the robots from the shared catalog with the Franka Hand's URDF finger boxes.

The cloth, the needle, the feed dog and the presser foot live in the cloth solver (sewing/cloth.py, sewing/machine.py).
Here the foot's shank and the needle bar are static keep-out colliders for the arms only (they never touch cloth in
MuJoCo). Visual dressing is in render.py and has no collision.
"""

import math

import numpy as np

from . import layout as LY

MU_TABLE = 0.4            # rigid-rigid (fingers on laminate/cast iron) [estimate]
# Franka Hand finger collision boxes (left-finger frame) from the manufacturer's fr3_franka_hand.urdf (as filleting,
# welding)
FRANKA_HAND_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                            ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                            ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                            ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))
# hand housing as the cloth sees it (hand frame: z along the approach, y across the fingers) [estimate from the Franka
# Hand drawing: 204 mm across, 60 mm thick, 63 mm tall to the finger base]
HAND_HOUSING = dict(center=(0.0, 0.0, 0.035), half=(0.030, 0.102, 0.032))


class Scene:
    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))   # z up; a scalar is deprecated since Newton 1.4
        self.b.rigid_gap = 0.
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.shapes = {}, {}
        self.finger_boxes = {}           # arm -> list of (body, xyz, quat_xyzw, half) in the finger body frame
        self.hand_bodies = {}            # arm -> hand body id
        self.hand_housing = {}           # arm -> (centre, half) of the hand body as the cloth sees it (hand frame)
        self.cameras = {}                # robot-mounted cameras (the G1's head camera)
        self.g1 = config.robot == "g1"
        self.room()
        self.machine()
        self.robots()
        # no contact detection gap on any shape: the robots' sub-builders keep Newton's default rigid_gap (0.1 m), which
        # under MuJoCo 3.9's semantics widens detection without generating force; on the G1 it carried ~1,000 pairs of
        # its own links up to 200 mm apart through every MuJoCo step
        self.b.shape_gap = [0.0]*len(self.b.shape_gap)

    def cfg(self, **kw):
        v = dict(mu=MU_TABLE, ke=2e4, kd=100., density=0., gap=0., margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def box(self, name, lo, hi, material, body=-1, rot=None, **kw):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, h = 0.5*(lo + hi), 0.5*(hi - lo)
        i = self.b.add_shape_box(body, xform=self.wp.transform(tuple(map(float, c)), rot or self.wp.quat_identity()),
                                 hx=float(h[0]), hy=float(h[1]), hz=float(h[2]), cfg=self.cfg(**kw), label=name)
        self.materials[i] = material
        self.shapes[name] = i
        return i

    def room(self):
        self.b.add_ground_plane(height=-LY.TABLE_H, cfg=self.cfg(mu=.6), label="floor")
        T = LY.TABLE
        (x0, x1), (y0, y1), t = T["x"], T["y"], T["thickness"]
        self.box("table_top", (x0, y0, -t), (x1, y1, 0.0), "table_laminate")
        for x in (x0 + .05, x1 - .05):
            for y in (y0 + .05, y1 - .05):
                self.box(f"table_leg_{'r' if x > 0 else 'l'}{'b' if y > 0 else 'f'}", (x - .02, y - .02, -LY.TABLE_H),
                         (x + .02, y + .02, -t), "table_steel")
        if not self.g1:              # the arms' pedestal in the operator's seat (the humanoid stands there instead)
            P = LY.PEDESTAL
            self.box("pedestal", (P["x"][0], P["y"][0], P["z"][0]), (P["x"][1], P["y"][1], P["z"][1] - 1e-3),
                     "pedestal_steel")

    def machine(self):
        """The head as static boxes at the class's dimensions (layout.py): what the arms can hit."""
        for name, d in (("pillar", LY.PILLAR), ("arm", LY.ARM), ("head", LY.HEAD), ("presser_bar", LY.PRESSER_BAR)):
            self.box(f"machine_{name}", (d["x"][0], d["y"][0], d["z"][0]), (d["x"][1], d["y"][1], d["z"][1]),
                     "machine_cast")
        r = LY.NEEDLE_BAR["radius"]
        self.box("machine_needle_bar", (-r, -r, LY.NEEDLE_BAR["z_bottom_up"]), (r, r, LY.HEAD["z"][0] + 1e-3),
                 "machine_steel")
        k = LY.FOOT_KEEP_OUT
        self.box("machine_foot_shank", (k["x"][0], k["y"][0], k["z"][0]), (k["x"][1], k["y"][1], k["z"][1]),
                 "machine_steel")
        # the handwheel's envelope as a box (a cylinder collider has no MuJoCo multi-contact with box/mesh pairs)
        h = LY.HANDWHEEL
        cx, cy, cz = h["center"]
        r, wd = h["radius"], h["width"]
        self.box("machine_handwheel", (cx - wd/2, cy - r, cz - r), (cx + wd/2, cy + r, cz + r), "machine_handwheel")

    def robots(self):
        if self.g1:
            return self._g1_robot()
        from robots.catalog import for_environment, model
        from robots.backends.newton import mount
        import robots
        setup = for_environment("sewing", self.c.robot)
        self.robot_setup = setup
        data = setup.to_dict()
        self.arm_hw, mounts = {}, {}
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            base = LY.ROBOT_BASES[name]
            mounts[inst["mount"]] = mount(tuple(base["pos"]), math.radians(base["yaw_deg"]))
            self.arm_hw[name] = dict(model=inst["model"].split("@")[0], profile=inst["controller_profile"],
                                     mount=(tuple(base["pos"][:2]), math.radians(base["yaw_deg"])),
                                     base_z=base["pos"][2], arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            self.arm_hw[owner]["hand"] = att["model"]
        first = self.b.body_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        self.joints = {}
        for name, handle in self.rig_pending.instances.items():
            self.joints.update(handle.joints)
            for i in handle.shape_ids:
                self.materials[i] = "robot_" + self.arm_hw[name]["model"]
            if self.arm_hw[name].get("hand"):
                self._fingers(name, handle)

    def _g1_robot(self):
        """Unitree G1 + two Dex3-1 hands standing on the floor in front of the table (g1.py; rig
        robots/rigs/sewing-g1.json). Floating base: no weld, no pelvis support. Its links collide in MuJoCo with their
        URDF meshes' convex hulls; the cloth sees the Dex3 finger links and the palm as boxes (g1.cloth_pads)."""
        from robots.catalog import resolve
        from robots.backends.newton import mount
        import robots
        from . import g1 as G1
        setup = resolve(G1.SETUP)
        self.robot_setup = setup
        pos, yaw = G1.mount_pose()
        first = self.b.body_count
        self.rig_pending = robots.add_to(self.b, setup, mounts={G1.MOUNT: mount(pos, yaw)})
        self.robot_bodies = list(range(first, self.b.body_count))
        handle = self.rig_pending.instances[G1.INSTANCE]
        self.joints = dict(handle.joints)
        for i in handle.shape_ids:
            self.materials[i] = "robot_g1"
        self.arm_hw = {G1.INSTANCE: dict(model="g1", profile="sewing", mount=(pos[:2], yaw), base_z=pos[2],
                                         arm_joints=[], gripper_joints=[])}
        fingers, palms = G1.cloth_pads(self.b, handle, self.newton)
        for side in ("left", "right"):
            self.finger_boxes[side] = fingers[side]
            body, c, h = palms[side]
            self.hand_bodies[side] = body
            self.hand_housing[side] = (c, h)
        G1.head_camera(self, handle)

    def _fingers(self, arm, handle):
        """The URDF finger boxes on each Franka Hand finger (MuJoCo colliders, as filleting and welding) and their
        poses in the finger body frame, which the cloth solver uses as kinematic pads (world.py)."""
        from scipy.spatial.transform import Rotation
        b = self.b
        bodies = sorted({b.shape_body[i] for i in handle.shape_ids if b.shape_body[i] >= 0})
        fingers = [k for k in bodies if "finger" in b.body_label[k]]
        hands = [k for k in bodies if b.body_label[k].endswith("hand") or b.body_label[k].endswith("/hand")]
        if hands:
            self.hand_bodies[arm] = hands[0]
        else:
            # the import merges the fixed-jointed fr3_link8 and fr3_hand into fr3_link7: the hand's housing rides on
            # link 7, 0.107 m along its axis and turned -45 deg about it (fr3_franka_hand.urdf: fr3_joint8,
            # fr3_hand_joint). Without this the cloth never meets the housing
            l7 = [k for k in range(b.body_count) if b.body_label[k].endswith("link7")
                  and b.body_label[k].startswith(arm + "/")]
            if l7:
                Rh = Rotation.from_euler("z", -np.pi/4)
                c = np.array([0.0, 0.0, 0.107]) + Rh.apply(HAND_HOUSING["center"])
                self.hand_bodies[arm] = l7[0]
                self.hand_housing[arm] = (tuple(c), HAND_HOUSING["half"], tuple(Rh.as_quat()))
        self.finger_boxes[arm] = []
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            tf = np.asarray(b.shape_transform[meshes[0]], float) if meshes else np.array([0, 0, 0, 0, 0, 0, 1.])
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            short = b.body_label[body].split("/")[-1]
            have_box = any(b.shape_type[i] == self.newton.GeoType.BOX for i in shapes)
            for k, (xyz, rpy, size) in enumerate(FRANKA_HAND_FINGER_BOXES):
                R = Rm*Rotation.from_euler("xyz", rpy)
                p = pm + Rm.apply(xyz)
                half = tuple(.5*v for v in size)
                if not have_box:
                    # corners about the centre p, rotated by R about it (box() places the centre and the rotation)
                    self.box(f"{arm}_{short}_box{k}", p - np.asarray(half), p + np.asarray(half), "robot", body=body,
                             rot=self.wp.quat(*map(float, R.as_quat())), density=0., mu=.8)
                self.finger_boxes[arm].append((body, p, R.as_quat(), half))

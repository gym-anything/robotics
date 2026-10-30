"""Rigid scene of the filleting cell (Newton ModelBuilder -> MuJoCo): bench, board, vessels, the two FR3 arms from
the shared robot catalog, the authored knife-handle inserts on the right Franka Hand, and the fillet knife as a
handle plus five hinged blade segments (the flexible blade).

All dimensions come from filleting/layout.py and filleting/params.py. Soft tissue and
bones are NOT here: they live in the tissue solver (tissue.py, bodies.py) and are coupled in world.py.
Visual-only dressing is in dressing.py (render path) and has no collision.
"""

import math
import numpy as np

from . import knife as KN
from . import layout as LY
from . import params as PR

# Franka Hand finger collision boxes (left-finger frame) from the manufacturer's fr3_franka_hand.urdf:
# (xyz, rpy, full size); the same table as label_application/scene.py (read from the URDF there)
FRANKA_HAND_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                            ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                            ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                            ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))

# Knife-handle inserts (authored hardware): per finger a 60 mm long (along the
# handle), 18 mm deep channel block fixed to the fingertip's inner face, with lips that wrap the handle's top and
# bottom edges (the 32 mm handle height) so pitch loads in the blade plane are carried geometrically; 2 mm TPU
# liner (mu 0.8, the repo's finger pad friction). Each insert ~25 g.
INSERT = dict(length=.060, depth=.018, lip=.004, lip_height=.004, back=.004, mass=.025, mu=.8)
HANDLE_H, HANDLE_W, HANDLE_L = PR.v(PR.HANDLE)["height"], PR.v(PR.HANDLE)["width"], PR.v(PR.HANDLE)["length"]
SEG_COM_Z_FRAC = .6        # blade segment COM height / blade height (flat grind: spine-heavy)


class Scene:
    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=-9.81)
        # no contact detection gap on any shape (the imported robots included; Newton's default is 0.1 m): MuJoCo
        # computes the contacts, and in MuJoCo 3.11 a gap only makes it generate inactive contacts out to that distance
        # (the active ones are the same). Its box-box collider, searching that far, reported the knife handle 38.8 mm
        # inside an insert lip 0.25 mm away and pushed with 5 kN during a legal wrist swing
        self.b.rigid_gap = 0.
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.visual, self.bodies = {}, set(), {}
        self.cameras = {}
        self.static_boxes = []          # (name, center, half, quat) static colliders shared with the tissue solver
        self.finger_boxes = []          # (arm, body, name, local pos, local quat xyzw, half, role)
        self.hand_boxes = []            # the Franka Hand housings (same tuple): posed into the tissue solver only
        self.insert_shapes = []         # Newton shape ids of the knife-handle inserts (MuJoCo contact priority)
        self.room()
        self.robots()
        self.knife()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=.45, ke=2e4, kd=100., density=1000., gap=0., margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, **kw):
        cfg = self.cfg(**kw) if collide else self.newton.ModelBuilder.ShapeConfig(
            density=0., has_shape_collision=False, has_particle_collision=False)
        i = self.b.add_shape_box(body, xform=self.wp.transform(pos, rot or self.wp.quat_identity()),
                                 hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        return i

    def static_box(self, name, center, half, material, yaw=0., tissue=True, quat=None, **kw):
        q = LY.yaw_quat(math.degrees(yaw)) if quat is None else np.asarray(quat, float)
        self.box(name, tuple(center), tuple(half), material, rot=self.wp.quat(*map(float, q)), **kw)
        if tissue:
            self.static_boxes.append((name, np.asarray(center, float), np.asarray(half, float), q))

    # ------------------------------------------------------------------ room, bench, vessels
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=.6), label="floor")
        T = LY.TABLE
        z = T["top_z"]
        cx, cy = T["center"]
        sx, sy = T["size"]
        self.static_box("table_top", (cx, cy, z - T["top_thickness"]/2), (sx/2, sy/2, T["top_thickness"]/2),
                        "stainless", mu=.3)
        for x in (cx - sx/2 + .05, cx + sx/2 - .05):
            for y in (cy - sy/2 + .05, cy + sy/2 - .05):
                self.static_box(f"table_leg_{x:+.2f}_{y:+.2f}", (x, y, (z - T["top_thickness"])/2),
                                (T["leg"]/2, T["leg"]/2, (z - T["top_thickness"])/2), "stainless", tissue=False)
        self.static_box("table_undershelf", (cx, cy, T["undershelf_z"]), (sx/2 - .05, sy/2 - .05, .01), "stainless",
                        tissue=False)
        P = LY.PEDESTAL
        self.static_box("pedestal", (P["center"][0], P["center"][1], P["size"][2]/2),
                        (P["size"][0]/2, P["size"][1]/2, P["size"][2]/2), "stainless", tissue=False)
        # damp cloth + HDPE board (a board on a damp cloth does not slide: static)
        B = LY.BOARD
        bx, by = B["center"]
        top = LY.BOARD_TOP_Z
        self.static_box("board", (bx, by, top - B["size"][2]/2), (B["size"][0]/2, B["size"][1]/2, B["size"][2]/2),
                        "hdpe_blue", mu=PR.v(PR.MU_FLESH_HDPE))
        # perforated GN 1/2 fillet pan: floor + four walls (0.8 mm stainless), standing on the bench
        self.pan("fillet_tray", LY.fillet_tray(self.c.task))
        # bench portion scale (presets B, C): its weighing platform is a static collider; the reading is the normal
        # load the tissue exerts on it (World.scale_reading_g), so it weighs whatever physically rests on it
        if self.c.preset in LY.SCALE["presets"]:
            S = LY.SCALE
            self.static_box("scale_platform", (S["center"][0], S["center"][1], z + S["size"][2]/2),
                            tuple(.5*np.asarray(S["size"])), "stainless", mu=PR.v(PR.MU_FLESH_HDPE))
        # stainless bowl as a 16-sided ring of wall slabs on a floor disc (square approximation of the floor)
        W = LY.BOWL
        r, d = W["diameter"]/2, W["depth"]
        bx, by = W["center"]
        self.static_box("bowl_floor", (bx, by, z + .001), (r*.62, r*.62, .001), "stainless")
        for k in range(16):
            a = 2*math.pi*k/16
            cxw, cyw = bx + r*.86*math.cos(a), by + r*.86*math.sin(a)
            self.static_box(f"bowl_wall_{k}", (cxw, cyw, z + d/2), (.0008, r*math.tan(math.pi/16)*1.05, d/2),
                            "stainless", yaw=a)
        if self.c.task in LY.FRAME_PAN["tasks"]:
            self.tapered_pan("frame_pan", LY.FRAME_PAN)
        self.knife_block(z)
        self.bench_stock(z)

    def bench_stock(self, z):
        """Stock and tools the render draws on the bench within the arms' reach get colliders sized from the drawn
        geometry (dressing.py), so a hand or the knife meets them instead of passing through: the GN 1/1 pan of
        crushed ice with its two stand-in trout (preset A), the honing steel behind the board, the tweezer tray and
        its tweezers (preset B), the sanitiser spray bottle and the paper roll at the bench's back corners (0.88-0.93
        m from the arm bases: beyond an arm, within a held knife's reach)."""
        from .dressing import trout_body
        if self.c.preset in LY.ICE_PAN["presets"]:
            I = dict(LY.ICE_PAN, center=LY.ice_pan_center(self.c.task))
            (cx, cy), (sx, sy, sz) = I["center"], I["size"]
            self.pan("ice_pan", I)
            fill = .050                                           # dressing.build_ice_pan: ice 50 mm deep
            ins = .012 + .018*(1 - fill/sz) + .002
            self.static_box("ice_pan_ice", (cx, cy, z + .0008 + fill/2), (sx/2 - ins, sy/2 - ins, fill/2), "stainless",
                            mu=.1)
            V = trout_body()["V"]
            lo, hi = V.min(0), V.max(0)
            for k, ((fx, fy), yaw) in enumerate([((-.005, -.065), math.radians(4.)), ((.012, .070), math.radians(-5.))]):
                c = .5*(lo + hi)
                ca, sa = math.cos(yaw), math.sin(yaw)
                self.static_box(f"ice_pan_trout_{k}", (cx + fx + ca*c[0] - sa*c[1], cy + fy + sa*c[0] + ca*c[1],
                                                       z + .0008 + fill - .003 + (hi[2] - lo[2])/2),
                                tuple(.5*(hi - lo)), "stainless", yaw=yaw, mu=.15)
        # honing steel lying behind the board (dressing: 260 mm rod, 40 mm guard, 125 mm handle, 4 deg yaw)
        yaw = math.radians(4.)
        xc = (.26 - .137)/2
        self.static_box("honing_steel", (-.12 + math.cos(yaw)*xc, .66 + math.sin(yaw)*xc, z + .02),
                        ((.26 + .137)/2, .02, .02), "stainless", yaw=yaw)
        # sanitiser spray (dressing.build_spray_bottle: 65 x 45 x 250 mm, -25 deg) and the centre-feed paper roll
        # (200 mm diameter, 250 mm tall) at the back corners
        self.static_box("spray_bottle", (-.80, .68, z + .125), (.0325, .0225, .125), "stainless", yaw=math.radians(-25))
        self.static_box("paper_roll", (.78, .64, z + .125), (.10*math.cos(math.pi/8), .10*math.cos(math.pi/8), .125),
                        "stainless")
        if self.c.preset == "B":
            T = LY.TWEEZER_TRAY
            self.pan("tweezer_tray", T)
            (cx, cy) = T["center"]
            self.static_box("tweezers", (cx - .06, cy, z + .0008 + .006), (.0055, .06, .006), "stainless")

    def knife_block(self, z):
        """The knife block as the render draws it (dressing.build_knife_block): a 100 x 160 x 200 mm block with a
        40 mm 45 deg front chamfer, and the fillet knife's slot (dressing.knife_block_fit: 70 deg below horizontal,
        blade running down-back, handle up toward the robots) as a 3.6 mm gap between two halves, with a slot floor
        0.3 mm under the blade's edge line and a ceiling 0.5 mm over its spine, so a knife stands in it at the slot
        angle, resting on its handle's front face. The boning knife's handle beside it is a static collider."""
        from scipy.spatial.transform import Rotation
        from .dressing import knife_block_fit, BLOCK_CHAMFER
        K = LY.KNIFE_BLOCK
        (cx, cy), (sx, sy, sz) = K["center"], K["size"]
        fit = knife_block_fit()
        self.block_fit = fit
        xs, g = K["slot_x"][0], 1.8e-3                     # fillet slot centre (block frame x) and half-width
        self.block_slot = dict(x=xs, half_gap=g, origin=np.array([cx, cy, z]))
        c = BLOCK_CHAMFER
        # profile pieces (y, z extents in the block frame): lower full-depth box, upper box behind the chamfer
        pieces = [((-sy/2, sy/2), (0., sz - c)), ((-sy/2 + c, sy/2), (sz - c, sz))]
        for k, (xa, xb) in enumerate(((-sx/2, xs - g), (xs + g, sx/2))):
            hx = (xb - xa)/2
            for j, ((ya, yb), (za, zb)) in enumerate(pieces):
                self.static_box(f"knife_block_{k}{j}", (cx + (xa + xb)/2, cy + (ya + yb)/2, z + (za + zb)/2),
                                (hx, (yb - ya)/2, (zb - za)/2), "wood_block")
            # the chamfer: a box rotated 45 deg about x whose outer face is the chamfer plane
            t = .02
            mid = np.array([(-sy/2 + (-sy/2 + c))/2, (sz - c + sz)/2])            # chamfer midpoint (y, z)
            nrm = np.array([-1., 1.])/math.sqrt(2)                                 # outward normal (y, z)
            ctr = mid - nrm*t/2
            q = Rotation.from_euler("x", 45., degrees=True).as_quat()          # local z -> (0, -1, 1)/sqrt 2
            self.static_box(f"knife_block_{k}c", (cx + (xa + xb)/2, cy + ctr[0], z + ctr[1]),
                            (hx, c/math.sqrt(2), t/2), "wood_block", quat=q)
        # slot floor and ceiling in the gap, rotated about x to the slot angle
        xd, zd = fit["xdir"], fit["zdir"]                 # (0, y, z) unit vectors: heel -> tip, edge -> spine
        heel = np.array([xs, *fit["heels"]["fillet"]])    # blade frame origin (heel, edge line), block frame
        Lb, Hb = PR.v(PR.FILLET_BLADE)["length"], PR.v(PR.FILLET_BLADE)["height"]
        R = np.column_stack([np.array([1., 0, 0]), xd, zd])  # box axes: x across the slot, y along it, z its normal
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        q = Rotation.from_matrix(R).as_quat()
        th = .004
        lam0, lam1 = -.0025, Lb + .005                    # floor from the mouth to beyond the tip
        ctr = heel + xd*(lam0 + lam1)/2 - zd*(.3e-3 + th/2)
        self.static_box("knife_block_slot_floor", (cx + ctr[0], cy + ctr[1], z + ctr[2]), (g, (lam1 - lam0)/2, th/2),
                        "wood_block", quat=q)
        lam0 = .015
        ctr = heel + xd*(lam0 + lam1)/2 + zd*(Hb + .5e-3 + th/2)
        self.static_box("knife_block_slot_ceiling", (cx + ctr[0], cy + ctr[1], z + ctr[2]), (g, (lam1 - lam0)/2, th/2),
                        "wood_block", quat=q)
        # the knife's pose standing in the slot (world): blade frame origin and rotation (x heel->tip, z edge->spine)
        Rk = np.column_stack([xd, np.cross(zd, xd), zd])
        self.block_knife_pose = (np.array([cx, cy, z]) + heel, Rk)
        # the boning knife in the second slot, 70 mm beside it (dressing.build_knife_block): its blade is inside the
        # block and its handle stands above it, 51 mm from the fillet knife's handle. The handle is a static collider
        # (as the dressing draws it: x in [-HANDLE_L, 0] of its blade frame, centred on its spine line), so a hand
        # reaching for the fillet knife meets it. Static: it does not wiggle in its slot when struck.
        heel_b = np.array([K["slot_x"][1], *fit["heels"]["boning"]])
        hc = heel_b + Rk @ np.array([-HANDLE_L/2, 0., PR.v(PR.BONING_BLADE)["height"]])
        self.static_box("knife_block_boning_handle", (cx + hc[0], cy + hc[1], z + hc[2]),
                        (HANDLE_L/2, HANDLE_W/2, HANDLE_H/2), "tpe_black", quat=Rotation.from_matrix(Rk).as_quat())

    def pan(self, name, spec):
        z = LY.TABLE["top_z"]
        (cx, cy), (sx, sy, sz), t = spec["center"], spec["size"], spec.get("wall", .0008)
        self.static_box(f"{name}_floor", (cx, cy, z + t/2), (sx/2, sy/2, t/2), "stainless")
        self.static_box(f"{name}_wall_n", (cx, cy + sy/2 - t/2, z + sz/2), (sx/2, t/2, sz/2), "stainless")
        self.static_box(f"{name}_wall_s", (cx, cy - sy/2 + t/2, z + sz/2), (sx/2, t/2, sz/2), "stainless")
        self.static_box(f"{name}_wall_e", (cx + sx/2 - t/2, cy, z + sz/2), (t/2, sy/2, sz/2), "stainless")
        self.static_box(f"{name}_wall_w", (cx - sx/2 + t/2, cy, z + sz/2), (t/2, sy/2, sz/2), "stainless")

    def tapered_pan(self, name, spec, wall=.002):
        """A solid GN pan as dressing.gn_pan_geometry draws it: floor, four walls whose inner faces run from `taper` +
        `rim` inside the outer size at the floor to `rim` inside it at the top (so a piece rests against the drawn
        sloping wall, not an invisible vertical one), and the rim flange (4 mm thick, `rim` wide). The walls are `wall`
        thick, outward of the drawn inner surface."""
        from scipy.spatial.transform import Rotation
        z, lip = LY.TABLE["top_z"], .004
        (cx, cy), (sx, sy, sz) = spec["center"], spec["size"]
        r, tp, t = spec["rim"], spec["taper"], spec["wall"]
        self.static_box(f"{name}_floor", (cx, cy, z + t/2), (sx/2 - r - tp + wall, sy/2 - r - tp + wall, t/2), "stainless")
        slant = math.hypot(sz, tp)
        up, out = np.array([tp, sz])/slant, np.array([sz, -tp])/slant   # along the wall / its outward normal (out, z)
        zh = np.array([0., 0., 1.])
        for ax, sgn in ((0, 1), (0, -1), (1, 1), (1, -1)):
            e = np.zeros(3); e[ax] = sgn                  # this wall's outward horizontal direction
            half_out, half_len = (sx, sy)[ax]/2, (sy, sx)[ax]/2
            mid = np.array([half_out - r - tp/2, sz/2]) + out*wall/2          # box centre (outward, z)
            ex, ez = e*out[0] + zh*out[1], e*up[0] + zh*up[1]
            q = Rotation.from_matrix(np.column_stack([ex, np.cross(ez, ex), ez])).as_quat()
            tag = f"{'xy'[ax]}{'+' if sgn > 0 else '-'}"
            # the walls run the rim's inner length (closing the corners there)
            self.static_box(f"{name}_wall_{tag}", tuple(np.array([cx, cy, z]) + e*mid[0] + zh*mid[1]),
                            (wall/2, half_len - r + wall, slant/2), "stainless", quat=q)
            fh = np.array([0., 0., lip/2]); fh[ax], fh[1 - ax] = r/2, half_len
            self.static_box(f"{name}_rim_{tag}", tuple(np.array([cx, cy, z + sz - lip/2]) + e*(half_out - r/2)),
                            tuple(fh), "stainless")

    # ------------------------------------------------------------------ robots
    def robots(self):
        from robots.catalog import for_environment, model
        from robots.backends.newton import mount
        import robots
        setup = for_environment("filleting", self.c.robot)
        self.robot_setup = setup
        data = setup.to_dict()
        self.arm_hw = {}
        mounts = {}
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            base = LY.ROBOT_BASES[name]
            mounts[inst["mount"]] = mount(tuple(base["pos"]), math.radians(base["yaw_deg"]))
            self.arm_hw[name] = dict(model=inst["model"], mount=(tuple(base["pos"][:2]), math.radians(base["yaw_deg"])),
                                     base_z=base["pos"][2],
                                     wrist_link=spec.get("frames", {}).get("wrist", {}).get("parent"),
                                     arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            pads = model(att["model"]).to_dict().get("contact", {}).get("fingertip_pads")
            if pads is None:
                raise ValueError(f"filleting needs fingertip pads; attachment {att['model']} declares none")
            self.arm_hw[owner].update(hand=att["model"], pads=pads)
        first_body = self.b.body_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first_body, self.b.body_count))
        self.joints = {}
        for name, handle in self.rig_pending.instances.items():
            for j, entry in handle.joints.items():
                self.joints[j] = entry
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            self._finger_colliders(name, handle)
            self._hand_collider(name, handle)
            self._wrist_camera(name, handle)

    def _hand_collider(self, arm, handle):
        """The Franka Hand housing (the hand meshes merged into link7) as one box for the tissue solver, fitted to the
        meshes' vertices in the hand frame: a free knife (and the fish) meet the palm there as well as in MuJoCo."""
        b = self.b
        from scipy.spatial.transform import Rotation
        link7 = [bd for bd in {b.shape_body[i] for i in handle.shape_ids if b.shape_body[i] >= 0}
                 if b.body_label[bd].endswith("link7")]
        if not link7:
            return
        body = link7[0]
        hand = [i for i in handle.shape_ids if b.shape_body[i] == body and b.shape_type[i] == self.newton.GeoType.MESH
                and float(b.shape_transform[i][2]) > .09]         # hand meshes sit on the flange, link7's own at z = 0
        if not hand:
            return
        tf = np.asarray(b.shape_transform[hand[0]], float)
        Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
        pts = []
        for i in hand:
            src = b.shape_source[i]
            v = np.asarray(src.vertices, float)*np.asarray(b.shape_scale[i], float)
            ti = np.asarray(b.shape_transform[i], float)
            v_body = Rotation.from_quat(ti[3:]).apply(v) + ti[:3]
            pts.append(Rm.inv().apply(v_body - pm))                  # into the hand frame
        pts = np.vstack(pts)
        lo, hi = pts.min(0), pts.max(0)
        c_hand = .5*(lo + hi)
        half = .5*(hi - lo)
        self.hand_boxes.append((arm, body, f"{arm}_hand_housing", pm + Rm.apply(c_hand), Rm.as_quat(), half, "hand"))

    def _finger_colliders(self, arm, handle):
        """The manufacturer URDF's finger collision boxes on every finger (rigid collision shapes, also posed into
        the tissue solver), plus the knife-handle inserts on the right hand's fingers."""
        b = self.b
        from scipy.spatial.transform import Rotation
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        pad_z = self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            tf = np.asarray(b.shape_transform[meshes[0]], float) if meshes else np.array([0, 0, 0, 0, 0, 0, 1.])
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            short = b.body_label[body].split("/")[-1]
            for k, (xyz, rpy, size) in enumerate(FRANKA_HAND_FINGER_BOXES):
                pad = abs(xyz[2] - pad_z) < 1e-6
                R = Rm*Rotation.from_euler("xyz", rpy)
                p = pm + Rm.apply(xyz)
                half = tuple(.5*v for v in size)
                has_box = any(b.shape_type[i] == self.newton.GeoType.BOX for i in shapes)
                if not has_box:
                    self.box(f"{arm}_{short}_hand_box{k}", tuple(p), half, "robot", body=body,
                             rot=self.wp.quat(*map(float, R.as_quat())), density=0., mu=PR.v(PR.MU_INSERT_HANDLE))
                self.finger_boxes.append((arm, body, f"{arm}_{short}_box{k}", p, R.as_quat(), np.asarray(half),
                                          "pad" if pad else "finger"))
            if arm == "right":
                self._insert(arm, body, short, Rm, pm, pad_z)

    def _insert(self, arm, body, short, Rm, pm, pad_z):
        """Knife-handle insert on one right-hand finger (finger frame: y toward the partner finger, z out along the
        finger). A back plate on the fingertip's inner face and two lips above and below the handle's edges; the
        channel is 32 mm (the handle height) + 2 x 0.5 mm clearance. Mass 25 g (authored)."""
        I = INSERT
        # the rubber tip box spans finger-frame y in [-0.02, 15.18] mm: its gripping face is at y ~ 0 and the partner
        # finger lies toward -y (both finger origins meet at the hand centre when closed). The insert sits on
        # that face: back plate y in [-back, 0], lips further in, above and below the handle's edges.
        tip_face_y = 7.58e-3 - 15.2e-3/2
        zc = pad_z                                    # channel centred on the pad
        yb = tip_face_y - I["back"]/2
        yl = tip_face_y - I["back"] - I["lip"]/2
        parts = [("back", (0., yb, zc), (I["length"]/2, I["back"]/2, HANDLE_H/2 + .0005 + I["lip_height"])),
                 ("lip_top", (0., yl, zc + HANDLE_H/2 + .0005 + I["lip_height"]/2),
                  (I["length"]/2, I["lip"]/2, I["lip_height"]/2)),
                 ("lip_bot", (0., yl, zc - HANDLE_H/2 - .0005 - I["lip_height"]/2),
                  (I["length"]/2, I["lip"]/2, I["lip_height"]/2))]
        self.insert_contact_q = HANDLE_W/2 + I["back"] - tip_face_y      # finger opening at which plates touch
        vol = sum(8*h[0]*h[1]*h[2] for _, _, h in parts)
        for nm, xyz, half in parts:
            p = pm + Rm.apply(xyz)
            q = Rm.as_quat()
            self.insert_shapes.append(self.box(f"{arm}_{short}_insert_{nm}", tuple(p), half, "insert_tpu", body=body,
                                               rot=self.wp.quat(*map(float, q)), density=I["mass"]/vol, mu=I["mu"]))
            self.finger_boxes.append((arm, body, f"{arm}_{short}_insert_{nm}", p, q, np.asarray(half), "insert"))

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist RGB on the arm's wrist link (same mount geometry as control_cable/label_application)."""
        labels = self.b.body_label
        wl = self.arm_hw[arm]["wrist_link"]
        link7 = [i for i in handle.body_ids if wl and labels[i].endswith(wl)]
        if not link7:
            return
        body = link7[0]
        self.box(f"{arm}_camera_case", (.075, 0, .07), (.021, .021, .0115), "camera_dark", body=body,
                 density=.060/(.042*.042*.023))
        self.box(f"{arm}_camera_bracket", (.05, 0, .046), (.025, .013, .003), "aluminum", body=body,
                 density=.010/(.050*.026*.006))
        self.cameras[f"{arm}_wrist_rgb"] = dict(body=body, eye=[.075, -.012, .083], target=[.075, -.012, .283],
                                                up=[0, 1, 0], fov=87., role="policy")

    # ------------------------------------------------------------------ knife
    def knife(self):
        """Fillet knife (Victorinox Fibrox 5.3703.18 geometry). MuJoCo holds only the HANDLE: a free body gripped by the
        right hand's inserts. The flexible blade (five hinged segments) is a set of rigid bodies of the tissue solver,
        hanging from a driven copy of this handle (workpiece.py): its flat-wise modes (~1.5 kHz for a 35 mm tip
        segment) and the stiff edge ties to the tissue cannot be coupled explicitly at MuJoCo's 1 ms tick (a MuJoCo
        blade went into a growing 50 Hz flat-wise oscillation inside the tissue: development probe), so they are
        integrated at the tissue substep and only the root reaction reaches the handle.
        Built at a parking pose; world.py places it into the right hand's inserts at reset."""
        spec = KN.BladeSpec("fillet", segments=5)
        self.blade = spec
        wp = self.wp
        steel = 7900.
        # blade steel mass from its volume: flat grind to the edge shoulder (mean thickness ~0.55 x spine)
        self.blade_seg_mass = [steel*spec.seg_len*spec.H*.55*spec.spine_at((s + .5)/spec.segments)
                               for s in range(spec.segments)]
        m_blade = float(sum(self.blade_seg_mass))
        m_handle = max(spec.mass - m_blade, .02)
        self.knife_mass = dict(blade=m_blade, handle=m_handle, total=m_blade + m_handle)
        # blade segment frames in the handle frame: edge centre of segment s (x along the blade, z edge -> spine);
        # the spine continues the handle's centre line [estimate from product photographs]
        self.blade_seg_local = [np.array([HANDLE_L/2 + spec.seg_len*(s + .5), 0., -spec.H]) for s in range(spec.segments)]
        self.blade_seg_com_z = SEG_COM_Z_FRAC*spec.H          # flat-ground profile: spine-heavy
        self.handle_ref_local = np.zeros(3)                   # the tissue solver's handle body: the grip box centre
        # MuJoCo's knife carries the WHOLE knife's rigid-body mass and inertia (handle + blade at rest shape; true
        # values), so the tissue solver passes it only the loads the world applies to the blade (contacts), not the
        # blade chain's stiff root-joint force: released from the hand, the 71 g handle against that 2e5 N/m joint
        # was explicitly unstable at the 1 ms tick (E15: k dt^2/m = 2.8)
        mb, cb, Ib = 0., np.zeros(3), np.zeros((3, 3))
        parts = []
        for s_ in range(spec.segments):
            m_ = self.blade_seg_mass[s_]
            t_ = spec.spine_at((s_ + .5)/spec.segments)
            c_ = self.blade_seg_local[s_] + np.array([0., 0., self.blade_seg_com_z])
            I_ = np.diag([m_*(spec.H**2 + t_**2)/12, m_*(spec.seg_len**2 + spec.H**2)/12, m_*(spec.seg_len**2 + t_**2)/12])
            parts.append((m_, c_, I_))
            mb += m_; cb += m_*c_
        cb /= mb
        for m_, c_, I_ in parts:
            d_ = c_ - cb
            Ib += I_ + m_*(np.dot(d_, d_)*np.eye(3) - np.outer(d_, d_))
        park = np.array([.36, .60, LY.TABLE["top_z"] + .25])
        h = self.b.add_link(xform=wp.transform(tuple(park - np.array([spec.L/2 + HANDLE_L/2, 0, 0])), wp.quat_identity()),
                            mass=float(mb), com=wp.vec3(*cb), inertia=wp.mat33(*Ib.ravel()), label="knife_handle")
        self.bodies["knife_handle"] = h
        self.box("knife_handle_grip", (0, 0, 0), (HANDLE_L/2, HANDLE_W/2, HANDLE_H/2), "tpe_black", body=h,
                 density=m_handle/(HANDLE_L*HANDLE_W*HANDLE_H), mu=PR.v(PR.MU_INSERT_HANDLE))
        # the blade's rigid envelope on MuJoCo's knife (massless: the link already carries the blade's mass): while a
        # hand holds the knife, the blade meets the board, bench, block and robot here, in MuJoCo's implicit contact
        # solver (explicitly coupled across the 1 ms tick, the blade's stiff contacts with rigid geometry went unstable:
        # the scripted A0 pickup, drawing the blade along its slot); the tissue solver keeps the blade against tissue
        # and bones. A free knife is integrated wholly in the tissue solver (World._set_knife_mode).
        self.blade_rigid_shapes = []
        for s_ in range(spec.segments):
            c_ = self.blade_seg_local[s_] + np.array([0., 0., spec.H/2])
            self.blade_rigid_shapes.append(self.box(f"knife_blade_rigid_{s_}", tuple(c_),
                                                    (spec.seg_len/2, spec.t/2, spec.H/2), "steel", body=h, density=0.,
                                                    mu=PR.v(PR.MU_BLADE_TISSUE)))
        self.b.add_articulation([self.b.add_joint_free(h, label="knife_free")], label="knife")

    def free_body(self, name, pos, rot=None):
        body = self.b.add_body(xform=self.wp.transform(pos, rot or self.wp.quat_identity()), label=name)
        self.bodies[name] = body
        return body

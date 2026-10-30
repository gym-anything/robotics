"""The paint system of one episode: paints, film charts, bulk reservoirs, applicators (roller, brushes, aerosol), drips
and the exact solids/vehicle ledger. Owned by the environment and coupled to the rigid world once
per 0.5 ms tick (painting.world.PaintingWorld.step):

    pre_step():  applicator contact wrenches into world.body_wrench (nap pressure and traction, tuft tip reactions)
    post_step(): paint exchange at those contacts, reservoir uptake, valve and spray, drips; films every
                 FILM.every_ticks (100 ms while any film is moving, 500 ms when every film has rested)
Nothing here is reachable from the policy interface: paint only moves because a tool touches, presses or sprays.
"""

import math
import numpy as np

from . import params as P
from . import layout as L
from . import tools as T
from .paint import make_paint
from .film import Chart
from .reservoir import Reservoir, box_cavity, cyl_cavity, ring
from . import contact as C


class PaintSystem:
    def __init__(self, world, spec):
        self.w, self.spec = world, spec
        sp = spec
        clim = sp["climate"]
        pk = sp["paint"]
        if "colors" in pk:
            self.paints = [make_paint(pk["kind"], c, clim) for c in pk["colors"]]
        else:
            self.paints = [make_paint(pk["kind"], pk["color"], clim)]
        self.charts, self.reservoirs, self.applicators, self.sinks = {}, {}, {}, {}
        self.prims = []
        self.drips = None
        self.tick_film = 0
        self.film_info = dict(calls=0, substeps=0, limited=0)
        self.timing = dict(pre_s=0.0, post_s=0.0, film_s=0.0)
        self.events = []
        self.enabled = True
        self.skins = {}                      # MuJoCo body id -> wet paint carried on a robot link / tool part
        self._skin_tick = 0
        self._build()
        self._stencil_block(self.w.state.body_q.numpy())
        self.initial_totals = self.totals()

    # ------------------------------------------------------------------ construction
    def _chart(self, name, **kw):
        ch = Chart(name, self.paints, **kw)
        self.charts[name] = ch
        return ch

    def _build(self):
        sp, sc = self.spec, self.w.scene
        preset = sp["preset"]
        Z, X, YW = L.Z, L.X, L.YW
        cw = P.CELL
        if preset == "wall":
            self._wall_charts()
        # contact primitives mirroring the static colliders
        wall_charts = ("wall",) if "wall" in self.charts else ()
        for name, c, h, mat in L.static_boxes():
            charts = ()
            if name.startswith("wall_n"):
                charts = wall_charts
            elif name.startswith("casing"):
                charts = tuple(n for n in self.charts if n.startswith("casing"))
            elif name in ("stool", "apron"):
                charts = tuple(n for n in ("stool_top", "casing_front") if n in self.charts)
            elif name == "base_n":
                charts = tuple(n for n in ("base_top", "base_front") if n in self.charts)
            elif name == "outlet_plate":
                charts = ("outlet_plate",) if "outlet_plate" in self.charts else ()
            elif name == "ceiling":
                charts = ("ceiling",) if "ceiling" in self.charts else ()
            sink = "radiator" if name.startswith("rad_") else name
            self.prims.append(C.Box(name, -1, c, (0, 0, 0, 1), h, charts=charts, sink=sink))
        pc = L.PIPE
        self.prims.append(C.Cylinder("rad_pipe", -1, (pc["x"], pc["y"], pc["z1"]/2), (0, 0, 0, 1), pc["r"], pc["z1"]/2,
                                     sink="radiator"))
        self.prims.append(C.Plane("floor", (0, 0, 0), (0, 0, 1), charts=("cloth",) if "cloth" in self.charts else (),
                                  sink="floor"))
        # tools and props
        for name, it in sc.items.items():
            kind = it["kind"]
            body = it["body"]
            if kind in ("roller", "roller_cover", "aerosol_button") or kind in ("sash", "round8", "stencil_brush", "longreach"):
                continue                         # applicators do not contact themselves; brushes: ferrule/handle via MuJoCo
            for p in sc.parts[name]:
                if not p.collide:
                    continue
                charts = ()
                holes = None
                if kind == "tray" and p.role == "ramp":
                    charts = ("tray_ramp",)
                if kind == "board":
                    charts = ("paper",)
                if kind == "crate":
                    charts = ("crate",)
                if kind == "stencil":
                    charts = ("stencil",)
                    holes = self._stencil_holes
                if kind == "palette":
                    holes = self._palette_holes
                if kind == "table":
                    charts = ("table",) if p.name == "top" else ()
                if kind == "spray_box":
                    # names, not a lookup: the box charts are created after the primitives (an empty lookup sends all
                    # spray landing in the box to its sink, P5/P8)
                    # the print lies on the booth's floor: it takes what lands on it first (with the floor's chart
                    # open under the print's 4 mm rim, listing the floor first starves the print's edges)
                    charts = (("print",) if p.name == "floor" else ()) + (f"box_{p.name}",)
                if p.kind == "box":
                    pos, size = p.pos, p.size
                    if (kind == "tray" and p.name in ("well_floor", "ramp", "lip")) or kind in ("stencil", "board", "crate"):
                        # thin slabs: the compliant layers (nap, tufts) may sink deeper than the slab is thick; extend
                        # the contact solid downward (into what lies under it anyway) so the nearest face stays the top
                        from .film import _quat_rot
                        ext = 0.012
                        pos = tuple(np.asarray(p.pos) + _quat_rot(p.quat) @ np.array([0., 0., -ext]))
                        size = (p.size[0], p.size[1], p.size[2] + ext)
                    self.prims.append(C.Box(f"{name}/{p.name}", body, pos, p.quat, size, charts=charts, sink=name, holes=holes))
                elif p.kind == "cylinder":
                    self.prims.append(C.Cylinder(f"{name}/{p.name}", body, p.pos, p.quat, p.size[0], p.size[1], charts=charts,
                                                 sink=name))
                elif p.kind == "capsule":
                    self.prims.append(C.Cylinder(f"{name}/{p.name}", body, p.pos, p.quat, p.size[0], p.size[1] + p.size[0],
                                                 charts=charts, sink=name))
        self._tool_charts()
        self._reservoirs()
        self._prepaint()
        self._pencil()
        from .applicators import build_applicators
        build_applicators(self)
        from .drips import Drips
        self.drips = Drips(self)

    def _wall_charts(self):
        sp = self.spec
        X, Z, YW = L.X, L.Z, L.YW
        W, CS, CT = L.WIN, L.CS, L.CT
        cw = P.CELL["wall"]
        nu, nv = int(round(X/cw)), int(round(Z/cw))
        xs = -X/2 + (np.arange(nu) + 0.5)*cw
        zs = (np.arange(nv) + 0.5)*cw
        XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
        mask = np.ones((nu, nv), bool)
        # window opening, casing, stool and apron footprints; baseboard; outlet plate
        mask &= ~((XX > L.WX0 - CS) & (XX < L.WX1 + CS) & (ZZ > W["sill"] - W["stool_t"] - W["apron"]) & (ZZ < W["head"] + CS))
        mask &= ~(ZZ < L.BB["h"])
        o = L.OUT
        mask &= ~((np.abs(XX - o["x"]) < o["w"]/2) & (np.abs(ZZ - o["z"]) < o["h"]/2))
        S = P.SUBSTRATE
        sorp = np.full((nu, nv), S["primed_drywall"]["sorptivity"], np.float32)
        cap = np.full((nu, nv), S["primed_drywall"]["capacity"], np.float32)
        for (px, pz, r) in sp.get("patches", []):
            m = (XX - px)**2 + (ZZ - pz)**2 < r*r
            sorp[m] = S["spackle"]["sorptivity"]; cap[m] = S["spackle"]["capacity"]
        from .paint import srgb_to_linear
        sub = np.broadcast_to(srgb_to_linear(P.COLORS[sp["old_color"]]), (nu, nv, 3)).copy()
        for (px, pz, r) in sp.get("patches", []):
            m = (XX - px)**2 + (ZZ - pz)**2 < r*r
            sub[m] = srgb_to_linear(P.COLORS["primer"])*0.97          # dry spackle, lighter than the old colour
        self._chart("wall", origin=(-X/2, YW, 0.0), u=(1, 0, 0), v=(0, 0, 1), size=(X, Z), cell=cw, substrate="primed_drywall",
                    substrate_color=sub, mask=mask, sorptivity=sorp, capacity=cap, role="target")
        self._chart("ceiling", origin=(-X/2, YW, Z), u=(1, 0, 0), v=(0, -1, 0), size=(X, 0.25), cell=P.CELL["ceiling"],
                    substrate_color="ceiling_white", role="protected")
        # casing front (legs, head and apron) on one chart, masked to the trim
        x0, x1 = L.WX0 - CS, L.WX1 + CS
        z0 = W["sill"] - W["stool_t"] - W["apron"]
        z1 = W["head"] + CS
        ct = P.CELL["trim"]
        cu, cv = int(round((x1 - x0)/ct)), int(round((z1 - z0)/ct))
        cx = x0 + (np.arange(cu) + 0.5)*ct; cz = z0 + (np.arange(cv) + 0.5)*ct
        CX, CZ = np.meshgrid(cx, cz, indexing="ij")
        legs = ((CX < L.WX0) | (CX > L.WX1)) & (CZ > W["sill"])
        head = CZ > W["head"]
        apron = (CZ < W["sill"] - W["stool_t"]) & (CX > L.WX0 - CS) & (CX < L.WX1 + CS)
        self._chart("casing_front", origin=(x0, YW - CT, z0), u=(1, 0, 0), v=(0, 0, 1), size=(x1 - x0, z1 - z0), cell=ct,
                    substrate_color="casing_white", mask=legs | head | apron, role="protected")
        self._chart("casing_side_l", origin=(x0, YW, W["sill"]), u=(0, -1, 0), v=(0, 0, 1), size=(CT, z1 - W["sill"]), cell=ct,
                    substrate_color="casing_white", role="protected")
        self._chart("casing_side_r", origin=(x1, YW - CT, W["sill"]), u=(0, 1, 0), v=(0, 0, 1), size=(CT, z1 - W["sill"]), cell=ct,
                    substrate_color="casing_white", role="protected")
        self._chart("casing_top", origin=(x0, YW - CT, z1), u=(1, 0, 0), v=(0, 1, 0), size=(x1 - x0, CT), cell=ct,
                    substrate_color="casing_white", role="protected")
        sw = W["w"]/2 + CS + 0.02
        self._chart("stool_top", origin=(W["x"] - sw, YW - W["stool_d"], W["sill"]), u=(1, 0, 0), v=(0, 1, 0),
                    size=(2*sw, W["stool_d"]), cell=ct, substrate_color="casing_white", role="protected")
        bb = L.BB
        self._chart("base_top", origin=(-X/2, YW - bb["t"], bb["h"]), u=(1, 0, 0), v=(0, 1, 0), size=(X, bb["t"]), cell=ct,
                    substrate_color="casing_white", role="protected")
        self._chart("base_front", origin=(-X/2, YW - bb["t"], 0.0), u=(1, 0, 0), v=(0, 0, 1), size=(X, bb["h"]), cell=ct,
                    substrate_color="casing_white", role="protected")
        o = L.OUT
        self._chart("outlet_plate", origin=(o["x"] - o["w"]/2, YW - o["t"], o["z"] - o["h"]/2), u=(1, 0, 0), v=(0, 0, 1),
                    size=(o["w"], o["h"]), cell=1e-3, substrate_color="casing_white", role="protected")
        cl = L.CLOTH
        self._chart("cloth", origin=(cl["x0"], cl["y0"], 0.0005), u=(1, 0, 0), v=(0, 1, 0), size=(cl["x1"] - cl["x0"], cl["y1"] - cl["y0"]),
                    cell=P.CELL["floor"], substrate="canvas_cloth", substrate_color="canvas_cloth", role="catch")

    def _tool_charts(self):
        sc = self.w.scene
        for name, it in sc.items.items():
            kind, body = it["kind"], it["body"]
            if kind == "tray":
                g = T.tray_geometry()
                t = g["t"]
                L_ = math.hypot(g["x_ramp_end"] - g["x_well"], g["z_ramp"] - g["z_bot"])
                u = (math.cos(g["ang"]), 0.0, math.sin(g["ang"]))
                self._chart("tray_ramp", body=body, origin=(g["x_well"], -(g["W"]/2 - t), g["z_bot"]), u=u, v=(0, 1, 0),
                            size=(L_, g["W"] - 2*t), cell=P.CELL["tray"], substrate_color=(0.07, 0.07, 0.075), role="tool",
                            sink_edges=("u0",))
            elif kind == "board":
                self._chart("paper", body=body, origin=(-0.1905, -0.1395, 0.0125), u=(1, 0, 0), v=(0, 1, 0), size=(0.381, 0.279),
                            cell=P.CELL["paper"], substrate="watercolour_paper", substrate_color="watercolour_paper", role="target")
            elif kind == "table":
                T_ = P.TABLE
                self._chart("table", body=body, origin=(-T_["size"][0]/2, -T_["size"][1]/2, T_["height"] + 0.0002), u=(1, 0, 0),
                            v=(0, 1, 0), size=T_["size"], cell=P.CELL["table"], substrate_color=(0.75, 0.74, 0.70), role="protected")
            elif kind == "crate":
                self._chart("crate", body=body, origin=(-0.225, -0.15, 0.0121), u=(1, 0, 0), v=(0, 1, 0), size=(0.45, 0.30),
                            cell=P.CELL["crate"], substrate="plywood_sealed", substrate_color="plywood", role="target")
            elif kind == "stencil":
                sx, sy = P.STENCIL["size"]
                d = P.CELL["stencil"]
                nu, nv = int(round(sx/d)), int(round(sy/d))
                xs = -sx/2 + (np.arange(nu) + 0.5)*d; ys = -sy/2 + (np.arange(nv) + 0.5)*d
                XX, YY = np.meshgrid(xs, ys, indexing="ij")
                ap = self._stencil_holes(np.stack([XX.ravel(), YY.ravel()], 1)).reshape(nu, nv)
                st = self._chart("stencil", body=body, origin=(-sx/2, -sy/2, P.STENCIL["thick"] + 0.0001), u=(1, 0, 0),
                                 v=(0, 1, 0), size=(sx, sy), cell=d, substrate="pet_on_plywood", substrate_color="pet_on_plywood",
                                 mask=~ap, role="tool")
                # the clear sheet's cut edges catch the light (appearance of the bare sheet only: a darker 1 mm rim
                # around each aperture and the sheet's outline, the film a touch lighter than the plywood under it)
                from scipy.ndimage import binary_erosion
                rim = st.mask & ~binary_erosion(np.pad(st.mask, 1, constant_values=False))[1:-1, 1:-1]
                st.substrate_rgb *= 1.06
                st.substrate_rgb[rim] *= 0.55
            elif kind == "spray_box":
                sx, sy, sz = T.SPRAY_BOX["size"]; t = T.SPRAY_BOX["t"]
                ix, iy = sx/2 - t, sy/2 - t
                cb = P.CELL["box"]
                mask = None
                # the floor's chart runs out to its open front edge, and under a print only the cells wholly beneath it
                # are masked, so paint on the 4 mm front strip and in the ring round the print has a chart
                fy0, fsy = -sy/2, iy + sy/2
                if self.spec.get("print"):
                    nu, nv = int(round(2*ix/cb)), int(round(fsy/cb))
                    xs = -ix + (np.arange(nu) + 0.5)*cb; ys = fy0 + (np.arange(nv) + 0.5)*cb
                    XX, YY = np.meshgrid(xs, ys, indexing="ij")
                    mask = ~((np.abs(XX) < 0.127 - cb/2) & (np.abs(YY) < 0.1016 - cb/2))
                self._chart("box_floor", body=body, origin=(-ix, fy0, t + 0.0002), u=(1, 0, 0), v=(0, 1, 0), size=(2*ix, fsy), cell=cb,
                            substrate="cardboard", substrate_color="cardboard", role="catch", mask=mask)
                self._chart("box_back", body=body, origin=(-ix, iy, t), u=(1, 0, 0), v=(0, 0, 1), size=(2*ix, sz - 2*t), cell=cb,
                            substrate="cardboard", substrate_color="cardboard", role="catch")
                self._chart("box_left", body=body, origin=(-ix, iy, t), u=(0, -1, 0), v=(0, 0, 1), size=(2*iy, sz - 2*t), cell=cb,
                            substrate="cardboard", substrate_color="cardboard", role="catch")
                self._chart("box_right", body=body, origin=(ix, -iy, t), u=(0, 1, 0), v=(0, 0, 1), size=(2*iy, sz - 2*t), cell=cb,
                            substrate="cardboard", substrate_color="cardboard", role="catch")
                if self.spec.get("print"):
                    pr = self._chart("print", body=body, origin=(-0.127, -0.1016, t + 0.0003), u=(1, 0, 0), v=(0, 1, 0),
                                     size=(0.254, 0.2032), cell=P.CELL["print"], substrate_color="photo", role="target")
                    self._photo(pr)

    @staticmethod
    def _photo(ch):
        """The 10 x 8 in print's image as its substrate colour per cell (what a clear coat goes over): a tone-mapped crop
        of Poly Haven's CC0 'Suburban Football Field' HDRI (painting/assets/photo_print.jpg)."""
        from pathlib import Path
        from PIL import Image
        from .paint import srgb_to_linear
        im = Image.open(Path(__file__).resolve().parent/"assets"/"photo_print.jpg").convert("RGB").resize((ch.nu, ch.nv), Image.LANCZOS)
        a = np.asarray(im, float)[::-1].transpose(1, 0, 2)                  # rows top-down -> cells (u right, v up)
        ch.substrate_rgb = np.ascontiguousarray(srgb_to_linear(a))

    # stencil apertures: a bold "3" (two bowls of a stroke 22 mm wide) cut in the 16 x 12 in sheet, no islands
    def _stencil_holes(self, xy):
        xy = np.asarray(xy, float)
        x, y = xy[:, 0], xy[:, 1]
        out = np.zeros(len(xy), bool)
        for cy in (0.036, -0.036):
            r = np.hypot(x - 0.0, y - cy)
            ring_ = (r > 0.030) & (r < 0.052)
            ang = np.degrees(np.arctan2(y - cy, x))
            out |= ring_ & ((ang > -105) & (ang < 125) if cy > 0 else (ang > -125) & (ang < 105))
        return out & (np.abs(x) < 0.13) & (np.abs(y) < 0.09)

    def _palette_holes(self, xy):
        xy = np.asarray(xy, float)
        out = np.zeros(len(xy), bool)
        for (wx, wy) in T.PALETTE_WELLS:
            out |= np.hypot(xy[:, 0] - wx, xy[:, 1] - wy) < T.PALETTE_WELL["r"]
        return out

    def _reservoirs(self):
        sc = self.w.scene
        P0 = len(self.paints)

        def fill(volume, k=0):
            s = np.zeros(P0); s[k] = volume*self.paints[k].phi0
            return s, volume*(1 - self.paints[k].phi0)

        for name, it in self.spec["items"].items():
            kind = it["kind"]
            if kind == "tray":
                g = T.tray_geometry()
                t = g["t"]
                floor = lambda x, g=g: g["z_bot"] if x < g["x_well"] else g["z_bot"] + (x - g["x_well"])*math.tan(g["ang"])
                pts, w = box_cavity(-g["L"]/2 + t, g["L"]/2 - t, -g["W"]/2 + t, g["W"]/2 - t, floor, g["H"])
                rim = np.array([[-g["L"]/2, -g["W"]/2, g["H"]], [g["L"]/2, -g["W"]/2, g["H"]], [g["L"]/2, g["W"]/2, g["H"]],
                                [-g["L"]/2, g["W"]/2, g["H"]], [-g["L"]/2, -g["W"]/2, g["H"]]])
                wall = lambda q, g=g: (np.abs(q[:, 0]) < g["L"]/2 - g["t"]) & (np.abs(q[:, 1]) < g["W"]/2 - g["t"])
                s, v = fill(it["fill"])
                self.reservoirs[name] = Reservoir(name, sc.bodies[name], self.paints, pts, w, rim, solids=s, vehicle=v, wall_fn=wall)
            elif kind in ("pail", "can"):
                G = P.PAIL if kind == "pail" else P.CAN
                r0 = G.get("r_bottom", G.get("r")) - 0.002
                r1 = G.get("r_top", G.get("r")) - 0.002
                pts, w = cyl_cavity(r0, r1, 0.002, G["height"])
                wall = lambda q, r0=r0, r1=r1, H=G["height"]: np.hypot(q[:, 0], q[:, 1]) < r0 + (r1 - r0)*np.clip(q[:, 2]/H, 0, 1)
                s, v = fill(it["fill"])
                self.reservoirs[name] = Reservoir(name, sc.bodies[name], self.paints, pts, w, ring(r1, G["height"]), solids=s,
                                                  vehicle=v, wall_fn=wall)
            elif kind == "palette":
                for k, (wx, wy) in enumerate(T.PALETTE_WELLS):
                    if k not in it.get("wells", {}):
                        continue
                    color, vol = it["wells"][k]
                    pk = [p.color for p in self.paints].index(color)
                    W_ = T.PALETTE_WELL
                    pts, w = cyl_cavity(W_["r"], W_["r"], W_["top"] - W_["depth"], W_["top"], step=0.002)
                    pts = pts + np.array([wx, wy, 0.0])
                    wall = lambda q, wx=wx, wy=wy, r=W_["r"]: np.hypot(q[:, 0] - wx, q[:, 1] - wy) < r
                    s, v = fill(vol, pk)
                    self.reservoirs[f"{name}_well{k}"] = Reservoir(f"{name}_well{k}", sc.bodies[name], self.paints, pts, w,
                                                                   ring(W_["r"], W_["top"]) + np.array([wx, wy, 0.0]), solids=s,
                                                                   vehicle=v, wall_fn=wall)

    def _pencil(self):
        """Graphite template lines on the paper (the craft tasks' pencilled circle, S-stroke outline, square): darker
        substrate cells ~0.3 mm wide, which keep showing faintly through thin acrylic (K-M over the substrate)."""
        ch = self.charts.get("paper")
        if ch is None:
            return
        from .task import S_CURVE, stroke_width
        xs = (np.arange(ch.nu) + 0.5)*ch.dx - ch.size[0]/2
        ys = (np.arange(ch.nv) + 0.5)*ch.dx - ch.size[1]/2
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        t = self.spec["task"]
        line = 0.6*ch.dx
        if t == "fill-circle":
            m = np.abs(np.hypot(X, Y) - 0.030) < line
        elif t == "swatch-card":
            m = ((np.abs(np.abs(X) - 0.025) < line) & (np.abs(Y) <= 0.025 + line)) | \
                ((np.abs(np.abs(Y) - 0.025) < line) & (np.abs(X) <= 0.025 + line))
        elif t == "brush-stroke":
            m = np.zeros(X.shape, bool)
            ts = np.linspace(0, 1, 800)
            c = S_CURVE(ts)
            tang = np.gradient(c, axis=0)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)/np.linalg.norm(tang, axis=1, keepdims=True)
            for side in (-1, 1):
                pts = c + side*nrm*(stroke_width(ts)/2)[:, None]
                ii = np.clip(((pts[:, 0] + ch.size[0]/2)/ch.dx).astype(int), 0, ch.nu - 1)
                jj = np.clip(((pts[:, 1] + ch.size[1]/2)/ch.dx).astype(int), 0, ch.nv - 1)
                m[ii, jj] = True
        else:
            return
        ch.substrate_rgb[m] = ch.substrate_rgb[m]*0.62

    def _prepaint(self):
        """Seeded wet films (bands brushed and a strip rolled earlier) with their age applied analytically: vehicle lost
        at the evaporation rate and structure recovered with t_r; and the dry coat of a painted wall (touch-up)."""
        sp = self.spec
        self._cloth_stains()
        ch = self.charts.get("wall")
        if ch is None:
            return
        pt = self.paints[0]
        xs = -L.X/2 + (np.arange(ch.nu) + 0.5)*ch.dx
        zs = (np.arange(ch.nv) + 0.5)*ch.dx
        fc = float(sp.get("first_coat") or 0.0)
        if fc and not sp.get("painted"):
            # the dry first coat of a deep colour, a little patchy (+-15 % over 0.1-0.3 m, as a rolled first coat is):
            # it hides nowhere on its own (E5), so the start is unsolved
            rng = np.random.default_rng(7000 + int(sp.get("seed", 0)))
            g = rng.normal(0.0, 1.0, (max(2, ch.nu//60), max(2, ch.nv//60)))
            from scipy.ndimage import zoom
            f = zoom(g, (ch.nu/g.shape[0], ch.nv/g.shape[1]), order=3)[:ch.nu, :ch.nv]
            f = f/max(float(np.abs(f).max()), 1e-9)
            ch.d[0][ch.mask] = (fc*(1.0 + 0.15*f))[ch.mask]
        if sp.get("painted"):
            # the finished wall: 45 um (one coat) of a light colour, or a deep colour's two coats (first + finish)
            ch.d[0][ch.mask] = 45e-6 + fc
            for (sx, sz) in sp.get("scuffs", []):
                XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
                m = ch.mask & (((XX - sx)/0.035)**2 + ((ZZ - sz)/0.012)**2 < 1)
                if fc:
                    ch.d[0][m] = fc                                            # finish coat worn through to the first
                else:
                    ch.d[0][m] = 0.0
                    ch.substrate_rgb[m] = ch.substrate_rgb[m]*0.45            # grey-black scuff on the old colour
        if fc or sp.get("painted"):
            ch.d_base = ch.d.sum(axis=0).copy()           # the coats already on the wall (laps are judged above them)
        for band in sp.get("prepainted", []):
            x0, x1, z0, z1 = band["region"]
            i0, i1 = int((x0 + L.X/2)/ch.dx), int((x1 + L.X/2)/ch.dx)
            j0, j1 = int(z0/ch.dx), int(z1/ch.dx)
            I, J = np.meshgrid(np.arange(max(0, i0), min(ch.nu, i1)), np.arange(max(0, j0), min(ch.nv, j1)), indexing="ij")
            I, J = I.ravel(), J.ravel()
            keep = ch.mask[I, J]
            I, J = I[keep], J[keep]
            h = band["h"]
            lost = min(h*(1 - pt.phi0)*0.9, pt.j_evap*band["age"])
            s = np.full((1, len(I)), h*pt.phi0)
            v = np.full(len(I), h*(1 - pt.phi0) - lost)
            ch.add(I, J, s, v, texture=(0.25, P.FILM["tex_wl_roller"] if band["age"] > 100 else P.FILM["tex_wl_brush"], 0.0),
                   shear=True)
            ch.lam[I, J] = 1.0 - math.exp(-band["age"]/pt.rec["t_r"])
            ch.evaporated[-1] += lost*len(I)*ch.area
            self._prepainted_vehicle = getattr(self, "_prepainted_vehicle", 0.0) + lost*len(I)*ch.area

    def _cloth_stains(self):
        """A used canvas drop cloth: old dry drips and smears from earlier jobs are part of its substrate colour (like the
        wall's scuffs; appearance of the substrate, not paint of this episode -- the cloth carries no task criterion)."""
        ch = self.charts.get("cloth")
        if ch is None:
            return
        rng = np.random.default_rng(77 + int(self.spec.get("seed", 0)))
        # the room's earlier coats: ceiling and trim whites, the old beige, a grey primer, rarely a dark accent
        old = [np.asarray(c, float)/255.0 for c in ((240, 238, 230), (236, 233, 222), (209, 199, 184), (205, 196, 178),
                                                    (176, 178, 176), (232, 228, 214), (60, 72, 92))]
        weights = np.array([3, 3, 3, 2, 1, 2, 0.5]); weights = weights/weights.sum()
        from .paint import srgb_to_linear
        us = (np.arange(ch.nu) + 0.5)*ch.dx
        vs = (np.arange(ch.nv) + 0.5)*ch.dx
        UU, VV = np.meshgrid(us, vs, indexing="ij")
        for k in range(110):
            c = srgb_to_linear(tuple(255*old[rng.choice(len(old), p=weights)]))
            u0, v0 = rng.uniform(0, ch.size[0]), rng.uniform(0, ch.size[1])
            if rng.random() < 0.8:                                       # a drip: 6-30 mm across, ragged
                r = rng.uniform(0.003, 0.015)
                ang = np.arctan2(VV - v0, UU - u0)
                ph = rng.uniform(0, 2*np.pi, 3)
                rr = r*(1 + 0.18*np.sin(3*ang + ph[0]) + 0.12*np.sin(5*ang + ph[1]) + 0.08*np.sin(8*ang + ph[2]))
                m = (UU - u0)**2 + (VV - v0)**2 < rr*rr
            else:                                                        # a smear where a tool was set down
                a, L_, w_ = rng.uniform(0, np.pi), rng.uniform(0.05, 0.18), rng.uniform(0.010, 0.03)
                du, dv = UU - u0, VV - v0
                along, across = du*np.cos(a) + dv*np.sin(a), -du*np.sin(a) + dv*np.cos(a)
                m = (np.abs(along) < L_/2) & (np.abs(across) < w_/2*(1 - (2*along/L_)**2).clip(0.2))
            f = rng.uniform(0.45, 0.8)                                  # soaked into the canvas, worn
            ch.substrate_rgb[m] = (1 - f)*ch.substrate_rgb[m] + f*c

    # ------------------------------------------------------------------ per tick
    def pre_step(self):
        if not self.enabled:
            return
        import time
        t0 = time.perf_counter()
        bq = self.w.state.body_q.numpy()
        for a in self.applicators.values():
            a.forces(bq)
        self.timing["pre_s"] += time.perf_counter() - t0

    def post_step(self):
        if not self.enabled:
            return
        import time
        t0 = time.perf_counter()
        bq = self.w.state.body_q.numpy()
        now = self.w.sim_time
        for a in self.applicators.values():
            a.transfer(bq, now)
        self.drips.step(bq, P.DT)
        self._skin_tick += 1
        if self._skin_tick >= self.SKIN["every"]:
            self._skin_tick = 0
            self._skin_step(bq)
        self.tick_film += 1
        every = P.FILM["every_ticks"]*(1 if self.films_active() else 5)
        if self.tick_film >= every:
            t1 = time.perf_counter()
            self.advance_films(self.tick_film*P.DT, bq, now)
            self.timing["film_s"] += time.perf_counter() - t1
            self.tick_film = 0
            self._lump(bq)
        self.timing["post_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ wet paint on robot links and tool parts
    SKIN = dict(every=10, r=4e-3, split=0.5, wet_min=0.5e-6, tools=("roller", "sash", "longreach", "round8", "aerosol"))

    def _skin_setup(self):
        """Which MuJoCo geoms carry a paint skin: every robot link, and the tools' parts that are not their applicators
        (the roller frame and grip, brush ferrules and handles, the aerosol can) -- not nap, bristles or cover."""
        w, m = self.w, self.w.mjm
        lab = list(w.model.shape_label)
        robot = {w._mj_body[b] for b in w.scene.robot_bodies if w._mj_body[b] >= 0}
        tools = {w._mj_body[it["body"]] for n, it in w.scene.items.items() if it["kind"] in self.SKIN["tools"]}
        skin = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            sh = int(w._geom_shape[g])
            name = lab[sh] if 0 <= sh < len(lab) else ""
            if any(k in name for k in ("bristles", "nap", "core", "roller_cover")):
                continue
            skin[g] = b in robot or b in tools
        self._skin_geom = skin

    def _skin_step(self, bq):
        """Robot links and non-applicator tool parts touching wet film carry part of it away and leave part of what
        they carry (a film split at separation, SKIN['split'], over a ~4 mm footprint around each MuJoCo contact); the
        touched wet cells are marked for the evaluator (`untouched`), so an elbow cannot rub across a wet strip with no
        mark."""
        if not hasattr(self, "_skin_geom"):
            self._skin_setup()
        d = self.w.mjd
        n = int(d.ncon)
        if n == 0:
            return
        con = d.contact
        g1, g2 = np.asarray(con.geom1[:n], int), np.asarray(con.geom2[:n], int)
        s1, s2 = self._skin_geom[g1], self._skin_geom[g2]
        one = s1 ^ s2
        if not np.any(one):
            return
        idx = np.nonzero(one)[0]
        pts = np.asarray(con.pos[:n], float)[idx]
        gs = np.where(s1[idx], g1[idx], g2[idx])
        bodies = np.asarray(self.w.mjm.geom_bodyid, int)[gs]
        carrying = any(float(sk["solids"].sum() + sk["vehicle"]) > 1e-15 for sk in self.skins.values())
        for ch in self.charts.values():
            if ch.bbox is None and not carrying:
                continue
            su, sv, hh = ch.local(pts, bq)
            ok = (np.abs(hh) < 3e-3) & (su >= 0) & (su < ch.size[0]) & (sv >= 0) & (sv < ch.size[1])
            for k in np.nonzero(ok)[0]:
                self.touch(int(bodies[k]), ch, float(su[k]), float(sv[k]))

    def touch(self, body, ch, su, sv):
        """A skin body pressed on chart ch at (su, sv): split the wet film over the footprint with the body's skin."""
        S = self.SKIN
        k = max(1, int(math.ceil(S["r"]/ch.dx)))
        g = np.arange(-k, k + 1)
        DI, DJ = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
        keep = (DI**2 + DJ**2)*ch.dx**2 <= S["r"]**2 + 1e-18
        I = int(su//ch.dx) + DI[keep]; J = int(sv//ch.dx) + DJ[keep]
        inb = (I >= 0) & (I < ch.nu) & (J >= 0) & (J < ch.nv)
        I, J = I[inb], J[inb]
        I, J = I[ch.mask[I, J]], J[ch.mask[I, J]]
        if I.size == 0:
            return
        sk = self.skins.setdefault(body, dict(solids=np.zeros(len(self.paints)), vehicle=0.0))
        give_s, give_v = sk["solids"]*S["split"], sk["vehicle"]*S["split"]
        wet = (ch.w[I, J] + ch.s[:, I, J].sum(axis=0)) > S["wet_min"]
        took_s, took_v = np.zeros(len(self.paints)), 0.0
        if np.any(wet):
            ch.touched[I[wet], J[wet]] = 1
            ch.marred[I[wet], J[wet]] = 1
            took_s, took_v = ch.take_frac(I[wet], J[wet], S["split"])
        if give_s.sum() + give_v > 1e-15:
            open_ = np.ones(I.size, bool) if ch.flow_block is None else ~ch.flow_block[I, J]
            if np.any(open_):
                nI = int(open_.sum())
                ch.add(I[open_], J[open_], np.repeat(give_s[:, None], nI, axis=1)/nI/ch.area,
                       np.full(nI, give_v/nI/ch.area), shear=True, now=self.w.sim_time)
                sk["solids"] = sk["solids"] - give_s
                sk["vehicle"] = sk["vehicle"] - give_v
        sk["solids"] = sk["solids"] + took_s
        sk["vehicle"] = sk["vehicle"] + took_v

    def films_active(self):
        """Any film still moving (set by Chart.advance: structure recovering or loaded above its yield stress), or
        fresh paint laid since the last film step."""
        return any((getattr(ch, "mobile", False) and ch.bbox is not None) or ch.version != getattr(ch, "_v_seen", -1)
                   for ch in self.charts.values())

    def _stencil_block(self, bq):
        """Local ideal-contact seal on the flat rigid substrate, not projected overlap.

        Only sheet material with underside clearance <= 20 um is treated as sealed.
        This is an explicit ideal smooth-contact approximation (not adhesive or a
        measured plywood seal). Resolved lifted cells remain open to film and spray.
        Tilt is evaluated cell by cell, including at each aperture edge. Sub-grid
        roughness/seepage at nominal contact is not resolved.
        """
        cr, st = self.charts.get("crate"), self.charts.get("stencil")
        if cr is None or st is None:
            return
        o_c, _, _, n_c = cr.frame(bq)
        o_s, u_s, v_s, n_s = st.frame(bq)
        key = tuple(np.round(np.r_[(o_s - o_c)/1e-6, u_s/1e-6, n_s/1e-6]).astype(int))
        if getattr(self, "_block_key", None) == key:
            return
        self._block_key = key
        if float(n_s @ n_c) <= 0:
            cr.flow_block = None
            return
        xs = (np.arange(cr.nu) + 0.5)*cr.dx
        ys = (np.arange(cr.nv) + 0.5)*cr.dx
        U, V = np.meshgrid(xs, ys, indexing="ij")
        su, sv, hh = st.local(cr.world(U.ravel(), V.ravel(), bq), bq)
        gap = -hh - P.STENCIL["thick"]
        on = (su >= 0) & (su <= st.size[0]) & (sv >= 0) & (sv <= st.size[1])
        hole = self._stencil_holes(np.stack([su - st.size[0]/2, sv - st.size[1]/2], 1))
        # Deep interpenetration is not a sealing condition either.
        contact = (gap >= -P.STENCIL["seal_tolerance"]) & (gap <= P.STENCIL["seal_tolerance"])
        cr.flow_block = (on & ~hole & contact).reshape(cr.nu, cr.nv)

    def advance_films(self, dt, bq, now):
        self._stencil_block(bq)
        for ch in self.charts.values():
            info = ch.advance(dt, bq, now)
            ch._v_seen = ch.version
            self.film_info["calls"] += 1
            self.film_info["substeps"] += info.get("substeps", 0)
            self.film_info["limited"] += info.get("limited", 0)
            if ch.drained.any() and ch.name == "tray_ramp":
                res = self.reservoirs.get("tray")
                if res is not None:
                    res.add(ch.drained[:-1], ch.drained[-1])
                    self._ramp_returned = getattr(self, "_ramp_returned", np.zeros(len(self.paints) + 1)) + ch.drained
                    ch.drained[:] = 0.0

    def _lump(self, bq):
        """Liquid mass into the bodies that carry it (containers' pools; applicator loads are lumped by applicators)."""
        for r in self.reservoirs.values():
            if r.body in self.w._mj_body:
                self.w.set_lumped_mass(r.body, r.mass(), r.centroid_local(bq))
        for a in self.applicators.values():
            a.lump(bq)

    # ------------------------------------------------------------------ broadphase
    def _index_prims(self):
        """World AABBs of the static primitives (vectorised culling) and the list of primitives on bodies."""
        from .film import _quat_rot
        lo, hi, idx = [], [], []
        self._dyn = []
        for k, pr in enumerate(self.prims):
            if pr.body >= 0:
                self._dyn.append(k)
                continue
            if isinstance(pr, C.Plane):
                lo.append((-1e3, -1e3, -1e3)); hi.append((1e3, 1e3, 0.0)); idx.append(k)
                continue
            if isinstance(pr, C.Box):
                ext = np.abs(pr.Rl) @ pr.h
            else:
                ext = np.abs(pr.Rl) @ np.array([pr.r, pr.r, pr.hh])
            lo.append(pr.c - ext); hi.append(pr.c + ext); idx.append(k)
        self._slo, self._shi, self._sidx = np.array(lo), np.array(hi), np.array(idx, int)
        self._cand_cache = {}

    def candidates(self, key, centre, radius, bq):
        """Primitives within `radius` of `centre`, cached per caller while it moves less than 10 mm (computed with a
        20 mm margin) and refreshed every 50 ticks for primitives on moving bodies."""
        if not hasattr(self, "_slo"):
            self._index_prims()
        c = np.asarray(centre, float)
        hit = self._cand_cache.get(key)
        # reuse only while the new query sphere lies inside the cached one (a spray cloud grows ~1 cm per tick: a
        # centre-only test returned stale lists without the box walls or the stencil, and parcels flew through them)
        if hit is not None and np.linalg.norm(c - hit[0]) + radius <= hit[3] and self.w.tick - hit[2] < 50:
            return hit[1]
        r = radius + 0.02
        d = np.linalg.norm(np.maximum(np.maximum(self._slo - c, c - self._shi), 0.0), axis=1)
        out = [self.prims[k] for k in self._sidx[d <= r]]
        for k in self._dyn:
            pr = self.prims[k]
            pc, _ = pr.frame(bq)
            if np.linalg.norm(pc - c) > pr.radius + r:
                continue
            holes = getattr(pr, "holes", None)
            if holes is not None:
                pr.holes = None
            dd, _ = pr.sdf(c[None], bq)
            if holes is not None:
                pr.holes = holes
            if dd[0] < r:
                out.append(pr)
        self._cand_cache[key] = (c, out, self.w.tick, r)
        return out

    # ------------------------------------------------------------------ routing paint to a surface point
    def deposit_at(self, prim, points, solids, vehicle, bq, texture=None, shear=True, now=0.0, normal=None, kernel=None):
        """Put liquid (absolute volumes, m^3: solids (P, N), vehicle (N)) at world points on a primitive's surface:
        into the chart of that face at each point if one covers it, else into the primitive's sink. kernel (m): spread
        each point's liquid as a Gaussian splat (sigma kernel/2, cut at kernel) over the open cells around it (spray
        parcels, each standing for ~2e4 droplets)."""
        points = np.atleast_2d(points)
        solids = np.asarray(solids, float).reshape(len(self.paints), -1)
        vehicle = np.asarray(vehicle, float).ravel()
        left = np.ones(len(points), bool)
        for cn in prim.charts:
            ch = self.charts.get(cn)
            if ch is None:
                continue
            su, sv, hh = ch.local(points, bq)
            i, j, ok = ch.cell_index(su, sv)
            ok &= left & (np.abs(hh) < 0.004)
            if normal is not None:
                _, _, _, n = ch.frame(bq)
                ok &= (np.asarray(normal) @ n) > 0.7
            if np.any(ok):
                if kernel:
                    done = self._splat(ch, su[ok], sv[ok], solids[:, ok], vehicle[ok], kernel, texture, shear, now)
                    idx = np.nonzero(ok)[0]
                    ok = np.zeros_like(ok); ok[idx[done]] = True
                else:
                    ch.add(i[ok], j[ok], solids[:, ok]/ch.area, vehicle[ok]/ch.area, texture=texture, shear=shear, now=now)
                left &= ~ok
        if np.any(left):
            self.to_sink(prim.sink, solids[:, left].sum(axis=1), float(vehicle[left].sum()), points=points[left])

    def _splat(self, ch, su, sv, solids, vehicle, r, texture, shear, now):
        """Gaussian splats (sigma r/2, cut at r) of each point's liquid over the chart's open cells: cells under a sheet
        lying on the chart (a stencil's flow block) and masked cells take none; a point with no open cell in reach
        keeps its own cell."""
        k = max(1, int(math.ceil(r/ch.dx)))
        g = np.arange(-k, k + 1)
        DI, DJ = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
        d2 = (DI**2 + DJ**2)*ch.dx**2
        keep = d2 <= r*r + 1e-18
        DI, DJ, d2 = DI[keep], DJ[keep], d2[keep]
        wgt = np.exp(-d2/(2*(r/2)**2))
        i0 = np.floor(su/ch.dx).astype(int); j0 = np.floor(sv/ch.dx).astype(int)
        I = i0[:, None] + DI[None]; J = j0[:, None] + DJ[None]
        inb = (I >= 0) & (I < ch.nu) & (J >= 0) & (J < ch.nv)
        Ic, Jc = np.clip(I, 0, ch.nu - 1), np.clip(J, 0, ch.nv - 1)
        open_ = inb & ch.mask[Ic, Jc]
        if ch.flow_block is not None:
            open_ &= ~ch.flow_block[Ic, Jc]
        W = np.where(open_, wgt[None], 0.0)
        tot = W.sum(axis=1)
        none = tot <= 0
        if np.any(none):                                   # nothing open in reach: the point's own cell
            c = int(np.nonzero((DI == 0) & (DJ == 0))[0][0])
            ownok = open_[none, c]
            W[np.nonzero(none)[0][ownok], c] = 1.0
            tot = W.sum(axis=1)
        W = W/np.maximum(tot, 1e-30)[:, None]
        sel = W > 0
        pi = np.nonzero(sel)[0]
        if pi.size:
            ch.add(Ic[sel], Jc[sel], solids[:, pi]*W[sel][None]/ch.area, vehicle[pi]*W[sel]/ch.area, texture=texture,
                   shear=shear, now=now)
        return tot > 0                                     # points that found no open cell are left to the caller

    SINK_VOXEL = 4e-3       # m: painted area on a surface without a chart is counted on this grid

    def to_sink(self, name, solids, vehicle, area=0.0, points=None):
        """Paint on a surface without a film chart. A container's own walls return it to the container's pool
        (wiping a brush on the pail, a roller in the tray well).

        The painted area is the union of 4 mm voxels that received >= 1 um of dry film equivalent (solids volume per
        voxel face); `points` are where the paint went, so repeated brush passes over the same radiator columns count
        once. Without points
        (airborne mist, chart-edge run-off) no area is recorded."""
        res = self.reservoirs.get(name) or next((r for n, r in self.reservoirs.items() if n.startswith(f"{name}_well")), None)
        if res is not None:
            res.add(np.asarray(solids, float), float(vehicle))
            return
        s = self.sinks.setdefault(name, dict(solids=np.zeros(len(self.paints)), vehicle=0.0, events=0, area=0.0, vox={}))
        s.setdefault("vox", {})
        s["solids"] += solids
        s["vehicle"] += vehicle
        s["events"] += 1
        if points is not None and len(points):
            pts = np.atleast_2d(np.asarray(points, float))
            per = float(np.sum(solids))/len(pts)
            vx = s["vox"]
            for key in map(tuple, np.floor(pts/self.SINK_VOXEL).astype(int)):
                vx[key] = vx.get(key, 0.0) + per
            s["area"] = sum(1 for v in vx.values() if v >= 1e-6*self.SINK_VOXEL**2)*self.SINK_VOXEL**2

    def chart_at(self, prim, points, bq, normal=None):
        """(chart, i, j, ok) for the first chart of the primitive covering each point (single chart per call)."""
        for cn in prim.charts:
            ch = self.charts.get(cn)
            if ch is None:
                continue
            su, sv, hh = ch.local(points, bq)
            i, j, ok = ch.cell_index(su, sv)
            ok &= np.abs(hh) < 0.006
            if normal is not None:
                _, _, _, n = ch.frame(bq)
                ok &= (np.asarray(normal) @ n) > 0.7
            if np.any(ok):
                return ch, i, j, ok
        return None, None, None, None

    # ------------------------------------------------------------------ ledger
    def totals(self):
        """System totals (m^3): solids per paint and vehicle (wet + absorbed + evaporated) over every holder."""
        Pn = len(self.paints)
        solids, vehicle = np.zeros(Pn), 0.0
        parts = {}
        for n, ch in self.charts.items():
            t = ch.totals()
            s = t["wet_solids"] + t["dry_solids"] + t["drained_solids"]
            v = t["wet_vehicle"] + t["absorbed"] + t["evaporated"] + t["drained_vehicle"]
            parts[f"chart:{n}"] = (s, v)
        for n, r in self.reservoirs.items():
            parts[f"res:{n}"] = (r.solids.copy(), r.vehicle)
        for n, a in self.applicators.items():
            parts[f"app:{n}"] = a.load()
        if self.drips is not None:
            parts["airborne"] = self.drips.load()
        for n, s in self.sinks.items():
            parts[f"sink:{n}"] = (s["solids"].copy(), s["vehicle"])
        for b, sk in getattr(self, "skins", {}).items():
            parts[f"skin:{b}"] = (sk["solids"].copy(), sk["vehicle"])
        for s, v in parts.values():
            solids = solids + np.asarray(s, float)
            vehicle += float(v)
        return dict(solids=solids, vehicle=vehicle, parts=parts)

    def ledger_residual(self):
        a, b = self.initial_totals, self.totals()
        scale = max(1e-12, float(a["solids"].sum() + a["vehicle"]))
        return dict(solids=(b["solids"] - a["solids"]).tolist(), vehicle=b["vehicle"] - a["vehicle"],
                    relative=float((np.abs(b["solids"] - a["solids"]).sum() + abs(b["vehicle"] - a["vehicle"]))/scale))

    # ------------------------------------------------------------------ checkpoint
    def snapshot(self):
        return dict(charts={n: c.snapshot() for n, c in self.charts.items()},
                    reservoirs={n: r.snapshot() for n, r in self.reservoirs.items()},
                    applicators={n: a.snapshot() for n, a in self.applicators.items()},
                    drips=self.drips.snapshot(), sinks={n: dict(solids=s["solids"].copy(), vehicle=s["vehicle"], events=s["events"],
                                                                 area=s["area"], vox=dict(s.get("vox", {})))
                                                        for n, s in self.sinks.items()},
                    skins={b: dict(solids=sk["solids"].copy(), vehicle=sk["vehicle"]) for b, sk in self.skins.items()},
                    tick_film=self.tick_film, ramp=getattr(self, "_ramp_returned", None),
                    lumped=dict(self.w._extra_mass))

    def restore(self, snap):
        for n, c in self.charts.items():
            c.restore(snap["charts"][n])
        for n, r in self.reservoirs.items():
            r.restore(snap["reservoirs"][n])
        for n, a in self.applicators.items():
            a.restore(snap["applicators"][n])
        self.drips.restore(snap["drips"])
        self.sinks = {n: dict(solids=s["solids"].copy(), vehicle=s["vehicle"], events=s["events"], area=s["area"],
                              vox=dict(s.get("vox", {}))) for n, s in snap["sinks"].items()}
        self.skins = {b: dict(solids=sk["solids"].copy(), vehicle=sk["vehicle"]) for b, sk in snap.get("skins", {}).items()}
        self.tick_film = snap["tick_film"]
        if snap["ramp"] is not None:
            self._ramp_returned = snap["ramp"].copy()
        elif hasattr(self, "_ramp_returned"):
            del self._ramp_returned
        self._lump(self.w.state.body_q.numpy())

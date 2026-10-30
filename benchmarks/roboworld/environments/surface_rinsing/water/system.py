"""The water system of one episode: surfaces and their films, outlets and valves, packets, pools, species,
soil, friction and the exact ledger, coupled to the rigid world once per 0.5 ms tick (world.RinsingWorld.step):

    pre_step():  valve and balancer springs, jet reactions, impact impulses (last tick), buoyancy and drag, film wall
                 shear into world.body_wrench; finger-pad friction scheduled from the soap state under each pad contact
    post_step(): body poses to the GPU; outlets (valve openings -> flow -> jets); packets (flight, hits, splatter, pool
                 capture); films every `stride` ticks with CFL substeps (transport, drips, lip/edge drops, species, soil,
                 submerged cells, drains); pools (inflow, drains, holes, overflow, flooding); lumped water masses
Nothing here is reachable from the policy interface: water only flows because a lever, blade or drain lever is moved by
contact, and it only moves things by the forces it computes.
"""

import math
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import warp as wp

from .. import params as P
from .. import layout as L
from .surfaces import SurfaceSet
from .engine import WaterEngine, NS, f64
from .network import Mixer, Outlet, SprayValve, Balancer
from .pools import Pool, PoolSet

RHO = P.WATER["rho"]


@wp.kernel
def k_gather(idx: wp.array(dtype=int), h: wp.array(dtype=f64), sp: wp.array2d(dtype=f64), gamma: wp.array(dtype=f64),
             out: wp.array2d(dtype=float)):
    i = wp.tid()
    c = idx[i]
    out[i, 0] = float(h[c])
    out[i, 1] = float(sp[0, c] + sp[5, c])
    out[i, 2] = float(gamma[c])
    out[i, 3] = float(sp[4, c])


@wp.kernel
def k_leak(idx: wp.array(dtype=int), frac: float, h_keep: float, h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3),
           sp: wp.array2d(dtype=f64), area: wp.array(dtype=float), out: wp.array(dtype=f64)):
    """Film on a perforated sheet runs into its holes: a fraction of the film above h_keep (with its dissolved species)
    leaves the cells into out[0 : 1 + 4] (m3, then species)."""
    i = wp.tid()
    c = idx[i]
    hc = h[c]
    if hc <= f64(h_keep):
        return
    A = f64(area[c])
    fr = f64(frac)*(hc - f64(h_keep))/hc
    h[c] = hc - fr*hc
    q[c] = q[c]*(1.0 - float(fr))
    wp.atomic_add(out, 0, fr*hc*A)
    for k in range(4):
        ds = sp[k, c]*fr
        sp[k, c] = sp[k, c] - ds
        wp.atomic_add(out, 1 + k, ds*A)


@wp.kernel
def k_sub_count(cell_body: wp.array(dtype=int), sub: wp.array(dtype=int), n_pool: int, out: wp.array2d(dtype=int)):
    c = wp.tid()
    b = cell_body[c]
    k = sub[c]
    if b >= 0 and k >= 0:
        wp.atomic_add(out, b, k, 1)


def shape_mesh(builder, i, newton):
    """Triangle mesh (V, F) of a builder shape in its body frame (static shapes: world frame)."""
    G = newton.GeoType
    t = int(builder.shape_type[i])
    sc = np.asarray(builder.shape_scale[i], float)
    if t == G.BOX:
        m = newton.Mesh.create_box(sc[0], sc[1], sc[2], compute_normals=False, compute_uvs=False, compute_inertia=False)
    elif t == G.SPHERE:
        m = newton.Mesh.create_sphere(sc[0], num_latitudes=10, num_longitudes=12, compute_normals=False, compute_uvs=False,
                                      compute_inertia=False)
    elif t == G.CAPSULE:
        m = newton.Mesh.create_capsule(sc[0], sc[1], up_axis=newton.Axis.Z, segments=12, compute_normals=False,
                                       compute_uvs=False, compute_inertia=False)
    elif t == G.CYLINDER:
        m = newton.Mesh.create_cylinder(sc[0], sc[1], up_axis=newton.Axis.Z, segments=16, compute_normals=False,
                                        compute_uvs=False, compute_inertia=False)
    elif t in (G.MESH, G.CONVEX_MESH):
        src = builder.shape_source[i]
        if src is None:
            return None
        V = np.asarray(src.vertices, float)*sc
        F = np.asarray(src.indices, np.int32).reshape(-1, 3)
        tf = np.asarray(builder.shape_transform[i], float)
        R = Rotation.from_quat(tf[3:]).as_matrix()
        return V @ R.T + tf[:3], F
    else:
        return None
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.indices, np.int32).reshape(-1, 3)
    tf = np.asarray(builder.shape_transform[i], float)
    R = Rotation.from_quat(tf[3:]).as_matrix()
    return V @ R.T + tf[:3], F


class WaterSystem:
    def __init__(self, world, spec, device="cuda:0"):
        t0 = time.perf_counter()
        self.w, self.spec, self.device = world, spec, device
        sc = world.scene
        self.station = spec["station"]
        self.supply = dict(spec["supply"])
        self.timing = dict(pre_s=0.0, post_s=0.0, film_s=0.0, packets_s=0.0, pools_s=0.0)
        self.film_info = dict(steps=0, substeps=0)
        self.tick = 0
        self._film_clock = 0
        self.stride = 1
        self.ratio_max = 0.0
        self.drain_kg = 0.0
        self.drain_sp = np.zeros(NS)
        self.events = []
        # ---------------------------------------------------------------- surfaces
        S = SurfaceSet()
        self.item_charts = {}                    # item name -> chart object
        for name, w_ in sc.ware.items():
            b = sc.items[name]["body"]
            for c in w_.charts:
                if c["type"] == "revolved":
                    ch = S.add_revolved(name, b, c["profile"], c["cell"], material=c["material"], lips=c["lips"])
                elif c["type"] == "ellipsoid":
                    ch = S.add_ellipsoid(name, b, c["centre"], c["radii"], c["cell"], material=c["material"])
                else:
                    ch = S.add_rect(name, b, c["origin"], c["u"], c["v"], c["size"], c["cell"], material=c["material"],
                                    open_edges=c["open_edges"])
                ch.item = name
                self.item_charts[name] = ch
        self.static_charts = {}
        for (name, origin, u, v, size, cell, mat, open_edges, role, extra) in L.static_charts(self.station):
            ch = S.add_rect(name, -1, origin, u, v, size, cell, role=role, material=mat, open_edges=open_edges)
            ch.extra = extra
            self.static_charts[name] = ch
        S.finish()
        self.check = S.check()
        self.surf = S
        # the bay floor slopes 1/8 in per ft to its drain: tilt the cells' normals (the rigid floor stays flat; 1 cm over
        # the 2 m bay)
        for name, ch in self.static_charts.items():
            if "drain" in ch.extra:
                xd, yd, rd = ch.extra["drain"]
                s = ch.extra["slope"]
                sl = slice(ch.first, ch.first + ch.count)
                d = S.pos[sl, :2] - (xd, yd)
                r = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
                n = np.concatenate([-s*d/r, np.ones((ch.count, 1))], 1)
                S.nrm[sl] = n/np.linalg.norm(n, axis=1, keepdims=True)
        self.eng = E = WaterEngine(S, world.model.body_count, device=device, room_lo=(-0.5, -0.5, -0.2),
                                   room_hi=(L.X + 0.5, L.Y + 0.5, L.Z + 0.2))
        if any("drain" in ch.extra for ch in self.static_charts.values()):
            E.cell_nrm.assign(S.nrm.astype(np.float32))
        # ---------------------------------------------------------------- pools
        self.pools = PoolSet(self)
        for name, d in L.static_pools(self.station).items():
            p = self.pools.add(Pool(name, "static", lo=d["lo"], hi=d["hi"], drain_d=d["drain_d"], drain_xy=d["drain_xy"]))
            p.drain_open = 1.0 if spec["drains"].get(name) else 0.0
        for name, w_ in sc.ware.items():
            if w_.cavity is not None:
                self.pools.add(Pool(f"cavity:{name}", "cavity", body=sc.items[name]["body"], ware=w_, item=name))
        from .sponge import SpongeModel
        self.sponges = {}
        for name, it in sc.items.items():
            if it["kind"] == "sponge":
                sm = SpongeModel(self, name)
                self.pools.add(sm.pool)
                self.sponges[name] = sm
        E.set_pools(self.pools.gpu_specs())
        # cells of a cavity's inside belong to its pool
        own = np.full(S.n, -1, np.int32)
        for name, ch in self.item_charts.items():
            k = self.pools.index.get(f"cavity:{name}")
            if k is None:
                continue
            inner = sc.ware[name].regions.get(sc.ware[name].kind, {}).get("inner", [])
            cells = np.arange(ch.first, ch.first + ch.count)
            own[cells[np.isin(ch.cell_seg, inner)]] = k
        E.own_pool.assign(own)
        # floor-drain strainer cells
        drain = np.full(S.n, -1, np.int32)
        for name, ch in self.static_charts.items():
            if "drain" in ch.extra:
                xd, yd, rd = ch.extra["drain"]
                sl = np.arange(ch.first, ch.first + ch.count)
                m = np.hypot(S.pos[sl, 0] - xd, S.pos[sl, 1] - yd) < rd
                drain[sl[m]] = E.sink_index("drain")
        E.drain.assign(drain)
        # ---------------------------------------------------------------- sink meshes
        self._sink_meshes()
        # ---------------------------------------------------------------- outlets and valves
        self.outlets, self.valves, self.balancers = {}, {}, {}
        fx = sc.fixtures
        mixer = Mixer(world, fx, self.supply) if ("hot" in fx or "cold" in fx) else None
        self.mixer = mixer
        F = P.FAUCET
        dead, turn = math.radians(F["dead_deg"]), math.radians(F["turn_deg"])
        for name, f in fx.items():
            if f["kind"] == "spray_valve":
                sv = SprayValve(world, f)
                self.valves[name] = sv
                line = (P.LINE["riser_L"], P.LINE["riser_D"]) if f.get("hose") is None else (f["hose"]["L"], f["hose"]["D"])
                mix = mixer if mixer is not None else _AlwaysOn(self.supply)
                self.outlets[name] = Outlet(name, world, f["valve"], f["face"], f["direction"], "spray", f["spec"], mix,
                                            gate=sv.opening, line=line)
                if f.get("balancer") is not None:
                    self.balancers[name] = Balancer(world, f)
            elif f["kind"] == "spout":
                gate = None
                if "faucet_valve" in fx:
                    blade = fx["faucet_valve"]
                    gate = (lambda b=blade: min(1.0, max(0.0, (world.joint_value(b["joint"])[0] - dead)/(turn - dead))))
                self.outlets[name] = Outlet(name, world, f["body"], f["outlet"], f["direction"], "stream", None, mixer,
                                            gate=gate, line=(P.LINE["faucet_L"], P.LINE["faucet_D"]))
            elif f["kind"] == "gooseneck":
                self.outlets[name] = Outlet(name, world, -1, f["outlet"], f["direction"], "stream", None, mixer,
                                            line=(P.LINE["faucet_L"], P.LINE["faucet_D"]))
        self.wastes = {f["pool"]: f for f in fx.values() if f["kind"] == "waste"}
        # ---------------------------------------------------------------- per-body data
        self.com_local = world.model.body_com.numpy().astype(float)
        self.base_mass = {}
        self.item_cells = {name: np.arange(ch.first, ch.first + ch.count) for name, ch in self.item_charts.items()}
        self.trees = {sc.items[name]["body"]: (cKDTree(S.pos[cells]), cells) for name, cells in self.item_cells.items()}
        # footprints of items standing on the room's charted surfaces (floor, drainboards, tables): no film runs under
        # them (see _cover). The footprint is the item's standing ring: its solid profile within 2.5 mm of the standing
        # plane. Perforated ware (water falls through it) and round produce are left out.
        self.cover_items = []
        for name, w_ in sc.ware.items():
            if w_.solid is None or w_.kind in ("colander", "tomato", "sponge") or name not in sc.items:
                continue
            prof = np.asarray(w_.solid, float)
            foot = prof[prof[:, 1] <= 0.0025]
            if len(foot):
                self.cover_items.append((sc.items[name]["body"], float(foot[:, 0].max())))
        cc = [np.arange(ch.first, ch.first + ch.count) for ch in self.static_charts.values()]
        self.cover_cells = np.concatenate(cc) if cc else np.zeros(0, int)
        self.cover_tree = cKDTree(S.pos[self.cover_cells][:, :2]) if len(self.cover_cells) else None
        self.cover_z = S.pos[self.cover_cells][:, 2] if len(self.cover_cells) else np.zeros(0)
        # perforated sheets: the inner cells within one pitch of a hole centre drain into the cavity (see _leak)
        self.leaky = {}
        for name, w_ in sc.ware.items():
            holes = (w_.cavity or {}).get("holes") if getattr(w_, "cavity", None) else None
            k = self.pools.index.get(f"cavity:{name}")
            if holes is None or k is None or name not in self.item_cells:
                continue
            cells = self.item_cells[name]
            ch = self.item_charts[name]
            inner = np.isin(ch.cell_seg, w_.regions[w_.kind]["inner"])
            d, _ = cKDTree(np.asarray(holes)[:, :3]).query(S.pos[cells])
            sel = cells[inner & (d < P.ITEMS[w_.kind]["pitch"])]
            if len(sel):
                self.leaky[name] = (wp.array(sel.astype(np.int32), dtype=int, device=self.eng.device), self.pools.pools[k])
        self.sample_bodies = {sc.items[n]["body"]: w_ for n, w_ in sc.ware.items() if w_.samples is not None}
        self.forces = np.zeros((world.model.body_count, 6))
        self.reaction = np.zeros((world.model.body_count, 6))
        self.impact = np.zeros((world.model.body_count, 6))
        self.shear = np.zeros((world.model.body_count, 6))
        self.lumped = {}
        self.sanitize_s = {n: 0.0 for n in sc.items}
        self.immersed = {}
        self.pad_mu = {}
        self.pad_geoms = {int(g): s for g, s in zip(world.pad_geoms, world.scene.pad_shapes)}
        self.mu_pad_dry = float(world.mjm.geom_friction[world.pad_geoms[0], 0]) if len(world.pad_geoms) else 0.8
        self._mj_item = {world._mj_body[b]: b for b in world.item_bodies.values() if world._mj_body.get(b, -1) >= 0}
        self.max_impact = {}
        self.pending = []
        self._conc = None
        self._initial_state()
        self.build_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ construction helpers
    def _sink_meshes(self):
        """Every collider that carries no chart has a named account: the robot's links, moving fixture parts ("tools":
        hoses, valves, levers, blades, the swing spout), racks, the room's static parts ("fixtures"; sink-bowl walls
        route to their bowl's pool), the dunnage shelf, the floor (just below the floor charts). Water striking one
        either stays in the account or runs off as a drop (engine.set_sink_modes below)."""
        w, sc, E = self.w, self.w.scene, self.eng
        b = sc.b
        newton = w.newton
        chart_items = {sc.items[n]["body"] for n in self.item_charts}
        charted_static = {"drainboard_soiled", "drainboard_clean", "v_drainboard", "table_top_s", "table_top_n", "clean_table"}
        charted_static = {n for n in charted_static if n in self.static_charts or n.replace("table_top_", "table_") in self.static_charts}
        per_body = {}
        robot = set(sc.robot_bodies)
        sponge_bodies = {}
        for name, sm in self.sponges.items():
            k = self.pools.index[sm.pool.name]
            sponge_bodies[sm.base] = k
            sponge_bodies[sm.top] = k
        for i in range(b.shape_count):
            body = b.shape_body[i]
            if not (int(b.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            label = b.shape_label[i]
            if body in chart_items:
                continue
            if body < 0:
                name = label.split("/")[-1]
                if name == "floor" or name in charted_static:
                    continue
                role = sc.static_role.get(i, "")
                if role.startswith("pool:") and role[5:] in self.pools.index:
                    key = ("pool", self.pools.index[role[5:]])
                elif name.startswith(("shelf_", "shelf_post")):
                    key = ("sink", "shelf")                    # the dunnage shelf: clean storage (contamination account)
                else:
                    key = ("sink", "fixtures")
            elif body in robot:
                key = ("sink", "robot")
            elif body in sc.fixture_bodies:
                key = ("sink", "tools")                        # hoses, valves, levers, blades, the swing spout
            elif body in sponge_bodies:
                key = ("pool", sponge_bodies[body])            # water striking a sponge's faces enters its pores
            else:
                key = ("sink", "racks")
            mesh = shape_mesh(b, i, newton)
            if mesh is None:
                continue
            per_body.setdefault((body, key), []).append(mesh)
        for (body, key), meshes in per_body.items():
            V, F, off = [], [], 0
            for v, f in meshes:
                V.append(v); F.append(f + off); off += len(v)
            V, F = np.concatenate(V), np.concatenate(F)
            if key[0] == "pool":
                E.add_pool_wall_mesh(body, V, F, key[1])
            else:
                E.add_sink_mesh(body, V, F, key[1])
        # floor below the floor charts (water reaching the floor elsewhere is counted there)
        zc = -0.0015            # = engine.FLOOR_CATCH_Z, where the packet kernel also keeps any segment crossing it unstruck
        Vf = np.array([[-1, -1, zc], [L.X + 1, -1, zc], [L.X + 1, L.Y + 1, zc], [-1, L.Y + 1, zc]], float)
        E.add_sink_mesh(-1, Vf, np.array([[0, 1, 2], [0, 2, 3]]), "floor")
        # what runs off: everything from thin moving parts and rack wires; from the robot and the fixtures what strikes
        # a steep or overhanging face (their ledges and tops keep it); the shelf and the floor keep everything
        E.set_sink_modes({"tools": 1, "racks": 1, "robot": 2, "fixtures": 2, "shelf": 0, "floor": 0})
        E.finalize_meshes()

    def _initial_state(self):
        """Wet soapy films with foam patches on washed ware, tomato-paste soil on pre-rinse plates, dust on produce,
        debris and soap in the can, soapy water on the bay floor, and the sink bowls' fills."""
        S, E, sc, sp = self.surf, self.eng, self.w.scene, self.spec
        N = S.n
        h = np.zeros(N); spc = np.zeros((NS, N)); gamma = np.zeros(N)
        soil = np.zeros(N); soil_tr = np.zeros(N, np.float32); sd = np.zeros(N, np.float32); sw = np.zeros(N, np.float32)
        ts = np.ones(N, np.float32)
        w0 = np.zeros(N, np.float32)                     # soil already soaked before the episode (the pan in the wash water)
        rng = np.random.default_rng(1009 + sp["seed"])
        gm = E.gamma_m.numpy().astype(float)
        Ka = P.SURFACTANT["K"]
        for name, it in sp["items"].items():
            if name not in self.item_cells:
                continue
            cells = self.item_cells[name]
            wet = it.get("wet")
            if wet:
                c = wet["surf"]
                h[cells] = wet["h"]
                spc[0, cells] = c*wet["h"]
                spc[3, cells] = wet["h"]*wet["T"]
                gamma[cells] = gm[cells]*Ka*c/(1 + Ka*c)
                # foam patches: 2-5 mm of foam (5 % liquid) over discs of 15-30 mm
                pos = S.pos[cells]
                for k in range(int(wet.get("foam_patches", 0))):
                    centre = pos[rng.integers(len(cells))]
                    r = rng.uniform(0.015, 0.030)
                    m = np.linalg.norm(pos - centre, axis=1) < r
                    fth = rng.uniform(0.002, 0.005)
                    liq = fth*(1 - P.FOAM["phi"])
                    spc[4, cells[m]] += liq
                    spc[5, cells[m]] += liq*c
            soil_spec = it.get("soil")
            if soil_spec:
                kind = soil_spec["kind"]
                ch = self.item_charts[name]
                if kind == "tomato":
                    # ASTM F2324-03 class soil: 15 mL of tomato paste at 70 % moisture brushed over the 7 in flat of the
                    # face, dried: 4.95 g of solids over 0.0248 m2 -> 0.20 kg/m2 (0.13 mm dry), ragged edge
                    face = np.isin(ch.cell_seg, sc.ware[name].regions[sc.ware[name].kind]["face"])
                    rr = np.hypot(S.pos[cells, 0], S.pos[cells, 1])
                    edge = 0.0889*(1 + 0.06*np.sin(5*np.arctan2(S.pos[cells, 1], S.pos[cells, 0]) + rng.uniform(0, 6)))
                    m = face & (rr < edge)
                    load = 0.20*soil_spec["load"]*(1 + 0.25*rng.standard_normal(m.sum()).clip(-2, 2))
                    soil[cells[m]] = np.maximum(load, 0.02)
                    sd[cells[m]], sw[cells[m]], ts[cells[m]] = P.SOIL["sigma_dry"], P.SOIL["sigma_wet"], P.SOIL["t_soak20"]
                elif kind == "egg":
                    # dried egg over the inside bottom of the pan
                    inner = np.isin(ch.cell_seg, sc.ware[name].regions[sc.ware[name].kind]["inner"])
                    flat = inner & (S.pos[cells, 2] < 0.012)
                    Eg = P.EGG
                    load = Eg["load"]*soil_spec["load"]*(1 + 0.3*rng.standard_normal(flat.sum()).clip(-2, 2))
                    soil[cells[flat]] = np.maximum(load, 0.01)
                    sd[cells[flat]], sw[cells[flat]], ts[cells[flat]] = Eg["sigma_dry"], Eg["sigma_wet"], Eg["t_soak20"]
                    # the pan has been soaking in the wash water for soaked_s before the episode starts
                    t_s = Eg["t_soak20"]*2.0**(-(sp["T_wash"] - 20.0)/P.SOIL["soak_doubling_C"])
                    w0[cells[flat]] = 1.0 - math.exp(-float(soil_spec.get("soaked_s", 0.0))/t_s)
                elif kind in ("dust", "debris"):
                    D = P.DUST
                    if kind == "dust":
                        frac = rng.uniform(0.3, 1.0, len(cells))*(rng.uniform(size=len(cells)) < 0.6)
                        sel = np.ones(len(cells), bool)
                    else:
                        reg = sc.ware[name].regions[sc.ware[name].kind]["inner"]
                        sel = np.isin(ch.cell_seg, reg)
                        frac = rng.uniform(0.2, 1.0, len(cells))*(rng.uniform(size=len(cells)) < 0.5)
                    soil[cells[sel]] = D["load"]*soil_spec["load"]*frac[sel]*2.0
                    soil_tr[cells[sel]] = 1.0
                    sd[cells[sel]], sw[cells[sel]], ts[cells[sel]] = D["sigma_dry"], D["sigma_wet"], D["t_soak20"]
        fs = sp.get("floor_soil")
        if fs is not None:
            for name, ch in self.static_charts.items():
                if "drain" not in ch.extra:
                    continue
                cells = np.arange(ch.first, ch.first + ch.count)
                x0, x1, y0, y1 = fs["region"]
                pos = S.pos[cells]
                blob = (pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 1] > y0) & (pos[:, 1] < y1)
                blob &= rng.uniform(size=len(cells)) < 0.85
                h[cells[blob]] = fs["h"]
                spc[0, cells[blob]] = fs["surf"]*fs["h"]
                spc[3, cells[blob]] = fs["h"]*20.0
                D = P.DUST
                soil[cells[blob]] = D["load"]*4*fs["load"]*rng.uniform(0.2, 1.0, blob.sum())
                soil_tr[cells[blob]] = 1.0
                sd[cells[blob]], sw[cells[blob]], ts[cells[blob]] = D["sigma_dry"], D["sigma_wet"], D["t_soak20"]
        E.h.assign(h); E.sp.assign(spc); E.gamma.assign(gamma); E.soil.assign(soil)
        E.soil_tr.assign(soil_tr); E.soil_sd.assign(sd); E.soil_sw.assign(sw); E.soil_ts.assign(ts)
        E.soil_w.assign(w0)
        E.soil_sig.assign(np.where(soil > 0, sd*(1 - w0)**2 + sw, 0).astype(np.float32))
        E.fl.T_amb = 20.0
        # sponge pores
        for name, it in sp["items"].items():
            if it.get("pores") and name in self.sponges:
                sm = self.sponges[name]
                pr = it["pores"]
                pool = sm.pool
                pool.vol = pr["fill"]*sm.v_full
                pool.sp[:] = 0.0
                pool.sp[0] = pr["surf"]*pool.vol
                pool.sp[3] = pr["T"]*pool.vol
                pool.T_default = pr["T"]
        # pools
        for name, f in sp["fills"].items():
            p = self.pools.pools[self.pools.index[name]]
            p.vol = f["depth"]*p.area
            p.sp[0] = f["surf"]*p.vol
            p.sp[3] = f["T"]*p.vol
            p.sp[4] = f.get("foam", 0.0)*p.area*(1 - P.FOAM["phi"])
            p.sp[5] = p.sp[4]*f["surf"]
            p.chlorine_ppm = f.get("chlorine_ppm", 0.0)
            p.T_default = f["T"]
        for p in self.pools.pools:
            if p.kind == "static":
                p.plane()
        self.initial = self.totals()

    # ------------------------------------------------------------------ coupling
    def pre_step(self):
        t0 = time.perf_counter()
        w = self.w
        wr = w.body_wrench
        for v in self.valves.values():
            v.spring(wr)
        for b in self.balancers.values():
            b.force(wr)
        for sm in self.sponges.values():
            sm.spring(wr)
        wr += (self.reaction + self.impact + self.forces + self.shear).astype(wr.dtype)
        if self.tick % 10 == 0:
            self._pad_friction()
        self.timing["pre_s"] += time.perf_counter() - t0

    def _up(self, body):
        """Upward unit vector of the effective gravity seen by a body (world frame): g minus its filtered acceleration."""
        a = self.eng._acc_f[body] if body >= 0 else np.zeros(3)
        g = np.array([0., 0., -P.GRAVITY]) - a
        return -g/np.linalg.norm(g)

    def post_step(self):
        t0 = time.perf_counter()
        w, E = self.w, self.eng
        dt = P.DT
        bq = w.state.body_q.numpy()
        bqd = w.state.body_qd.numpy()
        E.set_bodies(bq, bqd, self.com_local, dt)
        # outlets
        self.reaction[:] = 0.0
        xs, vs, ms, ds, ks, sps = [], [], [], [], [], []
        p_s = self.supply["p_static"]
        for name, o in self.outlets.items():
            out = o.step(dt, p_s)
            if out is None:
                continue
            x, v, m, d, k, sp, face = out
            xs.append(x); vs.append(v); ms.append(m); ds.append(d); ks.append(k); sps.append(sp)
            if o.body is not None and o.body >= 0:
                com = bq[o.body, :3] + Rotation.from_quat(bq[o.body, 3:]).apply(self.com_local[o.body])
                self.reaction[o.body, :3] += o.reaction
                self.reaction[o.body, 3:] += np.cross(face - com, o.reaction)
        if ms:
            E.emit(np.concatenate(xs), np.concatenate(vs), np.concatenate(ms), np.concatenate(ds), np.concatenate(ks),
                   np.concatenate(sps, axis=1), source=True)
        if self.pending:
            # pool outflows (rims, holes) of the last tick: water already accounted, re-entering as drops
            pend = self.pending
            self.pending = []
            E.emit(np.concatenate([p[0] for p in pend]), np.concatenate([p[1] for p in pend]),
                   np.concatenate([p[2] for p in pend]), np.concatenate([p[3] for p in pend]),
                   np.ones(sum(len(p[2]) for p in pend), np.int32), np.concatenate([p[4] for p in pend], axis=1), source=False)
        # pools: planes for this tick
        tp = time.perf_counter()
        for name, f in self.wastes.items():
            k = self.pools.index.get(name)
            if k is not None:
                q, _ = w.joint_value(f["joint"])
                self.pools.pools[k].drain_open = min(1.0, max(0.0, q/(0.5*math.pi)))
        n_list, d_list, conc, vol, foam = self.pools.update_planes(bq, self._up)
        budget = np.array([[p.vol, p.sp[0], p.sp[4]] for p in self.pools.pools]).T.reshape(3, -1)
        E.update_pool_planes(n_list, d_list, conc, vol, foam, budget)
        self._conc = conc
        self.timing["pools_s"] += time.perf_counter() - tp
        # packets
        tk = time.perf_counter()
        E.body_f.zero_(); E.body_t.zero_()
        E.pool_m.zero_(); E.pool_sp.zero_(); E.pool_imp.zero_()
        E.step_packets(dt)
        self.timing["packets_s"] += time.perf_counter() - tk
        # films
        self._film_clock += 1
        film_step = self._film_clock >= self.stride
        if film_step:
            tf = time.perf_counter()
            dtf = self._film_clock*dt
            self._film_clock = 0
            E.cell_frames()
            n_sub = max(1, int(math.ceil(dtf*self.ratio_max/P.FILM["cfl"])))
            n_sub = min(n_sub, 64)
            for _ in range(n_sub):
                E.film_substep(dtf/n_sub)
            E.emit_edges()
            E.drips()
            E.species_step(dtf)
            E.submerge(dtf)
            E.drains()
            E.body_film(dtf)
            r = E.max_ratio()
            self.film_info["steps"] += 1
            self.film_info["substeps"] += n_sub
            self.timing["film_s"] += time.perf_counter() - tf
        # read back this tick's accumulators (one sync)
        bf, bt = E.body_f.numpy(), E.body_t.numpy()
        self.impact[:, :3] = bf; self.impact[:, 3:] = bt
        npool = len(self.pools.pools)
        pm, psp = E.pool_m.numpy()[:npool], E.pool_sp.numpy()[:, :npool]
        for k, p in enumerate(self.pools.pools):
            if pm[k] != 0.0 or psp[:, k].any():
                p.add(pm[k]/RHO, psp[:, k])
        if film_step:
            self.ratio_max = float(r.numpy()[0])
            out, take = E.pool_out.numpy()[:, :npool], E.pool_foam_take.numpy()[:, :npool]
            for k, p in enumerate(self.pools.pools):
                # out row 0: adsorbed from the pool (surfactant), rows 1-4: species 0-3 carried out in films, row 5: volume
                p.sp[0] -= out[0, k] + out[1, k]
                p.sp[1:4] -= out[2:5, k]
                p.vol -= out[5, k]
                p.sp[4] -= take[0, k]
                p.sp[5] -= take[1, k]
            self._film_bodies(bq, dtf)
            self._choose_stride()
        tp = time.perf_counter()
        if self.sponges and self.tick % 10 == 0:
            for sm in self.sponges.values():
                sm.step(10*dt)
        if self.leaky and self.tick % 10 == 0:
            self._leak(10*dt)
        if self.cover_items and self.cover_tree is not None and self.tick % 10 == 0:
            self._cover(bq)
        self.pools.outflows(dt, bq, bqd, self.com_local, self._up, self._emit_later)
        self.pools.flooding(dt, bq)
        if self.tick % 4 == 0:
            self._buoyancy(bq, bqd)
        self.timing["pools_s"] += time.perf_counter() - tp
        self.tick += 1
        self.timing["post_s"] += time.perf_counter() - t0

    def _emit_later(self, x, v, m, d, sp):
        """Pool outflows (holes, rims) enter the packet ring at the next tick's emission (counted as pending)."""
        self.pending.append((x, v, m, d, sp))

    def to_drain(self, vol, sp):
        self.drain_kg += vol*RHO
        self.drain_sp += sp

    def _choose_stride(self):
        """Films advance every `stride` ticks while the CFL allows a longer step (thin draining films), every tick with
        CFL substeps while water moves fast (impact zones)."""
        r = self.ratio_max
        any_flow = any(o.Q > 0 for o in self.outlets.values())
        if r <= 0:
            self.stride = P.FILM["max_stride"]
        else:
            k = int(P.FILM["cfl"]/(r*P.DT))
            self.stride = max(1, min(P.FILM["max_stride"], k))
        if any_flow:
            self.stride = 1

    def _cover(self, bq):
        """Mark the room-surface cells under the footprint of every upright item standing on them (its standing ring
        within 4 mm of the cell): the film flux into and out of them is zero (engine.k_edge_flux). Water does not run
        under a can, a glass or a plate standing on a floor or a drainboard; it runs round it."""
        blk = np.zeros(self.eng.N, np.int32)
        for b, r in self.cover_items:
            p = bq[b, :3]
            R = Rotation.from_quat(bq[b, 3:]).as_matrix()
            if R[2, 2] < 0.94:                                     # tilted more than 20 deg: not standing on its ring
                continue
            idx = self.cover_tree.query_ball_point(p[:2], r)
            if not idx:
                continue
            idx = np.asarray(idx, int)
            dz = p[2] - self.cover_z[idx]
            blk[self.cover_cells[idx[(dz > -0.003) & (dz < 0.004)]]] = 1
        self.eng.blocked.assign(blk)
        self.covered = int(blk.sum())

    def _leak(self, dt):
        """Perforated ware (the colander): film standing on the perforated sheet runs into the holes (time constant
        0.1 s, the film a 4 mm hole on a 9 mm pitch cannot hold: more than the 60 um that clings) and joins the cavity's
        water, which leaves through the same holes (pools.outflows). (First build: only pooled water used the holes, so
        a drained colander kept a 1 mm sheet of "film" on its bottom and never read as drained; demo A3.)"""
        E = self.eng
        fr = min(0.5, dt/0.1)
        for name, (ia, pool) in self.leaky.items():
            out = wp.zeros(1 + NS, dtype=f64, device=E.device)
            wp.launch(k_leak, dim=ia.shape[0], inputs=[ia, float(fr), 6.0e-5, E.h, E.q, E.sp, E.area, out], device=E.device)
            o = out.numpy()
            if o[0] > 0.0:
                sp = np.zeros(NS); sp[:4] = o[1:5]
                pool.add(float(o[0]), sp)

    def _film_bodies(self, bq, dtf):
        """Film + foam water lumped into each free body (with its cavity pool); wall shear force and moment."""
        E, w = self.eng, self.w
        m, mx = E.bf_m.numpy(), E.bf_mx.numpy()
        f, t = E.bf_f.numpy(), E.bf_t.numpy()
        self.shear[:] = 0.0
        for name, b in w.item_bodies.items():
            R = Rotation.from_quat(bq[b, 3:]).as_matrix()
            fw = R @ f[b]
            com = self.com_local[b]
            # moment about the body's COM: r x f with r from the COM (body frame), rotated
            tw = R @ (t[b] - np.cross(com, f[b]))
            self.shear[b, :3], self.shear[b, 3:] = fw, tw
            mass = float(m[b])
            first = mx[b]/mass if mass > 0 else np.zeros(3)
            k = self.pools.index.get(f"cavity:{name}")
            if k is not None:
                p = self.pools.pools[k]
                if p.vol > 0:
                    pts, vw = p.wet_samples()
                    if len(vw):
                        pc = (pts*vw[:, None]).sum(0)/vw.sum()
                        mp = p.vol*RHO
                        first = (first*mass + pc*mp)/(mass + mp)
                        mass += mp
            extra_I = None
            if name in self.sponges:
                # pore water: lumped at the cellulose's centre in the base frame, with its own inertia (a uniform block
                # filling the 111 x 66 x 12 mm cellulose: the water is held in the pores and turns with the sponge)
                mp = self.sponges[name].pool.vol*RHO
                if mp > 0:
                    first = (first*mass + np.array([0., 0., 0.011])*mp)/(mass + mp)
                    mass += mp
                    d_ = P.ITEMS["sponge"]
                    a_, b_, h_ = d_["L"], d_["W"], d_["H"] - d_["scour"]
                    extra_I = mp/12.0*np.array([b_*b_ + h_*h_, a_*a_ + h_*h_, a_*a_ + b_*b_])
            prev = self.lumped.get(b, (0.0, None))
            if abs(mass - prev[0]) > 5e-4 or (prev[1] is None and mass > 0):
                if extra_I is not None:
                    w.set_lumped_mass(b, mass, first, extra_inertia=extra_I)
                else:
                    w.set_lumped_mass(b, mass, first)
                self.lumped[b] = (mass, first)

    def _buoyancy(self, bq, bqd):
        """Buoyancy and drag of free bodies in pools (static bowls and moving cavities), every 4 ticks (held)."""
        self.forces[:] = 0.0
        statics = [p for p in self.pools.pools if p.kind == "static" and p.vol > 0]
        cavs = [p for p in self.pools.pools if p.kind == "cavity" and p.vol > 0]
        disp = {id(p): 0.0 for p in statics}
        self.immersed = {}
        boxes = [(p, np.array(p.lo) - 0.3, np.array(p.hi) + 0.3) for p in statics]
        for b, w_ in self.sample_bodies.items():
            pos = bq[b, :3]
            near_static = any(np.all(pos > lo) and np.all(pos < hi) for _, lo, hi in boxes)
            near_cav = any(np.linalg.norm(pos - bq[p.body, :3]) < 0.8 for p in cavs if p.body != b)
            if not near_static and not near_cav:
                continue
            R = Rotation.from_quat(bq[b, 3:]).as_matrix()
            pts, vols = w_.samples
            X = bq[b, :3] + pts @ R.T
            cav = w_.cavity
            for p in statics:
                lo, hi = p.lo, p.hi
                ins = (X[:, 0] > lo[0]) & (X[:, 0] < hi[0]) & (X[:, 1] > lo[1]) & (X[:, 1] < hi[1]) & (X[:, 2] < p.level)
                Vs = float(vols[ins].sum())
                cen = (X[ins]*vols[ins, None]).sum(0)/Vs if Vs > 0 else None
                frac = Vs/float(vols.sum())
                if cav is not None:
                    # air pocket: cavity samples under the outer level but above the cavity's own liquid
                    k = self.pools.index[f"cavity:{self._name_of(b)}"]
                    ip = self.pools.pools[k]
                    Xc = bq[b, :3] + cav["samples"] @ R.T
                    inc = (Xc[:, 0] > lo[0]) & (Xc[:, 0] < hi[0]) & (Xc[:, 1] > lo[1]) & (Xc[:, 1] < hi[1]) & (Xc[:, 2] < p.level)
                    if ip.vol > 0:
                        inc &= cav["samples"] @ ip.n >= ip.d
                    Va = float(cav["volumes"][inc].sum())
                    if Va > 0:
                        ca = (Xc[inc]*cav["volumes"][inc, None]).sum(0)/Va
                        cen = ca if cen is None else (cen*Vs + ca*Va)/(Vs + Va)
                        Vs += Va
                if Vs <= 0:
                    continue
                disp[id(p)] += Vs
                self.immersed[self._name_of(b)] = (p.name, frac)
                self._apply_fluid_force(b, p, Vs, cen, bq, bqd, None)
            for p in cavs:
                if p.body == b:
                    continue
                Rc = Rotation.from_quat(bq[p.body, 3:]).as_matrix()
                Xl = (X - bq[p.body, :3]) @ Rc
                (z0, z1, _), (r0, r1, _) = p.region
                rr = r0 + (r1 - r0)*np.clip((Xl[:, 2] - z0)/(z1 - z0), 0, 1)
                ins = (Xl @ p.n < p.d) & (Xl[:, 2] > z0 - 0.002) & (np.hypot(Xl[:, 0], Xl[:, 1]) < rr)
                Vs = float(vols[ins].sum())
                if Vs <= 0:
                    continue
                cen = (X[ins]*vols[ins, None]).sum(0)/Vs
                self._apply_fluid_force(b, p, Vs, cen, bq, bqd, p.body)
        for p in statics:
            p.disp = disp[id(p)]

    def _name_of(self, b):
        if not hasattr(self, "_body_name"):
            self._body_name = {v: k for k, v in self.w.item_bodies.items()}
        return self._body_name[b]

    def _apply_fluid_force(self, b, pool, Vs, cen, bq, bqd, container):
        up = self._up(container if container is not None else -1)
        Fb = RHO*P.GRAVITY*Vs*up
        com = bq[b, :3] + Rotation.from_quat(bq[b, 3:]).apply(self.com_local[b])
        v = bqd[b, :3] + np.cross(bqd[b, 3:], cen - com)
        if container is not None:
            comc = bq[container, :3] + Rotation.from_quat(bq[container, 3:]).apply(self.com_local[container])
            v = v - (bqd[container, :3] + np.cross(bqd[container, 3:], cen - comc))
        A = Vs**(2/3)
        Fd = -0.5*RHO*P.POOL["Cd_body"]*A*np.linalg.norm(v)*v
        F = Fb + Fd
        self.forces[b, :3] += F
        self.forces[b, 3:] += np.cross(cen - com, F)
        if container is not None:
            comc = bq[container, :3] + Rotation.from_quat(bq[container, 3:]).apply(self.com_local[container])
            self.forces[container, :3] -= F
            self.forces[container, 3:] -= np.cross(cen - comc, F)

    def _pad_friction(self):
        """Finger-pad friction from the state of the cells under each pad contact: dry 0.8, wet clean
        0.7, soapy 0.2 on glass and glaze (params.SUBSTRATES), blended by phi_s = 1 - exp(-(Gamma + s)/Gamma_ref)."""
        w = self.w
        d = w.mjd
        if not d.ncon or not len(self.pad_geoms):
            return
        hits = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            for pg, og in ((g1, g2), (g2, g1)):
                if pg in self.pad_geoms:
                    body = self._mj_item.get(int(w.mjm.geom_bodyid[og]))
                    if body is not None and body in self.trees:
                        hits.append((pg, body, np.array(c.pos)))
        mu_new = {}
        if hits:
            bq = w.state.body_q.numpy()
            idx = []
            for pg, body, x in hits:
                R = Rotation.from_quat(bq[body, 3:]).as_matrix()
                xl = R.T @ (x - bq[body, :3])
                tree, cells = self.trees[body]
                _, j = tree.query(xl)
                idx.append(int(cells[j]))
            dev = self.device
            ia = wp.array(np.array(idx, np.int32), dtype=int, device=dev)
            out = wp.zeros((len(idx), 4), dtype=float, device=dev)
            wp.launch(k_gather, dim=len(idx), inputs=[ia, self.eng.h, self.eng.sp, self.eng.gamma, out], device=dev)
            vals = out.numpy()
            mats = self.surf.materials
            for (pg, body, x), c, v in zip(hits, idx, vals):
                sub = P.SUBSTRATES.get(mats[self.surf.mat_of[c]], P.SUBSTRATES["glass"])
                mu_dry, mu_wet, mu_soap = sub["mu_pad"]
                hcell, s, gam, foam = v
                phi = 1.0 - math.exp(-(gam + s)/P.SURFACTANT["gamma_ref"])
                if hcell > P.FILM["h_wet"] or foam > 0:
                    mu = mu_wet*(1 - phi) + mu_soap*phi
                else:
                    mu = mu_dry*(1 - phi) + mu_soap*phi
                mu_new[pg] = min(mu_new.get(pg, 9.0), mu)
        for pg in self.pad_geoms:
            mu = mu_new.get(pg, self.mu_pad_dry)
            if abs(self.pad_mu.get(pg, -1) - mu) > 1e-4:
                w.mjm.geom_friction[pg, 0] = mu
                self.pad_mu[pg] = mu

    # ------------------------------------------------------------------ accounts
    def totals(self):
        T = self.eng.totals()
        pv = sum(p.vol for p in self.pools.pools)
        psp = sum((p.sp for p in self.pools.pools), np.zeros(NS))
        pend = 0.0
        pend_sp = np.zeros(NS)
        for x, v, m, d, sp in self.pending:
            pend += float(np.sum(m)); pend_sp += sp.sum(axis=1)
        # foam liquid travels as species 4 (m3) wherever it goes (packets, edge accumulators, sinks, lost)
        foam_moving = (T["sp_packets"][4] + T["sp_eject"][4] + T["sp_sinks"][4] + T["sp_lost"][4] + pend_sp[4])*RHO
        water = T["film"] + T["foam"] + T["packets"] + T["eject"] + float(T["sinks"].sum()) + T["lost"] + \
            (pv + psp[4])*RHO + self.drain_kg + self.drain_sp[4]*RHO + pend + foam_moving
        surf = T["sp_film"][0] + T["sp_film"][5] + T["gamma"] + T["sp_packets"][0] + T["sp_packets"][5] + T["sp_eject"][0] + \
            T["sp_eject"][5] + T["sp_sinks"][0] + T["sp_sinks"][5] + T["sp_lost"][0] + T["sp_lost"][5] + psp[0] + psp[5] + \
            self.drain_sp[0] + self.drain_sp[5] + pend_sp[0] + pend_sp[5]
        solids = T["soil"] + T["sp_film"][1] + T["sp_packets"][1] + T["sp_eject"][1] + T["sp_sinks"][1] + T["sp_lost"][1] + \
            psp[1] + self.drain_sp[1] + pend_sp[1]
        return dict(water_kg=water, surfactant_kg=surf, solids_kg=solids, film_kg=T["film"], foam_kg=T["foam"],
                    packets_kg=T["packets"], pools_kg=(pv + psp[4])*RHO, sinks={n: float(v) for n, v in
                                                                                   zip(self.eng.sinks, T["sinks"])},
                    # everything that struck each uncharted account (kg): what it kept (sinks) plus what ran off it
                    struck={n: float(v + r) for n, v, r in zip(self.eng.sinks, T["sinks"], T["shed"])},
                    drain_kg=self.drain_kg, lost_kg=T["lost"], alive=T["alive"], emitted_kg=float(self.eng.emitted[0]),
                    emitted_surf=float(self.eng.emitted[1] + self.eng.emitted[6]))

    def ledger_residual(self):
        T = self.totals()
        I = self.initial
        res_w = T["water_kg"] - (I["water_kg"] + T["emitted_kg"])
        res_s = T["surfactant_kg"] - (I["surfactant_kg"] + T["emitted_surf"])
        res_m = T["solids_kg"] - I["solids_kg"]
        scale = max(I["water_kg"] + T["emitted_kg"], 1e-9)
        return dict(water_kg=res_w, surfactant_kg=res_s, solids_kg=res_m, relative=abs(res_w)/scale,
                    relative_surf=abs(res_s)/max(I["surfactant_kg"] + T["emitted_surf"], 1e-12),
                    relative_solids=abs(res_m)/max(I["solids_kg"], 1e-12))


class _AlwaysOn:
    """Supply that is always open at the outlet's own valve (the hose reel's wall valve stays on, as in practice): cold
    water, and no handle cartridge in the line (Outlet.target_flow)."""
    cartridges = False

    def __init__(self, supply):
        self.supply = supply

    def state(self):
        return 0.0, 1.0, self.supply["T_cold"]


GPU_STATE = ("h", "q", "sp", "gamma", "soil", "soil_w", "soil_sig", "sub", "mobile", "foam_mob", "ej_m", "ej_p", "ej_sp",
             "bej_m", "bej_p", "bej_sp", "alive", "px", "pv", "pm", "pd", "pkind", "page", "pdist", "psp", "head", "lost", "clamped", "diag",
             "sink_m", "sink_sp", "shed_m", "blocked")


def _snapshot(self):
    import copy
    E = self.eng
    gpu = {k: getattr(E, k).numpy().copy() for k in GPU_STATE}
    host = dict(acc_f=E._acc_f.copy(), alpha_f=E._alpha_f.copy(), prev_v=copy.deepcopy(E._prev_v), emitted=E.emitted.copy(),
                etick=E.tick)
    pools = [dict(vol=p.vol, sp=p.sp.copy(), drain_open=p.drain_open, disp=p.disp, drained=p.drained,
                  chlorine=p.chlorine_ppm) for p in self.pools.pools]
    outlets = {n: dict(Q=o.Q, T=o.T, emitted=o.emitted_kg) for n, o in self.outlets.items()}
    return dict(gpu=gpu, host=host, pools=pools, outlets=outlets, drain_kg=self.drain_kg, drain_sp=self.drain_sp.copy(),
                pending=copy.deepcopy(self.pending), tick=self.tick, film_clock=self._film_clock, stride=self.stride,
                ratio_max=self.ratio_max, lumped=dict(self.lumped), forces=self.forces.copy(), impact=self.impact.copy(),
                reaction=self.reaction.copy(), shear=self.shear.copy(), sanitize_s=dict(self.sanitize_s),
                pad_mu=dict(self.pad_mu), max_impact=dict(self.max_impact), conc=None if self._conc is None else self._conc.copy())


def _restore(self, snap):
    import copy
    E = self.eng
    for k, v in snap["gpu"].items():
        getattr(E, k).assign(v)
    h = snap["host"]
    E._acc_f[:], E._alpha_f[:] = h["acc_f"], h["alpha_f"]
    E._prev_v = copy.deepcopy(h["prev_v"])
    E.emitted[:] = h["emitted"]
    E.tick = h["etick"]
    for p, s in zip(self.pools.pools, snap["pools"]):
        p.vol, p.sp, p.drain_open, p.disp, p.drained, p.chlorine_ppm = s["vol"], s["sp"].copy(), s["drain_open"], s["disp"], \
            s["drained"], s["chlorine"]
        if p.kind == "static":
            p.plane()
    for n, s in snap["outlets"].items():
        o = self.outlets[n]
        o.Q, o.T, o.emitted_kg = s["Q"], s["T"], s["emitted"]
    self.drain_kg, self.drain_sp = snap["drain_kg"], snap["drain_sp"].copy()
    self.pending = copy.deepcopy(snap["pending"])
    self.tick, self._film_clock, self.stride, self.ratio_max = snap["tick"], snap["film_clock"], snap["stride"], snap["ratio_max"]
    self.lumped = dict(snap["lumped"])
    self.forces[:], self.impact[:], self.reaction[:], self.shear[:] = snap["forces"], snap["impact"], snap["reaction"], snap["shear"]
    self.sanitize_s = dict(snap["sanitize_s"])
    self.pad_mu = dict(snap["pad_mu"])
    for g, mu in self.pad_mu.items():
        self.w.mjm.geom_friction[g, 0] = mu
    self.max_impact = dict(snap["max_impact"])
    self._conc = None if snap["conc"] is None else snap["conc"].copy()


WaterSystem.snapshot = _snapshot
WaterSystem.restore = _restore

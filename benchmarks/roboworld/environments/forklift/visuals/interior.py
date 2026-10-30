"""The building from inside (warehouse and dock presets): floor slabs and painted markings, walls, roof structure and
high-bay luminaires, columns, selective pallet racking and the unit loads stored in it, the roller conveyor and the
press it feeds, the LPG cage, empty-pallet stacks, signs and dressing. Draws every layout part that
common.owner(...) gives to 'interior'.

Rules kept here: physics dimensions are authoritative. A part that has a collider (rack uprights and beams, stored
loads, guards, walls, columns, conveyor frames, the press, the cage, pallet stacks) is drawn filling that collider and
within its faces (bevels, panel recesses and labels of a few millimetres aside). Everything drawn without a collider is
listed by dressing(layout, preset, seed) with its bounding box, and draw() draws exactly that list: rack internals
(bracing, wire decks, labels), the roof structure and luminaires above 7 m, wall fixtures, floor stencils, and the
loose equipment along the walls.

Uses the drawing kit of visuals.loads (mesh buffer, materials, prototypes, unit loads).
"""

import math

import numpy as np

from . import loads as K
from .common import KIT, kit, layout_box, layout_surface, owner
from .loads import Buf, asset_proto, font, has, inst, mat, pbr, proto, quat_z, rot_z, save_tex, seed_of, tex

LAMP_GRID, LAMP_Z = 6.0, 8.4      # the grid and height render._lights puts the high-bay light sources at
JOINT = 6.0                       # floor slab joint grid (saw-cut control joints on the column lines)
ROOF_Z = 9.0                      # underside of the roof deck (top of the walls)
PURLIN = 1.5                      # purlin spacing; the luminaires (y = 3 + 6 k) hang under purlins
TRUSS_DEPTH = 1.30
LENS_EMISSION = 1500.0           # OmniPBR emissive intensity of a luminaire's diffuser (see _highbay_proto)
WALL_BAND = 2.40                  # height of the walls' lower band
DT_WH = K.DT_WH
ASSETS = dict(                    # dressing props (cache path, nominal size x, y, z in metres as measured from the meshes)
    pallet_truck=(DT_WH + "Equipment/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd", (0.65, 1.63, 1.27)),
    hand_truck=(DT_WH + "Equipment/Hand_Trucks/Convertible_Steel_A/ConvertibleSteelHandTruck_A01_PR_V_NVD_01.usd", (0.50, 0.48, 1.35)),
    drum_fan=(DT_WH + "Equipment/Fans/Drum_A/DrumFan_A01_01.usd", (0.99, 0.43, 1.02)),
    exit_sign=(DT_WH + "Equipment/Signage/PlasticHardWiredExit_A/PlasticHardwiredExitSign_A01_Red_01.usd", (0.31, 0.044, 0.19)),
    steel_drum=(DT_WH + "Storage/Drums/Steel_A/SteelDrum_A01_PR_NVD_01.usd", (0.64, 0.66, 0.87)),
    plastic_drum=(DT_WH + "Storage/Drums/Plastic_A/PlasticDrum_A04_PR_V_NVD_01.usd", (0.47, 0.47, 0.75)),
    bin=(DT_WH + "Facilities/Trash_Cans/TrashCanWithWheels_A/TrashCanWithWheels_A01_Blue_01.usd", (0.84, 0.88, 1.17)),
    fire_alarm=(DT_WH + "Safety/Fire_Protection/FireAlarm_A/FireAlarm_A01_01.usd", (0.041, 0.10, 0.14)),
    cart=(KIT + "SM_PushcartA_02.usd", (0.94, 1.65, 0.38)),
)
WALL_ASSETS = dict(               # kit props modelled on a wall: origin on the wall plane, +x out of the wall
    ext_board=(KIT + "SM_EmergencyBoardFull_01.usd", (0.19, 0.64, 1.17)),
    fuse_cabinet=(KIT + "SM_FuseBox_01.usd", (0.41, 0.54, 1.71)),
    fuse_box=(KIT + "SM_FuseBox_04.usd", (0.11, 0.33, 0.22)),
    sign_a=(KIT + "SM_SignA_02.usd", (0.004, 0.30, 0.21)), sign_b=(KIT + "SM_SignB_02.usd", (0.004, 0.33, 0.22)),
    sign_c=(KIT + "SM_SignCVer_01.usd", (0.005, 0.10, 0.33)),
)


# ====================================================================================================== entry
def draw(S, world):
    sc = world.scene
    L = sc.layout
    if L is None:                          # lab preset: a plain floor
        S.box("/World/floor", (120.0, 60.0, 0.02), S.mats["floor_concrete"], p=(20.0, 0.0, -0.01), uv_per_m=0.5)
        return
    if sc.c.preset not in ("warehouse", "dock"):
        return
    _floor(S, world)
    _lines(S, world)
    racks, stored, walls, conv = [], {}, [], []
    for bx in L.boxes:
        n = bx["name"]
        if owner(world, "box", n) != "interior":
            continue
        if n.startswith("wall"):
            walls.append(bx)
        elif n.startswith("stored_"):
            loc, part = n[len("stored_"):].rsplit("_", 1)
            stored.setdefault(loc, {})[part] = bx
        elif bx["material"] in ("rack_upright", "rack_beam") or (bx["material"] == "guard_yellow" and bx.get("watch") == "rack"):
            racks.append(bx)
        elif n.startswith("column"):
            (_column_guard if n.endswith("_guard") else _column)(S, bx)
        elif n.startswith("conv_"):
            conv.append(bx)
        elif n == "press":
            _press(S, bx, next((b for b in L.boxes if b["name"] == "press_guard"), None))
        elif n == "press_guard":
            pass                            # drawn with the press
        elif n == "lpg_cage":
            _lpg_cage(S, bx)
        elif n.startswith("pallet_stack"):
            _pallet_stack(S, f"/World/{n}", bx["centre"], bx["half"], seed_of(n, sc.c.seed))
        else:
            layout_box(S, world, bx)
    _walls(S, world, walls)
    _racks(S, world, racks)
    _stored(S, world, stored)
    if conv:
        _conveyor_frames(S, world, conv)
    _conveyor_rollers(S, world)
    # wet patches: a glossy film where the floor is wet (state of the floor, drawn from the start state)
    st = sc.task_start or {}
    for name, wp_ in st.get("floor", {}).items():
        film = pbr(S, "v_wet_film", color=(0.10, 0.10, 0.11), rough_const=0.03, opacity=0.55)
        c, h = wp_["centre"], wp_["half"]
        b = Buf()
        n = 40
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rr = 1.0 + 0.10 * np.sin(3 * ang + 1.0) + 0.06 * np.sin(7 * ang)          # an irregular puddle inside the patch
        V = np.vstack([[c[0], c[1], 0.0015], np.stack([c[0] + 0.92 * h[0] * rr * np.cos(ang) / 1.16,
                                                      c[1] + 0.92 * h[1] * rr * np.sin(ang) / 1.16, np.full(n, 0.0015)], 1)])
        j = np.arange(n)
        b.add(V, np.tile([0.0, 0.0, 1.0], (n + 1, 1)), V[:, :2], np.stack([np.zeros(n, int), 1 + j, 1 + (j + 1) % n], 1))
        b.write(S, f"/World/wet_{name}", film)
        kit(S, f"/World/kit/wet_sign_{name}", "S_WetFloorSign.usd", (c[0] - h[0] - 0.4, c[1] + h[1] - 0.1, 0.0))
    for it in dressing(L, sc.c.preset, sc.c.seed):
        DRAWERS[it["kind"]](S, world, it, f"/World/dressing/{it['name']}")
    K.flush(S)


# ====================================================================================================== floor
def _floor_materials(S):
    F = "polyhaven/concrete_floor_worn_001/concrete_floor_worn_001_"
    d, r, n = tex(F + "diff_2k.jpg"), tex(F + "rough_2k.jpg"), tex(F + "nor_gl_2k.jpg")
    # sealed and trafficked: lighter and smoother than the bare, dark slab the maps were photographed from (their mean
    # albedo is 0.09; a sealed warehouse slab is about 0.35)
    return [pbr(S, f"v_floor{k}", d, n, rough=r, color=(0.36, 0.36, 0.35), tint=t, rough_const=0.16, rough_infl=0.40,
                scale=1 / 3.0, bump=0.5, brightness=br, add=0.07, desat=0.25) for k, (t, br) in
            enumerate((((1.0, 1.0, 0.99), 2.2), ((0.99, 0.99, 0.98), 2.08), ((1.0, 0.995, 0.98), 2.32)))]


def _floor(S, world):
    """Every interior drivable surface as slabs on the 6 m joint grid: each slab takes one of three tones of the same
    sealed concrete and its own offset in the texture; the saw cuts between them show the dark joint filler."""
    L = world.scene.layout
    mats = _floor_materials(S)
    bufs = [Buf() for _ in mats]
    filler, body = Buf(), Buf()
    cut = 0.0025
    for s in L.surfaces:
        if owner(world, "surface", s["name"]) != "interior" or s.get("collider", True) is False:
            continue
        if s.get("grade"):                                          # no graded surface indoors today; keep the fallback
            layout_surface(S, world, s)
            continue
        (cx, cy), (hx, hy) = s["centre"], s["half"]
        z, th = s.get("z_top", 0.0), s.get("thickness", 0.2)
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
        xs = [x0] + [v for v in np.arange(math.floor(x0 / JOINT) * JOINT, x1, JOINT) if x0 + 0.3 < v < x1 - 0.3] + [x1]
        ys = [y0] + [v for v in np.arange(math.floor(y0 / JOINT) * JOINT, y1, JOINT) if y0 + 0.3 < v < y1 - 0.3] + [y1]
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                rng = np.random.default_rng(seed_of("slab", round(xs[i], 2), round(ys[j], 2)))
                a0, a1 = xs[i] + (cut if i else 0.0), xs[i + 1] - (cut if i < len(xs) - 2 else 0.0)
                b0, b1 = ys[j] + (cut if j else 0.0), ys[j + 1] - (cut if j < len(ys) - 2 else 0.0)
                o = rng.uniform(0.0, 9.0, 2)
                bufs[int(rng.integers(0, len(mats)))].quad(((a0 + a1) / 2, (b0 + b1) / 2, z), (1, 0, 0), (0, 1, 0), (a1 - a0, b1 - b0),
                                                           st=(o[0] + a0, o[1] + b0, o[0] + a1, o[1] + b1))
        filler.quad((cx, cy, z - 0.004), (1, 0, 0), (0, 1, 0), (2 * hx, 2 * hy))
        body.box((cx, cy, z - 0.004 - th / 2), (2 * hx, 2 * hy, th), bevel=0.0, uvs=1.0)
    for k, b in enumerate(bufs):
        b.write(S, f"/World/floor/slabs{k}", mats[k])
    filler.write(S, "/World/floor/joint_filler", pbr(S, "v_joint_filler", color=(0.035, 0.035, 0.038), rough_const=0.7))
    body.write(S, "/World/floor/slab_body", pbr(S, "v_slab_body", color=(0.30, 0.30, 0.29), rough_const=0.85))


def _line_material(S, colour):
    """Floor paint with worn edges, chips and scuffed stretches: an opacity mask that repeats every 4 m of line."""
    name = f"v_line_{colour}"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(3 if colour == "yellow" else 4)
    Wd, Ht = 2048, 64
    a = np.ones((Ht, Wd), np.float32)
    # edge erosion: a smooth random depth per column (periodic along the line)
    for edge in (0, 1):
        d = np.zeros(Wd)
        for f, amp in ((3, 2.0), (11, 1.6), (37, 1.2), (97, 0.8)):
            d += amp * (0.5 + 0.5 * np.sin(2 * math.pi * f * np.arange(Wd) / Wd + rng.uniform(0, 6.28)))
        d = np.clip(d - 1.6, 0, None)
        rows = np.arange(Ht)[:, None]
        a *= (rows >= d[None, :]) if edge == 0 else (rows <= Ht - 1 - d[None, :])
    chips = rng.random((Ht, Wd)) < 0.0018                                 # chips flaked off
    chips = np.asarray(Image.fromarray(np.uint8(chips * 255)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255
    a *= 1.0 - np.clip(chips * 1.6, 0, 1)
    scuff = np.zeros(Wd)                                                  # stretches worn thin by tyres
    for _ in range(5):
        c0, w = rng.uniform(0, Wd), rng.uniform(40, 160)
        dx = np.minimum(np.abs(np.arange(Wd) - c0), Wd - np.abs(np.arange(Wd) - c0))
        scuff += rng.uniform(0.25, 0.6) * np.exp(-(dx / w) ** 2)
    grain = np.asarray(Image.fromarray(np.uint8(rng.random((Ht, Wd)) * 255)).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    a *= np.clip(1.0 - scuff[None, :] * (0.4 + 1.2 * grain), 0.0, 1.0)
    rgb = dict(yellow=(0.80, 0.58, 0.02), white=(0.80, 0.80, 0.77))[colour]
    return pbr(S, name, color=rgb, rough_const=0.5, opacity_tex=save_tex(S, f"line_{colour}_mask.png", Image.fromarray(np.uint8(a * 255), "L")))


def _lines(S, world):
    sc = world.scene
    bufs = {"yellow": Buf(), "white": Buf()}
    for ln in sc.layout.lines:
        if owner(world, "line", ln["name"]) != "interior":
            continue
        a, b = np.array(ln["a"]), np.array(ln["b"])
        d = b - a
        Lg = float(np.linalg.norm(d))
        if Lg < 1e-6:
            continue
        u = np.array([d[0] / Lg, d[1] / Lg, 0.0])
        v = np.array([-u[1], u[0], 0.0])
        c = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.0015 + sc.ground_height((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)])
        o = (seed_of(ln["name"]) % 1000) / 250.0
        bufs["yellow" if ln["colour"] == "yellow" else "white"].quad(c, u, v, (Lg, ln["width"]), st=(o, 0.0, o + Lg / 4.0, 1.0))
    for colour, b in bufs.items():
        b.write(S, f"/World/floor/lines_{colour}", _line_material(S, colour))


# ====================================================================================================== walls
def _wall_materials(S):
    Pc = "polyhaven/painted_concrete_02/painted_concrete_02_"             # the dock wall (visuals.dockyard) uses the same
    T = K.KIT_TEX
    upper = pbr(S, "v_wall_upper", normal=tex(Pc + "nor_gl_2k.jpg"), rough=tex(Pc + "rough_2k.jpg"), color=(0.55, 0.56, 0.55),
                rough_const=0.75, bump=0.6, scale=1 / 4.0)
    lower = pbr(S, "v_wall_lower", tex(T + "T_WallA_01_D.png"), tex(T + "T_WallA_01_N.png"), tex(T + "T_WallA_01_ORM.png"),
                color=(0.50, 0.50, 0.49), tint=(0.66, 0.67, 0.68), rough_const=0.8, metal_infl=0.0, scale=1 / 2.8)
    trim = pbr(S, "v_wall_trim", color=(0.16, 0.17, 0.18), rough_const=0.6)
    return upper, lower, trim


def _walls(S, world, walls):
    """Perimeter walls from their collision boxes: the room face is a block band painted light grey under painted
    concrete panels with caulked joints every 6 m and a dark cap line between; the other faces are plain."""
    if not walls:
        return
    L = world.scene.layout
    x0, x1, y0, y1 = L.extent
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    upper, lower, trim = _wall_materials(S)
    bu, bl, bt, bo = Buf(), Buf(), Buf(), Buf()
    for bx in walls:
        c, h = np.array(bx["centre"]), np.array(bx["half"])
        thin = int(np.argmin(h[:2])); al = 1 - thin
        sgn = 1.0 if mid[thin] > c[thin] else -1.0
        n = np.zeros(3); n[thin] = sgn
        u = np.zeros(3); u[al] = 1.0
        if np.cross(u, [0, 0, 1.0])[thin] * sgn < 0:
            u = -u
        face = c[thin] + sgn * h[thin]
        a0, a1 = c[al] - h[al], c[al] + h[al]
        zt = c[2] + h[2]
        band = min(WALL_BAND, zt)

        def rect(buf, lo, hi, z_lo, z_hi, off=0.0, st=None):
            pc = np.zeros(3); pc[thin] = face + sgn * off; pc[al] = (lo + hi) / 2; pc[2] = (z_lo + z_hi) / 2
            buf.quad(pc, u, (0, 0, 1), (hi - lo, z_hi - z_lo), st=st or (lo, z_lo, hi, z_hi))
        rect(bl, a0, a1, 0.0, band)
        if zt > band:
            rect(bu, a0, a1, band, zt)
            rect(bt, a0, a1, band - 0.03, band + 0.03, off=0.003)             # cap line between band and panels
            for v in np.arange(math.floor(a0 / 6.0) * 6.0 + 3.0, a1, 6.0):     # caulked panel joints
                if a0 + 0.5 < v < a1 - 0.5:
                    rect(bt, v - 0.009, v + 0.009, band + 0.03, zt, off=0.0015)
        rect(bt, a0, a1, 0.0, 0.012, off=0.002)                                # the sealant line at the floor
        # the wall's body behind the room face (seen only from outside or through an opening)
        bc = c.copy(); bc[thin] -= sgn * 0.004
        hs = 2 * h.copy(); hs[thin] -= 0.008
        bo.box(bc, hs, bevel=0.0)
    bu.write(S, "/World/walls/panels", upper)
    bl.write(S, "/World/walls/band", lower)
    bt.write(S, "/World/walls/trim", trim)
    bo.write(S, "/World/walls/body", pbr(S, "v_wall_body", color=(0.48, 0.48, 0.47), rough_const=0.85))


# ====================================================================================================== columns
def _column(S, bx):
    """Concrete column with chamfered arrises, in the floor's concrete."""
    F = "polyhaven/concrete_floor_worn_001/concrete_floor_worn_001_"
    m = pbr(S, "v_column", tex(F + "diff_2k.jpg"), tex(F + "nor_gl_2k.jpg"), rough=tex(F + "rough_2k.jpg"), color=(0.42, 0.42, 0.40),
            add=0.07, brightness=2.5, desat=0.3, rough_const=0.8, rough_infl=1.0, scale=1 / 2.0)
    b = Buf()
    b.box(bx["centre"], 2 * np.array(bx["half"]), bevel=0.025, uvo=(bx["centre"][0], 0.0))
    b.write(S, f"/World/{bx['name']}", m)


def _column_guard(S, bx):
    """Moulded polyethylene column protector filling its collision box: ribbed, rounded-square, two straps."""
    c, h = np.array(bx["centre"]), np.array(bx["half"])
    yellow = pbr(S, "v_hdpe_yellow", normal=tex(K.DT_BASE + "Plastics/Textures/T_Plastic_Black_A_Normal.png"), color=(0.80, 0.56, 0.02),
                 rough_const=0.42, scale=3.0)
    zs = np.linspace(0.0, 2 * h[2], 73)
    ang = np.linspace(0, 2 * math.pi, 49)
    e = 5.0                                                     # superellipse exponent: a square with round corners
    cs, sn = np.cos(ang), np.sin(ang)
    k = (np.abs(cs) ** e + np.abs(sn) ** e) ** (-1 / e)
    rib = 0.014 * (0.5 - 0.5 * np.cos(2 * math.pi * zs / 0.10))       # ribs every 10 cm, 14 mm deep
    edge = 0.03 * np.clip(1 - np.minimum(zs, 2 * h[2] - zs) / 0.04, 0, 1) ** 2   # rounded top and bottom
    strap = np.zeros_like(zs, bool)
    for zc in (0.27, 0.63):
        strap |= np.abs(zs - zc * 2 * h[2] / 0.9) < 0.028
    rib = np.where(strap, 0.004, rib)
    R = (1.0 - (rib + edge) / h[0])[:, None] * k[None, :]
    V = np.stack([c[0] + h[0] * R * cs[None, :], c[1] + h[1] * R * sn[None, :], np.broadcast_to((c[2] - h[2] + zs)[:, None], R.shape)], 2)
    gz, ga = np.gradient(V, axis=0), np.gradient(V, axis=1)
    Nn = np.cross(ga, gz)
    Nn /= np.linalg.norm(Nn, axis=2)[:, :, None]
    UV = np.stack([np.broadcast_to(ang[None, :] * h[0], R.shape), np.broadcast_to(zs[:, None], R.shape)], 2)
    m, n = len(zs), len(ang)
    i, j = np.meshgrid(np.arange(m - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel(); b_ = a + 1; c_ = a + n; d_ = c_ + 1
    F = np.concatenate([np.stack([a, b_, d_], 1), np.stack([a, d_, c_], 1)])
    is_strap = np.repeat(strap[:-1] & strap[1:], n - 1)
    body, band = Buf(), Buf()
    body.add(V.reshape(-1, 3), Nn.reshape(-1, 3), UV.reshape(-1, 2), F[np.tile(~is_strap, 2)])
    band.add(V.reshape(-1, 3), Nn.reshape(-1, 3), UV.reshape(-1, 2), F[np.tile(is_strap, 2)])
    top = Buf()
    top.box((c[0], c[1], c[2] + h[2] - 0.016), (2 * h[0] * 0.90, 2 * h[1] * 0.90, 0.03), bevel=0.012)   # the cap around the column
    body.write(S, f"/World/{bx['name']}/body", yellow)
    top.write(S, f"/World/{bx['name']}/cap", yellow)
    band.write(S, f"/World/{bx['name']}/straps", mat(S, "black"))


# ====================================================================================================== racks
def _hole_material(S, kind):
    """The punched holes of a roll-formed upright as a decal: 'front' two columns of teardrops on a 2 in pitch (the beam
    connectors hook into them), 'side' one column of round holes (the bracing bolts)."""
    name = f"v_rack_holes_{kind}"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageDraw
    Wd, Ht = 300, 200                                            # 76.2 x 50.8 mm
    m = Image.new("L", (Wd * 2, Ht * 2), 0)
    d = ImageDraw.Draw(m)
    s = 2 * Wd / 76.2
    if kind == "front":
        for cx in (22.0, 54.2):
            x, y = cx * s, 20.0 * s
            d.ellipse([x - 6.5 * s, y - 6.5 * s, x + 6.5 * s, y + 6.5 * s], fill=255)
            d.polygon([(x - 6.3 * s, y + 1.5 * s), (x + 6.3 * s, y + 1.5 * s), (x + 3.2 * s, y + 20 * s), (x - 3.2 * s, y + 20 * s)], fill=255)
    else:
        x, y = 38.1 * s, 25.4 * s
        d.ellipse([x - 5.5 * s, y - 5.5 * s, x + 5.5 * s, y + 5.5 * s], fill=255)
    m = m.resize((Wd, Ht), Image.LANCZOS)
    return pbr(S, name, color=(0.004, 0.004, 0.005), rough_const=0.9, opacity_tex=save_tex(S, f"rack_holes_{kind}.png", m))


def _frame_proto(S, depth, up, height):
    """Upright frame prototype (origin on the floor at the frame's centre): two roll-formed posts `depth` apart outside
    to outside, horizontal and diagonal bracing between them, footplates with anchors."""
    P = proto(S, f"rack_frame_{int(depth * 1000)}_{int(height * 100)}")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    blue, holes_f, holes_s, anchors = Buf(), Buf(), Buf(), Buf()
    xo = depth / 2 - up / 2
    for sx in (-1, 1):
        blue.box((sx * xo, 0, height / 2), (up, up, height), bevel=0.004, uvo=(sx, 0.0))
        # hole decals: the aisle face and the two faces along the row
        e = 0.0008
        holes_f.quad((sx * (depth / 2 + e), 0, height / 2), (0, sx, 0), (0, 0, 1), (up, height), st=(0, 0, 1, height / 0.0508))
        for sy in (-1, 1):
            holes_s.quad((sx * xo, sy * (up / 2 + e), height / 2), (-sy, 0, 0), (0, 0, 1), (up, height), st=(0, 0, 1, height / 0.0508))
        # footplate: flush with the aisle face, 3 cm proud to the sides, reaching inward; two anchors
        blue.box((sx * (depth / 2 - 0.075), 0, 0.005), (0.15, up + 0.07, 0.010), bevel=0.002)
        for sy in (-1, 1):
            anchors.cyl((sx * (depth / 2 - 0.115), sy * (up / 2 + 0.012), 0.016), 0.011, 0.014, n=6)
            anchors.cyl((sx * (depth / 2 - 0.115), sy * (up / 2 + 0.012), 0.028), 0.006, 0.014, n=8)
    # bracing between the posts (in the frame's plane): horizontals at every other node, diagonals zig-zag
    inner = depth - 2 * up
    nodes = np.linspace(0.15, height - 0.25, 5)
    bw, bt = 0.038, 0.025
    for k, z in enumerate(nodes):
        if k % 2 == 0:
            blue.box((0, 0, z), (inner, bt, bw), bevel=0.003)
    for k in range(len(nodes) - 1):
        za, zb = (nodes[k] + 0.03, nodes[k + 1] - 0.03)
        a = np.array([-inner / 2, 0, za]) if k % 2 == 0 else np.array([inner / 2, 0, za])
        b = np.array([inner / 2, 0, zb]) if k % 2 == 0 else np.array([-inner / 2, 0, zb])
        d = b - a
        Lg = float(np.linalg.norm(d))
        ex = d / Lg
        ey = np.array([0.0, 1.0, 0.0])
        blue.box((a + b) / 2, (Lg, bt, bw), R=np.stack([ex, ey, np.cross(ex, ey)], 1), bevel=0.003)
    blue.write(S, P + "/steel", mat(S, "paint_blue"))
    holes_f.write(S, P + "/holes_front", _hole_material(S, "front"))
    holes_s.write(S, P + "/holes_side", _hole_material(S, "side"))
    anchors.write(S, P + "/anchors", mat(S, "galv"))
    return P


def _beam_proto(S, length, bw, bh):
    """Step beam prototype (origin: middle of the beam, top edge of its aisle face; +x into the rack): the 5 in box
    section with the deck step on the inside top edge, three-pin connector plates hooked onto the posts at both ends."""
    P = proto(S, f"rack_beam_{int(length * 1000)}")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    b = Buf()
    step_w, step_h = 0.018, 0.041
    b.box(((bw - step_w) / 2, 0, -bh / 2), (bw - step_w, length, bh), bevel=0.0035)
    b.box((bw - step_w / 2, 0, -(bh + step_h) / 2), (step_w, length, bh - step_h), bevel=0.002)
    pins = Buf()
    for sy in (-1, 1):
        # connector: an angle welded to the beam's end; its lip lies on the post's aisle face (4 mm proud) with three pins
        b.box((0.028, sy * (length / 2 - 0.003), -0.10), (0.056, 0.006, 0.20), bevel=0.001)
        b.box((-0.002, sy * (length / 2 + 0.017), -0.10), (0.004, 0.040, 0.20), bevel=0.001)
        for z in (-0.05, -0.10, -0.15):
            pins.cyl((-0.0045, sy * (length / 2 + 0.020), z), 0.0075, 0.003, axis=0, n=8)
    b.write(S, P + "/steel", mat(S, "paint_orange"))
    pins.write(S, P + "/pins", mat(S, "galv"))
    return P


def _deck_proto(S, depth, width):
    """Wire deck prototype (origin: centre, top of the wires): 2.5 x 4 in welded mesh on three support channels, lying
    on the beams' steps."""
    P = proto(S, f"wire_deck_{int(depth * 1000)}_{int(width * 1000)}")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    b = Buf()
    w = 0.0048
    ny = int(width / 0.0635)
    for y in np.linspace(-width / 2 + 0.01, width / 2 - 0.01, ny):          # wires front to back, on top
        b.box((0, y, -w / 2), (depth, w, w), bevel=0.0)
    nx = int(depth / 0.1016)
    for x in np.linspace(-depth / 2 + 0.012, depth / 2 - 0.012, nx):        # cross wires under them
        b.box((x, 0, -1.5 * w), (w, width - 0.01, w), bevel=0.0)
    for y in np.linspace(-width / 2 + 0.16, width / 2 - 0.16, 3):           # support channels (inverted U)
        b.box((0, y, -2 * w - 0.001), (depth - 0.004, 0.045, 0.002), bevel=0.0)
        for s in (-1, 1):
            b.box((0, y + s * 0.0225, -2 * w - 0.017), (depth - 0.004, 0.002, 0.032), bevel=0.0)
    b.write(S, P + "/wire", mat(S, "galv"))
    return P


def _hazard_material(S):
    name = "v_hazard"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 256), (222, 160, 8))
    d = ImageDraw.Draw(img)
    for k in range(-4, 8):
        d.polygon([(k * 64, 256), (k * 64 + 32, 256), (k * 64 + 32 + 256, 0), (k * 64 + 256, 0)], fill=(14, 14, 15))
    return pbr(S, name, save_tex(S, "hazard.png", img), rough_const=0.5)


def _guard_proto(S, size):
    """End-of-row rack guard prototype (origin: centre of its collision box): a formed steel rail, safety yellow with a
    hazard band, on two bolted base plates."""
    P = proto(S, f"rack_guard_{int(size[0] * 1000)}")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    sx, sy, sz = size
    b, hz, bolts = Buf(), Buf(), Buf()
    b.box((0, 0, 0.02), (sx, sy, sz - 0.04), bevel=0.012)
    b.box((0, 0, -sz / 2 + 0.006), (sx, sy, 0.012), bevel=0.002)
    for s in (-1, 1):
        hz.quad((0, s * (sy / 2 + 0.0008), sz / 2 - 0.10), (-s, 0, 0), (0, 0, 1), (sx - 0.05, 0.10), st=(0, 0, (sx - 0.05) / 0.2, 0.5))
        for dx in (-sx / 2 + 0.07, sx / 2 - 0.07):
            bolts.cyl((dx, s * (sy / 2 - 0.022), -sz / 2 + 0.017), 0.011, 0.010, n=6)
    b.write(S, P + "/steel", mat(S, "paint_yellow"))
    hz.write(S, P + "/hazard", _hazard_material(S))
    bolts.write(S, P + "/bolts", mat(S, "galv"))
    return P


def _rack_groups(boxes):
    """Rack collision boxes by role: frames {(row, k): [two posts]}, beams {(row, bay, level): [front, back]}, guards."""
    frames, beams, guards = {}, {}, []
    for bx in boxes:
        n = bx["name"]
        if bx["material"] == "rack_upright":
            row, rest = n.split("_up")
            frames.setdefault((row, int(rest.split("_")[0])), []).append(bx)
        elif bx["material"] == "rack_beam":
            row, rest = n.split("_beam")
            k, lv = rest.split("_")[:2]
            beams.setdefault((row, int(k), int(lv)), []).append(bx)
        else:
            guards.append(bx)
    return frames, beams, guards


def _racks(S, world, boxes):
    """Selective pallet racking on its collision boxes: posts and beams are the colliders; the bracing, footplates and
    connectors belong to them (see dressing: rack_internals carries the decks, labels and row spacers)."""
    frames, beams, guards = _rack_groups(boxes)
    for (row, k), posts in sorted(frames.items()):
        xs = sorted(p["centre"][0] for p in posts)
        h = posts[0]["half"]
        if len(posts) != 2:
            for p in posts:
                layout_box(S, world, p)
            continue
        depth = xs[1] - xs[0] + 2 * h[0]
        inst(S, f"/World/racks/{row}/frame{k}", _frame_proto(S, depth, 2 * h[0], 2 * h[2]), ((xs[0] + xs[1]) / 2, posts[0]["centre"][1], 0.0))
    for (row, k, lv), pair in sorted(beams.items()):
        xc = float(np.mean([b["centre"][0] for b in pair]))
        for b in pair:
            c, h = b["centre"], b["half"]
            front = c[0] < xc                                         # this beam's aisle face looks toward -x
            P = _beam_proto(S, 2 * h[1], 2 * h[0], 2 * h[2])
            inst(S, f"/World/racks/{row}/beam{k}_{lv}_{'a' if front else 'b'}", P,
                 (c[0] - h[0] if front else c[0] + h[0], c[1], c[2] + h[2]), None if front else quat_z(math.pi))
    for g in guards:
        inst(S, f"/World/racks/guards/{g['name']}", _guard_proto(S, 2 * np.array(g["half"])), g["centre"])


def _stored(S, world, stored):
    """The unit loads stored in the racks: a pallet (its collision box is the pallet's envelope) and the load filling
    its box: cartons in a pattern, most of them stretch-wrapped, a pallet label toward the aisle."""
    sc = world.scene
    locs = sc.layout.zones.get("rack_locations", {})
    for loc, parts in sorted(stored.items()):
        seed = seed_of("stored", loc, sc.c.seed)
        rng = np.random.default_rng(seed)
        path = f"/World/stored/{loc}"
        if "pallet" in parts:
            pb = parts["pallet"]
            c, h = pb["centre"], pb["half"]
            inst(S, path + "_pallet", K.pallet_proto(S, int(rng.choice(4, p=[0.38, 0.27, 0.2, 0.15]))),
                 (c[0], c[1], c[2] - h[2]), quat_z(math.pi * int(rng.integers(0, 2))))
        if "load" in parts:
            lb = parts["load"]
            c, h = np.array(lb["centre"]), np.array(lb["half"])
            face = locs.get(loc, {}).get("face", "e")
            K.unit_load(S, path + "_load", 2 * h, seed, p=(c[0], c[1], c[2] - h[2]), label_side="+x" if face == "e" else "-x")


def _pallet_stack(S, path, centre, half, seed, n=None):
    """Empty pallets stacked inside a box (a collision box, or a dressing item's bounds): slightly askew, as stacked by
    hand truck, never beyond the box."""
    g = K.gma_parts()
    Hp = max(c[2] + s[2] / 2 for c, s, _ in g)
    n = n or max(1, int(round(2 * half[2] / Hp)))
    dz = min(Hp, 2 * half[2] / n)
    rng = np.random.default_rng(seed)
    k = 0.984                                                   # a hair smaller, so the stack can sit askew inside
    S.xform(path, (centre[0], centre[1], centre[2] - half[2]), (0, 0, 0, 1))
    for i in range(n):
        yaw = float(rng.uniform(-0.007, 0.007)) + (math.pi if rng.random() < 0.5 else 0.0)
        inst(S, f"{path}/p{i}", K.pallet_proto(S, int(rng.choice(4, p=[0.35, 0.3, 0.25, 0.1]))),
             (rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004), i * dz),
             quat_z(yaw), (k, k, 1.0))


# ====================================================================================================== conveyor and press
def _conveyor_frames(S, world, boxes):
    """Chain-driven live roller conveyor on its collision boxes. Each side frame box is a channel on legs closed by
    sheet-steel side guards (the box is solid down to the floor); the guide rails and the end stop are their boxes."""
    frame, galv, yellow, dark, rubber = Buf(), Buf(), Buf(), Buf(), Buf()
    for bx in boxes:
        n = bx["name"]
        c, h = np.array(bx["centre"]), np.array(bx["half"])
        if n.startswith("conv_frame"):
            z1 = c[2] + h[2]
            ch = 0.16                                               # channel depth
            frame.box((c[0], c[1], z1 - ch / 2), (2 * h[0], 2 * h[1], ch), bevel=0.004)
            xs = np.linspace(c[0] - h[0] + 0.05, c[0] + h[0] - 0.05, 6)
            for x in xs:                                            # legs with foot plates
                frame.box((x, c[1], (z1 - ch) / 2), (0.08, 2 * h[1], z1 - ch), bevel=0.003)
                frame.box((x, c[1], 0.005), (0.16, 2 * h[1], 0.010), bevel=0.002)
            for a, b in zip(xs[:-1], xs[1:]):                       # side guards between the legs, 4 mm back
                galv.box(((a + b) / 2, c[1], (z1 - ch) / 2 + 0.01), (b - a - 0.08, 2 * h[1] - 0.008, z1 - ch - 0.02), bevel=0.0)
        elif n.startswith("conv_guide"):
            yellow.box(c, 2 * h, bevel=0.006)
        elif n == "conv_endstop":
            yellow.box((c[0] + 0.015, c[1], c[2]), (2 * h[0] - 0.03, 2 * h[1], 2 * h[2]), bevel=0.006)
            for dy in (-0.35, 0.35):                                # rubber buffers on the upstream face
                rubber.box((c[0] - h[0] + 0.015, c[1] + dy, c[2]), (0.03, 0.16, 0.12), bevel=0.008)
        else:
            dark.box(c, 2 * h, bevel=0.004)
    frame.write(S, "/World/conveyor/frames", mat(S, "paint_ltblue"))
    galv.write(S, "/World/conveyor/side_guards", mat(S, "galv"))
    yellow.write(S, "/World/conveyor/yellow", mat(S, "paint_yellow"))
    dark.write(S, "/World/conveyor/other", mat(S, "paint_grey"))
    rubber.write(S, "/World/conveyor/buffers", mat(S, "rubber"))


def _conveyor_rollers(S, world):
    """The rollers from their collision cylinders (bodies when the conveyor is powered, a static body otherwise):
    galvanised tubes with end caps."""
    sb = world.model.shape_body.numpy()
    scale = world.model.shape_scale.numpy()
    tf = world.model.shape_transform.numpy()
    by_body = {}
    for s in range(world.model.shape_count):
        if str(world.model.shape_label[s]).startswith("conveyor:roller"):
            by_body.setdefault(int(sb[s]), []).append(s)
    galv = mat(S, "galv")
    for b, shapes in by_body.items():
        buf, caps = Buf(), Buf()
        for s in shapes:
            r, hl = float(scale[s, 0]), float(scale[s, 1])
            R = K.quat_R(tf[s, 3:])
            buf.cyl(tf[s, :3], r, 2 * hl - 0.004, n=20, caps=False, R=R, uvs=1.0)
            for sg in (-1.0, 1.0):
                caps.cyl(np.asarray(tf[s, :3]) + R @ np.array([0, 0, sg * (hl - 0.002)]), r, 0.004, n=20, R=R)
        buf.write(S, f"/Lab/body_{b}/rollers", galv)
        caps.write(S, f"/Lab/body_{b}/roller_caps", mat(S, "paint_dark"))


def _mesh_material(S, kind="square"):
    """Welded or expanded steel mesh as a cut-out: opaque wires, open cells."""
    name = f"v_mesh_{kind}"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageDraw
    m = Image.new("L", (256, 256), 0)
    d = ImageDraw.Draw(m)
    if kind == "square":                                          # 50 mm welded mesh: the tile is one cell
        d.rectangle([0, 0, 255, 22], fill=255); d.rectangle([0, 0, 22, 255], fill=255)
    else:                                                         # expanded metal: diamonds
        for k in range(-2, 4):
            d.line([(k * 128, 0), (k * 128 + 128, 128), (k * 128, 256)], fill=255, width=20)
            d.line([(k * 128 + 128, 0), (k * 128, 128), (k * 128 + 128, 256)], fill=255, width=20)
    return pbr(S, name, color=(0.06, 0.06, 0.065) if kind == "square" else (0.42, 0.43, 0.45), metal=0.6, rough_const=0.45,
               opacity_tex=save_tex(S, f"mesh_{kind}.png", m))


def _plate_material(S, name, lines, size=(512, 256), fg=(20, 20, 20), bg=(235, 235, 230), border=None, scale=None, rough=0.45,
                    emissive=None):
    """A printed plate or sign: lines of (text, relative height) centred on a background."""
    key = "v_plate_" + name
    if key in S.mats:
        return S.mats[key]
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    if border:
        d.rectangle([6, 6, size[0] - 7, size[1] - 7], outline=border, width=max(4, size[1] // 40))
    total = sum(h for _, h in lines)
    y = size[1] * 0.08
    for text, h in lines:
        px = size[1] * 0.84 * h / total
        f = font(px * 0.78)
        w = d.textlength(text, font=f)
        if w > size[0] * 0.9:
            f = font(px * 0.78 * size[0] * 0.9 / w)
            w = d.textlength(text, font=f)
        d.text(((size[0] - w) / 2, y + px * 0.08), text, font=f, fill=fg)
        y += px
    return pbr(S, key, save_tex(S, f"plate_{name}.png", img), rough_const=rough, emissive=emissive)


def _press(S, bx, guard):
    """The press as an enclosed, guarded machine filling its collision box: a welded frame of posts and rails on a
    plinth with recessed panels, service doors with windows, an operator panel; toward the conveyor its infeed is closed
    by the yellow mesh guard (the guard plate's collider), behind which the bolster and ram stand in the die space."""
    c, h = np.array(bx["centre"]), np.array(bx["half"])
    x0, x1, y0, y1, z1 = c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] + h[2]
    green, dkgreen, grey, dark, steel, yellow, glass = Buf(), Buf(), Buf(), Buf(), Buf(), Buf(), Buf()
    rec = 0.03                                                     # panels sit this far behind the frame's faces
    plinth = 0.22
    dark.box((c[0], c[1], plinth / 2), (2 * h[0], 2 * h[1], plinth), bevel=0.01)
    post = 0.12
    for sx in (-1, 1):                                             # corner posts
        for sy in (-1, 1):
            dkgreen.box((c[0] + sx * (h[0] - post / 2), c[1] + sy * (h[1] - post / 2), (plinth + z1) / 2), (post, post, z1 - plinth), bevel=0.008)
    crown = 0.75                                                   # the crown (drive housing) above the tie rail
    for z, t in ((plinth + 0.06, 0.12), (z1 - crown, 0.12), (z1 - 0.06, 0.12)):
        for sy in (-1, 1):
            dkgreen.box((c[0], c[1] + sy * (h[1] - post / 2), z), (2 * h[0] - 2 * post, post, t), bevel=0.006)
        for sx in (-1, 1):
            dkgreen.box((c[0] + sx * (h[0] - post / 2), c[1], z), (post, 2 * h[1] - 2 * post, t), bevel=0.006)
    dkgreen.box((c[0], c[1], z1 - 0.01), (2 * h[0] - 0.02, 2 * h[1] - 0.02, 0.02), bevel=0.004)      # roof sheet
    # recessed panels: crown all round; body panels on the two long sides and the back
    zc0, zc1 = z1 - crown + 0.06, z1 - 0.12
    zb0, zb1 = plinth + 0.12, z1 - crown - 0.06
    if guard is not None:
        gc, gh = np.array(guard["centre"]), np.array(guard["half"])
    for sy in (-1, 1):
        yy = c[1] + sy * (h[1] - rec)
        green.quad((c[0], yy, (zc0 + zc1) / 2), (-sy, 0, 0), (0, 0, 1), (2 * h[0] - 2 * post, zc1 - zc0))
        # body: a service door with a window toward the front, a plain panel toward the back
        xm = c[0] + 0.15
        grey.quad(((x0 + post + xm) / 2, yy, (zb0 + zb1) / 2), (-sy, 0, 0), (0, 0, 1), (xm - x0 - post - 0.02, zb1 - zb0))
        green.quad(((xm + x1 - post) / 2, yy, (zb0 + zb1) / 2), (-sy, 0, 0), (0, 0, 1), (x1 - post - xm, zb1 - zb0))
        dkgreen.box((xm, c[1] + sy * (h[1] - 0.02), (zb0 + zb1) / 2), (0.06, 0.04, zb1 - zb0), bevel=0.004)      # mullion
        wx0, wx1 = x0 + post + 0.18, xm - 0.20
        glass.quad(((wx0 + wx1) / 2, yy + sy * 0.004, 1.55), (-sy, 0, 0), (0, 0, 1), (wx1 - wx0, 0.75))
        dark.box(((wx0 + wx1) / 2, yy + sy * 0.006, 1.55), (wx1 - wx0 + 0.06, 0.008, 0.81), bevel=0.0)             # window gasket
        steel.box((xm - 0.09, yy + sy * 0.014, 1.15), (0.022, 0.028, 0.16), bevel=0.004)                         # door handle
        for z in (zb0 + 0.25, zb1 - 0.25):
            steel.box((x0 + post + 0.015, yy + sy * 0.008, z), (0.03, 0.016, 0.10), bevel=0.003)                 # hinges
    xx = x1 - rec
    green.quad((xx, c[1], (zc0 + zc1) / 2), (0, 1, 0), (0, 0, 1), (2 * h[1] - 2 * post, zc1 - zc0))
    grey.quad((xx, c[1], (zb0 + zb1) / 2), (0, 1, 0), (0, 0, 1), (2 * h[1] - 2 * post, zb1 - zb0))
    # front (toward the conveyor, -x): crown panel with the maker's plate; below, the die space behind the guard
    xf = x0 + rec
    green.quad((xf, c[1], (zc0 + zc1) / 2), (0, -1, 0), (0, 0, 1), (2 * h[1] - 2 * post, zc1 - zc0))
    plate = Buf()
    plate.quad((xf - 0.002, c[1], (zc0 + zc1) / 2 + 0.02), (0, -1, 0), (0, 0, 1), (1.30, 0.26))
    gy = 0.9 if guard is None else float(gh[1])
    gz = 2.2 if guard is None else float(gc[2] + gh[2])
    for sy in (-1, 1):                                             # front panels either side of the guard
        w = (h[1] - post) - gy
        if w > 0.02:
            green.quad((xf, c[1] + sy * (gy + w / 2), (zb0 + zb1) / 2), (0, -1, 0), (0, 0, 1), (w, zb1 - zb0))
    if zb1 > gz:
        green.quad((xf, c[1], (gz + zb1) / 2), (0, -1, 0), (0, 0, 1), (2 * gy, zb1 - gz))
    # die space: a dark recess 0.9 m deep with the bolster (bed), the ram and its two guide columns
    depth = 0.9
    xi = x0 + 0.16
    cav = Buf()
    cav.quad((xi + depth, c[1], (plinth + gz) / 2), (0, -1, 0), (0, 0, 1), (2 * gy, gz - plinth))
    for sy in (-1, 1):
        cav.quad((xi + depth / 2, c[1] + sy * gy, (plinth + gz) / 2), (sy, 0, 0), (0, 0, 1), (depth, gz - plinth))
    cav.quad((xi + depth / 2, c[1], gz), (1, 0, 0), (0, -1, 0), (depth, 2 * gy))
    steel.box((xi + depth / 2, c[1], plinth + 0.33), (depth - 0.1, 2 * gy - 0.3, 0.66), bevel=0.012)              # bed
    steel.box((xi + depth / 2, c[1], plinth + 0.70), (depth - 0.2, 2 * gy - 0.5, 0.08), bevel=0.006)              # bolster plate
    steel.box((xi + depth / 2, c[1], gz - 0.42), (depth - 0.25, 2 * gy - 0.55, 0.50), bevel=0.012)                # ram (slide)
    for sy in (-1, 1):
        steel.cyl((xi + depth / 2, c[1] + sy * (gy - 0.2), (plinth + gz) / 2 + 0.3), 0.05, gz - plinth - 0.7, n=20)
    # the guard: a yellow frame carrying black welded mesh, filling the guard collider out to the press's front face
    mesh = Buf()
    fw = 0.06
    xg = x0 + 0.07
    for sy in (-1, 1):
        yellow.box((xg, c[1] + sy * (gy - fw / 2), gz / 2), (0.14, fw, gz), bevel=0.006)
    for z in (fw / 2, gz * 0.5, gz - fw / 2):
        yellow.box((xg, c[1], z), (0.14, 2 * gy - 2 * fw, fw), bevel=0.006)
    mesh.quad((x0 + 0.05, c[1], gz / 2), (0, -1, 0), (0, 0, 1), (2 * gy - 2 * fw, gz - 2 * fw), st=(0, 0, (2 * gy - 2 * fw) / 0.05, (gz - 2 * fw) / 0.05), both=True)
    # operator panel recessed in the south side's door post zone: screen, buttons, e-stop
    op = Buf()
    op.box((x0 + post + 0.11, y0 + rec - 0.011, 1.42), (0.16, 0.02, 0.36), bevel=0.004)
    scr = Buf()
    scr.quad((x0 + post + 0.11, y0 + rec - 0.0215, 1.50), (1, 0, 0), (0, 0, 1), (0.12, 0.09))
    red = Buf()
    red.cyl((x0 + post + 0.11, y0 + 0.005, 1.32), 0.018, 0.008, axis=1, n=16)
    # hazard band round the plinth's top edge
    hz = Buf()
    for (px, py, u, Lg) in ((c[0], y0 - 0.0008, (1, 0, 0), 2 * h[0]), (c[0], y1 + 0.0008, (-1, 0, 0), 2 * h[0]),
                            (x1 + 0.0008, c[1], (0, 1, 0), 2 * h[1]), (x0 - 0.0008, c[1], (0, -1, 0), 2 * h[1])):
        hz.quad((px, py, plinth - 0.05), u, (0, 0, 1), (Lg - 0.03, 0.08), st=(0, 0, (Lg - 0.03) / 0.16, 0.5))
    root = "/World/press"
    green.write(S, root + "/panels", mat(S, "paint_green"))
    dkgreen.write(S, root + "/frame", pbr(S, "v_press_frame", tex(K.DT_BASE + "Metals/Textures/T_Metal_Painted_Gray_A_Albedo.png"),
                                           tex(K.DT_BASE + "Metals/Textures/T_Metal_Painted_Green_A_Normal.png"),
                                           tex(K.DT_BASE + "Metals/Textures/T_Metal_Painted_Green_A_ORM.png"), color=(0.03, 0.14, 0.07),
                                           tint=(0.03, 0.14, 0.07), add=0.35, spec=0.35, scale=2.0))
    grey.write(S, root + "/doors", mat(S, "paint_cream"))
    dark.write(S, root + "/plinth", mat(S, "paint_dark"))
    steel.write(S, root + "/machine", mat(S, "steel"))
    yellow.write(S, root + "/guard_frame", mat(S, "paint_yellow"))
    cav.write(S, root + "/die_space", pbr(S, "v_cavity", color=(0.045, 0.05, 0.05), rough_const=0.7))
    mesh.write(S, root + "/guard_mesh", _mesh_material(S, "square"))
    glass.write(S, root + "/windows", pbr(S, "v_dark_glass", color=(0.02, 0.025, 0.03), rough_const=0.04, spec=1.0))
    plate.write(S, root + "/plate", _plate_material(S, "press", [("HALDEN", 1.0), ("250 t  SERVO PRESS", 0.42)], (1024, 208),
                                                    fg=(235, 235, 230), bg=(18, 22, 20)))
    op.write(S, root + "/operator_panel", mat(S, "paint_grey"))
    scr.write(S, root + "/screen", pbr(S, "v_hmi", color=(0.02, 0.10, 0.16), rough_const=0.15, emissive=(0.10, 0.45, 0.70, 180.0)))
    red.write(S, root + "/estop", pbr(S, "v_red_plastic", color=(0.65, 0.02, 0.02), rough_const=0.35))
    hz.write(S, root + "/hazard", _hazard_material(S))


# ====================================================================================================== LPG cage, cylinders
def _cylinder_proto(S):
    """A 33 lb forklift LPG cylinder lying along +x (origin at its centre): shell, foot ring, valve collar."""
    P = proto(S, "lpg_cylinder")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    r, Lc = 0.152, 0.50
    prof = [(0.0, -Lc / 2 - 0.095)] + [(r * math.sin(a), -Lc / 2 - 0.095 * math.cos(a)) for a in np.linspace(0.15, math.pi / 2, 7)]
    prof += [(r, Lc / 2)] + [(r * math.cos(a), Lc / 2 + 0.095 * math.sin(a)) for a in np.linspace(0.2, math.pi / 2 - 0.15, 6)] + [(0.0, Lc / 2 + 0.095)]
    S.revolve(P + "/shell", prof, pbr(S, "v_lpg_shell", tex(K.DT_BASE + "Metals/Textures/T_Steel_A1_Albedo.png"),
                                      tex(K.DT_BASE + "Metals/Textures/T_Steel_A1_Normal.png"), tex(K.DT_BASE + "Metals/Textures/T_Steel_A1_ORM.png"),
                                      color=(0.62, 0.63, 0.64), metal=1.0, tint=(0.95, 0.95, 0.96), scale=3.0), n=32, axis="x")
    b, v = Buf(), Buf()
    b.cyl((-Lc / 2 - 0.085, 0, 0), 0.105, 0.05, axis=0, n=24, caps=False)                         # foot ring
    b.cyl((Lc / 2 + 0.105, 0, 0), 0.108, 0.09, axis=0, n=24, caps=False, a0=0.5, a1=2 * math.pi - 0.5)   # collar, open one side
    v.cyl((Lc / 2 + 0.11, 0, 0), 0.017, 0.05, axis=0, n=10)
    v.box((Lc / 2 + 0.12, 0.03, 0), (0.03, 0.05, 0.02), bevel=0.003)
    b.write(S, P + "/rings", mat(S, "galv"))
    v.write(S, P + "/valve", pbr(S, "v_brass", color=(0.55, 0.40, 0.14), metal=1.0, rough_const=0.35))
    return P


def _lpg_cage(S, bx):
    """Cylinder storage cage filling its collision box: angle-iron frame, expanded-metal sides and doors, sheet roof,
    two shelves of cylinders lying valve-out, the hazard placards on the doors."""
    c, h = np.array(bx["centre"]), np.array(bx["half"])
    z0 = c[2] - h[2]
    frame, sheet, mesh, signs = Buf(), Buf(), Buf(), Buf()
    a = 0.045
    for sx in (-1, 1):
        for sy in (-1, 1):
            frame.box((c[0] + sx * (h[0] - a / 2), c[1] + sy * (h[1] - a / 2), c[2]), (a, a, 2 * h[2]), bevel=0.004)
    for z in (0.10, 0.92, 2 * h[2] - a / 2):
        for sy in (-1, 1):
            frame.box((c[0], c[1] + sy * (h[1] - a / 2), z0 + z), (2 * h[0] - 2 * a, a, a), bevel=0.004)
        for sx in (-1, 1):
            frame.box((c[0] + sx * (h[0] - a / 2), c[1], z0 + z), (a, 2 * h[1] - 2 * a, a), bevel=0.004)
    frame.box((c[0] + h[0] - a / 2, c[1], c[2]), (a * 0.7, a, 2 * h[2] - 2 * a), bevel=0.003)          # door meeting stile
    sheet.box((c[0], c[1], z0 + 2 * h[2] - 0.004), (2 * h[0], 2 * h[1], 0.008), bevel=0.002)          # roof
    for z in (0.10, 0.92):
        sheet.box((c[0], c[1], z0 + z + a / 2 + 0.002), (2 * h[0] - 2 * a, 2 * h[1] - 2 * a, 0.004), bevel=0.0)   # shelves
    e = 0.006                                                        # mesh panels just inside the frame's faces
    for sy in (-1, 1):
        mesh.quad((c[0], c[1] + sy * (h[1] - e), c[2]), (-sy, 0, 0), (0, 0, 1), (2 * h[0] - 2 * a, 2 * h[2] - 2 * a),
                  st=(0, 0, (2 * h[0]) / 0.06, (2 * h[2]) / 0.03), both=True)
    for sx in (-1, 1):
        mesh.quad((c[0] + sx * (h[0] - e), c[1], c[2]), (0, sx, 0), (0, 0, 1), (2 * h[1] - 2 * a, 2 * h[2] - 2 * a),
                  st=(0, 0, (2 * h[1]) / 0.06, (2 * h[2]) / 0.03), both=True)
    P = _cylinder_proto(S)
    k = 0
    for z in (0.10, 0.92):
        for y in np.linspace(c[1] - h[1] + 0.26, c[1] + h[1] - 0.26, 4):
            if (k * 7 + 3) % 8 == 0:                                 # a slot left empty
                k += 1
                continue
            inst(S, f"/World/{bx['name']}/cyl{k}", P, (c[0] - 0.04, y, z0 + z + a / 2 + 0.004 + 0.152))
            k += 1
    # placards on the door (the face toward the floor, +x): flammable gas, no smoking
    signs.quad((c[0] + h[0] + 0.0008, c[1] - 0.42, z0 + 1.42), (0, 1, 0), (0, 0, 1), (0.50, 0.25))
    s2 = Buf()
    s2.quad((c[0] + h[0] + 0.0008, c[1] + 0.42, z0 + 1.42), (0, 1, 0), (0, 0, 1), (0.50, 0.25))
    root = f"/World/{bx['name']}"
    frame.write(S, root + "/frame", mat(S, "paint_yellow"))
    sheet.write(S, root + "/sheet", mat(S, "galv"))
    mesh.write(S, root + "/mesh", _mesh_material(S, "expanded"))
    signs.write(S, root + "/sign_gas", _plate_material(S, "lpg", [("DANGER", 1.0), ("FLAMMABLE GAS", 0.7), ("LP-GAS CYLINDERS", 0.5)],
                                                       fg=(240, 240, 235), bg=(170, 20, 16), border=(240, 240, 235)))
    s2.write(S, root + "/sign_smoke", _plate_material(S, "nosmoke", [("NO SMOKING", 1.0), ("NO OPEN FLAME", 0.7), ("WITHIN 8 m", 0.5)],
                                                      fg=(170, 20, 16), bg=(240, 240, 235), border=(170, 20, 16)))


# ====================================================================================================== dressing: the list
def _item(name, kind, p, half, yaw=0.0, where="floor", **extra):
    d = dict(name=name, kind=kind, p=tuple(float(v) for v in p), yaw=float(yaw), half=tuple(float(v) for v in half), where=where)
    d.update(extra)
    return d


def _on_floor(name, kind, x, y, yaw=0.0, **extra):
    """A prop standing on the floor at (x, y): bounds from its nominal size."""
    sx, sy, sz = ASSETS[kind][1] if kind in ASSETS else extra.pop("size", (0.36, 0.36, 0.70))
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return _item(name, kind, (x, y, sz / 2), ((sx * c + sy * s) / 2, (sx * s + sy * c) / 2, sz / 2), yaw, **extra)


def _on_wall(name, kind, wall, along, z, size=None, **extra):
    """A fixture on a wall's room face. wall: (axis the wall is normal to, coordinate of its face, +1/-1 the direction
    into the room); along: the position along the wall; z: the fixture's centre height."""
    ax, face, sgn = wall
    d, w, hh = size or WALL_ASSETS.get(kind, ASSETS.get(kind, (None, (0.05, 0.3, 0.3))))[1]
    p = [0.0, 0.0, z]
    p[ax] = face + sgn * d / 2
    p[1 - ax] = along
    half = [0.0, 0.0, hh / 2]
    half[ax], half[1 - ax] = d / 2, w / 2
    yaw = {(0, 1): 0.0, (0, -1): math.pi, (1, 1): math.pi / 2, (1, -1): -math.pi / 2}[(ax, int(sgn))]
    return _item(name, kind, p, half, yaw, where="wall", **extra)


def dressing(layout, preset, seed):
    """Every object this module draws that has no collider: dict(name, kind, p, yaw, half, where[, ...]) with p the
    centre of its bounding box in world metres, half its half-extents (yaw about z; the box is axis-aligned) and
    where one of 'floor' (stands on the floor: equipment along the walls), 'wall' (fixed to a wall, at most 0.45 m
    proud), 'paint' (on the floor's surface), 'rack' (inside a rack row's footprint), 'overhead' (above 3 m),
    'machine' (within a machine's footprint). Depends on the layout and the preset only; draw() draws exactly this
    list. Nothing stands in a zone of layout.zones or on an aisle or lane."""
    L = layout
    out = []
    if L is None or preset not in ("warehouse", "dock"):
        return out
    x0, x1, y0, y1 = L.extent
    Ws, Wn, Ww, We = (1, y0, 1), (1, y1, -1), (0, x0, 1), (0, x1, -1)        # the room faces of the four walls
    # ---- overhead: roof deck, trusses, purlins, sprinkler pipes, luminaires
    out.append(_item("roof_deck", "roof_deck", ((x0 + x1) / 2, (y0 + y1) / 2, ROOF_Z + 0.06), ((x1 - x0) / 2 + 0.3, (y1 - y0) / 2 + 0.3, 0.06), where="overhead"))
    for k, xx in enumerate(np.arange(x0, x1 + 0.01, 6.0)):
        xx = float(min(max(xx, x0 + 0.12), x1 - 0.12))
        out.append(_item(f"truss{k}", "truss", (xx, (y0 + y1) / 2, ROOF_Z - 0.2 - TRUSS_DEPTH / 2), (0.09, (y1 - y0) / 2, TRUSS_DEPTH / 2), where="overhead"))
    out.append(_item("purlins", "purlins", ((x0 + x1) / 2, (y0 + y1) / 2, ROOF_Z - 0.10), ((x1 - x0) / 2, (y1 - y0) / 2, 0.10), where="overhead"))
    out.append(_item("sprinklers", "sprinklers", ((x0 + x1) / 2, (y0 + y1) / 2, 8.60), ((x1 - x0) / 2, (y1 - y0) / 2, 0.12), where="overhead"))
    k = 0
    for xx in np.arange(x0 + LAMP_GRID / 2, x1, LAMP_GRID):
        for yy in np.arange(y0 + LAMP_GRID / 2, y1, LAMP_GRID):
            out.append(_item(f"highbay{k}", "highbay", (float(xx), float(yy), LAMP_Z + 0.2), (0.64, 0.24, 0.22), where="overhead"))
            k += 1
    # exhaust louvres high in the walls and large ceiling fans under the trusses
    spans = (("s", Ws, x0, x1), ("e", We, y0, y1)) + ((("w", Ww, y0, y1),) if preset == "dock" else ())
    for nm, wall, a0, a1 in spans:
        for k, v in enumerate(np.arange(a0 + 9.0, a1 - 3.0, 12.0)):
            out.append(_on_wall(f"louvre_{nm}{k}", "louvre", wall, float(v), 6.6, size=(0.06, 1.30, 1.30)))
            out[-1]["where"] = "overhead"
    fans = ((30.0, 15.0), (42.0, 27.0), (18.0, 30.0)) if preset == "warehouse" else ((12.0, 9.0),)
    for k, (fx, fy) in enumerate(fans):
        if x0 + 4 < fx < x1 - 4 and y0 + 4 < fy < y1 - 4:
            out.append(_item(f"ceiling_fan{k}", "ceiling_fan", (fx, fy, 7.1), (2.75, 2.75, 0.45), where="overhead"))
    # ---- racks: what is drawn inside each row's footprint without a collider, and the signs hung over the aisles
    rows = {}
    for bx in L.boxes:
        if bx["material"] == "rack_upright":
            r = rows.setdefault(bx["name"].split("_up")[0], [np.inf, -np.inf, np.inf, -np.inf, 0.0])
            c, h = bx["centre"], bx["half"]
            r[0], r[1] = min(r[0], c[0] - h[0]), max(r[1], c[0] + h[0])
            r[2], r[3] = min(r[2], c[1] - h[1]), max(r[3], c[1] + h[1])
            r[4] = max(r[4], c[2] + h[2])
    for row, (a0, a1, b0, b1, top) in sorted(rows.items()):
        out.append(_item(f"rack_{row}_internals", "rack_internals", ((a0 + a1) / 2, (b0 + b1) / 2, top / 2),
                         ((a1 - a0) / 2, (b1 - b0) / 2, top / 2), where="rack", row=row))
    letters = sorted({r[0] for r in rows})
    for letter in letters:
        rs = [v for r, v in rows.items() if r[0] == letter]
        a0, a1 = min(v[0] for v in rs), max(v[1] for v in rs)
        for end, yy in (("s", min(v[2] for v in rs)), ("n", max(v[3] for v in rs))):
            sgn = -1 if end == "s" else 1
            out.append(_item(f"row_sign_{letter}_{end}", "row_sign", ((a0 + a1) / 2, yy + sgn * 0.012, 2.55), (0.30, 0.012, 0.22),
                             math.pi if end == "n" else 0.0, where="rack", text=letter))
    for name, (xa, xb) in sorted(L.zones.get("aisles", {}).items()):
        yn = max((v[3] for v in rows.values()), default=22.0)
        out.append(_item(f"aisle_sign_{name}", "aisle_sign", ((xa + xb) / 2, yn + 0.6, 5.2), (0.50, 0.03, 0.30), where="overhead", text=name))
    # ---- machines: the conveyor's parts between and on its frames, the press's top
    conv = L.zones.get("conveyor")
    if conv:
        cx, cy = sum(conv["x"]) / 2, sum(conv["y"]) / 2
        out.append(_item("conveyor_internals", "conveyor_internals", (cx, cy, 0.28), ((conv["x"][1] - conv["x"][0]) / 2, (conv["y"][1] - conv["y"][0]) / 2 + 0.10, 0.28),
                         where="machine"))
        for nm, xe in (("entry", conv["x"][0] + 0.35), ("end", conv["eye_x"])):
            out.append(_item(f"photo_eye_{nm}", "photo_eye", (xe, cy, 0.36), (0.05, (conv["y"][1] - conv["y"][0]) / 2 + 0.17, 0.20), where="machine"))
    press = next((b for b in L.boxes if b["name"] == "press"), None)
    if press:
        c, h = press["centre"], press["half"]
        out.append(_item("press_top", "press_top", (c[0] + 0.1, c[1], c[2] + h[2] + 0.30), (0.75, 0.55, 0.30), where="overhead"))
    # ---- the layout's own props
    for pr in L.props:
        kind = pr.get("kind")
        p = pr["p"]
        if kind == "convex_mirror":
            out.append(_item(pr["name"], "convex_mirror", (p[0], p[1], (p[2] - 0.33 + ROOF_Z - 0.2) / 2), (0.33, 0.33, (ROOF_Z - 0.2 - p[2] + 0.33) / 2),
                             pr.get("yaw", 0.0), where="overhead", z=p[2], note="hangs from the purlins; its lowest point is 0.33 m under z"))
        elif kind == "sign_speed8":
            out.append(_item(pr["name"], "speed_sign", (p[0], p[1], (p[2] + 0.25) / 2), (0.23, 0.23, (p[2] + 0.25) / 2), pr.get("yaw", 0.0), z=p[2]))
        elif kind == "extinguisher":
            out.append(_item(pr["name"], "extinguisher_stand", (p[0], p[1], 0.55), (0.16, 0.16, 0.55), pr.get("yaw", 0.0)))
        elif kind == "first_aid":
            yw = pr.get("yaw", 0.0)                                   # the board faces along yaw; 0.25 deep, 0.66 wide
            c, s_ = abs(math.cos(yw)), abs(math.sin(yw))
            out.append(_item(pr["name"], "safety_station", (p[0], p[1], 1.05), (0.25 * c + 0.33 * s_, 0.25 * s_ + 0.33 * c, 1.05), yw))
    # ---- paint: stencilled identifiers by the bays, lanes, aisles and stop lines
    def stencil(name, text, x, y, up, hgt=0.30):
        w = hgt * (72 * len(text) + 24) / 128.0
        a = abs(math.cos(up)), abs(math.sin(up))
        out.append(_item(f"stencil_{name}", "stencil", (x, y, 0.001), ((w * a[0] + hgt * a[1]) / 2, (w * a[1] + hgt * a[0]) / 2, 0.001),
                         up, where="paint", text=text, height=hgt))
    for nm, b in sorted(L.zones.get("bays", {}).items()):
        south_row = b["approach"] < 0
        stencil(nm, nm, sum(b["x"]) / 2, b["y"][1] + 0.26 if south_row else b["y"][0] - 0.26, math.pi if south_row else 0.0, 0.20)
    if preset == "warehouse":
        for nm, z in sorted(L.zones.get("receiving", {}).items()):
            stencil(nm, nm, z["x"][0] + 0.55, sum(z["y"]) / 2, -math.pi / 2, 0.45)
        for nm, z in sorted(L.zones.get("staging", {}).items()):
            stencil(nm, nm, sum(z["x"]) / 2, z["y"][0] + 0.45, 0.0, 0.45)
        for nm, (xa, xb) in sorted(L.zones.get("aisles", {}).items()):
            yn = max((v[3] for v in rows.values()), default=22.0)
            stencil(f"aisle{nm}", nm, (xa + xb) / 2, yn + 1.6, math.pi, 0.80)
        for nm, z in sorted(L.zones.get("intersections", {}).items()):
            ym = sum(z["y"]) / 2
            stencil(f"stop_{nm}_w", "STOP", z["stop_x"] - 0.75, ym, -math.pi / 2, 0.55)
            stencil(f"stop_{nm}_e", "STOP", z["x"][1] + 0.6 + 0.75, ym, math.pi / 2, 0.55)
    else:
        for nm, z in sorted(L.zones.get("staging", {}).items()):
            stencil(nm, nm.replace("lane", "LANE "), sum(z["x"]) / 2, z["y"][0] + 0.45, 0.0, 0.32)
        br = L.zones.get("bale_row")
        if br:
            stencil("bales", "BALES", br["x"][0] - 0.40, sum(br["y"]) / 2, -math.pi / 2, 0.40)
    # a painted pedestrian walkway along the open back of a single rack row (the side its mesh backing guards)
    locs = L.zones.get("rack_locations", {})
    for row, (a0, a1, b0, b1, top) in sorted(rows.items()):
        if sum(1 for r in rows if r[0] == row[0]) != 1:
            continue
        face = next((v["face"] for n, v in locs.items() if n.split("-")[0] == row[0]), None)
        xw = a1 + 0.95 if face == "w" else a0 - 0.95                     # centre line of a 1.2 m walkway behind the row
        if face is None or not (x0 + 1.6 < xw < x1 - 1.6):
            continue
        out.append(_item(f"walkway_{row}", "walkway", (xw, (b0 + b1) / 2, 0.001), (0.60, (b1 - b0) / 2, 0.001), where="paint"))
        for k, yy in enumerate(np.linspace(b0 + 2.0, b1 - 2.0, 3)):
            stencil(f"walk_{row}{k}", "WALKWAY", xw, float(yy), 0.0, 0.19)
    # ---- paint-thin: rubber left by turning and travelling trucks
    def marks(name, x, y, yaw, style):
        w, ln = (3.6, 3.6) if style == "arc" else (1.6, 4.5)
        yaw += ((seed_of("tyre", name) % 100) / 100.0 - 0.5) * (0.5 if style == "arc" else 0.12)
        r = math.hypot(w, ln) / 2 if style == "arc" else None
        c, s_ = abs(math.cos(yaw)), abs(math.sin(yaw))
        half = (r, r) if r else ((w * c + ln * s_) / 2, (w * s_ + ln * c) / 2)
        out.append(_item(f"tyre_{name}", "tyre_marks", (x, y, 0.001), (*half, 0.001), yaw, where="paint", style=style, size=(w, ln)))
    if preset == "warehouse":
        yn = max((v[3] for v in rows.values()), default=22.0)
        for nm, (xa, xb) in sorted(L.zones.get("aisles", {}).items()):
            xc = (xa + xb) / 2
            marks(f"aisle{nm}_in_e", xc + 1.2, yn + 3.2, math.pi, "arc"); marks(f"aisle{nm}_in_w", xc - 1.2, yn + 3.2, -math.pi / 2, "arc")
            marks(f"aisle{nm}_run1", xc + 0.2, yn - 4.0, 0.0, "run"); marks(f"aisle{nm}_run2", xc - 0.3, yn - 11.0, math.pi, "run")
        for nm, z in sorted(L.zones.get("intersections", {}).items()):
            xm, ym = sum(z["x"]) / 2, sum(z["y"]) / 2
            marks(f"{nm}_a", xm + 0.6, ym - 1.2, math.pi / 2, "arc"); marks(f"{nm}_b", xm - 0.6, ym - 1.6, math.pi, "arc")
            marks(f"{nm}_lane", xm, z["y"][0] - 4.5, 0.0, "run")
        ma = L.zones.get("main_aisle")
        if ma:
            for k, xx in enumerate(np.arange(ma["x"][0] + 6.0, ma["x"][1] - 4.0, 9.0)):
                marks(f"main{k}", float(xx), ma["y"][0] + (1.3 if k % 2 else 3.1), math.pi / 2, "run")
        for nm, z in sorted(L.zones.get("staging", {}).items()):
            marks(f"{nm}_run", sum(z["x"]) / 2, z["y"][0] + 1.9, 0.0, "run")
        for nm, z in sorted(L.zones.get("receiving", {}).items()):
            marks(f"{nm}_run", z["x"][0] + 3.5, sum(z["y"]) / 2, math.pi / 2, "run")
        if conv:
            marks("conv_in", conv["x"][0] - 2.6, sum(conv["y"]) / 2, math.pi / 2, "run")
            marks("conv_turn", conv["x"][0] - 4.0, sum(conv["y"]) / 2 + 2.2, 0.0, "arc")
    else:
        for nm, z in sorted(L.zones.get("staging", {}).items()):
            xm = sum(z["x"]) / 2
            marks(f"{nm}_dock", xm, z["y"][1] + 2.0, 0.0, "run"); marks(f"{nm}_turn", xm + 0.5, z["y"][0] - 2.4, math.pi / 2, "arc")
        br = L.zones.get("bale_row")
        if br:
            marks("bales_a", br["x"][0] - 2.2, sum(br["y"]) / 2 + 1.0, math.pi, "arc"); marks("bales_b", br["x"][0] - 3.0, sum(br["y"]) / 2 - 1.5, -math.pi / 2, "run")
    # ---- walls and the floor along them
    if preset == "warehouse":
        racks_e = max((v[1] for v in rows.values()), default=14.0)
        xs = racks_e + 1.6
        out += [
            _on_wall("door_s", "door", Ws, 26.5, 1.07, size=(0.09, 1.02, 2.14)), _on_wall("exit_s", "exit_sign", Ws, 26.5, 2.45, size=(0.05, 0.31, 0.19)),
            _on_wall("door_n", "door", Wn, 8.0, 1.07, size=(0.09, 1.02, 2.14)), _on_wall("exit_n", "exit_sign", Wn, 8.0, 2.45, size=(0.05, 0.31, 0.19)),
            _on_wall("door_e", "door", We, 24.8, 1.07, size=(0.09, 1.02, 2.14)), _on_wall("exit_e", "exit_sign", We, 24.8, 2.45, size=(0.05, 0.31, 0.19)),
            _on_wall("ext_s1", "ext_board", Ws, 18.6, 1.35), _on_wall("ext_s2", "ext_board", Ws, 40.2, 1.35),
            _on_wall("ext_n1", "ext_board", Wn, 14.2, 1.35), _on_wall("ext_e1", "ext_board", We, 13.4, 1.35),
            _on_wall("ext_e2", "ext_board", We, 33.0, 1.35),
            _on_wall("panel_s", "fuse_cabinet", Ws, 45.6, 0.86, conduit=True), _on_wall("panel_n", "fuse_cabinet", Wn, 21.0, 0.86, conduit=True),
            _on_wall("box_e", "fuse_box", We, 14.6, 1.45, conduit=True), _on_wall("box_s", "fuse_box", Ws, 31.0, 1.45, conduit=True),
            _on_wall("alarm_s", "fire_alarm", Ws, 27.4, 1.25, size=(0.045, 0.10, 0.14)), _on_wall("alarm_n", "fire_alarm", Wn, 8.9, 1.25, size=(0.045, 0.10, 0.14)),
            _on_wall("alarm_e", "fire_alarm", We, 25.7, 1.25, size=(0.045, 0.10, 0.14)),
            _on_wall("sign_s1", "sign_a", Ws, 17.6, 1.75), _on_wall("sign_s2", "sign_b", Ws, 39.2, 1.75), _on_wall("sign_n1", "sign_b", Wn, 13.2, 1.75),
            _on_wall("sign_n2", "sign_c", Wn, 20.3, 1.60), _on_wall("sign_e1", "sign_a", We, 12.4, 1.75),
            _on_wall("notice_n", "notice_board", Wn, 17.6, 1.55, size=(0.03, 1.20, 0.90)),
            _on_wall("lpg_sign", "wall_plate", Ww, 32.5, 2.35, size=(0.004, 0.60, 0.30), text="lpg_store"),
        ]
        ys = y0 + 0.75
        out += [
            _on_floor("pallets_s1", "pallet_stack", xs + 0.2, ys, size=(1.24, 1.04, 0.85), n=7),
            _on_floor("pallets_s2", "pallet_stack", xs + 1.75, ys, size=(1.24, 1.04, 1.09), n=9),
            _on_floor("bin_s1", "bin", 19.0 if xs + 3.2 < 19.0 else xs + 3.2, y0 + 0.62, math.pi),
            _on_floor("drum_s1", "steel_drum", 28.3, y0 + 0.50), _on_floor("drum_s2", "steel_drum", 29.05, y0 + 0.52),
            _on_floor("drum_s3", "plastic_drum", 29.75, y0 + 0.42), _on_floor("drum_s4", "plastic_drum", 30.3, y0 + 0.44),
            _on_floor("pallet_truck_s", "pallet_truck", 32.6, y0 + 1.75, math.pi), _on_floor("hand_truck_s", "hand_truck", 25.0, y0 + 0.36, math.pi),
            _on_floor("fan_s", "drum_fan", 39.0, y0 + 0.40, 0.0), _on_floor("bin_s2", "bin", 41.6, y0 + 0.62, math.pi),
            _on_floor("cart_s", "cart", 43.4, y0 + 1.0, math.pi / 2),
            _on_floor("pallets_n1", "pallet_stack", 4.6, y1 - 0.75, size=(1.24, 1.04, 1.21), n=10),
            _on_floor("pallets_n2", "pallet_stack", 6.1, y1 - 0.75, size=(1.24, 1.04, 0.61), n=5),
            _on_floor("bin_n1", "bin", 10.4, y1 - 0.62, 0.0), _on_floor("bin_n2", "bin", 11.35, y1 - 0.62, 0.0),
            _on_floor("pallet_truck_n", "pallet_truck", 16.0, y1 - 1.75, 0.0), _on_floor("hand_truck_n", "hand_truck", 19.2, y1 - 0.36, 0.0),
            _on_floor("cone_n1", "cone", 23.0, y1 - 0.35), _on_floor("cone_n2", "cone", 23.45, y1 - 0.33), _on_floor("cone_n3", "cone", 25.9, y1 - 0.36),
            _on_floor("fan_n", "drum_fan", 26.9, y1 - 0.40, math.pi),
            _on_floor("drum_e1", "steel_drum", x1 - 0.50, 11.5), _on_floor("drum_e2", "steel_drum", x1 - 0.50, 12.25),
            _on_floor("bin_e1", "bin", x1 - 0.62, 15.3, -math.pi / 2),
            _on_floor("bin_ne", "bin", x1 - 0.62, y1 - 0.9, -math.pi / 2), _on_floor("drum_ne", "plastic_drum", x1 - 0.45, y1 - 1.9),
        ]
    else:
        out += [
            _on_wall("door_s", "door", Ws, 4.6, 1.07, size=(0.09, 1.02, 2.14)), _on_wall("exit_s", "exit_sign", Ws, 4.6, 2.45, size=(0.05, 0.31, 0.19)),
            _on_wall("door_e", "door", We, 17.6, 1.07, size=(0.09, 1.02, 2.14)), _on_wall("exit_e", "exit_sign", We, 17.6, 2.45, size=(0.05, 0.31, 0.19)),
            _on_wall("ext_s1", "ext_board", Ws, 8.8, 1.35), _on_wall("ext_w1", "ext_board", Ww, 11.6, 1.35), _on_wall("ext_e1", "ext_board", We, 10.6, 1.35),
            _on_wall("panel_s", "fuse_cabinet", Ws, 12.9, 0.86, conduit=True), _on_wall("box_w", "fuse_box", Ww, 13.2, 1.45, conduit=True),
            _on_wall("alarm_s", "fire_alarm", Ws, 5.5, 1.25, size=(0.045, 0.10, 0.14)), _on_wall("alarm_e", "fire_alarm", We, 16.7, 1.25, size=(0.045, 0.10, 0.14)),
            _on_wall("sign_s1", "sign_a", Ws, 7.8, 1.75), _on_wall("sign_w1", "sign_b", Ww, 10.5, 1.75), _on_wall("sign_e1", "sign_c", We, 9.6, 1.60),
            _on_wall("notice_w", "notice_board", Ww, 15.4, 1.55, size=(0.03, 1.20, 0.90)),
            _on_floor("pallet_truck_s", "pallet_truck", 10.6, y0 + 1.75, math.pi), _on_floor("hand_truck_s", "hand_truck", 7.0, y0 + 0.36, math.pi),
            _on_floor("bin_s1", "bin", 14.6, y0 + 0.62, math.pi), _on_floor("bin_s2", "bin", 15.55, y0 + 0.62, math.pi),
            _on_floor("drum_s1", "steel_drum", 16.9, y0 + 0.50), _on_floor("drum_s2", "plastic_drum", 17.6, y0 + 0.42),
            _on_floor("fan_s", "drum_fan", 2.2, y0 + 0.40, 0.0),
            _on_floor("pallet_truck_w", "pallet_truck", x0 + 1.0, 12.9, -math.pi / 2), _on_floor("cone_w1", "cone", x0 + 0.40, 16.6),
            _on_floor("cone_w2", "cone", x0 + 0.42, 17.1), _on_floor("bin_w", "bin", x0 + 0.62, 18.4, math.pi / 2),
            _on_floor("bin_e1", "bin", x1 - 0.62, 11.6, -math.pi / 2), _on_floor("drum_e1", "steel_drum", x1 - 0.50, 12.9),
            _on_floor("drum_e2", "steel_drum", x1 - 0.50, 13.65), _on_floor("fan_e", "drum_fan", x1 - 0.42, 15.4, math.pi / 2),
            _on_floor("cart_e", "cart", x1 - 0.75, 1.7, 0.0),
        ]
    return out


# ====================================================================================================== dressing: drawers
def _deck_material(S):
    """Steel roof deck from below, painted white: the normal map of a trapezoidal rib profile on a 0.2 m pitch (flat
    crown, sloped webs, flat valley), a faint soot line in each valley."""
    name = "v_roof_deck"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image
    Wd = 256
    x = (np.arange(Wd) + 0.5) / Wd
    hgt = np.interp(x, [0.0, 0.30, 0.42, 0.88, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0]) * 0.038       # valley 38 mm above the crown
    slope = np.gradient(hgt, 0.2 / Wd)
    nrm = np.stack([-slope, np.zeros(Wd), np.ones(Wd)], 1)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    nimg = np.tile(np.uint8((nrm * 0.5 + 0.5) * 255)[None], (8, 1, 1))
    shade = 0.80 - 0.10 * (hgt / 0.038)
    dimg = np.tile(np.uint8(np.stack([shade, shade, shade * 0.99], 1) * 255)[None], (8, 1, 1))
    return pbr(S, name, save_tex(S, "roof_deck_d.png", Image.fromarray(dimg)), save_tex(S, "roof_deck_n.png", Image.fromarray(nimg)),
               rough_const=0.55, scale=(1 / 0.2, 1.0))


def _d_roof_deck(S, world, it, path):
    b = Buf()
    p, h = it["p"], it["half"]
    b.quad((p[0], p[1], ROOF_Z), (0, 1, 0), (1, 0, 0), (2 * h[1], 2 * h[0]), st=(0, 0, 2 * h[1], 2 * h[0]))    # ribs run along x
    b.box((p[0], p[1], ROOF_Z + 0.07), (2 * h[0], 2 * h[1], 0.12), bevel=0.0)
    b.write(S, path, _deck_material(S))


def _d_truss(S, world, it, path):
    """Parallel-chord roof truss across the building (double-angle chords, single-angle web), bearing on the walls."""
    p, h = it["p"], it["half"]
    zt, zb = ROOF_Z - 0.2 - 0.07, ROOF_Z - 0.2 - TRUSS_DEPTH + 0.06
    y0, y1 = p[1] - h[1], p[1] + h[1]
    b = Buf()
    b.box((p[0], p[1], zt), (0.16, y1 - y0, 0.13), bevel=0.006)
    b.box((p[0], p[1], zb), (0.16, y1 - y0 - 1.2, 0.11), bevel=0.006)
    n = max(4, int(round((y1 - y0) / 1.8 / 2)) * 2)
    ys = np.linspace(y0 + 0.3, y1 - 0.3, n + 1)
    for k in range(n):
        a = np.array([p[0], ys[k], zt if k % 2 == 0 else zb])
        c = np.array([p[0], ys[k + 1], zb if k % 2 == 0 else zt])
        d = c - a
        Lg = float(np.linalg.norm(d))
        ey = d / Lg
        ex = np.array([1.0, 0.0, 0.0])
        b.box((a + c) / 2, (0.075, Lg, 0.075), R=np.stack([ex, ey, np.cross(ex, ey)], 1), bevel=0.004)
        if k % 2 == 1 and k < n - 1:
            b.box((p[0], ys[k + 1], (zt + zb) / 2), (0.06, 0.06, zt - zb), bevel=0.003)
    for yy in (y0 + 0.09, y1 - 0.09):                                     # bearing plates on the walls
        b.box((p[0], yy, zt - 0.02), (0.30, 0.18, 0.20), bevel=0.006)
    b.write(S, path, mat(S, "paint_white"))


def _d_purlins(S, world, it, path):
    """Purlins along the building on top of the trusses, carrying the deck; a bridging line under them at mid-bay."""
    p, h = it["p"], it["half"]
    b = Buf()
    for yy in np.arange(p[1] - h[1] + PURLIN, p[1] + h[1] - 0.2, PURLIN):
        b.box((p[0], float(yy), ROOF_Z - 0.10), (2 * h[0], 0.065, 0.20), bevel=0.004)
    b.write(S, path, mat(S, "paint_white"))


def _d_sprinklers(S, world, it, path):
    """Wet-pipe sprinkler system: a cross main and branch lines every 3 m with upright heads."""
    p, h = it["p"], it["half"]
    x0, x1, y0, y1 = p[0] - h[0], p[0] + h[0], p[1] - h[1], p[1] + h[1]
    pipes, heads = Buf(), Buf()
    xm = round((x0 + x1) / 2 / 6.0) * 6.0 + 0.25
    pipes.tube([(xm, y0 + 0.4, 8.52), (xm, y1 - 0.4, 8.52)], 0.057, n=12)
    for yy in np.arange(y0 + 2.25, y1 - 0.5, 3.0):
        pipes.tube([(x0 + 0.5, float(yy), 8.62), (x1 - 0.5, float(yy), 8.62)], 0.021, n=8)
        pipes.tube([(xm, float(yy), 8.52), (xm, float(yy), 8.62)], 0.021, n=8)
        for xx in np.arange(x0 + 1.5, x1 - 0.5, 3.0):
            heads.cyl((float(xx), float(yy), 8.665), 0.012, 0.05, n=8)
            heads.cyl((float(xx), float(yy), 8.70), 0.032, 0.004, n=10)
    pipes.write(S, path + "/pipes", mat(S, "paint_red"))
    heads.write(S, path + "/heads", pbr(S, "v_brass", color=(0.55, 0.40, 0.14), metal=1.0, rough_const=0.35))


def _highbay_proto(S):
    """Linear LED high-bay housing (origin: centre of the light-emitting face, which render._lights places): body and
    bezel around and above the emitter, heat-sink fins, driver, two pendant rods to the purlin above."""
    P = proto(S, "highbay")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    white, fins, black = Buf(), Buf(), Buf()
    white.box((0, 0, 0.045), (1.27, 0.47, 0.07), bevel=0.01)
    for sy in (-1, 1):                                                      # bezel round the emitter (1.2 x 0.4 m)
        white.box((0, sy * 0.2175, -0.004), (1.27, 0.035, 0.028), bevel=0.004)
    for sx in (-1, 1):
        white.box((sx * 0.6175, 0, -0.004), (0.035, 0.40, 0.028), bevel=0.004)
    for x in np.linspace(-0.55, 0.55, 23):
        fins.box((x, 0, 0.098), (0.004, 0.40, 0.036), bevel=0.0)
    white.box((0, 0, 0.125), (0.32, 0.12, 0.05), bevel=0.006)               # driver
    hang = ROOF_Z - 0.20 - LAMP_Z
    for sx in (-1, 1):
        black.tube([(sx * 0.5, 0, 0.08), (sx * 0.5, 0, hang)], 0.004, n=6)
        white.box((sx * 0.5, 0, hang - 0.012), (0.05, 0.07, 0.024), bevel=0.003)
    black.tube([(0.12, 0, 0.15), (0.14, 0.02, 0.28), (0.10, 0.05, hang - 0.02)], 0.006, n=6)     # supply cord
    lens = Buf()
    lens.quad((0, 0, 0.004), (0, 1, 0), (1, 0, 0), (0.40, 1.20))
    white.write(S, P + "/body", mat(S, "paint_white"))
    fins.write(S, P + "/fins", mat(S, "galv"))
    black.write(S, P + "/cords", mat(S, "black"))
    # the lit diffuser. The RectLight that render._lights places here carries the light but is not itself visible to
    # the camera, so the housing would look switched off; this lens glows at LENS_NITS, a small fraction of the source
    lens.write(S, P + "/lens", pbr(S, "v_highbay_lens", color=(0.9, 0.9, 0.88), rough_const=0.6, emissive=(1.0, 0.98, 0.94, LENS_EMISSION)))
    return P


def _d_highbay(S, world, it, path):
    inst(S, path, _highbay_proto(S), (it["p"][0], it["p"][1], LAMP_Z))


def _label_atlas(S, row, labels):
    """Rack location labels on one sheet: white, the location in large type, a level colour bar, a Code 128 barcode,
    an arrow to the level it marks. Returns (material, {text: st rectangle})."""
    from PIL import Image, ImageDraw
    cw, ch, cols = 340, 128, 12
    rows = (len(labels) + cols - 1) // cols
    img = Image.new("RGB", (cw * cols, ch * max(rows, 1)), (238, 238, 234))
    d = ImageDraw.Draw(img)
    colours = {0: (40, 120, 60), 1: (220, 150, 10), 2: (30, 80, 160)}
    st = {}
    for i, (text, level, down) in enumerate(labels):
        x, y = (i % cols) * cw, (i // cols) * ch
        d.rectangle([x + 2, y + 2, x + cw - 3, y + ch - 3], outline=(60, 60, 60), width=2)
        d.rectangle([x + 4, y + 4, x + 44, y + ch - 5], fill=colours.get(level, (90, 90, 90)))
        ax, ay = x + 24, y + ch / 2
        s = 1 if down else -1
        d.polygon([(ax - 14, ay - s * 6), (ax + 14, ay - s * 6), (ax, ay + s * 26)], fill=(255, 255, 255))
        d.rectangle([ax - 6, ay - s * 30 if s > 0 else ay + 6, ax + 6, ay - s * 6 if s > 0 else ay + 30], fill=(255, 255, 255))
        d.text((x + 54, y + 8), text, font=font(52), fill=(10, 10, 10))
        K.draw_barcode(d, (x + 54, y + 74, x + cw - 10, y + ch - 12), text)
        H = ch * max(rows, 1)
        st[text] = (x / (cw * cols), 1.0 - (y + ch) / H, (x + cw) / (cw * cols), 1.0 - y / H)
    return pbr(S, f"v_rack_labels_{row}", save_tex(S, f"rack_labels_{row}.png", img), rough_const=0.5), st


def _d_rack_internals(S, world, it, path):
    """What a rack row holds besides its colliders: wire decks on the beam levels, location labels on the beams'
    aisle faces, row spacers across the flue of a back-to-back row, wire-mesh backing on a single row's open back."""
    L = world.scene.layout
    row = it["row"]
    frames, beams, _ = _rack_groups([b for b in L.boxes if b["material"] in ("rack_upright", "rack_beam") and b["name"].startswith(row + "_")])
    k = 0
    for (r, bay, lv), pair in sorted(beams.items()):
        if len(pair) != 2:
            continue
        xa, xb = sorted(b["centre"][0] for b in pair)
        h = pair[0]["half"]
        depth = (xb - xa) + 2 * h[0] - 2 * (2 * h[0] - 0.018) - 0.004          # between the two beams' steps
        width = h[1] - 0.012
        zt = pair[0]["centre"][2] + h[2] - 0.0015
        P = _deck_proto(S, depth, width)
        for s in (-1, 1):
            inst(S, f"{path}/deck{k}", P, ((xa + xb) / 2, pair[0]["centre"][1] + s * (h[1] / 2), zt))
            k += 1
    # labels
    locs = {n: v for n, v in L.zones.get("rack_locations", {}).items()}
    letter = row[0]
    fi = int(row[1:]) if row[1:].isdigit() else 0
    mine = []
    xs_row = sorted({round(b["centre"][0], 3) for bs in beams.values() for b in bs})
    for name, v in sorted(locs.items()):
        r, face, bay, lv, slot = name.split("-")
        if r != letter or not xs_row:
            continue
        if not (min(xs_row) - 0.1 < (v["x0"] + v["x1"]) / 2 < max(xs_row) + 0.1):
            continue
        mine.append((name, v, f"{r}-{bay}-{int(lv) + 1}{'AB'[int(slot)]}", int(lv)))
    if mine:
        from ..facility import BEAM_LEVELS, BEAM_H
        material, st = _label_atlas(S, row, [(t, lv, lv == 0) for _, _, t, lv in mine])
        b = Buf()
        for name, v, text, lv in mine:
            sgn = -1.0 if v["face"] == "w" else 1.0                         # the aisle side of the beam
            zb = BEAM_LEVELS[max(lv, 1) - 1] - BEAM_H / 2
            dy = (-0.20 if lv == 0 else 0.20) if lv < 2 else 0.0             # floor and first-level labels share a beam
            b.quad((v["aisle_x"] + sgn * 0.0009, v["y"] + dy * sgn, zb), (0, sgn, 0), (0, 0, 1), (0.150, 0.056), st=st[text])
        b.write(S, path + "/labels", material)
    # row spacers across a flue (this row has a neighbour 6 in behind it) and mesh backing on an open back
    ups = [b for b in L.boxes if b["material"] == "rack_upright"]
    mine_x = sorted({round(b["centre"][0], 3) for b in ups if b["name"].startswith(row + "_")})
    other = [b for b in ups if b["name"][0] == letter and not b["name"].startswith(row + "_")]
    sp, back = Buf(), Buf()
    if other and fi == 0:
        ox = sorted({round(b["centre"][0], 3) for b in other})
        gap0, gap1 = max(mine_x) + 0.038, min(ox) - 0.038
        if 0.02 < gap1 - gap0 < 0.5:
            for (r, kk), posts in frames.items():
                for z in (1.45, 3.05, 4.55):
                    sp.box(((gap0 + gap1) / 2, posts[0]["centre"][1], z), (gap1 - gap0, 0.05, 0.05), bevel=0.003)
    if not other:                                                           # a single row: mesh on the side away from its aisle
        face = next((v["face"] for n, v in locs.items() if n.split("-")[0] == letter), "e")
        xb_ = min(mine_x) if face == "e" else max(mine_x)
        if min(abs(xb_ - L.extent[0]), abs(xb_ - L.extent[1])) > 1.0:        # not for a row backed onto a wall
            ys = sorted({round(p[0]["centre"][1], 3) for p in frames.values()})
            for a, c in zip(ys[:-1], ys[1:]):
                back.quad((xb_, (a + c) / 2, 2.55), (0, 1, 0), (0, 0, 1), (c - a - 0.08, 4.5), st=(0, 0, (c - a) / 0.05, 4.5 / 0.05), both=True)
                for z in (0.31, 2.55, 4.79):
                    sp.box((xb_, (a + c) / 2, z), (0.022, c - a - 0.08, 0.03), bevel=0.002)
            back.write(S, path + "/back_mesh", _mesh_material(S, "square"))
    sp.write(S, path + "/row_spacers", mat(S, "paint_blue"))


def _d_row_sign(S, world, it, path):
    """Row identifier on the end frame: a board between the posts, readable from the cross aisle."""
    p = it["p"]
    R = rot_z(it["yaw"])
    b, f = Buf(), Buf()
    b.box(p, (0.60, 0.012, 0.44), R=R, bevel=0.002)
    n = R @ np.array([0.0, -1.0, 0.0])
    f.quad(np.array(p) + n * 0.0068, R @ np.array([1.0, 0, 0]), (0, 0, 1), (0.58, 0.42))
    b.write(S, path + "/board", mat(S, "galv"))
    f.write(S, path + "/face", _plate_material(S, f"row_{it['text']}", [(it["text"], 1.0), ("ROW", 0.28)], (512, 384), fg=(245, 245, 240),
                                               bg=(12, 50, 120), border=(245, 245, 240)))


def _d_aisle_sign(S, world, it, path):
    """Aisle number hung over the aisle mouth from the roof, readable from both directions."""
    p = it["p"]
    b, f, w = Buf(), Buf(), Buf()
    b.box(p, (1.0, 0.03, 0.60), bevel=0.004)
    m = _plate_material(S, f"aisle_{it['text']}", [("AISLE", 0.34), (it["text"], 1.0)], (640, 384), fg=(20, 20, 20), bg=(236, 196, 20), border=(20, 20, 20))
    f.quad((p[0], p[1] - 0.0158, p[2]), (1, 0, 0), (0, 0, 1), (0.98, 0.58))
    f.quad((p[0], p[1] + 0.0158, p[2]), (-1, 0, 0), (0, 0, 1), (0.98, 0.58))
    for sx in (-1, 1):
        w.tube([(p[0] + sx * 0.42, p[1], p[2] + 0.30), (p[0] + sx * 0.42, p[1], ROOF_Z - 0.2)], 0.0025, n=5)
    b.write(S, path + "/board", mat(S, "paint_dark"))
    f.write(S, path + "/faces", m)
    w.write(S, path + "/wires", mat(S, "galv"))


def _d_conveyor_internals(S, world, it, path):
    """Conveyor parts between and on its side frames: the roller shafts in their axle strips, the chain guard on the
    drive side, the guide rails' posts, cross ties, the gear motor under the discharge end."""
    conv = world.scene.layout.zones["conveyor"]
    (xa, xb), (ya, yb), top = conv["x"], conv["y"], conv["top"]
    from ..machines import ROLLER_PITCH, ROLLER_R
    n = int(round((xb - xa) / ROLLER_PITCH))
    zc = top - ROLLER_R
    ft = top / 2 - 0.02 + (top / 2 - 0.02)                                  # top of the side frames (their collision boxes)
    steel, blue, yellow, dark = Buf(), Buf(), Buf(), Buf()
    for i in range(n):
        x = xa + (i + 0.5) * ROLLER_PITCH
        steel.cyl((x, ya - 0.012, zc), 0.0085, 0.036, axis=1, n=6)          # hex shaft stubs into the south axle strip
    blue.box(((xa + xb) / 2, ya - 0.024, ft + 0.018), (xb - xa, 0.008, 0.036), bevel=0.0)        # south axle strip
    blue.box(((xa + xb) / 2, yb + 0.035, ft + 0.019), (xb - xa, 0.11, 0.036), bevel=0.004)       # chain guard, drive side
    for x in np.arange(xa + 0.5, xb, 1.0):                                  # guide rail posts on both frames
        for yy in (ya - 0.06, yb + 0.06):
            yellow.box((float(x), yy, (ft + top + 0.055) / 2), (0.035, 0.012, top + 0.055 - ft), bevel=0.002)
    for x in np.linspace(xa + 0.35, xb - 0.05, 6):                          # cross ties under the rollers
        blue.box((float(x), (ya + yb) / 2, 0.20), (0.05, yb - ya - 0.04, 0.08), bevel=0.003)
    xm = xb - 0.75                                                          # gear motor on a base between the frames
    dark.box((xm, (ya + yb) / 2 + 0.30, 0.03), (0.50, 0.40, 0.02), bevel=0.003)
    dark.cyl((xm, (ya + yb) / 2 + 0.12, 0.16), 0.085, 0.30, axis=1, n=20)
    dark.box((xm, (ya + yb) / 2 + 0.38, 0.16), (0.20, 0.20, 0.24), bevel=0.012)
    steel.write(S, path + "/shafts", mat(S, "steel"))
    blue.write(S, path + "/frame_parts", mat(S, "paint_ltblue"))
    yellow.write(S, path + "/guide_posts", mat(S, "paint_yellow"))
    dark.write(S, path + "/drive", mat(S, "paint_grey"))


def _d_photo_eye(S, world, it, path):
    """Retro-reflective photo-eye across the conveyor: sensor on the south frame, reflector on the north, on brackets."""
    p, h = it["p"], it["half"]
    conv = world.scene.layout.zones["conveyor"]
    z = conv["top"] + 0.05
    galv, black, lens = Buf(), Buf(), Buf()
    for s, yy in ((-1, p[1] - h[1] + 0.03), (1, p[1] + h[1] - 0.03)):
        galv.box((p[0], yy + s * 0.012, z - 0.12), (0.03, 0.004, 0.32), bevel=0.0)             # bracket
        galv.box((p[0], yy - s * 0.02, z - 0.27), (0.03, 0.06, 0.004), bevel=0.0)
        if s < 0:
            black.box((p[0], yy, z), (0.022, 0.04, 0.05), bevel=0.003)
            lens.cyl((p[0], yy + 0.021, z), 0.007, 0.002, axis=1, n=10)
        else:
            lens.box((p[0], yy, z), (0.05, 0.006, 0.05), bevel=0.001)
    galv.write(S, path + "/brackets", mat(S, "galv"))
    black.write(S, path + "/sensor", mat(S, "plastic_black"))
    lens.write(S, path + "/lens", pbr(S, "v_reflector", color=(0.75, 0.10, 0.06), rough_const=0.2))


def _d_press_top(S, world, it, path):
    """On the press's roof (above 3.2 m): the main motor with its flywheel guard, a stack light."""
    p, h = it["p"], it["half"]
    z0 = p[2] - h[2]
    g, s = Buf(), Buf()
    g.box((p[0] + 0.15, p[1], z0 + 0.20), (0.9, 0.9, 0.40), bevel=0.02)
    g.cyl((p[0] - 0.45, p[1], z0 + 0.20), 0.17, 0.42, axis=1, n=24)
    s.cyl((p[0] - 0.60, p[1] + 0.45, z0 + 0.20), 0.02, 0.40, n=10)
    g.write(S, path + "/drive", mat(S, "paint_green"))
    s.write(S, path + "/mast", mat(S, "galv"))
    for k, (col, lit) in enumerate((((0.05, 0.45, 0.08), True), ((0.75, 0.45, 0.02), False), ((0.6, 0.03, 0.02), False))):
        b = Buf()
        b.cyl((p[0] - 0.60, p[1] + 0.45, z0 + 0.43 + 0.065 * k), 0.032, 0.06, n=14)
        b.write(S, f"{path}/lamp{k}", pbr(S, f"v_stack{k}", color=col, rough_const=0.25, emissive=(*col, 900.0) if lit else None))


def _d_convex_mirror(S, world, it, path):
    """Convex traffic mirror hung from the roof on a rod, looking along its yaw."""
    p = it["p"]
    z = it["z"]
    R = rot_z(it["yaw"])
    n = 24
    prof = [(0.30 * math.sin(a), 0.085 * math.cos(a)) for a in np.linspace(math.pi / 2, 0.0, 12)]      # shallow dome
    b, bk, rod = Buf(), Buf(), Buf()
    ang = np.linspace(0, 2 * math.pi, n + 1)
    V, Nn = [], []
    for (r, d) in prof:
        for a in ang:
            V.append(R @ np.array([d, r * math.cos(a), r * math.sin(a)]))
            nn = np.array([d / 0.085 ** 2 * 0.3 ** 2 if r > 0 else 1.0, r * math.cos(a), r * math.sin(a)])
            Nn.append(R @ (nn / np.linalg.norm(nn)))
    m = len(prof)
    F = []
    for i in range(m - 1):
        for j in range(n):
            a, c = i * (n + 1) + j, (i + 1) * (n + 1) + j
            F += [(a, c, c + 1), (a, c + 1, a + 1)]
    c0 = np.array([p[0], p[1], z])
    b.add(np.array(V) + c0, Nn, np.zeros((len(V), 2)), F, fix=True)
    bk.cyl(c0 + R @ np.array([-0.012, 0, 0]), 0.315, 0.024, R=R @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T, n=24)
    rod.tube([c0 + R @ np.array([-0.03, 0, 0.30]), (c0[0], c0[1], z + 0.55), (c0[0], c0[1], ROOF_Z - 0.2)], 0.012, n=8)
    b.write(S, path + "/mirror", pbr(S, "v_mirror", color=(0.92, 0.93, 0.94), metal=1.0, rough_const=0.02))
    bk.write(S, path + "/back", mat(S, "paint_dark"))
    rod.write(S, path + "/rod", mat(S, "galv"))


def _d_speed_sign(S, world, it, path):
    """Speed limit sign on a post with a bolted base."""
    p, z = it["p"], it["z"]
    R = rot_z(it["yaw"])
    post, face = Buf(), Buf()
    post.cyl((p[0], p[1], (z + 0.2) / 2), 0.03, z + 0.2, n=14)
    post.box((p[0], p[1], 0.006), (0.22, 0.22, 0.012), bevel=0.002)
    n = R @ np.array([1.0, 0.0, 0.0])
    post.cyl(np.array([p[0], p[1], z]) + n * 0.034, 0.225, 0.004, R=R @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T, n=32)
    face.quad(np.array([p[0], p[1], z]) + n * 0.0368, R @ np.array([0, 1.0, 0]), (0, 0, 1), (0.45, 0.45))
    from PIL import Image, ImageDraw
    key = "v_plate_speed"
    if key not in S.mats:
        img = Image.new("RGB", (512, 512), (120, 122, 124))
        d = ImageDraw.Draw(img)
        d.ellipse([4, 4, 507, 507], fill=(200, 20, 16)); d.ellipse([60, 60, 451, 451], fill=(240, 240, 235))
        f = font(250); w = d.textlength("8", font=f)
        d.text(((512 - w) / 2, 96), "8", font=f, fill=(15, 15, 15))
        f2 = font(64); w2 = d.textlength("km/h", font=f2)
        d.text(((512 - w2) / 2, 352), "km/h", font=f2, fill=(15, 15, 15))
        pbr(S, key, save_tex(S, "plate_speed.png", img), rough_const=0.4)
    post.write(S, path + "/post", mat(S, "galv"))
    face.write(S, path + "/face", S.mats[key])


def _wall_asset(S, path, kind, p, yaw, half):
    """A kit prop modelled on a wall (origin on the wall plane, +x out of it), placed with its back on the wall."""
    key = WALL_ASSETS[kind][0]
    P = proto(S, "wall_" + kind)
    if not has(S, P):
        S.UsdGeom.Xform.Define(S.stage, P)
        src = tex(key)
        if src:
            S.stage.DefinePrim(P + "/asset").GetReferences().AddReference(src)
        else:
            b = Buf(); b.box((half[0], 0, 0), 2 * np.array(WALL_ASSETS[kind][1]) / 2, bevel=0.005)
            b.write(S, P + "/box", mat(S, "paint_grey"))
    lo = K.asset_bounds(tex(key))[0] if tex(key) else np.array([0.0, 0.0, -WALL_ASSETS[kind][1][2] / 2])
    hi = K.asset_bounds(tex(key))[1] if tex(key) else -lo
    R = rot_z(yaw)
    back = np.array(p) - R @ np.array([half[0] if abs(math.cos(yaw)) > 0.5 else half[1], 0.0, 0.0])
    base = back - np.array([0.0, 0.0, (lo[2] + hi[2]) / 2]) - R @ np.array([0.0, (lo[1] + hi[1]) / 2, 0.0])
    inst(S, path, P, base, quat_z(yaw))


def _d_wall_asset(S, world, it, path):
    _wall_asset(S, path, it["kind"], it["p"], it["yaw"], it["half"])
    if it.get("conduit"):                                                   # EMT conduit from the enclosure up the wall
        p, h = np.array(it["p"]), np.array(it["half"])
        R = rot_z(it["yaw"])
        back = p - R @ np.array([h[0] if abs(math.cos(it["yaw"])) > 0.5 else h[1], 0, 0]) + R @ np.array([0.022, 0, 0])
        b = Buf()
        for dy in (-0.06, 0.06):
            q = back + R @ np.array([0, dy, 0])
            b.tube([(q[0], q[1], p[2] + h[2]), (q[0], q[1], ROOF_Z - 0.25)], 0.011, n=8)
            for z in np.arange(p[2] + h[2] + 0.6, ROOF_Z - 0.4, 1.5):
                b.box((q[0], q[1], float(z)), (0.03, 0.03, 0.02), R=R, bevel=0.002)
        b.write(S, path + "_conduit", mat(S, "galv"))


def _floor_asset(S, path, kind, p, yaw, half):
    P, size = asset_proto(S, "prop_" + kind, ASSETS[kind][0], fallback=ASSETS[kind][1])
    inst(S, path, P, (p[0], p[1], p[2] - half[2]), quat_z(yaw))


def _d_floor_asset(S, world, it, path):
    _floor_asset(S, path, it["kind"], it["p"], it["yaw"], it["half"])


def _d_cone(S, world, it, path):
    """Traffic cone, 70 cm: PVC cone with two reflective collars on a black base. (The kit's cone glows: its
    reflective bands are emissive and lit the wall behind it.)"""
    P = proto(S, "cone")
    if not has(S, P):
        S.UsdGeom.Xform.Define(S.stage, P)
        o, w, k = Buf(), Buf(), Buf()
        k.box((0, 0, 0.015), (0.36, 0.36, 0.03), bevel=0.008)
        r = lambda z: 0.135 - (0.135 - 0.028) * (z - 0.03) / 0.67                     # the cone's radius at height z
        for z0, z1, buf in ((0.03, 0.33, o), (0.33, 0.43, w), (0.43, 0.50, o), (0.50, 0.57, w), (0.57, 0.70, o)):
            buf.cyl((0, 0, (z0 + z1) / 2), r(z0), z1 - z0, n=24, caps=(z1 == 0.70), r2=r(z1))
        o.write(S, P + "/pvc", pbr(S, "v_cone_orange", color=(0.85, 0.20, 0.02), rough_const=0.45))
        w.write(S, P + "/collars", pbr(S, "v_cone_white", color=(0.82, 0.82, 0.80), rough_const=0.25))
        k.write(S, P + "/base", mat(S, "rubber"))
    p = it["p"]
    inst(S, path, P, (p[0], p[1], p[2] - it["half"][2]), quat_z(it["yaw"]))


def _d_exit_sign(S, world, it, path):
    """Exit sign above a door (the prop's legend faces its -y)."""
    P, size = asset_proto(S, "prop_exit_sign", ASSETS["exit_sign"][0], fallback=ASSETS["exit_sign"][1])
    p = it["p"]
    inst(S, path, P, (p[0], p[1], p[2] - it["half"][2]), quat_z(it["yaw"] + math.pi / 2))


def _d_fire_alarm(S, world, it, path):
    P, size = asset_proto(S, "prop_fire_alarm", ASSETS["fire_alarm"][0], fallback=ASSETS["fire_alarm"][1])
    p = it["p"]
    inst(S, path, P, (p[0], p[1], p[2] - it["half"][2]), quat_z(it["yaw"] + math.pi))       # the prop's face looks along its -x


def _d_door(S, world, it, path):
    """Hollow-metal personnel door in its frame on the wall face: leaf with a vision lite, panic bar, closer."""
    p, h = np.array(it["p"]), np.array(it["half"])
    R = rot_z(it["yaw"])
    W_, H_ = 1.02, 2.14
    back = -(h[0] if abs(math.cos(it["yaw"])) > 0.5 else h[1])

    def at(x, y, z):
        return np.array([p[0], p[1], 0.0]) + R @ np.array([back + x, y, 0.0]) + np.array([0, 0, z])
    frame, leaf, steel, glass = Buf(), Buf(), Buf(), Buf()
    for sy in (-1, 1):
        frame.box(at(0.02, sy * (W_ / 2 - 0.025), H_ / 2), (0.04, 0.05, H_), R=R, bevel=0.004)
    frame.box(at(0.02, 0, H_ - 0.025), (0.04, W_, 0.05), R=R, bevel=0.004)
    leaf.box(at(0.014, 0, (H_ - 0.05) / 2 + 0.004), (0.022, W_ - 0.10, H_ - 0.058), R=R, bevel=0.003)
    glass.quad(at(0.0256, 0.20, 1.50), R @ np.array([0, 1.0, 0]), (0, 0, 1), (0.14, 0.62))
    steel.box(at(0.0262, 0.20, 1.50), (0.002, 0.20, 0.68), R=R, bevel=0.0)
    steel.box(at(0.055, 0.0, 1.02), (0.05, W_ - 0.22, 0.06), R=R, bevel=0.012)                    # panic bar
    steel.box(at(0.045, -0.30, H_ - 0.14), (0.05, 0.28, 0.06), R=R, bevel=0.008)                  # closer
    steel.box(at(0.026, 0, 0.14), (0.002, W_ - 0.12, 0.26), R=R, bevel=0.0)                       # kick plate
    frame.write(S, path + "/frame", mat(S, "paint_dark"))
    leaf.write(S, path + "/leaf", mat(S, "paint_ltblue"))
    steel.write(S, path + "/hardware", mat(S, "galv"))
    glass.write(S, path + "/lite", pbr(S, "v_dark_glass", color=(0.02, 0.025, 0.03), rough_const=0.04, spec=1.0))


def _d_notice_board(S, world, it, path):
    """Shift notice board: a framed pin board with sheets."""
    p, h = np.array(it["p"]), np.array(it["half"])
    R = rot_z(it["yaw"])
    n = R @ np.array([1.0, 0, 0])
    b, f = Buf(), Buf()
    b.box(p, (0.03, 1.20, 0.90), R=R, bevel=0.004)
    from PIL import Image, ImageDraw
    key = "v_plate_notice"
    if key not in S.mats:
        img = Image.new("RGB", (1024, 768), (168, 140, 96))
        d = ImageDraw.Draw(img)
        rng = np.random.default_rng(12)
        d.rectangle([0, 0, 1023, 90], fill=(20, 60, 120)); d.text((30, 18), "SAFETY  &  SHIFT NOTICES", font=font(52), fill=(240, 240, 235))
        for k in range(8):
            x, y = 40 + (k % 4) * 245 + int(rng.integers(-10, 10)), 130 + (k // 4) * 310 + int(rng.integers(-10, 10))
            d.rectangle([x, y, x + 200, y + 270], fill=(240, 240, 234) if k % 3 else (245, 225, 120))
            d.text((x + 12, y + 10), ("DOCK SCHEDULE", "LOCKOUT / TAGOUT", "PPE REQUIRED", "5S AUDIT", "FORKLIFT CHECKS", "EMERGENCY EXITS", "WEEKLY KPI", "NEAR MISSES")[k],
                   font=font(18), fill=(20, 20, 20))
            for r in range(10):
                d.line([x + 12, y + 50 + r * 21, x + 188 - int(rng.integers(0, 60)), y + 50 + r * 21], fill=(110, 110, 110), width=3)
        pbr(S, key, save_tex(S, "plate_notice.png", img), rough_const=0.7)
    f.quad(p + n * 0.0156, R @ np.array([0, 1.0, 0]), (0, 0, 1), (1.14, 0.84))
    b.write(S, path + "/frame", mat(S, "galv"))
    f.write(S, path + "/face", S.mats[key])


def _d_wall_plate(S, world, it, path):
    p = np.array(it["p"])
    R = rot_z(it["yaw"])
    n = R @ np.array([1.0, 0, 0])
    f = Buf()
    f.quad(p + n * 0.002, R @ np.array([0, 1.0, 0]), (0, 0, 1), (2 * max(it["half"][0], it["half"][1]), 2 * it["half"][2]))
    f.write(S, path, _plate_material(S, "lpg_store", [("LP-GAS CYLINDER STORAGE", 0.6), ("AUTHORISED PERSONNEL ONLY", 0.4)], (1024, 512),
                                     fg=(20, 20, 20), bg=(236, 196, 20), border=(20, 20, 20)))


def _d_pallet_stack(S, world, it, path):
    _pallet_stack(S, path, it["p"], it["half"], seed_of(it["name"]), n=it.get("n"))


def _d_extinguisher_stand(S, world, it, path):
    """A fire extinguisher on a floor stand with its sign."""
    p = it["p"]
    b, f = Buf(), Buf()
    b.box((p[0], p[1], 0.012), (0.30, 0.30, 0.024), bevel=0.006)
    b.box((p[0] - 0.12, p[1], 0.56), (0.03, 0.03, 1.10), bevel=0.004)
    b.box((p[0] - 0.10, p[1], 0.98), (0.008, 0.22, 0.22), bevel=0.002)
    f.quad((p[0] - 0.0955, p[1], 0.98), (0, 1, 0), (0, 0, 1), (0.20, 0.20))
    b.write(S, path + "/stand", mat(S, "paint_red"))
    f.write(S, path + "/sign", _plate_material(S, "fire_point", [("FIRE", 1.0), ("EXTINGUISHER", 0.5)], (384, 384), fg=(240, 240, 235), bg=(170, 20, 16)))
    key = KIT + "SM_FireExtinguisher_02.usd"
    if tex(key):
        lo, hi = K.asset_bounds(tex(key))[:2]
        P = proto(S, "kit_extinguisher")
        if not has(S, P):
            S.UsdGeom.Xform.Define(S.stage, P)
            S.stage.DefinePrim(P + "/asset").GetReferences().AddReference(tex(key))
        inst(S, path + "/extinguisher", P, (p[0] - 0.10 - lo[0], p[1], 0.024 - lo[2]))


def _d_safety_station(S, world, it, path):
    """First-aid and fire point on a floor stand (the kit's board with its first-aid box)."""
    p = it["p"]
    R = rot_z(it["yaw"])
    b = Buf()
    b.box((p[0], p[1], 0.010), (0.44, 0.60, 0.020), R=R, bevel=0.004)
    for sy in (-0.25, 0.25):
        q = np.array([p[0], p[1], 0.0]) + R @ np.array([0.0, sy, 0.0])
        b.box((q[0], q[1], 1.05), (0.04, 0.04, 2.10), bevel=0.004)
    b.write(S, path + "/stand", mat(S, "galv"))
    _wall_asset(S, path + "/board", "ext_board", (p[0], p[1], 1.45), it["yaw"], (0.0, 0.32, 0.585))


def _stencil_material(S):
    name = "v_stencil"
    if name not in S.mats:
        pbr(S, name, color=(0.80, 0.80, 0.77), rough_const=0.5)
    return S.mats[name]


def _d_stencil(S, world, it, path):
    """Stencilled floor paint: the text as a cut-out, its top toward the yaw direction's left (yaw 0: reads from -y)."""
    from PIL import Image, ImageDraw, ImageFilter
    text, hgt = it["text"], it["height"]
    key = "v_stencil_" + "".join(ch if ch.isalnum() else "_" for ch in text)
    wpx = 72 * len(text) + 24
    if key not in S.mats:
        img = Image.new("L", (wpx, 128), 0)
        d = ImageDraw.Draw(img)
        f = font(118, mono=True)
        w = d.textlength(text, font=f)
        d.text(((wpx - w) / 2, 0), text, font=f, fill=255)
        a = np.asarray(img, np.float32) / 255.0
        rng = np.random.default_rng(seed_of(text))
        for x in range(18, wpx, 34):                                       # stencil bridges
            a[:, x:x + 3] = 0.0
        wear = np.asarray(Image.fromarray(np.uint8(rng.random(a.shape) * 255)).filter(ImageFilter.GaussianBlur(2.0)), np.float32) / 255
        a *= np.clip(1.25 - 0.9 * wear, 0.35, 1.0)
        pbr(S, key, color=(0.80, 0.80, 0.77), rough_const=0.5, opacity_tex=save_tex(S, f"stencil_{key[10:]}.png", Image.fromarray(np.uint8(a * 255), "L")))
    R = rot_z(it["yaw"])
    w = hgt * wpx / 128.0
    b = Buf()
    p = it["p"]
    z = world.scene.ground_height(p[0], p[1]) + 0.0016
    b.quad((p[0], p[1], z), R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0]), (w, hgt))
    b.write(S, path, S.mats[key])


def _d_walkway(S, world, it, path):
    """Pedestrian walkway: a strip of green floor coating, scuffed and chipped, between two white edge lines."""
    from PIL import Image, ImageFilter
    name = "v_walkway"
    if name not in S.mats:
        rng = np.random.default_rng(8)
        N = 1024
        a = np.ones((N, N), np.float32)
        chips = np.asarray(Image.fromarray(np.uint8((rng.random((N, N)) < 0.00035) * 255)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
        wear = np.asarray(Image.fromarray(np.uint8(rng.random((N, N)) * 255)).filter(ImageFilter.GaussianBlur(40)), np.float32) / 255
        wear = np.clip((wear - wear.mean()) / max(float(wear.std()), 1e-6) * 0.22, -0.5, 0.5)
        grain = np.asarray(Image.fromarray(np.uint8(rng.random((N, N)) * 255)).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
        a = np.clip(a * (0.90 - np.clip(wear, 0, 1) * (0.6 + grain)) - 1.4 * chips, 0.0, 1.0)
        pbr(S, name, color=(0.055, 0.19, 0.105), rough_const=0.35, opacity_tex=save_tex(S, "walkway_mask.png", Image.fromarray(np.uint8(a * 255), "L")))
    p, h = it["p"], it["half"]
    z = world.scene.ground_height(p[0], p[1])
    b, e = Buf(), Buf()
    b.quad((p[0], p[1], z + 0.0012), (1, 0, 0), (0, 1, 0), (2 * h[0], 2 * h[1]), st=(0, 0, 2 * h[0] / 3.0, 2 * h[1] / 3.0))
    for sx in (-1, 1):
        e.quad((p[0] + sx * (h[0] - 0.04), p[1], z + 0.0017), (0, 1, 0), (-1, 0, 0), (2 * h[1], 0.075), st=(0.3 * sx, 0, 0.3 * sx + 2 * h[1] / 4.0, 1))
    b.write(S, path + "/coating", S.mats[name])
    e.write(S, path + "/edges", _line_material(S, "white"))


def _tyre_material(S, style):
    """Rubber left on the floor: 'arc' the scuffs of trucks turning (concentric arcs of a pair of tyres), 'run' the
    faint twin track of straight travel. Opacity masks; never more than half opaque."""
    name = f"v_tyre_{style}"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageDraw, ImageFilter
    rng = np.random.default_rng(21 if style == "arc" else 22)
    N = 1024
    m = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(m)
    if style == "arc":
        for _ in range(3):
            r = rng.uniform(0.45, 0.90) * N
            a0 = rng.uniform(8, 50); a1 = a0 + rng.uniform(14, 34)
            w = int(rng.uniform(0.16, 0.20) / 3.6 * N)                       # tyre width on a 3.6 m tile
            shade = int(rng.uniform(30, 70))
            for dr in (0.0, 0.95 / 3.6 * N):                                 # the two tyres of an axle
                d.arc([-r - dr, N - r - dr, r + dr, N + r + dr], 270 + a0, 270 + a1, fill=shade, width=w)
    else:
        w = int(0.19 / 1.6 * N)
        for x, shade in ((0.21 * N, int(rng.uniform(30, 55))), (0.80 * N, int(rng.uniform(30, 55)))):      # 0.95 m apart
            d.rectangle([x - w / 2, 0.12 * N, x + w / 2, 0.88 * N], fill=shade)
    a = np.asarray(m.filter(ImageFilter.GaussianBlur(7)), np.float32) / 255.0
    yy = (np.arange(N) + 0.5) / N
    if style == "run":
        a *= np.clip(np.minimum(yy - 0.10, 0.90 - yy) / 0.25, 0.0, 1.0)[:, None]          # the track fades in and out
    grain = np.asarray(Image.fromarray(np.uint8(rng.random((N, N)) * 255)).filter(ImageFilter.GaussianBlur(3.0)), np.float32) / 255.0
    a = np.clip(a * (0.25 + 1.5 * grain), 0.0, 0.26) * (0.62 if style == "arc" else 0.8)
    return pbr(S, name, color=(0.018, 0.018, 0.02), rough_const=0.55, opacity_tex=save_tex(S, f"tyre_{style}.png", Image.fromarray(np.uint8(a * 255), "L")))


def _d_tyre_marks(S, world, it, path):
    p = it["p"]
    R = rot_z(it["yaw"])
    w, ln = it["size"]
    z = world.scene.ground_height(p[0], p[1]) + 0.0011
    b = Buf()
    b.quad((p[0], p[1], z), R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0]), (w, ln))
    b.write(S, path, _tyre_material(S, it["style"]))


def _d_louvre(S, world, it, path):
    """Exhaust louvre set in the wall high up: a frame with fixed blades."""
    p, h = np.array(it["p"]), np.array(it["half"])
    R = rot_z(it["yaw"])
    w, hh = 1.30, 1.30
    back = -(h[0] if abs(math.cos(it["yaw"])) > 0.5 else h[1])
    b, d = Buf(), Buf()

    def at(x, y, z):
        return p + R @ np.array([back + x, y, z])
    for sy in (-1, 1):
        b.box(at(0.03, sy * (w / 2 - 0.03), 0), (0.06, 0.06, hh), R=R, bevel=0.004)
    for sz in (-1, 1):
        b.box(at(0.03, 0, sz * (hh / 2 - 0.03)), (0.06, w - 0.12, 0.06), R=R, bevel=0.004)
    tilt = math.radians(40)
    Rb = R @ np.array([[math.cos(tilt), 0, math.sin(tilt)], [0, 1, 0], [-math.sin(tilt), 0, math.cos(tilt)]])
    for z in np.linspace(-hh / 2 + 0.11, hh / 2 - 0.11, 11):
        b.box(at(0.03, 0, float(z)), (0.10, w - 0.13, 0.004), R=Rb, bevel=0.0)
    d.quad(at(0.004, 0, 0), R @ np.array([0, 1.0, 0]), (0, 0, 1), (w - 0.12, hh - 0.12))
    b.write(S, path + "/frame", mat(S, "galv"))
    d.write(S, path + "/dark", mat(S, "black"))


def _d_ceiling_fan(S, world, it, path):
    """High-volume low-speed ceiling fan hung from the roof: five 2.6 m aerofoil blades, hub motor, drop tube, guy wires."""
    p = it["p"]
    z = p[2] - 0.25
    alu, dark = Buf(), Buf()
    dark.cyl((p[0], p[1], z + 0.04), 0.26, 0.20, n=28)
    dark.cyl((p[0], p[1], (z + 0.14 + ROOF_Z - 0.2) / 2), 0.045, ROOF_Z - 0.2 - z - 0.14, n=12)
    for k in range(5):
        a = 2 * math.pi * k / 5 + 0.3
        R = rot_z(a) @ np.array([[1, 0, 0], [0, math.cos(0.14), -math.sin(0.14)], [0, math.sin(0.14), math.cos(0.14)]])
        alu.box(np.array([p[0], p[1], z]) + rot_z(a) @ np.array([1.52, 0, 0]), (2.55, 0.19, 0.022), R=R, bevel=0.008)
        alu.box(np.array([p[0], p[1], z + 0.02]) + rot_z(a) @ np.array([2.80, 0, 0.03]), (0.02, 0.19, 0.11), R=rot_z(a), bevel=0.004)   # winglet
        ga = a + 0.6
        dark.tube([(p[0] + 0.05 * math.cos(ga), p[1] + 0.05 * math.sin(ga), z + 0.45),
                   (p[0] + 1.7 * math.cos(ga), p[1] + 1.7 * math.sin(ga), ROOF_Z - 0.2)], 0.003, n=5) if k < 4 else None
    alu.write(S, path + "/blades", pbr(S, "v_fan_blade", color=(0.72, 0.73, 0.74), metal=1.0, rough_const=0.35))
    dark.write(S, path + "/hub", mat(S, "paint_dark"))


DRAWERS = dict(walkway=_d_walkway, louvre=_d_louvre, ceiling_fan=_d_ceiling_fan, tyre_marks=_d_tyre_marks, roof_deck=_d_roof_deck, truss=_d_truss, purlins=_d_purlins, sprinklers=_d_sprinklers, highbay=_d_highbay,
               rack_internals=_d_rack_internals, row_sign=_d_row_sign, aisle_sign=_d_aisle_sign,
               conveyor_internals=_d_conveyor_internals, photo_eye=_d_photo_eye, press_top=_d_press_top,
               convex_mirror=_d_convex_mirror, speed_sign=_d_speed_sign, extinguisher_stand=_d_extinguisher_stand,
               safety_station=_d_safety_station, stencil=_d_stencil, door=_d_door, exit_sign=_d_exit_sign,
               fire_alarm=_d_fire_alarm, notice_board=_d_notice_board, wall_plate=_d_wall_plate, pallet_stack=_d_pallet_stack,
               ext_board=_d_wall_asset, fuse_cabinet=_d_wall_asset, fuse_box=_d_wall_asset, sign_a=_d_wall_asset,
               sign_b=_d_wall_asset, sign_c=_d_wall_asset, pallet_truck=_d_floor_asset, hand_truck=_d_floor_asset,
               drum_fan=_d_floor_asset, steel_drum=_d_floor_asset, plastic_drum=_d_floor_asset, bin=_d_floor_asset,
               cone=_d_cone, cart=_d_floor_asset)

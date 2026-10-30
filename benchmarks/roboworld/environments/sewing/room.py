"""The sample room around the sewing cell, for the renderer only: walls with windows and a door, two more lockstitch
workstations, the cutting table with a cutting mat, a laid-out fabric and paper pattern pieces, pattern hooks, shelving
with fabric bolts, a dress form, an ironing station, a pegboard and signage. Nothing here has collision or physics:
every object is outside every robot's sweep (the cell's table, machine, pedestal and plies are built by render.py from
the simulated state). Dimensions are typical of an apparel sample room [estimate: industrial cutting tables 0.90 m
high; lockstitch workstations 1.2 x 0.53 m at 0.76 m as the cell's own; dress forms 1.3-1.6 m on a tripod stand].
Textures generated here (cutting-mat grid, kraft pattern paper, signs) are drawn deterministically with PIL; the
fictional signage names no brand.
"""
import math
from pathlib import Path

import numpy as np

from . import layout as LY

ROOM = dict(x=(-3.2, 3.6), y=(-2.6, 2.8), h=3.0, wall=0.10)
WINDOWS = [(-0.9, 1.5), (1.4, 1.5)]          # (y centre, width) in the left wall
WINDOW_Z = (0.95, 2.45)                      # sill and head above the floor
DOOR = dict(y=1.7, w=0.95, h=2.10)           # in the right wall
STATIONS = (2.15, -2.35)                     # x offsets of the two other workstations from the cell's table
CUTTING = dict(x=(-1.35, 1.25), y=(1.55, 2.55), h=0.90)


# ------------------------------------------------------------------------------------------------ textures
def _tex_cutting_mat(folder):
    from PIL import Image, ImageDraw
    p = Path(folder)/"tex_cutting_mat.png"
    if p.exists():
        return p
    W = 2048
    im = Image.new("RGB", (W, W), (34, 92, 66))            # green self-healing mat
    d = ImageDraw.Draw(im)
    for k in range(0, 61):                                   # 1 cm grid over 0.6 m tile, heavier every 5 cm
        x = int(k*W/60)
        c, w = ((210, 225, 210), 3) if k % 5 == 0 else ((120, 165, 140), 1)
        d.line([(x, 0), (x, W)], fill=c, width=w)
        d.line([(0, x), (W, x)], fill=c, width=w)
    for k in range(0, 61, 5):
        d.text((int(k*W/60) + 4, 6), str(k), fill=(230, 240, 230))
    d.line([(0, 0), (W, W)], fill=(200, 215, 200), width=2)     # 45 degree guide
    im.save(p)
    return p


def _tex_kraft(folder, name, seed, label):
    """Brown pattern paper with a pencil outline, notches, a grain line and the piece's label."""
    from PIL import Image, ImageDraw, ImageFilter
    p = Path(folder)/f"tex_pattern_{name}.png"
    if p.exists():
        return p
    rng = np.random.default_rng(seed)
    W, H = 1024, 1024
    base = np.full((H, W, 3), (196, 162, 118), float) + rng.normal(0, 6, (H, W, 1))
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(im)
    m = 40
    d.rectangle([m, m, W - m, H - m], outline=(70, 60, 55), width=4)
    d.line([(W//2, 160), (W//2, H - 160)], fill=(60, 50, 45), width=3)               # grain line
    d.polygon([(W//2, 150), (W//2 - 18, 190), (W//2 + 18, 190)], fill=(60, 50, 45))
    d.polygon([(W//2, H - 150), (W//2 - 18, H - 190), (W//2 + 18, H - 190)], fill=(60, 50, 45))
    for y in (H//3, 2*H//3):                                                          # notches
        d.line([(m, y), (m + 30, y)], fill=(60, 50, 45), width=4)
        d.line([(W - m - 30, y), (W - m, y)], fill=(60, 50, 45), width=4)
    d.text((W//2 + 30, H//2 - 40), label, fill=(50, 40, 40))
    d.text((W//2 + 30, H//2), "CUT 2  SA 10 MM", fill=(50, 40, 40))
    im.save(p)
    return p


def _tex_sign(folder, name, lines, bg, fg):
    from PIL import Image, ImageDraw
    p = Path(folder)/f"tex_sign_{name}.png"
    if p.exists():
        return p
    im = Image.new("RGB", (800, 560), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 788, 548], outline=fg, width=10)
    for k, (txt, scale) in enumerate(lines):
        y = 70 + k*110
        # PIL's default bitmap font, scaled by drawing on a small image and resizing
        small = Image.new("RGB", (len(txt)*6 + 4, 12), bg)
        ImageDraw.Draw(small).text((2, 0), txt, fill=fg)
        big = small.resize((int(small.width*scale), int(small.height*scale)), Image.NEAREST)
        im.paste(big, ((800 - big.width)//2, y))
    im.save(p)
    return p


# ------------------------------------------------------------------------------------------------ helpers
def _glass(A, name):
    """Window glass (OmniGlass, clear, IOR 1.52)."""
    if name in A.mats:
        return A.mats[name]
    Sdf, Gf, UsdShade = A.Sdf, A.Gf, A.UsdShade
    m = UsdShade.Material.Define(A.stage, "/Looks/" + name)
    sh = UsdShade.Shader.Define(A.stage, str(m.GetPath()) + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.52)
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.96, 0.98, 0.97))
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    A.mats[name] = m
    return m


def _lathe(A, path, profile, mat, sides=48, center=(0., 0., 0.), scale_xy=(1., 1.)):
    """Surface of revolution about z from a (radius, z) profile, elliptical section by scale_xy."""
    cx, cy, cz = center
    a = np.linspace(0, 2*math.pi, sides, endpoint=False)
    V, F = [], []
    for r, z in profile:
        for t in a:
            V.append((cx + r*math.cos(t)*scale_xy[0], cy + r*math.sin(t)*scale_xy[1], cz + z))
    n = len(profile)
    for i in range(n - 1):
        for k in range(sides):
            k1 = (k + 1) % sides
            F.append([i*sides + k, i*sides + k1, (i + 1)*sides + k1, (i + 1)*sides + k])
    return A.mesh(path, V, F, mat, smooth=True)


def _quad(A, path, corners, mat, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    from pxr import UsdGeom, Sdf, Vt
    m = A.mesh(path, corners, [[0, 1, 2, 3]], mat)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([tuple(map(float, u)) for u in uv]))
    m.CreateDoubleSidedAttr(True)
    return m


# ------------------------------------------------------------------------------------------------ the room
def build(A, zf, VA, weave_texture, machine_folder, folder, P):
    """Add the room, its dressing and the other workstations to the stage held by `A` (render.py's asset helper)."""
    x0, x1 = ROOM["x"]
    y0, y1 = ROOM["y"]
    h, t = ROOM["h"], ROOM["wall"]
    ft, wt = VA.texture("laminate_floor_02"), VA.texture("beige_wall_002")
    floor = A.pbr("floor", (180, 160, 130), rough=.6, diff=ft["diff"], rough_tex=ft["rough"], normal=ft["nor_gl"],
                  tile_m=1.5)
    wall = A.pbr("wall", (238, 236, 230), rough=.9, diff=wt["diff"], rough_tex=wt["rough"], normal=wt["nor_gl"],
                 tile_m=2.)
    skirting = A.pbr("skirting", (70, 72, 76), rough=.5)
    frame = A.pbr("window_frame", (236, 236, 232), rough=.35)
    A.box("/World/Room/floor", (x0, y0, zf - .02), (x1, y1, zf), floor, uv=True)
    A.box("/World/Room/ceiling", (x0, y0, zf + h), (x1, y1, zf + h + .05), A.pbr("ceiling", (238, 238, 234), rough=.9))
    # back and front walls, the right wall around the door, the left wall around the windows
    A.box("/World/Room/wall_back", (x0, y1, zf), (x1, y1 + t, zf + h), wall, uv=True)
    A.box("/World/Room/wall_front", (x0, y0 - t, zf), (x1, y0, zf + h), wall, uv=True)
    dy0, dy1 = DOOR["y"] - DOOR["w"]/2, DOOR["y"] + DOOR["w"]/2
    A.box("/World/Room/wall_right_a", (x1, y0, zf), (x1 + t, dy0, zf + h), wall, uv=True)
    A.box("/World/Room/wall_right_b", (x1, dy1, zf), (x1 + t, y1, zf + h), wall, uv=True)
    A.box("/World/Room/wall_right_c", (x1, dy0, zf + DOOR["h"]), (x1 + t, dy1, zf + h), wall, uv=True)
    ys = [y0] + [v for c, w in WINDOWS for v in (c - w/2, c + w/2)] + [y1]
    for k in range(0, len(ys), 2):
        A.box(f"/World/Room/wall_left_pier_{k}", (x0 - t, ys[k], zf), (x0, ys[k + 1], zf + h), wall, uv=True)
    glass = _glass(A, "window_glass")
    for k, (c, w) in enumerate(WINDOWS):
        a, b = c - w/2, c + w/2
        A.box(f"/World/Room/wall_left_sill_{k}", (x0 - t, a, zf), (x0, b, zf + WINDOW_Z[0]), wall, uv=True)
        A.box(f"/World/Room/wall_left_head_{k}", (x0 - t, a, zf + WINDOW_Z[1]), (x0, b, zf + h), wall, uv=True)
        zl, zh = zf + WINDOW_Z[0], zf + WINDOW_Z[1]
        for nm, lo, hi in (("bottom", (x0 - t, a, zl), (x0 + .02, b, zl + .05)), ("top", (x0 - t, a, zh - .05), (x0, b, zh)),
                           ("left", (x0 - t, a, zl), (x0, a + .05, zh)), ("right", (x0 - t, b - .05, zl), (x0, b, zh)),
                           ("mullion", (x0 - t, c - .025, zl), (x0, c + .025, zh)),
                           ("transom", (x0 - t, a, zl + 1.05), (x0, b, zl + 1.10))):
            A.box(f"/World/Room/window_{k}_{nm}", lo, hi, frame)
        A.box(f"/World/Room/window_{k}_sill", (x0, a - .03, zl - .02), (x0 + .16, b + .03, zl + .01), frame)
        A.box(f"/World/Room/window_{k}_glass", (x0 - t/2 - .003, a + .05, zl + .05), (x0 - t/2 + .003, b - .05, zh - .05),
              glass)
    # door: frame, leaf (painted), lever handle, kick plate
    leaf = A.pbr("door_leaf", (150, 110, 75), rough=.45)
    A.box("/World/Room/door_frame_l", (x1 - .02, dy0 - .05, zf), (x1 + t, dy0, zf + DOOR["h"] + .05), frame)
    A.box("/World/Room/door_frame_r", (x1 - .02, dy1, zf), (x1 + t, dy1 + .05, zf + DOOR["h"] + .05), frame)
    A.box("/World/Room/door_frame_t", (x1 - .02, dy0 - .05, zf + DOOR["h"]), (x1 + t, dy1 + .05, zf + DOOR["h"] + .05), frame)
    A.box("/World/Room/door_leaf", (x1 + .01, dy0, zf), (x1 + .055, dy1, zf + DOOR["h"]), leaf)
    A.box("/World/Room/door_kick", (x1 + .005, dy0 + .02, zf + .01), (x1 + .012, dy1 - .02, zf + .20),
          A.pbr("kick_plate", (190, 192, 196), rough=.3, metal=.8))
    A.cyl("/World/Room/door_handle", (x1 - .005, dy0 + .10, zf + 1.02), (x1 - .005, dy0 + .24, zf + 1.02), .011,
          A.pbr("handle_steel", (200, 202, 206), rough=.25, metal=1.))
    # skirting
    for nm, lo, hi in (("back", (x0, y1 - .012, zf), (x1, y1, zf + .10)), ("front", (x0, y0, zf), (x1, y0 + .012, zf + .10)),
                       ("left", (x0, y0, zf), (x0 + .012, y1, zf + .10)), ("right_a", (x1 - .012, y0, zf), (x1, dy0, zf + .10)),
                       ("right_b", (x1 - .012, dy1, zf), (x1, y1, zf + .10))):
        A.box(f"/World/Room/skirting_{nm}", lo, hi, skirting)

    # ---- the other two lockstitch workstations (the cell's table and machine, unthreaded, idle), each with a stool
    # and a bundle bin of cut parts
    T = LY.TABLE
    for k, dx in enumerate(STATIONS):
        root = f"/World/Stations/s{k}"
        _workstation(A, root, dx, zf, VA, machine_folder, P)
        A.reference(f"{root}/stool", VA.model("metal_stool_01"), (dx + .15, T["y"][0] - .45, zf), 10. + 25*k)
        _bundle_bin(A, f"{root}/bin", (dx + .75, T["y"][0] - .15, zf), weave_texture, k)

    # ---- the cutting table at the back: laminate top on a steel frame, a cutting mat, a length of fabric laid out
    # with paper pattern pieces on it, an L-square, shears, weights, a tape measure
    cx, cy, ch = CUTTING["x"], CUTTING["y"], CUTTING["h"]
    top = A.pbr("cutting_top", (232, 232, 228), rough=.35)
    steel = A.pbr("cutting_steel", (60, 62, 66), rough=.45, metal=.6)
    A.box("/World/Cutting/top", (cx[0], cy[0], zf + ch - .03), (cx[1], cy[1], zf + ch), top)
    for kx, x in enumerate((cx[0] + .06, cx[1] - .06)):
        for ky, y in enumerate((cy[0] + .06, cy[1] - .06)):
            A.box(f"/World/Cutting/leg_{kx}{ky}", (x - .025, y - .025, zf), (x + .025, y + .025, zf + ch - .03), steel)
    A.box("/World/Cutting/shelf", (cx[0] + .06, cy[0] + .06, zf + .18), (cx[1] - .06, cy[1] - .06, zf + .20), steel)
    mat_tex = _tex_cutting_mat(folder)
    mat_m = A.pbr("cutting_mat", (255, 255, 255), rough=.7, diff=str(mat_tex), tile_m=.6)
    A.box("/World/Cutting/mat", (cx[0] + .05, cy[0] + .05, zf + ch), (cx[1] - .05, cy[1] - .05, zf + ch + .003), mat_m,
          uv=True)
    lin = VA.texture("rough_linen")
    denim = A.pbr("laid_fabric", (52, 70, 110), rough=.9, diff=weave_texture(), tint=(52, 70, 110), normal=lin["nor_gl"],
                  tile_m=.02)
    zt = zf + ch + .0035
    A.box("/World/Cutting/fabric", (cx[0] + .25, cy[0] + .15, zt), (cx[1] - .45, cy[1] - .12, zt + .0006), denim, uv=True)
    for k, (name, lo, hi, label) in enumerate((("front", (-0.95, 1.75), (-0.40, 2.30), "FRONT BODICE"),
                                               ("back", (-0.30, 1.72), (0.22, 2.32), "BACK BODICE"),
                                               ("sleeve", (0.30, 1.80), (0.62, 2.28), "SLEEVE"))):
        tex = _tex_kraft(folder, name, 7 + k, label)
        m = A.pbr(f"pattern_{name}", (255, 255, 255), rough=.8, diff=str(tex))
        _quad(A, f"/World/Cutting/pattern_{name}", [(lo[0], lo[1], zt + .0012), (hi[0], lo[1], zt + .0012),
                                                     (hi[0], hi[1], zt + .0012), (lo[0], hi[1], zt + .0012)], m)
    alu = A.pbr("aluminium_rule", (205, 208, 212), rough=.3, metal=1.)
    A.box("/World/Cutting/l_square_a", (0.72, 1.70, zt), (0.76, 2.30, zt + .002), alu)
    A.box("/World/Cutting/l_square_b", (0.72, 1.70, zt), (1.05, 1.74, zt + .002), alu)
    black = A.pbr("shear_handle", (25, 25, 28), rough=.5)
    A.box("/World/Cutting/shears_blade", (-1.20, 1.78, zt), (-1.02, 1.80, zt + .004), alu)
    A.box("/World/Cutting/shears_handle", (-1.27, 1.765, zt), (-1.19, 1.815, zt + .012), black)
    for k, (x, y) in enumerate(((-0.62, 1.66), (0.05, 2.38), (0.48, 1.70))):
        A.cyl(f"/World/Cutting/weight_{k}", (x, y, zt), (x, y, zt + .03), .03, A.pbr("weight_iron", (40, 42, 46),
                                                                                       rough=.4, metal=.7))
    A.reference("/World/Cutting/tape", VA.model("measuring_tape_01"), (0.95, 2.35, zt), 30.)
    # pattern hooks on the back wall: kraft pieces hanging from a rail
    A.cyl("/World/Wall/pattern_rail", (-1.2, y1 - .05, zf + 2.05), (0.9, y1 - .05, zf + 2.05), .012,
          A.pbr("rail_steel", (180, 182, 186), rough=.3, metal=1.))
    for k, x in enumerate(np.linspace(-1.05, 0.70, 6)):
        tex = _tex_kraft(folder, f"hang{k}", 20 + k, ["SKIRT", "CUFF", "COLLAR", "YOKE", "POCKET", "FACING"][k])
        m = A.pbr(f"pattern_hang{k}", (255, 255, 255), rough=.8, diff=str(tex))
        w, hh = .22 + .04*(k % 2), .55 + .1*(k % 3)
        yy = y1 - .055 - .004*k
        _quad(A, f"/World/Wall/pattern_hang{k}", [(x, yy, zf + 2.04 - hh), (x + w, yy, zf + 2.04 - hh),
                                                  (x + w, yy, zf + 2.04), (x, yy, zf + 2.04)], m)
    # ---- shelving with fabric bolts and thread boxes (right part of the back wall)
    A.reference("/World/Props/shelves", VA.model("steel_frame_shelves_01"), (2.35, y1 - .30, zf), 180.)
    cols = ((150, 40, 45), (40, 70, 120), (226, 214, 188), (60, 100, 70), (200, 200, 205), (120, 90, 60),
            (230, 230, 225), (30, 30, 34), (170, 140, 110))
    for k, col in enumerate(cols):
        mat = A.pbr(f"bolt_{k}", col, rough=.85, diff=weave_texture(), tint=col, tile_m=.05)
        x = 1.95 + .14*(k % 3)
        z = (.45, .95, 1.45)[k//3]
        A.cyl(f"/World/Props/bolt_{k}", (x, y1 - .55, zf + z + .06), (x, y1 - .06, zf + z + .06), .055, mat, sides=32)
    A.reference("/World/Props/drawer_cabinet", VA.model("drawer_cabinet"), (3.25, 0.10, zf), 270.)
    A.reference("/World/Props/wall_clock", VA.model("wall_clock"), (-0.2, y1 - .03, zf + 2.45), 180., ground=False)
    A.reference("/World/Props/cardboard_box", VA.model("cardboard_box_01"), (-2.6, 2.3, zf), 15.)
    A.reference("/World/Props/basket", VA.model("wicker_basket_01"), (-1.55, 1.95, zf), 0.)
    # ---- dress form with a muslin toile, on a tripod stand
    cover = A.pbr("form_cover", (226, 214, 190), rough=.9, diff=weave_texture(), tint=(226, 214, 190), tile_m=.02)
    toile = A.pbr("toile_muslin", (236, 226, 200), rough=.9, diff=weave_texture(), tint=(236, 226, 200), tile_m=.02)
    fx, fy = 3.05, 1.05
    torso = [(0.02, 0.0), (0.12, 0.02), (0.15, 0.10), (0.16, 0.20), (0.135, 0.33), (0.125, 0.40), (0.15, 0.50),
             (0.16, 0.56), (0.13, 0.63), (0.07, 0.68), (0.055, 0.72), (0.05, 0.78), (0.0, 0.80)]
    _lathe(A, "/World/Props/form_torso", torso, cover, center=(fx, fy, zf + 0.78), scale_xy=(1.0, 0.72))
    skirt = [(0.17, 0.0), (0.165, 0.10), (0.15, 0.22), (0.14, 0.30)]
    _lathe(A, "/World/Props/form_toile", [(r + .006, z) for r, z in skirt], toile, center=(fx, fy, zf + 0.58),
           scale_xy=(1.0, 0.74))
    pole = A.pbr("form_pole", (30, 30, 32), rough=.4, metal=.5)
    A.cyl("/World/Props/form_pole", (fx, fy, zf + .05), (fx, fy, zf + .80), .012, pole)
    for k in range(3):
        a = math.radians(90 + 120*k)
        A.cyl(f"/World/Props/form_leg_{k}", (fx, fy, zf + .06), (fx + .32*math.cos(a), fy + .32*math.sin(a), zf + .01),
              .012, pole)
    # ---- ironing station by the right wall: padded board on legs, a steam iron
    ix, iy = 3.05, -1.55
    pad = A.pbr("ironing_cover", (150, 160, 170), rough=.9, diff=weave_texture(), tint=(150, 160, 170), tile_m=.03)
    A.box("/World/Props/ironing_board", (ix - .19, iy - .65, zf + .86), (ix + .19, iy + .55, zf + .89), pad)
    A.cyl("/World/Props/ironing_leg_a", (ix, iy - .45, zf), (ix, iy + .35, zf + .86), .012, pole)
    A.cyl("/World/Props/ironing_leg_b", (ix, iy + .35, zf), (ix, iy - .45, zf + .86), .012, pole)
    A.box("/World/Props/iron_sole", (ix - .06, iy - .45, zf + .89), (ix + .06, iy - .21, zf + .91),
          A.pbr("iron_sole", (190, 192, 196), rough=.2, metal=1.))
    A.box("/World/Props/iron_body", (ix - .05, iy - .42, zf + .91), (ix + .05, iy - .24, zf + .99),
          A.pbr("iron_body", (230, 232, 236), rough=.3))
    # ---- signage on the front-left wall (fictional)
    s1 = _tex_sign(folder, "safety", [("SAMPLE ROOM", 7), ("KEEP FINGERS", 6), ("CLEAR OF NEEDLE", 6)], (245, 245, 240),
                   (30, 60, 120))
    s2 = _tex_sign(folder, "spec", [("SEAM ALLOWANCE 10 MM", 5), ("10-12 SPI  BACKTACK 3", 5), ("TEX 30 POLY", 5)],
                   (250, 240, 200), (60, 40, 30))
    # on the wall between the windows (WINDOWS: y -1.65..-0.15 and 0.65..2.15); read from inside the room, facing -x,
    # the text runs toward +y (mapped the other way it reads mirror-reversed)
    for k, (tex, y) in enumerate(((s1, -2.4), (s2, 0.0))):
        m = A.pbr(f"sign_{k}", (255, 255, 255), rough=.6, diff=str(tex))
        _quad(A, f"/World/Wall/sign_{k}", [(x0 + .005, y, zf + 1.45), (x0 + .005, y + .5, zf + 1.45),
                                            (x0 + .005, y + .5, zf + 1.80), (x0 + .005, y, zf + 1.80)], m,
              uv=((0, 0), (1, 0), (1, 1), (0, 1)))


def _workstation(A, root, dx, zf, VA, machine_folder, P):
    """The cell's industrial table and lockstitch head, shifted by dx along x (no robots, no plies)."""
    T = LY.TABLE
    tt = VA.texture("white_oak_veneer")
    top = A.pbr("table_top", (200, 180, 150), rough=.45, diff=tt["diff"], rough_tex=tt["rough"], tile_m=.6)
    edge = A.pbr("table_edge", (60, 60, 62), rough=.5)
    steel = A.pbr("k_leg_steel", (70, 72, 76), rough=.45, metal=.6)
    (x0, x1), (y0, y1), th = T["x"], T["y"], T["thickness"]
    x0, x1 = x0 + dx, x1 + dx
    bx, by = (LY.BED["x"][0] + dx, LY.BED["x"][1] + dx), LY.BED["y"]
    for nm, lo, hi in (("front", (x0, y0, -th), (x1, by[0], 0.)), ("back", (x0, by[1], -th), (x1, y1, 0.)),
                       ("left", (x0, by[0], -th), (bx[0], by[1], 0.)), ("right", (bx[1], by[0], -th), (x1, by[1], 0.))):
        A.box(f"{root}/top_{nm}", lo, hi, top, uv=True)
    A.box(f"{root}/edge_front", (x0, y0 - .003, -th), (x1, y0, 0.), edge)
    for k, x in enumerate((x0 + .06, x1 - .06)):
        A.box(f"{root}/leg_post_{k}", (x - .02, -.02, zf + .03), (x + .02, .02, -th), steel)
        A.box(f"{root}/leg_foot_{k}", (x - .02, y0 + .02, zf), (x + .02, y1 - .02, zf + .03), steel)
        A.box(f"{root}/leg_arm_{k}", (x - .015, y0 + .03, -th - .03), (x + .015, y1 - .03, -th), steel)
    A.box(f"{root}/crossbar", (x0 + .06, -.02, zf + .25), (x1 - .06, .02, zf + .29), steel)
    A.box(f"{root}/control_box", (dx + .18, -.02, -th - .20), (dx + .42, .17, -th - .02),
          A.pbr("control_box", (215, 215, 210), rough=.4))
    A.box(f"{root}/pedal", (dx - .05, -.35, zf), (dx + .10, -.10, zf + .04), A.pbr("pedal", (40, 40, 42), rough=.6))
    A.usd(f"{root}/machine", Path(machine_folder)/"machine_static.usd", (dx, 0., 0.))
    A.usd(f"{root}/needle_bar", Path(machine_folder)/"needle_bar.usd", (dx, 0., float(P.POINT_BDC + P.STROKE)))
    A.usd(f"{root}/presser_foot", Path(machine_folder)/"presser_foot.usd", (dx, 0., 0.))
    A.usd(f"{root}/feed_dog", Path(machine_folder)/"feed_dog.usd", (dx, 0., -0.001))
    hx, hy, hz = LY.HANDWHEEL["center"]
    A.usd(f"{root}/handwheel", Path(machine_folder)/"handwheel.usd", (hx + dx, hy, hz))


def _bundle_bin(A, root, pos, weave_texture, k):
    """A plastic bin holding a bundle of cut parts (stacked fabric rectangles with a ticket)."""
    x, y, z = pos
    plastic = A.pbr("bin_plastic", (40, 90, 150), rough=.5)
    A.box(f"{root}/floor", (x - .22, y - .16, z), (x + .22, y + .16, z + .01), plastic)
    for nm, lo, hi in (("l", (x - .22, y - .16, z), (x - .21, y + .16, z + .28)), ("r", (x + .21, y - .16, z), (x + .22, y + .16, z + .28)),
                       ("f", (x - .22, y - .16, z), (x + .22, y - .15, z + .28)), ("b", (x - .22, y + .15, z), (x + .22, y + .16, z + .28))):
        A.box(f"{root}/wall_{nm}", lo, hi, plastic)
    cols = [(226, 214, 188), (52, 70, 110), (150, 40, 45)]
    for j in range(6):
        col = cols[(j + k) % 3]
        m = A.pbr(f"bundle_{(j + k) % 3}", col, rough=.9, diff=weave_texture(), tint=col, tile_m=.02)
        A.box(f"{root}/part_{j}", (x - .17 + .005*j, y - .11, z + .01 + .006*j), (x + .15 - .004*j, y + .10, z + .015 + .006*j), m)
    A.box(f"{root}/ticket", (x - .05, y - .03, z + .05), (x + .05, y + .03, z + .051), A.pbr("ticket", (245, 240, 220), rough=.8))

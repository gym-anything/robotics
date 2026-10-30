"""The teaching kitchen around the station and what stands on its counter (visual; called from dressing.dress).

  shell      floor, four walls with a real window opening in the right wall and a door in the front wall, ceiling
  furniture  a stainless prep table with two stools behind the robot, dry-store shelving against the left wall, a
             hand-wash basin with its towel dispenser and a pedal bin at the right end of the back wall, the lesson board,
             a rail of aprons, a wall clock, a fire extinguisher, a radiator under the window
  props      the packets and jars the task's recipe was measured from, on the counter beside the work

Positions and outlines come from params (ROOM, WINDOW, DOOR, BOARD, WORK_TABLE, STOOLS, SHELVING, BASIN, DISPENSER,
BIN, APRON_HOOKS, PROP, PROPS); scene.furniture and scene.props give the same outlines static collision proxies.
Labels are generic (no brand is drawn); their text is generated here.
"""

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import geometry as GEO

CZ = P.COUNTER_Z
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"


# ====================================================================================== generated textures
def _font(size, bold=False, serif=False):
    from PIL import ImageFont
    f = (FONT.replace("Sans", "Serif") if serif else FONT).format("-Bold" if bold else "")
    return ImageFont.truetype(f, size) if Path(f).exists() else ImageFont.load_default()


def label_map(folder, name, title, lines=(), bg=(245, 242, 232), band=(30, 70, 140), fg=(30, 30, 30), size=(512, 640)):
    """A printed pack front: a coloured band with the product's name, a few lines under it. Returns the PNG's path."""
    from PIL import Image, ImageDraw
    W, H = size
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(0.16*H), W, int(0.46*H)], fill=band)
    f = _font(int(0.13*H), True)
    while d.textlength(title, font=f) > 0.9*W and f.size > 12:
        f = _font(f.size - 4, True)
    d.text((W//2, int(0.31*H)), title, font=f, fill=(250, 248, 240), anchor="mm")
    y = int(0.54*H)
    for k, ln in enumerate(lines):
        fk = _font(int((0.060 if k == 0 else 0.045)*H), k == 0)
        d.text((W//2, y), ln, font=fk, fill=fg, anchor="mm")
        y += int(0.085*H)
    d.rectangle([0, H - int(0.05*H), W, H], fill=band)
    p = Path(folder)/f"label_{name}.png"
    im.save(p)
    return p


def board_map(folder, world):
    """The lesson board: today's lesson in marker, with the task's own method on it."""
    from PIL import Image, ImageDraw
    from . import task as T
    spec = T.task_spec(world.config)
    W, H = 1800, 1050
    im = Image.new("RGB", (W, H), (246, 247, 245))
    d = ImageDraw.Draw(im)
    blue, red, black = (28, 52, 130), (170, 36, 30), (34, 34, 38)
    title, f = "Today:  " + spec["title"], _font(86, True)
    while d.textlength(title, font=f) > W - 140 and f.size > 30:
        f = _font(f.size - 4, True)
    d.text((70, 60 + (86 - f.size)//2), title, font=f, fill=blue)
    d.line([70, 175, W - 70, 175], fill=blue, width=6)
    y = 220
    words, cur, f = spec["instruction"].split(), "", _font(50)
    for wd in words:
        trial = (cur + " " + wd).strip()
        if d.textlength(trial, font=f) > W - 160:
            d.text((80, y), cur, font=f, fill=black)
            y += 70
            cur = wd
        else:
            cur = trial
    d.text((80, y), cur, font=f, fill=black)
    y += 120
    d.text((80, y), "Remember", font=_font(60, True), fill=red)
    y += 86
    for ln in ("- weigh everything before you start", "- keep it in the bowl", "- tools back in the crock, hob and mixer OFF",
               "- wash hands, wipe the station"):
        d.text((110, y), ln, font=_font(46), fill=black)
        y += 66
    p = Path(folder)/f"board_{world.config.task}.png"
    im.save(p)
    return p


def notice_map(folder):
    """A cork board with the kitchen's notices pinned to it (generated: no real document is reproduced)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(7)
    W, H = 1400, 1050
    cork = (rng.normal(0., 9., (H//4, W//4, 1)) + np.array([150, 112, 72])).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(cork, "RGB").resize((W, H), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    sheets = [((60, 60, 520, 700), (248, 248, 244), "KITCHEN RULES", (150, 30, 24),
               ("1  Wash hands before you start", "2  Hair tied back, apron on", "3  Knives stay on the board",
                "4  Hob OFF when you leave it", "5  Spills wiped at once", "6  Tools back where they live")),
              ((570, 90, 990, 560), (240, 246, 250), "ALLERGENS", (20, 80, 140),
               ("wheat   flour, cake mix", "egg     batters, mayonnaise", "milk    batters, pudding", "mustard relish, mayonnaise")),
              ((1030, 60, 1350, 480), (250, 247, 225), "CLEANING ROTA", (40, 110, 50),
               ("Mon  group A", "Tue  group B", "Wed  group C", "Thu  group D", "Fri  everyone")),
              ((600, 610, 1010, 990), (252, 238, 236), "IN CASE OF FIRE", (170, 20, 16),
               ("hob OFF at the knob", "leave by the door", "blanket by the basin", "do not use water on fat")),
              ((1060, 540, 1340, 960), (246, 246, 246), "THIS WEEK", (60, 60, 70),
               ("dry mixes", "muffins", "pancakes", "white sauce", "porridge")),
              ((90, 760, 500, 1000), (250, 250, 240), "WEIGH - MIX - CHECK", (120, 70, 20), ("grams, not cups", "read the method twice"))]
    for (x0, y0, x1, y1), bg, title, tc, lines in sheets:
        d.rectangle([x0 + 6, y0 + 8, x1 + 6, y1 + 8], fill=(96, 70, 46))
        d.rectangle([x0, y0, x1, y1], fill=bg)
        f = _font(40, True)
        while d.textlength(title, font=f) > (x1 - x0) - 40 and f.size > 14:
            f = _font(f.size - 2, True)
        d.text((x0 + 24, y0 + 26), title, font=f, fill=tc)
        d.line([x0 + 24, y0 + 84, x1 - 24, y0 + 84], fill=tc, width=3)
        yy = y0 + 106
        for ln in lines:
            d.text((x0 + 26, yy), ln, font=_font(25), fill=(40, 40, 44))
            yy += 44
        d.ellipse([0.5*(x0 + x1) - 11, y0 + 4, 0.5*(x0 + x1) + 11, y0 + 26], fill=(190, 30, 30))
    p = Path(folder)/"notice_board.png"
    im.save(p)
    return p


def sign_map(folder):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (400, 280), (20, 90, 160))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 389, 269], outline=(245, 245, 245), width=6)
    d.text((200, 100), "WASH YOUR", font=_font(50, True), fill=(250, 250, 250), anchor="mm")
    d.text((200, 180), "HANDS", font=_font(74, True), fill=(250, 250, 250), anchor="mm")
    p = Path(folder)/"sign_wash_hands.png"
    im.save(p)
    return p


# ====================================================================================== helpers
class Frame:
    """A prop's own frame on the counter: origin at its footprint's centre, turned by yaw about z."""

    def __init__(self, x, y, z, yaw):
        self.o = np.array([x, y, z], float)
        self.R = Rotation.from_euler("z", yaw)
        self.q = self.R.as_quat()

    def p(self, dx=0., dy=0., dz=0.):
        return self.o + self.R.apply([dx, dy, dz])

    def quat(self, extra=None):
        return self.q if extra is None else (self.R*extra).as_quat()


def textured(S, name, path, rough=0.6):
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    return S.mdl(name, "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(path).resolve()))),
                                   "reflection_roughness_constant": (T.Float, float(rough))})


def panel(S, path, centre, right, up, w, h, mat):
    """A flat printed panel (a label, a board's face, a sign): centre, its right and up directions, width and height."""
    c, r, u = np.asarray(centre, float), np.asarray(right, float), np.asarray(up, float)
    r, u = r/np.linalg.norm(r), u/np.linalg.norm(u)
    v = [c - 0.5*w*r - 0.5*h*u, c + 0.5*w*r - 0.5*h*u, c + 0.5*w*r + 0.5*h*u, c - 0.5*w*r + 0.5*h*u]
    n = np.cross(r, u)
    return S.mesh(path, v, [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], normals=[n]*4, mat=mat, double=True)


def color(S, name, rgb, rough=0.5, metal=0.0):
    return S.color(name, rgb, rough, metal)


# ====================================================================================== the room's shell
def shell(S, d, floor, plaster, tile, tile_top):
    """Floor, walls and ceiling. The right wall has a window opening (frame, sill, glass; daylight from the dome behind
    it); the front wall carries the door."""
    R = P.ROOM
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["z1"]
    S.quad("/Dressing/room/floor", [(x0, y0, 0.), (x1, y0, 0.), (x1, y1, 0.), (x0, y1, 0.)], floor, double=True)
    S.quad("/Dressing/room/back_tiles", [(x0, y1, 0.), (x1, y1, 0.), (x1, y1, tile_top), (x0, y1, tile_top)], tile, double=True)
    S.quad("/Dressing/room/back_paint", [(x0, y1, tile_top), (x1, y1, tile_top), (x1, y1, h), (x0, y1, h)], plaster, double=True)
    S.box("/Dressing/room/tile_trim", (x1 - x0, 0.008, 0.012), S.mat("ceramic_white"), p=(0.5*(x0 + x1), y1 - 0.004, tile_top + 0.006), r=0.003)
    S.quad("/Dressing/room/front", [(x1, y0, 0.), (x0, y0, 0.), (x0, y0, h), (x1, y0, h)], plaster, double=True)
    S.quad("/Dressing/room/left", [(x0, y0, 0.), (x0, y1, 0.), (x0, y1, h), (x0, y0, h)], plaster, double=True)
    S.quad("/Dressing/room/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)], S.mat("ceiling"), double=True)
    # the right wall around the window opening
    (wy0, wy1), (wz0, wz1), rv = P.WINDOW["y"], P.WINDOW["z"], P.WINDOW["reveal"]
    for name, (ya, yb, za, zb) in dict(below=(y0, y1, 0., wz0), above=(y0, y1, wz1, h), near=(wy1, y1, wz0, wz1),
                                       far=(y0, wy0, wz0, wz1)).items():
        S.quad(f"/Dressing/room/right_{name}", [(x1, yb, za), (x1, ya, za), (x1, ya, zb), (x1, yb, zb)], plaster, double=True)
    white = S.mat("frame_white")
    # reveal (the wall's thickness), sill, frame with one mullion and a transom, glass
    for name, c in dict(sill=[(x1, wy0, wz0), (x1 + rv, wy0, wz0), (x1 + rv, wy1, wz0), (x1, wy1, wz0)],
                        head=[(x1, wy1, wz1), (x1 + rv, wy1, wz1), (x1 + rv, wy0, wz1), (x1, wy0, wz1)],
                        jamb0=[(x1, wy0, wz1), (x1 + rv, wy0, wz1), (x1 + rv, wy0, wz0), (x1, wy0, wz0)],
                        jamb1=[(x1, wy1, wz0), (x1 + rv, wy1, wz0), (x1 + rv, wy1, wz1), (x1, wy1, wz1)]).items():
        S.quad(f"/Dressing/window/reveal_{name}", c, white, double=True)
    S.box("/Dressing/window/sill_board", (0.16, wy1 - wy0 + 0.10, 0.028), white, p=(x1 - 0.035, 0.5*(wy0 + wy1), wz0 - 0.014), r=0.006)
    xf = x1 + 0.07
    fw = 0.055
    for name, (size, pos) in dict(top=((0.05, wy1 - wy0, fw), (xf, 0.5*(wy0 + wy1), wz1 - 0.5*fw)),
                                  bottom=((0.05, wy1 - wy0, fw), (xf, 0.5*(wy0 + wy1), wz0 + 0.5*fw)),
                                  side0=((0.05, fw, wz1 - wz0), (xf, wy0 + 0.5*fw, 0.5*(wz0 + wz1))),
                                  side1=((0.05, fw, wz1 - wz0), (xf, wy1 - 0.5*fw, 0.5*(wz0 + wz1))),
                                  mullion=((0.045, 0.045, wz1 - wz0), (xf, 0.5*(wy0 + wy1), 0.5*(wz0 + wz1))),
                                  transom=((0.045, wy1 - wy0, 0.045), (xf, 0.5*(wy0 + wy1), wz1 - 0.38))).items():
        S.box(f"/Dressing/window/frame_{name}", size, white, p=pos, r=0.004)
    g = S.box("/Dressing/window/glass", (0.004, wy1 - wy0 - 2*fw, wz1 - wz0 - 2*fw), S.mats["glass"], p=(xf, 0.5*(wy0 + wy1), 0.5*(wz0 + wz1)), r=0.0005)
    S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
    S.tube("/Dressing/window/handle", [(xf - 0.03, 0.5*(wy0 + wy1) - 0.035, wz0 + 0.55), (xf - 0.055, 0.5*(wy0 + wy1) - 0.035, wz0 + 0.55),
                                       (xf - 0.055, 0.5*(wy0 + wy1) - 0.035, wz0 + 0.44)], 0.007, S.mat("steel_satin"), sides=10)
    # a panel radiator under the window
    rad = S.mat("frame_white")
    ry0, ry1 = wy0 + 0.15, wy1 - 0.15
    S.box("/Dressing/window/radiator", (0.06, ry1 - ry0, 0.56), rad, p=(x1 - 0.075, 0.5*(ry0 + ry1), 0.44), r=0.008)
    for k, yy in enumerate(np.arange(ry0 + 0.05, ry1 - 0.02, 0.05)):
        S.box(f"/Dressing/window/radiator_rib{k}", (0.012, 0.018, 0.52), rad, p=(x1 - 0.110, float(yy), 0.44), r=0.004)
    for k, yy in enumerate((ry0 + 0.12, ry1 - 0.12)):
        S.box(f"/Dressing/window/radiator_bracket{k}", (0.045, 0.03, 0.05), rad, p=(x1 - 0.022, yy, 0.62), r=0.002)
    S.tube("/Dressing/window/radiator_pipe", [(x1 - 0.075, ry0 + 0.03, 0.16), (x1 - 0.075, ry0 + 0.03, 0.0)], 0.008, S.mat("steel_satin"), sides=10)
    S.cylinder("/Dressing/window/radiator_valve", 0.016, 0.045, white, p=(x1 - 0.075, ry0 + 0.03, 0.19))
    # skirting
    skirt = S.mat("skirting")
    S.box("/Dressing/room/skirt_front", (x1 - x0, 0.012, 0.08), skirt, p=(0.5*(x0 + x1), y0 + 0.006, 0.04), r=0.003)
    S.box("/Dressing/room/skirt_left", (0.012, y1 - y0, 0.08), skirt, p=(x0 + 0.006, 0.5*(y0 + y1), 0.04), r=0.003)
    S.box("/Dressing/room/skirt_right", (0.012, y1 - y0, 0.08), skirt, p=(x1 - 0.006, 0.5*(y0 + y1), 0.04), r=0.003)


def door(S):
    """The door in the front wall: architrave, a laminate leaf with a vision panel, kick plate and lever handle."""
    y0 = P.ROOM["y0"]
    (dx0, dx1), dz = P.DOOR["x"], P.DOOR["z"]
    white, satin = S.mat("frame_white"), S.mat("steel_satin")
    leaf = S.mat("door_leaf")
    cx = 0.5*(dx0 + dx1)
    S.box("/Dressing/door/leaf", (dx1 - dx0, 0.044, dz), leaf, p=(cx, y0 + 0.030, 0.5*dz), r=0.003)
    for name, (size, pos) in dict(jamb0=((0.07, 0.07, dz + 0.07), (dx0 - 0.035, y0 + 0.035, 0.5*(dz + 0.07))),
                                  jamb1=((0.07, 0.07, dz + 0.07), (dx1 + 0.035, y0 + 0.035, 0.5*(dz + 0.07))),
                                  head=((dx1 - dx0 + 0.14, 0.07, 0.07), (cx, y0 + 0.035, dz + 0.035))).items():
        S.box(f"/Dressing/door/{name}", size, white, p=pos, r=0.006)
    S.box("/Dressing/door/vision", (0.16, 0.006, 0.62), S.mat("vision_glass"), p=(dx0 + 0.19, y0 + 0.054, 1.42), r=0.002)
    S.box("/Dressing/door/vision_frame", (0.20, 0.004, 0.66), satin, p=(dx0 + 0.19, y0 + 0.052, 1.42), r=0.002)
    S.box("/Dressing/door/kick_plate", (dx1 - dx0 - 0.03, 0.003, 0.20), satin, p=(cx, y0 + 0.0535, 0.115), r=0.001)
    S.box("/Dressing/door/push_plate", (0.09, 0.003, 0.28), satin, p=(dx0 + 0.09, y0 + 0.0535, 1.05), r=0.001)
    S.cylinder("/Dressing/door/rose", 0.026, 0.010, satin, p=(dx0 + 0.075, y0 + 0.057, 1.02), axis="Y")
    S.tube("/Dressing/door/lever", [(dx0 + 0.075, y0 + 0.060, 1.02), (dx0 + 0.075, y0 + 0.105, 1.02), (dx0 + 0.20, y0 + 0.105, 1.02)],
           0.009, satin, sides=12)
    for k, z in enumerate((0.25, 1.02, 1.80)):
        S.cylinder(f"/Dressing/door/hinge{k}", 0.009, 0.10, satin, p=(dx1 + 0.004, y0 + 0.058, z))
    # exit sign over the door
    S.box("/Dressing/door/exit_sign", (0.34, 0.03, 0.14), S.color("exit_green", (.02, .40, .12), .5, 0., emissive=(.10, 1., .30), emissive_intensity=300.),
          p=(cx, y0 + 0.03, dz + 0.22), r=0.004)


# ====================================================================================== furniture
def furniture(S, d, world):
    R = P.ROOM
    x0, x1, y0, y1 = R["x0"], R["x1"], R["y0"], R["y1"]
    steel, satin, white = S.mat("steel_brushed"), S.mat("steel_satin"), S.mat("frame_white")
    tex = d.tex
    # ---- prep table: top, undershelf, four legs, with bowls, a board and trays on it and crates under it
    t = P.WORK_TABLE
    (tx0, tx1), (ty0, ty1), zt = t["x"], t["y"], t["z"]
    cx, cy = 0.5*(tx0 + tx1), 0.5*(ty0 + ty1)
    S.box("/Dressing/table/top", (tx1 - tx0, ty1 - ty0, t["top"]), steel, p=(cx, cy, zt - 0.5*t["top"]), r=0.004)
    S.box("/Dressing/table/shelf", (tx1 - tx0 - 0.06, ty1 - ty0 - 0.06, 0.02), steel, p=(cx, cy, t["shelf_z"] - 0.01), r=0.003)
    lg = t["leg"]
    for i, lx in enumerate((tx0 + 0.5*lg, tx1 - 0.5*lg)):
        for j, ly in enumerate((ty0 + 0.5*lg, ty1 - 0.5*lg)):
            S.box(f"/Dressing/table/leg{i}{j}", (lg, lg, zt - t["top"] - 0.03), satin, p=(lx, ly, 0.5*(zt - t["top"] + 0.03)), r=0.004)
            S.cylinder(f"/Dressing/table/foot{i}{j}", 0.022, 0.03, S.mat("vent_dark"), p=(lx, ly, 0.015))
    for k in range(4):                                   # a nest of mixing bowls
        prof = GEO.bowl_profile(0.150 - 0.012*k, 0.110 - 0.006*k, 0.045)
        S.revolve(f"/Dressing/table/bowl{k}", GEO.wall_polygon(prof, 0.0012), S.mat("bowl_steel"), segments=72, crease_deg=50.,
                  p=(tx0 + 0.30, cy + 0.05, zt + 0.001 + 0.012*k))
    oak = S.mats.get("shelf_oak") or S.mat("wood")
    S.box("/Dressing/table/board", (0.40, 0.28, 0.022), oak, p=(cx + 0.05, cy - 0.06, zt + 0.011), r=0.004,
          q=Rotation.from_euler("z", 0.12).as_quat(), uv_per_m=1., grain=0)
    S.box("/Dressing/table/tray", (0.44, 0.32, 0.020), S.mat("aluminum"), p=(tx1 - 0.33, cy + 0.02, zt + 0.010), r=0.006)
    for k in range(3):                                   # folded cloths on the tray
        S.box(f"/Dressing/table/cloth{k}", (0.20, 0.14, 0.012), S.color(f"cloth{k}", ((.85, .85, .83), (.20, .35, .55), (.85, .85, .83))[k], .9, 0.),
              p=(tx1 - 0.33, cy + 0.02, zt + 0.026 + 0.012*k), r=0.004, q=Rotation.from_euler("z", 0.05*k).as_quat())
    d.ref("/Dressing/table/crate0", "plastic_crate_03", (tx0 + 0.36, cy, t["shelf_z"]), yaw=0.04)
    d.ref("/Dressing/table/crate1", "plastic_crate_01", (tx1 - 0.40, cy, t["shelf_z"]), yaw=1.57)
    # ---- two stools at the table
    for k, (sx_, sy_) in enumerate(P.STOOLS):
        d.ref(f"/Dressing/stool{k}", "bar_chair_round_01", (sx_, sy_, 0.), yaw=0.6*k + 0.3)
    # ---- dry-store shelving against the left wall, with boxes and crates
    sh = P.SHELVING
    scx, scy = 0.5*sum(sh["x"]), 0.5*sum(sh["y"])
    d.ref("/Dressing/shelving/rack", "steel_frame_shelves_01", (scx, scy, 0.), yaw=math.pi/2)
    stock(S, d, scx, scy)
    # ---- hand-wash basin, tap, soap, towel dispenser, sign, pedal bin
    bs = P.BASIN
    (bx0, bx1), (by0, by1), (bz0, bz1) = bs["x"], bs["y"], bs["z"]
    cer = S.mat("ceramic_white")
    bcx, bcy = 0.5*(bx0 + bx1), 0.5*(by0 + by1)
    wall = 0.022
    S.box("/Dressing/basin/floor", (bx1 - bx0, by1 - by0, wall), cer, p=(bcx, bcy, bz0 + 0.5*wall), r=0.008)
    for name, (size, pos) in dict(front=((bx1 - bx0, wall, bz1 - bz0), (bcx, by0 + 0.5*wall, 0.5*(bz0 + bz1))),
                                  left=((wall, by1 - by0, bz1 - bz0), (bx0 + 0.5*wall, bcy, 0.5*(bz0 + bz1))),
                                  right=((wall, by1 - by0, bz1 - bz0), (bx1 - 0.5*wall, bcy, 0.5*(bz0 + bz1))),
                                  back=((bx1 - bx0, 0.07, bz1 - bz0), (bcx, by1 - 0.035, 0.5*(bz0 + bz1)))).items():
        S.box(f"/Dressing/basin/{name}", size, cer, p=pos, r=0.008)
    S.cylinder("/Dressing/basin/waste", 0.022, 0.004, satin, p=(bcx, bcy - 0.02, bz0 + wall + 0.002))
    S.tube("/Dressing/basin/trap", [(bcx, bcy - 0.02, bz0), (bcx, bcy - 0.02, bz0 - 0.16), (bcx, bcy + 0.06, bz0 - 0.22),
                                    (bcx, by1, bz0 - 0.22)], 0.018, S.mat("chrome"), sides=14)
    tz = bz1
    S.cylinder("/Dressing/basin/tap_base", 0.020, 0.030, S.mat("chrome"), p=(bcx, by1 - 0.035, tz + 0.015))
    S.tube("/Dressing/basin/tap", [(bcx, by1 - 0.035, tz + 0.03), (bcx, by1 - 0.035, bs["tap_z"]), (bcx, by1 - 0.085, bs["tap_z"] + 0.02),
                                   (bcx, by1 - 0.150, bs["tap_z"] - 0.01), (bcx, by1 - 0.155, bs["tap_z"] - 0.04)], 0.010, S.mat("chrome"), sides=14)
    S.tube("/Dressing/basin/tap_lever", [(bcx, by1 - 0.035, bs["tap_z"] + 0.005), (bcx, by1 - 0.035, bs["tap_z"] + 0.04),
                                         (bcx, by1 - 0.110, bs["tap_z"] + 0.075)], 0.006, S.mat("chrome"), sides=10)
    S.box("/Dressing/basin/soap", (0.085, 0.075, 0.17), white, p=(bx0 - 0.02, y1 - 0.0375, 1.20), r=0.012)
    S.box("/Dressing/basin/soap_window", (0.03, 0.002, 0.07), S.color("soap_pink", (.80, .35, .45), .2, 0.), p=(bx0 - 0.02, y1 - 0.076, 1.21), r=0.001)
    S.box("/Dressing/basin/soap_lever", (0.07, 0.03, 0.02), S.mat("vent_dark"), p=(bx0 - 0.02, y1 - 0.085, 1.125), r=0.004)
    ds = P.DISPENSER
    dcx = 0.5*sum(ds["x"])
    S.box("/Dressing/basin/towels", (ds["x"][1] - ds["x"][0], ds["y"][1] - ds["y"][0], ds["z"][1] - ds["z"][0]), white,
          p=(dcx, 0.5*sum(ds["y"]), 0.5*sum(ds["z"])), r=0.012)
    S.box("/Dressing/basin/towel_slot", (0.20, 0.03, 0.012), S.mat("vent_dark"), p=(dcx, ds["y"][0] + 0.035, ds["z"][0] + 0.004), r=0.002)
    S.box("/Dressing/basin/towel_tip", (0.18, 0.003, 0.07), S.color("paper_towel", (.86, .85, .80), .95, 0.), p=(dcx, ds["y"][0] + 0.035, ds["z"][0] - 0.03), r=0.001)
    sign = textured(S, "sign_wash", sign_map(tex), 0.5)
    panel(S, "/Dressing/basin/sign", (dcx, y1 - 0.003, 1.84), (1, 0, 0), (0, 0, 1), 0.26, 0.18, sign)
    bn = P.BIN
    bxy = bn["xy"]
    S.revolve("/Dressing/bin/body", [(0., 0.), (bn["r"] - 0.012, 0.), (bn["r"], 0.012), (bn["r"], bn["h"] - 0.05), (0., bn["h"] - 0.05)],
              S.mat("steel_satin"), segments=64, p=(bxy[0], bxy[1], 0.))
    S.revolve("/Dressing/bin/lid", [(0., bn["h"] - 0.05), (bn["r"] + 0.004, bn["h"] - 0.05), (bn["r"] + 0.004, bn["h"] - 0.02),
                                    (bn["r"] - 0.03, bn["h"]), (0., bn["h"])], S.mat("vent_dark"), segments=64, p=(bxy[0], bxy[1], 0.))
    S.box("/Dressing/bin/pedal", (0.10, 0.06, 0.012), S.mat("vent_dark"), p=(bxy[0], bxy[1] - bn["r"] - 0.03, 0.02), r=0.003)
    # ---- the lesson board on the front wall, with its tray and pens
    (ox0, ox1), (oz0, oz1) = P.BOARD["x"], P.BOARD["z"]
    ocx, ocz = 0.5*(ox0 + ox1), 0.5*(oz0 + oz1)
    alu = S.mat("aluminum")
    board = textured(S, "lesson_board", board_map(tex, world), 0.25)
    panel(S, "/Dressing/board/face", (ocx, y0 + 0.014, ocz), (-1, 0, 0), (0, 0, 1), ox1 - ox0, oz1 - oz0, board)
    S.box("/Dressing/board/back", (ox1 - ox0 + 0.02, 0.012, oz1 - oz0 + 0.02), white, p=(ocx, y0 + 0.006, ocz), r=0.002)
    for name, (size, pos) in dict(top=((ox1 - ox0 + 0.04, 0.022, 0.022), (ocx, y0 + 0.014, oz1 + 0.011)),
                                  bottom=((ox1 - ox0 + 0.04, 0.022, 0.022), (ocx, y0 + 0.014, oz0 - 0.011)),
                                  left=((0.022, 0.022, oz1 - oz0 + 0.044), (ox0 - 0.011, y0 + 0.014, ocz)),
                                  right=((0.022, 0.022, oz1 - oz0 + 0.044), (ox1 + 0.011, y0 + 0.014, ocz))).items():
        S.box(f"/Dressing/board/frame_{name}", size, alu, p=pos, r=0.003)
    S.box("/Dressing/board/tray", (0.9, 0.07, 0.012), alu, p=(ocx, y0 + 0.05, oz0 - 0.03), r=0.003)
    for k, rgb in enumerate(((.03, .08, .40), (.50, .03, .03), (.02, .02, .02))):
        S.cylinder(f"/Dressing/board/pen{k}", 0.008, 0.13, S.color(f"pen{k}", rgb, .4, 0.), p=(ocx - 0.2 + 0.16*k, y0 + 0.05, oz0 - 0.016), axis="X")
    S.box("/Dressing/board/eraser", (0.11, 0.05, 0.03), S.color("eraser_felt", (.12, .13, .16), .95, 0.), p=(ocx + 0.32, y0 + 0.05, oz0 - 0.009), r=0.006)
    # ---- a rail of aprons, the clock, the extinguisher
    (ax0, ax1), az = P.APRON_HOOKS["x"], P.APRON_HOOKS["z"]
    S.box("/Dressing/aprons/rail", (ax1 - ax0 + 0.10, 0.018, 0.07), oak, p=(0.5*(ax0 + ax1), y0 + 0.009, az), r=0.004, uv_per_m=1., grain=0)
    for k, hx in enumerate(np.linspace(ax0, ax1, 4)):
        S.tube(f"/Dressing/aprons/hook{k}", [(hx, y0 + 0.018, az), (hx, y0 + 0.055, az - 0.005), (hx, y0 + 0.060, az + 0.025)], 0.005, satin, sides=8)
    for k, (hx, rgb) in enumerate(zip(np.linspace(ax0, ax1, 4)[:3], ((.08, .14, .30), (.80, .80, .77), (.30, .08, .08)))):
        cloth = S.color(f"apron{k}", rgb, .92, 0.)
        # a bib apron hung by its neck loop: the bib, then the skirt gathered into soft folds that deepen downwards
        v, f, nu = [], [], 25
        rows = [(-0.13, 0.085), (-0.24, 0.10), (-0.40, 0.145), (-0.60, 0.165), (-0.82, 0.175), (-0.98, 0.18)]   # (z below the hook, half width)
        for (dz, hw) in rows:
            depth = 0.022*min(1.0, max(0.0, (-dz - 0.2)/0.4))
            for u in np.linspace(-1., 1., nu):
                fold = depth*math.sin(2.6*math.pi*u + 1.7*k) + 0.5*depth*math.sin(5.1*math.pi*u + 0.6*k)
                v.append((hx + u*hw, y0 + 0.030 + depth + fold, az + dz))
        for r_ in range(len(rows) - 1):
            for i in range(nu - 1):
                a_, b_ = r_*nu + i, r_*nu + i + 1
                f += [a_, b_, b_ + nu, a_, b_ + nu, a_ + nu]
        S.mesh(f"/Dressing/aprons/apron{k}", v, f, mat=cloth, smooth=True, double=True)
        S.tube(f"/Dressing/aprons/loop{k}", [(hx - 0.080, y0 + 0.030, az - 0.13), (hx - 0.035, y0 + 0.040, az - 0.03), (hx, y0 + 0.056, az + 0.012),
                                             (hx + 0.035, y0 + 0.040, az - 0.03), (hx + 0.080, y0 + 0.030, az - 0.13)], 0.0045, cloth, sides=6, squash=0.4)
        for j, sgn in enumerate((-1, 1)):                # the waist ties hang down the sides
            S.tube(f"/Dressing/aprons/tie{k}{j}", [(hx + sgn*0.142, y0 + 0.036, az - 0.40), (hx + sgn*0.165, y0 + 0.040, az - 0.52),
                                                  (hx + sgn*0.158, y0 + 0.034, az - 0.74), (hx + sgn*0.170, y0 + 0.032, az - 0.90)],
                   0.006, cloth, sides=6, squash=0.3)
    # ---- the left wall: a cork notice board in its frame, a first-aid cabinet
    x0w = P.ROOM["x0"]
    (ny0, ny1), (nz0, nz1) = P.NOTICE["y"], P.NOTICE["z"]
    ncy, ncz = 0.5*(ny0 + ny1), 0.5*(nz0 + nz1)
    panel(S, "/Dressing/notice/face", (x0w + 0.016, ncy, ncz), (0, 1, 0), (0, 0, 1), ny1 - ny0, nz1 - nz0, textured(S, "notice_cork", notice_map(tex), 0.9))
    S.box("/Dressing/notice/back", (0.014, ny1 - ny0 + 0.02, nz1 - nz0 + 0.02), oak, p=(x0w + 0.007, ncy, ncz), r=0.002)
    for name, (size, pos) in dict(top=((0.024, ny1 - ny0 + 0.05, 0.025), (x0w + 0.012, ncy, nz1 + 0.0125)),
                                  bottom=((0.024, ny1 - ny0 + 0.05, 0.025), (x0w + 0.012, ncy, nz0 - 0.0125)),
                                  a=((0.024, 0.025, nz1 - nz0), (x0w + 0.012, ny0 - 0.0125, ncz)),
                                  b=((0.024, 0.025, nz1 - nz0), (x0w + 0.012, ny1 + 0.0125, ncz))).items():
        S.box(f"/Dressing/notice/frame_{name}", size, oak, p=pos, r=0.003, uv_per_m=1., grain=1 if name in ("top", "bottom") else 2)
    fa = P.FIRST_AID
    fh = fa["half"]
    S.box("/Dressing/first_aid/box", (2*fh[0], 2*fh[1], 2*fh[2]), white, p=(x0w + fh[0], fa["y"], fa["z"]), r=0.008)
    green = S.color("first_aid_green", (.02, .38, .12), .5, 0.)
    S.box("/Dressing/first_aid/cross_v", (0.002, 0.045, 0.14), green, p=(x0w + 2*fh[0] + 0.0006, fa["y"], fa["z"]), r=0.0005)
    S.box("/Dressing/first_aid/cross_h", (0.002, 0.14, 0.045), green, p=(x0w + 2*fh[0] + 0.0006, fa["y"], fa["z"]), r=0.0005)
    S.box("/Dressing/first_aid/latch", (0.006, 0.018, 0.035), satin, p=(x0w + 2*fh[0] + 0.002, fa["y"] - fh[1] + 0.02, fa["z"]), r=0.001)
    d.ref("/Dressing/room/clock", "wall_clock", (P.DOOR["x"][0] - 0.45, y0 + 0.002, 2.25), yaw=math.pi)
    d.ref("/Dressing/room/extinguisher", "korean_fire_extinguisher_01", (P.DOOR["x"][1] + 0.22, y0 + 0.16, 0.), yaw=math.pi)


RACK_Z = (0.115, 0.625, 1.135, 1.635, 2.145)        # m, the top faces of the pinned rack's five shelves (measured from its mesh)


def stock(S, d, cx, cy):
    """The dry store on the rack (its long side along y): a sack and a crate at the bottom, cartons, a row of bulk jars,
    bowls, trays and kitchen roll above, a carton and a stock pot on top. Visual; the rack's collision box holds it all."""
    tex = d.tex
    z0, z1, z2, z3, z4 = RACK_Z
    satin, white = S.mat("steel_satin"), S.mat("frame_white")
    lbl = lambda name, title, lines, band, bg=(245, 242, 232): textured(S, f"label_{name}", label_map(tex, name, title, lines, bg=bg, band=band), 0.6)
    # bottom: a 16 kg sack of flour lying flat, a crate
    paper = S.color("sack_paper", (.74, .68, .56), .9, 0.)
    S.box("/Dressing/shelving/sack", (0.36, 0.56, 0.13), paper, p=(cx, cy - 0.22, z0 + 0.065), r=0.045)
    S.box("/Dressing/shelving/sack2", (0.35, 0.54, 0.12), paper, p=(cx + 0.01, cy - 0.21, z0 + 0.185), r=0.045, q=Rotation.from_euler("z", 0.06).as_quat())
    panel(S, "/Dressing/shelving/sack_label", (cx + 0.01, cy - 0.21, z0 + 0.2455), (0, -1, 0), (-1, 0, 0), 0.34, 0.24,
          lbl("sack", "FLOUR", ("strong white", "16 kg"), (36, 78, 150)))
    d.ref("/Dressing/shelving/crate", "plastic_crate_03", (cx, cy + 0.33, z0), yaw=math.pi/2)
    # second shelf: two cartons
    d.ref("/Dressing/shelving/carton0", "cardboard_box_01", (cx, cy - 0.24, z1 + 0.007), yaw=0.03)
    d.ref("/Dressing/shelving/carton1", "cardboard_box_01", (cx - 0.01, cy + 0.30, z1 + 0.007), yaw=math.pi - 0.05)
    # third shelf: bulk jars with their goods, a stack of tins
    goods = [((.80, .74, .58), "OATS"), ((.93, .93, .91), "SUGAR"), ((.86, .82, .70), "RICE"), ((.78, .60, .30), "PASTA"), ((.42, .25, .14), "COCOA")]
    for k, (rgb, name) in enumerate(goods):
        y = cy - 0.44 + 0.165*k
        base = f"/Dressing/shelving/jar{k}"
        _jar_local(S, base, (cx - 0.02, y, z2), 0.068, 0.25, (0.85, 0.6, 0.75, 0.9, 0.5)[k], S.color(f"bulk_{name.lower()}", rgb, .7, 0.), satin)
        panel(S, base + "/label", (cx - 0.02 + 0.0695, y, z2 + 0.11), (0, 1, 0), (0, 0, 1), 0.07, 0.05,
              lbl(f"bulk_{name.lower()}", name, (), (60, 60, 66), bg=(250, 250, 246)))
    for k in range(4):
        S.cylinder(f"/Dressing/shelving/tin{k}", 0.05, 0.11, S.color("tin_plain", (.70, .70, .70), .3, .8),
                   p=(cx + 0.02, cy + 0.44, z2 + 0.055 + 0.112*(k % 2)) if k < 2 else (cx - 0.09, cy + 0.44, z2 + 0.055 + 0.112*(k % 2)))
    # fourth shelf: mixing bowls, baking trays, kitchen roll
    for k in range(5):
        prof = GEO.bowl_profile(0.165 - 0.010*k, 0.115 - 0.005*k, 0.05)
        S.revolve(f"/Dressing/shelving/bowl{k}", GEO.wall_polygon(prof, 0.0012), S.mat("bowl_steel"), segments=72, crease_deg=50.,
                  p=(cx, cy - 0.33, z3 + 0.001 + 0.014*k))
    for k in range(6):
        S.box(f"/Dressing/shelving/tray{k}", (0.40, 0.30, 0.012), S.mat("aluminum"), p=(cx, cy + 0.06, z3 + 0.006 + 0.013*k), r=0.004,
              q=Rotation.from_euler("z", math.pi/2 + 0.02*((k*7) % 3 - 1)).as_quat())
    roll = S.color("kitchen_roll", (.88, .88, .85), .95, 0.)
    for k, (dx, dy) in enumerate(((-0.08, 0.36), (0.07, 0.38), (-0.02, 0.47))):
        S.cylinder(f"/Dressing/shelving/roll{k}", 0.058, 0.23, roll, p=(cx + dx, cy + dy, z3 + 0.115))
        S.cylinder(f"/Dressing/shelving/roll{k}_core", 0.021, 0.232, S.color("roll_core", (.50, .40, .28), .9, 0.), p=(cx + dx, cy + dy, z3 + 0.115))
    # top: a carton and a stock pot
    d.ref("/Dressing/shelving/carton2", "cardboard_box_01", (cx, cy + 0.25, z4 + 0.007), yaw=0.08)
    S.cylinder("/Dressing/shelving/pot", 0.15, 0.20, S.mat("steel_brushed"), p=(cx, cy - 0.28, z4 + 0.10))
    S.cylinder("/Dressing/shelving/pot_lid", 0.155, 0.012, S.mat("steel_brushed"), p=(cx, cy - 0.28, z4 + 0.206))
    S.cylinder("/Dressing/shelving/pot_knob", 0.018, 0.025, S.mat("vent_dark"), p=(cx, cy - 0.28, z4 + 0.225))
    for k, sgn in enumerate((-1, 1)):
        S.tube(f"/Dressing/shelving/pot_handle{k}", [(cx - 0.03, cy - 0.28 + sgn*0.15, z4 + 0.15), (cx - 0.03, cy - 0.28 + sgn*0.185, z4 + 0.15),
                                                      (cx + 0.03, cy - 0.28 + sgn*0.185, z4 + 0.15), (cx + 0.03, cy - 0.28 + sgn*0.15, z4 + 0.15)],
               0.006, satin, sides=8)


# ====================================================================================== what stands on the counter
def _carton(S, d, base, f, half, h, body, label, lean=0.0):
    """A printed carton or paper pack standing on the counter: its box and a label on its front (-y in its frame)."""
    S.box(base + "/pack", (2*half[0], 2*half[1], h), body, p=f.p(0, 0, 0.5*h), q=f.q, r=0.003)
    if label is not None:
        panel(S, base + "/label", f.p(0, -half[1] - 0.0006, 0.5*h), f.R.apply([1, 0, 0]), (0, 0, 1), 1.9*half[0], 0.92*h, label)


def props(S, d, world):
    sc = world.scene
    tex = d.tex
    glass = S.mats["glass"]
    satin, white = S.mat("steel_satin"), S.mat("frame_white")
    lbl = lambda name, title, lines, band, bg=(245, 242, 232): textured(S, f"label_{name}", label_map(tex, name, title, lines, bg=bg, band=band), 0.55)
    for k, pr in enumerate(getattr(sc, "prop_list", ())):
        kind = pr["kind"]
        if pr.get("body") is not None:           # a free body: drawn on it, in its own frame (the middle of its foot)
            f, base = Frame(0., 0., 0., 0.), f"/Lab/body_{pr['body']}/dress"
        else:
            f, base = Frame(pr["x"], pr["y"], CZ, pr["yaw"]), f"/Dressing/props/{pr['name']}"
        hx, hy = P.PROP[kind]["half"]
        h = P.PROP[kind]["h"]
        if kind == "flour_bag":
            paper = S.color("flour_paper", (.86, .83, .76), .9, 0.)
            S.box(base + "/sack", (2*hx, 2*hy, h - 0.03), paper, p=f.p(0, 0, 0.5*(h - 0.03)), q=f.q, r=0.010)
            S.box(base + "/roll", (2*hx - 0.006, 0.030, 0.030), paper, p=f.p(0, 0, h - 0.018), q=f.q, r=0.012)
            panel(S, base + "/label", f.p(0, -hy - 0.0008, 0.47*(h - 0.03)), f.R.apply([1, 0, 0]), (0, 0, 1), 1.75*hx, 0.72*(h - 0.03),
                  lbl("flour", "FLOUR", ("plain wheat flour", "all-purpose", "1 kg"), (36, 78, 150)))
        elif kind == "sugar_jar":
            _jar_local(S, base, f.p(), hx, h - 0.02, 0.78, S.color("sugar_goods", (.93, .93, .91), .55, 0.), satin)
        elif kind == "baking_tin":
            S.cylinder(base + "/tin", hx, h - 0.012, S.color("tin_red", (.55, .05, .04), .35, 0.), p=f.p(0, 0, 0.5*(h - 0.012)))
            S.cylinder(base + "/lid", hx + 0.0015, 0.012, S.color("tin_lid", (.80, .78, .70), .3, 0.), p=f.p(0, 0, h - 0.006))
            panel(S, base + "/label", f.p(0, -hx - 0.0006, 0.45*h), f.R.apply([1, 0, 0]), (0, 0, 1), 0.045, 0.045,
                  lbl("baking", "BAKING", ("POWDER",), (120, 20, 16), bg=(250, 246, 235)))
        elif kind == "salt":
            S.cylinder(base + "/body", hx, h - 0.02, S.color("salt_white", (.90, .90, .88), .35, 0.), p=f.p(0, 0, 0.5*(h - 0.02)))
            S.cylinder(base + "/cap", hx - 0.002, 0.020, satin, p=f.p(0, 0, h - 0.010))
            S.box(base + "/band", (0.03, 0.002, 0.03), S.color("salt_blue", (.05, .15, .45), .5, 0.), p=f.p(0, -hx + 0.0005, 0.45*h), q=f.q, r=0.0005)
        elif kind == "milk_carton":
            cart = S.color("carton_white", (.90, .90, .88), .45, 0.)
            hb = h - 0.050
            S.box(base + "/body", (2*hx, 2*hy, hb), cart, p=f.p(0, 0, 0.5*hb), q=f.q, r=0.003)
            v = [f.p(-hx, -hy, hb), f.p(hx, -hy, hb), f.p(hx, hy, hb), f.p(-hx, hy, hb), f.p(-hx, 0, hb + 0.036), f.p(hx, 0, hb + 0.036)]
            S.mesh(base + "/gable", v, [0, 1, 5, 0, 5, 4, 2, 3, 4, 2, 4, 5, 1, 2, 5, 3, 0, 4], mat=cart, double=True)
            S.box(base + "/fin", (2*hx, 0.004, 0.016), cart, p=f.p(0, 0, hb + 0.043), q=f.q, r=0.001)
            S.cylinder(base + "/cap", 0.014, 0.012, S.color("cap_blue", (.05, .20, .55), .4, 0.), p=f.p(0, -0.5*hy, hb + 0.022),
                       q=f.quat(Rotation.from_euler("x", -math.atan2(0.036, hy))))
            panel(S, base + "/label", f.p(0, -hy - 0.0006, 0.5*hb), f.R.apply([1, 0, 0]), (0, 0, 1), 1.9*hx, 0.9*hb,
                  lbl("milk", "MILK", ("whole", "3.5 % fat", "1 litre"), (30, 100, 170), bg=(250, 250, 248)))
        elif kind == "egg_carton":
            pulp = S.color("egg_pulp", (.52, .50, .45), .95, 0.)
            S.box(base + "/tray", (2*hx, 2*hy, 0.034), pulp, p=f.p(0, 0, 0.017), q=f.q, r=0.008)
            S.box(base + "/lid", (2*hx, 0.012, 2*hy - 0.01), pulp, p=f.p(0, hy + 0.012, 0.5*(2*hy)), q=f.quat(Rotation.from_euler("x", -0.25)), r=0.005)
            shell = S.color("egg_shell", (.78, .58, .40), .45, 0.)
            for i in range(3):
                for j in range(2):
                    g = S.sphere(f"{base}/egg{i}{j}", 1.0, shell)
                    S.place(g, f.p((i - 1)*0.048, (j - 0.5)*0.048, 0.046), scale=(0.0205, 0.0205, 0.027))
        elif kind == "oil_bottle":
            prof = [(0., 0.), (hx - 0.003, 0.), (hx, 0.006), (hx, 0.165), (0.012, 0.215), (0.012, h - 0.022), (0., h - 0.022)]
            g = S.revolve(base + "/bottle", prof, glass, segments=56, crease_deg=50., p=f.p())
            S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
            S.revolve(base + "/oil", [(0., 0.002), (hx - 0.004, 0.002), (hx - 0.003, 0.008), (hx - 0.003, 0.150), (0., 0.152)],
                      S.color("oil_gold", (.72, .50, .06), .15, 0.), segments=48, p=f.p())
            S.cylinder(base + "/cap", 0.014, 0.024, S.color("cap_yellow", (.75, .55, .05), .4, 0.), p=f.p(0, 0, h - 0.012))
            panel(S, base + "/label", f.p(0, -hx - 0.0008, 0.085), f.R.apply([1, 0, 0]), (0, 0, 1), 0.050, 0.075,
                  lbl("oil", "OIL", ("sunflower", "500 ml"), (190, 140, 20), bg=(252, 248, 225)))
        elif kind == "jug":
            r_ = 0.047
            prof = GEO.cylinder_profile(r_ - 0.003, h - 0.004, corner=0.006)
            g = S.revolve(base + "/glass", GEO.wall_polygon(prof, 0.003), glass, segments=64, crease_deg=50., p=f.p(-0.012, 0, 0.003))
            S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
            S.tube(base + "/handle", [f.p(r_ - 0.014, 0, h - 0.020), f.p(r_ + 0.022, 0, h - 0.024), f.p(r_ + 0.026, 0, 0.5*h),
                                      f.p(r_ - 0.012, 0, 0.022)], 0.006, glass, sides=10)
            for i, z in enumerate((0.030, 0.055, 0.080)):
                S.box(f"{base}/mark{i}", (0.014, 0.0008, 0.0012), S.color("jug_red", (.60, .04, .04), .5, 0.), p=f.p(-0.012, -r_ - 0.0003, z), q=f.q, r=0.0002)
        elif kind == "scale":
            S.box(base + "/body", (2*hx, 2*hy, h - 0.008), white, p=f.p(0, 0, 0.5*(h - 0.008)), q=f.q, r=0.008)
            S.box(base + "/platform", (2*hx - 0.05, 2*hy - 0.012, 0.006), S.mat("steel_brushed"), p=f.p(0.022, 0, h - 0.005), q=f.q, r=0.003)
            S.box(base + "/lcd", (0.034, 0.05, 0.0012), S.color("lcd_green", (.20, .26, .18), .25, 0.), p=f.p(-hx + 0.022, 0, h - 0.0074), q=f.q, r=0.0005)
        elif kind == "cake_box":
            _carton(S, d, base, f, (hx, hy), h, S.color("cakebox", (.86, .68, .25), .55, 0.),
                    lbl("cake", "CAKE MIX", ("yellow sponge", "add water, oil, 3 eggs", "432 g"), (150, 60, 20), bg=(250, 236, 190)))
        elif kind == "pudding_box":
            _carton(S, d, base, f, (hx, hy), h, S.color("pudbox", (.92, .86, .62), .55, 0.),
                    lbl("pudding", "PUDDING", ("instant, vanilla", "96 g"), (120, 80, 20), bg=(252, 244, 205)))
        elif kind == "starch_box":
            _carton(S, d, base, f, (hx, hy), h, S.color("starchbox", (.92, .85, .30), .55, 0.),
                    lbl("starch", "CORN STARCH", ("for sauces", "400 g"), (30, 80, 140), bg=(250, 240, 170)))
        elif kind == "mayo_jar":
            _jar_local(S, base, f.p(), hx, h - 0.02, 0.62, S.color("mayo_goods", (.90, .88, .76), .35, 0.),
                       S.color("lid_blue", (.05, .18, .50), .4, 0.))
        elif kind == "relish_jar":
            _jar_local(S, base, f.p(), hx, h - 0.02, 0.80, S.color("relish_goods", (.22, .30, .06), .4, 0.), satin)
        elif kind == "ketchup":
            red = S.color("ketchup_red", (.50, .03, .02), .30, 0.)
            S.revolve(base + "/bottle", [(0., 0.), (hx - 0.004, 0.), (hx, 0.008), (hx, 0.115), (hx - 0.008, 0.160), (0.013, 0.180), (0., 0.180)],
                      red, segments=48, p=f.p())
            S.revolve(base + "/cap", [(0., 0.178), (0.017, 0.178), (0.017, h - 0.004), (0.010, h), (0., h)], white, segments=32, p=f.p())
            panel(S, base + "/label", f.p(0, -hx - 0.0008, 0.07), f.R.apply([1, 0, 0]), (0, 0, 1), 0.046, 0.07,
                  lbl("ketchup", "KETCHUP", ("tomato", "340 g"), (150, 20, 16), bg=(252, 246, 236)))
        elif kind == "tub":
            S.box(base + "/tub", (2*hx, 2*hy, h), glass, p=f.p(0, 0, 0.5*h), q=f.q, r=0.012)
            S.box(base + "/porridge", (2*hx - 0.012, 2*hy - 0.012, 0.022), S.color("porridge_left", (.70, .62, .46), .6, 0.), p=f.p(0, 0, 0.017), q=f.q, r=0.008)
            S.box(base + "/lid", (2*hx + 0.004, 2*hy + 0.004, 0.008), S.color("lid_teal", (.10, .42, .45), .45, 0.),
                  p=f.p(0, 0, h - 0.004), q=f.q, r=0.004)
        elif kind == "towel":
            T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
            cloth = S.mats.get("tea_towel") or S.color("towel_plain", (.80, .80, .76), .92, 0.)
            for i in range(3):
                S.box(f"{base}/fold{i}", (2*hx - 0.01*i, 2*hy - 0.008*i, 0.0075), cloth, p=f.p(0.003*i, -0.002*i, 0.004 + 0.0072*i),
                      q=f.quat(Rotation.from_euler("z", 0.02*i)), r=0.0035, uv_per_m=4.)


def _jar_local(S, base, p, r, h, fill, contents, lid):
    """A straight glass jar with its contents; lid: a material for a screw lid, or None for an open jar."""
    prof = GEO.cylinder_profile(r - 0.003, h, corner=0.008)
    g = S.revolve(base + "/glass", GEO.wall_polygon(prof, 0.003), S.mats["glass"], segments=64, crease_deg=50., p=(p[0], p[1], p[2] + 0.003))
    S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
    S.revolve(base + "/goods", [(0., 0.0005), (r - 0.0045, 0.0005), (r - 0.0035, 0.009), (r - 0.0035, fill*h), (0., fill*h + 0.004)],
              contents, segments=48, p=(p[0], p[1], p[2] + 0.003))
    if lid is not None:
        S.revolve(base + "/lid", [(0., h + 0.002), (r + 0.001, h + 0.002), (r + 0.001, h + 0.016), (r - 0.003, h + 0.019), (0., h + 0.019)],
                  lid, segments=48, p=p)


# ====================================================================================== hand tools and the pan
def _oct_ring(a, b, inset, cut=0.001, per=7):
    """One ring of the spoon bowl's outline: the octagon of tools.dish_boxes (its flats touch the oval 2a x 2b) moved
    `inset` inward. Every corner of the outline itself is rounded as far as leaves `cut` of the collider's corner
    outside the drawn wood (23.5 mm radius where a side meets a diagonal, 7.2 mm where a diagonal meets an end); an
    inner ring's radii are smaller by its inset. 8*per points (x, z), counter-clockwise."""
    k = math.tan(math.pi/8)
    V = np.array([(a, -b*k), (a, b*k), (a*k, b), (-a*k, b), (-a, b*k), (-a, -b*k), (-a*k, -b), (a*k, -b)])
    E = np.roll(V, -1, axis=0) - V
    Tn = E/np.linalg.norm(E, axis=1)[:, None]
    N = np.column_stack([Tn[:, 1], -Tn[:, 0]])
    D = np.sum(N*V, axis=1) - inset
    pts = []
    for i in range(8):
        j = (i - 1) % 8                                   # the corner between flat j and flat i
        c = np.linalg.solve(np.array([N[j], N[i]]), np.array([D[j], D[i]]))
        turn = math.acos(float(np.clip(Tn[j] @ Tn[i], -1., 1.)))
        rc = max(cut/(1./math.cos(0.5*turn) - 1.) - inset, 1e-5)      # past its radius a corner's true offset is sharp
        o = c - rc*math.tan(0.5*turn)*Tn[j] - rc*N[j]
        for s in np.linspace(0., 1., per):
            ca, sa = math.cos(s*turn), math.sin(s*turn)
            pts.append(o + rc*np.array([N[j, 0]*ca - N[j, 1]*sa, N[j, 0]*sa + N[j, 1]*ca]))
    return np.asarray(pts)


def _spoon(S, t, wood, handle=True):
    """The wooden spoon on its body (tool frame: z from the butt to the tip, the hollow toward +y): the bowl as one
    skin over the collision dish of tools.dish_boxes and the round handle with its domed butt.

    Where the drawn spoon and its collider differ (the collider is the boxes; each difference about a millimetre):
    the outline's eight corners are rounded (_oct_ring: the collider's corners stand 1.0 mm outside the drawn wood);
    the back's edge is rounded to 3 mm and the lip to its half width (1.2 and 1.0 mm of collider outside the wood
    along those edges, about 1.6 mm where they meet an outline corner); the foot of the hollow's wall has a 2.5 mm
    fillet (1.0 mm of drawn wood inside the collider's hollow); the handle runs straight into the bowl where the
    collider's capsule ends in a dome (up to 1.0 mm of drawn wood outside it in that corner). The outline is the
    collider's octagon and reads as one: a true oval would leave 2.1 to 3.2 mm of collider outside the wood at the
    eight corners. The neck does not taper: the collider's bowl ends toward the handle in a flat 21 mm wide, and so
    does the drawn one."""
    import trimesh
    s = P.SPOON
    (bw, bl), tk, dep, rim, hr = s["bowl"], s["bowl_t"], s["bowl_depth"], s["bowl_rim"], s["handle_r"]
    a, b, hl = 0.5*bw, 0.5*bl, s["length"] - s["bowl"][1]
    zc, y0, y1, yf = hl + b, -0.5*tk, 0.5*tk, 0.5*tk - dep
    rb, rl, rf = 0.003, 0.5*rim, 0.0025
    arc = lambda n: [math.pi*(i + 1)/(2*(n + 1)) for i in range(n)]
    skin = [(rb, y0)] + [(rb - rb*math.sin(u), y0 + rb - rb*math.cos(u)) for u in arc(4)] + [(0., y0 + rb), (0., y1 - rl)]
    skin += [(rl - rl*math.cos(u), y1 - rl + rl*math.sin(u)) for u in np.linspace(0, math.pi, 9)[1:-1]]
    skin += [(rim, y1 - rl), (rim, yf + rf)] + [(rim + rf - rf*math.cos(u), yf + rf - rf*math.sin(u)) for u in arc(5)] + [(rim + rf, yf)]
    rings = [_oct_ring(a, b, d) for d, _ in skin]
    m = len(rings[0])
    v = [(0., y0, zc)] + [(p[0], y, zc + p[1]) for r, (_, y) in zip(rings, skin) for p in r] + [(0., yf, zc)]
    f = [(0, 1 + (i + 1) % m, 1 + i) for i in range(m)]
    for k in range(len(skin) - 1):
        for i in range(m):
            p, q = 1 + k*m + i, 1 + k*m + (i + 1) % m
            f += [(p, q, q + m), (p, q + m, p + m)]
    last = 1 + (len(skin) - 1)*m
    f += [(len(v) - 1, last + i, last + (i + 1) % m) for i in range(m)]
    mesh = trimesh.Trimesh(np.asarray(v, float), np.asarray(f, int), process=False)
    trimesh.repair.fix_normals(mesh)
    base = f"/Lab/body_{t.body}/dress"
    # the grain runs along the tool; the shear by the height keeps it from smearing down the bowl's upright sides
    uv = mesh.vertices[:, [0, 2]] + mesh.vertices[:, 1:2]*np.array([0.45, 0.25])
    S.mesh(base + "/bowl", mesh.vertices, mesh.faces, uv=uv, normals=mesh.vertex_normals, mat=wood)
    if handle:
        zs = [hr*(1 - math.cos(u)) for u in arc(5)] + [hr, hl + 0.001]
        rs = [hr*math.sin(u) for u in arc(5)] + [hr, hr]
        n = 20
        ang = np.linspace(0, 2*math.pi, n, endpoint=False)
        v = [(0., 0., 0.)] + [(r*math.cos(u), r*math.sin(u), z) for r, z in zip(rs, zs) for u in ang]
        uv = [(0., 0.)] + [(hr*u, z) for r, z in zip(rs, zs) for u in ang]
        f = [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)]
        for k in range(len(zs) - 1):
            for i in range(n):
                p, q = 1 + k*n + i, 1 + k*n + (i + 1) % n
                f += [(p, q, q + n), (p, q + n, p + n)]
        mesh = trimesh.Trimesh(np.asarray(v, float), np.asarray(f, int), process=False)
        S.mesh(base + "/handle", mesh.vertices, mesh.faces, uv=uv, normals=mesh.vertex_normals, mat=wood)


def _stud(S, path, r, h, mat, p, q=(0., 0., 0., 1.)):
    """A low dome of radius r standing h along its own z from p: a rivet's head, a push button."""
    S.revolve(path, [(0., 0.)] + [(r*math.cos(u), h*math.sin(u)) for u in np.linspace(0., 0.5*math.pi, 7)], mat, segments=28, p=p, q=q)


def _pan(S, v):
    """What makes the pan a saucepan, on its body (vessel frame: origin at the middle of the inside bottom): the rolled
    rim, the handle's root on the wall with its rivet plate and the two rivet heads inside. (No hanging hole: the
    collider's handle is solid to its end, and an eye drawn on it read as a black button.) Against the collider: the rim's bead stands 1.0 mm outside the wall and above the rim; the rivet plate 1.0 mm and
    the rivet heads 0.8 mm off the wall; the handle's root is the handle's own round bar carried to the wall where the
    collider's capsule ends in a dome (up to 2 mm of drawn steel outside it in the corner between dome and wall). The
    handle is the collider's 22 mm round bar; the maker's is a concave bar (not drawn: it would hide the collider).
    The pan is clad to the rim (tri-ply): it has no base disc, and none is drawn."""
    s, h = P.PAN, v.handle
    base = f"/Lab/body_{v.body}/dress"
    steel, satin = S.mat("bowl_steel"), S.mat("steel_satin")
    ro = s["radius"] + s["wall"]
    cr, cz, rb = ro - 0.0009, v.depth - 0.0009, 0.0019
    S.revolve(base + "/rim", [(cr + rb*math.cos(u), cz + rb*math.sin(u)) for u in np.linspace(0, 2*math.pi, 14, endpoint=False)],
              steel, segments=112, crease_deg=80.)
    p0, p1 = np.asarray(h["p0"], float), np.asarray(h["p1"], float)
    u = (p1 - p0)/np.linalg.norm(p1 - p0)
    S.tube(base + "/handle_root", [p0 - h["r"]*u, p0], h["r"], S.mat("steel_brushed"), sides=24, cap=False)
    ang = math.atan2(p0[1], p0[0])
    rad, tan = np.array([math.cos(ang), math.sin(ang), 0.]), np.array([-math.sin(ang), math.cos(ang), 0.])
    # rivet plate: a patch of the wall's own curve, 1 mm proud, 36 mm wide and 26 mm high round the handle's root
    da, zs = np.linspace(-0.018/ro, 0.018/ro, 9), (p0[2] - 0.014, p0[2] + 0.012)
    pv = [((ro + 0.001)*math.cos(ang + d_), (ro + 0.001)*math.sin(ang + d_), z) for z in zs for d_ in da]
    pf = [t_ for i in range(len(da) - 1) for t_ in ((i, i + 1, i + 1 + len(da)), (i, i + 1 + len(da), i + len(da)))]
    S.mesh(base + "/rivet_plate", pv, pf, mat=satin, smooth=True, double=True)
    for k, side in enumerate((-1., 1.)):
        _stud(S, f"{base}/rivet{k}", 0.0042, 0.0008, satin, s["radius"]*rad + side*0.009*tan + np.array([0., 0., p0[2]]),
              Rotation.from_matrix(np.column_stack([tan, np.cross(tan, rad), -rad])).as_quat())


def _mixer_trim(S, t):
    """The hand mixer's housing as an appliance, on its body (frame: x toward the beater end, z toward the handle):
    a grey gear case under the white motor housing, the nose plate with its air slots at the beater end, the beater
    eject button on the handle. All of it lies on the collision shapes (housing box, handle capsule), 0.3 to 1 mm
    proud. No cord is drawn: a cord would have no collider."""
    hx, hy, hz = (0.5*v for v in t.spec["body"])
    base = f"/Lab/body_{t.body}/dress"
    grey, dark = S.mat("mixer_grey"), S.mat("vent_dark")
    S.box(base + "/gear_case", (2*hx + 0.0006, 2*hy + 0.0006, 0.034), grey, p=(0., 0., -hz + 0.017 - 0.0003), r=0.010)
    S.box(base + "/nose", (0.002, 2*hy - 0.022, 0.050), grey, p=(hx, 0., 0.018), r=0.0009)
    for k in range(5):
        S.box(f"{base}/nose_slot{k}", (0.0006, 2*hy - 0.036, 0.0032), dark, p=(hx + 0.0009, 0., 0.002 + 0.008*k), r=0.0002)
    zh = hz + 0.045 + t.spec["handle_r"]
    _stud(S, base + "/eject", 0.0085, 0.0010, grey, (0.018, 0., zh - 0.0001))


def tools(S, d, world):
    """Authored shapes of the hand tools and the pan on their moving bodies (as props: /Lab/body_{body}/dress).
    render.authored names the collision shapes these replace; the rest of each tool is drawn by render.export."""
    sc = world.scene
    for name, t in sc.tools.items():
        if t.kind == "spoon" and "bowl_depth" in P.SPOON:
            # beech: the pinned oak veneer's grain along the tool, tinted toward beech's pale pink-brown
            _spoon(S, t, d.pbr("spoon_beech", "white_oak_veneer", 0.5, .62, tint=(1.10, 0.96, 0.82), bump=.25), handle=not sc.sleeves)
        elif t.kind == "mixer":
            _mixer_trim(S, t)
    if getattr(sc.vessel, "handle", None):
        _pan(S, sc.vessel)


# ====================================================================================== the yard outside the window
def outlook(S, d):
    """What the window looks onto: the yard behind a ground-floor teaching kitchen. Paving from the wall out, a
    close-boarded fence 5 m away with a wheeled bin against it, and the brick wall of the next building 8.5 m away with
    two windows. Lit by the dome (daylight) like the room; visual only, outside the room's walls, nothing reaches it."""
    x1 = P.ROOM["x1"] + P.WINDOW["reveal"]
    yc = 0.5*sum(P.WINDOW["y"])
    # Exposure, not material: the dome's sun lights the yard about ten times brighter than pictures exposed for the
    # room's 500 lx show (the first stills had the fence burnt white and green bins mint). Everything out here is drawn
    # at SHADE of its albedo, so the yard reads as a camera that held both would show it. Authored; nothing reads it.
    SHADE = 0.14
    paving = d.pbr("yard_paving", "large_grey_tiles", 2.0, .75, tint=(SHADE,)*3)
    planks = d.pbr("yard_fence", "dark_wooden_planks", 2.0, .80, tint=(1.6*SHADE,)*3)
    brick = d.pbr("yard_brick", "brick_wall_10", 2.5, .85, tint=(SHADE,)*3)
    white, dark = S.color("yard_frame", (.80*SHADE,)*3, .5, 0.), S.color("yard_glass", (.004, .005, .006), .25, 0.)
    S.quad("/Dressing/outside/yard", [(x1 + 0.01, yc - 9., -0.05), (x1 + 9., yc - 9., -0.05), (x1 + 9., yc + 9., -0.05),
                                      (x1 + 0.01, yc + 9., -0.05)], paving)
    xf = x1 + 5.0
    S.box("/Dressing/outside/fence", (0.022, 18., 1.80), planks, p=(xf, yc, 0.85), r=0.002, uv_per_m=1., grain=2)
    S.box("/Dressing/outside/fence_cap", (0.060, 18., 0.030), planks, p=(xf, yc, 1.765), r=0.004, uv_per_m=1., grain=1)
    for k, yy in enumerate(np.arange(yc - 8.4, yc + 8.5, 2.4)):
        S.box(f"/Dressing/outside/fence_post{k}", (0.095, 0.095, 1.90), planks, p=(xf - 0.055, float(yy), 0.90), r=0.006, uv_per_m=1., grain=2)
    green = S.color("bin_green", (.03*SHADE, .13*SHADE, .06*SHADE), .45, 0.)
    for k, yy in enumerate((yc + 1.4, yc + 2.1)):
        S.box(f"/Dressing/outside/bin{k}", (0.72, 0.56, 0.96), green, p=(xf - 0.50, yy, 0.43), r=0.03)
        S.box(f"/Dressing/outside/bin{k}_lid", (0.76, 0.60, 0.07), green, p=(xf - 0.50, yy, 0.945), r=0.02)
    xw = x1 + 8.5
    S.box("/Dressing/outside/next_wall", (0.30, 18., 6.5), brick, p=(xw + 0.15, yc, 3.2), r=0.002, uv_per_m=1.)
    for k, (yy, zz) in enumerate(((yc - 0.6, 1.75), (yc + 3.0, 1.75), (yc - 0.6, 4.55), (yc + 3.0, 4.55))):
        S.box(f"/Dressing/outside/next_window{k}", (0.06, 1.20, 1.40), white, p=(xw - 0.01, yy, zz), r=0.004)
        S.box(f"/Dressing/outside/next_glass{k}", (0.012, 1.06, 1.26), dark, p=(xw - 0.042, yy, zz), r=0.001)
        S.box(f"/Dressing/outside/next_sill{k}", (0.12, 1.32, 0.05), white, p=(xw - 0.05, yy, zz - 0.725), r=0.004)


# ====================================================================================== daylight
def daylight(S, intensity=700.):
    """The sky and the land outside the window: a dome light with a pinned Poly Haven HDRI (whiteboard's and titrate's).
    The room is closed (double-sided walls and ceiling), so it lights the room only through the window."""
    from whiteboard import visual_assets as WVA
    hdr = WVA.asset("kiara_5_noon", "hdri")
    dome = S.UsdLux.DomeLight.Define(S.stage, "/Lights/daylight")
    dome.CreateTextureFileAttr(S.Sdf.AssetPath(str(Path(hdr).resolve())))
    dome.CreateIntensityAttr(float(intensity))
    dome.CreateColorAttr(S.Gf.Vec3f(1., 1., 1.))
    S.UsdGeom.Xformable(dome).AddRotateZOp().Set(40.0)
    return "kiara_5_noon"

"""Visual dressing of the skills-laboratory bay (render only; no physics reads this file).

What is drawn here either stands on a collision box of scene.room (the counter run, the trunking, the wall fittings,
the hand-wash basin, the drip stand, the bed) or is out of every arm's reach (the rest of the laboratory beyond the
bay: its far wall with two windows, a desk, the ceiling's light panels, the curtain on its track).

Light: six recessed 600 x 600 mm LED panels at 4,000 K set so that the tray gets TARGET_LUX (the panel luminance is
computed from the panels' geometry, as in the teaching kitchen), and daylight through the far wall's windows from a
pinned sky.
"""

import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P

CZ = P.COUNTER_Z
TARGET_LUX = 500.
LAB_Y0 = -5.60                         # the laboratory's far wall (the bay itself ends at ROOM['y0'])
PANELS = [(x, y) for x in (-1.0, 0.8) for y in (-0.55, -2.45, -4.35)]
PANEL_SIZE = (0.595, 0.595)
LIGHT_RGB = (1.057, 0.992, 0.913)      # 4,000 K as a camera's white balance leaves it (ingredient_mixing/dressing.py)
WINDOWS = ((-1.25, -0.15), (0.35, 1.45))          # x spans of the far wall's two windows
WINDOW_Z = (0.95, 2.15)


def panel_gain(ref):
    """Illuminance (lx) at `ref` per nit of the ceiling panels: E = sum L A cos_l cos_s / r^2."""
    ref = np.asarray(ref, float)
    A = PANEL_SIZE[0]*PANEL_SIZE[1]
    g = 0.
    for x, y in PANELS:
        d = np.array([x, y, P.ROOM["z1"] - 0.01]) - ref
        r = float(np.linalg.norm(d))
        g += A*(d[2]/r)**2/r**2
    return g


def _bx(S, path, x0, x1, y0, y1, z0, z1, mat, r=0.002):
    size = (x1 - x0, y1 - y0, z1 - z0)
    return S.box(path, size, S.mat(mat) if isinstance(mat, str) else mat, p=(0.5*(x0 + x1), 0.5*(y0 + y1), 0.5*(z0 + z1)),
                 r=max(min(r, 0.45*min(size)), 0.0004))


def _annulus(S, path, centre, r0, r1, z, mat, n=64):
    ang = np.linspace(0, 2*math.pi, n + 1)
    v = np.array([(centre[0] + rr*math.cos(a), centre[1] + rr*math.sin(a), z) for a in ang for rr in (r0, r1)])
    t = [(2*i, 2*i + 1, 2*i + 2) for i in range(n)] + [(2*i + 1, 2*i + 3, 2*i + 2) for i in range(n)]
    return S.mesh(path, v, t, normals=[(0., 0., 1.)]*len(v), mat=S.mat(mat), double=True)


def _image_quad(S, path, corners, image, name):
    from .render import _decal
    c = np.asarray(corners, float)
    nrm = np.cross(c[1] - c[0], c[3] - c[0])
    nrm /= np.linalg.norm(nrm)
    return S.mesh(path, c, [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], normals=[nrm]*4, mat=_decal(S, name, image), double=True)


def _poster(folder, spec):
    """The laminated checklist on the wall: the task's own instruction, set as a skills-lab checklist."""
    from PIL import Image, ImageDraw
    from .render import _font
    W, Hh = 1260, 1782
    im = Image.new("RGB", (W, Hh), (247, 246, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 150], fill=(22, 78, 120))
    d.text((50, 40), "Clinical skills: syringe practice", font=_font(62, True), fill=(255, 255, 255))
    y = 210
    d.text((50, y), spec["task"].replace("-", " ").title(), font=_font(64, True), fill=(20, 30, 50))
    y += 120
    words, line, f = spec["instruction"].split(), "", _font(40)
    for w_ in words:
        trial = (line + " " + w_).strip()
        if d.textlength(trial, font=f) > W - 110:
            d.text((55, y), line, font=f, fill=(25, 30, 40))
            y += 58
            line = w_
        else:
            line = trial
    d.text((55, y), line, font=f, fill=(25, 30, 40))
    y += 110
    for k, rule in enumerate(("Read the scale at the stopper's front rib.", "Air out before the dose is set.",
                              "Never put a sheath back on by hand.", "A bare needle touches nothing but its target.",
                              "Sharps go into the red container.")):
        d.ellipse([60, y + 8, 92, y + 40], outline=(22, 78, 120), width=5)
        d.text((115, y), rule, font=_font(40), fill=(25, 30, 40))
        y += 70
    d.text((50, Hh - 80), "Simulated medication only. For training use.", font=_font(34), fill=(120, 30, 30))
    path = Path(folder)/"poster.png"
    im.save(path)
    return path


def _sharps_label(folder):
    from PIL import Image, ImageDraw
    from .render import _font
    im = Image.new("RGB", (900, 540), (236, 196, 20))
    d = ImageDraw.Draw(im)
    d.text((40, 30), "SHARPS", font=_font(150, True), fill=(15, 15, 15))
    d.text((44, 220), "Do not fill above the line", font=_font(54), fill=(15, 15, 15))
    d.rectangle([40, 330, 860, 344], fill=(15, 15, 15))
    d.text((44, 380), "Do not recap needles", font=_font(54), fill=(15, 15, 15))
    path = Path(folder)/"sharps_label.png"
    im.save(path)
    return path


def _periodic_noise(px, sx, sy, rng):
    """Tileable noise: white noise filtered in the Fourier domain (sx, sy: feature lengths in pixels along x and y)."""
    n = rng.normal(0., 1., (px, px))
    fy = np.fft.fftfreq(px)[:, None]
    fx = np.fft.fftfreq(px)[None, :]
    filt = np.exp(-0.5*((fx*sx*2*np.pi)**2 + (fy*sy*2*np.pi)**2))
    out = np.real(np.fft.ifft2(np.fft.fft2(n)*filt))
    return (out - out.mean())/max(out.std(), 1e-9)


def _beech_map(folder, px=1024, seed=5):
    """Beech-effect laminate for the worktop (grain along the image's x): tileable, authored; colours as sRGB."""
    from PIL import Image
    p = Path(folder)/"worktop_beech_diff.png"
    if p.exists():
        return {"diff": p}
    rng = np.random.default_rng(seed)
    broad = _periodic_noise(px, 220., 9., rng)            # the figure of the boards
    fine = _periodic_noise(px, 60., 1.2, rng)             # pores along the grain
    fleck = _periodic_noise(px, 6., 1.0, rng)
    t = np.clip(0.5 + 0.20*broad + 0.10*fine + 0.03*fleck, 0., 1.)[..., None]
    light, dark = np.array([212., 186., 146.]), np.array([182., 150., 110.])
    Image.fromarray(np.clip(dark + (light - dark)*t, 0, 255).astype(np.uint8)).save(p)
    return {"diff": p}


def _vinyl_map(folder, px=1024, seed=9):
    """Sheet vinyl with a fleck, the floor of a clinical room: tileable, authored; sRGB."""
    from PIL import Image
    p = Path(folder)/"floor_vinyl_diff.png"
    if p.exists():
        return {"diff": p}
    rng = np.random.default_rng(seed)
    cloud = _periodic_noise(px, 90., 90., rng)
    base = np.array([112., 128., 132.])[None, None, :] + 5.0*cloud[..., None]
    img = np.repeat(base, 1, axis=2)
    for frac, col in ((0.010, (150., 164., 166.)), (0.008, (78., 92., 98.)), (0.003, (176., 170., 150.))):
        k = int(frac*px*px)
        ys, xs = rng.integers(0, px, k), rng.integers(0, px, k)
        for dy in range(2):
            for dx in range(2):
                img[(ys + dy) % px, (xs + dx) % px] = col
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(p)
    return {"diff": p}


def dress(S, world):
    sc = world.scene
    credits = set()
    R = P.ROOM
    x0, x1, y0, y1, h = R["x0"], R["x1"], LAB_Y0, R["y1"], R["z1"]
    boxes = {name: (np.asarray(pos, float), np.asarray(half, float)) for name, pos, half, _, _ in sc.static_boxes}
    tex = S.folder/"textures"
    T, Gf = S.Sdf.ValueTypeNames, S.Gf

    # ---- shell: vinyl floor with a coved skirting, walls (a wipeable panel to 1.4 m behind the counter), ceiling
    tex.mkdir(parents=True, exist_ok=True)
    floor = S.textured("floor_vinyl_flecked", _vinyl_map(tex), 0.60, .38)
    worktop = S.textured("worktop_beech", _beech_map(tex), 1.20, .30)
    S.quad("/Dressing/room/floor", [(x0, y0, 0.), (x1, y0, 0.), (x1, y1, 0.), (x0, y1, 0.)], floor, double=True)
    S.quad("/Dressing/room/ceiling", [(x0, y1, h), (x1, y1, h), (x1, y0, h), (x0, y0, h)], S.mat("ceiling"), double=True)
    _bx(S, "/Dressing/room/back_lower", x0, x1, y1, y1 + 0.03, 0.0, 1.40, "wall_panel", 0.0)
    _bx(S, "/Dressing/room/back_upper", x0, x1, y1, y1 + 0.03, 1.40, h, "wall_paint", 0.0)
    _bx(S, "/Dressing/room/back_rail", x0, x1, y1 - 0.006, y1, 1.395, 1.425, "trunking", 0.002)
    _bx(S, "/Dressing/room/left", x0 - 0.03, x0, y0, y1, 0.0, h, "wall_paint", 0.0)
    _bx(S, "/Dressing/room/right", x1, x1 + 0.03, y0, y1, 0.0, h, "wall_paint", 0.0)
    for name, (xa, xb, ya, yb) in dict(back=(x0, x1, y1 - 0.012, y1), left=(x0, x0 + 0.012, y0, y1), right=(x1 - 0.012, x1, y0, y1)).items():
        _bx(S, f"/Dressing/room/skirting_{name}", xa, xb, ya, yb, 0.0, 0.10, "skirting", 0.004)
    # the far wall with two windows (openings: the sky is the dome light's picture)
    za, zb = WINDOW_Z
    cuts = sorted(WINDOWS)
    xs = [x0] + [v for w in cuts for v in w] + [x1]
    for k in range(0, len(xs), 2):
        _bx(S, f"/Dressing/room/far_pier{k}", xs[k], xs[k + 1], y0 - 0.03, y0, 0.0, h, "lab_far", 0.0)
    for k, (wa, wb) in enumerate(cuts):
        _bx(S, f"/Dressing/room/far_sill{k}", wa, wb, y0 - 0.03, y0, 0.0, za, "lab_far", 0.0)
        _bx(S, f"/Dressing/room/far_head{k}", wa, wb, y0 - 0.03, y0, zb, h, "lab_far", 0.0)
        for j, (fa, fb, fz0, fz1) in enumerate(((wa, wb, za - 0.03, za), (wa, wb, zb, zb + 0.03), (wa - 0.03, wa, za, zb), (wb, wb + 0.03, za, zb),
                                                (0.5*(wa + wb) - 0.015, 0.5*(wa + wb) + 0.015, za, zb))):
            _bx(S, f"/Dressing/room/window{k}_frame{j}", fa, fb, y0 - 0.005, y0 + 0.04, fz0, fz1, "window_frame", 0.004)
        _bx(S, f"/Dressing/room/window{k}_board", wa - 0.04, wb + 0.04, y0, y0 + 0.14, za - 0.035, za - 0.005, "window_frame", 0.006)

    # ---- the counter run: solid-surface top with its upstand, casework fronts, the hand-wash basin
    cp, ch = boxes["counter_top"]
    S.box("/Dressing/counter/top", 2*ch, worktop, p=cp, r=0.003, uv_per_m=1.0, grain=0)
    up, uh = boxes["counter_upstand"]
    S.box("/Dressing/counter/upstand", 2*uh, worktop, p=up, r=0.003, uv_per_m=1.0, grain=0)
    # ---- the robots' cart: castors, a push bar, mounting plates with their bolts, controller fronts, cable runs
    if "cart_deck" in boxes:
        dp, dh = boxes["cart_deck"]
        zt_ = dp[2] + dh[2]
        for ix in (-1, 1):
            for iy in (-1, 1):
                cxp, cyp = dp[0] + ix*0.41, dp[1] + iy*0.21
                S.cylinder(f"/Dressing/cart/castor{ix > 0:d}{iy > 0:d}", 0.038, 0.026, S.mat("castor"), p=(cxp, cyp + 0.03*iy, 0.038), axis="X")
                _bx(S, f"/Dressing/cart/fork{ix > 0:d}{iy > 0:d}", cxp - 0.020, cxp + 0.020, cyp + 0.03*iy - 0.012, cyp + 0.03*iy + 0.012, 0.038, 0.085, "steel_brushed", 0.002)
            # the plate each arm stands on, bolted through the deck
            ax_ = dp[0] + ix*0.5*P.ARM_SPACING
            _bx(S, f"/Dressing/cart/mount{ix > 0:d}", ax_ - 0.13, ax_ + 0.13, -P.ARM_SETBACK - 0.13, -P.ARM_SETBACK + 0.13, zt_, zt_ + 0.008, "aluminum", 0.002)
            for bx_ in (-1, 1):
                for by_ in (-1, 1):
                    S.cylinder(f"/Dressing/cart/bolt{ix > 0:d}{bx_ > 0:d}{by_ > 0:d}", 0.007, 0.006, S.mat("bolt"), p=(ax_ + bx_*0.105, -P.ARM_SETBACK + by_*0.105, zt_ + 0.011))
            # the controller under it: a front panel with vents, a status light and a label, and the arm's cable
            if f"cart_controller_{ix}" in boxes:
                kp_, kh_ = boxes[f"cart_controller_{ix}"]
                yfr = kp_[1] - kh_[1]
                _bx(S, f"/Dressing/cart/ctrl_face{ix > 0:d}", kp_[0] - kh_[0] + 0.008, kp_[0] + kh_[0] - 0.008, yfr - 0.003, yfr, kp_[2] - kh_[2] + 0.008,
                    kp_[2] + kh_[2] - 0.008, "robot_dark", 0.002)
                for k in range(7):
                    _bx(S, f"/Dressing/cart/ctrl_vent{ix > 0:d}_{k}", kp_[0] - 0.11, kp_[0] + 0.03, yfr - 0.0045, yfr - 0.003, kp_[2] - 0.055 + 0.012*k,
                        kp_[2] - 0.050 + 0.012*k, "outlet_black", 0.0005)
                S.cylinder(f"/Dressing/cart/ctrl_led{ix > 0:d}", 0.004, 0.003, S.mat("led_green"), p=(kp_[0] + 0.10, yfr - 0.004, kp_[2] + 0.05), axis="Y")
                _bx(S, f"/Dressing/cart/ctrl_label{ix > 0:d}", kp_[0] + 0.05, kp_[0] + 0.13, yfr - 0.0042, yfr - 0.003, kp_[2] - 0.04, kp_[2] - 0.015, "paper", 0.0005)
                S.tube(f"/Dressing/cart/cable{ix > 0:d}", [(ax_ - 0.02*ix, -P.ARM_SETBACK - 0.11, zt_ - 0.01), (ax_ - 0.02*ix, -P.ARM_SETBACK - 0.16, zt_ - 0.10),
                                                     (kp_[0], kp_[1] - kh_[1] - 0.03, kp_[2] + kh_[2] + 0.10), (kp_[0], kp_[1] - kh_[1] + 0.02, kp_[2] + kh_[2] + 0.005)],
                       0.008, S.mat("cable"), sides=10)
        # the push bar at the cart's outer end, and a brake pedal bar between the rear castors
        S.tube("/Dressing/cart/push_bar", [(dp[0] - dh[0] + 0.01, dp[1] - dh[1] - 0.005, zt_ - 0.02), (dp[0] - dh[0] + 0.01, dp[1] - dh[1] - 0.07, zt_ - 0.02),
                                           (dp[0] + dh[0] - 0.01, dp[1] - dh[1] - 0.07, zt_ - 0.02), (dp[0] + dh[0] - 0.01, dp[1] - dh[1] - 0.005, zt_ - 0.02)],
               0.012, S.mat("steel_brushed"), sides=12)
        _bx(S, "/Dressing/cart/deck_edge", dp[0] - dh[0] - 0.003, dp[0] + dh[0] + 0.003, dp[1] - dh[1] - 0.003, dp[1] + dh[1] + 0.003, zt_ - 0.034, zt_ - 0.012,
            "cart_frame", 0.003)
    kp, kh = boxes["casework"]
    S.box("/Dressing/counter/carcass", 2*kh, S.mat("casework"), p=kp, r=0.001)
    tp, th = boxes["toe_kick"]
    S.box("/Dressing/counter/toe_kick", 2*th, S.mat("toe_kick"), p=tp, r=0.001)
    cx0, cx1 = cp[0] - ch[0], cp[0] + ch[0]
    yf = kp[1] - kh[1]
    zt = CZ - P.COUNTER_TOP_THICK
    n_unit = 4
    wu = (cx1 - cx0)/n_unit
    for k in range(n_unit):
        ua, ub = cx0 + k*wu + 0.004, cx0 + (k + 1)*wu - 0.004
        if k % 2 == 0:                                   # a stack of three drawers
            for j, (da, db) in enumerate(((0.11, 0.36), (0.365, 0.62), (0.625, zt - 0.006))):
                _bx(S, f"/Dressing/counter/unit{k}_drawer{j}", ua, ub, yf - 0.018, yf, da, db, "casework_front", 0.003)
                S.tube(f"/Dressing/counter/unit{k}_pull{j}", [(0.5*(ua + ub) - 0.07, yf - 0.045, db - 0.045), (0.5*(ua + ub) + 0.07, yf - 0.045, db - 0.045)],
                       0.005, S.mat("steel_brushed"), sides=10)
        else:                                            # a pair of doors
            mid = 0.5*(ua + ub)
            for j, (da, db) in enumerate(((ua, mid - 0.002), (mid + 0.002, ub))):
                _bx(S, f"/Dressing/counter/unit{k}_door{j}", da, db, yf - 0.018, yf, 0.11, zt - 0.006, "casework_front", 0.003)
                hx = db - 0.035 if j == 0 else da + 0.035
                S.tube(f"/Dressing/counter/unit{k}_handle{j}", [(hx, yf - 0.045, zt - 0.22), (hx, yf - 0.045, zt - 0.08)], 0.005, S.mat("steel_brushed"), sides=10)
    sk = P.SINK
    sa, sb = sk["x"]
    ya, yb = sk["y"]
    _bx(S, "/Dressing/sink/rim", sa, sb, ya, yb, CZ, CZ + sk["rim"], "ceramic", 0.008)
    _bx(S, "/Dressing/sink/bowl", sa + 0.035, sb - 0.035, ya + 0.035, yb - 0.09, CZ + sk["rim"] - 0.004, CZ + sk["rim"] + 0.0006, "sink_dark", 0.0002)
    tx = 0.5*(sa + sb)
    S.tube("/Dressing/sink/tap", [(tx, yb - 0.035, CZ + sk["rim"]), (tx, yb - 0.035, CZ + sk["tap_z"] - 0.03), (tx, yb - 0.07, CZ + sk["tap_z"]),
                                  (tx, yb - 0.17, CZ + sk["tap_z"] - 0.01), (tx, yb - 0.19, CZ + sk["tap_z"] - 0.04)], 0.011, S.mat("chrome"), sides=14)
    S.tube("/Dressing/sink/lever", [(tx, yb - 0.035, CZ + sk["tap_z"] - 0.02), (tx + 0.10, yb - 0.02, CZ + sk["tap_z"] + 0.02)], 0.006, S.mat("chrome"), sides=10)

    # ---- the bed-head services trunking: gas outlets, sockets; the policy camera on its arm
    tr = P.TRUNKING
    tz0, tz1 = tr["z"], tr["z"] + tr["height"]
    yt = y1 - tr["depth"]
    bp, bh = boxes["trunking"]
    S.box("/Dressing/trunking/body", 2*bh, S.mat("trunking"), p=bp, r=0.006)
    for k, (ox, col) in enumerate(((0.95, "outlet_white"), (0.78, "outlet_black"), (0.61, "outlet_yellow"))):
        S.cylinder(f"/Dressing/trunking/outlet{k}", 0.027, 0.012, S.mat(col), p=(ox, yt - 0.004, 0.5*(tz0 + tz1)), axis="Y")
        S.cylinder(f"/Dressing/trunking/outlet{k}_core", 0.011, 0.02, S.mat("steel_brushed"), p=(ox, yt - 0.008, 0.5*(tz0 + tz1)), axis="Y")
    for k in range(4):
        sx = -0.95 + 0.11*k
        _bx(S, f"/Dressing/trunking/socket{k}", sx - 0.043, sx + 0.043, yt - 0.006, yt, 0.5*(tz0 + tz1) - 0.043, 0.5*(tz0 + tz1) + 0.043, "socket_white", 0.004)
    for name in ("fit_station_camera", "fit_station_camera_arm"):
        p_, h_ = boxes[name]
        S.box(f"/Dressing/trunking/{name}", 2*h_, S.mat("camera_dark"), p=p_, r=0.004)
    cam = boxes["fit_station_camera"][0]
    S.cylinder("/Dressing/trunking/lens", 0.012, 0.006, S.mat("outlet_black"), p=(cam[0], cam[1] - boxes["fit_station_camera"][1][1] - 0.002, cam[2]), axis="Y")

    # ---- the ceiling recording camera (a policy camera, bay_rgb) on a drop rod; its housing stands behind the lens
    bc = P.BAY_CAMERA
    eye, d = np.asarray(bc["eye"], float), np.asarray(bc["target"], float) - np.asarray(bc["eye"], float)
    d /= np.linalg.norm(d)
    hc = eye - 0.065*d
    S.box("/Dressing/ceiling/bay_camera", (0.07, 0.07, 0.06), S.mat("camera_dark"), p=tuple(hc), r=0.006)
    S.cylinder("/Dressing/ceiling/bay_camera_rod", 0.012, h - (hc[2] + 0.03), S.mat("steel_brushed"), p=(hc[0], hc[1], 0.5*(h + hc[2] + 0.03)))
    S.box("/Dressing/ceiling/bay_camera_plate", (0.10, 0.10, 0.008), S.mat("trunking"), p=(hc[0], hc[1], h - 0.004), r=0.002)

    # ---- what hangs on the wall: glove boxes in a rack, towel dispenser, hand rub, clock, the checklist
    f = P.WALL_FITTINGS
    gx, gw, gh, gd, gz = f["gloves"]
    _bx(S, "/Dressing/wall/glove_rack", gx - 0.5*gw, gx + 0.5*gw, y1 - 0.012, y1, gz, gz + gh, "steel_brushed", 0.002)
    for k in range(3):
        a = gx - 0.5*gw + 0.01 + k*(gw - 0.02)/3
        _bx(S, f"/Dressing/wall/glove_box{k}", a, a + (gw - 0.02)/3 - 0.008, y1 - gd, y1 - 0.012, gz + 0.012, gz + gh - 0.012, "glove_box", 0.004)
        _bx(S, f"/Dressing/wall/glove_band{k}", a, a + (gw - 0.02)/3 - 0.008, y1 - gd - 0.0006, y1 - gd, gz + 0.012, gz + 0.075, "glove_blue", 0.0002)
    px_, pw, ph, pd, pz = f["towels"]
    _bx(S, "/Dressing/wall/towels", px_ - 0.5*pw, px_ + 0.5*pw, y1 - pd, y1, pz, pz + ph, "dispenser", 0.02)
    _bx(S, "/Dressing/wall/towel_slot", px_ - 0.4*pw, px_ + 0.4*pw, y1 - pd + 0.02, y1 - pd + 0.10, pz - 0.001, pz + 0.004, "outlet_black", 0.001)
    hx_, hw, hh, hd, hz = f["handrub"]
    _bx(S, "/Dressing/wall/handrub_bracket", hx_ - 0.5*hw, hx_ + 0.5*hw, y1 - 0.02, y1, hz, hz + hh, "dispenser", 0.004)
    S.cylinder("/Dressing/wall/handrub_bottle", 0.036, hh - 0.06, S.mat("handrub"), p=(hx_, y1 - 0.058, hz + 0.5*(hh - 0.06)))
    S.cylinder("/Dressing/wall/handrub_pump", 0.012, 0.05, S.mat("dispenser"), p=(hx_, y1 - 0.058, hz + hh - 0.035))
    kx, kw, kh_, kd, kz = f["clock"]
    S.cylinder("/Dressing/wall/clock_rim", 0.5*kw, kd, S.mat("outlet_black"), p=(kx, y1 - 0.5*kd, kz + 0.5*kh_), axis="Y")
    S.cylinder("/Dressing/wall/clock_face", 0.5*kw - 0.012, kd + 0.002, S.mat("paper"), p=(kx, y1 - 0.5*kd - 0.001, kz + 0.5*kh_), axis="Y")
    for k in range(12):
        a = 2*math.pi*k/12
        rr = 0.5*kw - 0.03
        S.box(f"/Dressing/wall/clock_mark{k}", (0.006, 0.002, 0.018), S.mat("print_dark"), p=(kx + rr*math.sin(a), y1 - kd - 0.002, kz + 0.5*kh_ + rr*math.cos(a)),
              q=tuple(Rotation.from_euler("y", a).as_quat()), r=0.0005)
    # the hands are dynamic meshes that show the simulated time (render.clock_meshes)
    S.cylinder("/Dressing/wall/clock_boss", 0.006, 0.006, S.mat("print_dark"), p=(kx, y1 - kd - 0.006, kz + 0.5*kh_), axis="Y")
    ox, ow, oh, od, oz = f["poster"]
    from . import task as TK
    img = _poster(tex, TK.task_spec(world.config, sc.layout))
    _image_quad(S, "/Dressing/wall/poster", [(ox - 0.5*ow, y1 - 0.0045, oz), (ox + 0.5*ow, y1 - 0.0045, oz), (ox + 0.5*ow, y1 - 0.0045, oz + oh),
                                             (ox - 0.5*ow, y1 - 0.0045, oz + oh)], img, "poster")
    _bx(S, "/Dressing/wall/poster_back", ox - 0.5*ow - 0.006, ox + 0.5*ow + 0.006, y1 - 0.004, y1, oz - 0.006, oz + oh + 0.006, "dispenser", 0.002)

    # ---- on the counter: the vial's printed ring, the pad's mat, the sharps container's label
    if "vial" in sc.bodies and hasattr(sc, "vial_start"):
        vx, vy = sc.vial_start
        _annulus(S, "/Dressing/counter/vial_ring", (vx, vy), 0.019, 0.020, CZ + 0.0002, "print_dark")
    if "pad_mat" in boxes:
        mp, mh = boxes["pad_mat"]
        S.box("/Dressing/counter/pad_mat", 2*mh, S.mat("mat"), p=mp, r=0.0008)
    if "sharps_wall_front" in boxes:
        fp, fh = boxes["sharps_wall_front"]
        yq = fp[1] - fh[1] - 0.0006
        _image_quad(S, "/Dressing/counter/sharps_label", [(fp[0] - 0.09, yq, fp[2] - 0.02), (fp[0] + 0.09, yq, fp[2] - 0.02), (fp[0] + 0.09, yq, fp[2] + 0.088),
                                                           (fp[0] - 0.09, yq, fp[2] + 0.088)], _sharps_label(tex), "sharps_label")

    # ---- the drip stand: five-leg base on castors, pole, hooks, a bag of saline with its set clamped and coiled
    pp, ph_ = boxes["stand_pole"]
    dx, dy = pp[0], pp[1]
    steel = S.mat("steel_brushed")
    S.cylinder("/Dressing/stand/pole", 0.0125, 1.84, steel, p=(dx, dy, 0.98))
    for k in range(5):
        a = 2*math.pi*k/5 + 0.3
        ex, ey = dx + 0.27*math.cos(a), dy + 0.27*math.sin(a)
        S.tube(f"/Dressing/stand/leg{k}", [(dx, dy, 0.085), (ex, ey, 0.062)], 0.013, S.mat("stand_base"), sides=10)
        S.sphere(f"/Dressing/stand/castor{k}", 0.028, S.mat("stand_base"), p=(ex, ey, 0.028))
    S.cylinder("/Dressing/stand/hub", 0.045, 0.05, S.mat("stand_base"), p=(dx, dy, 0.085))
    for k in range(4):
        a = math.pi/4 + k*math.pi/2
        S.tube(f"/Dressing/stand/hook{k}", [(dx, dy, 1.88), (dx + 0.10*math.cos(a), dy + 0.10*math.sin(a), 1.90), (dx + 0.13*math.cos(a), dy + 0.13*math.sin(a), 1.87),
                                            (dx + 0.12*math.cos(a), dy + 0.12*math.sin(a), 1.85)], 0.004, steel, sides=8)
    from .render import _translucent
    bag = _translucent(S, "saline_bag_film", (.82, .86, .88), 0.55, 0.15)
    bxp = (dx + 0.12*math.cos(math.pi/4), dy + 0.12*math.sin(math.pi/4))
    S.box("/Dressing/stand/bag", (0.13, 0.035, 0.25), bag, p=(bxp[0], bxp[1], 1.70), r=0.016)
    S.cylinder("/Dressing/stand/drip_chamber", 0.008, 0.05, bag, p=(bxp[0], bxp[1], 1.535))
    coil = [(bxp[0], bxp[1], 1.51), (bxp[0], bxp[1], 1.30)] + [(bxp[0] + 0.05*math.cos(t) - 0.05, bxp[1] + 0.02*math.sin(2*t), 1.30 - 0.012*k + 0.05*math.sin(t))
                                                                for k, t in enumerate(np.linspace(0, 6*math.pi, 40))]
    S.tube("/Dressing/stand/set", coil, 0.002, bag, sides=8)

    # ---- the bed space behind the robot: bed with a manikin under a blanket, an over-bed table
    bd = P.BED
    (ba, bb), (bya, byb), bz = bd["x"], bd["y"], bd["z"]
    _bx(S, "/Dressing/bed/frame", ba, bb, bya, byb, 0.28, 0.40, "bed_frame", 0.02)
    _bx(S, "/Dressing/bed/mattress", ba + 0.03, bb - 0.03, bya + 0.03, byb - 0.03, 0.40, bz, "linen", 0.04)
    _bx(S, "/Dressing/bed/head", bb - 0.03, bb + 0.02, bya, byb, 0.28, 1.02, "bed", 0.02)
    _bx(S, "/Dressing/bed/foot", ba - 0.02, ba + 0.03, bya, byb, 0.28, 0.82, "bed", 0.02)
    for k, (lx, ly) in enumerate(((ba + 0.1, bya + 0.08), (ba + 0.1, byb - 0.08), (bb - 0.1, bya + 0.08), (bb - 0.1, byb - 0.08))):
        S.cylinder(f"/Dressing/bed/leg{k}", 0.025, 0.28, S.mat("bed_frame"), p=(lx, ly, 0.14))
    ym = 0.5*(bya + byb)
    _bx(S, "/Dressing/bed/pillow", bb - 0.50, bb - 0.08, ym - 0.28, ym + 0.28, bz - 0.01, bz + 0.10, "linen", 0.05)
    S.sphere("/Dressing/bed/manikin_head", 0.095, S.mat("manikin"), p=(bb - 0.30, ym, bz + 0.16))
    _bx(S, "/Dressing/bed/manikin_body", ba + 0.25, bb - 0.46, ym - 0.24, ym + 0.24, bz - 0.02, bz + 0.20, "blanket", 0.09)
    _bx(S, "/Dressing/bed/blanket", ba + 0.05, bb - 0.62, bya + 0.04, byb - 0.04, bz - 0.01, bz + 0.035, "blanket", 0.03)
    tb = (ba - 0.50, 0.5*(bya + byb) + 0.25)             # at the foot end, out of both arms' reach
    S.cylinder("/Dressing/bed/table_post", 0.02, 0.74, S.mat("bed_frame"), p=(tb[0], tb[1], 0.42))
    _bx(S, "/Dressing/bed/table_top", tb[0] - 0.38, tb[0] + 0.38, tb[1] - 0.52, tb[1] + 0.03, 0.80, 0.825, "solid_surface", 0.008)
    _bx(S, "/Dressing/bed/table_base", tb[0] - 0.33, tb[0] + 0.33, tb[1] - 0.03, tb[1] + 0.03, 0.03, 0.06, "bed_frame", 0.01)

    # ---- the curtain drawn back on its ceiling track at the bay's open side; the rest of the laboratory
    by0 = R["y0"]
    _bx(S, "/Dressing/curtain/track", x0, x1, by0 - 0.015, by0 + 0.015, h - 0.03, h, "curtain_track", 0.004)
    n_f = 9
    for k in range(n_f):
        cxk = x0 + 0.05 + 0.055*k
        S.cylinder(f"/Dressing/curtain/fold{k}", 0.034, 2.08, S.mat("curtain"), p=(cxk, by0 + 0.02*math.sin(1.7*k), h - 0.05 - 1.04))
    _bx(S, "/Dressing/lab/desk_top", -0.9, 0.9, y0 + 0.02, y0 + 0.72, 0.72, 0.75, "solid_surface", 0.006)
    for k, lx in enumerate((-0.85, 0.85)):
        _bx(S, f"/Dressing/lab/desk_side{k}", lx - 0.02, lx + 0.02, y0 + 0.04, y0 + 0.70, 0.0, 0.72, "casework", 0.004)
    for k, cx in enumerate((-0.45, 0.45)):
        _bx(S, f"/Dressing/lab/chair{k}_seat", cx - 0.22, cx + 0.22, y0 + 0.78, y0 + 1.22, 0.44, 0.50, "chair", 0.03)
        _bx(S, f"/Dressing/lab/chair{k}_back", cx - 0.21, cx + 0.21, y0 + 1.20, y0 + 1.25, 0.52, 0.92, "chair", 0.03)
        S.cylinder(f"/Dressing/lab/chair{k}_post", 0.025, 0.36, S.mat("bed_frame"), p=(cx, y0 + 1.0, 0.25))
        S.cylinder(f"/Dressing/lab/chair{k}_base", 0.26, 0.03, S.mat("stand_base"), p=(cx, y0 + 1.0, 0.045))
    _bx(S, "/Dressing/lab/monitor", -0.28, 0.28, y0 + 0.16, y0 + 0.19, 0.86, 1.20, "outlet_black", 0.008)
    _bx(S, "/Dressing/lab/monitor_foot", -0.10, 0.10, y0 + 0.10, y0 + 0.26, 0.75, 0.765, "outlet_black", 0.004)
    S.cylinder("/Dressing/lab/monitor_neck", 0.015, 0.12, S.mat("outlet_black"), p=(0., y0 + 0.18, 0.82))
    _bx(S, "/Dressing/lab/trolley", x1 - 0.62, x1 - 0.12, -3.6, -3.05, 0.12, 0.90, "casework", 0.01)
    _bx(S, "/Dressing/lab/trolley_top", x1 - 0.64, x1 - 0.10, -3.62, -3.03, 0.90, 0.93, "solid_surface", 0.006)

    # ---- light
    ref = (float(sc.layout["tray"][0]), float(sc.layout["tray"][1]), CZ)
    lum = TARGET_LUX/panel_gain(ref)
    emis = S.color("led_panel_lit", (.95, .95, .92), .6, 0., emissive=LIGHT_RGB, emissive_intensity=float(lum))
    for k, (lx, ly) in enumerate(PANELS):
        _bx(S, f"/Dressing/lights/panel{k}_frame", lx - 0.31, lx + 0.31, ly - 0.31, ly + 0.31, h - 0.012, h - 0.001, "trunking", 0.002)
        S.quad(f"/Dressing/lights/panel{k}_lens", [(lx - 0.297, ly + 0.297, h - 0.0125), (lx + 0.297, ly + 0.297, h - 0.0125), (lx + 0.297, ly - 0.297, h - 0.0125),
                                                   (lx - 0.297, ly - 0.297, h - 0.0125)], emis, double=True)
        lt = S.UsdLux.RectLight.Define(S.stage, f"/Lights/panel{k}")
        lt.CreateWidthAttr(PANEL_SIZE[0])
        lt.CreateHeightAttr(PANEL_SIZE[1])
        lt.CreateIntensityAttr(float(lum))
        lt.CreateColorAttr(Gf.Vec3f(*LIGHT_RGB))
        S.place(lt, (lx, ly, h - 0.02))
    try:
        from whiteboard import visual_assets as WVA
        hdr = WVA.asset("kiara_5_noon", "hdri")
        dome = S.UsdLux.DomeLight.Define(S.stage, "/Lights/daylight")
        dome.CreateTextureFileAttr(S.Sdf.AssetPath(str(Path(hdr).resolve())))
        dome.CreateIntensityAttr(700.0)
        S.UsdGeom.Xformable(dome).AddRotateZOp().Set(200.0)
        credits.add("kiara_5_noon")
    except Exception as exc:                              # no pinned sky on this machine: a plain dome
        dome = S.UsdLux.DomeLight.Define(S.stage, "/Lights/daylight")
        dome.CreateIntensityAttr(500.0)
        credits.add(f"plain sky ({type(exc).__name__})")
    S.stage.GetPrimAtPath("/Dressing").SetCustomDataByKey("lighting", dict(target_lux=TARGET_LUX, reference=list(ref), panel_nit=float(lum)))
    return credits

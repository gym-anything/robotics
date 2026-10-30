"""USD export of the bedroom being repainted and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; each request carries the body poses and only what changed in the paint: the film
charts' textures (diffuse = Kubelka-Munk reflectance of substrate -> dry coats -> wet layer; roughness = wet gloss vs
dry sheen; normal = slopes of the film's relief plus its sub-grid stipple/brush marks), the roller nap's paint load,
the brush tufts' bent shapes, the pool levels in the containers and the airborne spray and drops. Nothing is painted
on by the renderer: every pixel of paint is a field of the physics state.

Appearance rules: collision primitives are drawn as the parts they stand for; the robot uses the
franka_description visual meshes (classroom_arrangement's drawing); room dressing outside the working volume comes from
pinned CC0 Poly Haven assets (painting/visual_assets.py).
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
import zlib
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement.render import Stage, hdri, clip_mesh, EXTERIOR_HDRI
from . import params as P
from . import layout as L
from . import tools as T
from .paint import srgb_to_linear, linear_to_srgb8

COLORS = dict(
    trim=((.93, .93, .90), .25, 0.), ceiling=((.92, .92, .90), .9, 0.), glass=((.10, .12, .14), .02, 0.),
    sash=((.94, .94, .92), .35, 0.), radiator=((.86, .83, .74), .45, .0), pipe=((.80, .78, .72), .4, .3),
    plate=((.95, .95, .93), .3, 0.), outlet=((.92, .91, .87), .35, 0.), slot=((.02, .02, .02), .8, 0.),
    robot=((.88, .88, .87), .35, 0.), robot_dark=((.07, .075, .08), .45, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    tmr_black=((.05, .05, .055), .5, 0.), tmr_deck=((.80, .81, .80), .4, 0.), aluminum=((.70, .72, .74), .3, .85),
    tray=((.07, .07, .075), .5, 0.), steel=((.70, .71, .73), .3, .9), grip=((.10, .25, .55), .6, 0.),
    grip_band=((.85, .65, .10), .5, 0.), ferrule=((.80, .78, .70), .25, .95), alder=((.62, .44, .28), .55, 0.),
    bristle=((.80, .66, .42), .6, 0.), pail=((.95, .95, .95), .5, 0.), can_steel=((.75, .76, .78), .3, .9),
    can_label=((.94, .93, .90), .6, 0.), palette=((.97, .97, .96), .3, 0.), table_top=((.92, .92, .90), .5, 0.),
    steel_dark=((.25, .26, .28), .4, .6), mdf=((.55, .42, .30), .6, 0.), cardboard=((.36, .23, .11), .9, 0.),
    aerosol=((.70, .10, .12), .3, .4), aerosol_metal=((.80, .80, .82), .25, .9), button=((.95, .95, .95), .4, 0.),
    pet=((.90, .93, .95), .05, 0.), plastic_sheet=((.92, .94, .96), .15, 0.), nap=((.96, .95, .92), .9, 0.),
    tape_blue=((.20, .45, .85), .7, 0.), pencil=((.35, .35, .38), .8, 0.), floor_dark=((.25, .17, .10), .6, 0.),
    lamp=((1., 1., 1.), .3, 0.), stick=((.82, .70, .52), .7, 0.),
)


def _rot_to_wxyz(R):
    x, y, z, w = R.as_quat()
    return float(w), float(x), float(y), float(z)


def _lin8(x):
    return (np.clip(x, 0.0, 1.0)*255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------ film appearance (shared with probes)
class ChartImage:
    """Texture arrays of one chart, updated over the window that changed since the last request."""

    def __init__(self, chart, rough_srgb=True):
        self.ch = chart
        self.version = -1
        self.prev = None
        nu, nv = chart.nu, chart.nv
        self.rgba = np.zeros((nv, nu, 4), np.uint8); self.rgba[..., 3] = 255
        self.rough = np.zeros((nv, nu, 4), np.uint8); self.rough[..., 3] = 255
        self.normal = np.zeros((nv, nu, 4), np.uint8); self.normal[..., 3] = 255
        rng = np.random.default_rng(zlib.crc32(chart.name.encode()))
        from scipy.ndimage import gaussian_filter
        base = rng.standard_normal((nu, nv))
        self.n_iso = gaussian_filter(base, 0.8); self.n_iso /= self.n_iso.std() + 1e-9
        self.n_su = gaussian_filter(base, (3.0, 0.6)); self.n_su /= self.n_su.std() + 1e-9      # streaks along u
        self.n_sv = gaussian_filter(base, (0.6, 3.0)); self.n_sv /= self.n_sv.std() + 1e-9      # streaks along v
        # static relief of the bare substrate (appearance only): canvas drop-cloth folds, drywall orange peel,
        # cold-press paper tooth
        sub = chart.substrate
        if sub == "canvas_cloth":
            folds = gaussian_filter(rng.standard_normal((nu, nv)), 25.0)
            self.base = 2.5e-3*folds/(folds.std() + 1e-12)
        elif sub in ("primed_drywall", "spackle"):
            self.base = 15e-6*self.n_iso
        elif sub == "watercolour_paper":
            self.base = 40e-6*gaussian_filter(base, 1.2)/(gaussian_filter(base, 1.2).std() + 1e-12)
        elif chart.name == "tray_ramp":
            # the ramp's moulded ribs (2 mm high, 12 mm apart) as relief under the paint film (appearance only: their
            # effect on the nap's loading is lumped into the uptake rate, E8)
            u = (np.arange(nu) + 0.5)*chart.dx
            ph = (u % 0.012)/0.012
            self.base = np.repeat((P.TRAY["ramp_ribs"]*np.exp(-((ph - 0.5)/0.12)**2))[:, None], nv, axis=1)
        else:
            self.base = np.zeros((nu, nv))
        # the bare substrate's own pattern (canvas weave, corrugated board, plywood grain) as a relative albedo sampled
        # from the pinned texture at the chart's physical scale; paint hides it (weight exp(-film / 10 um))
        self.albedo = substrate_albedo(chart)
        self.full()

    def full(self):
        self._update((slice(0, self.ch.nu), slice(0, self.ch.nv)))
        self.version = self.ch.version
        self.prev = self.ch.window(margin=2)

    def refresh(self):
        """Recompute the changed window; returns True if anything changed."""
        if self.ch.version == self.version:
            return False
        win = self.ch.window(margin=2)
        boxes = [b for b in (win, self.prev) if b is not None]
        self.version = self.ch.version
        self.prev = win
        if not boxes:
            return True
        i0 = min(b[0] for b in boxes); i1 = max(b[1] for b in boxes)
        j0 = min(b[2] for b in boxes); j1 = max(b[3] for b in boxes)
        self._update((slice(i0, i1), slice(j0, j1)))
        return True

    def _update(self, sl):
        ch = self.ch
        R = ch.reflectance(sl)
        # relief: wet + dry thickness plus sub-grid texture (stipple, brush streaks along the stroke direction)
        H = ch.h(sl) + ch.dry_total(sl)
        if self.albedo is not None:
            R = R*(self.albedo[sl]**np.exp(-H/10e-6))[..., None]
        # the worker's dynamic textures are sampled as linear data (checked against the constant-colour side walls,
        # which use the same linear old colour: sRGB-encoded bytes rendered the work wall pale), so bytes are linear
        rgb = _lin8(R)                                                   # (ni, nj, 3)
        self.rgba[sl[1], sl[0], :3] = rgb.transpose(1, 0, 2)
        rough = ch.roughness()[sl]
        self.rough[sl[1], sl[0], :3] = _lin8(rough)[..., None].transpose(1, 0, 2)
        ang = ch.tdir[sl]
        streak = np.abs(np.cos(ang))*self.n_su[sl] + np.abs(np.sin(ang))*self.n_sv[sl]
        brushy = ch.twl[sl] < 1.5e-3
        Hs = H + ch.tex[sl]*np.where(brushy, streak, self.n_iso[sl]) + self.base[sl]
        gx = np.gradient(np.pad(Hs, 1, mode="edge"), ch.dx, axis=0)[1:-1, 1:-1]
        gy = np.gradient(np.pad(Hs, 1, mode="edge"), ch.dx, axis=1)[1:-1, 1:-1]
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        enc = _lin8(0.5*(n + 1.0))
        self.normal[sl[1], sl[0], :3] = enc.transpose(1, 0, 2)[::1]
        # image rows run top (v max) to bottom: flip at transmission
        return True

    def payload(self, kind):
        a = {"diffuse": self.rgba, "rough": self.rough, "normal": self.normal}[kind][::-1]
        if self.ch.dx >= 3.5e-3:
            # coarse (4 mm) charts are sent at twice the resolution, bilinearly interpolated: drawn cell by cell their
            # overspray would be a staircase of squares on the booth floor; appearance only
            from scipy.ndimage import zoom
            a = np.clip(zoom(a.astype(np.float32), (2, 2, 1), order=1), 0, 255).astype(np.uint8)
            a[..., 3] = 255
        return dict(w=int(a.shape[1]), h=int(a.shape[0]), data=base64.b64encode(np.ascontiguousarray(a).tobytes()).decode())


SUBSTRATE_PATTERN = {"canvas_cloth": ("texture", "rough_linen", 0.25, 0.6),
                     "cardboard": ("blotch", None, 0.0, 0.07),
                     "plywood_sealed": ("texture", "plywood", 0.6, 0.6),
                     # the clear 12 mil PET stencil lying on the plywood crate: drawn as the plywood seen through a faint
                     # frosted film (its colour and a damped grain); the renderer's opacity path did not show through
                     "pet_on_plywood": ("texture", "plywood", 0.6, 0.35)}


def substrate_albedo(chart):
    """Relative albedo (mean 1) of a chart's bare substrate from a pinned CC0 texture tiled at `tile` metres, its
    contrast scaled by `gain` (corrugated board: smooth blotches of the kraft liner; its 4.8 mm flutes would alias on
    the booth's 4 mm cells); None for substrates drawn uniform (primed drywall, paper, trim)."""
    spec = SUBSTRATE_PATTERN.get(chart.substrate)
    if spec is None:
        return None
    from PIL import Image
    from .visual_assets import texture
    kind, name, tile, gain = spec
    if kind == "blotch":
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(zlib.crc32(chart.name.encode()) + 1)
        blot = gaussian_filter(rng.standard_normal((chart.nu, chart.nv)), 6.0)
        blot /= blot.std() + 1e-12
        return np.clip(1.0 + gain*blot, 0.8, 1.2)
    from PIL import ImageFilter
    im = Image.open(texture(name)["diff"]).convert("L")
    # low-pass to the chart's cell before point sampling (threads finer than a 4 mm cell would alias into moire)
    im = im.filter(ImageFilter.GaussianBlur(radius=max(0.5, 0.5*chart.dx/tile*im.size[0])))
    img = np.asarray(im, float)/255.0
    H_, W_ = img.shape
    u = ((np.arange(chart.nu) + 0.5)*chart.dx/tile % 1.0)*W_
    v = ((np.arange(chart.nv) + 0.5)*chart.dx/tile % 1.0)*H_
    lum = img[np.clip(v.astype(int), 0, H_ - 1)][:, np.clip(u.astype(int), 0, W_ - 1)].T   # (nu, nv)
    lum = lum/max(float(lum.mean()), 1e-6)
    return np.clip(1.0 + gain*(lum - 1.0), 0.5, 1.5)


def wetted(load_fraction, scale=0.10):
    """Share of fibre surface coated with paint at a fill fraction of the fibres' uptake capacity: fibres look the
    paint's colour once their surfaces are wetted, long before the pores are full (1 - exp(-f / 0.10): 63 % at a tenth of
    capacity, 92 % at a quarter). A linear blend by the liquid fraction would render a nap holding 63 g of navy -- 85 %
    of capacity -- pale grey."""
    return 1.0 - np.exp(-np.clip(load_fraction, 0, None)/scale)


def nap_image(roller):
    """Roller cover texture (around x along): nap colour toward the wet paint's colour by the wetted fibre share."""
    from .roller import SAT
    V = (roller.s.sum(axis=0) + roller.v).reshape(roller.na, roller.nl)
    sat = np.clip(V/(SAT*roller.C0), 0, 1.6)
    pt = roller.sys.paints[0]
    base = srgb_to_linear((240, 238, 230))
    c = wetted(sat)[..., None]
    col = base[None, None]*(1 - c) + pt.rinf[None, None]*c*0.92
    col *= (1 - 0.15*np.clip(sat - 1, 0, 1))[..., None]
    img = np.zeros((roller.nl, roller.na, 4), np.uint8)
    img[..., :3] = _lin8(col).transpose(1, 0, 2)
    img[..., 3] = 255
    return dict(w=roller.na, h=roller.nl, data=base64.b64encode(np.ascontiguousarray(img).tobytes()).decode())


# ------------------------------------------------------------------------------ export
def export(env, folder, extra_cameras=None):
    """Export the episode's static appearance to folder/scene.usdc; returns (path, meta)."""
    import newton
    world, sc, paint = env.world, env.world.scene, env.paint
    b = sc.b
    S = Stage(folder)
    mats = {name: S.color(name, *v) for name, v in COLORS.items()}
    _textured_mats(S, mats, env.spec)
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    _robot(S, world, mats)
    _room(S, mats, env.spec)
    meta = dict(textures={}, charts={}, curves={}, pools={}, points={}, naps={})
    _tools(S, world, mats, env, meta)
    _charts(S, paint, meta)
    _dressing(S, mats, env.spec)
    meta["cameras"] = _cameras(S, world, env.spec)
    for name, (eye, target, f) in (extra_cameras or {}).items():
        if name in meta["cameras"]:
            raise ValueError(f"extra camera {name!r} would replace a built-in camera")
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); meta["cameras"][name] = "diagnostic"
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def _textured_mats(S, mats, spec):
    from .visual_assets import texture
    fl = texture("plank_flooring_02")
    # a neutral tint: a warm (1.0, 0.92, 0.80) multiplier on an already warm texture draws a saturated orange floor
    mats["floor"] = S.textured("floor_oak", fl["diff"], rough=.45, normal=fl["nor_gl"], rough_tex=fl["rough"], tint=(0.86, 0.84, 0.80))
    pl = texture("painted_plaster_wall")
    old = srgb_to_linear(P.COLORS[spec.get("old_color", "accessible_beige")])
    mats["wall_old"] = _pbr_const(S, "wall_old", old, rough=.82, normal=pl["nor_gl"], normal_strength=0.35)
    ln = texture("rough_linen")
    mats["linen_normal"] = ln
    ply = texture("plywood")
    mats["plywood"] = S.textured("plywood_tex", ply["diff"], rough=.6, normal=ply["nor_gl"])


def _pbr_const(S, name, rgb_linear, rough=.6, normal=None, normal_strength=1.0):
    """OmniPBR with a constant (linear) diffuse colour and an optional tangent-space normal map (surface relief only)."""
    T_, Sdf, UsdShade = S.Sdf.ValueTypeNames, S.Sdf, S.UsdShade
    m = UsdShade.Material.Define(S.stage, "/Looks/" + name)
    sh = UsdShade.Shader.Define(S.stage, f"/Looks/{name}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", T_.Color3f).Set(S.Gf.Vec3f(*map(float, rgb_linear)))
    sh.CreateInput("reflection_roughness_constant", T_.Float).Set(float(rough))
    if normal is not None:
        sh.CreateInput("normalmap_texture", T_.Asset).Set(Sdf.AssetPath(str(Path(normal).resolve())))
        sh.CreateInput("bump_factor", T_.Float).Set(float(normal_strength))
    m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    S.mats[name] = m
    return m


def _robot(S, world, mats):
    """franka_description visual meshes with their material colours (classroom_arrangement's drawing rules)."""
    import newton
    sc, b = world.scene, world.scene.b
    V, C = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    for i in sc.robot_shapes:
        flags = int(b.shape_flags[i])
        body = b.shape_body[i]
        label = b.body_label[body].split("/")[-1] if body >= 0 else ""
        mname = sc.materials.get(i)
        tf = np.asarray(b.shape_transform[i], float)
        if mname in ("camera_dark", "aluminum"):
            S.box(f"/Lab/body_{body}/shape_{i}", 2*np.asarray(b.shape_scale[i], float), mats[mname], tf[:3], tf[3:])
            continue
        if flags & C or not flags & V:
            continue
        src = b.shape_source[i]
        scale = np.asarray(b.shape_scale[i], float)
        path = f"/Lab/body_{body}/shape_{i}"
        col = tuple(float(x) for x in b.shape_color[i])
        if src is not None and hasattr(src, "vertices"):
            v = np.asarray(src.vertices)*scale
            if np.abs(v).max() > 5.:
                v = v*1e-3
            f = np.asarray(src.indices).reshape(-1, 3)
            if label == "base_link" and len(v) > 100000:
                v2, below, above = clip_mesh(v, f, 0.26)
                for part, faces, m in (("skirt", below, mats["tmr_black"]), ("deck", above, mats["tmr_deck"])):
                    S.mesh(f"{path}_{part}", v2, faces, mat=m, smooth=True, p=tf[:3], q=tf[3:])
                continue
            m = mats["robot"] if label == "base_link" else S.color(
                f"robot_{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}", col, .35, 0.)
            S.mesh(path, v, f, mat=m, smooth=True, p=tf[:3], q=tf[3:])
        elif b.shape_type[i] == newton.GeoType.BOX:
            S.box(path, 2*scale, mats["tmr_black"], tf[:3], tf[3:])
        elif b.shape_type[i] == newton.GeoType.CYLINDER:
            S.cylinder(path, scale[0], 2*scale[1], mats["robot_dark"], tf[:3], tf[3:])


def _room(S, mats, spec):
    X, Y, Z, WT, YW = L.X, L.Y, L.Z, L.WT, L.YW
    W, CS, CT = L.WIN, L.CS, L.CT
    UsdLux, Gf = S.UsdLux, S.Gf
    # oak strip floor (UVs in metres / 2 m tile)
    pts = [(-X/2, -Y/2, 0), (X/2, -Y/2, 0), (X/2, Y/2, 0), (-X/2, Y/2, 0)]
    S.mesh("/Room/floor", pts, [0, 1, 2, 0, 2, 3], uv=[(p[0]/2.0, p[1]/2.0) for p in pts], normals=[(0, 0, 1)]*4, mat=mats["floor"])
    # walls: the work wall's paintable face is its film chart (drawn by _charts); its body is set back 1 mm
    for name, c, h, mat in L.static_boxes():
        if name.startswith("rad_") or name in ("outlet_plate", "glazing"):
            continue
        m = {"drywall": mats["wall_old"], "trim": mats["trim"], "ceiling": mats["ceiling"]}.get(mat, mats["trim"])
        cc = np.array(c, float)
        if name.startswith("wall_n"):
            cc[1] += 0.001
        if name == "ceiling":
            S.box(f"/Room/{name}", 2*np.asarray(h), m, cc, (0, 0, 0, 1), bevel=0.001, uv_per_m=0.5)
            continue
        S.box(f"/Room/{name}", 2*np.asarray(h), m, cc, (0, 0, 0, 1), bevel=0.002 if mat == "trim" else 0.001,
              uv_per_m=0.5 if mat == "drywall" else None)
    # window: double-hung sashes, meeting rail, glass, exterior daylight
    wx = W["x"]; yg = YW + 0.07
    for k, (z0, z1) in enumerate(((W["sill"], (W["sill"] + W["head"])/2 + 0.02), ((W["sill"] + W["head"])/2 - 0.02, W["head"]))):
        yy = yg - 0.02 + 0.03*k
        S.box(f"/Room/sash_bot_{k}", (W["w"], 0.035, 0.05), mats["sash"], (wx, yy, z0 + 0.025), (0, 0, 0, 1))
        S.box(f"/Room/sash_top_{k}", (W["w"], 0.035, 0.05), mats["sash"], (wx, yy, z1 - 0.025), (0, 0, 0, 1))
        for j, x in enumerate((wx - W["w"]/2 + 0.025, wx + W["w"]/2 - 0.025)):
            S.box(f"/Room/sash_side_{k}_{j}", (0.05, 0.035, z1 - z0), mats["sash"], (x, yy, (z0 + z1)/2), (0, 0, 0, 1))
    for j, x in enumerate((L.WX0 - 0.005, L.WX1 + 0.005)):
        S.box(f"/Room/jamb_{j}", (0.012, 0.11, W["head"] - W["sill"]), mats["trim"], (x, YW + 0.055, (W["sill"] + W["head"])/2), (0, 0, 0, 1))
    # outlet: plate (physics box) with a duplex receptacle face
    o = L.OUT
    S.box("/Room/outlet_plate", (o["w"], o["t"], o["h"]), mats["plate"], (o["x"], YW - o["t"]/2, o["z"]), (0, 0, 0, 1), bevel=0.0015)
    for k, dz in enumerate((-0.019, 0.019)):
        S.box(f"/Room/receptacle_{k}", (0.034, 0.004, 0.030), mats["outlet"], (o["x"], YW - o["t"] - 0.0015, o["z"] + dz), (0, 0, 0, 1),
              bevel=0.003)
        for j, dx in enumerate((-0.0064, 0.0064)):
            S.box(f"/Room/slot_{k}_{j}", (0.0022, 0.002, 0.008 if j else 0.0065), mats["slot"],
                  (o["x"] + dx, YW - o["t"] - 0.0036, o["z"] + dz + 0.003), (0, 0, 0, 1), bevel=0.0003)
        S.box(f"/Room/ground_{k}", (0.004, 0.002, 0.004), mats["slot"], (o["x"], YW - o["t"] - 0.0036, o["z"] + dz - 0.007),
              (0, 0, 0, 1), bevel=0.001)
    S.cylinder("/Room/plate_screw", 0.003, 0.002, mats["steel"], (o["x"], YW - o["t"] - 0.001, o["z"]),
               Rotation.from_euler("x", 90, degrees=True).as_quat(), n=12)
    # cast-iron radiator: 10 sections of 4 columns (2.5 in pitch), hubs, feet, supply valve and pipe
    R = L.RAD
    ry0, ry1 = YW - R["gap"] - R["depth"], YW - R["gap"]
    pitch = R["len"]/R["sections"]
    for k in range(R["sections"]):
        x = wx - R["len"]/2 + (k + 0.5)*pitch
        for j in range(4):
            y = ry0 + (j + 0.5)*R["depth"]/4
            S.tube(f"/Room/rad_col_{k}_{j}", [(x, y, 0.07), (x, y, R["height"] - 0.03)], pitch*0.30, mats["radiator"], n=10)
        S.box(f"/Room/rad_cap_{k}", (pitch*0.62, R["depth"]*0.96, 0.03), mats["radiator"], (x, (ry0 + ry1)/2, R["height"] - 0.02),
              (0, 0, 0, 1), bevel=0.008)
        S.box(f"/Room/rad_base_{k}", (pitch*0.62, R["depth"]*0.96, 0.03), mats["radiator"], (x, (ry0 + ry1)/2, 0.075), (0, 0, 0, 1),
              bevel=0.008)
    for k, z in enumerate((0.11, R["height"] - 0.05)):
        S.tube(f"/Room/rad_hub_{k}", [(wx - R["len"]/2, (ry0 + ry1)/2, z), (wx + R["len"]/2, (ry0 + ry1)/2, z)], 0.02, mats["radiator"])
    for k, x in enumerate((wx - R["len"]/2 + 0.02, wx + R["len"]/2 - 0.02)):
        S.box(f"/Room/rad_foot_{k}", (0.036, R["depth"], 0.06), mats["radiator"], (x, (ry0 + ry1)/2, 0.03), (0, 0, 0, 1), bevel=0.006)
    pc = L.PIPE
    S.tube("/Room/pipe", [(pc["x"], pc["y"], 0.0), (pc["x"], pc["y"], pc["z1"] - 0.02), (pc["x"] + 0.03, pc["y"], pc["z1"])], pc["r"],
           mats["pipe"])
    S.cylinder("/Room/valve", 0.022, 0.05, mats["pipe"], (pc["x"], pc["y"], pc["z1"] + 0.03))
    # ceiling fixture light, daylight
    X2 = X/2
    for i, (x, y) in enumerate(((0.0, 0.0),)):
        lt = UsdLux.DiskLight.Define(S.stage, f"/Room/ceiling_light_{i}")
        lt.CreateRadiusAttr(0.16); lt.CreateIntensityAttr(30000.); lt.CreateColorAttr(Gf.Vec3f(1., .95, .88))
        xf = S.UsdGeom.Xformable(lt.GetPrim()); xf.AddTranslateOp().Set(Gf.Vec3d(x, y, Z - 0.09))
        xf.AddOrientOp().Set(Gf.Quatf(1., Gf.Vec3f(0, 0, 0)))
    dome = UsdLux.DomeLight.Define(S.stage, "/Room/sky")
    dome.CreateTextureFileAttr(S.Sdf.AssetPath(str(Path(hdri(EXTERIOR_HDRI)).resolve())))
    dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(1500.)
    sun = UsdLux.DistantLight.Define(S.stage, "/Room/sun")
    sun.CreateIntensityAttr(900.); sun.CreateAngleAttr(0.53); sun.CreateColorAttr(Gf.Vec3f(1., .95, .86))
    xf = S.UsdGeom.Xformable(sun.GetPrim())
    xf.AddOrientOp().Set(Gf.Quatf(*_rot_to_wxyz(Rotation.from_euler("xyz", (50., 0., 170.), degrees=True))))
    # painter's LED work light on a tripod in the room's west corner, aimed obliquely along the work wall (raking light)
    lt = UsdLux.RectLight.Define(S.stage, "/Room/work_light")
    lt.CreateWidthAttr(0.22); lt.CreateHeightAttr(0.16); lt.CreateIntensityAttr(15000.); lt.CreateColorAttr(Gf.Vec3f(1., .98, .95))
    xf = S.UsdGeom.Xformable(lt.GetPrim())
    # (in the west corner, outside every task's robot footprint: it is render-only and has no collider)
    LP = np.array([-1.60, 0.25, 1.60])
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, LP)))
    look = np.array([0.4, 1.675, 1.1]) - LP; look /= np.linalg.norm(look)
    # RectLight emits along its local -z
    zax = -look; xax = np.cross((0, 0, 1), zax); xax /= np.linalg.norm(xax); yax = np.cross(zax, xax)
    xf.AddOrientOp().Set(Gf.Quatf(*_rot_to_wxyz(Rotation.from_matrix(np.stack([xax, yax, zax], 1)))))
    # a job-site LED light: yellow housing, black bezel, the lit lens, on a tripod stand (pole, collar, three legs)
    qh = Rotation.from_matrix(np.stack([xax, yax, zax], 1))
    hc = LP - 0.02*look
    S.box("/Room/work_light_head", (0.26, 0.20, 0.07), S.color("worklight_yellow", (.85, .62, .05), .45, 0.), hc + qh.apply((0, 0, 0.035)),
          qh.as_quat(), bevel=0.01)
    S.box("/Room/work_light_bezel", (0.24, 0.18, 0.012), mats["steel_dark"], hc - qh.apply((0, 0, 0.003)), qh.as_quat(), bevel=0.004)
    S.box("/Room/work_light_lens", (0.21, 0.15, 0.004), S.color("worklight_lens", (1., 1., .97), .2, 0., emissive=(4., 3.9, 3.7)),
          hc - qh.apply((0, 0, 0.010)), qh.as_quat(), bevel=0.002)
    S.tube("/Room/work_light_yoke", [hc + qh.apply((-0.14, 0, 0.03)), hc + np.array([0, 0, -0.10]), hc + qh.apply((0.14, 0, 0.03))],
           0.006, mats["steel_dark"], n=10)
    base = np.array([hc[0], hc[1], 0.62])
    S.tube("/Room/tripod_pole", [base, hc + np.array([0, 0, -0.10])], 0.012, mats["steel_dark"], n=12)
    S.cylinder("/Room/tripod_collar", 0.02, 0.05, mats["steel_dark"], base + np.array([0, 0, 0.02]), n=16)
    for k, a in enumerate((0.3, 2.4, 4.5)):
        foot = np.array([hc[0] + 0.30*math.cos(a), hc[1] + 0.30*math.sin(a), 0.0])
        foot[0] = max(foot[0], -L.X/2 + 0.03)
        S.tube(f"/Room/tripod_{k}", [base, foot], 0.009, mats["steel_dark"], n=10)
        S.cylinder(f"/Room/tripod_foot_{k}", 0.014, 0.012, S.color("rubber_foot", (.05, .05, .05), .8, 0.), foot + np.array([0, 0, 0.006]), n=12)
    # blue painter's tape marks for the strip / bands (on the wall, render-only marks)
    if spec["preset"] == "wall":
        tape = []
        if spec["task"] == "roll-strip":
            tape = [(L.STRIP["x0"], 0.30), (L.STRIP["x1"], 0.30), (L.STRIP["x0"], 2.20), (L.STRIP["x1"], 2.20)]
        elif spec["task"] == "cut-in-ceiling":
            tape = [(L.CEILING_BAND["x0"], Z - 0.12), (L.CEILING_BAND["x1"], Z - 0.12)]
        for k, (x, z) in enumerate(tape):
            S.box(f"/Room/tape_{k}", (0.006, 0.0006, 0.08), mats["tape_blue"], (x, YW - 0.0012, z), (0, 0, 0, 1), bevel=0.0002)
        # touch-up: a blue tape arrow beside each scuff (two short strips forming a chevron pointing at it)
        for k, (sx, sz) in enumerate(spec.get("scuffs", [])):
            for j, a in enumerate((35.0, -35.0)):
                q = Rotation.from_euler("y", a, degrees=True).as_quat()
                S.box(f"/Room/arrow_{k}_{j}", (0.045, 0.0006, 0.012), mats["tape_blue"],
                      (sx + 0.075, YW - 0.0012, sz + (0.012 if j == 0 else -0.012)), q, bevel=0.0002)


def _tools(S, world, mats, env, meta):
    sc, paint = world.scene, env.paint
    for name, it in sc.items.items():
        kind, body = it["kind"], it["body"]
        root = f"/Lab/body_{body}/{name}"
        parts = sc.parts[name]
        if kind == "roller":
            R = P.ROLLER
            S.tube(f"{root}/wire", [(0, 0, 0), (0, 0, -R["wire_len"])], R["wire_r"], mats["steel"])
            S.tube(f"{root}/axle", [(0, 0, 0), (0, 0.006, 0)], R["wire_r"], mats["steel"])
            gz = -R["wire_len"] - R["grip_len"]/2
            S.cylinder(f"{root}/grip", R["grip_r"], R["grip_len"], mats["grip"], (0, 0, gz), n=24)
            S.cylinder(f"{root}/grip_band", R["grip_r"]*1.02, 0.015, mats["grip_band"], (0, 0, -R["wire_len"] - 0.02), n=24)
            S.cylinder(f"{root}/collar", R["grip_r"]*0.7, 0.02, mats["steel"], (0, 0, -R["wire_len"] + 0.005), n=16)
        elif kind == "roller_cover":
            R = P.ROLLER
            r_out = R["core_od"]/2 + R["nap"]
            v, f, uv, nr = [], [], [], []
            na, nl = R["n_around"], 24
            for i in range(na + 1):
                a = 2*math.pi*i/na
                for j in range(nl + 1):
                    y = -R["cover_len"]/2 + R["cover_len"]*j/nl
                    v.append((r_out*math.cos(a), y, r_out*math.sin(a))); uv.append((i/na, j/nl))
                    nr.append((math.cos(a), 0, math.sin(a)))
            for i in range(na):
                for j in range(nl):
                    p0 = i*(nl + 1) + j
                    f += [p0, p0 + nl + 1, p0 + nl + 2, p0, p0 + nl + 2, p0 + 1]
            mat = _dynamic_mat(S, f"nap_{name}")
            S.mesh(f"{root}/nap", v, f, uv=uv, normals=nr, mat=mat)
            for k, y in enumerate((-R["cover_len"]/2 - 0.001, R["cover_len"]/2 + 0.001)):
                S.cylinder(f"{root}/end_{k}", R["core_od"]/2, 0.002, mats["steel"], (0, y, 0),
                           Rotation.from_euler("x", 90, degrees=True).as_quat(), n=24)
            meta["naps"][name.replace("_cover", "")] = f"nap_{name}"
            meta["textures"][f"nap_{name}"] = dict(w=R["n_around"], h=R["n_along"])
        elif kind in ("sash", "round8", "stencil_brush", "longreach"):
            B = P.BRUSHES[{"stencil_brush": "stencil"}.get(kind, kind)]
            fx, fy, fz = B["ferrule"]
            if kind == "sash":
                S.box(f"{root}/ferrule", (fx, fy, fz), mats["ferrule"], (0, 0, -fz/2), (0, 0, 0, 1), bevel=0.003)
                S.box(f"{root}/handle", (0.025, 0.019, B["handle_len"]), mats["alder"], (0, 0, -fz - B["handle_len"]/2), (0, 0, 0, 1),
                      bevel=0.008)
                S.box(f"{root}/plug", (B["width"]*0.98, B["thick"]*0.98, 0.006), mats["bristle"], (0, 0, 0.002), (0, 0, 0, 1), bevel=0.002)
            elif B.get("dogleg_deg"):
                h = T.dogleg_dir(kind)
                S.box(f"{root}/ferrule", (fx, fy, fz), mats["ferrule"], (0, 0, -fz/2), (0, 0, 0, 1), bevel=0.003)
                S.box(f"{root}/plug", (B["width"]*0.98, B["thick"]*0.98, 0.006), mats["bristle"], (0, 0, 0.002), (0, 0, 0, 1), bevel=0.002)
                a0 = np.array([0., 0., -fz])
                S.tube(f"{root}/neck", [a0 + 0.004*np.array([0, 0, 1.]), a0 + 0.03*h], B["handle_r"]*0.8, mats["alder"], n=14)
                S.tube(f"{root}/handle", [a0 + 0.03*h, a0 + B["handle_len"]*h], B["handle_r"], mats["alder"], n=16)
                S.cylinder(f"{root}/end_cap", B["handle_r"]*1.05, 0.004, mats["steel_dark"], a0 + B["handle_len"]*h,
                           Rotation.align_vectors([h], [[0, 0, 1]])[0].as_quat(), n=16)
            else:
                S.cylinder(f"{root}/ferrule", fx/2, fz, mats["ferrule"], (0, 0, -fz/2), n=20)
                S.tube(f"{root}/handle", [(0, 0, -fz), (0, 0, -fz - B["handle_len"])], B["handle_r"],
                       mats["alder"] if kind != "round8" else S.color("lacquer_black", (.03, .03, .035), .25, 0.))
            app = paint.applicators[name]
            path = f"{root}/tufts"
            curves = S.UsdGeom.BasisCurves.Define(S.stage, path)
            curves.CreateTypeAttr("linear")
            n = app.n
            curves.CreateCurveVertexCountsAttr([3]*n)
            pts = np.concatenate([np.stack([app.clamp[i], app.pivot[i], app.pivot[i] + app.ell[i]*app.d[i]]) for i in range(n)])
            curves.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            curves.CreateWidthsAttr(S.Vt.FloatArray.FromNumpy(np.full(3*n, app.b*1.15, np.float32)))
            curves.SetWidthsInterpolation("vertex")
            S.UsdGeom.PrimvarsAPI(curves).CreatePrimvar("displayColor", S.Sdf.ValueTypeNames.Color3fArray, "uniform").Set(
                S.Vt.Vec3fArray.FromNumpy(np.tile(np.array(COLORS["bristle"][0], np.float32), (n, 1))))
            # one material per brush whose colour the worker sets each frame to the tufts' load-weighted wet colour:
            # the BasisCurves' displayColor primvar is not read by the RTX material, so per-tuft colours never reach
            # the image
            S.bind(curves, S.color(f"bristle_vc_{name}", COLORS["bristle"][0], .6, 0.))
            meta["curves"][name] = path
            meta.setdefault("curve_mats", {})[name] = f"/Looks/bristle_vc_{name}/Shader"
        elif kind == "aerosol":
            from . import appearance as AP
            A = P.AEROSOL
            bh = A["height"] - 0.026
            pc = env.spec.get("paint", {}).get("color", "gloss_red")
            clear = pc == "clear"
            body = S.color("aerosol_body_" + pc, (.86, .86, .84) if clear else tuple(srgb_to_linear(P.COLORS[pc])), .3, .4)
            S.cylinder(f"{root}/can", A["r"], bh, body, (0, 0, bh/2), n=40)
            lab = AP.aerosol_label(S.folder/"textures", pc, clear=clear)
            AP.wrapped(S, f"{root}/label", A["r"]*1.002, bh*0.55, bh*0.18, S.textured("aerosol_label_" + pc, lab, rough=.35),
                       n=48)
            S.cylinder(f"{root}/shoulder", A["r"]*0.72, 0.012, mats["aerosol_metal"], (0, 0, bh + 0.006), n=24)
            S.cylinder(f"{root}/cup", 0.0127, 0.008, mats["aerosol_metal"], (0, 0, bh + 0.016), n=16)
            S.cylinder(f"{root}/base_ring", A["r"]*0.97, 0.004, mats["aerosol_metal"], (0, 0, 0.002), n=40)
        elif kind == "aerosol_button":
            A = P.AEROSOL
            S.cylinder(f"{root}/button", A["button_r"], A["button_h"], mats["button"], (0, 0, 0), n=20)
            S.cylinder(f"{root}/orifice", 0.0012, 0.002, mats["slot"], (A["button_r"], 0, 0.002),
                       Rotation.from_euler("y", 90, degrees=True).as_quat(), n=10)
        elif kind == "tray":
            for p in parts:
                if p.collide:
                    S.box(f"{root}/{p.name}", 2*np.asarray(p.size), mats["tray"], p.pos, p.quat, bevel=0.0012)
            # moulded detail of a deep-well tray (appearance only): a 12 mm flange round the top edge (its feet are parts)
            g = P.TRAY["size"]
            Lx, Ly, H = g[0], g[1], g[2]
            for k, (cx, cy, sx, sy) in enumerate(((0, Ly/2 + 0.006, Lx + 0.024, 0.012), (0, -Ly/2 - 0.006, Lx + 0.024, 0.012),
                                                  (Lx/2 + 0.006, 0, 0.012, Ly), (-Lx/2 - 0.006, 0, 0.012, Ly))):
                S.box(f"{root}/flange_{k}", (sx, sy, 0.003), mats["tray"], (cx, cy, H - 0.0015), (0, 0, 0, 1), bevel=0.0008)
        elif kind == "pail":
            from . import appearance as AP
            G = P.PAIL
            _open_cylinder(S, f"{root}/shell", G["r_bottom"], G["r_top"], G["height"], mats["pail"])
            S.cylinder(f"{root}/bottom", G["r_bottom"], 0.002, mats["pail"], (0, 0, 0.001), n=32)
            AP.wrapped(S, f"{root}/decal", (G["r_bottom"] + G["r_top"])/2 + 0.0012, G["height"]*0.9, 0.006,
                       S.textured("pail_decal", AP.pail_decal(S.folder/"textures"), rough=.5), n=48)
            S.cylinder(f"{root}/rim", G["r_top"] + 0.002, 0.006, mats["pail"], (0, 0, G["height"] - 0.003), n=40, caps=False)
            S.tube(f"{root}/strap", [(-G["r_top"], 0, G["height"]*0.85), (0, 0, G["height"] + 0.06), (G["r_top"], 0, G["height"]*0.85)],
                   0.004, mats["steel_dark"], n=8)
        elif kind == "can":
            from . import appearance as AP
            G = P.CAN
            pc = env.spec.get("paint", {}).get("color", "extra_white")
            _open_cylinder(S, f"{root}/shell", G["r"], G["r"], G["height"], mats["can_steel"])
            AP.wrapped(S, f"{root}/label", G["r"]*1.003, G["height"]*0.70, G["height"]*0.12,
                       S.textured("can_label_" + pc, AP.can_label(S.folder/"textures", pc), rough=.55), n=64)
            for k, z in enumerate((0.004, G["height"] - 0.006)):              # rolled seams
                S.cylinder(f"{root}/seam_{k}", G["r"] + 0.0015, 0.005, mats["can_steel"], (0, 0, z), n=48, caps=False)
            S.cylinder(f"{root}/bottom", G["r"], 0.003, mats["can_steel"], (0, 0, 0.0015), n=32)
            # paint dried over the rim groove: a painter's can that has been opened and poured
            S.cylinder(f"{root}/rim_paint", G["r"]*0.93, 0.004, S.color("can_rim_" + pc, tuple(srgb_to_linear(P.COLORS.get(pc, (238, 239, 234)))), .4, 0.),
                       (0, 0, G["height"] - 0.002), n=48, caps=False)
            S.tube(f"{root}/bail", [(-G["r"], 0, G["height"]*0.8), (0, 0.0, G["height"] + 0.09), (G["r"], 0, G["height"]*0.8)], 0.0018,
                   mats["steel"], n=8)
            S.cylinder(f"{root}/bail_grip", 0.009, 0.08, S.color("bail_grip", (.12, .12, .13), .6, 0.), (0, 0, G["height"] + 0.09),
                       Rotation.from_euler("y", 90, degrees=True).as_quat(), n=12)
        elif kind == "palette":
            from . import appearance as AP
            W_ = T.PALETTE_WELL
            AP.plate_with_wells(S, root, mats["palette"], T.PALETTE_WELLS, W_["r"], (0.25, 0.17), W_["top"] - W_["depth"], W_["top"])
        elif kind == "table":
            for p in parts:
                if p.kind == "box":
                    S.box(f"{root}/{p.name}", 2*np.asarray(p.size), mats["table_top"], p.pos, p.quat, bevel=0.006)
                else:
                    S.cylinder(f"{root}/{p.name}", p.size[0], 2*(p.size[1] + p.size[0]), mats["steel_dark"], p.pos, p.quat, n=12)
        elif kind == "board":
            S.box(f"{root}/board", (0.60, 0.45, 0.012), mats["mdf"], (0, 0, 0.006), (0, 0, 0, 1), bevel=0.002)
            for k, (x, y) in enumerate(((-0.19, 0.0), (0.19, 0.0))):
                S.box(f"{root}/tape_{k}", (0.025, 0.30, 0.0004), S.color("masking_tape", (.86, .80, .62), .7, 0.),
                      (x, y, 0.0128), (0, 0, 0, 1), bevel=0.0001)
        elif kind == "spray_box":
            for p in parts:
                S.box(f"{root}/{p.name}", 2*np.asarray(p.size), mats["cardboard"], p.pos, p.quat, bevel=0.001)
        elif kind == "crate":
            S.box(f"{root}/panel", (0.45, 0.30, 0.012), mats["plywood"], (0, 0, 0.006), (0, 0, 0, 1), bevel=0.001, uv_per_m=1/0.6)
        elif kind == "stencil":
            pass                         # drawn by its film chart (the sheet's top face, apertures cut out)


def _open_cylinder(S, path, r0, r1, h, mat, z0=0.0, n=40):
    v, f, nr = [], [], []
    for i in range(n + 1):
        a = 2*math.pi*i/n
        for z, r in ((z0, r0), (z0 + h, r1)):
            v.append((r*math.cos(a), r*math.sin(a), z)); nr.append((math.cos(a), math.sin(a), 0))
    for i in range(n):
        a, b_, c, d = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
        f += [a, c, d, a, d, b_]
    S.mesh(path, v, f, normals=nr, mat=mat)
    # inside face
    v2 = [(x*0.985, y*0.985, z) for (x, y, z) in v]
    f2 = [f[k] for k in range(len(f))]
    f2 = [x for t in zip(f2[0::3], f2[2::3], f2[1::3]) for x in t]
    S.mesh(path + "_in", v2, f2, normals=[(-a, -b_, 0) for (a, b_, _) in nr], mat=mat)


def _dynamic_mat(S, name, rough_name=None, normal_name=None):
    """OmniPBR material whose maps are dynamic textures pushed by the worker (dynamic://<name>)."""
    T_, Sdf, UsdShade = S.Sdf.ValueTypeNames, S.Sdf, S.UsdShade
    m = UsdShade.Material.Define(S.stage, "/Looks/dyn_" + name)
    sh = UsdShade.Shader.Define(S.stage, f"/Looks/dyn_{name}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_texture", T_.Asset).Set(Sdf.AssetPath(f"dynamic://{name}"))
    sh.CreateInput("reflection_roughness_constant", T_.Float).Set(0.6)
    if rough_name:
        sh.CreateInput("reflectionroughness_texture", T_.Asset).Set(Sdf.AssetPath(f"dynamic://{rough_name}"))
        sh.CreateInput("reflection_roughness_texture_influence", T_.Float).Set(1.0)
    if normal_name:
        sh.CreateInput("normalmap_texture", T_.Asset).Set(Sdf.AssetPath(f"dynamic://{normal_name}"))
        sh.CreateInput("bump_factor", T_.Float).Set(1.0)
    m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return m


def _chart_mesh(ch, offset):
    """Quads over the chart's paintable cells (run-length merged along v per column band), UVs over the whole chart."""
    mask = ch.mask
    v, f, uv = [], [], []
    du, dv = ch.dx, ch.dx
    step = max(1, ch.nu//400)
    for i0 in range(0, ch.nu, step):
        i1 = min(ch.nu, i0 + step)
        col = mask[i0:i1].all(axis=0)
        j = 0
        while j < ch.nv:
            if not col[j]:
                j += 1
                continue
            j0 = j
            while j < ch.nv and col[j]:
                j += 1
            k = len(v)
            for (a, b_) in ((i0, j0), (i1, j0), (i1, j), (i0, j)):
                p = ch.origin + ch.u*a*du + ch.v*b_*dv + ch.n*offset
                v.append(p); uv.append((a/ch.nu, b_/ch.nv))
            f += [k, k + 1, k + 2, k, k + 2, k + 3]
        # columns partly masked (a mask edge inside the band): fall back to per-column runs
        part = mask[i0:i1].any(axis=0) & ~mask[i0:i1].all(axis=0)
        if np.any(part):
            for i in range(i0, i1):
                colm = mask[i] & part
                j = 0
                while j < ch.nv:
                    if not colm[j]:
                        j += 1
                        continue
                    j0 = j
                    while j < ch.nv and colm[j]:
                        j += 1
                    k = len(v)
                    for (a, b_) in ((i, j0), (i + 1, j0), (i + 1, j), (i, j)):
                        p = ch.origin + ch.u*a*du + ch.v*b_*dv + ch.n*offset
                        v.append(p); uv.append((a/ch.nu, b_/ch.nv))
                    f += [k, k + 1, k + 2, k, k + 2, k + 3]
    return np.array(v), np.array(f), np.array(uv)


def _charts(S, paint, meta):
    for name, ch in paint.charts.items():
        if not ch.visible:
            continue
        off = {"wall": 0.0, "ceiling": -0.0004}.get(name, 0.0004)
        v, f, uv = _chart_mesh(ch, off)
        if not len(f):
            continue
        mat = _dynamic_mat(S, f"paint_{name}", f"rough_{name}", f"normal_{name}")
        parent = f"/Lab/body_{ch.body}" if ch.body >= 0 else "/Room"
        S.mesh(f"{parent}/chart_{name}", v, f, uv=uv, normals=[tuple(ch.n)]*len(v), mat=mat)
        meta["charts"][name] = dict(diffuse=f"paint_{name}", rough=f"rough_{name}", normal=f"normal_{name}", w=ch.nu, h=ch.nv)
    # pools: one mesh per reservoir, rewritten per request (level and outline)
    for name, r in paint.reservoirs.items():
        path = f"/Lab/body_{r.body}/pool_{name}"
        g = S.UsdGeom.Mesh.Define(S.stage, path)
        g.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(np.zeros((32, 3), np.float32)))
        g.CreateFaceVertexCountsAttr([3]*31)                       # a closed fan: centre + 31 rim points
        g.CreateFaceVertexIndicesAttr([x for k in range(31) for x in (0, k + 1, (k + 1) % 31 + 1)])
        g.CreateSubdivisionSchemeAttr("none")
        col = tuple(float(x) for x in (r.paints[int(np.argmax(r.solids))] if r.solids.sum() > 0 else r.paints[0]).rinf)
        S.bind(g, S.color(f"pool_{name}", col, .08, 0.))
        meta["pools"][name] = path
    # airborne spray and drops: point clouds (positions per request)
    for key in ("spray", "drops", "sinks"):
        pts = S.UsdGeom.Points.Define(S.stage, f"/Air/{key}")
        pts.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(np.zeros((1, 3), np.float32) - 50))
        pts.CreateWidthsAttr(S.Vt.FloatArray.FromNumpy(np.full(1, 0.001, np.float32)))
        col = tuple(float(x) for x in paint.paints[0].rinf)
        m = S.color(f"air_{key}", col, .3, 0.)
        if key == "spray":
            # each airborne parcel stands for ~2e4 droplets spread over the jet's local width: drawn as a 7 mm translucent
            # puff, so the plume reads as a haze (2.8 mm puffs are invisible)
            sh = S.UsdShade.Shader(S.stage.GetPrimAtPath(f"/Looks/air_{key}/Shader"))
            sh.CreateInput("opacity", S.Sdf.ValueTypeNames.Float).Set(0.30)
        S.bind(pts, m)
        meta["points"][key] = f"/Air/{key}"


def pool_polygon(res, bq):
    """Fan of the pool's free surface in the container frame (convex hull of the cavity cross-section at the level)."""
    lvl, up, _ = res.level(bq)
    if res.volume <= 0:
        return np.zeros((32, 3))
    hgt = res.points @ up
    # cavity points just below the level only, the rim inset 1 mm: a band +-4 mm about the level with the rim 2 mm out
    # reached through the tray's sloped walls (navy triangles outside it)
    band = res.points[(hgt <= lvl) & (hgt > lvl - 0.004)]
    if len(band) < 3:
        band = res.points[hgt < lvl]
    if len(band) < 3:
        return np.zeros((32, 3))
    # project onto the level plane; the rim is the convex hull of the cavity points there, read along 31 rays from the
    # centroid and set 1 mm inside: its chords join real wall points, so it stays inside a convex cavity (the farthest
    # point of each angular sector placed at the sector's middle overshoots the walls at the corners)
    from scipy.spatial import ConvexHull
    c = band.mean(axis=0)
    e1 = np.cross(up, [1., 0., 0.] if abs(up[0]) < 0.9 else [0., 1., 0.]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    rel = band - c
    xy = np.stack([rel @ e1, rel @ e2], 1)
    base = c + up*(lvl - c @ up)
    try:
        hv = xy[ConvexHull(xy).vertices]
    except Exception:
        return np.zeros((32, 3))
    out = [base]
    for k in range(31):
        a = -math.pi + 2*math.pi*(k + 0.5)/31
        d = np.array([math.cos(a), math.sin(a)])
        best = 0.0
        for p0, p1 in zip(hv, np.roll(hv, -1, axis=0)):
            M = np.array([d, p0 - p1]).T
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            t, u = np.linalg.solve(M, p0)
            if t > 0 and -1e-9 <= u <= 1 + 1e-9:
                best = max(best, t)
        r = max(best - 0.001, 0.0)
        out.append(base + r*(d[0]*e1 + d[1]*e2))
    return np.array(out)                                         # (32, 3): centre + 31 rim points


def dynamic_payload(env, images, full=False):
    """Per-request dynamic state: changed chart textures, nap textures, tuft shapes, pool fans, airborne points."""
    paint = env.paint
    bq = env.world.state.body_q.numpy()
    clear = (env.spec.get("paint") or {}).get("color") == "clear"
    out = dict(textures={}, curves={}, pools={}, points={})
    for name, ch in paint.charts.items():
        if not ch.visible:
            continue
        img = images.get(name)
        if img is None:
            img = images[name] = ChartImage(ch)
            changed = True
        else:
            changed = img.refresh()
        if changed or full:
            out["textures"][f"paint_{name}"] = img.payload("diffuse")
            out["textures"][f"rough_{name}"] = img.payload("rough")
            out["textures"][f"normal_{name}"] = img.payload("normal")
    for name, a in paint.applicators.items():
        if a.kind == "roller":
            key = ("nap", name)
            ver = float(a.s.sum() + a.v.sum())
            if full or images.get(key) != ver:
                out["textures"][f"nap_{name}_cover"] = nap_image(a)
                images[key] = ver
        elif a.kind == "brush":
            X, Pw, R = a.tips(bq)
            p = np.asarray(bq[a.body][:3], float)
            loc_tip = (X - p) @ R
            pts = np.concatenate([np.stack([a.clamp[i], a.pivot[i], loc_tip[i]]) for i in range(a.n)])
            load = wetted((a.s.sum(axis=0) + a.v)/np.maximum(a.C_half, 1e-12))
            base = np.array(COLORS["bristle"][0]); pc = paint.paints[0].rinf
            cols = base[None]*(1 - 0.9*load[:, None]) + pc[None]*0.9*load[:, None]
            # the material colour: the mean over the tufts, weighted toward the loaded ones (what the eye reads)
            wgt = 0.2 + load
            mean = (cols*wgt[:, None]).sum(axis=0)/wgt.sum()
            out["curves"][name] = dict(points=np.round(pts, 6).tolist(), colors=np.round(cols, 4).tolist(),
                                       material=np.round(mean, 4).tolist())
    for name, r in paint.reservoirs.items():
        out["pools"][name] = np.round(pool_polygon(r, bq), 6).tolist()
    spray = [a for a in paint.applicators.values() if a.kind == "aerosol"]
    if spray:
        a = spray[0]
        pts = a.pos[:16000] if len(a.pos) and not clear else np.zeros((1, 3)) - 50
        out["points"]["spray"] = dict(points=np.round(pts, 5).tolist(), width=0.001)
    d = paint.drips
    out["points"]["drops"] = dict(points=(np.round(d.pos, 5).tolist() if len(d.pos) else [[0, 0, -50]]), width=0.004)
    # paint on surfaces without a film chart (radiator, pipe, glazing): its 4 mm voxels holding >= 1 um, drawn as flat
    # paint-coloured dots at their centres
    # a clear lacquer is drawn as gloss only: its airborne and sink points are not white dots
    vox = [] if clear else [k for n, s_ in paint.sinks.items() if n not in ("air",) for k, v in s_.get("vox", {}).items()
                            if v >= 1e-6*paint.SINK_VOXEL**2]
    # 6 mm dots at the 4 mm voxel centres, so a dot whose centre lies inside a column still shows on its surface
    out["points"]["sinks"] = dict(points=(np.round((np.array(vox, float) + 0.5)*paint.SINK_VOXEL, 5).tolist() if vox
                                          else [[0, 0, -50]]), width=1.5*paint.SINK_VOXEL)
    return out


def _dressing(S, mats, spec):
    """Render-only dressing outside the robot's working volume: furniture moved off the work wall, a plastic sheet over
    the bed, the ladder, a picture taken down, a plant, spare cans; the can lid and stir stick on the cloth."""
    from .visual_assets import model
    X, Y, YW = L.X, L.Y, L.YW

    def ref(name, path, p, yaw=0.0, scale=1.0):
        """Reference a pinned Poly Haven model; its MaterialX graphs are rebound to OmniPBR with the same texture
        files (MaterialX normal-map nodes crash the render-only worker, as in label_application and percussion)."""
        import re
        from pxr import Usd
        usd = model(name)
        outer = S.UsdGeom.Xform.Define(S.stage, path)
        outer.AddTranslateOp().Set(S.Gf.Vec3d(*map(float, p)))
        if yaw:
            outer.AddRotateZOp().Set(float(yaw))
        if scale != 1.0:
            outer.AddScaleOp().Set(S.Gf.Vec3f(scale, scale, scale))
        inner = S.UsdGeom.Xform.Define(S.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), f"/{name}")
        for pr in Usd.PrimRange(inner.GetPrim()):
            if pr.GetTypeName() == "BasisCurves":
                pr.SetActive(False)
        UsdShade, Tn, Sdf = S.UsdShade, S.Sdf.ValueTypeNames, S.Sdf
        made = {}
        for pr in Usd.PrimRange(S.stage.GetPrimAtPath(path)):
            if not pr.IsA(S.UsdGeom.Gprim):
                continue
            src, _ = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial()
            if not src:
                continue
            key = src.GetPrim().GetName()
            if key not in made:
                files = {}
                for sh in Usd.PrimRange(src.GetPrim()):
                    if sh.IsA(UsdShade.Shader) and UsdShade.Shader(sh).GetIdAttr().Get() == "UsdUVTexture":
                        f = UsdShade.Shader(sh).GetInput("file").Get()
                        if f:
                            files[Path(f.path).name] = usd.parent/"textures"/Path(f.path).name
                pick = lambda tag: next((v for k, v in files.items() if re.search(tag, k)), None)
                m = UsdShade.Material.Define(S.stage, f"/Looks/pbr_{name}_{key}")
                sh = UsdShade.Shader.Define(S.stage, f"/Looks/pbr_{name}_{key}/Shader")
                sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
                sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
                sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
                for tag, inp in (("_diff", "diffuse_texture"), ("_rough", "reflectionroughness_texture"), ("_nor_gl", "normalmap_texture")):
                    f = pick(tag)
                    if f is not None and f.exists() and f.suffix.lower() in (".jpg", ".png"):
                        sh.CreateInput(inp, Tn.Asset).Set(Sdf.AssetPath(str(f.resolve())))
                        if inp == "reflectionroughness_texture":
                            sh.CreateInput("reflection_roughness_texture_influence", Tn.Float).Set(1.0)
                m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
                made[key] = m
            UsdShade.MaterialBindingAPI.Apply(pr).Bind(made[key])
        for sp in [pr.GetPath() for pr in Usd.PrimRange(S.stage.GetPrimAtPath(path)) if pr.GetName() == "_materials"]:
            q = S.stage.GetPrimAtPath(sp)
            if q:
                q.SetActive(False)
        return outer

    ref("old_bed_frame", "/Dress/bed", (0.55, -1.0, 0.0), yaw=90.0)
    ref("vintage_wooden_drawer_01", "/Dress/dresser", (X/2 - 0.30, -0.6, 0.0), yaw=-90.0)
    ref("wooden_ladder", "/Dress/ladder", (-X/2 + 0.25, -0.9, 0.0), yaw=90.0)
    ref("modern_ceiling_lamp_01", "/Dress/ceiling_lamp", (0.0, 0.0, L.Z - 0.0), yaw=0.0)
    # the picture taken off the work wall stands on the floor leaning on the south wall (its model is centred: lift it
    # by its half height)
    pic = ref("hanging_picture_frame_01", "/Dress/picture", (-0.9, -Y/2 + 0.075, 0.415), yaw=180.0)   # face to the room
    pic.AddRotateXOp().Set(-8.0)                        # top toward the wall
    ref("potted_plant_02", "/Dress/plant", (X/2 - 0.30, -Y/2 + 0.35, 0.0))
    ref("spray_paint_bottles", "/Dress/spray_cans", (X/2 - 0.35, 0.6, 0.0), yaw=30.0)
    # packed boxes against the south wall, the painter's toolbox, a step stool and a jug of clean-up water
    # (footprints checked against the bed 0.9 x 2.0 m, the ladder, the dresser and the picture: no overlaps)
    ref("cardboard_box_01", "/Dress/box_0", (-1.05, -1.33, 0.0), yaw=90.0)
    ref("cardboard_box_01", "/Dress/box_1", (-1.06, -1.32, 0.335), yaw=84.0, scale=0.9)
    ref("metal_toolbox", "/Dress/toolbox", (X/2 - 0.28, 0.10, 0.0), yaw=-80.0)
    ref("folding_wooden_stool", "/Dress/stool", (-X/2 + 0.38, 0.05, 0.0), yaw=20.0)
    ref("plastic_bottle_gallon", "/Dress/water_jug", (X/2 - 0.21, 0.36, 0.0), yaw=40.0)
    if spec["preset"] == "wall":
        cl = L.CLOTH
        S.cylinder("/Dress/can_lid", P.CAN["r"]*1.02, 0.004, mats["can_steel"], (-1.25, 0.65, 0.003), n=32)
        S.box("/Dress/stir_stick", (0.30, 0.025, 0.004), mats["stick"], (-1.35, 0.78, 0.003), Rotation.from_euler("z", 20, degrees=True).as_quat())
        S.cylinder("/Dress/tape_roll", 0.045, 0.048, mats["tape_blue"], (0.85, 0.65, 0.024), n=28)


def _cameras(S, world, spec):
    from classroom_arrangement.render import cameras as classroom_cameras
    import types
    out = {}
    # policy cameras (head, wrists, base) exactly as the classroom robot
    sc = world.scene
    if not hasattr(sc, "g1_hw"):
        rig, inst = world.rig, sc.robot_instance
        fr = rig.frame_pose(f"{inst}/head_camera", world.state)
        carriage = sc.robot_body["franka_spine_mounting_point"]
        pb, Rb = world.body_pose(carriage)
        qw = fr["orientation_wxyz"]
        Rf = Rotation.from_quat((qw[1], qw[2], qw[3], qw[0]))
        pf = np.asarray(fr["position_m"])
        Rl = Rotation.from_matrix(Rb).inv()*Rf
        pl = Rotation.from_matrix(Rb).inv().apply(pf - pb)
        from classroom_arrangement import params as CP
        g = S.camera(f"/Lab/body_{carriage}/head_rgb", pl, pl + Rl.apply((1, 0, 0)), 18/math.tan(math.radians(CP.HEAD_HFOV/2)),
                     up=Rl.apply((0, 0, 1)), role="policy")
        S.alias("head_rgb", g.GetPath()); out["head_rgb"] = "policy"
    for name, m in sc.cameras.items():
        g = S.camera(f"/Lab/body_{m['body']}/{name}", m["eye"], m["target"], 18/math.tan(math.radians(m["fov"]/2)), up=m["up"],
                     role="policy")
        S.alias(name, g.GetPath()); out[name] = "policy"
    if hasattr(sc, "g1_hw"):
        # World-space view of the standing humanoid and craft table; the FR3 chase
        # offset placed a pelvis-mounted camera outside this room.
        g = S.camera("/Cameras/g1_chase", (-1.45, 0.25, 1.55), (-0.45, 1.10, 0.75), 14., role="diagnostic")
    else:
        base = sc.robot_body["base_link"]
        g = S.camera(f"/Lab/body_{base}/chase", (-1.6, 0.9, 1.9), (1.0, 0.0, 1.0), 14., role="diagnostic")
    S.alias("chase", g.GetPath()); out["chase"] = "diagnostic"
    YW, Z = L.YW, L.Z
    diag = dict(room=((1.55, -1.45, 1.75), (-0.45, 1.3, 1.0), 13.),
                wall=((-0.35, -0.75, 1.40), (-0.35, 1.675, 1.20), 13.),
                raking=((1.60, 0.55, 1.20), (-1.2, 1.675, 1.2), 16.),
                # oblique from the left: a camera behind the robot looks at the Spine column
                strip=((-1.50, 0.80, 1.50), (-0.45, 1.675, 1.20), 14.),
                tray_close=((-0.72, 0.85, 0.55), (-1.05, 1.25, 0.04), 18.),
                pail_close=((0.25, 0.95, 0.50), (-0.02, 1.30, 0.08), 20.),
                radiator_end=((1.25, 0.95, 1.05), (0.50, 1.63, 0.36), 16.),
                ceiling_line=((-0.70, 1.05, 2.05), (-0.70, 1.675, 2.42), 20.),
                floor_tools=((-0.6, 0.35, 1.05), (-0.9, 1.35, 0.05), 16.),
                table=((-0.25, 0.65, 1.60), (-0.25, 1.30, 0.74), 18.),
                paper=((-0.30, 1.05, 1.25), (-0.30, 1.30, 0.75), 22.),
                spray_box=((0.50, 0.80, 1.35), (-0.10, 1.33, 0.88), 16.),
                stencil=((-0.10, 1.12, 1.40), (-0.10, 1.34, 0.76), 22.),
                side=((1.40, 0.20, 1.40), (-0.40, 1.20, 0.90), 15.),
                # close views of the contact (the roller leaves the strip camera's frame on the upper halves; the
                # casing, outlet and scuffs are a few pixels in the wall view)
                strip_high=((-1.45, 0.75, 2.05), (-0.45, 1.675, 1.80), 14.),
                casing_close=((-0.40, 1.05, 1.45), (0.02, 1.675, 1.45), 13.),
                outlet_close=((-0.62, 1.20, 0.55), (-0.90, 1.675, 0.36), 18.),
                pail_side=((0.45, 1.05, 0.30), (-0.02, 1.30, 0.12), 20.),
                spray_front=((-0.10, 0.70, 1.20), (-0.10, 1.34, 0.80), 18.),
                paper_low=((0.20, 0.95, 0.92), (-0.30, 1.28, 0.75), 24.))
    for k, (sx, sz) in enumerate(spec.get("scuffs", [])):
        diag[f"scuff_{k + 1}"] = ((sx + 0.15, YW - 0.42, sz + 0.06), (sx + 0.01, YW, sz), 20.)
    pail = (spec.get("items") or {}).get("pail")
    if pail:
        px, py = float(pail["pose"][0]), float(pail["pose"][1])
        diag["pail_close"] = ((px + 0.27, 0.95, 0.50), (px, py, 0.08), 20.)
        diag["pail_side"] = ((px + 0.47, 1.05, 0.30), (px, py, 0.12), 20.)
    # down into the 64 mm gap behind the radiator from above its top: the wall band and the columns' backs (the
    # radiator's paint is not visible from any episode camera)
    wx_ = L.WIN["x"]
    diag["radiator_gap"] = ((wx_ - 0.05, YW - 0.30, 1.05), (wx_, YW - 0.03, 0.30), 18.)
    for name, (eye, target, f) in diag.items():
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); out[name] = "diagnostic"
    return out


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (painting/render_worker.py)."""

    def __init__(self, folder, scene, meta, *, width=640, height=360, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.meta = meta
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        self.images = {}
        self.first = True
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PNT_RENDER_PORT=str(listener.getsockname()[1]), PNT_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "painting.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            stdout=self.log, stderr=subprocess.STDOUT, env=worker_env(env), cwd=str(root))
            self.connection, _ = listener.accept()
            self.connection.settimeout(600)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed; see " + str(self.folder/"worker.log"))
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)

    def _call(self, message):
        import select
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            send(self.connection, message)
            # 30 min: on a GPU shared with other GPU simulations a worker's first frame (shader warm-up) took > 10 min
            deadline = time.monotonic() + 1800
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            reply = receive(self.connection)
        if not reply.get("ok") or reply.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(reply)[:400])
        return reply

    def observe(self, env, cameras, resolution=None, samples=None):
        """Render the named cameras from the environment's current state; returns {camera: PNG bytes}."""
        dyn = dynamic_payload(env, self.images, full=self.first)
        self.first = False
        msg = dict(op="observe", cameras=list(cameras), body_poses=np.round(env.world.state.body_q.numpy(), 7).tolist(),
                   tick=int(env.world.tick), dynamic=dyn)
        if resolution is not None:
            msg["resolution"] = list(resolution)
        if samples is not None:
            msg["spp"] = int(samples)
        reply = self._call(msg)
        self.last_timings = reply.get("timings", {})
        return {k: base64.b64decode(v) for k, v in reply["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

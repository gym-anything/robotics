"""USD export of the simulated scene and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses (and the scanner's light
state), so one exported stage serves a whole session. Appearance rules:
  * collision hulls are never drawn; robots use their visual meshes, fixtures their dimensioned boxes;
  * packaging is textured at 4 texels/mm; every printed symbol is its own patch mesh at 10 texels per module,
    posed exactly as in Scene.labels (flat, or curved to the can/bottle radius);
  * finishes: matte paper (OmniPBR), semi-gloss / glossy / film / metal as a clear coat over the print
    (OmniSurface), clear shrink film as a thin-walled transmissive shell;
  * the scanner's imager is a thin-lens camera at the entrance pupil inside an open-fronted head whose exit
    window (32 x 24 mm) matches the field of view 50 mm ahead of the pupil; two red LEDs beside the window;
    the aim line is an IES-profiled amber source tilted to cross the optical axis at 150 mm.
Depth of field: Isaac RTX reads a camera's focal length in tenths of a scene unit for its aperture, so the
f-stop that gives an entrance pupil A is 0.1 m x focalLength / A.
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import subprocess
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from .catalog import SKUS

RTX_FOCAL_UNIT_M = 0.1          # measured: RTX aperture = focalLength (tenths of a unit) / fStop
IMAGER_FOCAL_MM = 35.6          # virtual lens: 20.955 mm aperture -> 32.8 deg HFOV
IMAGER_APERTURE_MM = 2*IMAGER_FOCAL_MM*math.tan(math.radians(P.HFOV_DEG/2))
WINDOW = (0.032, 0.024)         # exit window (m): the FOV footprint 50 mm ahead of the pupil

COLORS = dict(
    floor=((.30, .31, .30), .75, 0.), robot=((.86, .87, .86), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.),
    camera_dark=((.03, .03, .035), .45, 0.), aluminum=((.62, .64, .66), .25, .9),
    cart_steel=((.30, .31, .33), .45, .5), cart_frame=((.16, .17, .18), .5, .3), tablet_black=((.02, .02, .025), .3, 0.),
    shelf_steel=((.42, .44, .46), .5, .4), scanner_body=((.055, .055, .06), .38, 0.), scanner_grip=((.03, .03, .032), .75, 0.),
    trigger_black=((.04, .04, .045), .35, 0.), cradle_black=((.05, .05, .055), .45, 0.), keypad_black=((.06, .06, .065), .5, 0.),
    keycap=((.80, .80, .79), .55, 0.), carton=((.62, .45, .28), .85, 0.), basket_blue=((.10, .32, .72), .45, 0.),
    tote_grey=((.36, .38, .40), .6, 0.), laminate=((.78, .76, .72), .35, 0.), wood_counter=((.55, .38, .22), .45, 0.),
    counter_panel=((.30, .30, .32), .55, 0.), bezel=((.02, .02, .022), .4, 0.), led_off=((.018, .026, .02), .35, 0.),
    lid_metal=((.72, .72, .70), .25, .9), cap_blue=((.10, .35, .75), .35, 0.), jar_glass=((.35, .05, .05), .1, 0.),
    pet_water=((.80, .88, .94), .06, 0.), bag_white=((.90, .90, .88), .55, 0.), caster_rubber=((.05, .05, .055), .8, 0.),
    lid_gold=((.72, .58, .30), .3, .9),
)
PRINT_WHITE = 0.85   # diffuse reflectance of unprinted label stock / coated board (estimate; artwork.INK)
FINISH = {  # clear coat over the print: coat roughness per finish, ESTIMATES. They are not anchored to the SE4710's
    # specular dead zone: dead-zone benches found none at any roughness from 0.07 to 0.35
    "matte": None, "semigloss": dict(coat=1., coat_rough=.25), "glossy": dict(coat=1., coat_rough=.07),
    "film": dict(coat=1., coat_rough=.03), "metal": dict(coat=1., coat_rough=.10, metal=.35),
    "glass": dict(coat=1., coat_rough=.05),
}
# Provenance of the render materials, a ledger like params.py's.
# COLORS are (linear base colour, roughness, metallic) for UsdPreviewSurface/OmniPBR; none is identified.
MATERIAL_NOTES = {
    "PRINT_WHITE": "estimate: diffuse reflectance of white label stock / coated board; the substrate of every printed symbol",
    "INK": "estimate (artwork.INK, sRGB 63 = 5 % of the substrate's white): carbon-black thermal-transfer and litho inks reflect 2-5 %",
    "FINISH": "estimates: clear-coat roughness per finish; not anchored (no specular dead zone at any roughness 0.07-0.35)",
    "jar_glass": "authored: the sauce seen through the jar's glass (opaque, glass coat); the headspace and shoulder are OmniGlass n 1.52 (render.jar_glass)",
    "pet_water": "authored: the PET bottle's label-free body is OmniGlass n 1.33 (render.water_pet); this colour is the fallback",
    "scanner_body": "authored appearance (dark grey moulded plastic, product photographs)",
    "default": "authored appearance estimate (scene dressing and fixtures; does not enter a decoded symbol's contrast)",
}


def quat_xyzw_from_matrix(R):
    return Rotation.from_matrix(np.asarray(R, float)).as_quat()


# camera->scanner frame: USD cameras look along -Z with +Y up; the scanner looks along +x with +z up
R_CAM = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def ies_line(path, fan_deg=P.AIM_FAN_DEG, thick_deg=P.AIM_THICK_DEG):
    half = fan_deg/2
    vert = [0.0] + [round(half*i/40, 4) for i in range(1, 41)] + [half + .05, 90.0, 180.0]
    t = thick_deg/2
    horiz = sorted({0.0, 90 - 3*t, 90 - t, 90.0, 90 + t, 90 + 3*t, 180.0, 270 - 3*t, 270 - t, 270.0, 270 + t, 270 + 3*t, 360.0})
    lines = ["IESNA:LM-63-2002", "[TEST] handheld imager aim line (authored)", "TILT=NONE",
             f"1 -1 1 {len(vert)} {len(horiz)} 1 2 0 0 0", "1.0 1.0 0.0",
             " ".join(f"{v:g}" for v in vert), " ".join(f"{h:g}" for h in horiz)]
    for h in horiz:
        on = abs(h - 90) <= t + 1e-9 or abs(h - 270) <= t + 1e-9
        lines.append(" ".join(("1" if (on and v <= half) else "0") for v in vert))
    Path(path).write_text("\n".join(lines) + "\n")


def ies_led(path, fwhm_h=P.LED_FWHM_H_DEG, fwhm_v=P.LED_FWHM_V_DEG):
    """IESNA LM-63 type-C profile of the illumination LEDs: an elliptical Gaussian with the published full widths at
    half maximum (params.LED_FWHM_*: the SE4710 pattern scaled to the field of view; C = 0/180 is the horizontal plane after the
    light's orientation, see add_imager). A hard cone at the half-intensity angle lit only the frame centre and left labels
    near the frame edge under a 5x illumination step."""
    import math
    vert = [float(v) for v in range(0, 91, 2)]
    horiz = [float(h) for h in range(0, 361, 10)]
    a, b = fwhm_h/2, fwhm_v/2
    lines = ["IESNA:LM-63-2002", "[TEST] handheld imager illumination LED (elliptical Gaussian)", "TILT=NONE",
             f"1 -1 1 {len(vert)} {len(horiz)} 1 2 0 0 0", "1.0 1.0 0.0",
             " ".join(f"{v:g}" for v in vert), " ".join(f"{h:g}" for h in horiz)]
    for h in horiz:
        c, s_ = math.cos(math.radians(h)), math.sin(math.radians(h))
        lines.append(" ".join(f"{math.exp(-math.log(2)*((v*c/a)**2 + (v*s_/b)**2)):.5f}" for v in vert))
    Path(path).write_text("\n".join(lines) + "\n")


class Stage:
    """Thin USD authoring helper for the export."""

    def __init__(self, folder):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        self.Gf, self.Sdf, self.Usd, self.UsdGeom, self.UsdLux, self.UsdShade, self.Vt = Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder/"textures").mkdir(exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(self.folder/"scene.usdc"))
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/Lab").GetPrim())
        self.mats = {}

    # ---------------------------------------------------------------- materials
    def color(self, name, rgb, rough=.5, metal=0., emissive=None, opacity=None):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        s = UsdShade.Shader.Define(self.stage, f"/Looks/{name}/Shader")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        if emissive is not None:
            s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, emissive)))
        if opacity is not None:
            s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        self.mats[name] = m
        return m

    def mdl(self, name, module, inputs):
        if name in self.mats:
            return self.mats[name]
        UsdShade = self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, f"/Looks/{name}/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(self.Sdf.AssetPath(module + ".mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier(module, "mdl")
        for k, (typ, v) in inputs.items():
            sh.CreateInput(k, typ).Set(v)
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        self.mats[name] = m
        return m

    def printed(self, name, texture, finish="matte"):
        """Print on a substrate with the item's finish: matte paper, or a clear coat of the given roughness."""
        T, Sdf = self.Sdf.ValueTypeNames, self.Sdf
        tex = Sdf.AssetPath(str(Path(texture).resolve()))
        f = FINISH.get(finish)
        if f is None:
            return self.mdl(name, "OmniPBR", {"diffuse_texture": (T.Asset, tex), "reflection_roughness_constant": (T.Float, .85),
                                             "specular_level": (T.Float, .35),
                                             "diffuse_tint": (T.Color3f, self.Gf.Vec3f(PRINT_WHITE))})
        inputs = {"diffuse_reflection_color_image": (T.Asset, tex), "diffuse_reflection_weight": (T.Float, PRINT_WHITE),
                  "specular_reflection_weight": (T.Float, 0.), "coat_weight": (T.Float, f["coat"]),
                  "coat_roughness": (T.Float, f["coat_rough"]), "coat_ior": (T.Float, 1.5)}
        if f.get("metal"):
            inputs.update({"metalness": (T.Float, f["metal"]), "specular_reflection_weight": (T.Float, 1.),
                           "specular_reflection_roughness": (T.Float, .3)})
        return self.mdl(name, "OmniSurface", inputs)

    def film(self, name):
        """Clear plastic film (shrink wrap, PVC label holder), thin enough not to refract: 92 % passes straight through,
        8 % is reflected specularly (two surfaces at n = 1.51, Fresnel ((n-1)/(n+1))^2 = 4.1 % each). Modelled as a smooth
        mirror at 8 % fractional opacity, so light and shadow rays from the imager's 1 mm LEDs pass it. Refractive
        versions (OmniSurface thin-walled transmission, OmniGlass) blocked the LEDs' direct light in the path tracer:
        behind them the card was lit by the room alone and nothing decoded."""
        T, Gf = self.Sdf.ValueTypeNames, self.Gf
        return self.mdl(name, "OmniPBR", {"diffuse_color_constant": (T.Color3f, Gf.Vec3f(1., 1., 1.)),
                                           "metallic_constant": (T.Float, 1.), "reflection_roughness_constant": (T.Float, .03),
                                           "enable_opacity": (T.Bool, True), "opacity_constant": (T.Float, .08)})

    def screen(self, name, texture):
        T, Sdf = self.Sdf.ValueTypeNames, self.Sdf
        return self.mdl(name, "OmniPBR", {"diffuse_color_constant": (T.Color3f, self.Gf.Vec3f(0.)),
                                          "enable_emission": (T.Bool, True),
                                          "emissive_color_texture": (T.Asset, Sdf.AssetPath(str(Path(texture).resolve()))),
                                          "emissive_intensity": (T.Float, 900.), "reflection_roughness_constant": (T.Float, .15)})

    def bind(self, prim, mat):
        self.UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    # ---------------------------------------------------------------- geometry
    def xform(self, path, p=None, q=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        if p is not None:
            x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            x.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        return x

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False):
        UsdGeom, Vt, Sdf = self.UsdGeom, self.Vt, self.Sdf
        v = np.asarray(v, np.float32); f = np.asarray(f, np.int32).reshape(-1, 3)
        g = UsdGeom.Mesh.Define(self.stage, path)
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
        g.CreateFaceVertexCountsAttr([3]*len(f))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
        g.CreateSubdivisionSchemeAttr("none")
        g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([v.min(0), v.max(0)], np.float32)))
        if normals is not None:
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
            g.SetNormalsInterpolation("vertex")
        elif smooth:
            import trimesh
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(trimesh.Trimesh(v, f, process=False).vertex_normals.astype(np.float32)))
            g.SetNormalsInterpolation("vertex")
        if uv is not None:
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        if mat is not None:
            self.bind(g, mat)
        return g

    def quad(self, path, centre, u, v, size, mat, flip_u=False, uv_rect=None):
        """Rectangle centred at `centre` spanning size[0] along unit u and size[1] along unit v (normal = u x v);
        `uv_rect` = (u0, v0, u1, v1) maps it onto a tile of an atlas texture."""
        c, u, v = (np.asarray(x, float) for x in (centre, u, v))
        a, b = size[0]/2, size[1]/2
        pts = [c - a*u - b*v, c + a*u - b*v, c + a*u + b*v, c - a*u + b*v]
        n = np.cross(u, v)
        u0, v0, u1, v1 = uv_rect or (0., 0., 1., 1.)
        uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)] if not flip_u else [(u1, v0), (u0, v0), (u0, v1), (u1, v1)]
        return self.mesh(path, pts, [0, 1, 2, 0, 2, 3], uv=uv, normals=[n]*4, mat=mat)

    def box(self, path, size, mat, p=None, q=None, bevel=.0015):
        from whiteboard.usd_scene import bevel_box
        g = bevel_box(self.stage, path, np.asarray(size, float), min(bevel, float(min(size))*.2))
        if p is not None:
            xf = self.UsdGeom.Xformable(g)
            xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            xf.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        if mat is not None:
            self.bind(g, mat)
        return g

    def box_uv(self, g, size, per_m):
        """Planar UVs per face at `per_m` cycles per metre (the texture keeps its physical scale)."""
        Sdf, Vt, UsdGeom = self.Sdf, self.Vt, self.UsdGeom
        pts = np.asarray(g.GetPointsAttr().Get()); idx = np.asarray(g.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(g.GetFaceVertexCountsAttr().Get())
        uv, k = [], 0
        for c in counts:
            face = pts[idx[k:k + c]]
            n = np.cross(face[1] - face[0], face[2] - face[0])
            ax = int(np.argmax(np.abs(n)))
            a, b_ = [i for i in range(3) if i != ax]
            uv.extend(face[:, [a, b_]]*per_m)
            k += c
        UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))

    def cylinder_side(self, path, r, h, mat, n=64, a0=0., a1=2*math.pi, z0=None, uv_mode="wrap"):
        """Cylinder side (axis z) between azimuths a0..a1 (azimuth 0 = -y, increasing toward +x)."""
        z0 = -h/2 if z0 is None else z0
        ks = np.linspace(a0, a1, n + 1)
        v, uv, nr, f = [], [], [], []
        for i, a in enumerate(ks):
            d = np.array([math.sin(a), -math.cos(a), 0.])
            for j, z in enumerate((z0, z0 + h)):
                v.append(d*r + np.array([0, 0, z])); nr.append(d)
                uv.append(((a - a0)/(a1 - a0), j) if uv_mode == "wrap" else (j, (a - a0)/(a1 - a0)))
        for i in range(n):
            a, b, c, d = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
            f += [a, c, d, a, d, b]
        return self.mesh(path, v, f, uv=uv, normals=nr, mat=mat)

    def tube(self, path, pts, radius, mat, n=12):
        """A round tube along a polyline (handles, cords)."""
        pts = np.asarray(pts, float)
        v, nr, f = [], [], []
        for i, p in enumerate(pts):
            t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]; t /= np.linalg.norm(t)
            a = np.array([0., 0., 1.]) if abs(t[2]) < .9 else np.array([1., 0., 0.])
            u = np.cross(t, a); u /= np.linalg.norm(u); w_ = np.cross(t, u)
            for k in range(n):
                ang = 2*math.pi*k/n
                d = math.cos(ang)*u + math.sin(ang)*w_
                v.append(p + radius*d); nr.append(d)
        for i in range(len(pts) - 1):
            for k in range(n):
                a, b = i*n + k, i*n + (k + 1) % n
                c, d = a + n, b + n
                f += [a, c, d, a, d, b]
        return self.mesh(path, v, f, normals=nr, mat=mat)

    def frustum_side(self, path, r0, r1, z0, z1, mat, n=48):
        """Side of a truncated cone (axis z) from radius r0 at z0 to r1 at z1, outward normals."""
        slope = (r0 - r1)/max(z1 - z0, 1e-9)
        v, nr, f = [], [], []
        for a in np.linspace(0, 2*math.pi, n + 1):
            d = np.array([math.sin(a), -math.cos(a), 0.])
            nrm = d + np.array([0., 0., slope]); nrm /= np.linalg.norm(nrm)
            for rad, z in ((r0, z0), (r1, z1)):
                v.append(d*rad + np.array([0, 0, z])); nr.append(nrm)
        for i in range(n):
            a, b, c, d = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
            f += [a, c, d, a, d, b]
        return self.mesh(path, v, f, normals=nr, mat=mat)

    def revolve(self, path, profile, mat, n=48):
        """Closed surface of revolution (axis z) through the (radius, z) profile, bottom to top; profile ends with
        radius 0 close it. Smooth outward normals from the profile's tangents."""
        prof = np.asarray(profile, float)
        v, nr, f = [], [], []
        m = len(prof)
        for a in np.linspace(0, 2*math.pi, n, endpoint=False):
            d = np.array([math.sin(a), -math.cos(a), 0.])
            for i, (rad, z) in enumerate(prof):
                t = prof[min(i + 1, m - 1)] - prof[max(i - 1, 0)]          # (dr, dz) along the profile
                nrm = d*t[1] - np.array([0., 0., t[0]])                    # outward normal of the revolved curve
                nrm = nrm/np.linalg.norm(nrm) if np.linalg.norm(nrm) > 1e-12 else np.array([0., 0., np.sign(z) or 1.])
                v.append(d*rad + np.array([0., 0., z])); nr.append(nrm)
        for k in range(n):
            k2 = (k + 1) % n
            for i in range(m - 1):
                a, b, c, d = k*m + i, k*m + i + 1, k2*m + i, k2*m + i + 1
                f += [a, c, d, a, d, b]
        return self.mesh(path, v, f, normals=nr, mat=mat)

    def water_pet(self):
        """A filled PET water bottle's body: clear, refracting like the water it holds (n = 1.33; PET's 0.3 mm wall,
        n = 1.57, is not resolved). The printed label and cap are separate opaque surfaces. (A refracting surface keeps
        the imager's LED light off what is behind it; nothing read lies behind a bottle's body.)"""
        T, Gf = self.Sdf.ValueTypeNames, self.Gf
        return self.mdl("pet_water_glass", "OmniGlass", {"glass_ior": (T.Float, 1.33), "glass_color": (T.Color3f, Gf.Vec3f(.97, .99, 1.)),
                                                        "thin_walled": (T.Bool, False), "frosting_roughness": (T.Float, 0.)})

    def jar_glass(self):
        """Soda-lime glass of a jar's empty headspace and shoulder (n = 1.52, a faint green tint), as one closed refracting
        volume like the PET bottle's water; the sauce below the fill line is an opaque surface under a glass coat."""
        T, Gf = self.Sdf.ValueTypeNames, self.Gf
        return self.mdl("jar_clear_glass", "OmniGlass", {"glass_ior": (T.Float, 1.52), "glass_color": (T.Color3f, Gf.Vec3f(.95, .99, .96)),
                                                        "thin_walled": (T.Bool, False), "frosting_roughness": (T.Float, 0.)})

    def disc(self, path, r, z, mat, up=True, n=48):
        v = [(0., 0., z)] + [(r*math.cos(a), r*math.sin(a), z) for a in np.linspace(0, 2*math.pi, n, endpoint=False)]
        f = []
        for i in range(n):
            a, b = 1 + i, 1 + (i + 1) % n
            f += [0, a, b] if up else [0, b, a]
        return self.mesh(path, v, f, normals=[(0, 0, 1 if up else -1)]*len(v), mat=mat)

    def annulus(self, path, r0, r1, z, mat, n=64):
        """Flat ring (axis z, facing up) between radii r0 < r1."""
        ks = np.linspace(0, 2*math.pi, n, endpoint=False)
        v = [(r*math.cos(a), r*math.sin(a), z) for a in ks for r in (r0, r1)]
        f = []
        for i in range(n):
            a, b, c, d = 2*i, 2*i + 1, 2*((i + 1) % n), 2*((i + 1) % n) + 1
            f += [a, b, d, a, d, c]
        return self.mesh(path, v, f, normals=[(0, 0, 1)]*len(v), mat=mat)

    def camera(self, path, eye, target, focal_mm, up=(0, 0, 1), role="diagnostic", aperture=36.):
        c = self.UsdGeom.Camera.Define(self.stage, path)
        c.AddTransformOp().Set(self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*eye), self.Gf.Vec3d(*target),
                                                            self.Gf.Vec3d(*up)).GetInverse())
        c.CreateHorizontalApertureAttr(float(aperture)); c.CreateVerticalApertureAttr(float(aperture*9/16))
        c.CreateFocalLengthAttr(float(focal_mm)); c.CreateClippingRangeAttr(self.Gf.Vec2f(.005, 40.))
        c.GetPrim().SetCustomDataByKey("observation_role", role)
        return c

    def alias(self, name, mounted):
        a = self.UsdGeom.Xform.Define(self.stage, "/Cameras/" + name)
        a.GetPrim().SetCustomDataByKey("mounted_camera_path", str(mounted))

    def save(self):
        self.stage.GetRootLayer().Save()


# ------------------------------------------------------------------------------ export
def export(world, folder, extra=None):
    """Export the world's static appearance + posable bodies to folder/scene.usdc; returns (path, meta). `extra(S)`
    (probes only: room_lux's white cards) adds prims to the stage before it is saved."""
    from . import artwork, packaging as PK, dressing
    import newton
    sc = world.scene
    b = sc.b
    S = Stage(folder)
    tex = S.folder/"textures"
    for name, (rgb, rough, metal) in COLORS.items():
        S.color(name, rgb, rough, metal)
    S.color("led_on", (.1, 1., .2), .3, 0., emissive=(0.2, 6., 0.6))
    S.color("imager_glass", (.02, .02, .02), .05, 0.)
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    V = int(newton.ShapeFlags.VISIBLE)
    item_bodies = {it["body"] for it in sc.items.values() if it.get("sku")}
    item_bodies |= {it["cover_body"] for it in sc.items.values() if it.get("cover_body") is not None}   # drawn by _hinged_book
    skip = {"scanner_head", "scanner_handle", "trigger_blade", "display", "floor"}
    deck_mat = S.printed("particle_board", PK.particle_board(tex), "matte")
    # ---- shapes: robots (visual meshes), fixtures, containers
    for i in range(b.shape_count):
        body, lab = b.shape_body[i], b.shape_label[i]
        if lab in skip or body in item_bodies or lab.startswith(("keycap_", "cradle_", "basket_wall", "bag_tote_wall", "returns_bin_wall")):
            continue
        flags = int(b.shape_flags[i])
        mname = sc.materials.get(i, "robot")
        is_robot = mname in ("robot",) or (body >= 0 and body in sc.robot_bodies and mname == "robot")
        if is_robot and not flags & V:
            continue                                          # robot collision hulls are never drawn
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        src = b.shape_source[i]
        scale = np.asarray(b.shape_scale[i], float)
        tf = np.asarray(b.shape_transform[i], float)
        if i in sc.hull_boxes:
            g = S.box(path, 2*np.asarray(sc.hull_boxes[i]), None)
        elif src is not None and hasattr(src, "vertices"):
            g = S.mesh(path, np.asarray(src.vertices)*scale, src.indices, smooth=True)
        elif b.shape_type[i] == newton.GeoType.BOX:
            g = S.box(path, 2*scale, None)
            if lab.startswith("deck_"):
                S.box_uv(g, 2*scale, 1.)
        else:
            continue
        xf = S.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(S.Gf.Vec3d(*tf[:3]))
        xf.AddOrientOp().Set(S.Gf.Quatf(float(tf[6]), S.Gf.Vec3f(*tf[3:6])))
        if mname == "robot":
            blabel = b.body_label[body] if body >= 0 else ""
            dark = any(k in blabel for k in ("link7", "finger", "hand")) or blabel.endswith("link0")
            mat = S.mats["robot_dark" if dark else "robot"]
        elif lab.startswith("deck_"):
            mat = deck_mat
        else:
            mat = S.mats.get(mname) or S.mats["cart_frame"]
        S.bind(g, mat)
    _cradle(S, sc, b)
    _basket(S, sc)
    _crates(S, sc)
    # ---- stock units: packaging + printed symbols
    art = S.folder/"art"; art.mkdir(exist_ok=True)
    made = {}
    for name, it in sc.items.items():
        if not it.get("sku"):
            continue
        s = SKUS[it["sku"]]
        base = f"/Lab/body_{it['body']}/pkg"
        if it.get("cover_body") is not None:
            _hinged_book(S, s, it, art, made)
        else:
            _package(S, base, s, art, made, it)
    # ---- printed symbols (items, locations, inner units)
    atlas = _location_atlases(S, sc, art)
    for k, L in enumerate(sc.labels):
        _symbol_patch(S, k, L, art, atlas)
    # ---- scanner, cradle, keypad legends, display
    meta = _scanner(S, sc, world)
    _keypad(S, sc, art)
    meta["display"] = _display(S, sc, art)
    # ---- room dressing and cameras
    credits = dressing.dress(S, world)
    meta["cameras"] = _cameras(S, sc, world)
    meta["visual_assets"] = sorted(credits)
    if extra is not None:
        extra(S)
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def _package(S, base, s, art, made, it):
    from . import packaging as PK
    key = s.id
    if s.shape == "box":
        if key not in made:
            made[key] = PK.box_faces(s, art)
        faces = made[key]
        W, D, H = s.dims
        e = .0002
        spec = {"front": ((0, -D/2 - e, 0), (1, 0, 0), (0, 0, 1), (W, H)), "back": ((0, D/2 + e, 0), (-1, 0, 0), (0, 0, 1), (W, H)),
                "right": ((W/2 + e, 0, 0), (0, 1, 0), (0, 0, 1), (D, H)), "left": ((-W/2 - e, 0, 0), (0, -1, 0), (0, 0, 1), (D, H)),
                "top": ((0, 0, H/2 + e), (1, 0, 0), (0, 1, 0), (W, D)), "bottom": ((0, 0, -H/2 - e), (1, 0, 0), (0, -1, 0), (W, D))}
        film = s.finish == "film" and s.extra.get("inner")
        for face, (c, u, v, size) in spec.items():
            if film and face not in (s.barcode.face,):
                continue                              # clear film: only the printed panel is opaque
            mat = S.printed(f"pkg_{key}_{face}", faces[face][0], "matte" if s.category == "library" and face in ("right", "top", "bottom") else s.finish)
            if film:
                # printed panel on the film: a band across the face, not the whole face
                size = (size[0]*.62, size[1]*.40)
            S.quad(f"{base}/{face}", c, u, v, size, mat)
        if film:
            inner = SKUS[s.extra["inner"]]
            if inner.id not in made:
                made[inner.id] = PK.wrap_texture(inner, art)
            for j, (cx, cy, cz, az) in enumerate(it.get("inner", [])):
                x = S.xform(f"{base}/inner_{j}", (cx, cy, cz), Rotation.from_euler("z", az, degrees=True).as_quat())
                _can(S, str(x.GetPath()), inner, made[inner.id], art)
            S.box(f"{base}/film", (W + .002, D + .002, H + .002), S.film("shrink_film"), bevel=.004)
    else:
        if key not in made:
            made[key] = PK.wrap_texture(s, art)
        _can(S, base, s, made[key], art)


def _hinged_book(S, s, it, art, made):
    """A book with a hinged front cover: the text block (back cover, spine, page edges, first endpaper) on the item
    body and the front board or card (cover art outside, endpaper inside, edges) on the cover body."""
    from . import packaging as PK
    if s.id not in made:
        made[s.id] = PK.box_faces(s, art)
    faces = made[s.id]
    W, D, H = s.dims
    t = P.book_cover(s)[0]
    e = .0002
    blk = f"/Lab/body_{it['body']}/pkg"
    Db, yc = D - t, t/2
    spec = {"back": ((0, D/2 + e, 0), (-1, 0, 0), (0, 0, 1), (W, H)),
            "right": ((W/2 + e, yc, 0), (0, 1, 0), (0, 0, 1), (Db, H)), "left": ((-W/2 - e, yc, 0), (0, -1, 0), (0, 0, 1), (Db, H)),
            "top": ((0, yc, H/2 + e), (1, 0, 0), (0, 1, 0), (W, Db)), "bottom": ((0, yc, -H/2 - e), (1, 0, 0), (0, -1, 0), (W, Db))}
    for face, (c, u, v, size) in spec.items():
        mat = S.printed(f"pkg_{s.id}_{face}", faces[face][0], "matte" if face in ("right", "top", "bottom") else s.finish)
        S.quad(f"{blk}/{face}", c, u, v, size, mat)
    endpaper = S.color("endpaper", (.86, .84, .78), .8, 0.)
    S.quad(f"{blk}/endpaper", (0, -D/2 + t - e, 0), (1, 0, 0), (0, 0, 1), (W - .002, H - .002), endpaper)
    cov = f"/Lab/body_{it['cover_body']}/pkg"
    S.quad(f"{cov}/front", (W/2, -t/2 - e, 0), (1, 0, 0), (0, 0, 1), (W, H), S.printed(f"pkg_{s.id}_front", faces["front"][0], s.finish))
    S.quad(f"{cov}/inside", (W/2, t/2 + e, 0), (-1, 0, 0), (0, 0, 1), (W, H), endpaper)
    edge = S.color(f"board_edge_{s.id}", tuple(c/255*.8 for c in s.palette[0]), .6, 0.)
    S.quad(f"{cov}/edge", (W + e, 0, 0), (0, 1, 0), (0, 0, 1), (t, H), edge)
    S.quad(f"{cov}/edge_top", (W/2, 0, H/2 + e), (1, 0, 0), (0, 1, 0), (W, t), edge)
    S.quad(f"{cov}/edge_bottom", (W/2, 0, -H/2 - e), (1, 0, 0), (0, -1, 0), (W, t), edge)


def _crates(S, sc):
    """The bagging tote and the library returns bin (the physics is each one's five 4 mm walls): moulded polypropylene
    with vertical ribs, a thickened rim and, on the returns bin, a printed sign. Visual only."""
    from PIL import Image, ImageDraw
    from .dressing import _sign
    for name, base_rgb, rib_rgb, sign in (("bag_tote", (58, 62, 66), (40, 43, 46), None),
                                          ("returns_bin", (28, 70, 140), (18, 48, 100), ("BOOK RETURNS", "PLEASE PLACE BOOKS HERE"))):
        it = sc.items.get(name)
        if not it:
            continue
        L, W, H = it["container_size"]
        wall = .004
        tile = S.folder/"art"/f"{name}_ribs.png"
        tile.parent.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (400, 400), base_rgb); d = ImageDraw.Draw(im)
        for x0 in (40, 240):                                # two 12 mm ribs per 40 mm tile
            d.rounded_rectangle([x0, 20, x0 + 120, 380], radius=50, fill=rib_rgb)
        im.save(tile)
        mat = S.printed(f"{name}_moulded", tile, "semigloss")
        rim = S.color(f"{name}_rim", tuple(c/255*1.1 for c in base_rgb), .45, 0.)
        root = f"/Lab/body_{it['body']}/{name}"
        parts = [((0, 0, -H/2 + wall/2), (L, W, wall)),
                 ((L/2 - wall/2, 0, 0), (wall, W, H)), ((-L/2 + wall/2, 0, 0), (wall, W, H)),
                 ((0, W/2 - wall/2, 0), (L, wall, H)), ((0, -W/2 + wall/2, 0), (L, wall, H))]
        for k, (c, size) in enumerate(parts):
            g = S.box(f"{root}/wall_{k}", size, mat, p=c, q=(0., 0., 0., 1.), bevel=.0008)
            S.box_uv(g, size, 1/.04)
        for k, (c, size) in enumerate((((0, W/2, H/2 - .005), (L + .010, .010, .010)), ((0, -W/2, H/2 - .005), (L + .010, .010, .010)),
                                       ((L/2, 0, H/2 - .005), (.010, W, .010)), ((-L/2, 0, H/2 - .005), (.010, W, .010)))):
            S.box(f"{root}/rim_{k}", size, rim, p=c, q=(0., 0., 0., 1.), bevel=.003)
        if sign:
            png = _sign(S.folder/"art", f"{name}_sign", sign[0], sign[1], (245, 245, 240), (20, 50, 110))
            smat = S.printed(f"{name}_sign", png, "matte")
            for side in (-1., 1.):                          # on both long walls, outside
                S.quad(f"{root}/sign_{int(side > 0)}", (0., side*(W/2 + .0006), .01), (-side, 0., 0.), (0., 0., 1.),
                       (min(.30, L*.8), min(.075, H*.45)), smat)


def _basket(S, sc):
    """The customer's hand basket (the physics is its five 4 mm walls): moulded blue polypropylene walls with rows
    of rounded slots, a thickened rim, and two carry handles folded down along the long sides, as on a store
    hand basket. Visual only; the handles lie outside the walls and have no collision."""
    it = sc.items.get("basket")
    if not it:
        return
    from PIL import Image, ImageDraw
    L, W, H = it["container_size"]
    wall = .004
    tile = S.folder/"art"/"basket_slots.png"
    tile.parent.mkdir(parents=True, exist_ok=True)
    if not tile.exists():
        px = 400                                   # one 40 mm tile: a 26 x 9 mm slot with 4 mm corner radius
        im = Image.new("RGB", (px, px), (30, 88, 190)); d = ImageDraw.Draw(im)
        d.rounded_rectangle([70, 155, 330, 245], radius=40, fill=(8, 26, 70))
        im.save(tile)
    mat = S.printed("basket_moulded", tile, "semigloss")
    base = f"/Lab/body_{it['body']}/basket"
    parts = [((0, 0, -H/2 + wall/2), (L, W, wall)),
             ((L/2 - wall/2, 0, 0), (wall, W, H)), ((-L/2 + wall/2, 0, 0), (wall, W, H)),
             ((0, W/2 - wall/2, 0), (L, wall, H)), ((0, -W/2 + wall/2, 0), (L, wall, H))]
    for k, (c, size) in enumerate(parts):
        g = S.box(f"{base}/wall_{k}", size, mat, p=c, q=(0., 0., 0., 1.), bevel=.0008)
        S.box_uv(g, size, 1/.04)
    rim = S.mats["basket_blue"]
    for k, (c, size) in enumerate((((0, W/2, H/2 - .006), (L + .012, .012, .012)), ((0, -W/2, H/2 - .006), (L + .012, .012, .012)),
                                   ((L/2, 0, H/2 - .006), (.012, W, .012)), ((-L/2, 0, H/2 - .006), (.012, W, .012)))):
        S.box(f"{base}/rim_{k}", size, rim, p=c, q=(0., 0., 0., 1.), bevel=.003)
    handle = S.color("basket_handle", (.06, .14, .34), .5, 0.)
    for side in (-1., 1.):                          # folded down: from the rim near each end, bowed out and down
        pts = []
        for t in np.linspace(0., 1., 25):
            x = (-L/2 + .06) + t*(L - .12)
            bow = math.sin(math.pi*t)
            pts.append((x, side*(W/2 + .010 + .030*bow), H/2 - .004 - .070*bow))
        S.tube(f"{base}/handle_{int(side > 0)}", pts, .0045, handle)


def _can(S, base, s, wrap, art):
    r, h = s.dims
    path, C, H = wrap
    from .catalog import neck
    g = neck(s)
    if g:                                   # jar / bottle: glass or PET body, printed panel, shoulder, cap
        (b0, b1), (s0, s1), (c0, c1), (p0, p1) = g["body"], g["shoulder"], g["cap"], g["panel"]
        S.cylinder_side(f"{base}/wrap", r + .0002, p1 - p0, S.printed(f"wrap_{s.id}", path, s.finish), z0=p0)
        if g["body_mat"] == "pet_water":         # clear bottle full of water: one closed refracting volume to the neck
            S.revolve(f"{base}/water", [(0., b0), (r*.93, b0), (r, b0 + .004), (r, b1), (g["cap_r"]*.9, c0), (0., c0)], S.water_pet())
            S.cylinder_side(f"{base}/cap", g["cap_r"] + .0002, c1 - c0, S.mats[g["cap_mat"]], z0=c0)
            S.disc(f"{base}/top", g["cap_r"], c1 + .0003, S.mats[g["cap_mat"]], up=True)
            return
        body = S.mats[g["body_mat"]]
        if g["body_mat"] == "jar_glass":
            # a glass jar of sauce: the sauce fills the body to 12 mm below the shoulder, seen through the glass (an opaque
            # surface with a glass coat, FINISH); the headspace, shoulder and neck are clear glass up to the lid
            fill = min(b1 - .012, max(p1, b1 - .012))
            if p0 > b0:
                S.cylinder_side(f"{base}/body_low", r, p0 - b0, body, z0=b0)
            if fill > p1:
                S.cylinder_side(f"{base}/body_high", r, fill - p1, body, z0=p1)
            S.disc(f"{base}/sauce_top", r*.97, fill, body, up=True)
            S.revolve(f"{base}/headspace", [(0., fill + .0005), (r, fill + .0005), (r, s0), (g["cap_r"]*.92, s1),
                                            (g["cap_r"]*.92, c0), (0., c0)], S.jar_glass())
            S.cylinder_side(f"{base}/cap", g["cap_r"] + .0002, c1 - c0, S.mats[g["cap_mat"]], z0=c0)
            S.disc(f"{base}/top", g["cap_r"], c1 + .0003, S.mats[g["cap_mat"]], up=True)
            S.disc(f"{base}/bottom", r*.98, b0 - .0003, body, up=False)
            return
        if p0 > b0:
            S.cylinder_side(f"{base}/body_low", r, p0 - b0, body, z0=b0)
        if b1 > p1:
            S.cylinder_side(f"{base}/body_high", r, b1 - p1, body, z0=p1)
        S.frustum_side(f"{base}/shoulder", r, g["cap_r"], s0, s1, S.mats[g["shoulder_mat"]])
        S.cylinder_side(f"{base}/cap", g["cap_r"] + .0002, c1 - c0, S.mats[g["cap_mat"]], z0=c0)
        S.disc(f"{base}/top", g["cap_r"], c1 + .0003, S.mats[g["cap_mat"]], up=True)
        S.disc(f"{base}/bottom", r*.98, b0 - .0003, body, up=False)
        return
    S.cylinder_side(f"{base}/wrap", r + .0002, h, S.printed(f"wrap_{s.id}", path, s.finish))
    top = {"K05": "cap_blue", "K04": "lid_gold", "K10": "cap_blue"}.get(s.id, "lid_metal")
    S.disc(f"{base}/top", r*.98, h/2 - .0025, S.mats[top], up=True)
    if top == "lid_metal":
        # the double seam: a rolled metal rim 2.5 mm above the recessed lid (render only; the collider is the cylinder)
        S.cylinder_side(f"{base}/seam", r + .0004, .003, S.mats["lid_metal"], z0=h/2 - .003)
        S.cylinder_side(f"{base}/seam_in", r*.95, .0028, S.mats["lid_metal"], z0=h/2 - .0028)
        S.annulus(f"{base}/seam_top", r*.95, r + .0004, h/2 + .0002, S.mats["lid_metal"])
    else:
        S.disc(f"{base}/top_cover", r*.98, h/2 + .0003, S.mats[top], up=True)
    S.disc(f"{base}/bottom", r*.98, -h/2 - .0003, S.mats["lid_metal" if top == "lid_metal" else "tote_grey"], up=False)


def _atlas(images, path, cols, pad=12, fill=(255, 255, 255)):
    """Pack equal-sized PIL images into one texture (white margins stop filtering from bleeding between tiles);
    returns the uv rect (u0, v0, u1, v1) of each image (USD uv: origin at the bottom-left)."""
    from PIL import Image
    w, h = images[0].size
    rows = (len(images) + cols - 1)//cols
    W, H = cols*(w + 2*pad), rows*(h + 2*pad)
    sheet = Image.new("RGB", (W, H), fill)
    rects = []
    for i, im in enumerate(images):
        c, r = i % cols, i//cols
        x, y = c*(w + 2*pad) + pad, r*(h + 2*pad) + pad
        sheet.paste(im.resize((w, h)), (x, y))
        rects.append((x/W, 1 - (y + h)/H, (x + w)/W, 1 - y/H))
    sheet.save(path)
    return rects


def _location_atlases(S, sc, art, per=16):
    """All shelf location labels in atlases of `per` labels (4 x 4), one material each: 45 label materials cost the
    render worker's first frame ~2 s of material creation; returns {label name: (material, uv)}."""
    from PIL import Image
    from .artwork import location_label
    locs = [L for L in sc.labels if L["kind"] == "location"]
    out = {}
    for a0 in range(0, len(locs), per):
        chunk = locs[a0:a0 + per]
        ims = [Image.open(location_label(L["text"], art)[0]).convert("RGB") for L in chunk]
        path = art/f"location_atlas_{a0//per}.png"
        rects = _atlas(ims, path, cols=4)
        mat = S.printed(f"loc_atlas_{a0//per}", path, "matte")
        for L, r in zip(chunk, rects):
            out[L["name"]] = (mat, r)
    return out


def _symbol_patch(S, k, L, art, atlas=None):
    """A printed symbol as its own patch at 10 texels/module, posed from Scene.labels (flat or curved)."""
    from .artwork import barcode_patch, location_label
    sym = L["symbol"]
    if L["kind"] == "location":
        p, w, h, _ = location_label(L["text"], art)
    elif L["kind"] == "library":
        from .packaging import library_label
        p, w, h, _ = library_label(L["symbol"].data, art)
    else:
        p, w, h = barcode_patch(sym, art, f"sym_{sym.kind}_{sym.data}".replace("/", "_"))
    finish = L["finish"] if L["kind"] != "location" else "matte"
    finish = finish if finish in ("matte", "semigloss", "glossy", "film", "metal") else "semigloss"
    uv_rect = None
    if atlas and L["name"] in atlas:
        mat, uv_rect = atlas[L["name"]]
    else:
        # identical prints share one material (each material costs a render worker ~46 ms at its first frame)
        import hashlib
        mat = S.printed(f"sym_{hashlib.sha1(f'{Path(p).name}|{finish}'.encode()).hexdigest()[:12]}", p, finish)
    base = f"/Lab/body_{L['body']}" if L["body"] >= 0 else "/Lab/static_labels"
    if L["body"] < 0 and not S.stage.GetPrimAtPath(base):
        S.xform(base)
    n, up, c = (np.asarray(L[x], float) for x in ("normal", "up", "center"))
    if L.get("radius"):
        r = L["radius"]
        az_c = math.atan2(n[0], -n[1])
        ladder = L.get("text") == "ladder"
        span = (h if ladder else w)/r
        zc = c[2]
        g = S.cylinder_side(f"{base}/sym_{k}", r + .0004, (w if ladder else h), mat, n=24, a0=az_c - span/2, a1=az_c + span/2,
                            z0=zc - (w if ladder else h)/2, uv_mode="ladder" if ladder else "wrap")
        # cylinder centre is the item axis: recover it from the label centre
        axis_c = c - n*r
        xf = S.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(S.Gf.Vec3d(float(axis_c[0]), float(axis_c[1]), 0.))
        if ladder:
            pass
        return
    u = np.cross(up, n)
    S.quad(f"{base}/sym_{k}", c + n*.0003, u, up, (w, h), mat, uv_rect=uv_rect)


def _scanner(S, sc, world):
    """Scanner visual (open-fronted head shell, window bezel, raked grip, trigger, good-read LED), the imager
    camera at the entrance pupil, two illumination LEDs and the aim-line source."""
    UsdLux, Gf, Sdf = S.UsdLux, S.Gf, S.Sdf
    sb, tb = sc.bodies["scanner"], sc.bodies["scanner_trigger"]
    base = f"/Lab/body_{sb}/vis"
    S.xform(base)
    hh = sc.HEAD_HALF
    body, bz = S.mats["scanner_body"], S.mats["scanner_grip"]
    I = (0, 0, 0, 1)
    # moulded housing (render-only rounded meshes, housing.py; the colliders are the boxes of scene.py)
    from .housing import scanner_meshes, trigger_mesh
    hc, hq = sc.handle_frame()
    ms = scanner_meshes(hh, hc, hq, sc.HANDLE_HALF)
    for part, mat in (("head", body), ("lining", S.mats["imager_glass"]), ("handle", bz)):
        m = ms[part].smooth_shaded                       # vertices split at creases (the CSG window rim stays crisp)
        S.mesh(f"{base}/{part}", m.vertices, m.faces, mat=mat, normals=m.vertex_normals)
    # interior baffle behind the optics (dark), seen through the window
    S.box(f"{base}/baffle", (.002, .030, .024), S.mats["imager_glass"], (-P.PUPIL_SETBACK_M - .01, 0, 0), I)
    # good-read LED on the top rear: off and on versions, visibility switched per frame
    led_x = -2*hh[0] + .015
    S.box(f"{base}/led_off", (.008, .004, .0015), S.mats["led_off"], (led_x, 0, hh[2] + .0005), I)
    on = S.box(f"{base}/led_on", (.008, .004, .0016), S.mats["led_on"], (led_x, 0, hh[2] + .0006), I)
    on.GetPrim().GetAttribute("visibility").Set("invisible")
    # trigger blade on the trigger body
    tm = trigger_mesh(P.TRIGGER_ARM_M).smooth_shaded
    S.mesh(f"/Lab/body_{tb}/blade", tm.vertices, tm.faces, mat=S.mats["trigger_black"], normals=tm.vertex_normals)
    # imager camera at the entrance pupil, thin lens; LEDs; aim line (shared with the optics bench)
    leds, aim = add_imager(S, f"/Lab/body_{sb}", sc.scanner["pupil"], sc.scanner["leds"], sc.scanner["aim"])
    return dict(scanner=dict(body=sb, camera="imager", leds=leds, aim=aim,
                             led_on=f"{base}/led_on", fstop=RTX_FOCAL_UNIT_M*IMAGER_FOCAL_MM/P.PUPIL_DIAMETER_M))


def add_imager(S, parent, pupil, led_positions, aim_position, alias="imager"):
    """The imager (thin-lens camera at the entrance pupil), its two illumination LEDs and the IES aim-line source,
    under `parent` (scanner body frame: +x out of the window, +z up). Returns (led prim paths, aim prim path)."""
    UsdLux, Gf, Sdf = S.UsdLux, S.Gf, S.Sdf
    qc = quat_xyzw_from_matrix(R_CAM)
    cam_x = S.xform(f"{parent}/imager_frame", pupil, qc)
    cam = S.UsdGeom.Camera.Define(S.stage, str(cam_x.GetPath()) + "/imager")
    cam.CreateFocalLengthAttr(IMAGER_FOCAL_MM); cam.CreateHorizontalApertureAttr(IMAGER_APERTURE_MM)
    cam.CreateVerticalApertureAttr(IMAGER_APERTURE_MM*P.HEIGHT_PX/P.WIDTH_PX)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.004, 20.))
    cam.CreateFStopAttr(float(RTX_FOCAL_UNIT_M*IMAGER_FOCAL_MM/P.PUPIL_DIAMETER_M))
    cam.CreateFocusDistanceAttr(float(P.FOCUS_DISTANCE_M))
    cam.GetPrim().SetCustomDataByKey("observation_role", "device_internal")
    S.alias(alias, cam.GetPath())
    leds = []
    for j, pos in enumerate(led_positions):
        lx = S.xform(f"{parent}/led_frame_{j}", pos, qc)
        s = UsdLux.SphereLight.Define(S.stage, str(lx.GetPath()) + "/led")
        s.CreateRadiusAttr(.001); s.CreateIntensityAttr(0.); s.CreateColorAttr(Gf.Vec3f(1., .02, 0.))
        sh = UsdLux.ShapingAPI.Apply(s.GetPrim())
        led_ies = S.folder/"led.ies"
        if j == 0:                  # always rewritten: an export folder reused after a pattern change kept the old file
            ies_led(led_ies)
        sh.CreateShapingIesFileAttr(Sdf.AssetPath(str(led_ies))); sh.CreateShapingIesNormalizeAttr(True)
        leds.append(str(s.GetPath()))
    ies = S.folder/"aim_line.ies"
    ies_line(ies)
    tilt = math.atan2(P.AIM_OFFSET_M, P.AIM_CROSSING_M)
    Ra = Rotation.from_euler("y", tilt)*Rotation.from_matrix(R_CAM)*Rotation.from_euler("z", 90, degrees=True)
    ax = S.xform(f"{parent}/aim_frame", aim_position, Ra.as_quat())
    aim = UsdLux.SphereLight.Define(S.stage, str(ax.GetPath()) + "/aim")
    aim.CreateRadiusAttr(.0005); aim.CreateIntensityAttr(0.); aim.CreateColorAttr(Gf.Vec3f(1., .55, .05))
    ash = UsdLux.ShapingAPI.Apply(aim.GetPrim())
    ash.CreateShapingIesFileAttr(Sdf.AssetPath(str(ies))); ash.CreateShapingIesNormalizeAttr(True)
    return leds, str(aim.GetPath())


def _side_dir(sc):
    """Horizontal unit vector from the seated scanner towards its right-hand side (-y of the scanner frame), for a
    profile view of the scanner in its cradle."""
    from scipy.spatial.transform import Rotation
    y = Rotation.from_quat(sc.cradle["scanner_quat"]).apply((0., -1., 0.))
    y[2] = 0.
    return tuple(y/np.linalg.norm(y))


def _cradle(S, sc, b):
    """Moulded cradle (render-only): the union of the cradle's colliders (scene.build_cradle) -- the teardrop base, the
    pedestal and the handle's socket (floor, front lip, side walls, raised rear wall) -- with rounded edges, so that
    nothing is drawn that has no collider and nothing collides that is not drawn. (The first moulded shell filled the
    space under the cup with a hull that had no collider.)"""
    import trimesh
    from scipy.spatial.transform import Rotation
    from .housing import rounded_box
    parts = []
    for i in range(b.shape_count):
        lab = b.shape_label[i]
        if not lab.startswith("cradle_") or lab == "cradle_base":
            continue
        if lab == "cradle_pedestal":
            parts.append(trimesh.convex.convex_hull(np.asarray(b.shape_source[i].vertices, float)))
            continue
        tf = np.asarray(b.shape_transform[i], float)
        half = np.asarray(b.shape_scale[i], float)
        m = rounded_box(half, min(.004, .8*float(min(half))))
        T = np.eye(4); T[:3, :3] = Rotation.from_quat(tf[3:7]).as_matrix(); T[:3, 3] = tf[:3]
        parts.append(m.apply_transform(T))
    if not parts:
        return
    cr = sc.cradle
    base = sc.cradle_base_mesh()                                          # the teardrop base, as its collider
    Tb = np.eye(4); Tb[:3, :3] = Rotation.from_quat(cr["quat"]).as_matrix(); Tb[:3, 3] = np.asarray(cr["pos"], float)
    base.apply_transform(Tb)
    shell = trimesh.boolean.union([base] + parts, engine="manifold").smooth_shaded
    mat = S.mats["cradle_black"]
    S.mesh("/Lab/cradle_body", shell.vertices, shell.faces, mat=mat, normals=shell.vertex_normals)


def _keypad(S, sc, art):
    """Keycaps with their legends (one atlas texture and material for all 17 legends)."""
    from PIL import Image, ImageDraw
    from .artwork import _font, FONT_BOLD
    names = list(sc.keys)
    ims = []
    for name in names:
        w, h = sc.keys[name]["size"]
        im = Image.new("RGB", (int(w*4000), int(h*4000)), (206, 206, 202))
        d = ImageDraw.Draw(im)
        d.text((im.width/2, im.height/2), name if name != "NUM" else "Num", font=_font(FONT_BOLD, 26 if len(name) < 3 else 18),
               fill=(30, 30, 30), anchor="mm")
        ims.append(im)
    # tiles must share one size: pad each legend onto the largest key's canvas, centred
    W = max(i.width for i in ims); H = max(i.height for i in ims)
    tiles, scale = [], []
    for im in ims:
        t = Image.new("RGB", (W, H), (206, 206, 202)); t.paste(im, ((W - im.width)//2, (H - im.height)//2))
        tiles.append(t); scale.append((im.width/W, im.height/H))
    rects = _atlas(tiles, art/"keycap_atlas.png", cols=6, pad=4, fill=(206, 206, 202))
    mat = S.printed("keycap_atlas", art/"keycap_atlas.png", "matte")
    for name, (u0, v0, u1, v1), (fx, fy) in zip(names, rects, scale):
        k = sc.keys[name]
        w, h = k["size"]
        cu, cv, hu, hv = (u0 + u1)/2, (v0 + v1)/2, (u1 - u0)*fx/2, (v1 - v0)*fy/2
        base = f"/Lab/body_{k['body']}"
        S.box(f"{base}/cap", (w - .0025, h - .0025, .008), S.mats["keycap"], (0, 0, 0), (0, 0, 0, 1), bevel=.0012)
        S.quad(f"{base}/legend", (0, 0, .0041), (1, 0, 0), (0, 1, 0), (w - .005, h - .005), mat,
               uv_rect=(cu - hu, cv - hv, cu + hu, cv + hv))


def _display(S, sc, art):
    """Host display screen (emissive), texture swapped per observation by the app state."""
    from .device import HostApp
    fx = sc.fixtures["tablet"]
    Rq = Rotation.from_quat(fx["quat"])
    c = np.asarray(fx["center"]) + Rq.apply((0, 0, .0085))
    first = art/"display_initial.png"
    HostApp.draw_screen({"title": "READY", "lines": ["Scanner in cradle"]}, first)
    mat = S.screen("display_screen", first)
    S.quad("/Lab/static_display/screen", c, Rq.apply((1, 0, 0)), Rq.apply((0, 1, 0)), fx["screen"], mat)
    return dict(texture_attr="/Looks/display_screen/Shader.inputs:emissive_color_texture")


def _cameras(S, sc, world):
    """Policy cameras (mast + wrists) and diagnostic cameras; returns {name: role}."""
    out = {}
    mast = sc.fixtures["mast_camera"]["eye"]
    c = S.camera("/Cameras/cart_rgb_cam", mast, sc.fixtures["mast_camera"].get("target", (0., .30, .78)),
                 18/math.tan(math.radians(69/2)), role="policy")
    S.alias("cart_rgb", c.GetPath()); out["cart_rgb"] = "policy"
    for name, m in sc.cameras.items():
        g = S.UsdGeom.Camera.Define(S.stage, f"/Lab/body_{m['body']}/{name}")
        g.AddTransformOp().Set(S.Gf.Matrix4d().SetLookAt(S.Gf.Vec3d(*m["eye"]), S.Gf.Vec3d(*m["target"]), S.Gf.Vec3d(*m["up"])).GetInverse())
        g.CreateHorizontalApertureAttr(36.); g.CreateVerticalApertureAttr(20.25)
        g.CreateFocalLengthAttr(18/math.tan(math.radians(m["fov"]/2))); g.CreateClippingRangeAttr(S.Gf.Vec2f(.005, 30.))
        g.GetPrim().SetCustomDataByKey("observation_role", "policy")
        S.alias(name, g.GetPath()); out[name] = "policy"
    preset = world.config.preset
    diag = dict(room=((2.0, -2.6, 1.9), (0., .10, .95), 22.), front=((0.35, -1.55, 1.35), (0., .15, 1.0), 28.),
                tray=((.45, -.55, 1.10), (.02, -.21, .80), 36.), slot=((.62, -.42, 1.40), (0., .15, 1.10), 30.),
                handover=((.30, -.62, 1.42), (.08, -.28, 1.19), 38.),
                side=((1.25, -.35, 1.25), (0., -.05, 1.0), 26.))
    if preset != "stockroom":
        # the counter/library room ends 1.6 m behind the robot (the stockroom overview eye would be outside it)
        diag.update(front=((0.2, -1.55, 1.45), (0., .30, .95), 26.), slot=((.35, -.45, 1.35), (-.1, .3, .95), 28.),
                    room=((2.6, -1.35, 2.1), (-.1, .35, .95), 20.), customer=((.9, 2.3, 1.55), (-.05, .1, .95), 24.))
        if preset == "counter":                                          # close-up of the basket and the scan zone
            diag.update(basket=((-.05, -.12, 1.55), (-.32, .22, .98), 40.))
            # from the customer's side, where no arm passes (from the robot's side the hand-over above the basket is
            # hidden behind the arm, and the tote's inside is out of every view)
            diag.update(handover_basket=((-.55, 1.05, 1.42), (-.32, .22, 1.10), 34.),
                        tote_inside=((.40, .78, 1.50), (.34, .24, .97), 34.))
        else:                                                            # close-up of the returns stack
            diag.update(books=((.05, -.14, 1.30), (-.30, .22, .99), 36.))
            # the desk and the returns bin from the patron's side, where no arm passes (the robot-side books, slot and
            # cradle views had an arm link in front of the eye in 17-19 frames of a library run)
            diag.update(desk_far=((-.10, 1.15, 1.50), (-.12, .22, .99), 30.))
    if preset == "stockroom":
        # the floor around the cart, wide: a scanner knocked out of the hand lands there (E17's cradle jam fell out of
        # every other view)
        diag.update(floor=((1.55, -1.55, 1.05), (.15, -.25, .30), 16.))
    if getattr(sc, "cart_pose", None) is not None:
        # G1 configuration: the tray close-up follows the parked cart; two views of the humanoid at work
        eye, look, f = diag["tray"]
        diag["tray"] = (sc.cart_point(eye)[0], sc.cart_point(look)[0], f)
        pel = np.array(list(sc.g1_hw["layout"]["pelvis"]) + [0.])
        diag.update(g1_left=(tuple(pel + (-1.25, 0.25, 1.35)), tuple(pel + (0.12, 0.35, 0.95)), 26.),
                    g1_back=(tuple(pel + (-0.35, -1.25, 1.75)), tuple(pel + (0.15, 0.35, 0.95)), 26.))
    target = (getattr(sc, "task_meta", {}) or {}).get("target_slot")
    loc = next((L for L in sc.labels if L["kind"] == "location" and L["text"] == target), None)
    if loc is not None:                                                  # the scan zone in front of the counted slot
        c = np.asarray(loc["center"], float)
        diag.update(scan_zone=(tuple(c + (.34, -.42, .12)), tuple(c + (0., -.13, -.01)), 30.),
                    scan_zone_under=(tuple(c + (-.10, -.36, -.16)), tuple(c + (0., -.14, .0)), 32.))
    seat = np.asarray(sc.cradle["seat"], float)                       # close-ups of the scanner in its cradle
    if preset == "counter":                     # over the cradled scanner's shoulder, across its window: the pass zone
        diag.update(presentation=(tuple(seat + (.15, .23, .41)), tuple(seat + (0., -.22, .065)), 45.))
    diag.update(cradle=(tuple(seat + (.42, -.30, .22)), tuple(seat + (0., 0., .04)), 40.),
                cradle_top=(tuple(seat + (-.06, .16, .34)), tuple(seat + (0., 0., .03)), 36.),
                cradle_side=(tuple(seat + 0.42*np.asarray(_side_dir(sc))), tuple(seat + (0., 0., .05)), 45.))
    if preset == "library":                     # the cradled scanner from above the patron's side of the desk
        diag.update(cradle_far=(tuple(seat + (.20, .55, .40)), tuple(seat + (0., 0., .04)), 40.))
    for name, (eye, target, f) in diag.items():
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); out[name] = "diagnostic"
    return out


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, world, folder, scene, *, width=1280, height=720, samples=24, scan_samples=None):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, BCS_RENDER_PORT=str(listener.getsockname()[1]), BCS_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        scan_samples = scan_samples or world.config.render_samples
        try:
            self.process = subprocess.Popen([executable, "-m", "barcode_scanning.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples),
                                             "--scan-samples", str(scan_samples)],
                                            cwd=root, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept(); self.connection.settimeout(600)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)

    def _call(self, message):
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            send(self.connection, message)
            deadline = time.monotonic() + 600
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
        if not result.get("ok") or result.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(result)[:400])
        return result

    def observe(self, cameras, body_q, tick, lights=None, visibility=None, textures=None, resolution=None):
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(body_q, float).tolist(), cameras=list(cameras),
                   lights=lights or {}, visibility=visibility or {}, textures=textures or {})
        if resolution:
            msg["resolution"] = list(resolution)
        r = self._call(msg)
        self.last_timings = r.get("timings", {})
        return {k: base64.b64decode(v) for k, v in r["images"].items()}

    def scan(self, subframes, tick, lights, visibility=None, samples=None, max_bounces=None):
        """Linear radiance (H, W, 3) float32 of the imager, averaged over the sub-exposure pose sets; `samples` per pixel
        for the whole frame (default: the worker's scan samples); path-tracing depth `max_bounces` (default
        params.RENDER_IMAGER_MAX_BOUNCES)."""
        msg = dict(op="scan", tick=int(tick), camera="imager", resolution=[P.WIDTH_PX, P.HEIGHT_PX],
                   subframe_poses=[np.asarray(b, float).tolist() for b in subframes], lights=lights, visibility=visibility or {},
                   max_bounces=int(max_bounces or P.RENDER_IMAGER_MAX_BOUNCES))
        if samples:
            msg["spp"] = int(samples)
        r = self._call(msg)
        self.last_timings = r.get("timings", {})
        return np.frombuffer(base64.b64decode(r["radiance"]), np.float16).reshape(r["shape"]).astype(np.float32)

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

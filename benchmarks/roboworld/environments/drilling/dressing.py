"""Visual-only dressing of the basement maintenance workshop (no collision, no physics): floor finish and safety tape,
ceiling LED troffers and their light, the drill-press task light, stock shelving, tool chest on its roller cabinet,
electrical panel, extinguisher, signs and small tools, and the other station's equipment where the preset does not
build it. The room's services and context: a sprinkler main with branch lines and heads, a copper water line, EMT
conduit runs with junction boxes and a disconnect switch, a galvanised supply duct with registers, a steel door, a
painted dado band and rubber cove base; at the bench a pegboard with hand tools on hooks, a bench vise at the far end,
a multi-outlet strip and a caged hanging light; beside the wall bay steel shelving with stock, cement bags, a crate of
anchor boxes and a ladder. Nothing is placed where an arm, a tool or a part moves: wall-bay context stays beyond
|x| > 1.9 m below 2.2 m (the arms reach |x| < 1.88 m and z < 2.13 m), bench context beyond x > 3.9 m above the bench
or at its far (north) end, out of the arms' 0.86 m reach (procedural models: visual_models.py).

The component benches (preset `probe`) get the same workshop's walls and ceiling, and a welded steel test stand
under each fixed test block (visual only: in the physics the block is welded to the world).

Everything a robot can touch is built in scene.py / stations.py; models and textures are pinned CC0 Poly Haven assets
(visual_assets.py). Lighting: six 2 x 4 ft LED troffers (4000 K) whose luminance gives 500 lx on the work surfaces
(EN 12464-1: 500 lx for metal working / fine bench work), a task light over the drill press and a weak ambient dome.
The Dresser class (asset references re-bound to OmniPBR) follows wine_pouring/dressing.py.
"""

import math
import re
import shutil
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import visual_assets as va

TARGET_LUX = 500.
TROFFERS = [(x, y) for x in (-2.4, 0.0, 2.4) for y in (-1.3, 1.3)]
TROFFER_SIZE = (1.2, 0.6)
CEILING_Z = 3.0
KELVIN = 4000.


def troffer_luminance(reference):
    """Rect-light luminance (nit) giving TARGET_LUX at `reference` from all troffers: E = sum L A cos_l cos_s / r^2."""
    ref = np.asarray(reference, float)
    A = TROFFER_SIZE[0]*TROFFER_SIZE[1]
    g = 0.
    for x, y in TROFFERS:
        v = np.array([x, y, CEILING_Z - 0.04]) - ref
        r = float(np.linalg.norm(v))
        c = max(0., v[2]/r)
        g += A*c*c/r**2
    return TARGET_LUX/g


class Dresser:
    def __init__(self, S):
        self.S, self.stage = S, S.stage
        self.credits = set()
        self.folder = S.folder
        self.tex = self.folder/"textures"

    def pbr(self, name, tex_name, tile_m, roughness=.6, tint=None):
        maps = va.texture(tex_name)
        if tex_name not in va.PROCEDURAL:              # generated textures carry no third-party credit
            self.credits.add(tex_name)
        return self.S.textured(name, maps, tile_m, roughness, tint=tint)

    def _copy_textures(self, asset, usd):
        if asset not in self.credits:
            for f in (usd.parent/"textures").glob("*"):
                if not (self.tex/f.name).exists():
                    shutil.copy2(f, self.tex/f.name)
        self.credits.add(asset)

    def ref(self, path, asset, pos, yaw=0., scale=None):
        """Reference a pinned model (visual only) at a world pose with a yaw, re-bound to OmniPBR."""
        q = Rotation.from_euler("z", yaw).as_quat()
        return self.ref_parts(path, asset, pos, q, scale or 1., None)

    def ref_parts(self, path, asset, pos, quat, scale=1., prims=None):
        """Reference a pinned model under `path` at (pos, quat xyzw) in its parent's frame; `prims` keeps only those
        children of the model's root (the parts a tool's moving bodies do not draw)."""
        from pxr import Usd
        usd = va.model(asset)
        self._copy_textures(asset, usd)
        S = self.S
        outer = S.UsdGeom.Xform.Define(self.stage, path)
        S.place(outer, pos, quat, scale=(scale, scale, scale) if scale != 1. else None)
        inner = S.UsdGeom.Xform.Define(self.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), f"/{asset}")
        for p in Usd.PrimRange(inner.GetPrim()):
            if p.GetTypeName() == "BasisCurves":
                p.SetActive(False)
        if prims is not None:
            for child in inner.GetPrim().GetChildren():
                if child.GetName() not in prims and child.GetName() != "_materials":
                    child.SetActive(False)
        self._rebind(path, asset, usd)
        return outer

    def _rebind(self, path, asset, usd):
        from pxr import Usd
        S = self.S
        UsdShade, T, Sdf = S.UsdShade, S.Sdf.ValueTypeNames, S.Sdf
        root = self.stage.GetPrimAtPath(path)
        made = {}
        for prim in Usd.PrimRange(root):
            if not prim.IsA(S.UsdGeom.Gprim):
                continue
            src, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
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
                inputs = {}
                for tag, inp in (("_diff", "diffuse_texture"), ("_rough", "reflectionroughness_texture"),
                                 ("_nor_gl", "normalmap_texture"), ("_metal", "metallic_texture")):
                    f = pick(tag)
                    if f is not None and f.exists():
                        inputs[inp] = (T.Asset, Sdf.AssetPath(str(f.resolve())))
                if "reflectionroughness_texture" in inputs:
                    inputs["reflection_roughness_texture_influence"] = (T.Float, 1.)
                if "metallic_texture" in inputs:
                    inputs["metallic_texture_influence"] = (T.Float, 1.)
                made[key] = S.mdl(f"pbr_{asset}_{key}", "OmniPBR", inputs)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(made[key])
        for sp in [prim.GetPath() for prim in Usd.PrimRange(root) if prim.GetName() == "_materials"]:
            prim = self.stage.GetPrimAtPath(sp)
            if prim:
                prim.SetActive(False)

    def draw(self, path, parts, p=None, q=None):
        """Procedural parts (visual_models) under `path` (render.draw_parts: crease normals, UVs, materials)."""
        from . import render as RD
        return RD.draw_parts(self.S, path, parts, self, p, q)

    def ref_R(self, path, asset, pos, R, scale=1., prims=None):
        """Reference a pinned model with a full rotation matrix."""
        q = Rotation.from_matrix(np.asarray(R, float)).as_quat()
        return self.ref_parts(path, asset, pos, q, scale, prims)

    def fit(self, path, asset, R, scale=1., prims=None, **at):
        """Reference a model rotated by R and scaled, translated so that its rotated bounds meet `at` (keys x0, x1,
        y0, y1, z0, z1 = a face of the bounds; xc, yc, zc = a centre). Bounds from the model's own geometry."""
        lo, hi = _bounds(asset, tuple(prims) if prims else None)
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        w = (corners*scale) @ np.asarray(R, float).T
        mn, mx = w.min(0), w.max(0)
        t = np.zeros(3)
        for k, ax in enumerate("xyz"):
            if ax + "0" in at:
                t[k] = at[ax + "0"] - mn[k]
            elif ax + "1" in at:
                t[k] = at[ax + "1"] - mx[k]
            elif ax + "c" in at:
                t[k] = at[ax + "c"] - 0.5*(mn[k] + mx[k])
        return self.ref_R(path, asset, t, R, scale, prims)

    def quad(self, path, corners, mat, uv_per_m=1.):
        c = np.asarray(corners, float)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        uv = np.array([[0, 0], [l1, 0], [l1, l2], [0, l2]])*uv_per_m
        return self.S.mesh(path, c, [0, 1, 2, 0, 2, 3], uv=uv, normals=[n]*4, mat=mat)

    # ------------------------------------------------------------------ the room
    def dress(self, world):
        S = self.S
        L = world.scene.layout
        R = P.ROOM
        floor = self.pbr("floor_finish", "concrete_floor_worn_001", 3.0, .65)
        self.quad("/Dressing/floor", [(R["x0"], R["y0"], 0.), (R["x1"], R["y0"], 0.), (R["x1"], R["y1"], 0.),
                                      (R["x0"], R["y1"], 0.)], floor, uv_per_m=1/3.)
        if L["preset"] == "probe":
            # the component benches build no room (scene.room stops before the walls), so their films showed a blue
            # void above the dressing: draw the same workshop's walls and ceiling (visual only)
            t = 0.2
            wx, wy, h = R["x1"] - R["x0"], R["y1"] - R["y0"], R["h"]
            paint = self.pbr("probe_wall_paint", "painted_plaster_wall", 2.5, .85)
            conc = self.pbr("probe_wall_concrete", "concrete", 2.4, .9)
            ceil = self.pbr("probe_ceiling", "painted_plaster_wall", 3.0, .9)
            for name, (p, size, mat) in dict(
                    south=((0., R["y0"] - 0.5*t, 0.5*h), (wx + 2*t, t, h), paint),
                    north=((0., R["y1"] + 0.5*t, 0.5*h), (wx + 2*t, t, h), conc),
                    west=((R["x0"] - 0.5*t, 0., 0.5*h), (t, wy, h), paint),
                    east=((R["x1"] + 0.5*t, 0., 0.5*h), (t, wy, h), paint),
                    ceiling=((0., 0., h + 0.5*t), (wx + 2*t, wy + 2*t, t), ceil)).items():
                S.box(f"/Dressing/probe_{name}", size, mat, p=p, q=(0., 0., 0., 1.), uv_per_m=1.)
            # a fixed test block stands in the air in the physics (welded to the world): draw the welded steel stand a
            # test rig would carry it on (visual only; its top plate is cut out under the block, so whatever leaves
            # the block's underside - a bit, a bolt - passes freely, as in the physics)
            from . import visual_models as VM
            boxes = []
            for d in world.scene.drillables.values():
                if not d.static:
                    continue
                Rm = Rotation.from_quat(world.bq[d.body][3:]).as_matrix()
                ext = np.abs(Rm) @ np.asarray(d.half, float)
                c = np.asarray(world.bq[d.body][:3], float)
                boxes.append((d.name, c - ext, c + ext))
            for name, lo, hi in boxes:
                below = any(o != name and lo2[2] < lo[2] and hi2[2] > lo[2] - 0.01 and lo2[0] < hi[0] and hi2[0] > lo[0]
                            and lo2[1] < hi[1] and hi2[1] > lo[1] for o, lo2, hi2 in boxes)
                if lo[2] > 0.05 and not below:
                    self.draw(f"/Dressing/test_stand_{name}", VM.test_stand(lo[:2], hi[:2], float(lo[2])))
        tape = S.color("floor_tape", (.75, .55, .02), .6, 0.)
        # yellow floor tape around the two docking boxes
        for name, (x0, x1, y0, y1) in dict(dock_w=(-1.1, 1.1, 1.65, 2.55), dock_b=(2.25, 3.15, -1.70, -0.50)).items():
            for k, (a, bb) in enumerate((((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)))):
                a, bb = np.array([*a, 0.0008]), np.array([*bb, 0.0008])
                d = bb - a
                nrm = np.array([-d[1], d[0], 0.])/np.linalg.norm(d)*0.025
                self.quad(f"/Dressing/tape_{name}_{k}", [a - nrm, bb - nrm, bb + nrm, a + nrm], tape)
        # ceiling troffers (model + a downward rect light each)
        lum = troffer_luminance((0.0, 2.5, 1.2))
        for k, (x, y) in enumerate(TROFFERS):
            self.ref(f"/Dressing/troffer_{k}", "mounted_fluorescent_lights", (x, y, CEILING_Z), 0.)
            lt = S.UsdLux.RectLight.Define(self.stage, f"/Lights/troffer_{k}")
            lt.CreateWidthAttr(TROFFER_SIZE[0])
            lt.CreateHeightAttr(TROFFER_SIZE[1])
            lt.CreateIntensityAttr(float(lum))
            lt.CreateEnableColorTemperatureAttr(True)
            lt.CreateColorTemperatureAttr(KELVIN)
            lt.GetPrim().CreateAttribute("visibleInPrimaryRay", S.Sdf.ValueTypeNames.Bool).Set(True)
            S.place(lt, (x, y, CEILING_Z - 0.045), (0., 0., 0., 1.))      # RectLight emits along its -z: downwards
        dome = S.UsdLux.DomeLight.Define(self.stage, "/Lights/ambient")
        dome.CreateIntensityAttr(25.)
        dome.CreateColorAttr(S.Gf.Vec3f(0.9, 0.92, 1.0))
        # drill-press task light (an industrial wall lamp on the east wall above the press)
        self.ref("/Dressing/press_lamp", "industrial_wall_lamp", (3.95, -1.45, 1.95), math.pi/2)
        sl = S.UsdLux.SphereLight.Define(self.stage, "/Lights/press_task")
        sl.CreateRadiusAttr(0.04)
        sl.CreateIntensityAttr(9000.)
        sl.CreateEnableColorTemperatureAttr(True)
        sl.CreateColorTemperatureAttr(3500.)
        S.place(sl, (3.80, -1.45, 1.80), (0., 0., 0., 1.))
        # west wall: stock shelving with boxes and crates, extinguisher by the door; south wall: electrical panel
        self.ref("/Dressing/shelves_a", "steel_frame_shelves_01", (-3.70, 0.6, 0.), math.pi/2)
        self.ref("/Dressing/shelves_b", "steel_frame_shelves_01", (-3.70, -0.6, 0.), math.pi/2)
        for k, (y, z) in enumerate(((0.75, 0.42), (0.35, 0.42), (-0.45, 1.13), (-0.8, 0.42))):
            self.ref(f"/Dressing/box_{k}", "cardboard_box_01", (-3.68, y, z), math.pi/2 + 0.1*k, scale=0.8)
        self.ref("/Dressing/toolbox", "metal_toolbox", (-3.68, -0.55, 0.42), math.pi/2)
        self.ref("/Dressing/crate", "wooden_crate_01", (-3.45, -2.2, 0.), 0.3)
        self.ref("/Dressing/extinguisher", "korean_fire_extinguisher_01", (-3.85, -2.75, 0.), math.pi/2)
        self.ref("/Dressing/panel", "power_box_01", (1.2, -2.80, 1.55), math.pi)
        self.ref("/Dressing/broom", "plastic_broom", (1.75, 2.85, 0.), math.pi)
        self.ref("/Dressing/sign", "WetFloorSign_01", (-1.55, 2.25, 0.), -0.4)
        # the top chest stands on its roller cabinet (on the floor its feet hung 0.18 m in the air: the model's
        # origin is 0.176 m below its base)
        from . import visual_models as VM
        self.draw("/Dressing/roller_cabinet", VM.roller_cabinet((3.56, 1.55, 0.), -math.pi/2))
        self.fit("/Dressing/tool_chest", "metal_tool_chest", Rotation.from_euler("z", -math.pi/2).as_matrix(),
                 xc=3.565, yc=1.55, z0=0.84, prims=("metal_tool_chest", "metal_tool_chest_lid", "metal_tool_chest_lock",
                                                    "metal_tool_chest_handle_left", "metal_tool_chest_handle_right",
                                                    "metal_tool_chest_hinge_lid", "metal_tool_chest_hinge_chest"))
        if L["preset"] == "bench":
            self.ref("/Dressing/tape", "measuring_tape_01", (3.85, 0.30, P.BENCH_TOP_Z), 0.4)
            # both models stand on end in their files: laid flat on the bench (rotated about x)
            lay = lambda yaw: (Rotation.from_euler("z", yaw)*Rotation.from_euler("x", math.pi/2)).as_quat()
            self.ref_parts("/Dressing/hammer", "cross_pein_hammer", (3.88, 0.10, P.BENCH_TOP_Z + 0.0145), lay(1.2), 1., None)
            self.ref_parts("/Dressing/wrench", "adjustable_wrench", (3.86, -0.05, P.BENCH_TOP_Z + 0.0155), lay(0.2), 1., None)
        else:
            # the bench station seen from the wall bay: the workbench, the press cabinet and the drill press as dressing
            # (not built here), drawn by the same models as preset `bench`
            self.draw("/Dressing/workbench", VM.workbench((3.24, 4.0), (-1.18, 0.65), P.BENCH_TOP_Z))
            self.draw("/Dressing/press_cabinet", VM.press_cabinet((0.21, 0.15, 0.5*P.PRESS_STAND_M)),
                      (3.52, -1.45, 0.5*P.PRESS_STAND_M), (0., 0., 1., 0.))
            qz = Rotation.from_euler("z", math.pi*1.5).as_quat()
            self.ref_parts("/Dressing/drill_press", "drill_press_01", (3.55, -1.45, P.PRESS_STAND_M), qz, 1.,
                           None)
        self.services(L)
        self.bench_context(L)
        if L["preset"] == "wall":
            self.wall_context(L)
        return self.credits

    # ------------------------------------------------------------------ room services and finishes
    def services(self, L):
        from . import visual_models as VM
        R = P.ROOM
        S = self.S
        # walls: painted dado band (the wall paint darkened) and a rubber cove base on the block walls
        dado = self.pbr("dado_paint", "painted_plaster_wall", 2.5, .75, tint=(0.42, 0.45, 0.47))
        for name, (a, b) in dict(s=((R["x0"], R["y0"] + 0.0015), (R["x1"], R["y0"] + 0.0015)),
                                 w=((R["x0"] + 0.0015, R["y1"]), (R["x0"] + 0.0015, R["y0"])),
                                 e=((R["x1"] - 0.0015, R["y0"]), (R["x1"] - 0.0015, R["y1"]))).items():
            self.quad(f"/Dressing/dado_{name}", [(*b, 0.10), (*a, 0.10), (*a, 1.10), (*b, 1.10)], dado, uv_per_m=1/2.5)
            S.box(f"/Dressing/dado_line_{name}", (abs(b[0] - a[0]) or 0.004, abs(b[1] - a[1]) or 0.004, 0.012),
                  self.S.color("dado_line", (.05, .07, .09), .5, 0.), p=(0.5*(a[0] + b[0]), 0.5*(a[1] + b[1]), 1.106),
                  q=(0, 0, 0, 1))
        segs = [((R["x0"], R["y0"], 0.), (R["x1"], R["y0"], 0.), (0., 1., 0.)),
                ((R["x0"], R["y0"], 0.), (R["x0"], R["y1"], 0.), (1., 0., 0.)),
                ((R["x1"], R["y0"], 0.), (R["x1"], R["y1"], 0.), (-1., 0., 0.)),
                ((R["x0"], R["y1"], 0.), (R["x1"], R["y1"], 0.), (0., -1., 0.))]
        self.draw("/Dressing/cove_base", VM.wall_base(segs))
        # steel door in the south wall by the electrical panel, light switch beside it
        self.draw("/Dressing/door", VM.steel_door(2.45, 3.35, R["y0"], normal=1.))
        self.draw("/Dressing/light_switch", VM.junction_box((2.22, R["y0"], 1.20), (0., 1., 0.), size=0.075, depth=0.04,
                                                            mat="white_paint"))
        self.draw("/Dressing/door_conduit", VM.pipe_run([(2.22, R["y0"] + 0.012, 1.24), (2.22, R["y0"] + 0.012, 2.98)],
                                                         0.0105, "emt_galv"))
        # sprinkler main along the north wall (100 mm, red), two branch lines with pendent heads, a gate valve
        yn = R["y1"]
        zm = 2.70
        self.draw("/Dressing/sprinkler_main", VM.pipe_run([(R["x0"], yn - 0.14, zm), (R["x1"], yn - 0.14, zm)], 0.057,
                                                           "pipe_red", flanges=[(x, yn - 0.14, zm) for x in
                                                                                (-2.0, 0.6, 3.2)]))
        for k, x in enumerate((-3.6, -1.5, 1.4, 3.6)):
            self.draw(f"/Dressing/main_bracket_{k}", VM.strut((x, yn - 0.0001, zm - 0.0775), (x, yn - 0.24, zm - 0.0775))
                      + VM.pipe_clamp((x, yn - 0.14, zm), 0.057, (1., 0., 0.), (0., 0., -1.)))
        self.draw("/Dressing/main_valve", VM.gate_valve((-3.0, yn - 0.14, zm), (1., 0., 0.), 0.057, (0., -0.6, -0.8)))
        for k, x in enumerate((-1.2, 1.2)):
            zb = 2.86
            self.draw(f"/Dressing/branch_{k}", VM.pipe_run([(x, yn - 0.14, zm), (x, yn - 0.14, zb), (x, -1.70, zb)],
                                                            0.021, "pipe_red", bend=0.05))
            for j, y in enumerate((1.75, 0.0, -1.45)):
                self.draw(f"/Dressing/sprinkler_{k}_{j}", _sprinkler_head((x, y, zb)))
            for j, y in enumerate((2.3, 0.85, -0.75)):
                self.draw(f"/Dressing/branch_hanger_{k}_{j}",
                          VM.pipe_run([(x, y, zb + 0.026), (x, y, R["h"])], 0.0048, "zinc")
                          + VM.pipe_clamp((x, y, zb), 0.021, (0., 1., 0.), (0., 0., 1.), mat="zinc"))
        # copper water line and an EMT conduit run along the north wall top
        self.draw("/Dressing/copper_line", VM.pipe_run([(R["x0"], yn - 0.060, 2.50), (R["x1"], yn - 0.060, 2.50)],
                                                        0.0143, "copper", flanges=[(x, yn - 0.060, 2.50)
                                                                                   for x in (-2.6, -0.2, 2.2)]))
        for k, x in enumerate(np.arange(R["x0"] + 0.5, R["x1"], 1.2)):
            self.draw(f"/Dressing/copper_clip_{k}", VM.pipe_clamp((x, yn - 0.060, 2.50), 0.0143, (1., 0., 0.),
                                                                  (0., 1., 0.), mat="copper")
                      + VM.pipe_run([(x, yn - 0.060 + 0.020, 2.50), (x, yn, 2.50)], 0.004, "copper"))
        self.draw("/Dressing/emt_north", VM.pipe_run([(R["x0"] + 0.3, yn - 0.022, 2.95), (R["x1"] - 0.25, yn - 0.022,
                                                                                               2.95)], 0.0105, "emt_galv",
                                                      flanges=[(x, yn - 0.022, 2.95) for x in (-2.0, 1.0)]))
        # supply duct along the south side of the ceiling with two side registers
        self.draw("/Dressing/duct", VM.duct(R["x0"], R["x1"], -2.2, R["h"] - 0.02, registers=[(-1.6, 1), (1.6, 1)]))

    def bench_context(self, L):
        """East wall over the bench: pegboard with hand tools on hooks, outlet strip and its conduit, a bench vise at
        the far (north) end, a caged hanging light over the bench, stock on the lower shelf (both presets)."""
        from . import visual_models as VM
        R = P.ROOM
        xw = R["x1"]
        parts, xf = VM.pegboard(-1.10, 0.62, 1.00, 1.92, xw)
        self.draw("/Dressing/pegboard", parts)
        # tools: (asset, prims, model axis hung vertical, model thin axis -> +x (into the board), y centre, z top, scale)
        tools = [("combination_wrench", None, "y", "z", -0.06, 1.80, 1.00),
                 ("combination_wrench", None, "y", "z", 0.00, 1.80, 0.86),
                 ("combination_wrench", None, "y", "z", 0.055, 1.80, 0.74),
                 ("pliers", None, "z", "y", -0.20, 1.80, 1.0),
                 ("tongue_groove_pliers", None, "y", "z", -0.30, 1.82, 1.0),
                 ("pipe_wrench", None, "z", "y", 0.47, 1.84, 1.0),
                 ("screwdrivers_02", ("screwdrivers_02_a",), "y", "z", 0.20, 1.76, 1.0),
                 ("screwdrivers_02", ("screwdrivers_02_b",), "y", "z", 0.245, 1.73, 1.0),
                 ("flathead_screwdriver", None, "z", "x", 0.30, 1.76, 1.0),
                 ("handsaw_wood", None, "y", "x", -0.70, 1.86, 1.0),
                 ("measuring_tape_01", None, "z", "x", -0.42, 1.52, 1.0)]
        for k, (asset, prims, up, thin, yc, zt, sc_) in enumerate(tools):
            R_ = _axes(up, thin) if asset != "handsaw_wood" else np.eye(3)
            self.fit(f"/Dressing/peg_tool_{k}", asset, R_, sc_, prims, x1=xf - 0.004, yc=yc, z1=zt)
            self.draw(f"/Dressing/peg_hook_{k}", VM.peg_hook(xf, yc, zt - 0.03, length=0.04))
        # multi-outlet strip along the bench back, fed by conduit from the ceiling
        self.draw("/Dressing/outlet_strip", VM.outlet_strip((xw - 0.001, -1.05, 0.93), (xw - 0.001, -0.25, 0.93),
                                                            (-1., 0., 0.)))
        self.draw("/Dressing/outlet_conduit", VM.pipe_run([(xw - 0.012, -1.06, 0.95), (xw - 0.012, -1.16, 0.95),
                                                           (xw - 0.012, -1.16, 2.98)], 0.0105, "emt_galv", bend=0.04))
        # bench vise bolted at the far (north) end of the bench, its fixed jaw at the front edge
        # its round swivel base (r 0.10 m about the model origin, measured on the mesh) 0.16 m from the edge: the jaws
        # overhang the edge and the screw bar (0.20 m in front of the base centre) hangs clear of it
        vx, vy = 3.40, 0.43
        self.ref("/Dressing/bench_vise", "bench_vice_01", (vx, vy, P.BENCH_TOP_Z), -math.pi/2)
        for k, a in enumerate((math.pi/2, math.pi*7/6, math.pi*11/6)):
            self.draw(f"/Dressing/vise_bolt_{k}", _bolt_head((vx + 0.093*math.cos(a + math.pi), vy + 0.093*math.sin(a + math.pi),
                                                              P.BENCH_TOP_Z + 0.008)))
        # stock on the lower shelf at the far end, spray cans at the bench's back corner
        self.ref("/Dressing/shelf_toolbox", "metal_toolbox", (3.62, -0.15, 0.211), -math.pi/2, scale=0.95)
        self.ref("/Dressing/shelf_crate", "plastic_crate_02", (3.62, -0.78, 0.211), math.pi/2)
        self.draw("/Dressing/shelf_crate_boxes", VM.anchor_boxes((3.62, -0.78, 0.226), 3, 4, yaw=math.pi/2))
        self.ref("/Dressing/bench_spray", "lubricant_spray", (3.93, 0.52, P.BENCH_TOP_Z), 2.4)
        self.ref("/Dressing/bench_oil", "small_oil_can_01", (3.90, 0.36, P.BENCH_TOP_Z), math.pi*0.9)
        # caged hanging light over the bench on its chains (switched off: the room's 500 lx comes from the troffers)
        self.fit("/Dressing/caged_light", "caged_hanging_light", Rotation.from_euler("z", math.pi/2).as_matrix(), 1.0,
                 None, xc=3.55, yc=-0.20, z1=R["h"])

    def wall_context(self, L):
        """Context beside the wall bay, beyond the arms' reach (|x| > 1.9 m): steel shelving with stock in the
        north-west corner, cement bags and a crate of anchor boxes, a ladder against the west wall, a disconnect
        switch with its conduit and junction box, a caged light over the corner (preset `wall`)."""
        from . import visual_models as VM
        R = P.ROOM
        yn = R["y1"]
        self.ref("/Dressing/bay_shelves", "steel_frame_shelves_01", (-3.30, yn - 0.27, 0.), 0.)
        for k, (x, z, yaw, sc_) in enumerate(((-3.62, 0.42, 0.05, 0.75), (-3.05, 0.42, -0.08, 0.8),
                                             (-3.55, 1.13, 0.02, 0.7))):
            self.ref(f"/Dressing/bay_box_{k}", "cardboard_box_01", (x, yn - 0.27, z), yaw, scale=sc_)
        self.ref("/Dressing/bay_crate_shelf", "plastic_crate_02", (-3.0, yn - 0.27, 1.13), 0.)
        self.draw("/Dressing/bay_crate_shelf_boxes", VM.anchor_boxes((-3.0, yn - 0.27, 1.145), 4, 4))
        self.ref("/Dressing/bay_oil", "oil_tin", (-3.62, yn - 0.30, 1.84), 0.3)
        self.ref("/Dressing/bay_spray", "spray_paint_bottles", (-3.20, yn - 0.30, 1.84), 0.1)
        # cement bags on the floor against the wall, a crate of anchor boxes in front of them
        for k, (x, y, z, yaw) in enumerate(((-2.30, yn - 0.38, 0.0, math.pi/2), (-2.30, yn - 0.38, 0.172, math.pi/2 + 0.06),
                                            (-2.27, yn - 0.40, 0.344, math.pi/2 - 0.05))):
            self.ref(f"/Dressing/cement_{k}", "cement_bag", (x, y, z), yaw)
        self.ref("/Dressing/anchor_crate", "plastic_crate_03", (-2.40, yn - 0.95, 0.), 0.25)
        self.draw("/Dressing/anchor_crate_boxes", VM.anchor_boxes((-2.40, yn - 0.95, 0.015), 4, 3, yaw=0.25))
        # aluminium ladder leaning against the west wall
        Rl = Rotation.from_euler("y", -14., degrees=True).as_matrix() @ Rotation.from_euler("z", math.pi/2).as_matrix()
        self.fit("/Dressing/ladder", "ladder_sectioned_01", Rl, 1.0, None, x0=R["x0"] + 0.012, yc=1.95, z0=0.)
        # disconnect switch with conduit up to a junction box under the ceiling
        self.draw("/Dressing/disconnect", VM.disconnect_switch((-2.30, yn, 1.45), (0., -1., 0.)))
        self.draw("/Dressing/disconnect_conduit", VM.pipe_run([(-2.30, yn - 0.022, 1.64), (-2.30, yn - 0.022, 2.90)],
                                                             0.0105, "emt_galv"))
        self.draw("/Dressing/disconnect_jbox", VM.junction_box((-2.30, yn, 2.948), (0., -1., 0.)))
        self.draw("/Dressing/bay_outlet", VM.junction_box((-1.98, yn, 0.45), (0., -1., 0.), size=0.075, depth=0.045,
                                                          mat="box_grey"))
        self.draw("/Dressing/bay_outlet_conduit", VM.pipe_run([(-1.98, yn - 0.022, 0.49), (-1.98, yn - 0.022, 1.0),
                                                              (-2.22, yn - 0.022, 1.15), (-2.22, yn - 0.022, 1.28)],
                                                             0.0105, "emt_galv", bend=0.05))
        self.fit("/Dressing/bay_caged_light", "caged_hanging_light", np.eye(3), 1.0, None, xc=-2.9, yc=2.25, z1=R["h"])
        # east of the bay (seen from the room, beyond the right arm's reach): a rolling tool cart, stacked crates, a
        # first-aid cabinet and an outlet with its conduit
        self.fit("/Dressing/bay_tool_cart", "tool_cart", np.eye(3), 1.0, None, x0=2.55, y1=yn - 0.04, z0=0.)
        for k, (x, z, yaw) in enumerate(((2.08, 0.0, 0.0), (2.08, 0.254, 0.04), (2.10, 0.508, -0.03))):
            self.ref(f"/Dressing/bay_crates_{k}", "plastic_crate_02", (x, yn - 0.24, z), math.pi/2 + yaw)
        self.draw("/Dressing/first_aid", _first_aid((2.25, yn, 1.45), (0., -1., 0.)))
        self.draw("/Dressing/bay_outlet_e", VM.junction_box((3.10, yn, 1.25), (0., -1., 0.), size=0.075, depth=0.045))
        self.draw("/Dressing/bay_outlet_e_conduit", VM.pipe_run([(3.10, yn - 0.022, 1.29), (3.10, yn - 0.022, 2.95)],
                                                                0.0105, "emt_galv"))


@lru_cache(maxsize=None)
def _bounds(asset, prims=None):
    """(min, max) of a pinned model's geometry in its own frame (optionally only some children of its root)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(va.model(asset)))
    root = st.GetPrimAtPath("/" + asset)
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    if prims:
        rngs = [bc.ComputeWorldBound(root.GetChild(p)).ComputeAlignedRange() for p in prims if root.GetChild(p)]
        lo = np.min([list(r.GetMin()) for r in rngs], axis=0)
        hi = np.max([list(r.GetMax()) for r in rngs], axis=0)
    else:
        r = bc.ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = np.array(list(r.GetMin())), np.array(list(r.GetMax()))
    return np.asarray(lo, float), np.asarray(hi, float)


def _axes(up, thin):
    """Rotation taking the model axis `up` to world +z and the model axis `thin` to world +x (a tool hung flat)."""
    idx = {"x": 0, "y": 1, "z": 2}
    img = [None, None, None]
    su = -1. if up.startswith("-") else 1.
    st = -1. if thin.startswith("-") else 1.
    img[idx[up[-1]]] = su*np.array([0., 0., 1.])
    img[idx[thin[-1]]] = st*np.array([1., 0., 0.])
    k = [i for i in range(3) if img[i] is None][0]
    a, b = [(k + 1) % 3, (k + 2) % 3]
    img[k] = np.cross(img[a], img[b])
    return np.column_stack(img)


def _sprinkler_head(p):
    """Pendent sprinkler head under a branch line: drop nipple, frame arms, red glass bulb, deflector."""
    from . import visual_models as VM
    M = VM.Model()
    x, y, z = p
    M.add(VM.cyl((x, y, z - 0.012), (x, y, z - 0.045), 0.009, 0.009, 16), "pipe_red", "nipple")
    M.add(VM.hexagon((x, y, z - 0.045), (x, y, z - 0.058), 0.022), "copper", "body")
    for sgn in (-1, 1):
        M.add(VM.cyl((x + sgn*0.007, y, z - 0.058), (x + sgn*0.007, y, z - 0.082), 0.0025, 0.0025, 8), "copper", "frame")
    M.add(VM.cyl((x, y, z - 0.060), (x, y, z - 0.078), 0.0028, 0.0028, 10), "estop_red", "bulb")
    M.add(VM.cyl((x, y, z - 0.082), (x, y, z - 0.084), 0.016, 0.016, 24), "copper", "deflector")
    return M.parts()


def _bolt_head(p):
    """A hex bolt head with washer (the bench vise's hold-down bolts)."""
    from . import visual_models as VM
    M = VM.Model()
    x, y, z = p
    M.add(VM.cyl((x, y, z), (x, y, z + 0.0025), 0.012, 0.012, 24), "zinc", "washer")
    M.add(VM.hexagon((x, y, z + 0.0025), (x, y, z + 0.010), 0.017), "zinc", "head")
    return M.parts()


def _first_aid(p, normal):
    """Wall-mounted first-aid cabinet: white steel box, green cross."""
    from . import visual_models as VM
    M = VM.Model()
    n = np.asarray(normal, float)
    u = np.array([0., 0., 1.])
    sd = np.cross(u, n)
    R = np.column_stack([sd, u, n])
    M.add(VM.rbox((0., 0., 0.06), (0.30, 0.38, 0.12), 0.008), "white_paint", "box", crease=35)
    M.add(VM.box((0., 0.03, 0.1205), (0.11, 0.035, 0.002)), "aid_green", "cross")
    M.add(VM.box((0., 0.03, 0.1205), (0.035, 0.11, 0.002)), "aid_green", "cross")
    M.add(VM.rbox((0.12, -0.08, 0.125), (0.02, 0.06, 0.012), 0.004, 6), "plastic_grey", "latch")
    return M.parts(R, p)

"""Physical instrument printing; no task-answer overlay or phase colour."""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def strokes(stage, name, paths, width, material):
    curve = UsdGeom.BasisCurves.Define(stage, name)
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(p) for p in paths])
    curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.concatenate(paths).astype(np.float32)))
    curve.CreateWidthsAttr([width])
    curve.SetWidthsInterpolation("constant")
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)


def cylinder_text(stage, name, text, origin, radius, height, material, angle_offset=0.):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    glyph = TextPath((0, 0), text, size=height, prop=FontProperties(family="DejaVu Sans"))
    center = (glyph.get_extents().x0+glyph.get_extents().x1)/2
    paths = []
    for contour in glyph.to_polygons():
        zs = origin[2]+contour[:, 1]
        rs = radius(zs) if callable(radius) else radius
        angle = (contour[:, 0]-center)/rs+angle_offset
        paths.append(np.c_[origin[0]+rs*np.sin(angle), origin[1]-rs*np.cos(angle), zs])
    strokes(stage, name, paths, height/18, material)


def add_markings(stage, lab, mats, output):
    c = lab.config
    x, y, vz = c.valve_position
    radius = c.burette_outer_radius+.000025
    dz_ml = 1e-6/(np.pi*c.burette_inner_radius**2)
    zero = vz+.048+50*dz_ml
    paths = []
    for tenth in range(501):
        z = zero-tenth*dz_ml/10
        width = .004 if tenth % 10 == 0 else .0025 if tenth % 5 == 0 else .0014
        angle = np.linspace(-width/(2*radius), width/(2*radius), 9)
        paths.append(np.c_[x+radius*np.sin(angle), y-radius*np.cos(angle), np.full(len(angle), z)])
    strokes(stage, "/Lab/burette_scale", paths, .000070, mats["ink"])
    for ml in range(0, 51, 5):
        cylinder_text(stage, f"/Lab/scale_label_{ml}", str(ml),
                       (x, y, zero-ml*dz_ml+.001), radius, .0023, mats["ink"])
    cylinder_text(stage, "/Lab/burette_id", "50 mL 20 C", (x, y, zero+.012), radius, .0015, mats["ink"])
    # Labels are attached to their rigid vessel frame, so they move with it.
    from .label_art import reagent_label, cylinder_label

    for name, formula, title in (("acid_bottle", "HCl", "Hydrochloric acid"), ("base_bottle", "NaOH", "Sodium hydroxide")):
        body = lab.apparatus.bodies[name]
        texture = reagent_label(output/"textures", name, formula, title, .100)
        cylinder_label(stage, f"/Lab/body_{body}/label", .02908, .021, .073, .058, texture)
    flask = lab.apparatus.bodies["flask"]
    cylinder_text(stage, f"/Lab/body_{flask}/capacity", "125 mL", (0., 0., .033),
                   lambda z: np.interp(z, [.015, .040], [.0335, .029])+.000025, .0028, mats["ink"], angle_offset=.70)
    # Graduation heights follow the modeled cavity, not a fitted fluid level.
    from scipy.integrate import quad
    from scipy.optimize import brentq

    heights = np.array([.002, .003, .007, .015, .040, .067, .086, .091, .114])
    inner = np.array([.025, .029, .032, .032, .0275, .0205, .0135, .0125, .0125])
    outer_z = np.array([0., .002, .006, .015, .040, .067, .085, .091, .110, .112, .114])
    outer_r = np.array([.025, .031, .0335, .0335, .029, .022, .015, .014, .014, .015, .015])
    for ml in (50, 75, 100, 125):
        def volume(z):
            return quad(lambda h: np.pi*np.interp(h, heights, inner)**2,
                        .002, z, points=heights[(heights > .002) & (heights < z)], epsabs=1e-12)[0]
        z = brentq(lambda h: volume(h)-ml*1e-6, .002, .114)
        r = np.interp(z, outer_z, outer_r)+.000035
        angle = np.linspace(-.9, -.30, 21)
        line = np.c_[r*np.sin(angle), -r*np.cos(angle), np.full(len(angle), z)]
        strokes(stage, f"/Lab/body_{flask}/graduation_{ml}", [line], .00011, mats["ink"])
        cylinder_text(stage, f"/Lab/body_{flask}/volume_{ml}", str(ml), (0., 0., z+.0005),
            lambda z: np.interp(z, outer_z, outer_r)+.000035, .0026, mats["ink"])
    pipette = lab.apparatus.bodies["volumetric_pipette"]
    from .pipette_geometry import MARK_Z_M
    t = np.linspace(0., 2*np.pi, 129)
    strokes(stage, f"/Lab/body_{pipette}/calibration_line", [np.c_[.002835*np.cos(t), .002835*np.sin(t),
            np.full(len(t), MARK_Z_M)]], .00015, mats["ink"])

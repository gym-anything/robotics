"""Construction details bound to the moving cut panels, never world-fixed.

Thread and fastener appearance remains render-only, as in the baseline showcase.
Sewn constraints and lumped construction mass are owned by the physical model.
"""
import numpy as np


class Details:
    def __init__(self, env, stage):
        from pxr import Gf, UsdGeom, UsdShade
        from ..laundry_scene import _omnipbr_material
        from ..garment_asset import FeatureKind
        self.curves, self.buttons = {}, {}
        self.triangles = env.sim.inputs.physical_triangles
        self.panels = env.sim.topology.triangle_panel_index
        self.panel_names = list(env.sim.topology.panel_ids)
        pattern, topology = env.sim.pattern_mesh, env.sim.topology
        construction = env.sim.construction
        denim = construction.id == "denim_five_pocket_jeans"
        thread = _omnipbr_material(stage, "/Materials/detail_thread",
            (.93, .55, .16) if denim else (.86, .87, .83), roughness=.8)
        button = _omnipbr_material(stage, "/Materials/detail_button", (.90, .93, .92), roughness=.28)
        brass = _omnipbr_material(stage, "/Materials/detail_brass", (.62, .38, .12), roughness=.3, metallic=.9)
        def path_ids(ref):
            ids = pattern.path_vertices[(ref.panel_id, ref.path_id)]
            fraction = pattern.path_fractions[(ref.panel_id, ref.path_id)]
            ids = ids[(fraction >= ref.start_fraction-1e-12) & (fraction <= ref.end_fraction+1e-12)]
            if ref.reverse:
                ids = ids[::-1]
            ids = topology.material_to_physical_vertex[ids]
            keep = np.r_[True, ids[1:] != ids[:-1]]
            return ids[keep]
        def curve(path, ref, width):
            ids = path_ids(ref)
            if len(ids) < 2:
                return
            g = UsdGeom.BasisCurves.Define(stage, path)
            g.CreateTypeAttr("linear"); g.CreateWrapAttr("nonperiodic")
            g.CreateCurveVertexCountsAttr([len(ids)])
            g.CreatePointsAttr([Gf.Vec3f(0.)]*len(ids))
            g.CreateWidthsAttr([width]); g.SetWidthsInterpolation("constant")
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(thread)
            self.curves[path] = (ids, self.panel_names.index(ref.panel_id))
        root = "/Lab/Garment/Construction"
        for i, seam in enumerate(construction.seams):
            if not any(s in seam.side_a.panel_id for s in ("pocket_inner", "pocket_outer", "fly_shield", "fly_facing")):
                curve(f"{root}/seam_{i}", seam.side_a, .00135 if denim else .00085)
        for i, feature in enumerate(construction.features):
            if feature.kind == FeatureKind.DOBBY_BAND:
                continue  # finite-width material-space bands are in the textile shader
            for j, ref in enumerate(feature.path_refs):
                curve(f"{root}/feature_{i}_{j}", ref, min(feature.width_m or .002, .0032))
        from ..garment_asset import PathRef
        fasteners = []
        if construction.id == "woven_button_down_shirt":
            fasteners.append((PathRef("right_front", "button_line"), 7, .0052, button))
        if denim:
            fasteners.extend((PathRef(f"front_{side}", "pocket_opening"), 2, .0032, brass) for side in ("left", "right"))
            fasteners.append((PathRef("front_left_waistband", "waist_opening"), 1, .007, brass))
            fasteners.append((PathRef("front_left", "rise", .45, 1.), 12, .0015, brass))
        number = 0
        for ref, count, radius, material in fasteners:
            ids = path_ids(ref)
            # Choose attachments in undeformed material-path distance once.
            rest = topology.physical_positions_m[ids]
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(rest, axis=0), axis=1))]
            for distance in np.linspace(.08, .92, count)*length[-1]:
                i = min(int(np.searchsorted(length, distance, side="right")-1), len(ids)-2)
                u = (distance-length[i])/max(length[i+1]-length[i], 1e-12)
                path = f"{root}/fastener_{number}"
                number += 1
                g = UsdGeom.Cylinder.Define(stage, path)
                g.CreateRadiusAttr(radius); g.CreateHeightAttr(.0022); g.CreateAxisAttr("Z")
                xform = UsdGeom.Xformable(g)
                xform.AddTranslateOp(); xform.AddOrientOp()
                UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
                self.buttons[path] = (ids[i:i+2], np.array([1-u, u]), self.panel_names.index(ref.panel_id))
        apply(stage, self.sample(env.state.particle_q.numpy()))

    def sample(self, positions):
        normals = {}
        for p in {p for _,p in self.curves.values()} | {p for _,_,p in self.buttons.values()}:
            triangles = self.triangles[self.panels == p]
            n = np.zeros_like(positions, dtype=float)
            face = np.cross(positions[triangles[:, 1]]-positions[triangles[:, 0]], positions[triangles[:, 2]]-positions[triangles[:, 0]])
            for corner in range(3):
                np.add.at(n, triangles[:, corner], face)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            normals[p] = n
        curves = {path:(positions[ids]+normals[p][ids]*.0018).tolist() for path,(ids,p) in self.curves.items()}
        buttons = {}
        for path,(ids,w,p) in self.buttons.items():
            n = w@normals[p][ids]
            n /= max(np.linalg.norm(n), 1e-12)
            # Quaternion taking the cylinder's local +Z into the cloth normal.
            q = np.r_[np.cross([0., 0., 1.], n), 1+n[2]]
            if np.linalg.norm(q) < 1e-8:
                q = np.array([1., 0., 0., 0.])
            q /= np.linalg.norm(q)
            buttons[path] = dict(position=(w@positions[ids]+n*.0021).tolist(), quaternion=q.tolist())
        return dict(curves=curves, buttons=buttons)


def attribute_values(payload):
    from pxr import Gf, Vt
    values = {}
    for path, points in payload.get("curves", {}).items():
        points = np.asarray(points, dtype=np.float32)
        if not np.isfinite(points).all():
            raise ValueError("Non-finite construction detail")
        values[path+".points"] = Vt.Vec3fArray.FromNumpy(points)
    for path, data in payload.get("buttons", {}).items():
        p, q = data["position"], data["quaternion"]
        if not np.isfinite([*p, *q]).all():
            raise ValueError("Non-finite fastener pose")
        values[path+".xformOp:translate"] = Gf.Vec3d(*p)
        values[path+".xformOp:orient"] = Gf.Quatf(q[3], Gf.Vec3f(*q[:3]))
    return values


def apply(stage, payload):
    for path, value in attribute_values(payload).items():
        stage.GetAttributeAtPath(path).Set(value)

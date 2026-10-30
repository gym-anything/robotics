"""Render-only terry yarn loops attached to the actual panel triangles.

The 3 mm declared textile envelope includes the pile: the base is recessed to
0.25 mm on either side and yarn centre lines stay within 1.35 mm of the physical
mid-surface. No yarn geometry is fed into dynamics, collision or task scoring.
Loop placement is deterministic and saved with the renderer source snapshot.
"""
import numpy as np


class TerryPile:
    def __init__(self, env, stage):
        from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt
        import matplotlib.tri as mtri
        self.entries = {}
        pattern, top = env.sim.pattern_mesh, env.sim.topology
        for p, panel in enumerate(env.sim.construction.panels):
            if panel.material_profile != "terry_cotton_prior":
                continue
            idx = pattern.panel_ids.index(panel.id)
            mt = pattern.triangles[pattern.triangle_panel_index == idx]
            uv = pattern.material_coordinates_m
            used = np.unique(mt)
            local = np.full(len(uv), -1, int); local[used] = np.arange(len(used))
            finder = mtri.Triangulation(uv[used, 0], uv[used, 1], local[mt]).get_trifinder()
            rng = np.random.default_rng(150513+p)
            # 3 mm placement pitch; variation breaks the artificial oval grid.
            u, v = np.meshgrid(np.arange(.028, .422, .003), np.arange(.028, .672, .003))
            centres = np.c_[u.ravel(), v.ravel()]
            centres += rng.uniform(-.00065, .00065, centres.shape)
            centres = centres[(abs(centres[:, 1]-.075) > .019) & (abs(centres[:, 1]-.625) > .019)]
            n = len(centres)
            theta = np.linspace(0., np.pi, 7)
            angle = rng.uniform(-.65, .65, n)
            axis = np.c_[np.cos(angle), np.sin(angle)]
            cross = np.c_[-axis[:, 1], axis[:, 0]]
            # Arch rises from the cloth and leans in a varying weave direction.
            span = rng.uniform(.00035, .00060, n)
            lean = rng.uniform(-.0005, .0005, n)
            sample_uv = centres[:, None, :]+axis[:, None, :]*span[:, None, None]*np.cos(theta)[None, :, None]
            sample_uv += cross[:, None, :]*lean[:, None, None]*np.sin(theta)[None, :, None]
            flat = sample_uv.reshape(-1, 2)
            triangle = finder(flat[:, 0], flat[:, 1])
            if np.any(triangle < 0):
                raise ValueError("Pile sample outside its physical panel")
            material = uv[mt[triangle]]
            matrix = np.stack((material[:, 1]-material[:, 0], material[:, 2]-material[:, 0]), axis=2)
            weights12 = np.linalg.solve(matrix, (flat-material[:, 0])[..., None])[..., 0]
            weights = np.c_[1-weights12.sum(1), weights12]
            physical = top.material_to_physical_vertex[mt[triangle]]
            heights = (.00025+rng.uniform(.00065, .00105, n)[:, None]*np.sin(theta)).ravel()
            material_path = f"/Materials/terry_yarn_{p}"
            mat = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path+"/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*panel.display_color_rgb))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.92)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for side in (-1, 1):
                path = f"/Lab/Garment/pile_{p}_{'front' if side == 1 else 'back'}"
                curve = UsdGeom.BasisCurves.Define(stage, path)
                curve.CreateTypeAttr("linear")
                curve.CreateWrapAttr("nonperiodic")
                curve.CreateCurveVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n, 7, np.int32)))
                curve.CreateWidthsAttr([.00024])
                curve.SetWidthsInterpolation("constant")
                curve.CreatePointsAttr()
                curve.CreateExtentAttr()
                UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(mat)
                self.entries[path] = physical, weights, heights*side
        for path, points in self.sample(env.state.particle_q.numpy()).items():
            stage.GetAttributeAtPath(path+".points").Set(Vt.Vec3fArray.FromNumpy(points))
            stage.GetAttributeAtPath(path+".extent").Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], np.float32)))

    def sample(self, particles):
        result = {}
        for path, (indices, weights, heights) in self.entries.items():
            triangle = particles[indices]
            normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            result[path] = (np.sum(triangle*weights[..., None], axis=1)+normal*heights[:, None]).astype(np.float32)
        return result

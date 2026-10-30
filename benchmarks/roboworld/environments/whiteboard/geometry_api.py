"""Teacher-only metric collision geometry, with bounded mesh pagination."""

import numpy as np
from scipy.spatial import ConvexHull


class GeometryView:
    def __init__(self, env):
        self.env = env
        self.hulls = {}

    def describe(self):
        from .deformable_contact import render_meshes
        from .elasticity import tool_solid

        m = self.env.solver.mj_model
        mapping = self.env.solver.mjc_body_to_newton.numpy()[0]
        return {
            "elastic_materials": {
                name: {
                    **tool_solid(data["kind"]).metadata(),
                    "vertices_local_m": data["vertices"].tolist(),
                    "triangles": data["faces"].tolist(),
                    "tetrahedra": tool_solid(data["kind"]).tetra.tolist(),
                    "contact_scope": "Board/nib/felt and cap/barrel interior. Other contacts use listed rigid proxies.",
                }
                for name, data in render_meshes(self.env, full=True).items()
            },
            "units": "meters, kilograms, seconds",
            "quaternion_order": "wxyz",
            "frame": "Geometry poses are body-local. /v1/state body poses are world xyzw.",
            "bodies": [
                {
                    "id": i,
                    "name": m.body(i).name,
                    "state_body_label": self.env.model.body_label[int(mapping[i])]
                    if int(mapping[i]) >= 0
                    else None,
                    "mass_kg": float(m.body_mass[i]),
                    "com_local_m": m.body_ipos[i].tolist(),
                    "principal_inertia_kg_m2": m.body_inertia[i].tolist(),
                    "inertia_rotation_wxyz": m.body_iquat[i].tolist(),
                }
                for i in range(m.nbody)
            ],
            "geometries": [
                {
                    "id": i,
                    "name": m.geom(i).name,
                    "body_id": int(m.geom_bodyid[i]),
                    "type": int(m.geom_type[i]),
                    "size": m.geom_size[i].tolist(),
                    "position_local_m": m.geom_pos[i].tolist(),
                    "rotation_wxyz": m.geom_quat[i].tolist(),
                    "mesh_id": int(m.geom_dataid[i]) if int(m.geom_type[i]) == 7 else None,
                    "friction": m.geom_friction[i].tolist(),
                }
                for i in range(m.ngeom)
            ],
            "geometry_types": {
                "0": "plane",
                "1": "heightfield",
                "2": "sphere",
                "3": "capsule",
                "4": "ellipsoid",
                "5": "cylinder",
                "6": "box",
                "7": "convex mesh",
            },
            "size_convention": "Sphere: radius; capsule/cylinder: radius and half-length; box: half-extents",
            "meshes": [
                {
                    "id": i,
                    "vertices": int(m.mesh_vertnum[i]),
                    "triangles_source": int(m.mesh_facenum[i]),
                }
                for i in range(m.nmesh)
            ],
            "mesh_route": "POST /v1/geometry/mesh, {mesh_id,array: vertices|hull_triangles,start,limit}",
            "contact_note": "Nib/pad-porcelain use the documented compliant patches instead of rigid proxy contact.",
        }

    def mesh(self, mesh_id, array, start=0, limit=1000):
        m = self.env.solver.mj_model
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (mesh_id, start, limit)):
            raise ValueError("Mesh id/start/limit must be integers")
        if not 0 <= mesh_id < m.nmesh or start < 0 or not 1 <= limit <= 10000:
            raise ValueError("Invalid mesh range; limit must be in [1,10000]")
        adr, count = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
        vertices = np.asarray(m.mesh_vert[adr : adr + count])
        if array == "vertices":
            values = vertices
        elif array == "hull_triangles":
            if mesh_id not in self.hulls:
                self.hulls[mesh_id] = ConvexHull(vertices).simplices
            values = self.hulls[mesh_id]
        else:
            raise ValueError("array must be vertices or hull_triangles")
        if start > len(values):
            raise ValueError("start exceeds array length")
        end = min(start + limit, len(values))
        return {
            "mesh_id": mesh_id,
            "array": array,
            "start": start,
            "next_start": end,
            "total": len(values),
            "complete": end == len(values),
            "values": values[start:end].tolist(),
        }

"""Exact Newton body-local geometry, including hollow liquid collision meshes."""

import hashlib
import numpy as np


class GeometryCatalog:
    def __init__(self, world):
        from .runtime import require_newton
        newton, _ = require_newton()
        self.objects, self.assets = {}, {}
        from .usd_lab import PALETTE
        b, m = world.lab.builder, world.model
        for i in range(b.shape_count):
            source = b.shape_source[i]
            asset = None
            if source is not None and hasattr(source, "vertices"):
                v, f = np.asarray(source.vertices), np.asarray(source.indices).reshape(-1, 3)
                asset = hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()
                self.assets[asset] = {"asset_id": asset, "vertices_local_m": v.tolist(),
                    "triangles": f.tolist(), "apply_shape_scale": True}
            self.objects[str(i)] = {"id": str(i), "name": b.shape_label[i],
                "body_id": b.shape_body[i], "type": newton.GeoType(b.shape_type[i]).name,
                "pose_local_xyzw": np.asarray(b.shape_transform[i]).tolist(),
                "scale": np.asarray(b.shape_scale[i]).tolist(), "asset_id": asset,
                "collides_with_rigids": bool(b.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)),
                "collides_with_liquid": bool(b.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)),
                "visible": bool(b.shape_flags[i] & int(newton.ShapeFlags.VISIBLE))}
            material = world.lab.materials.get(i)
            if material in PALETTE:
                color, roughness, metallic = PALETTE[material]
            else:
                source_color = getattr(source, 'color', None)
                color = np.asarray(source_color if source_color is not None else b.shape_color[i])[:3]
                roughness, metallic = .35, 0.
            self.objects[str(i)]['preview_material'] = {'name': material,
                'linear_rgb': np.asarray(color).tolist(), 'roughness': roughness, 'metallic': metallic,
                'transparent': material in ('glass', 'water'),
                'qualification': 'Operator preview only. RTX camera materials/textures are exported separately.'}
            analytic = getattr(b, 'titration_analytic_solids', {}).get(i)
            if analytic is not None:
                kind, dimensions = analytic
                self.objects[str(i)]['fluid_boundary'] = {
                    'type': 'cylinder_y_minus_bore_z' if kind == 1 else 'annular_socket_y_minus_ports_z',
                    'inner_or_plug_radius_m': dimensions[0], 'outer_radius_m': dimensions[1] if kind == 2 else dimensions[0],
                    'half_length_y_m': dimensions[2], 'bore_radius_m': dimensions[3],
                    'frame': 'shape local; same dimensional source as the tessellated asset',
                    'field': 'Exact CSG solid sign; conservative distance bound at Boolean intersections'}
        mass, com, inertia = m.body_mass.numpy(), m.body_com.numpy(), m.body_inertia.numpy()
        self.bodies = [{"id": i, "name": name, "mass_kg": float(mass[i]),
            "com_local_m": com[i].tolist(), "inertia_local_kg_m2": inertia[i].tolist()}
            for i, name in enumerate(m.body_label)]

    def index(self):
        return {"units": "metres, kilograms, seconds", "quaternion_order": "xyzw",
            "bodies": self.bodies, "geometries": list(self.objects.values()),
            "assets": [{"id": k, "vertex_count": len(v["vertices_local_m"]),
                "triangle_count": len(v["triangles"])} for k, v in self.assets.items()],
            "size_convention": "BOX: half extents; CYLINDER/CAPSULE: radius and half length along local Z; mesh: component scale",
            "cavity_note": "Whole hollow meshes are liquid boundaries/visuals; separate convex wall cells handle rigid contacts."}

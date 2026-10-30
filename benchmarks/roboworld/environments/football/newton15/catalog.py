"""Privileged collision geometry; exact scale/frame, never exposed to policy."""

import hashlib
import numpy as np


class Catalog:
    def __init__(self, w):
        self.w = w
        self.objects, self.assets = {}, {}
        b = w.builder
        for name, body in {"world": -1, **w.body_ids}.items():
            shapes = []
            for i, owner in enumerate(b.shape_body):
                if owner != body:
                    continue
                source = b.shape_source[i]
                item = {
                    "id": i,
                    "label": b.shape_label[i],
                    "kind": w.n.GeoType(b.shape_type[i]).name,
                    "scale_m": np.asarray(b.shape_scale[i]).tolist(),
                    "local_pose_xyz_xyzw": np.asarray(b.shape_transform[i]).tolist(),
                    "flags": int(b.shape_flags[i]),
                }
                if source is not None and hasattr(source, "vertices"):
                    vertices = np.asarray(source.vertices) * np.asarray(
                        b.shape_scale[i]
                    )
                    faces = np.asarray(source.indices).reshape(-1, 3)
                    key = hashlib.sha256(
                        vertices.tobytes() + faces.tobytes()
                    ).hexdigest()
                    self.assets[key] = {
                        "id": key,
                        "units": "m",
                        "vertices": vertices.tolist(),
                        "triangles": faces.tolist(),
                    }
                    item["mesh_asset_id"] = key
                shapes.append(item)
            self.objects[name] = {
                "id": name,
                "body_id": body,
                "frame": "body local, metres; quaternions XYZW",
                "mass_kg": 0.0 if body < 0 else float(w.model.body_mass.numpy()[body]),
                "com_local_m": None if body < 0 else w.com[body].tolist(),
                "inertia_com_kg_m2": None
                if body < 0
                else w.model.body_inertia.numpy()[body].tolist(),
                "shapes": shapes,
            }

    def index(self):
        return {
            "schema": "football-geometry-v1",
            "objects": [
                {"id": k, "body_id": v["body_id"]} for k, v in self.objects.items()
            ],
            "net_ids": ["net_" + name for name in self.w.nets],
            "asset_count": len(self.assets),
            "units": "m,kg,s",
            "scale_convention": "BOX=half-extents xyz; SPHERE=radius x; CYLINDER/CAPSULE=radius x,half-height y; mesh vertices already scaled",
            "custom_contact": "right shoe pressure shell; pitch; cable ball/net",
            "collision_filter_pairs": [
                list(pair) for pair in self.w.builder.shape_collision_filter_pairs
            ],
        }

    def get(self, key):
        if key.startswith("net_"):
            n = self.w.nets[key[4:]]
            return {
                "id": key,
                "frame": "world",
                "tick": self.w.tick,
                "vertices_m": n.position.tolist(),
                "edges": np.stack([n.edge_a, n.edge_b], axis=1).tolist(),
                "cord_diameter_m": n.prior.cord_diameter_m,
                "node_mass_kg": n.mass.tolist(),
                "anchors": np.flatnonzero(n.anchored).tolist(),
            }
        return self.objects[key]

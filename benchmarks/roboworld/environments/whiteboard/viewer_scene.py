"""Diagnostic WebGL scene from the same object and imported robot visual geometry.

The browser renders poses only. It never simulates or interpolates robot motion.
"""

import hashlib
import numpy as np
from .usd_scene import PALETTE


def describe(env):
    meshes, objects = {}, []
    from .deformable_contact import render_meshes

    deformable = render_meshes(env)

    def mesh(vertices, faces):
        points = np.asarray(vertices, np.float32)
        index = np.asarray(faces, np.int32).reshape(-1)
        key = hashlib.sha256(points.tobytes() + index.tobytes()).hexdigest()
        if key not in meshes:
            meshes[key] = {"vertices": points.reshape(-1).tolist(), "indices": index.tolist()}
        return key

    for obj in env.objects:
        parts = []
        for p in obj.parts:
            if obj.name in deformable and (p.name in ("nib", "pad") or p.name.startswith("wall_")):
                continue
            item = {
                "name": p.name,
                "kind": p.kind,
                "size": p.size,
                "position": p.position,
                "rotation": p.rotation,
                "material": p.material,
            }
            if p.kind == "mesh":
                item["mesh"] = mesh(p.vertices, p.faces)
            parts.append(item)
        if obj.name in deformable:
            data = deformable[obj.name]
            parts.append(
                {
                    "name": "elastic_material",
                    "kind": "mesh",
                    "mesh": mesh(data["vertices"], data["faces"]),
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0, 1],
                    "material": "felt" if obj.name == "eraser" else obj.name.split("_", 1)[1],
                }
            )
        objects.append(
            {
                "name": obj.name,
                "body": env.body_ids.get(obj.name, -1),
                "position": obj.position,
                "rotation": obj.rotation,
                "parts": parts,
            }
        )
    b = env.builder
    for i in range(len(env.shape_ids), b.shape_count):
        if not b.shape_flags[i] & int(env.newton.ShapeFlags.VISIBLE) or b.shape_body[i] < 0:
            continue
        source = b.shape_source[i]
        if source is None or not hasattr(source, "vertices"):
            continue
        local = np.asarray(b.shape_transform[i])
        label = b.body_label[b.shape_body[i]]
        material = "paint_white" if np.asarray(b.shape_color[i]).mean() > 0.40 else "charcoal"
        objects.append(
            {
                "name": f"robot_{i}",
                "body": int(b.shape_body[i]),
                "position": [0, 0, 0],
                "rotation": [0, 0, 0, 1],
                "parts": [
                    {
                        "name": label,
                        "kind": "mesh",
                        "mesh": mesh(
                            np.asarray(source.vertices) * np.asarray(b.shape_scale[i]),
                            source.indices,
                        ),
                        "position": local[:3].tolist(),
                        "rotation": local[3:].tolist(),
                        "material": material,
                    }
                ],
            }
        )
    return {
        "objects": objects,
        "meshes": meshes,
        "materials": PALETTE,
        "board": {
            "width": env.config.board.writing_width,
            "height": env.config.board.writing_height,
            "face_y": env.config.board.face_y,
            "center_z": env.config.board.center_z,
        },
        "renderer": "Diagnostic WebGL, authored physical geometry; policy RGB uses separate RTX renderer",
    }


def state(env):
    from .ik import HandIK
    from .deformable_contact import render_meshes

    ik = HandIK(env)
    return {
        "tick": env.tick,
        "ready": env.healthy,
        "time_s": env.tick * env.config.physics_dt,
        "poses": env.state_0.body_q.numpy().tolist(),
        "joints": env.proprioception()["joints"],
        "left": ik.pose("left"),
        "right": ik.pose("right"),
        "ink_revision": env.ink.revision,
        "elastic_vertices": {
            name + "/elastic_material": data["vertices"].reshape(-1).tolist()
            for name, data in render_meshes(env).items()
        },
    }

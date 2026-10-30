"""Reuse the original mesh construction without importing an Isaac application.

The AST adapter changes only the old script-style sibling import. The original
geometry implementation and original Isaac entry points remain unmodified.
"""

import ast
from functools import lru_cache
from pathlib import Path
from types import ModuleType
import numpy as np
from .runtime import require_engine


@lru_cache(maxsize=1)
def legacy_mesh():
    path = Path(__file__).resolve().parents[1] / "lock_mesh.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "lock_specs":
            tree.body[i] = ast.ImportFrom(module="lockkey", names=node.names, level=0)
            matches += 1
    if matches != 1:
        raise RuntimeError("Legacy mesh import changed; review geometry adapter")
    module = ModuleType("lockkey.newton15._legacy_mesh")
    module.__file__ = str(path)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def mesh(tm):
    newton, _ = require_engine()
    if not tm.is_watertight or tm.volume <= 0:
        raise ValueError("Collision solid must have a closed, outward surface")
    return newton.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).ravel())


def add_mesh(builder, body, tm, *, label, cfg, transform=None, convex=True, color=(.64, .43, .12)):
    # Legacy convex colliders are defined by their vertex hulls. Reconstructing
    # that hull also closes the original revolved pin's duplicate pole topology.
    if convex:
        tm = tm.convex_hull
    fn = builder.add_shape_convex_hull if convex else builder.add_shape_mesh
    return fn(body, mesh=mesh(tm), xform=transform, cfg=cfg, color=color, label=label)


def set_mass(builder, body, mass):
    old = builder.body_mass[body]
    if old <= 0 or mass <= 0:
        raise ValueError("Dynamic solid must have positive mass")
    builder.body_inertia[body] = builder.body_inertia[body] * float(mass / old)
    builder.body_mass[body] = float(mass)


def key_parts(c, bitting):
    parts, _ = legacy_mesh().make_key_blade_prisms([int(d) for d in bitting],
        length=.030, spine_z=c.spine_z, width=2*c.keyway_half_width-c.blade_clearance,
        flat_root=None, root_tare=c.root_tare)
    flip = np.array([[-1., 0, 0, c.face_x+c.insertion_travel],
                     [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]])
    for part in parts:
        part.apply_transform(flip)
    return parts


def complete_plug(c):
    """Cylinder with five round blind bores and an open-bottom narrow keyway.

    No collision walls are omitted to accommodate old PhysX convex cooking.
    The bore floors are the same ledge plane, not bitting-shaped supports.
    """
    import trimesh
    from lockkey import lock_specs as LK
    lm = legacy_mesh()
    solid = lm._cyl_x(LK.PLUG_R, c.plug_length, sections=96)
    height = LK.PLUG_R+.001-c.ledge_z
    holes = [lm._cyl_z(LK.PIN_R+c.bore_clearance, height, cx=x,
                       cz=c.ledge_z+height/2) for x in c.chambers_x]
    low = -LK.PLUG_R-.001
    slot = trimesh.creation.box(extents=(c.plug_length+.002, 2*c.keyway_half_width, c.keyway_top-low))
    slot.apply_translation((0., 0., (low+c.keyway_top)/2))
    return solid.difference([*holes, slot])

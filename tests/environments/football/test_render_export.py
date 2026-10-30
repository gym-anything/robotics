"""USD export of the task scenes with a stand-in world: no Newton, no Isaac, no rendering.

goal (no props or markers) must take exactly the base export path: task_scene is never called and no task
material or marker is authored. The other tasks' props and markers must author only valid, plain-typed values.
The export tests need the export's own dependencies (usd-core, scipy, pillow, trimesh) and are skipped without them;
the value-type check of variants.layout needs nothing.
"""

from enum import IntEnum, IntFlag
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from football.newton15 import variants

EXPORT_DEPENDENCIES = all(importlib.util.find_spec(m) for m in ("pxr", "scipy", "PIL", "trimesh"))
BASE_MATERIALS = {"white", "dark", "cord", "steel", "grass", "line", "path", "bench"}


class GeoType(IntEnum):
    SPHERE = 1
    BOX = 2
    CAPSULE = 3
    CYLINDER = 4


class ShapeFlags(IntFlag):
    VISIBLE = 1


def stand_in_world(task, seed=0):
    """The fields render.export reads, with the frame, a ball, two robot links and the task's props added the way
    FootballWorld adds them (static boxes on body -1 labelled prop/<name>)."""
    layout = variants.layout(task, seed)
    shapes = []

    def add(body, label, kind, scale, position, quat=(0.0, 0.0, 0.0, 1.0), color=(0.8, 0.8, 0.8)):
        shapes.append((body, label, kind, np.asarray(scale, np.float32), np.asarray((*position, *quat), np.float32),
                       color))

    add(-1, "pitch", GeoType.BOX, (18.5, 12.0, 0.10), (-15.5, 0.0, -0.10))
    for y in (-3.72, 3.72):
        add(-1, f"goal_post_{y}", GeoType.CYLINDER, (0.06, 1.22, 0.0), (0.0, y, 1.22))
    add(0, "ball/shell", GeoType.SPHERE, (variants.R, 0.0, 0.0), (0.0, 0.0, 0.0))
    add(1, "robot/pelvis", GeoType.SPHERE, (0.07, 0.0, 0.0), (0.0, 0.0, 0.0), color=(0.2, 0.2, 0.2))
    add(2, "robot/torso_link", GeoType.BOX, (0.10, 0.15, 0.20), (0.0, 0.0, 0.1))
    prop_materials = {}
    for prop in layout["props"]:
        label = "prop/" + prop["name"]
        half = prop["yaw_rad"] / 2
        add(-1, label, GeoType.BOX, prop["half_extents_m"], prop["position_m"], (0.0, 0.0, math.sin(half), math.cos(half)))
        prop_materials[label] = prop["material"]
    builder = SimpleNamespace(
        shape_count=len(shapes), shape_flags=[int(ShapeFlags.VISIBLE)] * len(shapes),
        shape_body=[s[0] for s in shapes], shape_label=[s[1] for s in shapes], shape_source=[None] * len(shapes),
        shape_scale=[s[3] for s in shapes], shape_type=[s[2] for s in shapes],
        shape_transform=[s[4] for s in shapes], shape_color=[s[5] for s in shapes])
    poses = np.zeros((3, 7), np.float32)
    poses[:, 6] = 1.0
    poses[:, :3] = ((-11.0, 0.0, 0.11), (-11.42, 0.0, 0.80), (-11.42, 0.0, 1.0))
    net = SimpleNamespace(position=np.linspace(0.0, 1.0, 30, dtype=np.float32).reshape(10, 3), edge_a=np.arange(9),
                          edge_b=np.arange(1, 10), prior=SimpleNamespace(cord_diameter_m=0.004))
    net.edges = list(zip(net.edge_a, net.edge_b))
    return SimpleNamespace(builder=builder, n=SimpleNamespace(ShapeFlags=ShapeFlags, GeoType=GeoType), poses=poses,
                           ball_id=0, ball=SimpleNamespace(radius_m=variants.R), nets={"back": net}, robot=True,
                           body_ids={"torso_link": 2}, layout=layout, prop_materials=prop_materials)


class LayoutValueTests(unittest.TestCase):
    def test_props_and_markers_hold_plain_python_values(self):
        def plain(value):
            if isinstance(value, (tuple, list)):
                return bool(value) and all(plain(v) for v in value)
            return type(value) in (float, str)

        for task in variants.TASKS:
            for seed in (0, 7) if task in variants.SEEDED else (0,):
                layout = variants.layout(task, seed)
                for item in layout["props"] + layout["markers"]:
                    for key, value in item.items():
                        with self.subTest(task=task, seed=seed, item=item["name"], key=key):
                            self.assertTrue(plain(value), repr(value))


@unittest.skipUnless(EXPORT_DEPENDENCIES, "needs the render export's dependencies (usd-core, scipy, pillow, trimesh)")
class RenderExportTests(unittest.TestCase):
    def setUp(self):
        import football.newton15.visual_assets as assets

        folder = Path(tempfile.mkdtemp())
        maps = {key: folder / f"{key}.jpg" for key in ("diff", "nor_gl", "rough")}
        for path in [*maps.values(), folder / "sky.hdr"]:
            path.write_bytes(b"stand-in")
        for name, value in (("grass", lambda: maps), ("sky", lambda: folder / "sky.hdr"),
                            ("grass_blades", lambda stage, material: None)):
            patcher = patch.object(assets, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def export(self, world):
        from pxr import Usd
        from football.newton15 import render

        stage = Usd.Stage.Open(str(render.export(world, Path(tempfile.mkdtemp()))))
        return stage

    def test_goal_takes_the_base_export_path(self):
        from football.newton15 import render

        with patch.object(render, "task_scene", side_effect=AssertionError("goal must not export task content")):
            stage = self.export(stand_in_world("goal"))
        materials = {p.GetName() for p in stage.GetPrimAtPath("/Materials").GetChildren()}
        self.assertEqual(materials, BASE_MATERIALS)
        self.assertFalse(stage.GetPrimAtPath("/Lab/task_markers").IsValid())

    def test_task_props_and_markers_author_valid_values(self):
        from pxr import UsdShade

        for task in variants.TASKS[1:]:
            with self.subTest(task=task):
                world = stand_in_world(task)
                stage = self.export(world)
                layout = world.layout
                for i, label in enumerate(world.builder.shape_label):
                    if label in world.prop_materials:
                        prim = stage.GetPrimAtPath(f"/Lab/static_{i}")
                        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                        self.assertEqual(bound.GetPrim().GetName(), world.prop_materials[label])
                for marker in layout["markers"]:
                    prim = stage.GetPrimAtPath("/Lab/task_markers/" + marker["name"])
                    self.assertTrue(prim.IsValid())
                    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    self.assertEqual(bound.GetPrim().GetName(), marker["material"])
                task_prims = [p for p in stage.Traverse() if str(p.GetPath()).startswith("/Lab/task_markers")]
                task_prims += [stage.GetPrimAtPath(f"/Lab/static_{i}")
                               for i, label in enumerate(world.builder.shape_label) if label in world.prop_materials]
                task_prims += [p for name in set(variants_materials(layout))
                               for p in stage.GetPrimAtPath("/Materials/" + name).GetAllChildren()]
                for prim in task_prims:
                    for attribute in prim.GetAttributes():
                        if not attribute.HasAuthoredValue():
                            continue
                        value = attribute.Get()
                        self.assertIsNotNone(value, attribute.GetPath())
                        type_name = attribute.GetTypeName()
                        if type_name.isArray:
                            self.assertGreater(len(value), 0, attribute.GetPath())
                            if str(type_name) not in ("token[]", "string[]", "asset[]"):
                                self.assertTrue(np.isfinite(np.asarray(value, dtype=float)).all(),
                                                attribute.GetPath())
                        elif isinstance(value, float):
                            self.assertTrue(math.isfinite(value), attribute.GetPath())


def variants_materials(layout):
    return [item["material"] for item in layout["props"] + layout["markers"]]


if __name__ == "__main__":
    unittest.main()

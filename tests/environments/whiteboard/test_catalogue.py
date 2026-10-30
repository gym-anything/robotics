"""Constructed pigment/body states only; never a simulator or policy solution."""
import copy
import json
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from whiteboard.config import Board, Config, MaterialPriors
from whiteboard.ink import InkSurface
from whiteboard.tasks import BRIDGE_INSTRUCTION
from whiteboard.task_set.catalogue import (
    CONTRACT, CatalogueTask, NAMES, SELECTIONS, preset_config, raster, resolve,
)
from whiteboard.task_set.presets import seed_board, tool_poses, OPEN_COLORS
from whiteboard.task_set.mechanical import bounds, tool_geometry, tray_measure
from whiteboard.task_set.verifier import evaluate
from tests.packages import PACKAGES

ROOT = PACKAGES


def surface(task=None, start=False, resolution=(2048, 1024)):
    ink = InkSurface(Board(), *resolution, MaterialPriors())
    if task:
        channels = raster(task.strokes(start), ink.board, ink.width, ink.height)
        paint(ink, channels)
    return ink


def paint(ink, channels, density=.2):
    ink.tiles.clear()
    for ty in range((ink.height+127)//128):
        for tx in range((ink.width+127)//128):
            block = channels[ty*128:(ty+1)*128, tx*128:(tx+1)*128]
            if not (block < 6).any():
                continue
            h, w = block.shape
            p = np.zeros((128, 128, 1, 6), np.float32)
            for c in range(6):
                p[:h, :w, 0, c] = (block == c) * density
            ink.tiles[tx, ty] = p, np.full((128, 128, 1), -600., np.float32)
    ink.revision += 1


def state(task, config=None, capped=False):
    config = config or preset_config(task.name)[0]
    poses = tool_poses(config, CatalogueTask("write-text") if capped else task)
    return dict(poses=poses, velocities={n: np.zeros(6) for n in poses}, touching=set(), pelvis_z=.8)


def judge(task, ink, mechanical=None):
    return evaluate(task, ink, preset_config(task.name)[0], mechanical or state(task))


class CatalogueTests(unittest.TestCase):
    def test_native_default_and_filters_are_unchanged(self):
        from whiteboard.benchmark import resolve as native
        for selection in ({}, {"variation_id": 0}, {"variation_id": 1}, {"kind": "text", "selection_seed": 7},
                          {"split": "test", "kind": "bridge", "selection_seed": 9}):
            self.assertEqual(resolve(selection), native(selection))
        self.assertEqual(resolve({})["description"], BRIDGE_INSTRUCTION)
        self.assertEqual(resolve({"task": "bridge"}), native({"variation_id": 0}))
        for selection in ({"variation_id": True}, {"selection_seed": "2"}, {"kind": "invalid"}):
            with self.assertRaises((ValueError, TypeError)):
                resolve(selection)

    def test_exact_eight_unique_selections_and_explicit_versions(self):
        benchmark = json.loads((ROOT / "whiteboard/benchmark.json").read_text())
        group = benchmark["task_set"]
        self.assertEqual(group["selections"], list(SELECTIONS))
        self.assertEqual(len({resolve(s)["task_id"] for s in SELECTIONS}), 8)
        for name in NAMES[:-1]:
            with self.assertRaises(ValueError):
                resolve({"task": name})
        for selection in ({"task": "bad", "contract": CONTRACT}, {"contract": CONTRACT},
                          {"task": "write-text", "contract": "v0"},
                          {"task": "write-text", "contract": CONTRACT, "variation_id": 1}):
            with self.assertRaises(ValueError):
                resolve(selection)

    def test_new_placements_and_primitive_bounds_fit_reach_and_tray(self):
        for name in NAMES[:-1]:
            config, task = preset_config(name)
            task.validate_board(config.board)
            s = state(task, config)
            geometry = tool_geometry(config)
            for body, pose in s["poses"].items():
                lo, hi = bounds(geometry[body], pose)
                self.assertLessEqual(max(abs(lo[0]-config.layout.robot_x), abs(hi[0]-config.layout.robot_x)), .5)
                self.assertTrue(tray_measure(config, body, pose, np.zeros(6), set())["pass"], body)
            # Every pair of new tool footprints has a non-intersecting AABB except its own seated barrel/cap.
            boxes = {body: bounds(geometry[body], pose) for body, pose in s["poses"].items()}
            for a, (alo, ahi) in boxes.items():
                for b, (blo, bhi) in boxes.items():
                    if a >= b or a.removeprefix("marker_").removeprefix("cap_") == b.removeprefix("marker_").removeprefix("cap_"):
                        continue
                    self.assertTrue(np.any(ahi <= blo) or np.any(bhi <= alo), (name, a, b))

    def test_presets_preserve_nonplacement_options(self):
        base = Config(physics_dt=.001, seed=17, texture_width=1000, texture_height=501, episode_ticks=678)
        config, _ = preset_config("write-text", base)
        for field in base.__dataclass_fields__:
            if field != "layout":
                self.assertEqual(getattr(config, field), getattr(base, field))
        self.assertEqual(config.layout.marker_order, base.layout.marker_order)
        self.assertEqual(config.layout.robot_y, base.layout.robot_y)

    def test_front_reference_and_geometry_agree(self):
        for name in ("write-text", "write-in-colour", "correct-misspelling"):
            task, ink = CatalogueTask(name), surface()
            np.testing.assert_array_equal(task.reference_view(ink), task.reference(ink)[:, ::-1])
            glyph0 = [p[0] for s in task.strokes() if s[0] == "letter 0" for p in s[3]]
            last = max(int(s[0].split()[1]) for s in task.strokes())
            glyphlast = [p[0] for s in task.strokes() if s[0] == f"letter {last}" for p in s[3]]
            self.assertGreater(min(glyph0), max(glyphlast))


class PigmentTests(unittest.TestCase):
    def test_native_bridge_untouched_is_zero(self):
        from whiteboard.variations import resolve as native
        task = native(0)[1]
        out = task.evaluate(surface())
        self.assertEqual((out["success"], out["score"], out["progress"]), (False, 0., 0.))
        self.assertEqual(out["revision"], "bridge-copy-v1")

    def test_all_seven_untouched_starts_zero(self):
        for name in NAMES[:-1]:
            with self.subTest(name=name):
                task, ink = CatalogueTask(name), surface()
                seed_board(ink, task)
                out = judge(task, ink)
                self.assertEqual((out["success"], out["score"], out["progress"]), (False, 0., 0.))
                self.assertTrue(out["contract"].startswith(CONTRACT))

    def test_all_seven_constructed_end_states_pass(self):
        for name in NAMES[:-1]:
            with self.subTest(name=name):
                task = CatalogueTask(name)
                out = judge(task, surface(task), state(task, capped=True))
                self.assertEqual((out["success"], out["score"], out["progress"]), (True, 1., 1.), out)

    def test_seed_is_dry_conserved_and_no_marker_reservoir_consumed(self):
        task, ink = CatalogueTask("erase-board"), surface(resolution=(1001, 503))
        initial = dict(ink.reservoir)
        record = seed_board(ink, task)
        self.assertGreater(record["pigment_ul"], 0)
        self.assertEqual(ink.reservoir, initial)
        self.assertEqual(ink.total_deposited_ul, ink.pigment_volume_ul())
        for _, stamp in ink.tiles.values():
            self.assertTrue((stamp == -600).all())
        with self.assertRaises(ValueError):
            seed_board(ink, task)

    def test_text_faint_partial_wrong_color_stray_mirror_and_shift(self):
        for name in ("write-text", "write-in-colour"):
            task, ink = CatalogueTask(name), surface()
            channels = raster(task.strokes(), ink.board, ink.width, ink.height)
            scores = []
            for density in (.03, .05, .07, .09):
                paint(ink, channels, density)
                out = judge(task, ink)
                scores.append(out["score"])
                self.assertFalse(out["success"])
                self.assertAlmostEqual(out["score"], .9 * out["progress"])
            self.assertEqual(scores, sorted(scores))
            self.assertEqual(len(set(scores)), len(scores))
            for altered in (np.where(channels < 6, 1, 255), channels[:, ::-1], np.roll(channels, 80, 1)):
                paint(ink, altered)
                self.assertFalse(judge(task, ink)["success"])
            paint(ink, np.where(channels < 6, 1, 255))
            self.assertEqual(judge(task, ink)["score"], 0)
            altered = channels.copy()
            altered[100:160, 100:160] = 1
            paint(ink, altered)
            self.assertFalse(judge(task, ink)["success"])

    def test_erasing_is_graded_and_new_marks_fail(self):
        task, ink = CatalogueTask("erase-board"), surface()
        seed_board(ink, task)
        scores = []
        for density in (.14, .10, .06, .03):
            for pigment, _ in ink.tiles.values():
                pigment[pigment > 0] = density
            out = judge(task, ink)
            scores.append(out["score"])
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))
        channels = np.full((ink.height, ink.width), 255, np.uint8)
        channels[100:150, 100:150] = 1
        paint(ink, channels)
        self.assertFalse(judge(task, ink)["success"])

    def test_keep_region_damage_and_recolour_fail(self):
        task = CatalogueTask("erase-except-boxed")
        self.assertEqual(judge(task, surface())["score"], 0)
        ink = surface(task)
        target = raster(task.strokes(), ink.board, ink.width, ink.height)
        paint(ink, np.where(target < 6, 1, 255))
        self.assertFalse(judge(task, ink)["success"])
        self.assertEqual(judge(task, ink)["score"], 0)
        damaged = target.copy()
        damaged[:, :ink.width//2] = 255
        paint(ink, damaged)
        self.assertFalse(judge(task, ink)["success"])

    def test_correction_earns_only_change_and_each_neighbour_matters(self):
        task = CatalogueTask("correct-misspelling")
        ink = surface(task, start=True)
        out = judge(task, ink)
        self.assertEqual(out["progress"], 0)
        self.assertGreater(out["factors"]["initial_glyph_f1"], 0)
        # Erasing whole word cannot collect retention credit.
        self.assertEqual(judge(task, surface())["score"], 0)
        for i in (0, 1, 2, 3, 4, 6):
            strokes = tuple(s for s in task.strokes() if s[0] != f"letter {i}")
            paint(ink, raster(strokes, ink.board, ink.width, ink.height))
            self.assertFalse(judge(task, ink)["success"])
        # Final-state equivalence permits erasing and rewriting neighbours in place.
        paint(ink, raster(task.strokes(), ink.board, ink.width, ink.height))
        self.assertTrue(judge(task, ink)["success"])
        ink.tiles = dict(reversed(list(ink.tiles.items())))
        self.assertTrue(judge(task, ink)["success"])

    def test_return_release_and_standing_are_end_state_requirements(self):
        for name in ("erase-board", "erase-except-boxed", "correct-misspelling", "recap-and-tidy"):
            task, ink = CatalogueTask(name), surface(CatalogueTask(name))
            good = state(task, capped=True)
            for modification in ("held", "fallen", "airborne", "floor", "fast"):
                s = copy.deepcopy(good)
                tool = "marker_black" if name == "recap-and-tidy" else "eraser"
                if modification == "held": s["touching"].add(tool)
                if modification == "fallen": s["pelvis_z"] = .3
                if modification == "airborne": s["poses"][tool][2] += .05
                if modification == "floor": s["poses"][tool][2] = .03
                if modification == "fast": s["velocities"][tool][0] = 1.
                out = judge(task, ink, s)
                self.assertFalse(out["success"], (name, modification))
                self.assertLessEqual(out["score"], .9)

    def test_recap_axial_radial_orientation_and_collateral(self):
        task, ink = CatalogueTask("recap-and-tidy"), surface()
        s = state(task)
        scores = []
        for distance in (0, .015, .030, .045, .060):
            candidate = copy.deepcopy(s)
            for color in OPEN_COLORS:
                candidate["poses"][f"cap_{color}"][0] -= distance
            scores.append(judge(task, ink, candidate)["progress"])
        self.assertEqual(scores[0], 0)
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])), scores)
        quantized = copy.deepcopy(s)
        quantized["poses"] = {n: p.astype(np.float32) for n, p in s["poses"].items()}
        self.assertEqual(judge(task, ink, quantized)["progress"], 0)
        for kind in ("radial", "reversed", "overinserted", "collateral"):
            candidate = state(task, capped=True)
            cap = candidate["poses"]["cap_green" if kind == "collateral" else "cap_black"]
            if kind == "radial": cap[1] += .005
            elif kind == "reversed": cap[3:] = Rotation.from_euler("y", -math.pi/2).as_quat()
            else: cap[0] += .04
            self.assertFalse(judge(task, ink, candidate)["success"], kind)

    def test_table_registered_copy_partial_gap_and_extra_line(self):
        task, ink = CatalogueTask("draw-table"), surface()
        strokes = task.strokes()
        scores = []
        for count in (0, 2, 4, 6):
            paint(ink, raster(strokes[:count], ink.board, ink.width, ink.height))
            out = judge(task, ink)
            self.assertFalse(out["success"])
            scores.append(out["progress"])
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])), scores)
        moved = tuple((n,c,w,tuple((x+.012,z+.008) for x,z in p)) for n,c,w,p in strokes)
        paint(ink, raster(moved, ink.board, ink.width, ink.height))
        self.assertTrue(judge(task, ink)["success"])
        target = raster(strokes, ink.board, ink.width, ink.height).copy()
        # 30 mm gap in top frame, away from junctions.
        u,v = ink.uv(-.29, 1.46)
        target[round(v)-6:round(v)+6, round(u)-13:round(u)+13] = 255
        paint(ink, target)
        self.assertFalse(judge(task, ink)["success"])
        extra = strokes + (("extra", "red", "normal", ((-.4,1.05),(.4,1.05))),)
        paint(ink, raster(extra, ink.board, ink.width, ink.height))
        self.assertFalse(judge(task, ink)["success"])


if __name__ == "__main__":
    unittest.main()

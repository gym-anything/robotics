"""Verifier-unit fixtures author pigment directly; never a policy solution."""

import hashlib
import math
import unittest
import numpy as np
from PIL import Image, ImageDraw
from whiteboard.config import Board, MaterialPriors, COLORS
from whiteboard.ink import InkSurface
from whiteboard.tasks import BRIDGE_INSTRUCTION, Task, bridge_strokes, reference_channels
from whiteboard.variations import resolve


class TaskTests(unittest.TestCase):
    def surface(self, task, shift=0, wrong_color=False, noise=False):
        ink = InkSurface(Board(), 1024, 512, MaterialPriors())
        reference = np.roll(task.reference(ink), shift, axis=1)
        palette = np.asarray([[round(v * 255) for v in rgb] for rgb in COLORS.values()])
        for ty in range(4):
            for tx in range(8):
                pixels = reference[ty * 128 : (ty + 1) * 128, tx * 128 : (tx + 1) * 128]
                a = np.zeros((128, 128, 6), np.float32)
                for channel, color in enumerate(palette):
                    a[:, :, (channel + 1) % 6 if wrong_color else channel] = np.all(
                        pixels == color, axis=2
                    )
                if noise:
                    a[:, :, :] = 1
                if a.any():
                    ink.tiles[tx, ty] = (
                        np.repeat(a[:, :, None, :], ink.subcells**2, axis=2),
                        np.zeros((128, 128, ink.subcells**2), np.float32),
                    )
        return ink

    def test_exact_reference_positive_text_and_bridge(self):
        for kind in ("text", "bridge"):
            task = Task(kind=kind)
            result = task.evaluate(self.surface(task))
            self.assertTrue(result["success"])
            self.assertEqual(result["score"], 1)

    def test_shift_wrong_color_and_full_board_fail(self):
        task = Task(kind="bridge")
        for opts in ({"shift": 70}, {"wrong_color": True}, {"noise": True}):
            with self.subTest(opts=opts):
                self.assertFalse(task.evaluate(self.surface(task, **opts))["success"])

    def test_blank_and_invisible_reference_rejected(self):
        with self.assertRaises(ValueError):
            Task(text="   ")
        with self.assertRaises(ValueError):
            Task(center_z=float("nan"))

    def test_reference_cache_never_caches_actual_pigment_or_scores(self):
        task = Task()
        ink = self.surface(task)
        self.assertTrue(task.evaluate(ink)["success"])
        ink.tiles.clear()
        self.assertFalse(task.evaluate(ink)["success"])
        self.assertEqual(task.evaluate(ink)["score"], 0)

    def test_reference_cache_key_covers_task_board_and_resolution(self):
        reference_channels.cache_clear()
        task = Task()
        args = (task.kind, task.text, task.center_x, task.center_z, task.width_m)
        first = reference_channels(args, Board(), 1024, 512)
        self.assertIs(first, reference_channels(args, Board(), 1024, 512))
        self.assertFalse(first.flags.writeable)
        with self.assertRaises(ValueError):
            first[0, 0] = 0
        for changed in (
            (("bridge", *args[1:]), Board(), 1024, 512),
            (("text", "WORLD", *args[2:]), Board(), 1024, 512),
            ((*args[:2], 0.1, *args[3:]), Board(), 1024, 512),
            ((*args[:3], 1.48, args[4]), Board(), 1024, 512),
            ((*args[:4], 0.6), Board(), 1024, 512),
            (args, Board(width=2.0), 1024, 512),
            (args, Board(), 2048, 1024),
        ):
            self.assertIsNot(first, reference_channels(*changed))


class BridgeTaskTests(unittest.TestCase):
    """Bridge copy job (revision bridge-copy-v1). Boards are constructed
    pigment: 3.5 mm marker-width lines rasterised from the reference strokes."""

    RES = (2048, 1024)

    def setUp(self):
        self.task = resolve(0)[1]

    def board(self, transform=None, colours=None, skip=(), extra=(), width_px=3, density=1.0, dash=None, keep=None,
              moved=None, res=None):
        """dash=(on_m, off_m) draws every stroke as dashes; keep(name, index, arc_m, x, z) -> bool keeps only part of
        a stroke; moved maps an element name to its own transform; res overrides the texture size."""
        res = res or self.RES
        ink = InkSurface(Board(), *res, MaterialPriors())
        layers = {}
        for index, (name, colour, _, points) in enumerate(bridge_strokes()):
            if name in skip:
                continue
            colour = (colours or {}).get(colour, colour)
            world = [(self.task.center_x + x * self.task.width_m, self.task.center_z + y * self.task.width_m)
                     for x, y in points]
            step = (moved or {}).get(name, transform)
            world = np.asarray([step(*p) for p in world] if step else world)
            image = layers.setdefault(colour, Image.new("L", res, 0))
            pieces = [world]
            if dash or keep:
                cum = np.r_[0, np.cumsum(np.hypot(*np.diff(world, axis=0).T))]
                arc = np.arange(0, cum[-1], 0.00025)
                dense = np.c_[np.interp(arc, cum, world[:, 0]), np.interp(arc, cum, world[:, 1])]
                on = np.ones(len(arc), bool)
                if dash:
                    on &= arc % (dash[0] + dash[1]) < dash[0]
                if keep:
                    on &= np.asarray([keep(name, index, a, x, z) for a, (x, z) in zip(arc, dense)])
                edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
                pieces = [dense[a:b] for a, b in zip(edges[::2], edges[1::2])]
            for piece in pieces:
                uv = [ink.uv(*q) for q in piece]
                ImageDraw.Draw(image).line(uv if len(uv) > 1 else uv * 2, fill=255, width=width_px,
                                           joint=None if dash or keep else "curve")  # pieces end flat
        masks = {c: np.asarray(im) > 0 for c, im in layers.items()}
        for colour, mask in extra:
            masks[colour] = masks.get(colour, np.zeros_like(mask)) | mask
        for ty in range(res[1] // 128):
            for tx in range(res[0] // 128):
                window = np.s_[ty * 128:(ty + 1) * 128, tx * 128:(tx + 1) * 128]
                if any(m[window].any() for m in masks.values()):
                    pigment = np.zeros((128, 128, 1, 6), np.float32)
                    for colour, mask in masks.items():
                        pigment[:, :, 0, tuple(COLORS).index(colour)] = mask[window] * density
                    ink.tiles[tx, ty] = (pigment, np.zeros((128, 128, 1), np.float32))
        return ink

    def square(self, side_m, centre=(-0.95, 1.10)):
        ink = InkSurface(Board(), *self.RES, MaterialPriors())
        (u0, v0), (u1, v1) = ink.uv(centre[0] - side_m / 2, centre[1] + side_m / 2), ink.uv(
            centre[0] + side_m / 2, centre[1] - side_m / 2)
        mask = np.zeros(self.RES[::-1], bool)
        mask[round(v0):max(round(v1), round(v0) + 1), round(u0):max(round(u1), round(u0) + 1)] = True
        return mask

    def similarity(self, dx=0.0, dz=0.0, scale=1.0, degrees=0.0, mirror=False):
        cx, cz, c, s = self.task.center_x, self.task.center_z, math.cos(math.radians(degrees)), math.sin(math.radians(degrees))

        def apply(x, z):
            x, z = (2 * cx - x if mirror else x) - cx, z - cz
            return cx + dx + scale * (c * x - s * z), cz + dz + scale * (s * x + c * z)
        return apply

    def test_scored_reference_raster_is_unchanged(self):
        for task, size, digest in (
            (Task(kind="bridge"), (1024, 512), "a71d1bf240d7192bc0aaf9f5331e30a8d721c176e6df3daafcb5583ccab0b2d0"),
            # Variation 0, its drawing placed within the robot's reach.
            (self.task, (4096, 2048), "e08634131bd4996cfab1674818cc53e3cc4729e725cd4ffabe813cd894888005"),
        ):
            image = task.reference(InkSurface(Board(), *size, MaterialPriors()))
            self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), digest)

    def test_supplied_picture_is_the_front_view_and_text_is_unchanged(self):
        ink = InkSurface(Board(), 1024, 512, MaterialPriors())
        np.testing.assert_array_equal(self.task.reference_view(ink), self.task.reference(ink)[:, ::-1])
        np.testing.assert_array_equal(Task().reference_view(ink), Task().reference(ink))
        self.assertEqual(Task().reference_role, "Public desired drawing, not current board state")

    def test_instruction_states_the_job_not_the_tolerances(self):
        self.assertEqual(self.task.description, BRIDGE_INSTRUCTION)
        self.assertFalse(any(ch.isdigit() for ch in BRIDGE_INSTRUCTION))
        self.assertEqual(Task(text="HELLO").description, "Use the black marker to write 'HELLO' on the whiteboard, "
                         "matching the supplied reference image. Leave other board areas clean.")

    def test_faithful_copies_get_full_credit(self):
        for label, transform in (
            ("exact", None), ("12 mm right, 8 mm low", self.similarity(-0.012, -0.008)),
            ("3 % small", self.similarity(scale=0.97)), ("tilted 1.5 deg", self.similarity(degrees=1.5)),
            ("freehand wobble", lambda x, z: (x + 0.002 * math.sin(90 * (x + z)), z + 0.002 * math.sin(90 * (x - z)))),
        ):
            with self.subTest(label):
                result = self.task.evaluate(self.board(transform))
                self.assertTrue(result["success"], result["parts"])
                self.assertEqual((result["score"], result["progress"]), (1.0, 1.0))

    def test_wrong_or_incomplete_copies_fail(self):
        for label, kwargs in (
            ("no main cable", dict(skip=("main cable",))),
            ("towers red, cables orange", dict(colours={"orange": "red", "red": "orange"})),
            ("mirror image at the shown place", dict(transform=self.similarity(mirror=True))),
            ("60 mm away", dict(transform=self.similarity(0.06))),
            ("whole drawing in green", dict(colours={c: "green" for c in COLORS})),
        ):
            with self.subTest(label):
                result = self.task.evaluate(self.board(**kwargs))
                self.assertFalse(result["success"])
                self.assertLess(result["score"], 0.9)

    def test_stray_marks_cost_the_same_in_every_colour(self):
        dab = self.square(0.004)
        self.assertTrue(self.task.evaluate(self.board(extra=[("green", dab)]))["success"])
        scores = {c: self.task.evaluate(self.board(extra=[(c, self.square(0.05))]))["score"]
                  for c in ("green", "black", "orange")}
        self.assertTrue(all(s < 0.9 for s in scores.values()))
        self.assertAlmostEqual(max(scores.values()), min(scores.values()), places=3)

    def test_progress_falls_with_displacement_and_defacement(self):
        progress = [self.task.evaluate(self.board(self.similarity(d)))["progress"] for d in (0.0, 0.04, 0.08, 0.12)]
        self.assertEqual(progress[0], 1.0)
        self.assertTrue(all(a > b for a, b in zip(progress, progress[1:])), progress)
        box = np.zeros(self.RES[::-1], bool)
        ink = InkSurface(Board(), *self.RES, MaterialPriors())
        (u0, v0), (u1, v1) = (ink.uv(self.task.center_x - 0.28, self.task.center_z + 0.16),
                              ink.uv(self.task.center_x + 0.28, self.task.center_z - 0.12))
        box[round(v0):round(v1), round(u0):round(u1)] = True
        self.assertLess(self.task.evaluate(self.board(extra=[("black", box)]))["progress"], 0.1)
        faint = self.task.evaluate(self.board(density=0.08))
        self.assertFalse(faint["success"])
        self.assertTrue(0.2 < faint["progress"] < 0.8, faint["progress"])

    def test_bridge_drawings_stay_within_the_robots_writing_reach(self):
        from whiteboard.variations import BRIDGE_TOP_Z_M, specification
        for index in range(0, 10000, 7):
            task = specification(index)["task"]
            if task["kind"] == "bridge":
                self.assertLessEqual(task["center_z"] + 0.27 * task["width_m"], BRIDGE_TOP_Z_M + 1e-9)

    def test_partial_copies_fail_and_parts_slightly_apart_pass(self):
        dabs = lambda x, z: (x, z)  # noqa: E731
        for label, kwargs in (
            ("no cross-bars", dict(skip=("right tower cross-bars", "left tower cross-bars"))),
            ("no main cable", dict(skip=("main cable",))),
        ):
            with self.subTest(label):
                self.assertFalse(self.task.evaluate(self.board(**kwargs))["success"])
        ink = self.board(transform=dabs)
        for (tx, ty), (pigment, _) in ink.tiles.items():  # dotted copy: keep 2 of every 9 columns and rows
            pigment[(np.arange(128)[:, None] % 9 > 1) | (np.arange(128)[None, :] % 9 > 1)] = 0
        self.assertFalse(self.task.evaluate(ink)["success"])
        water_low = self.board(transform=lambda x, z: (x, z - 0.006) if z < self.task.center_z - 0.045 else (x, z))
        self.assertTrue(self.task.evaluate(water_low)["success"])

    def test_misplaced_complete_copies_rank_by_distance(self):
        scores = [self.task.evaluate(self.board(self.similarity(-d)))["score"] for d in (0.10, 0.20, 0.30)]
        self.assertTrue(scores[0] > scores[1] > scores[2] > 0, scores)
        tilted = self.task.evaluate(self.board(self.similarity(degrees=5)))["score"]
        self.assertGreater(tilted, 0.5)


    def test_dotted_copies_fail_and_a_skipping_marker_passes(self):
        full = (4096, 2048)  # the environment's texture: short dashes survive rasterisation (5 texels per lines' width)
        dotted = self.task.evaluate(self.board(dash=(0.002, 0.004), width_px=5, res=full))
        self.assertFalse(dotted["success"])
        self.assertTrue(0.25 < dotted["progress"] < 0.7, dotted["progress"])  # tracks the inked share, no cliff
        sparse = self.task.evaluate(self.board(dash=(0.004, 0.004), width_px=5, res=full))["progress"]
        self.assertTrue(dotted["progress"] < sparse < 0.95, sparse)
        skipping = self.task.evaluate(self.board(dash=(0.010, 0.004), width_px=5, res=full))
        self.assertTrue(skipping["success"], skipping["parts"])

    def test_breaks_beside_junctions_are_measured_along_the_line(self):
        hangers = sorted({self.task.center_x + p[-1][0] * self.task.width_m
                          for name, _, _, p in bridge_strokes() if name == "hangers"})
        holes = [(hangers[5] + hangers[6]) / 2, (hangers[11] + hangers[12]) / 2]
        broken = self.task.evaluate(self.board(keep=lambda name, i, a, x, z: name != "main cable"
                                               or all(abs(x - h) > 0.011 for h in holes)))
        self.assertFalse(broken["success"])
        self.assertIn("main cable", broken["parts"]["cables and hangers"]["gap_excess_mm"])
        # A cable drawn in pieces that meet at hanger tops is complete.
        cut = hangers[8]
        pieces = self.board(keep=lambda name, i, a, x, z: name != "main cable" or abs(x - cut) > 0.0005)
        self.assertTrue(self.task.evaluate(pieces)["success"])
        legs = [i for i, s in enumerate(bridge_strokes()) if s[0].endswith("tower legs")]
        scores = []
        for leg in legs:
            with self.subTest(stub=leg):
                stub = self.task.evaluate(self.board(keep=lambda name, i, a, x, z: i != leg or a < 0.026))
                self.assertFalse(stub["success"])
                scores.append(self.task.evaluate(self.board(keep=lambda name, i, a, x, z: i != leg))["score"])
        self.assertLess(max(scores) - min(scores), 0.02, scores)

    def test_partial_lines_earn_partial_credit_once(self):
        path = np.asarray(next(s[3] for s in bridge_strokes() if s[0] == "roadway")) * self.task.width_m
        length = np.hypot(*np.diff(path, axis=0).T).sum()
        roadway = [self.task.evaluate(self.board(keep=lambda name, i, a, x, z: name != "roadway" or a < f * length))
                   ["parts"]["roadway"]["credit"] for f in (0.0, 0.25, 0.5, 0.75)]
        self.assertTrue(all(a < b for a, b in zip(roadway, roadway[1:])), roadway)
        for drawn, credit in zip((0.25, 0.5, 0.75), roadway[1:]):
            self.assertLess(abs(credit - drawn), 0.12, roadway)
        exact = self.task.evaluate(self.board())
        for part in exact["parts"].values():
            self.assertTrue(all(v >= 0.99 for v in part["coverage"].values()), part["coverage"])

    def test_one_misplaced_part_does_not_drag_the_rest(self):
        results = [self.task.evaluate(self.board(moved={"roadway": self.similarity(0, -d)})) for d in (0.010, 0.015, 0.020, 0.025)]
        scores = [r["score"] for r in results]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(scores, scores[1:])), scores)
        for part in ("towers", "cables and hangers", "water"):
            self.assertEqual(results[2]["parts"][part]["credit"], 1.0, results[2]["parts"])
        for colour, (dx, dz) in (("orange", (0.006, 0)), ("red", (0, 0.006)), ("black", (0, -0.006)),
                                 ("blue", (-0.006, 0)), ("red", (-0.006, 0)), ("orange", (0, 0.006))):
            with self.subTest(colour=colour, shift=(dx, dz)):
                names = {s[0] for s in bridge_strokes() if s[1] == colour}
                result = self.task.evaluate(self.board(moved={n: self.similarity(dx, dz) for n in names}))
                self.assertTrue(result["success"], result["parts"])


if __name__ == "__main__":
    unittest.main()

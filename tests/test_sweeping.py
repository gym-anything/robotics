"""Sweeping: discovery, task catalogue, start states, the bristle and debris solver's contact laws, the evaluator and the
public backend contract.

Run (the solver tests need the GPU; the backend tests build the full room, about a minute each):
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_sweeping.py
"""
import math
import unittest

import numpy as np


class Discovery(unittest.TestCase):


    def test_tools(self):
        from sweeping import toolkit as T, params as P
        for name, spec in (("angle_broom", P.ANGLE_BROOM), ("push_broom", P.PUSH_BROOM), ("lobby_broom", P.LOBBY_BROOM),
                           ("counter_brush", P.COUNTER_BRUSH)):
            t = T.BUILDERS[name]()
            s = T.summary(t)
            self.assertAlmostEqual(s["mass"], spec["mass"], places=6)      # the parts carry the catalogue mass
            self.assertEqual(s["clusters"]*int(round(t["bank"].filaments)), spec["filaments"])
            self.assertLess(s["bristle_mass"], spec["mass"])
            self.assertTrue(np.allclose(np.linalg.norm(t["bank"].dirs, axis=1), 1.0))
        pan = T.lobby_pan()
        self.assertAlmostEqual(T.summary(pan)["mass"], P.LOBBY_PAN["mass"], places=6)
        self.assertGreater(pan["pivot"][1], T.summary(dict(parts=pan["parts"]))["com"][1])   # pivot ahead of the pan's COM


class Dust(unittest.TestCase):
    """The fine-dust film (sweeping/dust.py): a reduced-order model whose one hard promise is that no mass appears
    or disappears."""

    def test_tasks_that_have_dust(self):
        from sweeping import layout as L, params as P
        from sweeping.task import instruction
        from sweeping.variants import resolve
        self.assertAlmostEqual(sum(p["grams"] for p in L.setup("pan-and-barrel", 0)["dust"]), P.DUST_LOADS["pile_g"])
        self.assertEqual(L.setup("gather-open", 0)["dust"], [])
        self.assertEqual(len(L.setup("gather-open", 0, variant="dusty")["dust"]), 1)
        self.assertEqual(len(L.setup("closing-sweep", 0)["dust"]), 2)
        self.assertIn("at least half", instruction(None, dict(task="pan-and-barrel")))
        self.assertIn("at least half", instruction(None, dict(task="gather-open", variant="dusty")))
        self.assertNotIn("dust", instruction(None, dict(task="gather-open")).lower())
        with self.assertRaises(ValueError):
            resolve(task="table-scraps", variant="dusty")


    def test_a_tipped_pan_empties_into_the_barrel(self):
        import types
        from scipy.spatial.transform import Rotation
        from sweeping import toolkit as T, params as P
        from sweeping.dust import DustFilm
        pose = dict(lobby_pan=(np.array([2.0, 4.0, 0.0006]), np.eye(3)), barrel=(np.array([2.0, 4.6, 0.0006]), np.eye(3)))
        clock = dict(t=0.0)
        film = DustFilm(types.SimpleNamespace(rod_slices=[]), dict(lobby_pan=T.lobby_pan(), barrel=T.barrel()), [], pose=lambda n: pose[n],
                        active=lambda: [], clock=lambda: clock["t"])
        pts = np.array([[2.0 + dx, 4.0 + dy, 0.05] for dx in (-0.08, 0.0, 0.08) for dy in (-0.08, 0.0, 0.08)])
        film.drop(pts, np.full(len(pts), 1.0e-4), pose)                      # 0.9 g let go over the pan
        film.total0 = film.total()
        self.assertAlmostEqual(film.summary()["pans"]["lobby_pan"], 0.9, places=6)

        def run(seconds):
            for _ in range(int(seconds/0.005)):
                clock["t"] += 0.005
                film.update(0.005)
        run(1.0)                                                            # level: nothing moves
        self.assertAlmostEqual(film.summary()["pans"]["lobby_pan"], 0.9, places=6)
        # carried mouth-up (the pan hangs like this from its handle): the film piles against the back wall and stays
        # (for five seconds: its speed against the wall must not go on growing, or it does not come out afterwards)
        up = Rotation.from_euler("x", math.radians(50.0)).as_matrix()
        pose["lobby_pan"] = (np.array([2.0, 4.0, 0.8]), up)
        run(5.0)
        self.assertAlmostEqual(film.summary()["pans"]["lobby_pan"], 0.9, places=6)
        self.assertLess(float(np.abs(film.pans["lobby_pan"]["speed"]).max()), 0.5)
        # held over the barrel, mouth down by 60 degrees: it runs out over the lip and lands in the barrel
        down = Rotation.from_euler("x", math.radians(-60.0)).as_matrix()
        lip = down @ np.array([0.0, P.LOBBY_PAN["pan"][1]/2, 0.0])
        pose["lobby_pan"] = (np.array([2.0, 4.6, 0.9]) - lip, down)
        run(1.0)
        s = film.summary()
        self.assertAlmostEqual(s["containers"]["barrel"], 0.9, places=6)
        self.assertAlmostEqual(s["pans"]["lobby_pan"], 0.0, places=9)
        self.assertLess(abs(s["unaccounted"]), 1e-9)
        # the same tip over the bare floor leaves it on the floor under the lip
        film.drop(pts + (0.0, 0.6, 0.9), np.full(len(pts), 1.0e-4), pose)   # (falls into the barrel: points are over it)
        self.assertAlmostEqual(film.summary()["containers"]["barrel"], 1.8, places=6)


class ClosingSweep(unittest.TestCase):
    """closing-sweep: its text and its reward contract: score 1 on success, else 0.9 x progress, and an untouched room
    scores 0."""

    def test_instruction(self):
        from sweeping.task import instruction
        from sweeping.variants import task_resolve
        text = task_resolve({"task": "closing-sweep"})["description"]
        for tolerance in ("at least half", "%", "90", "0.1"):
            self.assertNotIn(tolerance, text)
        for words in ("grey barrel", "put back any chairs you move", "put the tools away where they belong", "fine dust", "table b1",
                      "the wall with the tool rail", "between the cubbies and the cabinets"):
            self.assertIn(words, text.lower())
        self.assertNotIn("right-hand", text)                  # a frame the robot, facing the rail, does not share
        self.assertTrue(instruction(None, dict(task="closing-sweep", variant="push-broom-only")).endswith(
            "put the tools away where they belong. Only the push broom is on the rail."))

    def test_progress_contract(self):
        """Evaluator.progress on its inputs alone: what lies in a dustpan counts, less than in the barrel; bristle dust
        is credited only as the verifier credits it; tipped bodies are charged beyond the cap."""
        from sweeping.task import Evaluator, PROGRESS
        none = dict(mass_binned=0.0, mass_in_pans=0.0, wads=2, wads_binned=0, wads_in_pans=0, scraps=11, scraps_binned=0, scraps_in_pans=0,
                    pencil="left out", dust_start=1e-3, dust_binned=0.0, dust_lodged_credit=0.0, dust_in_pans=0.0)
        p = lambda q, oop=None, tipped=(): Evaluator.progress(None, dict(none, **q), oop or {}, set(tipped))["progress"]
        lane = dict(mass_binned=0.45, wads_binned=2, scraps_binned=7, dust_binned=3.5e-4)
        panned = dict(mass_in_pans=0.45, wads_in_pans=2, scraps_in_pans=7, dust_in_pans=3.5e-4)
        self.assertEqual(p({}), 0.0)
        self.assertLess(0.0, p(panned))
        self.assertLess(p(panned), p(lane))
        done = dict(mass_binned=1.0, wads_binned=2, scraps_binned=11, pencil="put away", dust_binned=6e-4)
        self.assertEqual(p(done), 1.0)
        chairs = {f"chair_{k}": 1.0 for k in range(25)}
        self.assertAlmostEqual(p(done, chairs), 1.0 - PROGRESS["cap"], places=6)                        # graded: capped
        self.assertAlmostEqual(p(done, chairs, chairs), 1.0 - 25*PROGRESS["per_item"], places=6)       # tipped: each one
        # a saturated term does not let a dustpan's contents stand in for emptying them
        part = dict(done, pencil="left out", dust_binned=4.5e-4)
        self.assertLess(p(dict(part, dust_in_pans=1e-4)), p(dict(part, dust_binned=5.5e-4)))
        self.assertLess(p(dict(done, pencil="left out", scraps_binned=3, scraps_in_pans=5)), p(dict(done, pencil="left out", scraps_binned=8)))


if __name__ == "__main__":
    unittest.main()

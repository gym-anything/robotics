import unittest
import numpy as np
from whiteboard.config import Board, MaterialPriors, Config
from whiteboard.geometry import classroom, tools
from whiteboard.ink import InkSurface
from whiteboard.deformable_contact import board_patch
from scipy.spatial.transform import Rotation
from types import SimpleNamespace


class MaterialTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        for parameters in (
            {"texture_width": 2.5},
            {"seed": True},
            {"physics_dt": float("nan")},
            {"device": "cuda:0"},
        ):
            with self.assertRaises(ValueError):
                Config(**parameters)
        with self.assertRaises(ValueError):
            Board(width=0.01)

    def setUp(self):
        self.board = Board()
        self.ink = InkSurface(self.board, 2048, 1024, MaterialPriors())
        self.center = (0, self.board.center_z)
        self.axes = np.diag([0.0025, 0.001])

    def test_no_load_and_no_time_never_write(self):
        self.assertEqual(self.ink.deposit("red", self.center, self.axes, 0, 0.01, 0), 0)
        self.assertEqual(self.ink.deposit("red", self.center, self.axes, 1, 0, 0), 0)
        self.assertEqual(len(self.ink.tiles), 0)

    def test_frame_excluded(self):
        self.ink.deposit("red", (self.board.width / 2, self.board.center_z), self.axes, 1, 0.1, 0)
        self.assertEqual(self.ink.total_deposited_ul, 0)

    def test_deposition_subdivision_invariant(self):
        other = InkSurface(self.board, 2048, 1024, MaterialPriors())
        self.ink.deposit("red", self.center, self.axes, 1, 0.02, 0)
        for i in range(20):
            other.deposit("red", self.center, self.axes, 1, 0.001, i * 0.001)
        self.assertAlmostEqual(self.ink.total_deposited_ul, other.total_deposited_ul, places=7)

    def test_finite_reservoir_and_mass_balance(self):
        self.ink.reservoir["red"] = 0.005
        for i in range(100):
            self.ink.deposit("red", (i * 0.001, self.board.center_z), self.axes, 1, 0.01, i * 0.01)
        self.assertAlmostEqual(self.ink.total_deposited_ul, 0.005, places=9)
        self.assertAlmostEqual(self.ink.pigment_volume_ul(), 0.005, places=7)
        self.assertGreaterEqual(self.ink.reservoir["red"], 0)

    def test_eraser_needs_pressure_and_travel_conserves_pigment(self):
        self.ink.deposit("blue", self.center, self.axes, 2, 0.1, 0)
        before = self.ink.pigment_volume_ul()
        self.ink.erase(self.center, self.axes, 0, 0.1, 10)
        self.ink.erase(self.center, self.axes, 2, 0, 10)
        self.assertAlmostEqual(before, self.ink.pigment_volume_ul())
        self.ink.erase(self.center, np.diag([0.052, 0.027]), 3, 0.10, 10)
        after = self.ink.pigment_volume_ul()
        self.assertLess(after, before * 0.1)
        self.assertAlmostEqual(before, after + self.ink.eraser_load.sum(), places=7)

    def test_each_color_is_distinct(self):
        from whiteboard.config import COLORS

        colors = []
        for i, color in enumerate(COLORS):
            center = (-0.06 + i * 0.02, self.board.center_z)
            self.ink.deposit(color, center, np.diag([0.005, 0.005]), 2, 0.1, 0)
            u, v = self.ink.uv(*center)
            colors.append(tuple(self.ink.rgb()[int(v), int(u)]))
        self.assertEqual(len(set(colors)), 6)

    def material_fixture(self, gravity=(0, 0, 0)):
        """Unit-level apparatus for the actual resolved contact implementation."""
        return SimpleNamespace(
            config=Config(robot=False, furniture=False),
            body_ids={"tool": 0},
            com=np.zeros((1, 3)),
            deformations={},
            model=SimpleNamespace(
                body_mass=SimpleNamespace(numpy=lambda: np.array([0.03])),
                body_inertia=SimpleNamespace(numpy=lambda: np.diag([1e-5, 3e-5, 4e-5])[None]),
            ),
            solver=SimpleNamespace(mj_model=SimpleNamespace(opt=SimpleNamespace(gravity=gravity))),
        )

    def test_resolved_normal_force_and_dissipation(self):
        env = self.material_fixture()
        r = Rotation.from_euler("x", np.pi / 2).as_quat()
        q = np.r_[[0, self.board.face_y + 0.063 - 0.0001, self.board.center_z], r]
        patch = board_patch(
            env,
            "nib",
            "tool",
            q,
            np.array([0.1, 0, 0, 0, 0, 0]),
            (0, 0, 0.063),
            ((0.0025, 0, 0), (0, 0.001, 0)),
            0.35,
            0.45,
        )
        self.assertGreater(patch["load"], 0)
        self.assertLess(patch["wrench"][0] * 0.1, 0)
        self.assertAlmostEqual(patch["compression"], 0.0001, places=8)
        self.assertGreater(env.deformations["tool"]["energy_j"], 0)
        self.assertLessEqual(env.deformations["tool"]["friction"]["power_w"], 0)

    def test_tools_carry_held_press_loads_inside_material_domain(self):
        # JIS S 6037 pushes a marker nib with 6 N; a hand-held duster is pressed far harder than 2 N.
        nib = board_patch(
            self.material_fixture(),
            "nib",
            "tool",
            np.r_[[0, self.board.face_y + 0.063 - 0.0016, self.board.center_z],
                  Rotation.from_euler("x", np.pi / 2).as_quat()],
            np.zeros(6),
            (0, 0, 0.063),
            ((0.0025, 0, 0), (0, 0.001, 0)),
            0.35,
            0.45,
        )
        self.assertGreater(nib["load"], 6.0)
        felt = board_patch(
            self.material_fixture(),
            "felt",
            "tool",
            np.r_[[0, self.board.face_y + 0.010 - 0.0001, self.board.center_z],
                  Rotation.from_euler("x", -np.pi / 2).as_quat()],
            np.zeros(6),
            (0, 0, -0.01),
            ((0.0515, 0, 0), (0, 0.0265, 0)),
            0.55,
            1.0,
        )
        self.assertGreater(felt["load"], 20.0)

    def test_named_dimensions_and_separate_caps(self):
        objects = tools(Config())
        self.assertEqual(len(objects), 13)
        self.assertAlmostEqual(sum(o.mass for o in objects[:2]), 0.0196)
        self.assertAlmostEqual(self.board.width, 96 * 0.0254)
        self.assertEqual(sum(o.name.startswith("chair") for o in classroom(Config())), 12)

    def test_resolved_pad_hover_and_domain_guard(self):
        env = self.material_fixture()
        q = np.r_[
            [0, self.board.face_y + 0.020, self.board.center_z],
            Rotation.from_euler("x", -np.pi / 2).as_quat(),
        ]
        args = (
            env,
            "felt",
            "tool",
            q,
            np.zeros(6),
            (0, 0, -0.01),
            ((0.0515, 0, 0), (0, 0.0265, 0)),
            0.55,
            1.0,
        )
        self.assertIsNone(board_patch(*args))
        q[1] = self.board.face_y + 0.004
        with self.assertRaises(FloatingPointError):
            board_patch(*args)

    def test_pad_sliding_force_is_coulomb_bounded(self):
        q = np.r_[
            [0, self.board.face_y + 0.009, self.board.center_z],
            Rotation.from_euler("x", -np.pi / 2).as_quat(),
        ]
        patch = board_patch(
            self.material_fixture(),
            "felt",
            "tool",
            q,
            np.array([1.0, 0, 0, 0, 0, 0]),
            (0, 0, -0.01),
            ((0.0515, 0, 0), (0, 0.0265, 0)),
            0.55,
            1.0,
        )
        self.assertLess(patch["wrench"][0], 0)
        self.assertLessEqual(np.linalg.norm(patch["wrench"][[0, 2]]), 0.55 * patch["load"] + 1e-12)

    def test_chair_shell_is_closed_with_outward_faces(self):
        import trimesh

        obj = next(o for o in classroom(Config()) if o.name == "chair_0_0")
        part = next(p for p in obj.parts if p.name == "back_shell")
        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)


if __name__ == "__main__":
    unittest.main()

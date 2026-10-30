"""Soil excavation: discovery, catalogue, robot validation, layout and material-mapping units without the worlds; the
backend's public path (registry, actions, state, evaluation, refusals, checkpoints) with a scene build when
GAR_GPU_TESTS=1 (MuJoCo on the CPU, the soil on cuda:0; cameras=() so no render worker).

python -m unittest tests.test_soil_excavation
GAR_GPU_TESTS=1 python -m unittest tests.test_soil_excavation
"""
import math
import os
import unittest

import numpy as np


class Discovery(unittest.TestCase):
    def test_describe_and_robots(self):
        from gym_anything_robotics.launch import describe
        d = describe("soil-excavation")
        self.assertEqual(d["default_robot"], "mobile-fr3-duo")
        self.assertEqual({r["id"] for r in d["robots"]}, {"mobile-fr3-duo", "mobile-fr3-single"})
        self.assertEqual(describe("dig-soil")["environment_id"], "soil-excavation")

    def test_catalogue(self):
        from gym_anything_robotics import tasks
        from soil_excavation.layout import TASKS
        from soil_excavation.variants import task_resolve
        cat = tasks.catalog("soil-excavation")
        self.assertEqual(cat["count"], 11)
        r = task_resolve(dict(task="cable-trench", seed=7))
        self.assertEqual(r["task_id"], "soil-excavation/v1/trench/cable-trench/seed-7")
        self.assertIn(r["split"], ("train", "validation", "test"))
        r = task_resolve(dict(task="planting-hole", seed=3, moisture="wet"))
        self.assertTrue(r["task_id"].endswith("/moisture-wet"))
        self.assertEqual(len(TASKS), 11)
        with self.assertRaises(ValueError):
            task_resolve(dict(task="planting-hole", seed=10000))
        with self.assertRaises(ValueError):
            task_resolve(dict(task="planting-hole", moisture="soggy"))

    def test_robot_validation(self):
        from soil_excavation.robot_setup import validate, check_task
        self.assertEqual(validate("mobile-fr3-duo"), "mobile-fr3-duo")
        self.assertEqual(validate("soil-mobile-fr3-single"), "mobile-fr3-single")
        for bad in ("g1", "dual-fr3"):
            with self.assertRaises(ValueError):
                validate(bad)
        check_task("mobile-fr3-single", "expose-line")
        with self.assertRaises(ValueError):
            check_task("mobile-fr3-single", "penetrate-marked")       # two-handed spade


class Layout(unittest.TestCase):
    def test_every_task_builds_a_spec(self):
        from soil_excavation.layout import TASKS, setup, ground_boxes
        for t in TASKS:
            sp = setup(t, 0)
            self.assertEqual(sp["task"], t)
            b = sp["block"]
            self.assertLess(b["lo"][2], b["hi"][2])
            (tx0, ty0), (tx1, ty1) = sp["tarp"]
            # the tarp never overlaps the soil block
            self.assertTrue(tx1 <= b["lo"][0] or tx0 >= b["hi"][0] or ty1 <= b["lo"][1] or ty0 >= b["hi"][1], t)
            names = [g["name"] for g in ground_boxes(sp["preset"])]
            self.assertIn("block_floor", names)

    def test_seeds_are_deterministic(self):
        from soil_excavation.layout import setup
        a, b = setup("break-hardpan", 5), setup("break-hardpan", 5)
        self.assertEqual(a["target_square"], b["target_square"])
        self.assertEqual([i["pose"] for i in a["items"].values()], [i["pose"] for i in b["items"].values()])


class Materials(unittest.TestCase):
    def test_bank_strength_and_loose_state(self):
        from soil_excavation import materials as M
        for n in M.NAMES:
            sp = M.solver_params(n)
            H, J0, Jl = M.hardening(n)
            h = lambda J: 1.0 if H == 0 else math.sinh(-H*math.log(min(J, 1.0)))
            self.assertAlmostEqual(h(J0), 1.0, places=9, msg=n)        # bank: the targets as given
            if n not in M.GRANULAR:
                from soil_excavation import params as P
                self.assertAlmostEqual(h(Jl), P.LOOSE_STRENGTH, places=9, msg=n)
            self.assertGreater(sp["attrs"]["friction"], 0.0)

    def test_moisture_scales_cohesion(self):
        from soil_excavation import materials as M
        dry, moist = M.solver_params("topsoil", "dry"), M.solver_params("topsoil", "moist")
        self.assertLess(dry["attrs"]["yield_stress"], moist["attrs"]["yield_stress"])
        for clay in ("subsoil", "hardpan"):
            self.assertGreater(M.solver_params(clay, "dry")["attrs"]["yield_stress"],
                               M.solver_params(clay, "moist")["attrs"]["yield_stress"])


class CutKernel(unittest.TestCase):


    def test_cut_does_not_remold_below_edge_or_past_sides(self):
        import warp as wp
        from soil_excavation.soil import k_cut
        wp.init()
        # Plate runs downwards. Inside, ahead, past side, and outside the thickness band.
        points = [[0, 0, -.099], [0, 0, -.101], [0, .101, 0], [.02, 0, 0]]
        def arr(x, dtype): return wp.array(x, dtype=dtype, device="cpu")
        jp = arr([.5]*4, float)
        args = [arr(points, wp.vec3), jp, arr([.9]*4, float),
                arr([[0, 0, 0, 0, 0, 0, 1]], wp.transform),
                arr([[0, 0, -.1, 0, 0, 0]], wp.spatial_vector), arr([[0, 0, .4]], wp.vec3),
                arr([0], int), arr([[0, 0, 0, 0, 0, 0, 1]], wp.transform),
                arr([[.002, .1, .1]], wp.vec3), arr([0], int), arr([[0, 0, -1]], wp.vec3),
                1, .01, .02]
        wp.launch(k_cut, dim=4, inputs=args, device="cpu")
        np.testing.assert_allclose(jp.numpy(), [.9, .5, .5, .5])
        # Withdrawal does not cut, including at the floor.
        jp.fill_(.5)
        args[4] = arr([[0, 0, .1, 0, 0, 0]], wp.spatial_vector)
        wp.launch(k_cut, dim=4, inputs=args, device="cpu")
        np.testing.assert_allclose(jp.numpy(), [.5]*4)
        # A rotating blade's COM is offset from its origin. That offset supplies the edge's downward speed.
        args[4] = arr([[0, 0, 0, 0, -1, 0]], wp.spatial_vector)
        args[5] = arr([[.4, 0, 0]], wp.vec3)
        wp.launch(k_cut, dim=4, inputs=args, device="cpu")
        np.testing.assert_allclose(jp.numpy(), [.9, .5, .5, .5])


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "builds the scene and the GPU soil (set GAR_GPU_TESTS=1)")
class PublicPath(unittest.TestCase):
    def test_backend(self):
        import tempfile
        from gym_anything_robotics import registry
        from gym_anything_robotics.actions import BaseTwist, Gripper
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        with self.assertRaises((UnsupportedConfigurationError, ValueError)):
            registry.create("soil-excavation", task="penetrate-marked", robot="g1", cameras=())
        with self.assertRaises((UnsupportedConfigurationError, ValueError)):
            registry.create("soil-excavation", task="penetrate-marked", robot="mobile-fr3-single", cameras=())
        tmp = tempfile.TemporaryDirectory()
        env = registry.create("soil-excavation", task="penetrate-marked", seed=0, cameras=(), output_dir=tmp.name + "/s")
        b = env._backend
        b.act(BaseTwist(0.0, 0.0, 0.0, duration_s=0.1))
        ev = b.evaluate()
        self.assertFalse(ev["success"])                              # the spade starts above the mark
        cp = b.save_checkpoint(label="a")
        b.act(Gripper("right", half_width_m=0.04, duration_s=0.1))
        b.restore_checkpoint(label="a")
        self.assertEqual(b.env.world.tick, cp["tick"])
        idx = b.get_geometry_index()
        self.assertIn("soil", idx)
        env.close()


if __name__ == "__main__":
    unittest.main()

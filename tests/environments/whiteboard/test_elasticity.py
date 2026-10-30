import unittest
import numpy as np
from whiteboard.elasticity import Solid, box_mesh, tool_solid


class ElasticityTests(unittest.TestCase):
    def test_condensed_columns_preserve_scalar_contractions(self):
        rng = np.random.default_rng(310)
        blocks = {i: rng.normal(size=(33, 3, 3)) for i in range(39)}
        rows = np.array([0, 3, 5, 12, 17, 25, 31])
        nodes = np.arange(39)
        left = rng.normal(size=(len(rows), 3))
        right = rng.normal(size=(len(nodes), 3))
        reference = np.stack(
            [
                np.einsum("ic,ic->i", left, np.einsum("icd,d->ic", blocks[int(node)][rows], vector))
                for node, vector in zip(nodes, right)
            ],
            axis=1,
        )
        actual = Solid._normal_columns(blocks, nodes, rows, left, right)
        np.testing.assert_array_equal(actual, reference)
        previous = np.zeros(len(rows))
        for column in reference.T:
            previous += column
        np.testing.assert_array_equal(np.add.accumulate(actual, axis=1)[:, -1], previous)
        from collections import OrderedDict

        cache = OrderedDict()
        for factor in (1.0, 2.0, -0.25):
            changed = Solid._normal_columns(
                blocks, nodes, rows, left, factor * right, packed_cache=cache
            )
            oracle = Solid._normal_columns(blocks, nodes, rows, left, factor * right)
            np.testing.assert_array_equal(changed, oracle)
        self.assertEqual(len(cache), 1)

    def test_specialized_cross_preserves_broadcasting_and_values(self):
        from whiteboard.deformable_contact import cross3

        rng = np.random.default_rng(712)
        for shape_a, shape_b in (((3,), (3,)), ((20, 3), (3,)), ((20, 1, 3), (1, 8, 3))):
            a, b = rng.normal(size=shape_a), rng.normal(size=shape_b)
            np.testing.assert_array_equal(cross3(a, b), np.cross(a, b))

    def test_affine_patch_energy(self):
        x, t = box_mesh((0.02, 0.03, 0.04), (2, 2, 2))
        s = Solid(x, t, np.isclose(x[:, 2], -0.02), 1e6, 0.25)
        strain = np.array([0.001, -0.002, 0.003, 0, 0, 0])
        u = x * strain[:3]
        np.testing.assert_allclose(s.strain(u), np.tile(strain, (len(t), 1)), atol=1e-14)
        expected = 0.5 * 0.02 * 0.03 * 0.04 * strain @ s.constitutive @ strain
        self.assertAlmostEqual(s.energy(u), expected, places=12)
        self.assertLess(abs(s.energy(np.ones_like(x) * 0.001)), 1e-12)

    def test_axial_compression_and_contact(self):
        x, t = box_mesh((0.01, 0.01, 0.02), (2, 2, 3))
        s = Solid(x, t, np.isclose(x[:, 2], -0.01), 1e6, 0)
        nodes = np.flatnonzero(np.isclose(x[:, 2], 0.01))
        u, force, check = s.contact(
            nodes, np.tile([0, 0, -1], (len(nodes), 1)), np.full(len(nodes), 0.0001)
        )
        self.assertAlmostEqual(-force[:, 2].sum(), 0.5, places=10)
        self.assertAlmostEqual(s.energy(u), 0.5 * 0.5 * 0.0001, places=12)
        self.assertLess(check["residual_m"], 2e-9)
        self.assertGreater(np.max(np.linalg.norm(u, axis=1)), 0)
        np.testing.assert_array_equal(u[s.fixed], 0)

    def test_lazy_field_reconstructs_same_force_displacement_and_energy(self):
        from whiteboard.deformable_contact import displacement

        s = tool_solid("cap", 1)
        radius = np.linalg.norm(s.rest[:, :2], axis=1)
        nodes = np.array(
            [
                i
                for i in np.unique(s.faces)
                if radius[i] < 0.008 and not s.fixed[i] and s.rest[i, 2] <= 0
            ],
            int,
        )
        normals = np.c_[s.rest[nodes, :2] / radius[nodes, None], np.zeros(len(nodes))]
        for offset in (0.0, 0.00002, -0.00002):
            p = 0.007 - radius[nodes] + offset
            full, f1, q1 = s.contact(nodes, normals, p)
            lazy, f2, q2 = s.contact(nodes, normals, p, resolve_displacement=False)
            self.assertIsNone(lazy)
            np.testing.assert_array_equal(f1, f2)
            reconstructed = displacement(s, {"force_nodes": nodes, "force_local_n": f2})
            np.testing.assert_allclose(full, reconstructed, atol=1e-14, rtol=1e-12)
            self.assertAlmostEqual(s.energy(full), q2["elastic_energy_j"], places=12)

    def test_meshes_are_connected_bounded_and_elastic(self):
        for kind in ("nib", "felt", "cap"):
            s = tool_solid(kind)
            self.assertTrue(np.all(s.volumes > 0))
            difference = s.K - s.K.T
            self.assertLess(np.max(np.abs(difference.data), initial=0), 1e-8)
            node = np.flatnonzero(~s.fixed)[0]
            f = np.zeros_like(s.rest)
            f[node, 0] = 0.001
            u = s.displacement(f)
            self.assertGreater(s.energy(u), 0)
            self.assertAlmostEqual(s.energy(u) * 2, float((f * u).sum()), places=9)
            self.assertGreater(len(s.tetra), 20)

    def test_mobility_warm_start_and_eviction_preserve_solution(self):
        x, t = box_mesh((0.01, 0.01, 0.02), (2, 2, 3))
        s = Solid(x, t, np.isclose(x[:, 2], -0.01), 1e6, 0.2)
        nodes = np.flatnonzero(np.isclose(x[:, 2], 0.01))
        normals = np.tile([0.0, 0.0, -1.0], (len(nodes), 1))
        jac = np.c_[np.ones(len(nodes)), x[nodes, :2]]
        mass_inverse = np.diag([1e-7, 2e-5, 2e-5])
        penetration = 0.0001 + x[nodes, 0] * 0.01
        full, force, quality = s.contact(
            nodes, normals, penetration, mobility=jac @ mass_inverse @ jac.T
        )
        # An incomplete previous support must discover all remaining contacts.
        # A tiny cache must not evict references used by the current solve.
        s._contact_cache_budget = 1
        lazy, other, check = s.contact(
            nodes,
            normals,
            penetration,
            mobility=(jac, mass_inverse),
            resolve_displacement=False,
            warm_start_nodes=nodes[:1],
        )
        self.assertIsNone(lazy)
        np.testing.assert_allclose(force, other, atol=1e-10, rtol=1e-10)
        self.assertLess(check["residual_m"], 2e-9)
        self.assertAlmostEqual(s.energy(full), check["elastic_energy_j"], places=12)
        self.assertLessEqual(s._contact_cache_bytes, 1)


if __name__ == "__main__":
    unittest.main()

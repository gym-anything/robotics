import unittest
from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from whiteboard.solver import (
    _CONTACT_KEYS,
    contact_order,
    canonical_contacts,
    native_ccd_compilation,
)


class ContactOrderTests(unittest.TestCase):
    def test_scoped_compilation_bounds_and_factory_restoration(self):
        import warp as wp
        from mujoco_warp._src import collision_convex, collision_driver

        ccd = SimpleNamespace(
            module=SimpleNamespace(
                options={"deterministic": wp.DeterministicMode.RUN_TO_RUN}, mark_modified=Mock()
            )
        )
        broad = SimpleNamespace(
            module=SimpleNamespace(
                options={"deterministic_max_records": 6141}, mark_modified=Mock()
            )
        )
        ccd_factory, broad_factory = Mock(return_value=ccd), Mock(return_value=broad)
        with (
            patch.object(collision_convex, "ccd_kernel_builder", ccd_factory),
            patch.object(collision_driver, "_nxn_broadphase", broad_factory),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with native_ccd_compilation():
                    self.assertIs(collision_convex.ccd_kernel_builder(), ccd)
                    self.assertIs(collision_driver._nxn_broadphase(), broad)
                    self.assertEqual(
                        ccd.module.options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED
                    )
                    self.assertEqual(broad.module.options["deterministic_max_records"], 1)
                    raise RuntimeError("fixture")
            self.assertIs(collision_convex.ccd_kernel_builder, ccd_factory)
            self.assertIs(collision_driver._nxn_broadphase, broad_factory)
        ccd.module.mark_modified.assert_called_once()
        broad.module.mark_modified.assert_called_once()

    def values(self):
        values = {name: np.zeros(4, np.float32) for name in _CONTACT_KEYS}
        values["worldid"] = np.array([1, 0, 0, 0], np.int32)
        values["geom"] = np.array([[1, 5], [2, 3], [2, 3], [1, 5]], np.int32)
        values["pos"] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]], np.float32)
        return values

    def test_order_is_independent_of_atomic_slot_arrival(self):
        values = self.values()
        order = contact_order(values, 4)
        expected = {name: value[order] for name, value in values.items()}
        rng = np.random.default_rng(529)
        for _ in range(20):
            permutation = rng.permutation(4)
            shuffled = {name: value[permutation] for name, value in values.items()}
            actual = contact_order(shuffled, 4)
            for name, value in shuffled.items():
                np.testing.assert_array_equal(value[actual], expected[name])

    def test_invalid_contact_count_and_nonfinite_values_are_rejected(self):
        values = self.values()
        with self.assertRaises(ValueError):
            contact_order(values, 5)
        with self.assertRaises(ValueError):
            contact_order(values, -1)
        values["dist"][0] = np.nan
        with self.assertRaises(FloatingPointError):
            contact_order(values, 4)

    def test_empty_and_inactive_tail_are_not_fabricated_contacts(self):
        values = self.values()
        values["dist"][3] = np.nan
        self.assertEqual(len(contact_order(values, 0)), 0)
        order = contact_order(values, 3)
        self.assertEqual(sorted(order.tolist()), [0, 1, 2])

    def test_canonical_copy_keeps_metadata_aligned_and_allows_absent_flex_arrays(self):
        import warp as wp

        values = self.values()
        order = contact_order(values, 4)
        values["geomcollisionid"] = np.arange(90, 94, dtype=np.int32)
        values.update({name: np.empty((0, 2), np.int32) for name in ("flex", "elem", "vert")})
        native = {name: wp.array(value, device="cpu") for name, value in values.items()}
        for name in ("flex", "elem", "vert"):
            native[name] = wp.empty(0, dtype=wp.vec2i, device="cpu")
        contact = make_dataclass("Contact", [(name, object) for name in native])(**native)
        data = SimpleNamespace(
            nacon=wp.array([4], dtype=int, device="cpu"), naconmax=4, contact=contact
        )
        canonical_contacts(SimpleNamespace(nflex=0), data)
        for name in _CONTACT_KEYS + ("geomcollisionid",):
            np.testing.assert_array_equal(native[name].numpy(), values[name][order])
        self.assertEqual(native["flex"].shape, (0,))

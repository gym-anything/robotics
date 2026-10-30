"""Pure configuration regressions; no simulation is advanced."""
from dataclasses import replace
import unittest

from deposit_cans.config import Config,resolved_contact_capacity


class ContactCapacityTest(unittest.TestCase):
    def test_full_surface_capacity_scales_for_high_fill_inventory(self):
        reference=Config(contact_capacity=65536)
        high_fill=replace(reference,preload_count=120)
        self.assertEqual(resolved_contact_capacity(reference),65536)
        self.assertEqual(resolved_contact_capacity(high_fill),131072)


if __name__=='__main__':unittest.main()

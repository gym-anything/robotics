from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football.net_model import CableNetPatch, CableNetPrior
from football.net_state_cache import settle_or_load


class NetStateCacheTests(unittest.TestCase):
    def test_cache_is_keyed_and_reproduces_node_state(self) -> None:
        prior = CableNetPrior(mesh_spacing_m=0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settled.json"
            first_net = CableNetPatch(width_m=0.6, height_m=0.6, prior=prior)
            first = settle_or_load(
                first_net, path, duration_s=0.002, physics_hz=1000
            )
            expected = [node.position_m for node in first_net.nodes]
            second_net = CableNetPatch(width_m=0.6, height_m=0.6, prior=prior)
            second = settle_or_load(
                second_net, path, duration_s=0.002, physics_hz=1000
            )
            self.assertEqual(first["source"], "computed and cached")
            self.assertEqual(second["source"], "validated deterministic cache")
            self.assertEqual(expected, [node.position_m for node in second_net.nodes])


if __name__ == "__main__":
    unittest.main()

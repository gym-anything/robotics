from __future__ import annotations

import unittest

import numpy as np

from laundry.api import BackendDescriptor, LaundryPhysicsBackend, ReplayLaundryBackend
from laundry.control import FIXTURE_VELOCITY_ACTION_SCHEMA
from laundry.mesh import build_rectangular_mesh
from laundry.observation import LaundryObservation, ToolState


class BackendApiTests(unittest.TestCase):
    def test_replay_backend_satisfies_shared_contract(self) -> None:
        mesh = build_rectangular_mesh(
            length_m=0.6,
            width_m=0.4,
            nx=4,
            ny=2,
            areal_density_kg_m2=0.3,
            origin_m=(0.0, 0.0, 0.0004),
        )
        tool = ToolState(
            "gripper",
            (0.0, 0.0, 0.1),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        observation = LaundryObservation(
            time_s=0.0,
            dt_s=1.0 / 60.0,
            particle_positions_m=mesh.positions_m,
            particle_velocities_m_s=np.zeros_like(mesh.positions_m),
            material_coordinates_m=mesh.material_coordinates_m,
            tools=(tool,),
            pinched_particle_count=0,
            jaw_normal_preload_n=0.0,
            roller_normal_force_n=0.0,
            roller_rolling_force_n=0.0,
            maximum_structural_strain=0.0,
            nonlocal_intersection_count=0,
            runtime_pose_writes=0,
            external_particle_force_applications=0,
            fixtures_clear=True,
            released=True,
        )
        descriptor = BackendDescriptor(
            id="replay-test",
            garment_profile_id="canonical_cotton_towel",
            particle_count=mesh.vertex_count,
            tool_names=("gripper",),
            action_schema=FIXTURE_VELOCITY_ACTION_SCHEMA,
        )
        backend = ReplayLaundryBackend(descriptor, (observation, observation))
        self.assertIsInstance(backend, LaundryPhysicsBackend)
        self.assertIs(backend.reset(seed=4), observation)
        backend.step(np.zeros(6, dtype=np.float32))
        self.assertEqual(len(backend.actions), 1)
        backend.close()
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)

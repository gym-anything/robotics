from __future__ import annotations

import unittest

import numpy as np

from laundry.episode import EpisodePhase, HalfFoldEpisodeTask
from laundry.mesh import build_rectangular_mesh
from laundry.observation import LaundryObservation, ToolState


IDENTITY = (1.0, 0.0, 0.0, 0.0)
ZERO3 = (0.0, 0.0, 0.0)


class HalfFoldEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = build_rectangular_mesh(
            length_m=0.6,
            width_m=0.4,
            nx=12,
            ny=8,
            areal_density_kg_m2=0.3,
            origin_m=(0.0, 0.0, 0.0004),
        )
        self.task = HalfFoldEpisodeTask(self.mesh, thickness_m=0.0008)
        self.tools = (
            ToolState("gripper", ZERO3, IDENTITY, ZERO3, ZERO3),
            ToolState("crease_roller", (0.0, 0.0, 0.45), IDENTITY, ZERO3, ZERO3),
        )

    def observation(
        self,
        time_s: float,
        positions: np.ndarray,
        *,
        pinched: int = 0,
        released: bool = True,
        clear: bool = False,
        intersections: int = 0,
        strain: float = 0.01,
    ) -> LaundryObservation:
        return LaundryObservation(
            time_s=time_s,
            dt_s=0.1,
            particle_positions_m=positions.copy(),
            particle_velocities_m_s=np.zeros_like(positions),
            material_coordinates_m=self.mesh.material_coordinates_m.copy(),
            tools=self.tools,
            pinched_particle_count=pinched,
            jaw_normal_preload_n=2.0 if pinched else 0.0,
            roller_normal_force_n=0.0,
            roller_rolling_force_n=0.0,
            maximum_structural_strain=strain,
            nonlocal_intersection_count=intersections,
            runtime_pose_writes=0,
            external_particle_force_applications=0,
            fixtures_clear=clear,
            released=released,
        )

    def perfect_fold(self) -> np.ndarray:
        positions = self.mesh.positions_m.copy()
        left = self.mesh.material_coordinates_m[:, 0] < 0.0
        positions[left, 0] *= -1.0
        positions[left, 2] += 0.0008
        positions[:, 0] += 0.020
        return positions

    def test_success_requires_grasp_release_clearance_and_geometry(self) -> None:
        flat = self.mesh.positions_m
        initial = self.task.update(self.observation(0.0, flat))
        self.assertEqual(initial.phase, EpisodePhase.READY)
        grasp = self.task.update(
            self.observation(0.1, flat, pinched=8, released=False)
        )
        self.assertEqual(grasp.phase, EpisodePhase.GRASPED)
        held_fold = self.task.update(
            self.observation(
                0.2,
                self.perfect_fold(),
                pinched=8,
                released=False,
            )
        )
        self.assertEqual(held_fold.phase, EpisodePhase.PLACED)
        released = self.task.update(
            self.observation(
                0.3,
                self.perfect_fold(),
                released=True,
                clear=True,
            )
        )
        self.assertTrue(released.success)
        self.assertTrue(released.terminated)
        self.assertFalse(released.truncated)
        self.assertEqual(released.phase, EpisodePhase.SUCCESS)
        self.assertGreater(released.reward, 4.0)

    def test_intersection_is_a_history_preserving_hard_failure(self) -> None:
        self.task.update(
            self.observation(0.0, self.mesh.positions_m, pinched=8, released=False)
        )
        failure = self.task.update(
            self.observation(
                0.1,
                self.perfect_fold(),
                intersections=1,
                released=False,
            )
        )
        self.assertTrue(failure.terminated)
        self.assertFalse(failure.success)
        self.assertEqual(failure.phase, EpisodePhase.FAILED)
        self.assertIn("self-intersection", failure.info["failure_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

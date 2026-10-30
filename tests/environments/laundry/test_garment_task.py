from __future__ import annotations

import unittest

import numpy as np

from laundry.garment_assembly import weld_garment_topology
from laundry.garment_meshing import compile_garment_pattern_mesh
from laundry.garment_patterns import GARMENT_CONSTRUCTIONS
from laundry.garment_task import (
    GARMENT_TASK_SPECS,
    GarmentEpisodePhase,
    GarmentEpisodeState,
    GarmentEpisodeTask,
    get_garment_task_spec,
)


class SemanticGarmentTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bindings = {}
        for construction in GARMENT_CONSTRUCTIONS.values():
            pattern = compile_garment_pattern_mesh(
                construction,
                target_edge_length_m=0.040,
            )
            topology = weld_garment_topology(construction, pattern, relax=False)
            cls.bindings[construction.id] = (construction, pattern, topology)

    def test_one_distinct_semantic_benchmark_is_bound_for_every_garment(self) -> None:
        self.assertEqual(
            {construction_id for construction_id, _ in GARMENT_TASK_SPECS},
            set(GARMENT_CONSTRUCTIONS),
        )
        for (construction_id, task_id), spec in GARMENT_TASK_SPECS.items():
            with self.subTest(construction=construction_id, task=task_id):
                evaluator = spec.bind(*self.bindings[construction_id])
                self.assertEqual(evaluator.index.particle_count, self.bindings[construction_id][2].physical_vertex_count)

    def test_unfolded_state_does_not_pass_fold_tasks(self) -> None:
        for (construction_id, task_id), spec in GARMENT_TASK_SPECS.items():
            if task_id == "tie_untangle":
                continue
            construction, pattern, topology = self.bindings[construction_id]
            evaluator = spec.bind(construction, pattern, topology)
            result = evaluator.evaluate(
                topology.physical_positions_m,
                np.zeros_like(topology.physical_positions_m),
                maximum_structural_strain=0.0,
                nonlocal_intersection_count=0,
                runtime_pose_writes=0,
                external_particle_force_applications=0,
                released=True,
                fixtures_clear=True,
            )
            with self.subTest(construction=construction_id, task=task_id):
                self.assertFalse(result.success)
                self.assertTrue(any(name.startswith("goal:") for name in result.failed_gates))
                self.assertGreaterEqual(result.potential, 0.0)
                self.assertLessEqual(result.potential, 1.0)

    def test_task_integrity_gates_reject_pose_and_force_cheats(self) -> None:
        construction_id = "asymmetric_draped_one_sleeve_top"
        construction, pattern, topology = self.bindings[construction_id]
        evaluator = get_garment_task_spec(construction_id, "tie_untangle").bind(
            construction, pattern, topology
        )
        result = evaluator.evaluate(
            topology.physical_positions_m,
            np.zeros_like(topology.physical_positions_m),
            maximum_structural_strain=0.0,
            nonlocal_intersection_count=0,
            runtime_pose_writes=1,
            external_particle_force_applications=2,
            released=True,
            fixtures_clear=True,
        )
        self.assertFalse(result.success)
        self.assertFalse(result.gates["integrity:no_runtime_pose_writes"])
        self.assertFalse(result.gates["integrity:no_direct_particle_forces"])

    def test_analytical_towel_oracle_passes_without_weakening_integrity_gates(self) -> None:
        construction_id = "terry_cotton_bath_towel"
        construction, pattern, topology = self.bindings[construction_id]
        evaluator = get_garment_task_spec(construction_id, "half_fold").bind(
            construction, pattern, topology
        )
        positions = topology.physical_positions_m.copy()
        fold_axis_x = 0.5 * (
            float(np.min(positions[:, 0])) + float(np.max(positions[:, 0]))
        )
        moving = positions[:, 0] < fold_axis_x - 1.0e-9
        positions[moving, 0] = 2.0 * fold_axis_x - positions[moving, 0]
        positions[moving, 2] += 3.2e-3
        result = evaluator.evaluate(
            positions,
            np.zeros_like(positions),
            maximum_structural_strain=0.0,
            nonlocal_intersection_count=0,
            runtime_pose_writes=0,
            external_particle_force_applications=0,
            released=True,
            fixtures_clear=True,
        )
        self.assertTrue(result.success)
        cheated = evaluator.evaluate(
            positions,
            np.zeros_like(positions),
            maximum_structural_strain=0.0,
            nonlocal_intersection_count=0,
            runtime_pose_writes=1,
            external_particle_force_applications=0,
            released=True,
            fixtures_clear=True,
        )
        self.assertFalse(cheated.success)
        self.assertFalse(cheated.gates["integrity:no_runtime_pose_writes"])

    def test_wrong_construction_and_unknown_task_are_rejected(self) -> None:
        tee = self.bindings["cotton_jersey_half_sleeve_tee"]
        shirt_spec = get_garment_task_spec("woven_button_down_shirt", "sleeve_fold")
        with self.assertRaises(ValueError):
            shirt_spec.bind(*tee)
        with self.assertRaises(KeyError):
            get_garment_task_spec("cotton_jersey_half_sleeve_tee", "rectangle_fold")

    def test_episode_requires_grasp_release_history_and_uses_delta_potential(self) -> None:
        construction_id = "asymmetric_draped_one_sleeve_top"
        construction, pattern, topology = self.bindings[construction_id]
        evaluator = get_garment_task_spec(construction_id, "tie_untangle").bind(
            construction, pattern, topology
        )
        episode = GarmentEpisodeTask(evaluator)

        def state(time_s: float, pinched: int, released: bool) -> GarmentEpisodeState:
            return GarmentEpisodeState(
                time_s=time_s,
                particle_positions_m=topology.physical_positions_m,
                particle_velocities_m_s=np.zeros_like(topology.physical_positions_m),
                maximum_structural_strain=0.0,
                nonlocal_intersection_count=0,
                runtime_pose_writes=0,
                external_particle_force_applications=0,
                pinched_particle_count=pinched,
                released=released,
                fixtures_clear=True,
            )

        ready = episode.update(state(0.0, 0, True))
        self.assertFalse(ready.success)
        self.assertEqual(ready.phase, GarmentEpisodePhase.READY)
        grasp = episode.update(state(0.1, 4, False), action=np.zeros(14))
        self.assertFalse(grasp.success)
        self.assertGreater(grasp.reward, 0.0)
        released = episode.update(state(0.2, 0, True), action=np.zeros(14))
        self.assertTrue(released.success)
        self.assertTrue(released.terminated)
        self.assertEqual(released.phase, GarmentEpisodePhase.SUCCESS)
        self.assertGreater(released.reward, 5.0)

    def test_episode_hard_fails_on_direct_particle_force(self) -> None:
        construction_id = "asymmetric_draped_one_sleeve_top"
        construction, pattern, topology = self.bindings[construction_id]
        evaluator = get_garment_task_spec(construction_id, "tie_untangle").bind(
            construction, pattern, topology
        )
        episode = GarmentEpisodeTask(evaluator, require_grasp_history=False)
        state = GarmentEpisodeState(
            time_s=0.0,
            particle_positions_m=topology.physical_positions_m,
            particle_velocities_m_s=np.zeros_like(topology.physical_positions_m),
            maximum_structural_strain=0.0,
            nonlocal_intersection_count=0,
            runtime_pose_writes=0,
            external_particle_force_applications=1,
            pinched_particle_count=0,
            released=True,
            fixtures_clear=True,
        )
        result = episode.update(state)
        self.assertTrue(result.terminated)
        self.assertFalse(result.success)
        self.assertEqual(result.phase, GarmentEpisodePhase.FAILED)
        self.assertIn("direct external particle force", result.info["failure_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

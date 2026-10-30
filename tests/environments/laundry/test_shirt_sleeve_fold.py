"""Constructed-state checks of the shirt sleeve-fold verifier and score (no physics, no GPU).

States start from the committed P13 shirt reset with the workcell's Z lift. Folded states are geometric
constructions (sleeves turned over the side seams, cloth layers 7 mm apart, the measured settled spacing).
They test what the verifier accepts and rejects; they are not claims that a robot produced them.
"""
from __future__ import annotations

import unittest

import numpy as np

from laundry.garment_task import (
    GARMENT_TASK_SPECS,
    PanelContainmentGoal,
    PanelStackGoal,
    SleeveOnBodyGoal,
    TabletopFrame,
    get_garment_task_spec,
)
from laundry.newton15.config import Config
from laundry.newton15.garment import load

LAYER = 0.007


class ShirtSleeveFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = Config(garment="shirt")
        construction, pattern, topology, positions = load(config)
        lift = config.table_top + config.initial_table_clearance - float(
            np.min(positions[:, 2] - topology.physical_vertex_radius_m)
        )
        cls.reset = positions + np.array([0.0, 0.0, lift])
        cls.table = TabletopFrame(config.table_top, config.table_half_x, config.table_half_y)
        cls.base = config.table_top + 0.002
        spec = get_garment_task_spec("woven_button_down_shirt", "sleeve_fold")
        cls.evaluator = spec.bind(construction, pattern, topology, reference_positions_m=cls.reset)
        index = cls.evaluator.index
        cls.front = index.panel_vertices(("left_front", "right_front"))
        cls.back = index.panel_vertices(("back",))
        body = index.panel_vertices(("left_front", "right_front", "back"))
        cls.body = body
        cls.moving = {
            side: np.setdiff1d(index.panel_vertices((f"{side}_sleeve", f"{side}_cuff")), body)
            for side in ("left", "right")
        }
        cls.cuff = {side: index.panel_vertices((f"{side}_cuff",)) for side in ("left", "right")}
        cls.seam_x = {"left": float(cls.reset[body, 0].min()), "right": float(cls.reset[body, 0].max())}

    def evaluate(self, positions, *, history=None, intersections=0, released=True, stretch=(1.0, 1.0)):
        recent = np.stack([positions, positions]) if history is None else history
        return self.evaluator.evaluate(
            positions,
            np.zeros_like(positions),
            maximum_structural_strain=max(abs(stretch[0] - 1.0), abs(stretch[1] - 1.0)),
            nonlocal_intersection_count=intersections,
            runtime_pose_writes=0,
            external_particle_force_applications=0,
            released=released,
            fixtures_clear=released,
            tabletop=self.table,
            recent_positions_m=recent,
            recent_span_s=0.5,
            principal_stretch_range=stretch,
        )

    def flat(self) -> np.ndarray:
        y = self.reset.copy()
        y[self.back, 2] = self.base
        y[self.front, 2] = self.base + LAYER
        for side in ("left", "right"):
            z = self.reset[self.moving[side], 2]
            y[self.moving[side], 2] = self.base + LAYER * (z - z.min()) / np.ptp(z)
        return y

    def folded(self, *, behind=False, sides=("left", "right")) -> np.ndarray:
        """Each sleeve turned over its side-seam line like a page and laid on the front (or under the back)."""
        y = self.flat()
        for side in sides:
            m = self.moving[side]
            sign = -1.0 if side == "left" else 1.0
            outside = sign * (y[m, 0] - self.seam_x[side]) > 0.0
            z = self.reset[m, 2]
            rank = (z - z.min()) / np.ptp(z)
            y[m[outside], 0] = 2.0 * self.seam_x[side] - y[m[outside], 0]
            layer = np.where(outside, 2.0 + (1.0 - rank), rank)
            y[m, 2] = self.base + LAYER * (layer if not behind else np.where(outside, 1.0 - rank, rank))
        if behind:
            y[self.body, 2] += 2.0 * LAYER
        elif len(sides) == 2:
            over = np.linalg.norm(y[self.moving["right"], None, :2] - y[None, self.moving["left"], :2], axis=2).min(1) < 0.04
            y[self.moving["right"][over], 2] += 2.0 * LAYER
        return y

    def test_untouched_reset_scores_zero_and_fails(self) -> None:
        result = self.evaluate(self.reset)
        self.assertFalse(result.success)
        self.assertEqual(result.potential, 0.0)
        self.assertIn("goal:left_sleeve_and_cuff_on_body:cuff_on_body", result.failed_gates)

    def test_valid_folds_get_exactly_full_credit(self) -> None:
        for name, state in (("front, crossed", self.folded()), ("behind the body", self.folded(behind=True))):
            with self.subTest(state=name):
                result = self.evaluate(state)
                self.assertTrue(result.success, result.failed_gates)
                self.assertEqual(result.potential, 1.0)

    def test_one_finished_sleeve_is_partial_and_below_success(self) -> None:
        result = self.evaluate(self.folded(sides=("left",)))
        self.assertFalse(result.success)
        self.assertGreater(result.progress, 0.35)
        self.assertLess(result.progress, 0.65)
        self.assertLess(result.potential, 0.9 + 1e-12)

    def test_cuff_left_past_the_hem_fails(self) -> None:
        y = self.folded()
        for side in ("left", "right"):
            cuff = self.cuff[side]
            y[cuff, 1] = float(y[self.body, 1].min()) - 0.06
            y[cuff, 2] = self.base
        result = self.evaluate(y)
        self.assertFalse(result.success)
        self.assertIn("goal:right_sleeve_and_cuff_on_body:cuff_on_body", result.failed_gates)
        self.assertLess(result.potential, 1.0)

    def test_off_table_moving_and_crossing_states_fail(self) -> None:
        floor = self.folded()
        floor[:, 0] += 1.25
        floor[:, 2] -= 0.78
        result = self.evaluate(floor)
        self.assertFalse(result.success)
        self.assertFalse(result.gates["state:on_table"])
        self.assertEqual(result.potential, 0.0)
        valid = self.folded()
        moving = self.evaluate(valid, history=np.stack([valid + [0.0, 0.0, 0.012], valid]))
        self.assertFalse(moving.gates["state:at_rest"])
        self.assertFalse(self.evaluate(valid, history=np.stack([valid])[:0]).gates["state:at_rest"])
        crossing = self.evaluate(valid, intersections=3)
        self.assertFalse(crossing.success)
        self.assertEqual(crossing.potential, 0.0)
        held = self.evaluate(valid, released=False)
        self.assertFalse(held.success)
        self.assertLess(held.potential, 1.0)

    def test_sleeve_enclosed_inside_the_open_front_fails(self) -> None:
        y = self.folded()
        for side in ("left", "right"):
            m = self.moving[side]
            folded = np.abs(y[m, 0] - self.reset[m, 0]) > 1.0e-9
            y[m[folded], 2] = self.base + 0.5 * LAYER
        y[self.front, 2] = self.base + 3.0 * LAYER
        result = self.evaluate(y)
        self.assertFalse(result.success)
        self.assertIn("goal:left_sleeve_and_cuff_on_body:sleeve_on_body", result.failed_gates)

    def axis_coordinate(self, y, side):
        m = self.moving[side]
        seam = y[np.intersect1d(self.evaluator.index.panel_vertices((f"{side}_sleeve",)), self.body), :2].mean(0)
        axis = y[self.cuff[side], :2].mean(0) - seam
        axis /= np.linalg.norm(axis)
        return m, (y[m, :2] - seam) @ axis, axis

    def test_sleeve_bunched_into_a_pile_fails(self) -> None:
        y = self.folded()
        for side in ("left", "right"):
            m, s, axis = self.axis_coordinate(y, side)
            u = s - 0.20
            sel = u > 0
            k = np.floor(u[sel] / 0.08).astype(int)
            r = u[sel] - k * 0.08
            s_new = 0.20 + np.where(k % 2 == 0, r, 0.08 - r)
            y[m[sel], :2] -= (s[sel] - s_new)[:, None] * axis
            y[m[sel], 2] += 2.0 * LAYER * k + (8.0 * LAYER if side == "right" else 0.0)
        result = self.evaluate(y)
        self.assertFalse(result.success)
        self.assertGreater(result.metrics.goal_metrics["left_sleeve_and_cuff_on_body"]["bunched_fraction"], 0.15)
        self.assertLess(result.potential, 0.9)

    def test_cloth_held_up_by_its_own_lower_ply_does_not_rest(self) -> None:
        y = self.folded()
        for side in ("left", "right"):
            m = self.moving[side]
            turned = np.abs(y[m, 0] - self.reset[m, 0]) > 1.0e-9
            y[m[turned], 2] += 1.0
        result = self.evaluate(y, released=False)
        metrics = result.metrics.goal_metrics["left_sleeve_and_cuff_on_body"]
        self.assertEqual(metrics["sleeve_on_body_fraction"], 0.0)
        self.assertLess(result.progress, 0.05)
        band = self.folded()
        prepared = self.evaluator._prepared["left_sleeve_and_cuff_on_body"]
        for side in ("left", "right"):
            m, s, _ = self.axis_coordinate(band, side)
            goal_sleeve = self.evaluator._prepared[f"{side}_sleeve_and_cuff_on_body"].sleeve
            order = np.argsort(s)
            cum = np.cumsum(np.isin(m[order], goal_sleeve)) / len(goal_sleeve)
            raised = m[order][(cum > 0.40) & (cum <= 0.60)]
            band[raised, 2] += 0.15
        self.assertGreater(len(prepared.sleeve), 0)
        raised_result = self.evaluate(band)
        self.assertFalse(raised_result.success)
        self.assertIn("goal:left_sleeve_and_cuff_on_body:sleeve_on_body", raised_result.failed_gates)

    def test_cuff_tip_under_the_opposite_front_is_tolerated_a_tucked_cuff_is_not(self) -> None:
        def tucked(edge):
            y = self.folded()
            m = self.moving["left"]
            folded = np.abs(y[m, 0] - self.reset[m, 0]) > 1.0e-9
            tuck = folded & (y[m, 0] > edge)
            y[m[tuck], 2] = self.base + 0.5 * LAYER
            right_front = self.evaluator.index.panel_vertices(("right_front",))
            near = np.linalg.norm(y[right_front, None, :2] - y[None, m[tuck], :2], axis=2).min(1) < 0.04
            y[right_front[near], 2] = self.base + 2.0 * LAYER
            return y
        cuff_x = self.folded()[self.cuff["left"], 0]
        tip = float(np.quantile(cuff_x, 0.92))
        self.assertTrue(self.evaluate(tucked(tip)).success)
        half = self.evaluate(tucked(float(np.median(cuff_x))))
        self.assertFalse(half.success)
        self.assertIn("goal:left_sleeve_and_cuff_on_body:cuff_on_body", half.failed_gates)

    def test_body_folded_over_the_finished_sleeves_fails_with_reduced_credit(self) -> None:
        def fold_over(y, coordinate, line, sign):
            ids = np.where(sign * (y[:, coordinate] - line) > 0)[0]
            top = float(np.max(np.delete(y[:, 2], ids)))
            y[ids, coordinate] = 2.0 * line - y[ids, coordinate]
            y[ids, 2] = top + LAYER + (float(y[ids, 2].max()) - y[ids, 2])
            return y
        sides = fold_over(fold_over(self.folded(), 0, -0.12, -1.0), 0, 0.12, 1.0)
        hem, neck = float(self.reset[self.body, 1].min()), float(self.reset[self.body, 1].max())
        hem_up = fold_over(self.folded(), 1, 0.5 * (hem + neck), -1.0)
        one_sleeve = self.evaluate(self.folded(sides=("left",))).potential
        for name, state in (("sides over sleeves", sides), ("hem over sleeves", hem_up)):
            with self.subTest(state=name):
                result = self.evaluate(state)
                self.assertFalse(result.success)
                self.assertGreater(result.potential, 0.25)
                self.assertLess(result.potential, one_sleeve)

    def test_recorded_at_rest_start_lies_on_the_table_and_reads_zero(self) -> None:
        from laundry.newton15.garment import rest_start

        config = Config(garment="shirt")
        start = rest_start(config, self.evaluator.index.topology)
        self.assertTrue(self.table.supports(start, 0.0, 0.30).all())
        evaluator = get_garment_task_spec("woven_button_down_shirt", "sleeve_fold").bind(
            self.evaluator.index.construction, self.evaluator.index.pattern_mesh, self.evaluator.index.topology,
            reference_positions_m=start,
        )
        result = evaluator.evaluate(
            start, np.zeros_like(start), maximum_structural_strain=0.0, nonlocal_intersection_count=0,
            runtime_pose_writes=0, external_particle_force_applications=0, released=True, fixtures_clear=True,
            tabletop=self.table, recent_positions_m=np.stack([start, start]), recent_span_s=0.5,
            principal_stretch_range=(1.0, 1.0),
        )
        self.assertEqual(result.potential, 0.0)
        self.assertIsNone(rest_start(Config(garment="tee"), self.evaluator.index.topology))


    def test_strain_gate_bounds_stretch_not_compression(self) -> None:
        valid = self.folded()
        self.assertTrue(self.evaluate(valid, stretch=(0.60, 1.20)).success)
        self.assertFalse(self.evaluate(valid, stretch=(0.95, 1.35)).gates["integrity:structural_strain"])

    def test_other_garment_tasks_keep_the_default_settings(self) -> None:
        for (construction_id, task_id), spec in GARMENT_TASK_SPECS.items():
            if task_id == "sleeve_fold":
                self.assertTrue(all(isinstance(goal, SleeveOnBodyGoal) for goal in spec.goals))
                continue
            with self.subTest(task=task_id):
                self.assertIsNone(spec.rest_window_s)
                self.assertTrue(spec.limit_compression)
                self.assertIsNone(spec.tabletop_support_fraction)
                self.assertFalse(spec.full_credit_requires_success)
        tee = get_garment_task_spec("cotton_jersey_half_sleeve_tee", "sleeve_tuck_fold")
        self.assertEqual(
            [type(goal) for goal in tee.goals],
            [PanelContainmentGoal, PanelStackGoal, PanelContainmentGoal, PanelStackGoal],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Outcome of half_fold (the bath towel's): the towel folded once, long edge to long edge: the two long hems together
in plan (RMS distance at most 25 mm, at most 60 mm anywhere along them) and the towel's plan footprint at most 62 % of
its flat one, strain within 20 %.

spec() gives the released spec (garment_task.GARMENT_TASK_SPECS: the bare selection {"garment": "towel"} and contract
laundry-semantic-legacy). laundry-tasks-v1 and -v2 score its goals with continuous credit and add their shared gates
(task_set.task_spec, task_set_v2.task_spec). verify(task, positions_m, velocities_m_s, **evidence) judges an episode
state with the evaluator bound to the selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain,
crossing, release, table and rest evidence) and returns its garment_task.GarmentTaskResult."""


def spec():
    """garment_task builds GARMENT_TASK_SPECS from this while it is being imported, so laundry is imported here."""
    from laundry.garment_task import GarmentTaskSpec, PathAlignmentGoal, ProjectedAreaFractionGoal, _ref

    return GarmentTaskSpec(
        id="half_fold",
        label="Fold the terry towel long edge to long edge",
        construction_id="terry_cotton_bath_towel",
        goals=(
            PathAlignmentGoal(
                "long_hem_alignment",
                _ref("terry_body", "long_left_hem"),
                _ref("terry_body", "long_right_hem", reverse=True),
                rms_threshold_m=0.025,
                maximum_threshold_m=0.060,
            ),
            ProjectedAreaFractionGoal(
                "half_footprint",
                maximum_fraction=0.62,
                panel_ids=("terry_body",),
            ),
        ),
        maximum_structural_strain=0.20,
    )


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)

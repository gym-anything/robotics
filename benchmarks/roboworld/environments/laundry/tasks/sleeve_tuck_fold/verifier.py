"""Outcome of sleeve_tuck_fold: both short sleeves folded in onto the T-shirt's body: for each sleeve at least 90 % of
its own particles inside the plan hull of the front and back panels, and lying on them as a layer (plan RMS distance to
the nearest body particle at most 45 mm, 90th-percentile height gap at most 25 mm).

spec() gives the released spec (garment_task.GARMENT_TASK_SPECS: the bare selection {"garment": "tee"} and contract
laundry-semantic-legacy). laundry-tasks-v1 and -v2 score its goals with continuous credit and add their shared gates
(task_set.task_spec, task_set_v2.task_spec). verify(task, positions_m, velocities_m_s, **evidence) judges an episode
state with the evaluator bound to the selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain,
crossing, release, table and rest evidence) and returns its garment_task.GarmentTaskResult."""


def spec():
    """garment_task builds GARMENT_TASK_SPECS from this while it is being imported, so laundry is imported here."""
    from laundry.garment_task import GarmentTaskSpec, PanelContainmentGoal, PanelStackGoal

    return GarmentTaskSpec(
        id="sleeve_tuck_fold",
        label="Tuck both short sleeves into the T-shirt body footprint",
        construction_id="cotton_jersey_half_sleeve_tee",
        goals=(
            PanelContainmentGoal(
                "left_sleeve_contained",
                ("left_sleeve",),
                ("front", "back"),
                required_fraction=0.90,
            ),
            PanelStackGoal(
                "left_sleeve_stacked",
                ("left_sleeve",),
                ("front", "back"),
            ),
            PanelContainmentGoal(
                "right_sleeve_contained",
                ("right_sleeve",),
                ("front", "back"),
                required_fraction=0.90,
            ),
            PanelStackGoal(
                "right_sleeve_stacked",
                ("right_sleeve",),
                ("front", "back"),
            ),
        ),
    )


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)

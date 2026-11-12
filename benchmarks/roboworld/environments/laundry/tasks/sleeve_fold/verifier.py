"""Outcome of sleeve_fold: both long sleeves and their cuffs resting on the laid-out shirt body, by any folding method:
for each side at least 85 % of the sleeve's cloth and 80 % of its cuff's (by area) lying over the body cloth, resting,
not enclosed by the body and not bunched in more than seven of its own layers, with the body still laid out (its
hem-to-neckline length at least 75 % of the start's) (garment_task.SleeveOnBodyGoal); at rest over 0.5 s, released, on
the table, stretch within 30 %, no cloth crossings. The score is 1 on success, else 0.9 x progress from the start.

spec() gives the spec (garment_task.GARMENT_TASK_SPECS; contract laundry-shirt-sleeve-fold-v1, the only one that
selects the shirt). verify(task, positions_m, velocities_m_s, **evidence) judges an episode state with the evaluator
bound to that spec (newton15/env.LaundryEnv.evaluate passes the strain, crossing, release, table and rest evidence)
and returns its garment_task.GarmentTaskResult."""

_SHIRT_BODY_PANELS = (
    "left_front", "right_front", "back", "back_yoke", "chest_pocket", "collar_stand", "collar",
)


def spec():
    """garment_task builds GARMENT_TASK_SPECS from this while it is being imported, so laundry is imported here."""
    from laundry.garment_task import GarmentTaskSpec, SleeveOnBodyGoal, _ref

    return GarmentTaskSpec(
        id="sleeve_fold",
        label="Fold both long sleeves and cuffs onto the shirt body",
        construction_id="woven_button_down_shirt",
        goals=tuple(
            SleeveOnBodyGoal(
                f"{side}_sleeve_and_cuff_on_body",
                f"{side}_sleeve",
                f"{side}_cuff",
                _SHIRT_BODY_PANELS,
                ("left_front", "right_front", "back"),
                ("cap_front", "cap_back"),
                (_ref("back", "bottom_hem"), _ref("back", "neckline")),
            )
            for side in ("left", "right")
        ),
        rest_window_s=0.5,
        limit_compression=False,
        tabletop_support_fraction=0.95,
        full_credit_requires_success=True,
        partial_credit_cap=0.9,
    )


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)

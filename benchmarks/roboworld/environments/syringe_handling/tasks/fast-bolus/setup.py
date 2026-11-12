"""Start of fast-bolus: a 10 mL syringe full of amber contrast and the prefilled saline flush syringe, both capped, on
the tray, and the training arm with its saline lock clamped; the 10 mL bolus goes in within 8 s, then 5 mL of flush.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing; this task changes nothing more (its variants, clamp-open and g18, are shared with the other
iv-lock tasks: scene.task_layout). DOSE_LABEL is the label the dose syringe shows (render.liquid_state)."""

SETUP = dict(trainer="lock", syringes=(("dose", "s10", "contrast", 10.0, 0.0, "cap", -3),
                                       ("flush", "flush", "saline", 10.0, 0.3, "cap", 3)),
             dose_ml=10.0, bolus_s=8.0, flush_ml=5.0)
DOSE_LABEL = "CONTRAST"


def setup(L, config, jit):
    """Nothing beyond SETUP: the task's variants are shared ones and it has no seeded contents of its own."""

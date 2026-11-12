"""Start of flush-lock: the prefilled 10 mL saline flush syringe, capped, on the tray (10 mL with 0.3 mL of air), and
the training arm with its saline lock clamped; 5 mL to flush.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing: it sets up the task's own variant blocked (a cannula that takes nothing)."""

SETUP = dict(trainer="lock", syringes=(("flush", "flush", "saline", 10.0, 0.3, "cap", 0),), flush_ml=5.0)


def setup(L, config, jit):
    if config.variant == "blocked":
        L["blocked"] = True

"""Start of slow-push: a 10 mL syringe with 4 mL of blue medication and the prefilled saline flush syringe, both
capped, on the tray, and the training arm with its saline lock clamped; the dose goes in over 120 s, then 5 mL of
flush.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing: it sets up the task's own variant short (1 mL from a 3 mL syringe over 60 s)."""

SETUP = dict(trainer="lock", syringes=(("dose", "s10", "medication", 4.0, 0.0, "cap", -3),
                                       ("flush", "flush", "saline", 10.0, 0.3, "cap", 3)),
             dose_ml=4.0, dose_s=120.0, flush_ml=5.0)


def setup(L, config, jit):
    if config.variant == "short":
        L["syringes"] = (("dose", "s3", "medication", 1.0, 0.0, "cap", -3), L["syringes"][1])
        L["dose_ml"], L["dose_s"] = 1.0, 60.0

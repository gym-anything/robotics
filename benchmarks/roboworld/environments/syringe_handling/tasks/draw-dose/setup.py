"""Start of draw-dose: an empty 3 mL syringe with a sheathed 18 G blunt fill needle on the tray and a 10 mL vial of blue
medication on the counter (from seed 1 on a used one, up to 3 mL down); the dose to draw is 2.5 mL.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing: it sets up the task's own variants, adapter (the empty syringe without a needle, a vented
needle-free adapter on the vial) and vacuum-vial (7 mL left, 25 kPa under ambient)."""

SETUP = dict(trainer="pad", syringes=(("syringe", "s3", None, 0.0, 0.0, "g18b", 0),), vial_ml=10.0, dose_ml=2.5)


def setup(L, config, jit):
    v = config.variant
    if v == "adapter":
        L["adapter"] = True
        L["syringes"] = (("syringe", "s3", None, 0.0, 0.0, None, 0),)
    elif v == "vacuum-vial":
        L["vial_ml"], L["vial_gauge_kpa"] = 7.0, -25.0

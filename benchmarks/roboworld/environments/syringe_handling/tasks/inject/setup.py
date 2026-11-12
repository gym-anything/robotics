"""Start of inject: the 3 mL syringe on the tray, its 23 G needle sheathed, holds the 0.5 mL dose for the muscle of the
injection pad; the pad and a vial stand on the counter.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing: it sets up the task's own variant subcut (a 1 mL syringe with a 27 G needle holding 0.3 mL, for
the fat under the skin). PROP_VIAL: the vial stands on the counter although the task does not use it
(scene.Scene.station)."""

SETUP = dict(trainer="pad", syringes=(("syringe", "s3", "medication", 0.5, 0.0, "g23", 0),), dose_ml=0.5)
PROP_VIAL = True


def setup(L, config, jit):
    if config.variant == "subcut":
        L["syringes"] = (("syringe", "s1", "medication", 0.3, 0.0, "g27", 0),)
        L["dose_ml"], L["layer"] = 0.3, "fat"

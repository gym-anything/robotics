"""Start of sharps-away: three used syringes with bare needles on the tray (a 3 mL with a 25 G, a 10 mL with a 21 G and
a 1 mL with a 27 G, each holding a few hundredths of a millilitre), the sheaths of the first two and a used gauze square
loose beside them; the injection pad and a vial stand on the counter.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing; this task changes nothing more. PROP_VIAL: the vial stands on the counter although the task does
not use it (scene.Scene.station)."""

SETUP = dict(trainer="pad", syringes=(("syringe_a", "s3", "medication", 0.05, 0.0, "g25", -4),
                                      ("syringe_b", "s10", "saline", 0.07, 0.0, "g21", 0),
                                      ("syringe_c", "s1", "medication", 0.04, 0.0, "g27", 4)),
             bare=True, loose_sheaths=("syringe_a", "syringe_b"))
PROP_VIAL = True


def setup(L, config, jit):
    """Nothing beyond SETUP: the task has no variant of its own and no seeded contents."""

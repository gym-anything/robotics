"""Start of vial-to-pad: an empty 3 mL syringe with a sheathed 18 G blunt fill needle and a spare sheathed 25 G needle
on the tray, a 10 mL vial of blue medication (from seed 1 on a used one, up to 3 mL down) and the injection pad on the
counter; the dose is 1 mL, into the muscle.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot); spare_needles: (name, kind, slot)).
setup(L, config, jit) receives that layout as it is built (a dict, changed in place), the launch configuration and the
seeded draw jit(s) (uniform in -s..s, 0 at seed 0) and returns nothing; this task changes nothing more (its variants,
upright and half-vial, are shared with other tasks: scene.task_layout)."""

SETUP = dict(trainer="pad", syringes=(("syringe", "s3", None, 0.0, 0.0, "g18b", -2),), vial_ml=10.0, dose_ml=1.0,
             spare_needles=(("needle_im", "g25", 3),))


def setup(L, config, jit):
    """Nothing beyond SETUP: the task's variants are shared ones and it has no seeded contents of its own."""

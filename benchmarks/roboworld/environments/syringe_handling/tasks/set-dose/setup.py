"""Start of set-dose: the 3 mL syringe on the tray, its 25 G needle sheathed, holds 2.8 mL of blue medication and
0.3 mL of air; the dose to leave in it is 2.5 mL. The injection pad and a vial stand on the counter.

SETUP is what lies on the tray and stands at the working position; scene.task_layout starts the bay's layout from it
(syringes: (name, kind, liquid, liquid mL, air mL, fitting, slot)). setup(L, config, jit) receives that layout as it is
built (a dict, changed in place), the launch configuration and the seeded draw jit(s) (uniform in -s..s, 0 at seed 0)
and returns nothing: from seed 1 on it varies the syringe's medication and air by up to 0.1 mL each. PROP_VIAL: the
vial stands on the counter although the task does not use it (scene.Scene.station)."""

SETUP = dict(trainer="pad", syringes=(("syringe", "s3", "medication", 2.8, 0.3, "g25", 0),), dose_ml=2.5)
PROP_VIAL = True


def setup(L, config, jit):
    if config.seed:
        # contents vary a little: what a previous user left, how much air came with the draw
        out = []
        for (name, kind, liq, ml, air, fit, slot) in L["syringes"]:
            if liq and ml > 0.2:
                ml, air = round(ml + jit(0.1), 3), round(max(air + jit(0.1), 0.1), 3)       # at most 3.3 mL in all: the barrel holds 3.36
            out.append((name, kind, liq, ml, air, fit, slot))
        L["syringes"] = tuple(out)

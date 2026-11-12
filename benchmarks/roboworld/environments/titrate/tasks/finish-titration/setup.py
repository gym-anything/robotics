"""Start of finish-titration: a titration paused just before its endpoint: all but 0.3125, 0.5, 0.75 or 1 mL (a
seeded choice) of the equivalence volume of base already run into the flask and mixed in, the burette holding the
rest, and the stirrer turning at 500 rpm.

arrange(s) sets the paused charge, the base already in and the stirrer dial on task_presets.Start
(task_presets.preset, seeded by the variation); precharge_chemistry(liquid, setup) mixes that base into the flask's
chemistry once the liquid is charged, and rebase(setup, initial) records the represented base in the flask as the
start's base0_ml (both called by task_session.catalogue_session). setup(config, **kwargs) receives
task_session.make_session's arguments and returns the task's session (task_session.catalogue_session)."""
from dataclasses import replace
import math

from titrate.newton15.task_session import catalogue_session


def arrange(s):
    reagents, rng = s.reagents, s.rng
    # All offsets are exact multiples of the reference parcel volume.
    delta = rng.choice((.3125, .5, .75, 1.))
    base0 = reagents.aliquot_ml*reagents.acid_molar/reagents.base_molar-delta
    if not 0 < base0 < reagents.burette_ml:
        raise ValueError('Paused titration requires enough base for a positive precharge')
    s.actual = replace(reagents, dilution_ml=reagents.dilution_ml+base0,
                       burette_ml=reagents.burette_ml-base0)
    s.base0 = base0
    s.dial = (500.-100.)/1400.*(5*math.pi/3)


def precharge_chemistry(liquid, setup):
    """A prepared, mixed starting sample, not a reaction-time intervention.

    Volume moved from burette to flask was included in the ordinary lattice
    initialization. Preserve original aliquot, tracer, all sodium/chloride and
    analytical H/OH totals; initialize the already reacted amount explicitly.
    """
    import numpy as np
    base0 = setup['base0_ml']
    if not base0:
        return
    from titrate.newton15.liquid_charge import Reagents
    r = Reagents(**setup['nominal_reagents'])
    v0 = r.aliquot_ml+r.dilution_ml
    volume = v0+base0
    first, last = liquid.initial_groups['flask']
    chem = liquid.chemistry
    chem.charge(slice(first, last), acid_m=r.acid_molar*r.aliquot_ml/volume,
                base_m=r.base_molar*base0/volume, indicator_m=r.indicator_molar*v0/volume)
    liquid.initial_totals = chem.observe()['totals_mol']
    n = chem.moles.numpy()
    reacted = np.minimum(n[:, 0], n[:, 1])
    n[:, 0] -= reacted
    n[:, 1] -= reacted
    chem.moles.assign(n)
    chem.next.assign(n)
    chem.reacted.assign(reacted)
    chem.next_reacted.assign(reacted)


def rebase(setup, initial):
    setup['base0_ml'] = initial['base_in_flask_ml']


def setup(config=None, **kwargs):
    return catalogue_session('finish-titration', config, **kwargs)

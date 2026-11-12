"""Start of run-in: the selected variation's bench as it is, nothing moved; a charge whose equivalence volume is
1.5 mL or less is refused, since the task runs in all but about 1 mL of it.

arrange(s) checks the charge on task_presets.Start (task_presets.preset) and raises ValueError for such a charge;
setup(config, **kwargs) receives task_session.make_session's arguments and returns the task's session
(task_session.catalogue_session)."""
from titrate.newton15.task_session import catalogue_session


def arrange(s):
    reagents = s.reagents
    if reagents.aliquot_ml*reagents.acid_molar/reagents.base_molar <= 1.5:
        raise ValueError('Run-in needs an equivalence volume above 1.5 mL')


def setup(config=None, **kwargs):
    return catalogue_session('run-in', config, **kwargs)

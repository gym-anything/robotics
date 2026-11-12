"""Start of set-up-gun: the tacked start with the source left on the cell's documented short-circuit job
(params.SCHEDULES sc_0p9_C25_175A: 6.4 m/min at 20 V, task.SETUP_LEFT_JOB) and the 1/4 in of wire that job leaves
(SETUP_LEFT_STICKOUT); the gas valve stays closed, as the tacked start leaves it. A variation's own stick-out replaces
the wire.

START_STATE is the start state env.WeldingEnv builds for the task (task.START_STATE) and PRESET names its preset
(task.START_PRESET); setup(env) receives the freshly built WeldingEnv before tick 0 (presets.apply_start_preset), sets
the source and the wire through the robot's own I/O (presets._source_left_on_last_job) and returns the preset's
assisted-log record."""
from welding.presets import _source_left_on_last_job

START_STATE = "tacked"
PRESET = "source left on the last job's short-circuit setting, 1/4 in of wire"
# that job's 1/4 in of wire (the lower end of the variations' stick-out range, on the procedure's basis: verifier.py
# SETUP_STICKOUT); a catalogue variation's own stick-out replaces the wire (variations.apply_initial)
SETUP_LEFT_STICKOUT = 6.35e-3


def setup(env):
    return _source_left_on_last_job(env, SETUP_LEFT_STICKOUT)

"""Start of set-up-and-weld: the tacked start with the source left on the cell's documented short-circuit job
(params.SCHEDULES sc_0p9_C25_175A: 6.4 m/min at 20 V, task.SETUP_LEFT_JOB); the wire is the tacked start's own (or the
variation's), so the recorded aim_tacks program, which commands the procedure's set points itself before it strikes
(scripted.WeldProgram.weld), runs from it as it ran from the tacked start.

START_STATE is the start state env.WeldingEnv builds for the task (task.START_STATE) and PRESET names its preset
(task.START_PRESET); setup(env) receives the freshly built WeldingEnv before tick 0 (presets.apply_start_preset), sets
the source through the robot's own I/O (presets._source_left_on_last_job) and returns the preset's assisted-log
record."""
from welding.presets import _source_left_on_last_job

START_STATE = "tacked"
PRESET = "source left on the last job's short-circuit setting"


def setup(env):
    return _source_left_on_last_job(env)

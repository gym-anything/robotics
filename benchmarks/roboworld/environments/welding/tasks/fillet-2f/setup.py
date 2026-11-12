"""Start of fillet-2f: the tacked start (env.py: the bracket tacked at both ends of the near side by the assisted
fixture's two spray tacks), the robots at their welding poses, the source at the cell's settings.

START_STATE is the start state env.WeldingEnv builds for the task (task.START_STATE); setup(env) receives the freshly
built WeldingEnv before tick 0 (presets.apply_start_preset) and returns the task's start preset record: None, as the
task sets nothing on that start state."""

START_STATE = "tacked"


def setup(env):
    return None

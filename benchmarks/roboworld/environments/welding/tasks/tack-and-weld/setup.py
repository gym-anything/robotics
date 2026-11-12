"""Start of tack-and-weld: the loose start (env.py: the bracket standing unattached on its layout on the clamped base
plate), no weld on the joint, the robots at their welding poses.

START_STATE is the start state env.WeldingEnv builds for the task (task.START_STATE); setup(env) receives the freshly
built WeldingEnv before tick 0 (presets.apply_start_preset) and returns the task's start preset record: None, as the
task sets nothing on that start state."""

START_STATE = "loose"


def setup(env):
    return None

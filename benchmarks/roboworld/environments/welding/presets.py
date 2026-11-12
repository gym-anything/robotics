"""Task start presets: the initial condition a task sets on its built start state before tick 0, disclosed
in the assisted log, as welding/variations.py apply_initial sets a catalogue variation's wire stick-out. The backend
applies both, the preset first (gym_anything_robotics/backends/newton_welding.py, _make_env); a task with a preset
refuses to be evaluated without it (task.Evaluator.evaluate). A preset places a part or sets the source's set points
through the robot's own I/O; it changes no physical model, so this module and the task folders are not hashed with the
physics (env.SOURCE_FILES) and the cached start states stay valid.

A benchmark task's preset is in its setup.py (tasks/<task>/: PRESET names it, setup(env) sets it and returns its
record): set-up-gun (the source left on the last job's short-circuit setting and that job's wire), set-up-and-weld (the
same set points only) and fit-up-bracket (the bracket stood off its layout). The other tasks have none.
"""

import math

from . import params as PR
from .task import SETUP_LEFT_JOB, task_setup


def apply_start_preset(env):
    """Set the task's preset on a freshly built WeldingEnv (before tick 0) and take its initial checkpoint again, so a
    reset returns to it. Returns the assisted-log record, or None for a task without a preset (a benchmark task's setup
    is called all the same: it sets nothing then)."""
    task = env.config.task
    folder = task_setup(task)
    preset = getattr(folder, "PRESET", None)
    if preset is None:
        return None if folder is None else folder.setup(env)
    if getattr(env, "start_preset", None) is not None:
        raise RuntimeError(f"{task}: its start preset is already applied")
    if env.steps:
        raise RuntimeError(f"{task}: a start preset is an initial condition; apply it before tick 0")
    record = folder.setup(env)
    record = dict(preset=preset, task=task, **record)
    env.assisted_log.append(record)
    env.start_preset = record
    env.start_state = f"{env.start_state}; {preset}"
    env.initial = env.checkpoint()
    return record


def _source_left_on_last_job(env, stickout=None):
    """set-up-gun's and set-up-and-weld's preset: the source's set points left on the cell's short-circuit job
    (task.SETUP_LEFT_JOB) and, given a stickout (m: set-up-gun's), the wire that job left in the gun."""
    wire = stickout is not None
    w = env.world
    src = w.process.src
    st = src.st
    if st.arc or st.on_cmd or st.inch:
        raise RuntimeError(f"{env.config.task}'s preset is an initial condition: the source must be idle")
    if wire and math.isfinite(st.d_tw) and stickout >= st.d_tw:
        raise ValueError(f"a {stickout*1e3:.2f} mm stick-out would reach the metal {st.d_tw*1e3:.2f} mm from "
                         "the contact tip at the start pose")
    job = PR.v(PR.SCHEDULES)[SETUP_LEFT_JOB]
    before = dict(wire_feed_m_min=float(src.s.wfs_m_min), voltage_v=float(src.s.voltage_v), stickout_m=float(st.l_s),
                  gas_on=bool(st.gas_on))
    w.tool_io(wire_feed_m_min=job["wfs_m_min"], voltage_v=job["voltage_v"])
    if wire:
        st.l_s = st.l_s0 = float(stickout)    # an initial condition: the wire balance starts here
    env._clear_world_logs()
    what = "source set points" + (" and wire stick-out" if wire else "")
    return dict(initial_condition=what + " left by the last job", job=SETUP_LEFT_JOB,
                before=before, set=dict(wire_feed_m_min=float(src.s.wfs_m_min), voltage_v=float(src.s.voltage_v),
                                        stickout_m=float(st.l_s), gas_on=bool(st.gas_on)))

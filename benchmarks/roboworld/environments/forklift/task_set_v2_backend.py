"""The forklift-tasks-v2 session: task wiring only (forklift/task_set_v2.py). World, operator, cameras, trace, refusals,
reset and the owner/evaluation routing are NewtonForkliftBackend's.

NewtonForkliftBackend.__new__ builds this subclass only for a launch whose backend options name forklift-tasks-v2 (the
resolver adds it for the v2 selections). Every other launch is the released class, unchanged. What differs, as on the
fix branch:
- the adapter is forklift/task_progress_v2.TaskProgress, built after settling, with no tick hook;
- command execution is the base's (act): no command is cut at the time limit and none is refused after it;
- checkpoints carry no adapter state (there is none);
- the policy's capabilities declare the limit (hardware.time_limit_s and the episode note) with the note on how the
  evaluation applies it.
The released reset, _evaluate_task, get_state, evaluate and policy_access serve the v2 adapter as they serve v1's.
"""

import time

from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
from gym_anything_robotics.errors import BackendProtocolError, UnsupportedConfigurationError
from gym_anything_robotics.types import BackendStep


class NewtonForkliftTaskSetV2Backend(NewtonForkliftBackend):
    """A session launched with contract=forklift-tasks-v2."""

    def __init__(self, *, task="start-and-park", preset=None, contract=None, **kwargs):
        from forklift.config import TASKS
        from forklift.task_set_v2 import resolve_contract
        # The released constructor's order: an unknown task or a wrong preset is refused first (by it, below), then
        # the contract, before any robot validation or world.
        presets = {t: p for p, ts in TASKS.items() for t in ts if p != "lab"}
        if task in presets and (preset is None or preset == presets[task]):
            try:
                chosen = resolve_contract(task, contract)
            except ValueError as exc:
                raise UnsupportedConfigurationError(str(exc)) from exc
        super().__init__(task=task, preset=preset, **kwargs)   # the released checks, world and session (bare)
        from forklift.task_progress_v2 import TaskProgress
        self.task_contract = chosen
        w = self.env.world
        self._task_progress = TaskProgress(w, w.scene.task_start)

    def _episode_note(self):
        """The task's simulated-time limit as the evaluation enforces it (the instruction does not state it): the
        evaluator's time_limit criterion, sim_time <= TIME_LIMIT_S. Command execution is the base's."""
        limit = self._time_limit_s()
        return (f"The task's time limit is {limit:g} s of simulated time ({self.config.task}; forklift/task.py "
                "TIME_LIMIT_S), counted from the settled start; simulated time advances only while a command runs. "
                "The evaluation's time_limit criterion requires the task to be complete within it: an evaluation "
                f"succeeds only at a simulated time of at most {limit:g} s. Commands run as usual past the limit; "
                "they are not cut or refused.")

    def act(self, action):
        """The base's command execution: the whole command runs, at any simulated time."""
        with self.env.lock:
            w = self.env.world
            cmd, ticks = self.validate_action(action)
            before = w.tick
            self._prestart_worker(ticks)
            t0 = time.perf_counter()
            try:
                state = self.env.act(cmd, ticks)
            except (RuntimeError, FloatingPointError) as exc:
                self._discard_prestart()
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            except BaseException:
                self._discard_prestart()
                raise
            wall = time.perf_counter() - t0
            self.last_timing = {"physics_s": wall}
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": w.tick - before,
                                       "wall_per_sim": wall / max(1e-9, (w.tick - before) * self.config.dt)}, None, False)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.world.tick, "generation": self.env.generation}

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = self._checkpoints.get(label)
            if snap is None:
                raise KeyError(label)
            self.env.restore(snap)
            self.env.generation += 1
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.tick,
                                          "generation": self.env.generation, "continuity": "restored_discontinuity",
                                          "storage": "in_memory_session"}

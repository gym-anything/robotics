"""Task-only facade around an already constructed, unmodified FilletingEnv.

No global monkeypatches: two sessions may use different contracts concurrently.
Engine configuration remains the original scene carrier; public state names the
selected task. Task histories follow checkpoints and reset with engine histories.
"""
from .task_set_catalogue import CONTRACT, INSTRUCTIONS, LIMITS
from .task_set_evaluator import TaskSetEvaluator


class TaskSetEnv:
    def __init__(self, base, task):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_task", task)
        old_trace = base.trace_callback
        base.evaluator = TaskSetEvaluator(self, task)

        def record(env):
            base.evaluator.monitor()
            if old_trace is not None:
                old_trace(env)

        base.trace_callback = record
        base.initial = self.checkpoint()

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        setattr(self._base, name, value)

    def checkpoint(self):
        snap = self._base.checkpoint()
        snap["task_set"] = self.evaluator.snapshot()
        return snap

    def restore(self, snap):
        self._base.restore(snap)
        self.evaluator.restore(snap["task_set"])

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()

    def _state(self, state):
        state["config"] = dict(state["config"], task=self._task, task_contract=CONTRACT)
        state["instruction"] = INSTRUCTIONS[self._task]
        state["terminated"] = self.sim_time >= LIMITS[self._task]
        if self.world.fault:
            state["evaluation"] = self.evaluator.evaluate()
        return state

    def get_state(self, evaluate=True):
        return self._state(self._base.get_state(evaluate=evaluate))

    def act(self, targets, *, ticks, guard=None, state=True):
        # Match existing input validation before limiting the episode.
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if self.world.fault:
                raise RuntimeError(self.world.fault)
            left = max(0, int(round(LIMITS[self._task]/self.config.rigid_dt)) - self.steps)
            if left:
                self._base.act(targets, ticks=min(ticks, left), guard=guard, state=False)
            return self.get_state() if state else None

    def step_once(self):
        # Owner-side single ticks also obey the task budget and record task histories.
        self.act({}, ticks=1, state=False)

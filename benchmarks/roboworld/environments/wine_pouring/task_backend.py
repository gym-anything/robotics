"""Task-contract adapter for GAR; the physics backend (newton_wine_pouring.py) is used unchanged.

Only task configuration, instruction/evaluator selection and variation identity are
extended. World construction and all controls/rendering use the original backend.
"""
from dataclasses import replace

from gym_anything_robotics.backends.newton_wine_pouring import NewtonWinePouringBackend as PhysicsBackend
from gym_anything_robotics.errors import ActionRejectedError
from .config import Config as PhysicsConfig, TASKS
from .contracts import TaskConfig, reward_contract
from .task_contracts import Evaluator, instruction


class NewtonWinePouringBackend(PhysicsBackend):
    def __init__(self, *, contract=None, config=None, **kwargs):
        values = dict(config or {})
        selected = values.pop("task_contract", "")
        if contract is not None:
            selected = contract
        preset = kwargs.get("preset", "service")
        task = kwargs.get("task") or TASKS[preset][0]
        reward_contract(task, selected)  # refuse an invalid contract before constructing the world
        super().__init__(config=values, **kwargs)
        self.config = TaskConfig(**vars(self.config), task_contract=selected)
        self._attach_task(self.env, self.config)

    @staticmethod
    def _attach_task(env, config):
        # Construction/preparation has finished. Only the task-facing config and
        # evaluator are replaced; world/scene retain the unmodified PhysicsConfig.
        env.config = config
        env.evaluator = Evaluator(env)

    def _make_env(self, config):
        from .env import WineEnv
        values = {k: v for k, v in vars(config).items() if k != "task_contract"}
        env = WineEnv(PhysicsConfig(**values), output_dir=self.output)
        self._attach_task(env, config)
        return env

    def policy_access(self):
        return replace(super().policy_access(), instruction=lambda: instruction(self.config))

    def reset(self, *, seed=None, options=None):
        """Reset to the session's task, or - with options {preset, task} and/or a seed - to another catalogue variation:
        the world is rebuilt for it in this process (about the cost of a session start's physics part, 30-45 s, but the
        server, its keys and history stay)."""
        from dataclasses import replace
        from wine_pouring.variants import resolve, config_values
        from wine_pouring.kinematics import ArmKinematics
        values = dict(options or {})
        if set(values) - {"preset", "task", "contract"}:
            raise ActionRejectedError("Reset selectors are preset, task, contract (options) and seed")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        preset = values.get("preset", None if "task" in values else self.config.preset)     # a task alone selects its preset
        task = values.get("task", self.config.task if preset == self.config.preset else None)
        try:
            # A reset/seed change without a task selector retains the active contract;
            # an explicit task selection starts from that task's default contract.
            selected_contract = values.get("contract", (self.config.task_contract or None)
                                           if not {"preset", "task"} & set(values) else None)
            sel = resolve(preset, task, self.config.seed if seed is None else seed, selected_contract)
        except (ValueError, KeyError) as exc:
            # refused before the world is touched: a pre-execution rejection, so the session stays ready
            # (a plain ValueError would latch a runtime fault in the HTTP service)
            raise ActionRejectedError(str(exc)) from exc
        from wine_pouring.contracts import reward_contract
        if ((sel["preset"], sel["task"], sel["seed"]) == (self.config.preset, self.config.task, self.config.seed) and
                reward_contract(sel["task"], sel.get("contract", "")) ==
                reward_contract(self.config.task, self.config.task_contract)):
            return self.env.reset(), {"continuity": "reset", "variation": sel}
        config = replace(self.config, **config_values(sel))
        candidate = self._make_env(config)
        with self.env.lock:
            self._end_operation(final=True)
            for t in getattr(self, "_reapers", []):
                t.join()
            self._reapers = []
            old, self.env, self.config = self.env, candidate, config
            candidate.generation = old.generation + 1
            self.ik = ArmKinematics(candidate.world)
            self._scene_path = None
            if self.trace:
                path = self.trace.store.path
                self.trace.close()
                from wine_pouring.trace import Trace
                self.trace = Trace(candidate, path)
            self.capabilities.notes["task"] = f"{config.preset}/{config.task} seed {config.seed}"
        del old
        return self.env.get_state(), {"continuity": "explicit_reset", "variation": sel}

    def select_variation(self, request):
        from wine_pouring.variants import resolve, config_values
        from wine_pouring.task_contracts import instruction
        if set(request) - {"preset", "task", "seed", "contract"}:
            raise ValueError("Selectors are preset, task, seed and contract")
        sel = resolve(request.get("preset", None if "task" in request else self.config.preset), request.get("task"),
                      request.get("seed", 0), request.get("contract"))
        return dict(variation=sel, task=instruction(type(self.config)(**{**self.config.__dict__, **config_values(sel)})),
                    reset_request=dict(seed=sel["seed"], options={k: v for k, v in sel.items() if k != "seed"}))

    def variation_contract(self):
        from wine_pouring.variants import contract
        from wine_pouring.contracts import reward_contract
        return dict(**contract(), current=dict(preset=self.config.preset, task=self.config.task, seed=self.config.seed,
                                              contract=reward_contract(self.config.task, self.config.task_contract)),
                    reset="POST /v1/variations/select (or reset with options {preset, task, contract} and a seed) rebuilds the "
                          "world for the selection in this session (30-45 s)")

    def _variation_key(self):
        from wine_pouring.contracts import reward_contract
        c = self.config
        return dict(preset=c.preset, task=c.task, seed=c.seed, robot=c.robot,
                    contract=reward_contract(c.task, c.task_contract))


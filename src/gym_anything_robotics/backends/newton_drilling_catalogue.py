"""Opt-in task adapter; newton_drilling.py is in drilling's protected physics digest."""

from dataclasses import replace

from .newton_drilling import NewtonDrillingBackend
from ..errors import ActionRejectedError
from drilling import catalogue
from drilling.catalogue_evaluator import configure_environment


class NewtonDrillingCatalogueBackend(NewtonDrillingBackend):
    def __init__(self, *, contract=None, **kwargs):
        # Preserve the base constructor's coercions, defaults and options (notably int(seed)).
        preset = kwargs.get("preset", "wall")
        default = catalogue.legacy.PUBLIC.get(preset, (None,))[0] if isinstance(preset, str) else None
        task = kwargs.get("task") or default
        self._task_contract = catalogue.validate_contract(task, contract)
        super().__init__(**kwargs)

    @property
    def env(self):
        return self._catalogue_env

    @env.setter
    def env(self, value):
        # Called for both the initial world and a replacement world inside the base reset.
        self._catalogue_env = configure_environment(value, self._task_contract)

    def reset(self, *, seed=None, options=None):
        # Validate the added selector before touching the current evaluator. The existing selector
        # validation mirrors the base reset, including None/falsy defaults and strict reset seed typing.
        from drilling.robot_setup import check_task
        try:
            values = dict(options or {})
            requested = values.pop("contract", None)
            if set(values) - {"preset", "task"}:
                raise ValueError("Reset selectors are preset, task, contract (options) and seed")
            for key in ("preset", "task"):
                if values.get(key) is not None and not isinstance(values[key], str):
                    raise ValueError(f"Reset option {key} must be a name (string)")
            preset = values.get("preset", None if "task" in values else self.config.preset)
            task = values.get("task", self.config.task if preset == self.config.preset else None)
            sel = catalogue.legacy.resolve(preset, task, self.config.seed if seed is None else seed)
            check_task(self.config.robot, sel["preset"], sel["task"])
            same_task = sel["task"] == self.config.task
            chosen = catalogue.validate_contract(sel["task"], requested if "contract" in (options or {}) else
                                                 self._task_contract if same_task else None)
        except (ValueError, TypeError) as exc:
            raise ActionRejectedError(f"Reset refused, nothing changed: {exc}") from exc
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        previous, previous_env = self._task_contract, self.env
        self._task_contract = chosen
        same_world = same_task and sel["preset"] == self.config.preset and sel["seed"] == self.config.seed
        if same_world:
            configure_environment(self.env, chosen)
        try:
            state, info = super().reset(seed=seed, options=values)
        except Exception:
            # A candidate-build failure has not replaced the world. Keep its original contract too.
            if self.env is previous_env:
                self._task_contract = previous
                configure_environment(self.env, previous)
            raise
        if chosen != catalogue.LEGACY:
            info = dict(info, variation=dict(info["variation"], contract=chosen))
        return state, info

    def select_variation(self, request):
        values = dict(request)
        requested = values.pop("contract", None)
        result = super().select_variation(values)
        sel = result["variation"]
        chosen = catalogue.validate_contract(sel["task"], requested)
        # A selection without contract always means legacy, even in a session on the task-set contract.
        result["reset_request"]["options"]["contract"] = chosen
        if chosen != catalogue.LEGACY:
            sel["contract"] = chosen
            result["task"] = catalogue.instruction(sel["task"], chosen)
        return result

    def policy_access(self):
        return replace(super().policy_access(), instruction=lambda: self.env.evaluator.instruction)

    def variation_contract(self):
        result = super().variation_contract()
        result.update(catalogue.contract())
        if self._task_contract != catalogue.LEGACY:
            result["current"]["contract"] = self._task_contract
        result["reset"] += f"; contract selects v1 (default) or {catalogue.CONTRACT}"
        return result

    def _variation_key(self):
        key = super()._variation_key()
        if self._task_contract != catalogue.LEGACY:
            key["contract"] = self._task_contract
        return key

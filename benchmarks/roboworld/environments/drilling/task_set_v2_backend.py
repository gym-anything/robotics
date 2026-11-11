"""The drilling-tasks-v2 session: task wiring only (drilling/task_set_v2.py). World, actions, cameras, refusals and the
base reset are NewtonDrillingBackend's.

The registered drilling factory builds NewtonDrillingCatalogueBackend; its __new__ builds this subclass only for a
launch whose backend options name drilling-tasks-v2 (the resolver adds it for the v2 selections). Every other launch is
the released adapter, unchanged. This class keeps the adapter's shape but takes the base backend's behaviour where the
fix branch did: the base constructor, reset, variation selection, variation contract and variation key (reached with
super(NewtonDrillingCatalogueBackend, self), past the adapter's v1 versions), with the v2 catalogue and evaluator, and
the base's reset refusal form ("...; got [...]").
"""

from gym_anything_robotics.backends.newton_drilling_catalogue import NewtonDrillingCatalogueBackend
from gym_anything_robotics.errors import ActionRejectedError


class NewtonDrillingTaskSetV2Backend(NewtonDrillingCatalogueBackend):
    """A session launched with contract=drilling-tasks-v2: its evaluator, instruction, selection, reset and variation
    contract. A reset can return to v1 (the legacy evaluator) or v2; drilling-tasks-v1 is a released session's."""

    def __init__(self, *, contract=None, **kwargs):
        from drilling import task_set_v2 as catalogue
        # Preserve the base constructor's coercions, defaults and options (notably int(seed)).
        preset = kwargs.get("preset", "wall")
        default = catalogue.legacy.PUBLIC.get(preset, (None,))[0] if isinstance(preset, str) else None
        task = kwargs.get("task") or default
        self._task_contract = catalogue.validate_contract(task, contract)
        super(NewtonDrillingCatalogueBackend, self).__init__(**kwargs)

    @property
    def env(self):
        return self._catalogue_env

    @env.setter
    def env(self, value):
        # Called for both the initial world and a replacement world inside the base reset.
        from drilling.task_set_v2_evaluator import configure_environment
        self._catalogue_env = configure_environment(value, self._task_contract)

    def reset(self, *, seed=None, options=None):
        # Validate the added selector before touching the current evaluator. The existing selector
        # validation mirrors the base reset, including None/falsy defaults and strict reset seed typing.
        from drilling import task_set_v2 as catalogue
        from drilling.task_set_v2_evaluator import configure_environment
        from drilling.robot_setup import check_task
        try:
            values = dict(options or {})
            requested = values.pop("contract", None)
            if set(values) - {"preset", "task"}:
                raise ValueError("Reset selectors are preset, task, contract (options) and seed; got "
                                 f"{sorted(map(str, set(values) - {'preset', 'task'}))}")
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
            state, info = super(NewtonDrillingCatalogueBackend, self).reset(seed=seed, options=values)
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
        from drilling import task_set_v2 as catalogue
        values = dict(request)
        requested = values.pop("contract", None)
        result = super(NewtonDrillingCatalogueBackend, self).select_variation(values)
        sel = result["variation"]
        chosen = catalogue.validate_contract(sel["task"], requested)
        # A selection without contract always means legacy, even in a current v2 session.
        result["reset_request"]["options"]["contract"] = chosen
        if chosen != catalogue.LEGACY:
            sel["contract"] = chosen
            result["task"] = catalogue.instruction(sel["task"], chosen)
        return result

    # policy_access: the released adapter's (the policy reads the current evaluator's instruction)

    def variation_contract(self):
        from drilling import task_set_v2 as catalogue
        result = super(NewtonDrillingCatalogueBackend, self).variation_contract()
        result.update(catalogue.contract())
        if self._task_contract != catalogue.LEGACY:
            result["current"]["contract"] = self._task_contract
        result["reset"] += f"; contract selects v1 (default) or {catalogue.CONTRACT}"
        return result

    def _variation_key(self):
        from drilling import task_set_v2 as catalogue
        key = super(NewtonDrillingCatalogueBackend, self)._variation_key()
        if self._task_contract != catalogue.LEGACY:
            key["contract"] = self._task_contract
        return key

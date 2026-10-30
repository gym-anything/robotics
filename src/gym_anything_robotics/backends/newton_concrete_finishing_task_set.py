"""Task-only bridge outside the environment's frozen physics/source digest.

Inherit all physics, input coercion, controls, checkpoints and rendering. Select the
new evaluator explicitly; old launches keep their original evaluator and scores.
"""

from concrete_finishing import task_set as C
from .newton_concrete_finishing import NewtonConcreteFinishingBackend as LegacyBackend
from ..errors import UnsupportedActionError, UnsupportedConfigurationError


class NewtonConcreteFinishingBackend(LegacyBackend):
    def __init__(self, *, contract=None, **options):
        from concrete_finishing.layout import PRESETS
        task = options.get("task") or PRESETS.get(options.get("preset", "b012"), ("",))[0]
        try:
            self._task_contract = C.selected_contract(task, contract)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        super().__init__(**options)
        self._install_evaluator()

    def _install_evaluator(self):
        from concrete_finishing.task_set_progress import Evaluator
        # Called only on a freshly built or reset world, never on an attempted state.
        self._legacy_evaluator = self.env.evaluator
        if self._task_contract == C.CONTRACT:
            self.env.evaluator = Evaluator(self.env)

    def _variation(self):
        result = super()._variation()
        if self._task_contract == C.CONTRACT:
            result["contract"] = self._task_contract
        return result

    def _resolve(self, options):
        result = super()._resolve(options)
        current = self._task_contract if result["task"] == self.config.task else None
        chosen = C.selected_contract(result["task"], options.get("contract", current))
        if chosen == C.CONTRACT:
            result["contract"] = chosen
        return result

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        unknown = set(options) - set(C.SELECTORS)
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; nothing changed")
        if seed is not None:
            options["seed"] = seed
        try:
            want = self._resolve(options)
        except (ValueError, TypeError) as exc:
            raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
        chosen = C.selected_contract(want["task"], want.get("contract"))
        # The superclass's reset selector list is frozen. Its rebuild accepts the
        # four original fields; changing just the contract needs no physics rebuild.
        original = {k: v for k, v in want.items() if k != "contract"}
        if original != super()._variation():
            super()._rebuild(original)
            self._task_contract = chosen
            self._install_evaluator()
            return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        self.env.evaluator = self._legacy_evaluator
        self.env.reset()
        self._task_contract = chosen
        self._install_evaluator()
        return self.env.get_state(), {"continuity": "reset", "variation": self._variation()}

    def variation_contract(self):
        return dict(C.contract(), current=self._variation())

    def select_variation(self, request):
        from concrete_finishing.task import instruction
        if set(request) - set(C.SELECTORS):
            raise ValueError(f"Selectors are {', '.join(C.SELECTORS)}")
        v = self._resolve(request)
        # Explicitly carry legacy when previewing a switch on the same task;
        # otherwise preserve the original reset request shape for bare selections.
        options = {k: value for k, value in v.items() if k != "seed"}
        if self._task_contract == C.CONTRACT and v["task"] == self.config.task and "contract" not in v:
            options["contract"] = C.selected_contract(v["task"])
        return dict(variation=v, task=instruction(v), reset_request=dict(seed=v["seed"], options=options))

    def evaluate(self):
        result = super().evaluate()
        if self.config.task != "pad-finish":
            result = dict(result, reward_version=self._task_contract)
        return result

"""Task-contract adapter; the original backend is left unchanged."""
from dataclasses import replace

from .newton_soil_excavation import NewtonSoilExcavationBackend as LegacyBackend
from ..errors import UnsupportedActionError, UnsupportedConfigurationError
from soil_excavation import catalogue as C


class NewtonSoilExcavationTasksBackend(LegacyBackend):
    def __init__(self, *, contract=None, **options):
        try:
            self._task_contract = C.selected_contract(options.get("task") or "penetrate-marked", contract)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        super().__init__(**options)
        self._install_contract()

    def _install_contract(self):
        from soil_excavation.task_set import install
        install(self.env, self._task_contract)
        self.capabilities = replace(self.capabilities, notes=self._notes())

    def _variation(self):
        value = super()._variation()
        if self._task_contract == C.CONTRACT:
            value["contract"] = self._task_contract
        return value

    def _resolve_request(self, request):
        current = self._variation()
        # A different bare task retains that task's original default meaning.
        if "task" in request and request["task"] != current["task"] and "contract" not in request:
            current.pop("contract", None)
        value = dict(current, **request)
        return C.resolve(value["task"], int(value["seed"]), value["moisture"], value["topsoil"], value["stones"],
                         value.get("contract"))

    def _rebuild(self, value):
        self._task_contract = C.selected_contract(value["task"], value.get("contract"))
        super()._rebuild({k: value[k] for k in C.legacy.SELECTORS})
        self._install_contract()

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        unknown = set(options) - set(C.SELECTORS)
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are "
                                         f"{', '.join(C.SELECTORS)}; nothing changed")
        if seed is not None:
            options["seed"] = seed
        if options:
            try:
                want = self._resolve_request(options)
                from soil_excavation.robot_setup import check_task
                check_task(self.config.robot, want["task"])
            except (TypeError, ValueError) as exc:
                raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
            want = {k: v for k, v in want.items() if k in C.SELECTORS}
            if want != self._variation():
                self._rebuild(want)
                return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        with self.env.lock:
            if self._task_contract == C.CONTRACT:
                self.env.evaluator.base_off_slab = 0.0
            return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def variation_contract(self):
        return dict(**C.contract(), current=self._variation())

    def select_variation(self, request):
        if set(request) - set(C.SELECTORS):
            raise ValueError(f"Selectors are {', '.join(C.SELECTORS)}")
        value = self._resolve_request(request)
        return dict(variation=value, task=C.instruction(value["task"], value.get("contract")),
                    reset_request=dict(seed=value["seed"], options={k: value.get(k) for k in C.SELECTORS if k != "seed"}))

    def save_checkpoint(self, *, label):
        with self.env.lock:
            result = super().save_checkpoint(label=label)
            if self._task_contract == C.CONTRACT:
                self._checkpoints[label]["task_set_base_off_slab"] = self.env.evaluator.base_off_slab
            return result

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            if self._task_contract == C.CONTRACT:
                self.env.evaluator.base_off_slab = snap["task_set_base_off_slab"]
            return super().restore_checkpoint(label, **kwargs)

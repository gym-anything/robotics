"""soil-tasks-v2 sessions: the released task-contract adapter, extended to the v2 contract (task wiring only).

GAR's soil factory builds NewtonSoilExcavationTasksBackend; that class builds this subclass only when the launch
names contract soil-tasks-v2, so every released selection keeps the released adapter and code path. Here the v2
judgement (task_set_v2.py) is installed on the settled start, its recorded history (the slab history and
break-hardpan's work windows) restarts with each episode and is saved and restored with checkpoints, and reset
options may switch to any task, seed or contract that catalogue_v2 accepts. The physics, the world and the
legacy backend are unchanged.
"""
from dataclasses import replace

from gym_anything_robotics.backends.newton_soil_excavation_tasks import NewtonSoilExcavationTasksBackend
from gym_anything_robotics.errors import UnsupportedActionError, UnsupportedConfigurationError
from . import catalogue as C
from . import catalogue_v2 as V2


class NewtonSoilExcavationTaskSetV2Backend(NewtonSoilExcavationTasksBackend):
    def __init__(self, *, contract=None, **options):
        try:
            self._task_contract = V2.selected_contract(options.get("task") or "penetrate-marked", contract)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        # The legacy backend's constructor, as the released adapter calls it (without the contract).
        super(NewtonSoilExcavationTasksBackend, self).__init__(**options)
        self._install_contract()

    def _v2(self):
        return self._task_contract == V2.CONTRACT

    def _install_contract(self):
        if not self._v2():
            return super()._install_contract()
        from .task_set_v2 import install
        install(self.env, self._task_contract)
        self.capabilities = replace(self.capabilities, notes=self._notes())

    def _variation(self):
        value = super()._variation()
        if self._v2():
            value["contract"] = self._task_contract
        return value

    def _resolve_request(self, request):
        current = self._variation()
        # A different bare task retains that task's original default meaning.
        if "task" in request and request["task"] != current["task"] and "contract" not in request:
            current.pop("contract", None)
        value = dict(current, **request)
        return V2.resolve(value["task"], int(value["seed"]), value["moisture"], value["topsoil"], value["stones"],
                          value.get("contract"))

    def _rebuild(self, value):
        self._task_contract = V2.selected_contract(value["task"], value.get("contract"))
        super(NewtonSoilExcavationTasksBackend, self)._rebuild({k: value[k] for k in C.legacy.SELECTORS})
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
            if self._v2():
                self.env.evaluator.restart_history()       # the v2 verdict's history restarts with the episode
            elif self._task_contract == C.CONTRACT:
                self.env.evaluator.base_off_slab = 0.0
            return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def variation_contract(self):
        return dict(**V2.contract(), current=self._variation())

    def select_variation(self, request):
        if set(request) - set(C.SELECTORS):
            raise ValueError(f"Selectors are {', '.join(C.SELECTORS)}")
        value = self._resolve_request(request)
        return dict(variation=value, task=V2.instruction(value["task"], value.get("contract")),
                    reset_request=dict(seed=value["seed"], options={k: value.get(k) for k in C.SELECTORS if k != "seed"}))

    def save_checkpoint(self, *, label):
        if not self._v2():
            return super().save_checkpoint(label=label)
        with self.env.lock:
            result = super(NewtonSoilExcavationTasksBackend, self).save_checkpoint(label=label)
            self._checkpoints[label]["task_set_v2_history"] = self.env.evaluator.history()
            return result

    def restore_checkpoint(self, label, **kwargs):
        if not self._v2():
            return super().restore_checkpoint(label, **kwargs)
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            self.env.evaluator.restore_history(snap["task_set_v2_history"])
            return super(NewtonSoilExcavationTasksBackend, self).restore_checkpoint(label, **kwargs)

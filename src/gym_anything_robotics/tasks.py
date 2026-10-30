"""Environment-owned task discovery; resolving a task never starts physics."""

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path

from .launch import LaunchSpec, describe
from .paths import project_file

CATALOGS = {
    "label-application": "label_application/benchmark.json",
    "barcode-scanning": "barcode_scanning/benchmark.json",
    "filleting": "filleting/benchmark.json",
    "forklift": "forklift/benchmark.json",
    "welding": "welding/benchmark.json",
    "sewing": "sewing/benchmark.json",
    "wine-pouring": "wine_pouring/benchmark.json",
    "percussion": "percussion/benchmark.json",
    "classroom-arrangement": "classroom_arrangement/benchmark.json",
    "painting": "painting/benchmark.json",
    "surface-rinsing": "surface_rinsing/benchmark.json",
    "soil-excavation": "soil_excavation/benchmark.json",
    "drilling": "drilling/benchmark.json",
    "gift-bags": "gift_bags/benchmark.json",
    "ingredient-mixing": "ingredient_mixing/benchmark.json",
    "syringe-handling": "syringe_handling/benchmark.json",
    "concrete-finishing": "concrete_finishing/benchmark.json",
    "tighten-codex": "tighten_codex/benchmark.json",
    "deposit-cans": "deposit_cans/benchmark.json",
    "football": "football/newton15/benchmark.json",
    "pencap": "pencap/newton15/benchmark.json",
    "lockkey": "lockkey/newton15/benchmark.json",
    "laundry": "laundry/newton15/benchmark.json",
    "whiteboard": "whiteboard/benchmark.json",
    "titration": "titrate/newton15/benchmark.json",
}


def catalog(environment_id):
    name = describe(environment_id)["environment_id"]
    if name not in CATALOGS:
        raise ValueError(f"No task catalogue registered for {name}")
    value = json.loads(project_file(CATALOGS[name]).read_text(encoding="utf-8"))
    if value.get("schema") != "gar.task-catalog.v1" or value.get("environment_id") != name:
        raise ValueError("Task catalogue identity/schema mismatch")
    return value


def catalogs():
    return [catalog(name) for name in CATALOGS]


def merge_exact(first, second, path="backend_options"):
    """Add recipe fields; never replace a conflicting explicit launch value."""
    result = deepcopy(first)
    for key, value in second.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_exact(result[key], value, path + "." + key)
        elif result[key] != value or type(result[key]) is not type(value):
            raise ValueError(f"Task/launch conflict at {path}.{key}; no value overwritten")
    return result


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    launch: LaunchSpec
    selection: dict
    evaluator: dict
    split: str | None = None
    schema: str = "gar.task.v1"

    def __post_init__(self):
        if self.schema != "gar.task.v1" or not isinstance(self.launch, LaunchSpec):
            raise ValueError("Invalid task schema/launch")
        for value in (self.task_id, self.description):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Task ID and description must be nonempty strings")
        if self.split is not None and (not isinstance(self.split, str) or not self.split):
            raise ValueError("Split must be a nonempty string or null")
        for key in ("selection", "evaluator"):
            value = getattr(self, key)
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            object.__setattr__(self, key, json.loads(json.dumps(value, allow_nan=False)))

    def create(self):
        """Construct through the same public environment registry."""
        return self.launch.create()

    def to_dict(self):
        return dict(schema=self.schema, task_id=self.task_id, description=self.description,
                    launch=self.launch.to_dict(), selection=deepcopy(self.selection),
                    evaluator=deepcopy(self.evaluator), split=self.split)

    @classmethod
    def from_dict(cls, value):
        required = {"schema", "task_id", "description", "launch", "selection", "evaluator", "split"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"Task manifest requires exactly {sorted(required)}")
        return cls(**{**value, "launch": LaunchSpec.from_dict(value["launch"])})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def resolve(environment_id, selection=None, *, base=None):
    descriptor = catalog(environment_id)
    selection = {} if selection is None else deepcopy(selection)
    if not isinstance(selection, dict):
        raise ValueError("Task selection must be an object")
    json.dumps(selection, allow_nan=False)
    unknown = set(selection) - set(descriptor["selectors"])
    if unknown:
        raise ValueError(f"Unsupported task selectors: {sorted(unknown)}")
    if base is not None and (
        base.environment_id != descriptor["environment_id"] or base.backend != "newton"
    ):
        raise ValueError("Task and base launch must name the same canonical environment/backend")
    if descriptor.get("resolver"):
        module, name = descriptor["resolver"].split(":")
        resolved = getattr(import_module(module), name)(selection)
    else:
        if selection:
            raise ValueError("This task has no selectors")
        resolved = dict(task_id=descriptor["default_task_id"],
                        description=descriptor["description"], backend_options={},
                        selection={}, split=None)
    options = merge_exact(base.backend_options if base else {}, resolved["backend_options"])
    launch = LaunchSpec(descriptor["environment_id"],
                        robot=base.robot if base else None, backend_options=options)
    return TaskSpec(resolved["task_id"], resolved["description"], launch,
                    resolved["selection"], descriptor["evaluator"], resolved["split"])

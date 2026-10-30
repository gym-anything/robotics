"""One existing semantic task per authored garment; no solver or reset changes."""


def resolve(selection):
    from .config import GARMENTS
    from ..garment_task import GARMENT_TASK_SPECS
    garment = selection.get("garment", "towel")
    if garment not in GARMENTS:
        raise ValueError(f"Unknown garment: {garment}")
    choices = [v for (construction, _), v in GARMENT_TASK_SPECS.items()
               if construction == GARMENTS[garment]]
    if len(choices) != 1:
        raise ValueError("Expected exactly one existing task for this garment")
    spec = choices[0]
    return dict(task_id=f"laundry/{garment}/{spec.id}",
                description=spec.label + ". Use the robot arms; leave the garment released and resting on the table.",
                backend_options={"config": {"garment": garment}},
                selection={"garment": garment, "construction_id": GARMENTS[garment]}, split=None)

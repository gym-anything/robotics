"""The opt-in eight-task selection; no scene or physics changes.

Bare selections keep their original IDs, instructions and evaluators. The seven
revised tasks explicitly select CONTRACT; group-work keeps its own evaluator.
"""

CONTRACT = "classroom-tasks-v1"
LEGACY = "v1"
SET_TASKS = (
    "push-in-chairs", "straighten-row", "seat-tutoring-group", "clear-floor",
    "exam-row", "move-teacher-desk", "chairs-up", "group-work",
)
TASK_SET = tuple(dict(task=t, seed=0, **({"contract": CONTRACT} if t != "group-work" else {})) for t in SET_TASKS)
DIFFICULTY = dict(zip(SET_TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))


def validate_contract(task, contract):
    if contract not in (LEGACY, CONTRACT):
        raise ValueError(f"contract must be {LEGACY!r} or {CONTRACT!r}")
    if contract == CONTRACT and task not in SET_TASKS[:-1]:
        raise ValueError(f"{task} keeps its existing evaluator; select contract={LEGACY!r}")


def description(task, contract=LEGACY):
    from .layout import TASKS
    validate_contract(task, contract)
    text = TASKS[task]["instruction"]
    if contract == CONTRACT:
        if task == "seat-tutoring-group":
            text = text.replace(" (seat front 20-35 cm from the table edge)", "")
        # Furniture is identical and unlabelled; the outcome names places, not serial identities.
        text = text.replace("its own desk", "a desk in the row").replace("their own desks", "the row's desks")
        if "Leave the rest of the room as it is." not in text:
            text += " Leave the rest of the room as it is."
    return text

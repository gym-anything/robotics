"""Physics-free task discovery. The empty selection retains the original seat-cap launch."""

DEFAULT_TASK = "seat-cap"
TASKS = (DEFAULT_TASK, "pen-on-notebook", "unload-fixture", "reseat-cap",
         "load-fixture", "pass-pen", "uncap", "cap-in-fixture")
EIGHT_TASKS = TASKS[1:] + (DEFAULT_TASK,)
DIFFICULTY = dict(zip(EIGHT_TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
INSTRUCTIONS = {
    "seat-cap": "Put the blue cap on the yellow pen's writing tip and press it until it snaps fully into place, "
                "then leave the capped pen still, whether held or set down.",
    "pen-on-notebook": "Lay the yellow pen on the notebook.",
    "unload-fixture": "Take the capped pen out of the capping fixture and lay it down on the bench.",
    "reseat-cap": "The cap is only partly on the yellow pen. Press it on until it snaps fully into place, "
                  "then leave the capped pen still, whether held or set down.",
    "load-fixture": "Stand the yellow pen in the capping fixture, tail end down, ready for its cap.",
    "pass-pen": "Lay the yellow pen on the notebook.",
    "uncap": "Take the cap off the yellow pen and lay the pen and the cap down on the bench.",
    "cap-in-fixture": "The yellow pen stands in the capping fixture. Put the blue cap on its writing tip and "
                      "press it until it snaps fully into place, leaving the capped pen standing in the fixture.",
}
CONTRACTS = {task: f"pencap/{task}-v1" for task in TASKS}
FIXTURE_TASKS = frozenset(("unload-fixture", "load-fixture", "cap-in-fixture"))


def validate(task):
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError(f"Unknown pencap task {task!r}; choose one of {', '.join(TASKS)}")
    return task


def task_resolve(selection):
    if not isinstance(selection, dict) or set(selection) - {"task"}:
        raise ValueError("Pencap supports only the task selector")
    task = validate(selection.get("task", DEFAULT_TASK))
    return dict(task_id=f"pencap/{task}", description=INSTRUCTIONS[task],
                backend_options={} if task == DEFAULT_TASK else {"task": task},
                selection={} if not selection else {"task": task}, split=None)

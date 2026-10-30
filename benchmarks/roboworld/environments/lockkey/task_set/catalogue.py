"""Eight jobs, discoverable without importing a simulator or robot assets."""

from lockkey.newton15.task import INSTRUCTION, DESCRIPTION

DEFAULT_TASK = "unlock"
# Append to the existing catalogue; the benchmark set has its own difficulty order.
TASKS = (DEFAULT_TASK, "swing-open", "turn-to-open", "insert-key", "return-key",
         "remove-key", "which-key", "unlock-and-stow")
TASK_SET_ORDER = TASKS[1:6] + (DEFAULT_TASK,) + TASKS[6:]
CONTRACT = "lockkey-tasks-v1"
UNLOCK_CONTRACT = "open-padlock-v1"  # lockkey.newton15.evaluation.CONTRACT
INSTRUCTIONS = {
    "unlock": INSTRUCTION,
    "swing-open": "The padlock on the chest has just been unlocked and its key is still in it. "
                  "Swing the shackle open. Leave the other key on its stand.",
    "turn-to-open": "The key from the near stand is already in the padlock on the chest. "
                    "Unlock the padlock and swing the shackle open. Leave the other key on its stand.",
    "insert-key": "Take the key from the near (-Y) stand and put it all the way into the padlock on the chest, "
                  "then let go of it. Leave the other key on its stand.",
    "return-key": "The padlock's key has been left lying on the work mat. Put it back on its stand, "
                  "the empty one on the near (-Y) side. Leave the other key on its stand.",
    "remove-key": "The padlock on the chest is open with its key still in it. Take the key out and put it back "
                  "on its stand on the near (-Y) side. Leave the shackle open and the other key on its stand.",
    "which-key": "Two keys rest on the stands beside the padlock on the chest; only one of them opens it. "
                 "Unlock the padlock and swing the shackle open. Leave the key that does not open it on its stand.",
    "unlock-and-stow": "Unlock the padlock on the chest with its key from the near (-Y) stand and swing the "
                       "shackle open, then put the key back on its stand. Leave the other key on its stand.",
}
DIFFICULTY = dict(zip(TASK_SET_ORDER, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))


def validate(task):
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError(f"Unknown lockkey task {task!r}; choose from {TASKS}")
    return task


def contract(task):
    return UNLOCK_CONTRACT if validate(task) == DEFAULT_TASK else CONTRACT


def description(task):
    return INSTRUCTIONS[validate(task)] + DESCRIPTION[len(INSTRUCTION):]


def task_resolve(selection):
    if not isinstance(selection, dict) or set(selection) - {"task"}:
        raise ValueError("Lockkey accepts only the task selector; no seeds or reset options")
    task = validate(selection.get("task", DEFAULT_TASK))
    # Preserve the old empty selection, backend options, id and instruction exactly.
    return dict(task_id=f"lockkey/{task}", description=INSTRUCTIONS[task],
                backend_options={} if task == DEFAULT_TASK else {"task": task},
                selection={} if task == DEFAULT_TASK else {"task": task}, split=None)

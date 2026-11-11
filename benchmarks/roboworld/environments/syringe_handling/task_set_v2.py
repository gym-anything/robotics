"""syringe-tasks-v2: the render-checked eight-task set, beside the released set (config.EIGHT_TASK_CONTRACT,
syringe-tasks-v1).

The render check changed no task of the set: on the fix branch the starts, instructions, success
predicates, progress (scoring.py) and 0.9 failure factor are the earlier set's, which the release ships as
syringe-tasks-v1. Its other fixes were to the environment (the trace's world events and a backend help note, both put
back as at the base there; the release keeps its own, also under this contract) and to a source manifest the release
does not ship. So this contract is judged as syringe-tasks-v1 (task.py and scoring.py) and reports itself, and its
selections get their own task ids by the released rule (contract-syringe-tasks-v2/ before the seed). vial-to-pad
keeps its released selection (splits/test.json) and its own evaluator. Selections without this contract resolve,
launch and score as released.
"""

from .config import EIGHT_TASK_CONTRACT

CONTRACT = "syringe-tasks-v2"
TASK_SET_CONTRACTS = (EIGHT_TASK_CONTRACT, CONTRACT)          # judged alike (task.py, scoring.py)
HARDEST_SELECTION = {"preset": "injection", "task": "vial-to-pad"}     # splits/test.json, unchanged
NOTE = (f"Render-checked eight-task set, contract {CONTRACT}; each of the seven other tasks selects it "
        "and gets its own task id with the contract before the seed "
        f"(syringe-handling/v1/<preset>/<task>/contract-{CONTRACT}/seed-0). Starts, instructions, success predicates, "
        f"progress and the 0.9 failure factor are task_set_v1's ({EIGHT_TASK_CONTRACT}); only the ids and the reported "
        "contract differ. vial-to-pad keeps its released selection (splits/test.json) and its own evaluator.")


def resolve_contract(task):
    """config.resolve_contract for this contract (it hands it here)."""
    if task == "vial-to-pad":
        raise ValueError(f"vial-to-pad keeps its own evaluator; {CONTRACT} is for the seven other tasks "
                         "(select contract='v1' or omit it)")
    return CONTRACT


def selections():
    """The eight exact selections, in difficulty order: the seven under this contract, then vial-to-pad as released."""
    from .variants import eight_task_selections
    return [dict(s, contract=CONTRACT) for s in eight_task_selections() if s["task"] != "vial-to-pad"] \
        + [dict(HARDEST_SELECTION)]


def reward(env, criteria, success):
    """task.evaluate's reward under this contract: syringe-tasks-v1's (scoring.reward), reporting this contract."""
    from .scoring import reward as task_set_v1_reward
    return dict(task_set_v1_reward(env, criteria, success), contract=CONTRACT)


def contract():
    """GET /v1/variations of a session under this contract: variants.contract() with this contract and this set."""
    from .variants import contract as released
    c = released()
    c["contracts"] = dict(c["contracts"], **{CONTRACT: f"the render-checked set: {EIGHT_TASK_CONTRACT}'s reward, "
                                                       "its own task ids"})
    c["task_set"] = selections()
    return c


def catalogue_fields():
    """The catalogue fields of benchmark.json (variants.main writes them): variants.catalogue_fields() with the released
    set under task_set_v1, this contract among the contract choices and this set under task_set."""
    from .variants import catalogue_fields as released
    fields = released()
    out = {("task_set_v1" if k == "task_set" else k): v for k, v in fields.items()}
    choices = fields["selector_choices"]
    out["selector_choices"] = dict(choices, contract=[*choices["contract"], CONTRACT])
    out["task_set"] = dict(selections=selections(), note=NOTE)
    return out

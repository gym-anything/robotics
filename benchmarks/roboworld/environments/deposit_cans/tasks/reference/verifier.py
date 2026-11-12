"""Outcome of reference: success once all 24 target cans lie released and wholly inside the bag, with the station
upright, the bag retained, the empty tub back on the counter, the reusable cups on their tray, no can on the floor or
out of the bag, and all of it still for a second. Under v1 the score is the share of targets deposited.

verify(progress,e) receives the episode's task.Progress and the environment after a physics substep and returns the
evaluation (success, score and their evidence) of the shared evaluator, Progress.evaluate_criteria, which judges by
the configuration's contract. CONTRACT is the variant's default contract (task.task_contract).
reference has a choice (variants.CONTRACT_CHOICES): v1 by default, TASKS_V2_CONTRACT, the high_fill rule, when the
selection names it."""

CONTRACT='v1'

def verify(progress,e):
    return progress.evaluate_criteria(e)

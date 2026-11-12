"""Outcome of single_can: success once every target can rests released and wholly inside the bag, with the station
upright, the bag retained, the empty tub back on the counter and the reusable cups on their tray, none of them held,
and all of it still over the last second. Otherwise the score is 0.9 times the signed progress: the target credit less
one unit per item out of its place.

verify(progress,e) receives the episode's task.Progress and the environment after a physics substep and returns the
evaluation (success, score and their evidence) of the shared evaluator, Progress.evaluate_criteria, which judges by
the configuration's contract. CONTRACT is the variant's default contract (task.task_contract)."""
from deposit_cans.task import TASKS_CONTRACT

CONTRACT=TASKS_CONTRACT

def verify(progress,e):
    return progress.evaluate_criteria(e)

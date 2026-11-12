"""Outcome of tub_cups: success once every target can rests released and wholly inside the bag and the three reusable
cups are back on their tray, with the station upright, the bag retained, the empty tub back on the counter, none of
them held, and all of it still over the last second. Otherwise the score is 0.9 times the signed progress: the target
credit plus one credit per cup on the tray, less one unit per item out of its place (a cup in the bag or on the floor).

verify(progress,e) receives the episode's task.Progress and the environment after a physics substep and returns the
evaluation (success, score and their evidence) of the shared evaluator, Progress.evaluate_criteria, which judges by
the configuration's contract. CONTRACT is the variant's default contract (task.task_contract).
CUPS_OFF_TRAY makes putting the cups back part of the job."""
from deposit_cans.variants import TASKS_V2_CONTRACT

CONTRACT=TASKS_V2_CONTRACT
CUPS_OFF_TRAY=True   # the reusable cups start off the tray: put back is credited, not charged

def verify(progress,e):
    return progress.evaluate_criteria(e)

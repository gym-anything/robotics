"""Outcome of sort_waste: success once every target can rests released and wholly inside the bag and both paper cups
stand untouched in the upright landfill bin, with the station upright, the bag retained, the empty tub back on the
counter, the reusable cups on their tray, none of them held, and all of it still over the last second. Otherwise the
score is 0.9 times the signed progress: the target credit plus one credit per paper cup in the bin, less one unit per
item out of its place (a paper cup in the bag or on the floor).

verify(progress,e) receives the episode's task.Progress and the environment after a physics substep and returns the
evaluation (success, score and their evidence) of the shared evaluator, Progress.evaluate_criteria, which judges by
the configuration's contract. CONTRACT is the variant's default contract (task.task_contract).
The evaluator judges the paper cups because the start has them (props.build, assembly objects 'paper_cups')."""
from deposit_cans.variants import TASKS_V2_CONTRACT

CONTRACT=TASKS_V2_CONTRACT

def verify(progress,e):
    return progress.evaluate_criteria(e)

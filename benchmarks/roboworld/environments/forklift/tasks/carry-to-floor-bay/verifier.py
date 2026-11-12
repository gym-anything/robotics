"""Outcome of carry-to-floor-bay: the pallet set down inside the target bay's lines, square to them and resting on the
floor, the forks backed out at least 0.5 m and lowered, the truck at rest (evaluator._set_down_in_bay, as for
carry-loose-stack).

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met); its details are empty. measurements(world, start, by),
progress(now, initial, approach) and site_details(world, start) are the task's forklift-tasks-v1 and -v2 grading
(task_progress.py)."""
from forklift import evaluator as E
from forklift.evaluator import _set_down_in_bay
from forklift.task_progress import set_down, set_down_progress


def verify(world, start):
    return _set_down_in_bay(world, start)


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): the set-down measures in the target bay."""
    return set_down(world, by, world.scene.zones["bays"][start["bay"]], yard=False)


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): the set-down progress."""
    return set_down_progress(now, initial, approach)


def site_details(world, start):
    """The shared site rules' details (strikes, dropped loads) for the contracts' workmanship
    (task_progress.workmanship): verify returns no details, though shared() measured its latches."""
    zone = world.scene.zones["bays"][start["bay"]]
    def destination(name, xy):
        return name == "carried" and E.inside_rect(xy, (zone["x"][0] - .05, zone["x"][1] + .05),
                                                   (zone["y"][0] - .05, zone["y"][1] + .05))
    _, det = E.shared(world, start, dest_floor=destination)
    return det

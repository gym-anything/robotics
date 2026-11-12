"""Outcome of turn_face_down: the T-shirt turned over and laid out flat: at least 90 % of the front panel's own
particles below a nearby back panel (within 45 mm in plan), the body's plan hull at least 85 % of the start's and the
body within 30 mm of height (FaceDownGoal).
Both task-set contracts add the shared gates (at rest over 0.5 s, released, on the table, stretch within the
garment's limit, no cloth crossings) and score 1 on success, else 0.9 x progress from the start.

goals(contract) gives the spec's goals under laundry-tasks-v1 and -v2 (task_set.task_spec, task_set_v2.task_spec);
verify(task, positions_m, velocities_m_s, **evidence) judges an episode state with the evaluator bound to the
selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain, crossing, release, table and rest
evidence) and returns its garment_task.GarmentTaskResult."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from laundry.garment_task import GoalEvaluation, _convex_hull_area, _project
from laundry.task_set_goals import above, below


def goals(contract):
    return (FaceDownGoal(),)


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)


@dataclass(frozen=True)
class FaceDownGoal:
    id: str = "face_down"
    weight: float = 1.
    uses_context: ClassVar[bool] = True

    def validate(self, index):
        index.panel_vertices(("front", "back"))

    def prepare(self, index, reference):
        front, back = (index.panel_vertices((p,)) for p in ("front", "back"))
        return (np.setdiff1d(front, back), np.setdiff1d(back, front),
                _convex_hull_area(reference[np.union1d(front, back), :2]))

    def evaluate(self, index, positions_m, table_normal, context):
        front, back, reference_area = context.prepared
        fp, bp = positions_m[front], positions_m[back]
        squared = np.sum((_project(fp, table_normal)[:, None] - _project(bp, table_normal)[None]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        gap = (bp[nearest] - fp) @ table_normal
        # A nearby backing surface is necessary: an unrelated raised corner is not a flipped shirt.
        plan_distance = np.sqrt(squared[np.arange(len(front)), nearest])
        fraction = float(np.mean((gap > .0001) & (plan_distance <= .045)))
        area = _convex_hull_area(_project(np.concatenate((fp, bp)), table_normal)) / reference_area
        thickness = float(np.ptp(np.concatenate((fp, bp)) @ table_normal))
        flip = float(np.mean(np.clip((gap + .002) / .004, 0., 1.) / (1 + plan_distance / .045)))
        layout = above(area, .85) * below(thickness, .030)
        # Layout initially passes, so it modulates flip credit instead of awarding idle credit.
        quality = flip * (.5 + .5 * layout)
        return GoalEvaluation(self.id, dict(flipped_fraction=fraction, body_area_fraction=area,
                              body_normal_extent_m=thickness, flip_quality=flip),
                              dict(face_down=fraction >= .90, laid_out=area >= .85, flat=thickness <= .030),
                              quality, "front below the nearby back panel, body laid out flat")

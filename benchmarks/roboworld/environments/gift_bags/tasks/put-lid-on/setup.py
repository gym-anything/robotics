"""Start of put-lid-on: lift-lid's gift box standing open on the dining table with the photo frame inside, its lid
upright on the table beside it, toward the robots (lid_offset: at least 170 mm, more for the larger boxes).

setup(spec, contract) changes the home preset's scene spec (scene.SceneSpec of lift-lid's world) in place before the
contact world is built. The task-set env adapters call it through task_set_presets.prepare (gift-bags-tasks-v1) and
task_set_v2_presets.prepare (gift-bags-tasks-v2); both contracts start the same way."""
import numpy as np


def lid_offset(half_y, clearance, wall):
    # The seed-0 robot demo carried the lid 170 mm toward the front. Larger boxes need more space.
    return max(.17, 2*half_y + clearance + wall + .035)


def setup(spec, contract):
    task = "put-lid-on"
    bodies = {b.name: b for b in spec.bodies}
    base, lid, g = bodies["gift_base"], bodies["gift_lid"], spec.geometry["box"]
    origin = base.p - base.R @ base.render["com_offset"]
    offset = lid_offset(g["hy"], g["clearance"], g["wall"])
    lid.p = origin + base.R @ np.array([0., -offset, 0.]) + lid.R @ lid.render["com_offset"]
    low = float((np.asarray(lid.verts) @ lid.R.T + lid.p)[:, 2].min())
    lid.p[2] += spec.surface_z + .00005 - low
    spec.geometry["task_start"] = dict(task=task, lid_forward_m=offset)

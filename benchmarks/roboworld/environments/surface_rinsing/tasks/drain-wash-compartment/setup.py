"""Start of drain-wash-compartment: the pot sink as rinse-sponge starts it, wash, rinse and sanitizer compartments
filled and their drains shut, with no movable ware or tool.

setup(task, seed) gives the episode spec of the task and seed (task_set.catalogue.setup adds the instruction and the
contract): rinse-sponge's scene and seeded draws (layout.setup), its items removed. The task is not in layout.TASKS; it
exists only under the task-set contracts."""
from surface_rinsing import layout


def setup(task, seed=0):
    # Only the new task has a new scene: same filled sink, no movable ware/tool.
    spec = layout.setup("rinse-sponge", seed)
    spec["items"] = {}
    spec["task"] = task
    return spec

"""Start of set-post: the post preset with the post hole dug before tick 0 (its spoil loose on the tarp, a gravel
base), the 4x4 post standing in it braced plumb on two faces, the trowel in the right hand, the 2x4 tamper in the left
and the trenching shovel lying on the walkway.

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands);
predug, the excavation dug before tick 0 (layout.predug); and kit, the tools lying on the walkway (the ~35 L backfill is
shovel work, not 100+ trowel loads). SECOND is the second tool it holds and that tool's hand (env.SECOND)."""

SECOND = ("tamper", "left")


def setup():
    return dict(held=dict(right="trowel", left="tamper"), predug=dict(kind="post_hole", post=True),
                kit=["trench_shovel"])

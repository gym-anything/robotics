"""Start of take-knife: the gutted trout on the board and the fillet knife standing handle-up in the knife block's slot,
right of the board; the right hand, with its knife-handle inserts, starts empty.

setup() gives the choices this start makes (layout.task_start): knife, where the knife starts when the selection names
no start (world.World builds the knife in the block's slot)."""


def setup():
    return dict(knife="block")

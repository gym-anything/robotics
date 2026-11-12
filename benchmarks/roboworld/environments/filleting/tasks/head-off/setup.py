"""Start of head-off: the gutted trout on its side on the board, head toward the holding (left) arm, back toward the
robots, the fillet knife in the right hand's insert grasp; seeds vary the fish's length, its rib and pin-bone counts
and its pose.

setup() gives the choices this start makes (layout.task_start): collision_profile, the robots' collision thresholds
when the selection names none (libfranka's contact-task thresholds, as for every cutting task:
variants.collision_profile)."""


def setup():
    return dict(collision_profile="contact")

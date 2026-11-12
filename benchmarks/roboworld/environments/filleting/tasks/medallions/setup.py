"""Start of medallions: the pork tenderloin on the board, thick end toward the knife hand, the fillet knife in the
right hand's insert grasp; seeds vary the loin's length and its pose.

setup() gives the choices this start makes (layout.task_start): collision_profile, the robots' collision thresholds
when the selection names none (libfranka's contact-task thresholds, as for every cutting task:
variants.collision_profile)."""


def setup():
    return dict(collision_profile="contact")

"""Start of excise-defect: the salmon half side skin-down on the board with a dark melanin spot set into its belly
flap, the bench scale beside the board, the fillet knife in the right hand's insert grasp; seeds vary the side's length,
the spot's size and place, and the side's pose.

setup() gives the choices this start makes (layout.task_start): collision_profile, the robots' collision thresholds
when the selection names none (libfranka's contact-task thresholds, as for every cutting task:
variants.collision_profile)."""


def setup():
    return dict(collision_profile="contact")

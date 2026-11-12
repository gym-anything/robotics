"""Start of put-away-knife: take-knife's scene (the trout on the board, the knife block right of it) with the fillet
knife already in the right hand's insert grasp, as the cutting tasks start; the task is to return it to its slot.

setup() gives the choices this start makes (layout.task_start): scene, the task whose scene it is built in (A0,
take-knife; a scene carrier only, never the public task or its evaluator), and knife_start, the knife start the world
is built with, which a selection may not contradict (task_set_catalogue.start_options)."""


def setup():
    return dict(scene="A0", knife_start="hand")

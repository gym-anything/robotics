"""Outcome of write-text: HELLO in black where the reference picture shows it and at its size, and no ink of another
colour on the board. Read from the final board pigment alone.

verify(task, ink, config, state) gives the verdict of a write-text preset (task_set/verifier.py's evaluate; ink the
board's InkSurface, config and state unused: READS_STATE tells the task-set backends not to read the body state): a
dict with score, success, progress, checks (lettering, other_colors_clean) and factors, from the lettering judgement
shared with write-in-colour (task_set/verifier.py).
"""
from whiteboard.task_set.verifier import lettering

READS_STATE = False


def verify(task, ink, config, state=None):
    return lettering(task, ink, "black")

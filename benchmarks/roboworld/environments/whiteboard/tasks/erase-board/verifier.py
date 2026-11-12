"""Outcome of erase-board: the board wiped clean with no marks added, the wiper resting in the tray and the robot
standing clear of it.

verify(task, ink, config, state) gives the verdict of an erase-board preset (task_set/verifier.py's evaluate): ink is
the board's InkSurface, config the episode's Config, state the body state the task-set backends read
(mechanical.read_state: tool poses and velocities, robot contacts, pelvis height; None raises ValueError). It returns
a dict with score, success, progress, checks (erased, no_added_marks, kept, standing, returned), factors and the tray
readings, from the erasing judgement shared with erase-except-boxed (task_set/verifier.py).
"""
from whiteboard.task_set.verifier import erasing


def verify(task, ink, config, state=None):
    return erasing(task, ink, config, state)

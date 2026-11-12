"""Start of carry-to-floor-bay: the truck in receiving lane R2 heading west, forks at travel height and tilted back, a
wrapped unit load of 600-1,200 kg and 0.9-1.5 m on them; the target floor bay is one of task.FLOOR_BAYS.

setup(start) gets the task.Start that task.resolve builds on, after the start this task shares with carry-loose-stack
(the truck's pose and fork position, then the bay), and puts the task's load on the forks in start.s. It returns
nothing."""


def setup(start):
    s, rng = start.s, start.rng
    s["load_on_forks"] = dict(kind="wrapped", mass=float(rng.uniform(600.0, 1200.0)),
                              height=float(rng.uniform(0.9, 1.5)))

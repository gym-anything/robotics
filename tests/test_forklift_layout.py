"""Forklift warehouse layout (no physics): a refused start state names its problems."""

import numpy as np


def _layout(task, seed):
    from forklift import facility as F, task as TK
    from forklift.config import Config
    cfg = Config(preset="warehouse", task=task, seed=seed, robot="none")
    return F.warehouse(np.random.default_rng(1000 + seed), reserved=TK.reserved_locations(cfg))


def test_refused_start_state_names_its_problems():
    from forklift.config import Config
    from forklift.env import StartStateRefused
    check = dict(problems=["penetration 25.1 mm (over the 10 mm limit)", "contact force 88395 kN at rest between a and b"])
    exc = StartStateRefused(Config(preset="warehouse", task="put-away-rack", seed=900, robot="none"), check,
                            dict(settled=True))
    assert isinstance(exc, RuntimeError)
    assert "put-away-rack seed 900" in str(exc) and "penetration 25.1 mm" in str(exc) and "88395 kN" in str(exc)
    assert exc.problems == check["problems"] and exc.start_check is check

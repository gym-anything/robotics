"""Hold unchanged named joint targets. Interface demonstration, NOT a solver."""


def create(context, options):
    if set(options) != {"ticks"} or type(options["ticks"]) is not int or options["ticks"] < 1:
        raise ValueError("Explicit positive integer ticks is required")
    def step(observation):
        return {"type": "named_joint_target", "targets_si": {}, "ticks": options["ticks"]}
    return step

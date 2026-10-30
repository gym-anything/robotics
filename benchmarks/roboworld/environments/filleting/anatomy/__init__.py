"""Anatomical workpiece generators.

    from filleting.anatomy import load
    wp = load("trout", seed=0)          # also "salmon" (half side, Trim A) and "tenderloin"
    wp.sdf(p), wp.region(p), wp.reference(p), wp.rigid_bodies, wp.rods, wp.measured ...

Seed 0 is the nominal workpiece; other seeds apply a seeded variation.
"""

PRESETS = ("trout", "salmon", "tenderloin")


def load(preset, seed=0, **kw):
    """Build a workpiece by preset name ('trout' | 'salmon' | 'tenderloin', aliases 'A' | 'B' | 'C')."""
    name = {"a": "trout", "b": "salmon", "c": "tenderloin"}.get(str(preset).lower(), str(preset).lower())
    if name == "trout_fillet_rib_strip":
        from filleting.anatomy import trout as mod
        return mod.build_fillet_rib_strip(seed=seed, **kw)
    if name == "trout":
        from filleting.anatomy import trout as mod
    elif name == "salmon":
        from filleting.anatomy import salmon as mod
    elif name == "tenderloin":
        from filleting.anatomy import tenderloin as mod
    else:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    return mod.build(seed=seed, **kw)

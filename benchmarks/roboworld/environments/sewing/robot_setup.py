"""Which robot setups sewing can host (robots.catalog ENVIRONMENTS["sewing"]["validator"]).

Four catalog rigs by alias: dual-fr3 (sewing-fr3, the default), dual-panda (sewing-panda), single-fr3
(sewing-fr3-single) and g1 (sewing-g1, the humanoid standing at the table: sewing/g1.py). A setup passed as an object
(a RobotSetup, its dict, or a rig JSON path) is accepted when it has one of those rigs' structure: the same instances
(names, robot models, mounts, controller profiles), attachments and roles, whatever its id or description (a renamed
copy of a rig is accepted). It then runs exactly as that alias. Other structures are
refused: the cell has a left and a right arm mount on the operator's pedestal (layout.ROBOT_BASES) or the humanoid's
standing place, the cloth couples to the Franka Hand's finger boxes and housing or the Dex3 links and palm, and a
setup no rig has (a lone left arm, a single Panda) has no mounting or start state built for it.
"""

import json

ALIASES = {"dual-fr3": "sewing-fr3", "dual-panda": "sewing-panda", "single-fr3": "sewing-fr3-single", "g1": "sewing-g1"}


def structure(setup):
    """What a setup builds: its instances, attachments and roles; not its id or description."""
    d = setup.to_dict()
    return json.dumps({k: d.get(k, {}) for k in ("instances", "attachments", "roles")}, sort_keys=True)


def validate_setup(selected):
    from robots.catalog import resolve
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, FileNotFoundError, KeyError) as exc:
        raise ValueError(f"sewing: not a robot setup ({exc}); choose one of {', '.join(ALIASES)}") from exc
    for alias, rig in ALIASES.items():
        if setup.fingerprint == resolve(rig).fingerprint:
            return alias
    s = structure(setup)
    for alias, rig in ALIASES.items():
        if s == structure(resolve(rig)):
            return alias
    raise ValueError(
        "sewing hosts its four catalog rigs, by alias or as setups with one of their structures (instances, mounts, "
        "models, controller profiles, attachments; the id and description may differ): "
        + ", ".join(f"{a} ({r})" for a, r in ALIASES.items())
        + ". Other setups are not built: the cell has a left and a right arm mount on the operator's pedestal, the "
          "cloth couples to a Franka Hand's fingers and housing, and each arm needs the hand TCP for IK")

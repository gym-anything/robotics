"""Which robot setups filleting can host (robots.catalog ENVIRONMENTS["filleting"]["validator"]).

The environment offers three catalog rigs by alias: dual-fr3 (filleting-fr3, the default), dual-panda (filleting-panda)
and single-fr3 (filleting-fr3-single, take-knife only: variants.check_robot). A setup passed as an object (a
RobotSetup, its dict, or a rig JSON path) is accepted when it is one of those rigs (same catalog fingerprint) and then
runs exactly as its alias; any other setup is refused with what the environment needs. Custom setups are not built:
the bench's two arm mounts (layout.ROBOT_BASES), the knife-handle inserts authored on the right Franka Hand
(scene._finger_colliders), the per-arm IK tool frames and the wrist cameras are qualified for these three rigs only.
"""

ALIASES = {"dual-fr3": "filleting-fr3", "dual-panda": "filleting-panda", "single-fr3": "filleting-fr3-single"}


def validate_setup(selected):
    from robots.catalog import resolve
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, FileNotFoundError, KeyError) as exc:
        raise ValueError(f"filleting: not a robot setup ({exc}); choose one of {', '.join(ALIASES)}") from exc
    for alias, rig in ALIASES.items():
        if setup.fingerprint == resolve(rig).fingerprint:
            return alias
    raise ValueError(
        "filleting hosts its three catalog rigs, by alias or as the rig objects themselves: "
        + ", ".join(f"{a} ({r})" for a, r in ALIASES.items())
        + ". Other setups are not built: the bench has a left and a right arm mount, the right hand must carry the "
          "authored knife-handle inserts of a Franka Hand (the knife is held by friction in them), and each arm needs "
          "an IK tool frame, a parallel hand with gripper joints and a wrist camera")

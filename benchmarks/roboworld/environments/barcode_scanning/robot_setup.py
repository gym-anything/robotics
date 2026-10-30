"""Which robot setups barcode-scanning can host (robots.catalog ENVIRONMENTS["barcode-scanning"]["validator"]).

The environment offers four catalog rigs by alias: dual-fr3 (scan-fr3), dual-panda (scan-panda), single-fr3
(scan-fr3-single) and g1 (scan-g1). A setup passed as an object (a RobotSetup, its dict, or a rig JSON path) is accepted
when it is one of those rigs (same catalog fingerprint) and then runs exactly as its alias; any other setup is refused
with what the environment needs. Custom setups are not built: the cart's two arm mounts, the per-arm IK tool frames, the
fingertip-pad contact model, the cameras and the reach table are qualified for these four rigs only.
"""

ALIASES = {"dual-fr3": "scan-fr3", "dual-panda": "scan-panda", "single-fr3": "scan-fr3-single", "g1": "scan-g1"}


def validate_setup(selected):
    from robots.catalog import resolve
    try:
        setup = resolve(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"barcode-scanning: not a robot setup ({exc}); choose one of {', '.join(ALIASES)}") from exc
    for alias, rig in ALIASES.items():
        if setup.fingerprint == resolve(rig).fingerprint:
            return alias
    raise ValueError(
        "barcode-scanning hosts its four catalog rigs, by alias or as the rig objects themselves: "
        + ", ".join(f"{a} ({r})" for a, r in ALIASES.items())
        + ". Other setups are not built: the cart has a left and a right arm mount, and each arm needs an IK tool frame, "
          "a parallel hand with fingertip pads and gripper joints, wrist cameras and a measured reach to the cradle, "
          "keypad and tote")

"""Which robot setups welding can host (robots.catalog ENVIRONMENTS["welding"]["validator"]).

Six catalog rigs by alias: ur10e-fr3 (welding-ur10e-fr3, the default: UR10e torch arm + FR3 with a Franka Hand),
ur10e (welding-ur10e, the torch arm alone), dual-fr3 (welding-fr3, the right FR3 carrying the light torch, the left a
Franka Hand), single-fr3 (welding-fr3-single, that FR3 torch arm alone), dual-panda (welding-panda, the light torch on
the right Panda, the left a Franka Hand) and g1 (welding-g1, the Unitree G1 standing at the table: the light torch
clamped to its right Dex3 palm, the left Dex3 handling; welding/humanoid.py); the two torch-only rigs are refused for
the tack task by variants.check_robot (at launch through variants.check_launch). A setup passed as an object (a RobotSetup, its dict, or a rig
JSON path) is accepted when it is one of those rigs (same catalog fingerprint) and then runs exactly as its alias; any
other setup is refused with what the cell needs.
"""

ALIASES = {"ur10e-fr3": "welding-ur10e-fr3", "ur10e": "welding-ur10e", "dual-fr3": "welding-fr3",
           "single-fr3": "welding-fr3-single", "dual-panda": "welding-panda", "g1": "welding-g1"}


def validate_setup(selected):
    from robots.catalog import resolve
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, FileNotFoundError, KeyError) as exc:
        raise ValueError(f"welding: not a robot setup ({exc}); choose one of {', '.join(ALIASES)}") from exc
    for alias, rig in ALIASES.items():
        if setup.fingerprint == resolve(rig).fingerprint:
            return alias
    raise ValueError(
        "welding hosts its six catalog rigs, by alias or as the rig objects themselves: "
        + ", ".join(f"{a} ({r})" for a, r in ALIASES.items())
        + ". Other setups are not built: the cell has a torch-arm mount and a handling-arm mount at the fixture table, "
          "the torch arm must carry a qualified gun within its payload (the 2.76 kg cobot gun on the UR10e, the 1.6 kg "
          "light torch on the FR3, the Panda or the G1's right hand), and the handling arm needs a Franka Hand or the "
          "G1's Dex3")

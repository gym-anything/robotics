"""Task instruction text shared by the catalogue and the policy; imports no physics.

Revision 2 (offset-lower-service) names each lead by what it
is and by the station's printed markings, states the outcome (PASS) instead of a
button action, and states the hands-off requirement as the job's outcome. Scoring
tolerances are documented in task.py, not in the instruction.
Revision 1 presets keep their original texts in benchmark.py and task.py.
"""


def _value(config, name, default):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def revision(config):
    return _value(config, "task_revision", 1)


def instruction(config):
    """Revision-2 instruction, or None for a revision-1 configuration."""
    if revision(config) < 2:
        return None
    port = _value(config, "sensor_port", "X1")
    return ("Restore the inspection station's control connections following the wiring schedule: "
            f"sensor harness from {port} to the B1 photoeye, network cable from LAN1 to the PLC, "
            "brown lead from SUPPLY L+ to I/O L+, and blue lead from SUPPLY M to I/O M. "
            "Route the sensor harness through guide G1, the network cable through G2, the brown lead "
            "through G3 and the blue lead through G4. Keep the existing LAN2 uplink connected throughout. "
            "Commission the finished installation with the COMMISSION/TEST button so that the station "
            "shows PASS. Then leave every connection holding by itself and keep the robot clear of the "
            "cables and connectors for two seconds before you finish.")

"""One standard Panda; actuator limits from the pinned manufacturer URDF."""

import math
import xml.etree.ElementTree as ET
from .runtime import require_engine

HOME = [0., -.4, 0., -2.2, 0., 1.9, .79, .04, .04]
# Delicate lock work uses the Panda controller's 20% operating-speed setting.
LOCK_OPERATION_SPEED = .2


def add_robot(builder, c):
    from robots.backends.newton import add_robot as build, mount
    addresses = build(builder,'panda','lockkey',instance='panda',
        placement=mount((.54,0,c.table_height+.11),math.pi)).metadata['legacy_addresses']
    for name, spec in addresses.items():
        if "finger" not in name:
            spec["velocity_limit"] *= LOCK_OPERATION_SPEED
    return addresses

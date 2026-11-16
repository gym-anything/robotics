"""One standard Panda; actuator limits from the pinned manufacturer URDF."""

import math
import xml.etree.ElementTree as ET
from .runtime import require_engine

HOME = [0., -.4, 0., -2.2, 0., 1.9, .79, .04, .04]
# Delicate lock work uses the Panda controller's 20% operating-speed setting.
LOCK_OPERATION_SPEED = .2


def add_robot(builder, c, materials):
    import newton
    from robots.backends.newton import add_robot as build, mount
    from pencap.newton15.wrist_camera import install
    addresses = build(builder,'panda','lockkey',instance='panda',
        placement=mount((.54,0,c.table_height+.11),math.pi)).metadata['legacy_addresses']
    for name, spec in addresses.items():
        if "finger" not in name:
            spec["velocity_limit"] *= LOCK_OPERATION_SPEED
    # The D405 wrist camera (pencap/newton15/wrist_camera.py), named left_wrist_rgb as the actions call this arm
    # 'left'; its case and bracket touch the scene like the workshop furniture (scene.py).
    install(builder, "panda", newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4, ke=c.contact_ke,
        kd=c.contact_kd, mu_torsional=0., mu_rolling=0.), materials, "left_wrist_rgb")
    return addresses

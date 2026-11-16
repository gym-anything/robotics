"""Pinned manufacturer Panda geometry, inertia and limits; physical servos."""
import math
import xml.etree.ElementTree as ET
from .runtime import require_engine


def add_robot(builder, arm, config, materials):
    import newton
    from robots.backends.newton import add_robot as build, mount
    from pencap.newton15.wrist_camera import install
    sign=1 if arm=='left' else -1
    addresses = build(builder,'panda','laundry',instance=arm,
        placement=mount((0.,sign*config.robot_base_y,config.table_top),-sign*math.pi/2)).metadata['legacy_addresses']
    # The D405 wrist camera on every arm (pencap/newton15/wrist_camera.py); its case and bracket touch the scene
    # like the workstation furniture (scene.py).
    install(builder, arm, newton.ModelBuilder.ShapeConfig(mu=.4, ke=1e7, kd=1000., gap=.0003), materials)
    return addresses

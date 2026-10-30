"""Pinned manufacturer Panda geometry, inertia and limits; physical servos."""
import math
import xml.etree.ElementTree as ET
from .runtime import require_engine


def add_robot(builder, arm, config):
    from robots.backends.newton import add_robot as build, mount
    sign=1 if arm=='left' else -1
    return build(builder,'panda','laundry',instance=arm,
        placement=mount((0.,sign*config.robot_base_y,config.table_top),-sign*math.pi/2)).metadata['legacy_addresses']

"""Manufacturer-description Panda with normal joint/contact actuation."""

import xml.etree.ElementTree as ET
from .runtime import require_newton


def add_panda(builder, position):
    from robots.backends.newton import add_robot as build, mount
    return build(builder,'panda','titration',instance='panda',prefix='',
        placement=mount(position)).metadata['legacy_addresses']

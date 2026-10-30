"""FR3 manufacturer URDF plus an explicitly authored WSG50 mounting assembly.
No alteration of arm limits. Gripper geometry/inertia estimates are recorded.
"""
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import newton
import newton.utils
import warp as wp
from newton.solvers import SolverMuJoCo

HOME=(0.,-.4,0.,-2.2,0.,1.9,.79)
TCP_Z=.107+.008+.0725+.0245+.045

def generated_urdf():
    from robots.attachments.wsg50.geometry import generated_urdf as generate
    return generate()

def add_robot(a,name,position,yaw):
    from robots.backends.newton import add_robot as build, mount
    selection=getattr(a,'robot_selection',{}).get(name,{})
    handle=build(a.builder,selection.get('model','fr3'),selection.get('controller_profile','deposit_cans'),
        instance=name,placement=mount(position,yaw),shape_config=a.shape_config(),sdf=True,
        parameters={'finger_effort_limit':a.config.gripper_force/2,'finger_velocity_limit':a.config.gripper_speed})
    a.sources['fr3_urdf']=handle.source  # Retained legacy provenance key.
    a.robot_joints.update(handle.metadata['legacy_addresses'])
    for i in handle.shape_ids:a.materials[i]='robot'
    a.robot_bases[name]=position;a.robot_yaws[name]=yaw

"""Reconstructed manual air-displacement filler, mechanically part of pipette.

Reference: Kartell plunger filler article 198 (40 mm diameter, 100 mm body,
10 mL). Interior geometry, dead volume and seal friction are disclosed
engineering estimates. This is a tool assembly, never a robot attachment.
"""

import math
import numpy as np
from .runtime import require_newton
from .geometry import revolve, profile_wall_cells

PISTON_RADIUS_M = .010
PISTON_AREA_M2 = math.pi*PISTON_RADIUS_M**2
# Nominal liquid capacity is not piston swept volume. Drawing 10 mL upward
# expands the initial air cushion as well: V_swept = V_liquid +
# V_air0*(p0/(p0-rho*g*h)-1). At the 0.34 m mark this is 10.446 mL.
# A 12 mL swept-volume limit leaves the piston face at z=0.4876 m, inside
# the existing 0.505 m barrel end. No barrel or pipette geometry is enlarged.
STROKE_M = 12.e-6/PISTON_AREA_M2
DEAD_VOLUME_M3 = 2.e-6
VENT_RADIUS_M = .0005
VENT_TRAVEL_M = .002
CHAMBER_BASE_Z_M = .443
PISTON_FACE_Z_M = CHAMBER_BASE_Z_M+DEAD_VOLUME_M3/PISTON_AREA_M2


def add_filler(lab, pipette):
    newton, wp = require_newton()
    b = lab.builder
    base = b.body_q[pipette]
    free = b.add_joint_free(child=pipette, label='volumetric_pipette_free_joint')
    # Silicone chuck and ABS barrel are fixed components of the same tool.
    for name, polygon, material, density in (
        ('pipette_silicone_chuck', [(.006, .419), (.006, .439), (.0028, .439), (.0028, .419)], 'rubber', 1100.),
        ('pipette_filler_barrel', [(.016, .439), (.016, .505), (.010, .505),
            (.010, .443), (.002, .443), (.002, .439)], 'instrument_blue', 1040.)):
        surface = revolve(polygon)
        shape = b.add_shape_mesh(pipette, mesh=newton.Mesh(surface.vertices, surface.triangles.flatten()),
            cfg=b.ShapeConfig(density=density, mu=.5, has_shape_collision=False), label=name)
        lab.materials[shape] = material
        for i, cell in enumerate(profile_wall_cells(polygon)):
            b.add_shape_convex_hull(pipette,
                mesh=newton.Mesh(cell.vertices, cell.triangles.flatten()),
                cfg=b.ShapeConfig(density=0., mu=.5, is_visible=False,
                                  has_particle_collision=False), label=f'{name}_contact_{i}')
    piston_origin = wp.transform((0., 0., PISTON_FACE_Z_M+.001), wp.quat_identity())
    piston = b.add_link(xform=wp.transform_multiply(base, piston_origin), label='pipette_plunger')
    lab.cylinder('filler_piston', (0., 0., 0.), .00995, .002, 'polypropylene', body=piston, density=950.)
    shaft_length = .519-(PISTON_FACE_Z_M+.002)
    lab.cylinder('filler_piston_rod', (0., 0., .001+shaft_length/2), .003, shaft_length,
                 'polypropylene', body=piston, density=950.)
    lab.cylinder('filler_thumb_disc', (0., 0., .520-PISTON_FACE_Z_M-.001), .020, .006,
                 'instrument_blue', body=piston, density=1040.)
    pj = b.add_joint_prismatic(pipette, piston, parent_xform=piston_origin,
        axis=(0., 0., 1.), limit_lower=0., limit_upper=STROKE_M,
        target_ke=0., target_kd=0., friction=.5, damping=.5, label='pipette_plunger_slide')
    # The button faces upward in the authored horizontal rack pose.
    vent_origin = wp.transform((-.019, 0., .478), wp.quat_identity())
    vent = b.add_link(xform=wp.transform_multiply(base, vent_origin), label='pipette_vent_button')
    lab.box('pipette_vent_button', (0., 0., 0.), (.006, .010, .012),
            'polypropylene', body=vent, density=950.)
    vj = b.add_joint_prismatic(pipette, vent, parent_xform=vent_origin,
        axis=(1., 0., 0.), limit_lower=0., limit_upper=VENT_TRAVEL_M,
        target_ke=1000., target_kd=2., friction=.05, label='pipette_vent_travel')
    b.add_articulation([free, pj, vj], label='pipette_with_manual_filler')
    lab.apparatus.bodies.update(pipette_plunger=piston, pipette_vent_button=vent)
    lab.apparatus.joints.update(pipette_plunger=pj, pipette_vent=vj)
    return {'pipette': pipette, 'plunger': piston, 'vent': vent,
            'plunger_joint': pj, 'vent_joint': vj}


class FillerForces:
    """Apply gas load through the actual piston joint, with casing reaction."""
    def __init__(self, world, pressure):
        self.world, self.pressure = world, pressure
        self.last = {}

    def apply(self, state, dt):
        from .solid_sampling import transformed
        w, p = self.world, self.pressure
        ids = w.lab.apparatus.bodies
        barrel, piston = ids['volumetric_pipette'], ids['pipette_plunger']
        poses, coms = state.body_q.numpy(), w.model.body_com.numpy()
        axis = transformed(np.array([[0., 0., 1.], [0., 0., 0.]]), poses[barrel])
        axis = axis[0]-axis[1]
        force = p.gauge_pa*PISTON_AREA_M2*axis
        # Both equal/opposite loads act at the physical piston face center.
        point = transformed(np.array([[0., 0., -.001]]), poses[piston])[0]
        loads = state.body_f.numpy().copy()
        for body, sign in ((piston, 1.), (barrel, -1.)):
            center = transformed(coms[body:body+1], poses[body])[0]
            loads[body, :3] += sign*force
            loads[body, 3:] += np.cross(point-center, sign*force)
        state.body_f.assign(loads)
        self.last = {'piston_force_n': float(p.gauge_pa*PISTON_AREA_M2)}

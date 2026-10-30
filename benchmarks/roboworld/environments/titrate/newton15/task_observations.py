"""Privileged task-only GPU inventories and rigid measurements.

The original endpoint measurements and spill/conservation definitions are
unchanged. The small extra reduction counts current cavities with precedence
pipette/valve/bottles/flask/beaker, never original parcel ownership.
"""

import math
import numpy as np
import warp as wp

from .task_readings import TaskReadings, cavity, reduce, finish, REST_SPEED_M_S
from .analytic_solids import field
from .task_evaluators import motor_rpm


@wp.kernel
def inventory(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], moles: wp.array2d[wp.float64],
              poses: wp.array[wp.transform], bodies: wp.array[int], profile: wp.array[wp.vec2],
              layout: wp.array[wp.vec2i], valve_pose: wp.transform, plug_body: int,
              plug_radius: float, plug_half: float, bore: float, parcel_ml: wp.float64,
              values: wp.array2d[wp.float64]):
    i = wp.tid()
    beaker, other, valve, tip = bool(False), bool(False), bool(False), bool(False)
    for vessel in range(bodies.shape[0]):
        pose = valve_pose
        if bodies[vessel] >= 0:
            pose = poses[bodies[vessel]]
        local = wp.transform_point(wp.transform_inverse(pose), q[i])
        row = layout[vessel]
        inside = cavity(local, profile, row[0], row[1])
        if vessel == 3:
            beaker = inside
        else:
            other = other or inside
        if vessel >= 5:
            valve = valve or inside
        if vessel == 6:
            tip = inside
    local = wp.transform_point(wp.transform_inverse(poses[plug_body]), q[i])
    static = wp.transform_point(wp.transform_inverse(valve_pose), q[i])
    in_bore = wp.length(wp.vec2(local[0], local[1])) < bore and wp.abs(local[2]) < plug_radius
    port = wp.length(wp.vec2(static[0], static[1])) < bore and wp.abs(static[2]) < .008
    if port:
        port = (field(static, 2, wp.vec4(plug_radius, .010, .010, bore))[3] >= 0.
                and field(local, 1, wp.vec4(plug_radius, 0., plug_half, bore))[3] >= 0.)
    valve = valve or in_bore or port
    for k in range(4):
        values[k, i] = wp.float64(0.)
    if beaker and not other and not valve:
        values[0, i] = moles[i, 2]
        values[1, i] = moles[i, 3]
    if valve:
        values[2, i] = parcel_ml
    if wp.length(qd[i]) > REST_SPEED_M_S and (tip or (not other and not beaker and not valve)):
        values[3, i] = moles[i, 2]


def tilt(pose):
    x, y = pose[3:5]
    return math.acos(float(np.clip(1-2*(x*x+y*y), -1., 1.)))


def rigid_measurements(world, setup):
    from .solid_sampling import inverse_transformed
    poses, velocities = world.state.body_q.numpy(), world.state.body_qd.numpy()
    names = world.lab.apparatus.bodies
    flask, bottle = poses[names['flask']], poses[names['acid_bottle']]
    local = inverse_transformed(flask[None, :3], poses[names['stirrer']])[0]
    # Flask bottom disk radius is 25 mm. The entire bottom must be on the plate.
    plate_error = max(abs(float(local[2])-.105), max(0., abs(float(local[0]))-.025),
                      max(0., abs(float(local[1]))-.025))
    result = dict(valve_rad=max(0., world.joint('stopcock')[0]),
                  motor_rpm=motor_rpm(world.joint('stirrer_knob')[0]),
                  bar_rpm=abs(float(velocities[names['stir_bar'], 5]))*60/(2*math.pi),
                  flask_distance_m=math.dist(flask[:2], world.lab.config.tip_position[:2]),
                  flask_start_distance_m=math.dist(flask[:3], setup['flask_start']),
                  flask_tilt_rad=tilt(flask), flask_plate_error_m=plate_error,
                  beaker_tilt_rad=tilt(poses[names['rinse_beaker']]), bottle_tilt_rad=tilt(bottle))
    # Resting anywhere on the bench is accepted, including a different return spot.
    c = world.lab.config
    result['bottle_bench_error_m'] = max(abs(float(bottle[2])-c.bench_height),
        max(0., abs(float(bottle[0]))+.029-c.bench_width/2),
        max(0., abs(float(bottle[1]))+.029-c.bench_depth/2))
    for prefix, name in (('flask', 'flask'), ('bottle', 'acid_bottle')):
        v = velocities[names[name]]
        result[prefix+'_speed_m_s'] = float(np.linalg.norm(v[:3]))
        result[prefix+'_spin_rad_s'] = float(np.linalg.norm(v[3:]))
    # Native MuJoCo contact list from the just-completed rigid step. Contact
    # names are mapped through bodies, not the policy's choice of finger/hand.
    mj, data = world.solver.mj_model, world.solver.mj_data
    touched = set()
    for contact in data.contact:
        if contact.dist > .0001:
            continue
        labels = [mj.body(int(mj.geom_bodyid[int(g)])).name for g in contact.geom]
        if any('panda' in label.lower() for label in labels):
            touched.update(labels)
    for key, bodies in (('flask_robot_contact', ('flask',)),
                        ('bottle_robot_contact', ('acid_bottle',)),
                        ('stirrer_robot_contact', ('stirrer', 'stirrer_knob'))):
        result[key] = any(label == b or label.endswith('/'+b) for label in touched for b in bodies)
    return result


class CatalogueReadings:
    def __init__(self, liquid, setup):
        self.original = TaskReadings(liquid)
        self.liquid, self.setup = liquid, setup
        n, device = liquid.water.model.particle_count, liquid.water.config.device
        self.values = wp.empty((4, n), dtype=wp.float64, device=device)
        self.partial = wp.empty((4, (n+255)//256), dtype=wp.float64, device=device)
        self.total = wp.empty(4, dtype=wp.float64, device=device)

    def read(self):
        result = self.original.read()
        o, liquid = self.original, self.liquid
        w, world = liquid.water, liquid.world
        c, names = world.lab.config, world.lab.apparatus.bodies
        wp.launch(inventory, dim=w.model.particle_count, inputs=[w.state.particle_q, w.state.particle_qd,
            liquid.chemistry.moles, world.state.body_q, o.bodies, o.profile, o.layout, o.valve_pose,
            names['stopcock'], c.valve_plug_radius, c.valve_plug_half_length, c.valve_bore_radius,
            wp.float64(w.volume_per_particle_m3*1.e6)], outputs=[self.values], device=w.config.device)
        wp.launch(reduce, dim=(4, self.partial.shape[1], 256), block_dim=256,
                  inputs=[self.values], outputs=[self.partial], device=w.config.device)
        wp.launch(finish, dim=4, inputs=[self.partial], outputs=[self.total], device=w.config.device)
        totals = self.total.numpy()
        result.update(beaker_base_ml=float(totals[0]*1000/liquid.reagents.base_molar),
                      beaker_acid_ml=float(totals[1]*1000/liquid.reagents.acid_molar),
                      burette_volume_ml=float(totals[2]),
                      delivery_pending_ml=float(totals[3]*1000/liquid.reagents.base_molar))
        result.update(rigid_measurements(world, self.setup))
        return result

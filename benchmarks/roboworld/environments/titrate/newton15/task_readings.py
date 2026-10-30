"""GPU read-only task measurements at every physical step, not client polls.

* base_in_flask_ml / acid_in_flask_mol: sodium and chloride in the current
  flask cavity, excluding an inserted pipette's lumen and the burette's own
  column, tip, bore and ports (a tip inside the flask neck is not delivered).
* pink_fraction: the share of the flask's indicator held in parcels at or above
  PINK_ONSET_PH, the textbook onset of phenolphthalein colour ("colorless below
  pH 8.2, pink 8.2-10.0"). Weighting by indicator means titrant
  that carries only a trace of indicator does not count as coloured solution.
* Liquid outside every vessel cavity, the bore and the ports is either
  arriving or stray. Arriving: moving faster than REST_SPEED_M_S and inside the
  vertical column above an open vessel's mouth disc (flask, bottles, rinse
  beaker, in their current poses), i.e. falling or poured into that vessel.
  Stray: everything else, moving or at rest, except a drop at rest hanging from
  the burette tip (within DELIVERY_PATH_RADIUS_M of the burette axis above the
  flask's starting mouth height).
* spill_ml: stray volume. Liquid thrown or knocked about is spill while still
  in the air; a stream into any vessel, from any tool, is not.
* arriving_ml / stray_moving_ml: diagnostics of the in-flight state.
* indicator_retained: flask indicator over the initial indicator. Only the
  flask's starting solution carries indicator, a passive tracer, so this is the
  share of the original solution (and aliquot) still in the flask.
* loose_base_ml / loose_acid_mol: sodium and chloride that have left the
  burette column, bottles, beaker and pipette and can still reach the flask:
  the burette tip, bore and ports, and liquid arriving at the flask.
* pending_base_ml: sodium on its way into the flask: moving liquid in the
  burette tip below the stopcock, and liquid arriving at the flask.
"""

import numpy as np
import warp as wp
from .analytic_solids import field
from .chemistry import KW

FIELDS = 16
PINK_ONSET_PH = 8.2
DELIVERY_PATH_RADIUS_M = .004
REST_SPEED_M_S = .05


@wp.func
def cavity(local: wp.vec3, profile: wp.array[wp.vec2], first: int, count: int):
    z = local[2]
    inside = bool(False)
    if z >= profile[first][1] and z <= profile[first+count-1][1]:
        width = 2.*wp.pi/96.
        theta = (wp.floor(wp.atan2(local[1], local[0])/width)+.5)*width
        radial = (local[0]*wp.cos(theta)+local[1]*wp.sin(theta))/wp.cos(wp.pi/96.)
        for k in range(first, first+count-1):
            a, b = profile[k], profile[k+1]
            if b[1] > a[1] and a[1] <= z and z <= b[1]:
                radius = a[0]+(b[0]-a[0])*(z-a[1])/(b[1]-a[1])
                inside = inside or radial < radius
    return inside


@wp.kernel
def measure(q: wp.array[wp.vec3], moles: wp.array2d[wp.float64], reacted: wp.array[wp.float64],
            poses: wp.array[wp.transform], bodies: wp.array[int], profile: wp.array[wp.vec2],
            layout: wp.array[wp.vec2i], valve_pose: wp.transform, plug_body: int,
            plug_radius: float, plug_half: float, bore: float, tip: wp.vec3,
            path_radius: float, path_floor: float, onset_ph: wp.float64, qd: wp.array[wp.vec3],
            rest_speed: float, parcel_ml: wp.float64, volume_l: wp.float64, values: wp.array2d[wp.float64]):
    i = wp.tid()
    in_flask, contained, in_tip = bool(False), bool(False), bool(False)
    for vessel in range(bodies.shape[0]):
        pose = valve_pose
        if bodies[vessel] >= 0:
            pose = poses[bodies[vessel]]
        local = wp.transform_point(wp.transform_inverse(pose), q[i])
        row = layout[vessel]
        inside = cavity(local, profile, row[0], row[1])
        contained = contained or inside
        if vessel == 0:
            in_flask = inside
        elif vessel >= 4 and inside:
            # Liquid still inside an inserted pipette, or in a burette tip
            # inside the flask neck, is not delivered base.
            in_flask = False
        if vessel == 6 and inside:
            in_tip = True
    local = wp.transform_point(wp.transform_inverse(poses[plug_body]), q[i])
    static = wp.transform_point(wp.transform_inverse(valve_pose), q[i])
    in_bore = wp.length(wp.vec2(local[0], local[1])) < bore and wp.abs(local[2]) < plug_radius
    port = wp.length(wp.vec2(static[0], static[1])) < bore and wp.abs(static[2]) < .008
    if port:
        port = (field(static, 2, wp.vec4(plug_radius, .010, .010, bore))[3] >= 0.
                and field(local, 1, wp.vec4(plug_radius, 0., plug_half, bore))[3] >= 0.)
    contained = contained or in_bore or port
    if in_bore or port:
        in_flask = False
    for k in range(FIELDS):
        values[k, i] = wp.float64(0.)
    h, oh = moles[i, 0], moles[i, 1]
    marker = moles[i, 4]
    if in_flask:
        if marker > wp.float64(0.):
            net = (h-oh)/volume_l
            root = wp.sqrt(net*net+wp.float64(4.*KW))
            free_h = (net+root)*wp.float64(.5)
            if net < wp.float64(0.):
                free_h = wp.float64(2.*KW)/(root-net)
            ph = -wp.log(free_h)/wp.log(wp.float64(10.))
            values[0, i] = marker
            if ph >= onset_ph:
                values[1, i] = marker
        values[2, i] = moles[i, 2]
        values[3, i] = moles[i, 3]
    dx, dy = q[i][0]-tip[0], q[i][1]-tip[1]
    on_path = dx*dx+dy*dy < path_radius*path_radius and q[i][2] > path_floor
    moving = wp.length(qd[i]) > rest_speed
    arriving, to_flask = bool(False), bool(False)
    if not contained and moving:
        for vessel in range(4):
            row = layout[vessel]
            top = profile[row[0]+row[1]-1]
            mouth = wp.transform_point(poses[bodies[vessel]], wp.vec3(0., 0., top[1]))
            if wp.length(wp.vec2(q[i][0]-mouth[0], q[i][1]-mouth[1])) < top[0] and q[i][2] > mouth[2]:
                arriving = True
                if vessel == 0:
                    to_flask = True
    if not contained and not arriving and not (on_path and not moving):
        values[4, i] = parcel_ml
        if moving:
            values[15, i] = parcel_ml
    if arriving:
        values[14, i] = parcel_ml
    lumen = not in_flask and (in_tip or in_bore or port)
    if lumen or to_flask:
        values[11, i] = moles[i, 2]
        values[12, i] = moles[i, 3]
    if (in_tip and not in_flask and moving) or to_flask:
        values[13, i] = moles[i, 2]
    values[5, i] = moles[i, 2]
    values[6, i] = moles[i, 3]
    values[7, i] = h+reacted[i]
    values[8, i] = oh+reacted[i]
    values[9, i] = marker
    for k in range(5):
        if not wp.isfinite(moles[i, k]) or moles[i, k] < wp.float64(0.):
            values[10, i] = wp.float64(1.)
    if not wp.isfinite(q[i][0]) or not wp.isfinite(q[i][1]) or not wp.isfinite(q[i][2]):
        values[10, i] = wp.float64(1.)


@wp.kernel
def reduce(values: wp.array2d[wp.float64], partial: wp.array2d[wp.float64]):
    field_id, block, lane = wp.tid()
    tile = wp.tile_load(values, shape=(1, 256), offset=(field_id, block*256))
    total = wp.tile_sum(tile)
    if lane == 0:
        partial[field_id, block] = total[0]


@wp.kernel
def finish(partial: wp.array2d[wp.float64], total: wp.array[wp.float64]):
    k = wp.tid()
    value = wp.float64(0.)
    for j in range(partial.shape[1]):
        value += partial[k, j]
    total[k] = value


class TaskReadings:
    def __init__(self, liquid):
        from .liquid_charge import FLASK_CAVITY, BOTTLE_CAVITY
        from .pipette_geometry import profile as pipette
        self.liquid = liquid
        w, world = liquid.water, liquid.world
        c, names = world.lab.config, world.lab.apparatus.bodies
        outline = np.asarray(pipette())
        profiles = [FLASK_CAVITY, BOTTLE_CAVITY, BOTTLE_CAVITY,
            [(.0285, .002), (.0285, .082)], outline[len(outline)//2:][::-1],
            [(c.valve_bore_radius, .008), (c.valve_bore_radius, .014),
                (c.burette_inner_radius, .040), (c.burette_inner_radius, c.burette_height-.055)],
            [(c.tip_inner_radius, -.055), (c.valve_bore_radius, -.013), (c.valve_bore_radius, -.008)]]
        bodies = [names[k] for k in ('flask', 'acid_bottle', 'base_bottle', 'rinse_beaker', 'volumetric_pipette')]+[-1, -1]
        starts = np.cumsum([0]+[len(p) for p in profiles[:-1]])
        device = w.config.device
        self.bodies = wp.array(bodies, dtype=int, device=device)
        self.profile = wp.array(np.concatenate(profiles), dtype=wp.vec2, device=device)
        self.layout = wp.array(list(zip(starts, map(len, profiles))), dtype=wp.vec2i, device=device)
        n = w.model.particle_count
        self.values = wp.empty((FIELDS, n), dtype=wp.float64, device=device)
        self.partial = wp.empty((FIELDS, (n+255)//256), dtype=wp.float64, device=device)
        self.total = wp.empty(FIELDS, dtype=wp.float64, device=device)
        self.valve_pose = wp.transform(c.valve_position, wp.quat_identity())
        self.tip = wp.vec3(*c.tip_position)
        self.path_floor = c.tip_position[2]-c.tip_clearance

    def read(self):
        liquid = self.liquid
        w, world, chem = liquid.water, liquid.world, liquid.chemistry
        c, names = world.lab.config, world.lab.apparatus.bodies
        wp.launch(measure, dim=w.model.particle_count, inputs=[w.state.particle_q, chem.moles,
            chem.reacted, world.state.body_q, self.bodies, self.profile, self.layout,
            self.valve_pose, names['stopcock'], c.valve_plug_radius, c.valve_plug_half_length,
            c.valve_bore_radius, self.tip, DELIVERY_PATH_RADIUS_M, self.path_floor,
            wp.float64(PINK_ONSET_PH), w.state.particle_qd, REST_SPEED_M_S,
            w.volume_per_particle_m3*1.e6, chem.volume_l], outputs=[self.values], device=w.config.device)
        wp.launch(reduce, dim=(FIELDS, self.partial.shape[1], 256), block_dim=256,
            inputs=[self.values], outputs=[self.partial], device=w.config.device)
        wp.launch(finish, dim=FIELDS, inputs=[self.partial], outputs=[self.total], device=w.config.device)
        total = self.total.numpy()
        initial = liquid.initial_totals
        expected = np.array([initial[k] for k in ('Na', 'Cl', 'H', 'OH', 'indicator')])
        conserved = np.all(np.abs(total[5:10]-expected) <= 1.e-10*np.maximum(expected, 1.e-15))
        return dict(pink_fraction=float(total[1]/total[0]) if total[0] else 0.,
            base_in_flask_ml=float(total[2]/liquid.reagents.base_molar*1000.),
            acid_in_flask_mol=float(total[3]), spill_ml=float(total[4]),
            invariants_ok=bool(conserved and total[10] == 0. and np.isfinite(total).all()),
            indicator_retained=float(total[0]/initial['indicator']) if initial['indicator'] > 0 else 0.,
            loose_base_ml=float(total[11]/liquid.reagents.base_molar*1000.), loose_acid_mol=float(total[12]),
            pending_base_ml=float(total[13]/liquid.reagents.base_molar*1000.), arriving_ml=float(total[14]),
            stray_moving_ml=float(total[15]))

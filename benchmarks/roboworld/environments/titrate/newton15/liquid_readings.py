"""Privileged, read-only liquid inventory measurements in moving glassware.

No initial parcel identity is used to decide the current vessel. Classification
uses current world positions and body-local cavity geometry. These measurements
are teacher/evaluator information, not deployable robot observations.
"""

import numpy as np
from .liquid_charge import FLASK_CAVITY, BOTTLE_CAVITY
from .solid_sampling import inverse_transformed
from .chemistry import equilibrium_ph, SPECIES


def analytic_intrusions(points, poses, builder, volume_per_particle_m3):
    """Read-only membership of the authored analytic valve solids.

    Report raw parcel-centre signs without clipping, deleting material, or
    labelling the signed CSG bound as an exact intersection-distance metric.
    """
    from .bored_geometry import plug_sdf, socket_sdf
    result = {}
    for shape, (kind, dimensions) in getattr(builder, 'titration_analytic_solids', {}).items():
        body = builder.shape_body[shape]
        local = inverse_transformed(points, poses[body]) if body >= 0 else points
        local = inverse_transformed(local, builder.shape_transform[shape])
        radius, outer, half_length, bore = dimensions
        value = (plug_sdf(local, radius, half_length, bore) if kind == 1 else
                 socket_sdf(local, radius, outer, half_length, bore))
        mask = value < 0.
        result[builder.shape_label[shape]] = {
            'parcel_centres_inside': int(mask.sum()),
            'material_volume_with_centres_inside_ml': float(mask.sum()*volume_per_particle_m3*1e6),
            'maximum_negative_csg_bound_m': float(np.maximum(-value, 0.).max(initial=0.)),
            'method': 'Analytic solid sign in current shape frame; no tolerance-based omission'}
    return result


def in_cavity(points, pose, profile, *, sectors=96):
    local = inverse_transformed(points, pose)
    p = np.asarray(profile, float)
    z = local[:, 2]
    width = 2*np.pi/sectors
    theta = np.arctan2(local[:, 1], local[:, 0])
    center = (np.floor(theta/width)+.5)*width
    # Match the authored faceted cavity, not a larger smooth cylinder.
    radial = (local[:, 0]*np.cos(center)+local[:, 1]*np.sin(center))/np.cos(np.pi/sectors)
    return (z >= p[0, 1]) & (z <= p[-1, 1]) & (radial < np.interp(z, p[:, 1], p[:, 0]))


def valve_inventory(points, plug_pose, config):
    """Non-overlapping current regions, including socket-to-tube junctions."""
    from .bored_geometry import plug_sdf, socket_sdf
    c = config
    column = [(c.valve_bore_radius, .008), (c.valve_bore_radius, .014),
        (c.burette_inner_radius, .040), (c.burette_inner_radius, c.burette_height-.055)]
    tip = [(c.tip_inner_radius, -.055), (c.valve_bore_radius, -.013), (c.valve_bore_radius, -.008)]
    static_pose = (*c.valve_position, 0., 0., 0., 1.)
    local = inverse_transformed(points, plug_pose)
    static = inverse_transformed(points, static_pose)
    masks = {'burette_column': in_cavity(points, static_pose, column),
        'burette_tip': in_cavity(points, static_pose, tip),
        'stopcock_bore': (np.linalg.norm(local[:, :2], axis=1) < c.valve_bore_radius)
            & (np.abs(local[:, 2]) < c.valve_plug_radius)}
    # The 7..8 mm socket ports are real fluid space, not an uncontained spill.
    # At rotated angles, exclude the plug's actual solid and already assigned
    # rotating-bore points. There is no nearest-vessel/identity assignment.
    port = (np.linalg.norm(static[:, :2], axis=1) < c.valve_bore_radius) & (np.abs(static[:, 2]) < .008)
    port &= socket_sdf(static, c.valve_plug_radius, .010, .010, c.valve_bore_radius) >= 0.
    port &= plug_sdf(local, c.valve_plug_radius, c.valve_plug_half_length, c.valve_bore_radius) >= 0.
    port &= ~np.logical_or.reduce(list(masks.values()))
    masks['socket_ports'] = port
    return masks


def readings(liquid, *, positions=None, moles=None):
    w, world = liquid.water, liquid.world
    points = w.state.particle_q.numpy() if positions is None else np.asarray(positions)
    n = liquid.chemistry.moles.numpy() if moles is None else np.asarray(moles)
    if points.shape != (w.model.particle_count, 3) or n.shape != (len(points), 5) or not np.isfinite(points).all() or not np.isfinite(n).all():
        raise ValueError('Finite complete material-point state required for readings')
    poses, names = world.state.body_q.numpy(), world.lab.apparatus.bodies
    masks = {}
    for name, profile in (('flask', FLASK_CAVITY), ('acid_bottle', BOTTLE_CAVITY),
            ('base_bottle', BOTTLE_CAVITY), ('rinse_beaker', [(0.0285, .002), (.0285, .082)])):
        masks[name] = in_cavity(points, poses[names[name]], profile)
    from .pipette_geometry import profile as pipette_profile
    outline = np.asarray(pipette_profile())
    masks['volumetric_pipette'] = in_cavity(points, poses[names['volumetric_pipette']], outline[len(outline)//2:][::-1])
    c = world.lab.config
    valve = valve_inventory(points, poses[names['stopcock']], c)
    # An immersed pipette's lumen, a burette tip inside a vessel neck, or a
    # bottle or flask neck lowered into the rinse beaker is geometrically inside
    # the outer vessel's cavity. Its contents belong to the inner vessel, never
    # twice to both: precedence pipette, burette, bottles, flask, rinse beaker.
    masks.update(valve)
    precedence = ('volumetric_pipette', *valve, 'acid_bottle', 'base_bottle', 'flask', 'rinse_beaker')
    claimed = np.zeros(len(points), bool)
    for name in precedence:
        masks[name] = masks[name] & ~claimed
        claimed |= masks[name]
    occupied = np.zeros(len(points), bool)
    inventory = {}
    vml = w.volume_per_particle_m3*1e6
    for name, mask in masks.items():
        if np.any(mask & occupied):
            raise ValueError(f'Overlapping vessel inventory regions: {name}')
        occupied |= mask
        total = n[mask].sum(axis=0)
        ph = equilibrium_ph((n[mask, 0]-n[mask, 1])/liquid.chemistry.volume_l)
        inventory[name] = {'parcel_count': int(mask.sum()), 'volume_ml': float(mask.sum()*vml),
            'species_mol': dict(zip(SPECIES, total.tolist())),
            'ph_quantiles_05_50_95': np.percentile(ph, [5, 50, 95]).tolist() if len(ph) else None,
            'world_z_min_max_m': [float(points[mask, 2].min()), float(points[mask, 2].max())] if mask.any() else None}
    outside = ~occupied
    # Do not mislabel a falling drop above the bench as deposited spill.
    low_outside = outside & (points[:, 2] <= c.bench_height+w.config.spacing_m)
    return {'inventory': inventory, 'uncontained_volume_ml': float(outside.sum()*vml),
        'uncontained_at_or_below_bench_ml': float(low_outside.sum()*vml),
        'material_volume_ml': float(len(points)*vml),
        'volume_discretization_ml': vml,
        'analytic_solid_intrusions': analytic_intrusions(points, poses, world.lab.builder, w.volume_per_particle_m3),
        'method': 'Current body-local faceted cavity membership; parcel-centre volume quadrature. No particle moves or implicit level fitting.'}

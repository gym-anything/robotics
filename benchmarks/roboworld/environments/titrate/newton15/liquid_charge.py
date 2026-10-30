"""Deterministic initial reagent inventory in physical cavities.

This is episode initialization, never a motion-time emitter. Material volume
is the requested integer number of SPH parcels, with no density calibration.
"""

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Reagents:
    aliquot_ml: float = 10.
    dilution_ml: float = 15.
    burette_ml: float = 20.
    acid_bottle_ml: float = 20.
    base_bottle_ml: float = 20.
    acid_molar: float = .100
    base_molar: float = .100
    indicator_molar: float = 6.283e-5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Nonnegative finite reagent value required: {name}")
        if self.acid_molar <= 0 or self.base_molar <= 0:
            raise ValueError("Positive specified analytical concentrations required")


FLASK_CAVITY = [(0., .002), (.025, .002), (.029, .003), (.032, .007),
    (.032, .015), (.0275, .040), (.0205, .067), (.0135, .086),
    (.0125, .091), (.0125, .114)]
BOTTLE_CAVITY = [(.027, .003), (.027, .094), (.015, .115), (.012, .118), (.012, .140)]


def parcel_count(volume_ml, spacing):
    count = volume_ml*1.e-6/spacing**3
    if not np.isclose(count, round(count), rtol=0., atol=1.e-6):
        raise ValueError(f"{volume_ml} mL is not an integer parcel volume at {spacing} m; no rounding")
    return int(round(count))


def fill_cavity(profile, volume_ml, spacing, *, exclude=None, wall_clearance_fraction=.5):
    """Fill bottom-up on a stable centred lattice, respecting wall clearance."""
    count = parcel_count(volume_ml, spacing)
    if count == 0:
        return np.empty((0, 3), np.float32)
    p = np.asarray(profile, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.any(np.diff(p[:, 1]) < 0):
        raise ValueError("An ordered radius/height cavity profile is required")
    radius = float(p[:, 0].max())
    xy = (np.arange(-math.ceil(radius/spacing), math.ceil(radius/spacing))+.5)*spacing
    xx, yy = np.meshgrid(xy, xy, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.isfinite(wall_clearance_fraction) or not 0 <= wall_clearance_fraction <= .5:
        raise ValueError('Initial wall clearance must lie within 0..0.5 parcel spacings')
    width = 2*np.pi/96
    middle = (np.floor(np.arctan2(xy[:, 1], xy[:, 0])/width)+.5)*width
    facet_distance = xy[:, 0]*np.cos(middle)+xy[:, 1]*np.sin(middle)
    chunks, used = [], 0
    first_z = math.ceil((p[0, 1]+spacing*wall_clearance_fraction)/spacing-.5)
    last_z = math.ceil((p[-1, 1]-spacing*wall_clearance_fraction)/spacing-.5)
    for z in (np.arange(first_z, last_z)+.5)*spacing:
        # Signed radial clearance on sloping walls, rather than merely
        # subtracting a radius horizontally from a steep shoulder.
        k = min(np.searchsorted(p[:, 1], z, side="right")-1, len(p)-2)
        slope = (p[k+1, 0]-p[k, 0])/(p[k+1, 1]-p[k, 1])
        # Use the actual 96-sided cavity, not the circumcircle. The selected
        # numerical centre clearance also applies during initialization; a
        # zero-clearance configuration must not retain a hidden half-dx gap.
        cosine = math.cos(width/2)
        available = np.interp(z, p[:, 1], p[:, 0])*cosine-spacing*wall_clearance_fraction*math.sqrt(1+(slope*cosine)**2)
        selected = facet_distance < available
        layer = np.column_stack((xy[selected], np.full(np.count_nonzero(selected), z)))
        if exclude is not None:
            layer = layer[~exclude(layer)]
        # Preserve reflection symmetry at the partially filled top layer by
        # choosing nearest-to-centre parcels first; this is not an action.
        layer = layer[np.argsort(np.linalg.norm(layer[:, :2], axis=1), kind="stable")]
        take = min(len(layer), count-used)
        if take:
            chunks.append(layer[:take])
            used += take
        if used == count:
            return np.concatenate(chunks).astype(np.float32)
    raise ValueError(f"Requested {volume_ml} mL exceeds the represented cavity's initial charge capacity")


def laboratory_charge(lab, state, spacing, reagents, *, wall_clearance_fraction=.5):
    from .solid_sampling import transformed, inverse_transformed
    bodies, q = lab.apparatus.bodies, state.body_q.numpy()
    bar = inverse_transformed(transformed(np.zeros((1, 3)), q[bodies['stir_bar']]), q[bodies['flask']])[0]

    def excludes_bar(points):
        p = points-bar
        p[:, 0] -= np.clip(p[:, 0], -.007, .007)
        return np.linalg.norm(p, axis=1) < .003+spacing*wall_clearance_fraction

    points, inventories, start = [], {}, 0
    config = lab.config
    column = [(config.valve_bore_radius, .008), (config.valve_bore_radius, .014),
              (config.burette_inner_radius, .040),
              (config.burette_inner_radius, config.burette_height-.055)]
    definitions = (
        ("flask", FLASK_CAVITY, reagents.aliquot_ml+reagents.dilution_ml, excludes_bar),
        ("burette", column, reagents.burette_ml, None),
        ("acid_bottle", BOTTLE_CAVITY, reagents.acid_bottle_ml, None),
        ("base_bottle", BOTTLE_CAVITY, reagents.base_bottle_ml, None))
    for name, profile, volume, exclude in definitions:
        local = fill_cavity(profile, volume, spacing, exclude=exclude,
            wall_clearance_fraction=wall_clearance_fraction)
        pose = q[bodies[name]] if name != "burette" else (*config.valve_position, 0., 0., 0., 1.)
        points.append(transformed(local, pose))
        inventories[name] = [start, start+len(local)]
        start += len(local)
    if not start:
        raise ValueError("Use the dry session explicitly for an empty laboratory")
    return np.concatenate(points).astype(np.float32), inventories

"""Task-only initial arrangements, outside both checkpoint physics source lists.

No setter here is exposed as an action. The factory applies these once before
liquid initialization and before recording the tick-zero checkpoint.
"""

from dataclasses import asdict, replace
import math
import random

from .task_catalogue import task_name


def preset(name, config, reagents, spacing_m=None):
    task_name(name)
    poses, base0, dial = {}, 0., 0.
    rng = random.Random(config.seed)
    if name == 'centre-flask':
        # Keep the entire 67 mm flask footprint on the 100 mm plate.
        radius, angle = rng.uniform(.013, .0165), rng.uniform(0., 2*math.pi)
        dx, dy = radius*math.cos(angle), radius*math.sin(angle)
        x, y, z = config.flask_position
        poses['flask'] = (x+dx, y+dy, z)
        poses['stir_bar'] = (x+dx, y+dy, z+.005)
    if name in ('prime-burette', 'drain-burette'):
        poses['rinse_beaker'] = (.10, -.27, config.bench_height+.0002)
    if name == 'decant-acid':
        poses['acid_bottle'] = (-.05, -.28, config.bench_height+.0002)
        poses['rinse_beaker'] = (.12, -.28, config.bench_height+.0002)
    actual = reagents
    if name == 'run-in' and reagents.aliquot_ml*reagents.acid_molar/reagents.base_molar <= 1.5:
        raise ValueError('Run-in needs an equivalence volume above 1.5 mL')
    if name == 'finish-titration':
        # All offsets are exact multiples of the reference parcel volume.
        delta = rng.choice((.3125, .5, .75, 1.))
        base0 = reagents.aliquot_ml*reagents.acid_molar/reagents.base_molar-delta
        if not 0 < base0 < reagents.burette_ml:
            raise ValueError('Paused titration requires enough base for a positive precharge')
        actual = replace(reagents, dilution_ml=reagents.dilution_ml+base0,
                         burette_ml=reagents.burette_ml-base0)
        dial = (500.-100.)/1400.*(5*math.pi/3)
    if spacing_m is not None:
        from .liquid_charge import parcel_count
        for key in ('aliquot_ml', 'dilution_ml', 'burette_ml', 'acid_bottle_ml', 'base_bottle_ml'):
            parcel_count(getattr(actual, key), spacing_m)
    result = dict(task=name, positions=poses, base0_ml=base0, dial_rad=dial,
                  reagents=asdict(actual), nominal_reagents=asdict(reagents),
                  flask_target=config.flask_position,
                  flask_start=poses.get('flask', config.flask_position))
    result['reach_m'] = check_placements(config, result)
    return result


def check_placements(config, setup):
    """Conservative shoulder sphere and authored bench-envelope clearance.

    This is a necessary geometric check, not an IK or grasp demonstration.
    Include each whole moved vessel's lower and upper extents, the rotary
    controls, and the destination under the tip, not just each object's origin.
    """
    shoulder = (-.43, -.04, config.bench_height+.025+.333)
    points = {'flask_destination': config.flask_position,
              'flask_mouth_destination': (*config.flask_position[:2], config.flask_position[2]+.114),
              'valve_handle': (config.flask_x, config.flask_y-.020, config.valve_position[2]),
              'dial': (config.flask_x, config.flask_y-.195, config.bench_height+.0382)}
    dimensions = {'flask': (.0335, .114), 'stir_bar': (.010, .003),
                  'acid_bottle': (.029, .140), 'rinse_beaker': (.030, .082)}
    for name, (x, y, z) in setup['positions'].items():
        r, height = dimensions[name]
        if abs(x)+r > config.bench_width/2 or abs(y)+r > config.bench_depth/2:
            raise ValueError(f'{name} is outside the bench')
        for dx in (-r, r):
            for dy in (-r, r):
                for dz in (0., height):
                    points[f'{name}/{dx}/{dy}/{dz}'] = (x+dx, y+dy, z+dz)
        if name in ('acid_bottle', 'rinse_beaker'):
            # Stirrer housing including front panel, stand base, pipette rack,
            # and Panda mount; all bounds read from laboratory.py.
            obstacles = ((config.flask_x-.075, config.flask_x+.075,
                          config.flask_y-.203, config.flask_y+.075),
                         (config.flask_x-.295, config.flask_x-.105,
                          config.flask_y+.05, config.flask_y+.23),
                         (.275, .765, -.285, -.215), (-.525, -.335, -.135, .055))
            for x0, x1, y0, y1 in obstacles:
                if x+r > x0 and x-r < x1 and y+r > y0 and y-r < y1:
                    raise ValueError(f'{name} overlaps an authored fixture envelope')
    distances = {key: math.dist(point, shoulder) for key, point in points.items()}
    if max(distances.values()) >= .855:
        raise ValueError('A task placement is outside the Panda shoulder reach sphere')
    return distances

"""Task-only initial arrangements, outside both checkpoint physics source lists.

No setter here is exposed as an action. The factory applies these once before
liquid initialization and before recording the tick-zero checkpoint. A task's own
arrangement is arrange(s) in its setup.py (tasks/<task>/); preset() builds the
Start it works on, then checks the volumes and placements every task shares.
"""

from dataclasses import asdict
import math
import random

from .task_catalogue import task_name, task_setup


class Start:
    """What a task's setup.py arranges its start on (preset() below calls its arrange(s), if it has one).

    name      the task
    config    the selected bench layout (lab_config.LabConfig)
    reagents  the selected nominal charge (liquid_charge.Reagents)
    rng       random.Random(config.seed), for the task's seeded draws
    poses     free body name -> start position (x, y, z), for the bodies the task moves
    actual    the charge the liquid starts with: the nominal one unless the task prepares another
    base0     base already run into the flask at the start, mL
    dial      the stirrer dial's start angle, rad"""

    def __init__(self, name, config, reagents, rng, poses, actual, base0, dial):
        self.name, self.config, self.reagents, self.rng = name, config, reagents, rng
        self.poses, self.actual, self.base0, self.dial = poses, actual, base0, dial


def preset(name, config, reagents, spacing_m=None):
    task_name(name)
    poses, base0, dial = {}, 0., 0.
    rng = random.Random(config.seed)
    actual = reagents
    arrange = getattr(task_setup(name), 'arrange', None)
    if arrange is not None:                 # the task's own start (tasks/<task>/setup.py)
        s = Start(name, config, reagents, rng, poses, actual, base0, dial)
        arrange(s)
        actual, base0, dial = s.actual, s.base0, s.dial
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

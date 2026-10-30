"""Finite, deterministic titration catalogue; resolution is never varied.

25 sensible bench positions x 11 aliquots x 4 burette starting charges.
ID zero is the reference 10+15 mL flask and 20 mL burette. All catalogue
combinations are specified, not asserted to have passed physical qualification.
"""

from dataclasses import asdict, replace
from .lab_config import LabConfig
from .liquid_charge import Reagents, parcel_count
from .task import description

XS = (.150, .120, .135, .165, .180)
YS = (.035, .005, .020, .050, .065)
ALIQUOTS = (10., 5., 6., 7., 8., 9., 11., 12., 13., 14., 15.)
BURETTE_ML = (20., 18., 22., 24.)
COUNT = len(XS)*len(YS)*len(ALIQUOTS)*len(BURETTE_ML)


def index(value):
    if isinstance(value, str) and value.startswith('titration-n15-'):
        suffix = value.removeprefix('titration-n15-')
        if len(suffix) != 4 or not suffix.isascii() or not suffix.isdecimal():
            raise ValueError('Use titration-n15-NNNN')
        value = int(suffix)
    if type(value) is not int or not 0 <= value < COUNT:
        raise ValueError(f'Variation ID must lie in 0..{COUNT-1}; no wrapping')
    return value


def resolve(value, *, base_config=None, base_reagents=None, spacing_m=None):
    identity = n = index(value)
    x, n = XS[n % len(XS)], n//len(XS)
    y, n = YS[n % len(YS)], n//len(YS)
    aliquot, n = ALIQUOTS[n % len(ALIQUOTS)], n//len(ALIQUOTS)
    burette = BURETTE_ML[n]
    config = replace(base_config or LabConfig(), seed=identity, flask_x=x, flask_y=y)
    reagents = replace(base_reagents or Reagents(), aliquot_ml=aliquot,
        dilution_ml=25.-aliquot, burette_ml=burette)
    if spacing_m is not None:
        for name in ('aliquot_ml', 'dilution_ml', 'burette_ml', 'acid_bottle_ml', 'base_bottle_ml'):
            parcel_count(getattr(reagents, name), spacing_m)
    return config, reagents


def contract(current=None):
    from .task_catalogue import TASKS, TASK_SET
    return dict(supported=True, count=COUNT, id_format='titration-n15-NNNN',
        integer_id_range=[0, COUNT-1], current=current,
        selectors=['variation_id', 'task'], constraints=['aliquot_ml', 'burette_ml'],
        tasks=list(TASKS), task_count=len(TASKS), task_set=list(TASK_SET),
        qualifications='Defined reproducible catalogue; not all combinations physically qualified',
        preview_mutates=False, start_replaces_episode=True)


def select(request, *, base_config=None, base_reagents=None, spacing_m=None):
    from .task_catalogue import task_name, description as task_description, contract as task_contract
    if not isinstance(request, dict) or set(request)-{'variation_id', 'constraints', 'task'}:
        raise ValueError('Choose variation_id and/or constraints; unknown fields are rejected')
    name = task_name(request.get('task', 'titration'))
    constraints = request.get('constraints', {})
    if not isinstance(constraints, dict) or set(constraints)-{'aliquot_ml', 'burette_ml'}:
        raise ValueError('Supported exact constraints: aliquot_ml, burette_ml')
    for key, value in constraints.items():
        allowed = ALIQUOTS if key == 'aliquot_ml' else BURETTE_ML
        if type(value) not in (int, float) or value not in allowed:
            raise ValueError(f'No catalogue variation has {key}={value!r}')
    candidates = [index(request['variation_id'])] if 'variation_id' in request else range(COUNT)
    for identity in candidates:
        cfg, r = resolve(identity, base_config=base_config, base_reagents=base_reagents, spacing_m=spacing_m)
        if all(getattr(r, key) == value for key, value in constraints.items()):
            result = dict(variation_id=f'titration-n15-{identity:04d}',
                config=asdict(cfg), reagents=asdict(r), task=description(r),
                reset_request={'options': {'variation_id': identity}})
            if name != 'titration':
                from .task_presets import preset
                result.update(task=task_description(name, r), task_name=name, task_contract=task_contract(name),
                              start=preset(name, cfg, r, spacing_m))
                result['reset_request']['options']['task'] = name
            elif 'task' in request:
                # An explicit request can switch a live new-task episode back.
                result['reset_request']['options']['task'] = name
            return result
    raise ValueError('Requested variation does not satisfy the exact constraints')

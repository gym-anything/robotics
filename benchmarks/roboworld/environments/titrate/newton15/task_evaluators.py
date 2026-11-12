"""Read-only task values from measurements at every physics tick.

The titration evaluator (task.py) is composed unchanged, including its validity gate and
endpoint persistence window. No policy actions or parcel identities are read. Each task's
own checks, reward terms and progress are verify() in its verifier.py (tasks/<task>/);
CatalogueEvaluator computes what the tasks share and calls it at every tick.
"""

from collections import deque
import copy
import json
import math

from .task import EndpointEvaluator, FAIL_SCORE_SCALE, READINGS
from .task_catalogue import contract, task_name, task_verifier


def clip(value):
    return max(0., min(1., value))


def grade(error, full, zero):
    return clip((zero-error)/(zero-full))


def motor_rpm(angle):
    fraction = clip(angle/(5*math.pi/3))
    return 0. if fraction < .01 else 100.+1400.*fraction


EXTRA = ('beaker_base_ml', 'beaker_acid_ml', 'burette_volume_ml', 'delivery_pending_ml',
         'valve_rad', 'motor_rpm', 'bar_rpm', 'flask_distance_m', 'flask_start_distance_m',
         'flask_tilt_rad', 'flask_plate_error_m', 'flask_speed_m_s', 'flask_spin_rad_s',
         'beaker_tilt_rad', 'bottle_tilt_rad', 'bottle_bench_error_m',
         'bottle_speed_m_s', 'bottle_spin_rad_s')
CONTACTS = ('flask_robot_contact', 'stirrer_robot_contact', 'bottle_robot_contact')


class CatalogueEvaluator:
    def __init__(self, name, dt_s, reagents, setup):
        self.name = task_name(name)
        if name == 'titration':
            raise ValueError('Use task.EndpointEvaluator for titration')
        self.endpoint = EndpointEvaluator(dt_s, reagents.aliquot_ml, reagents.acid_molar, reagents.base_molar)
        self.dt_s, self.reagents, self.setup = dt_s, reagents, json.loads(json.dumps(setup))
        self.closed_start = self.stir_start = None
        self.spin = deque()
        self.spin_bar_sum = self.spin_motor_sum = 0.
        self.last_tick, self.result, self.measurements = -1, None, None

    def _closed(self, m):
        return m['valve_rad'] >= 1.2 and m['delivery_pending_ml'] <= .002

    def _stir_safe(self, m):
        return (200 <= m['motor_rpm'] <= 800 and m['flask_start_distance_m'] <= .010
                and m['flask_tilt_rad'] < math.radians(5) and m['flask_plate_error_m'] <= .001
                and not m['stirrer_robot_contact'] and m['base_in_flask_ml'] <= .01
                and m['spill_ml'] <= .05 and self.reagents.aliquot_ml*(1-min(1., m['indicator_retained'])) <= .05)

    def update(self, tick, **m):
        if type(tick) is not int or tick != self.last_tick+1:
            raise ValueError('Task evaluator requires consecutive ticks, including zero')
        if set(m) != set(READINGS) | {'invariants_ok'} | set(EXTRA) | set(CONTACTS):
            raise ValueError('Incomplete task measurements')
        if any(type(m[k]) is not bool for k in CONTACTS):
            raise ValueError('Explicit contact measurements required')
        if any(isinstance(m[k], bool) or not math.isfinite(m[k]) or m[k] < 0 for k in EXTRA):
            raise ValueError('Finite nonnegative task measurements required')
        self.endpoint.update(tick, **{k: m[k] for k in (*READINGS, 'invariants_ok')})
        for field, good in (('closed_start', self._closed(m)), ('stir_start', self._stir_safe(m))):
            if not good:
                setattr(self, field, None)
            elif getattr(self, field) is None:
                setattr(self, field, tick)
        track = getattr(task_verifier(self.name), 'track', None)
        if track is not None:         # samples a task keeps every tick (start-stirring: the last 5 s of stirring)
            track(self, tick, m)
        self.last_tick, self.measurements = tick, copy.deepcopy(m)
        self.result = self._evaluate(tick, m)
        return copy.deepcopy(self.result)

    def _seconds(self, tick, start):
        return 0. if start is None else (tick-start)*self.dt_s

    def _evaluate(self, tick, m):
        e, r, name = self.endpoint.result, self.reagents, self.name
        closed_s = self._seconds(tick, self.closed_start)
        stir_s = self._seconds(tick, self.stir_start)
        k = clip((m['valve_rad']-.2)/1.)*clip(closed_s/2.)
        upright = grade(m['flask_tilt_rad'], math.radians(5), math.radians(30))
        plate = grade(m['flask_plate_error_m'], .001, .03)
        placed = (m['flask_distance_m'] <= .006 and m['flask_tilt_rad'] < math.radians(5)
                  and m['flask_plate_error_m'] <= .001 and m['flask_speed_m_s'] <= .01
                  and m['flask_spin_rad_s'] <= .1 and not m['flask_robot_contact'])
        checks = {key: e['checks'][key] for key in ('spill', 'acid_retained')}
        penalty = (max(0., m['spill_ml']-.05)+max(0., e['acid_loss_ml']-.05))/.6
        terms = dict(spill_aliquot_penalty=penalty, closed_grade=k)
        base = m['base_in_flask_ml']
        # The task's own checks, reward terms and progress (tasks/<task>/verifier.py).
        p, checks, terms, penalty = task_verifier(name).verify(
            self, m, e=e, r=r, base=base, closed_s=closed_s, stir_s=stir_s, k=k, upright=upright, plate=plate,
            placed=placed, checks=checks, terms=terms, penalty=penalty)
        progress = clip(p-penalty)
        valid = e['valid']
        success = valid and all(checks.values())
        terms.update(progress=progress, contract=contract(name),
                     score_rule='1 if success else 0.9 x progress; None if not valid')
        result = dict(tick=tick, task=name, contract=contract(name), valid=valid, success=success,
                      score=1. if success else FAIL_SCORE_SCALE*progress if valid else None,
                      progress=progress, checks=checks, reward=terms,
                      closed_seconds=closed_s, steady_seconds=stir_s,
                      stable_pink_seconds=e['stable_pink_seconds'], measurements=copy.deepcopy(m),
                      qualification='Task criterion only; no new physical qualification')
        if not valid:
            result['invalid_reason'] = e['invalid_reason']
        return result

    def snapshot(self):
        return dict(schema='titration.task-set.snapshot.v1', task=self.name, contract=contract(self.name),
                    setup=copy.deepcopy(self.setup), endpoint=self.endpoint.snapshot(),
                    last_tick=self.last_tick, closed_start=self.closed_start, stir_start=self.stir_start,
                    spin=list(self.spin), measurements=copy.deepcopy(self.measurements), result=copy.deepcopy(self.result))

    def _validated(self, state):
        current = self.snapshot()
        if set(state) != set(current) or any(state[k] != current[k] for k in ('schema', 'task', 'contract', 'setup')):
            raise ValueError('Task checkpoint identity mismatch')
        trial = CatalogueEvaluator(self.name, self.dt_s, self.reagents, self.setup)
        trial.endpoint.restore(state['endpoint'])
        tick = state['last_tick']
        if type(tick) is not int or tick != trial.endpoint.last_tick:
            raise ValueError('Task checkpoint clocks differ')
        for field in ('closed_start', 'stir_start'):
            start = state[field]
            if start is not None and (type(start) is not int or not 0 <= start <= tick):
                raise ValueError('Invalid task checkpoint window')
            setattr(trial, field, start)
        m = state['measurements']
        if not isinstance(m, dict) or set(m) != set(READINGS) | {'invariants_ok'} | set(EXTRA) | set(CONTACTS):
            raise ValueError('Invalid task checkpoint measurements')
        if any(type(m[k]) is not bool for k in (*CONTACTS, 'invariants_ok')) or any(
                isinstance(m[k], bool) or not math.isfinite(m[k]) or m[k] < 0 for k in (*READINGS, *EXTRA)):
            raise ValueError('Invalid task checkpoint measurement values')
        er = trial.endpoint.result
        if any(m[k] != er[k] for k in READINGS) or m['invariants_ok'] != er['valid']:
            raise ValueError('Endpoint/task checkpoint measurements differ')
        if (trial.closed_start is not None) != trial._closed(m) or (trial.stir_start is not None) != trial._stir_safe(m):
            raise ValueError('Checkpoint windows contradict measured state')
        spin = state['spin']
        window = getattr(task_verifier(self.name), 'spin_ticks', None)    # the ticks a task's samples cover
        expected = window(self, tick) if window is not None else []
        if not isinstance(spin, list) or [row[0] for row in spin] != expected or any(
                len(row) != 3 or any(not math.isfinite(v) or v < 0 for v in row[1:]) for row in spin):
            raise ValueError('Invalid stirring checkpoint samples')
        if spin and tuple(spin[-1][1:]) != (m['bar_rpm'], m['motor_rpm']):
            raise ValueError('Stirring checkpoint ends at a different state')
        trial.spin = deque(tuple(row) for row in spin)
        trial.spin_bar_sum = sum(row[1] for row in spin)
        trial.spin_motor_sum = sum(row[2] for row in spin)
        trial.last_tick, trial.measurements = tick, copy.deepcopy(m)
        trial.result = trial._evaluate(tick, m)
        # Running sums can differ in their last bits after serialization.
        if not _equivalent(trial.result, state['result']):
            raise ValueError('Task checkpoint result is inconsistent')
        return trial

    def validate(self, state):
        self._validated(state)

    def restore(self, state):
        trial = self._validated(state)
        self.__dict__.update(trial.__dict__)


def _equivalent(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_equivalent(v, b[k]) for k, v in a.items())
    if isinstance(a, float):
        return type(b) in (int, float) and math.isclose(a, b, rel_tol=1.e-12, abs_tol=1.e-12)
    return a == b

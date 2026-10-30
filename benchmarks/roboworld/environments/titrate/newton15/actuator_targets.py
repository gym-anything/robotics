"""Rate-limited motor references; never alter physical joint state.

Newton 1.5's MuJoCo backend does not implement joint_velocity_limit. The
requested destination is retained exactly; this controller moves the motor
reference towards it at the URDF rate. Actual motion remains force/contact
driven and can differ from the reference under external loading.
"""

import math
import numpy as np


class ActuatorTargets:
    def __init__(self, joints, initial):
        self.joints = joints
        self.requested = np.asarray(initial, dtype=np.float64).copy()
        self.setpoint = self.requested.copy()
        self.indices = np.array([s['q'] for s in joints.values()], dtype=int)
        self.rates = np.array([s['velocity_limit'] for s in joints.values()])
        if not np.isfinite(self.rates).all() or np.any(self.rates <= 0.):
            raise ValueError('Actuator reference rates must be positive and finite')

    def command(self, targets):
        if not isinstance(targets, dict) or not targets:
            raise ValueError('Nonempty named joint target object required')
        proposed = self.requested.copy()
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f'Unknown robot actuator: {name}')
            spec = self.joints[name]
            if isinstance(value, bool) or not np.isscalar(value) or not math.isfinite(value):
                raise ValueError(f'Invalid target for {name}')
            if not spec['lower'] <= value <= spec['upper']:
                raise ValueError(f'Target outside {name} limits; no clamping')
            proposed[spec['q']] = value
        self.requested[:] = proposed

    def advance(self, dt):
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError('Positive finite reference timestep required')
        i = self.indices
        delta = self.requested[i]-self.setpoint[i]
        # This limits the commanded reference rate, not an API target or a
        # physical velocity. No requested destination or duration is changed.
        self.setpoint[i] += np.sign(delta)*np.minimum(np.abs(delta), self.rates*dt)
        return self.setpoint

    def capture(self):
        return {'requested': self.requested.copy(), 'setpoint': self.setpoint.copy()}

    def validate(self, data):
        if not isinstance(data, dict) or set(data) != {'requested', 'setpoint'}:
            raise ValueError('Invalid actuator reference checkpoint layout')
        result = {}
        for key in ('requested', 'setpoint'):
            value = np.asarray(data[key])
            if value.shape != self.requested.shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f'Invalid actuator reference checkpoint: {key}')
            for spec in self.joints.values():
                # Initial model references are float32, so allow their
                # representational roundoff at an exact joint boundary.
                if not spec['lower']-1e-7 <= value[spec['q']] <= spec['upper']+1e-7:
                    raise ValueError(f'Out-of-range actuator checkpoint: {key}')
            result[key] = value.copy()
        return result

    def restore_validated(self, data):
        self.requested[:] = data['requested']
        self.setpoint[:] = data['setpoint']

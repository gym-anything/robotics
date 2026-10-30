"""Compose physical filler joints, gas circuit and existing liquid stepping."""

import copy
import math
import numpy as np
from .pipette_pressure import PipettePressure
from .pipette_airway import PipetteAirway
from .pipette_filler import (FillerForces, DEAD_VOLUME_M3, PISTON_AREA_M2,
    VENT_RADIUS_M, VENT_TRAVEL_M)
from .pneumatics import IsothermalChamber
from .solid_sampling import transformed


class FillerRuntime:
    def __init__(self, liquid):
        self.liquid = liquid
        self.world = liquid.world
        self.port = PipettePressure(liquid.water, dead_volume_m3=DEAD_VOLUME_M3)
        self.airway = PipetteAirway(liquid.water)
        self.piston = FillerForces(self.world, self.port)
        liquid.water.force_models = (self.port,)
        self.body = self.world.lab.apparatus.bodies['volumetric_pipette']
        self.prepare()

    def coordinates(self):
        pose = self.world.state.body_q.numpy()[self.body]
        local = self.world.model.body_com.numpy()[self.body:self.body+1]
        com = transformed(local, pose)[0]
        position, _ = self.world.joint('pipette_plunger')
        vent, _ = self.world.joint('pipette_vent')
        return pose, com, position*PISTON_AREA_M2, vent

    def prepare(self):
        pose, com, volume, vent = self.coordinates()
        wp = self.liquid.water.wp
        transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
        area = self.airway.area(transform)
        # The actual valve travel opens a port. This is geometry-to-area
        # mapping, not an API-side gripper/pipette shortcut.
        opening = float(np.clip(vent/VENT_TRAVEL_M, 0., 1.))
        self.port.prepare(pose, com, displaced_volume_m3=volume,
            vent_area_m2=math.pi*VENT_RADIUS_M**2*opening, tip_air_area_m2=area)

    def finish(self):
        # End-of-step gas state uses the actual new piston/liquid positions.
        # Gas exchange happens once, in prepare(); this adds no physics step.
        pose, _, displacement, _ = self.coordinates()
        occupied = self.port.measure(pose)
        volume = self.port.capacity_m3+DEAD_VOLUME_M3+displacement-occupied
        work = self.port.gas.change_volume(volume)
        self.port.gauge_pa = self.port.gas.pressure_pa-self.port.gas.air.ambient_pressure_pa
        self.port.last.update(end_step_volume_work_j=work['gauge_work_j'],
            occupied_volume_ml=occupied*1.e6, gas_volume_ml=volume*1.e6,
            gas_pressure_pa=self.port.gas.pressure_pa, pressure_pa=self.port.gas.pressure_pa,
            mass_kg=self.port.gas.mass_kg, gauge_pressure_pa=self.port.gauge_pa,
            displaced_volume_ml=displacement*1.e6)

    def state(self):
        return {**copy.deepcopy(self.port.last),
            'airway_step_start': copy.deepcopy(self.airway.last),
            'piston_step_start': copy.deepcopy(self.piston.last),
            'model': 'Lumped connected air cushion; resolved particle liquid; physical plunger/vent',
            'gas_state_tick': self.world.tick}

    def capture(self):
        return {'mass_kg': self.port.gas.mass_kg, 'volume_m3': self.port.gas.volume_m3,
            'port_last': copy.deepcopy(self.port.last), 'airway_last': copy.deepcopy(self.airway.last),
            'piston_last': copy.deepcopy(self.piston.last)}

    def validate(self, snapshot):
        if not isinstance(snapshot, dict) or set(snapshot) != {'mass_kg', 'volume_m3', 'port_last', 'airway_last', 'piston_last'}:
            raise ValueError('Invalid pipette gas checkpoint layout')
        for key in ('mass_kg', 'volume_m3'):
            if not isinstance(snapshot[key], (float, int)) or not math.isfinite(snapshot[key]) or snapshot[key] <= 0.:
                raise ValueError('Invalid pipette gas mass/volume')
        for key in ('port_last', 'airway_last', 'piston_last'):
            if not isinstance(snapshot[key], dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in snapshot[key].values()):
                raise ValueError('Invalid pipette gas diagnostic checkpoint')

    def restore(self, snapshot):
        self.validate(snapshot)
        gas = IsothermalChamber(snapshot['volume_m3'])
        gas.mass_kg = snapshot['mass_kg']
        self.port.gas = gas
        self.port.gauge_pa = gas.pressure_pa-gas.air.ambient_pressure_pa
        self.port.last = copy.deepcopy(snapshot['port_last'])
        self.airway.last = copy.deepcopy(snapshot['airway_last'])
        self.piston.last = copy.deepcopy(snapshot['piston_last'])

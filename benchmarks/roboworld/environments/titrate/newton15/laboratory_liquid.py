"""Liquid/chemistry subsystem composed into the existing laboratory clock."""

from dataclasses import asdict
import numpy as np
from .liquid_charge import laboratory_charge, Reagents
from .sph import ProjectedWater, SphNumerics
from .sph_solid import SphSolidCoupling
from .chemistry import Chemistry


class LaboratoryLiquid:
    def __init__(self, world, numerics, reagents=None):
        self.world = world
        self.reagents = reagents or Reagents()
        positions, self.initial_groups = laboratory_charge(world.lab, world.state,
            numerics.spacing_m, self.reagents, wall_clearance_fraction=numerics.wall_clearance_fraction)
        self.water = ProjectedWater(positions, numerics)
        if numerics.hydrostatic_initial_guess:
            # A pressure-solver initial iterate, NOT an imposed force, particle
            # velocity, position, or water level. Each initially separate
            # vessel starts with rho*g*depth; ordinary residual convergence
            # must still accept it before any integration. Never reapplied.
            guess = np.zeros(len(positions), np.float32)
            for first, last in self.initial_groups.values():
                if last > first:
                    surface_z = positions[first:last, 2].max()+numerics.spacing_m/2
                    guess[first:last] = numerics.dt_s*9.80665*(surface_z-positions[first:last, 2])
            self.water.warm_density.assign(guess/numerics.warm_start_fraction)
        # Contact-angle estimates are explicit model parameters. The glass
        # value is the researched clean-silica reference, not a universal
        # property of all specimens. Other materials retain declared estimates.
        bodies = world.lab.apparatus.bodies
        angles = {i: 80. for i in range(-1, world.model.body_count)}
        for name in ("flask", "acid_bottle", "base_bottle", "rinse_beaker", "volumetric_pipette"):
            angles[bodies[name]] = 25.
        angles[bodies["stir_bar"]] = 108.
        angles[bodies["stopcock"]] = 108.
        # Static glass must not inherit the bench material's angle. Per-shape
        # angles are carried by SolidSamples below; each quadrature point uses
        # the actual containing shape material, including world-static glass.
        shape_angles = {i: {"glass": 25., "ptfe": 108.}.get(world.lab.materials.get(i),
            angles[world.lab.builder.shape_body[i]]) for i in range(world.model.shape_count)}
        self.coupling = SphSolidCoupling(world.lab.builder, world.model, self.water, angles,
            shape_contact_angles=shape_angles,
            particle_regions=tuple((first, last) for first, last in self.initial_groups.values() if last > first))
        self.chemistry = Chemistry(len(positions), self.water.volume_per_particle_m3,
            device=numerics.device)
        r = self.reagents
        for name, (first, last) in self.initial_groups.items():
            indices = slice(first, last)
            if name == "flask":
                volume = r.aliquot_ml+r.dilution_ml
                self.chemistry.charge(indices, acid_m=r.acid_molar*r.aliquot_ml/volume if volume else 0.,
                    indicator_m=r.indicator_molar)
            else:
                self.chemistry.charge(indices, acid_m=r.acid_molar if name == "acid_bottle" else 0.,
                    base_m=r.base_molar if name in ("burette", "base_bottle") else 0.)
        self.initial_totals = self.chemistry.observe()["totals_mol"]
        self.filler = None
        if 'pipette_plunger' in world.lab.apparatus.bodies:
            from .filler_runtime import FillerRuntime
            self.filler = FillerRuntime(self)

    def advance(self):
        if self.filler is not None:
            self.filler.prepare()
        forces = self.coupling.advance(self.world.state)
        if self.filler is not None:
            forces[self.filler.body] += self.filler.port.reaction.numpy()[0]
        self.chemistry.step(self.water.state.particle_q, self.water.config.dt_s,
            self.water.config.support_m)
        return forces

    def state(self):
        q, v = self.water.read()
        chemistry = self.chemistry.observe()
        from .liquid_readings import readings
        return {"tick": self.water.tick, "positions_m": q, "velocities_m_s": v,
            "parcel_volume_m3": self.water.volume_per_particle_m3,
            "material_volume_ml": len(q)*self.water.volume_per_particle_m3*1.e6,
            "numerics": asdict(self.water.config), "reagents": asdict(self.reagents),
            "chemistry": chemistry, "projection": self.water.last_projection,
            "pipette_filler": self.filler.state() if self.filler else None,
            "readings": readings(self, positions=q, moles=chemistry['moles']),
            "rigid_loads": self.coupling.last_applied_body_forces if self.water.tick else
                np.zeros((self.world.model.body_count, 6))}

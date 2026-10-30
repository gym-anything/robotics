"""Data-only liquid checkpoints, including moving boundary quadrature state."""

from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import numpy as np
from .mechanical_snapshot import arrays
from .runtime import require_newton


def source_identity():
    names = ("sph.py", "sph_kernels.py", "sph_kernel_functions.py", "neighbor_cache.py",
        "pressure_graph.py", "implicit_interface.py", "nonlinear_density.py", "sph_solid.py", "solid_sampling.py", "particle_bounds.py", "boundary_coverage.py", "fluid_contacts.py", "fluid_broadphase.py", "mesh_contacts.py", "revolved_queries.py",
        "pressure_unrolled.py", "analytic_solids.py", "cavity_contacts.py", "rigid_reduction.py", "interface_energy.py", "chemistry.py", "chemistry_kernels.py", "materials.py",
        "task.py", "task_readings.py", "liquid_charge.py", "laboratory_liquid.py", "liquid_readings.py", "liquid_snapshot.py", "runtime.py",
        "pipette_pressure.py", "pipette_airway.py", "pipette_filler.py", "filler_runtime.py", "pneumatics.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def capture(liquid):
    w, s, c = liquid.water, liquid.coupling.samples, liquid.chemistry
    sampling = None if s.local is None else {
        "local": s.local.numpy(), "body_ids": s.body_ids.numpy(),
        "angles": None if s.sample_contact_angles is None else s.sample_contact_angles.copy(),
        "coverage": {str(k): np.array(v, copy=True) for k, v in s.coverage.items()},
        "candidate_shapes": list(s.candidate_shape_ids), "rebuilds": s.rebuilds}
    return {"schema": "titration.liquid.snapshot.v2", "source": source_identity(), "numerics": asdict(w.config),
        "filler": liquid.filler.capture() if liquid.filler else None,
        "reagents": asdict(liquid.reagents), "tick": w.tick,
        "projection": copy.deepcopy(w.last_projection), "sampling": sampling,
        "last_rigid_loads": liquid.coupling.last_applied_body_forces.copy(),
        "pressure_history": {"density": w.warm_density.numpy(), "divergence": w.warm_divergence.numpy()},
        "initial_totals": copy.deepcopy(liquid.initial_totals),
        "initial_groups": copy.deepcopy(liquid.initial_groups),
        "arrays": {name: {key: value.numpy().copy() for key, value in arrays(obj).items()}
                   for name, obj in (("state", w.state), ("next_state", w.next_state), ("chemistry", c))}}


def validate(liquid, snapshot):
    w, s = liquid.water, liquid.coupling.samples
    keys = {"schema", "source", "numerics", "reagents", "tick", "projection", "sampling", "arrays", "initial_totals", "initial_groups", "last_rigid_loads", "pressure_history", "filler"}
    if set(snapshot) != keys or snapshot["schema"] != "titration.liquid.snapshot.v2" or snapshot["numerics"] != asdict(w.config) or snapshot["reagents"] != asdict(liquid.reagents):
        raise ValueError("Liquid checkpoint configuration/schema mismatch")
    if snapshot["source"] != source_identity():
        raise ValueError("Liquid checkpoint source identity mismatch")
    if liquid.filler is None:
        if snapshot['filler'] is not None:
            raise ValueError('Pipette filler checkpoint configuration differs')
    else:
        liquid.filler.validate(snapshot['filler'])
    if type(snapshot["tick"]) is not int or snapshot["tick"] < 0:
        raise ValueError("Invalid liquid checkpoint clock")
    if snapshot["initial_totals"] != liquid.initial_totals or snapshot["initial_groups"] != liquid.initial_groups:
        raise ValueError("Liquid checkpoint initial inventory mismatch")
    writes = []
    if set(snapshot["pressure_history"]) != {"density", "divergence"}:
        raise ValueError("Invalid pressure history layout")
    for key, target in (("density", w.warm_density), ("divergence", w.warm_divergence)):
        incoming = np.asarray(snapshot["pressure_history"][key])
        if incoming.shape != (w.model.particle_count,) or incoming.dtype != np.dtype("float32") or not np.isfinite(incoming).all():
            raise ValueError("Invalid checkpoint pressure history")
        writes.append((target, incoming))
    loads = np.asarray(snapshot["last_rigid_loads"])
    if loads.shape != (liquid.world.model.body_count, 6) or not np.isfinite(loads).all():
        raise ValueError("Invalid checkpoint liquid rigid loads")
    if set(snapshot["arrays"]) != {"state", "next_state", "chemistry"}:
        raise ValueError("Invalid liquid array namespaces")
    for name, obj in (("state", w.state), ("next_state", w.next_state), ("chemistry", liquid.chemistry)):
        targets = arrays(obj)
        if set(targets) != set(snapshot["arrays"][name]):
            raise ValueError("Liquid checkpoint array layout mismatch")
        for key, target in targets.items():
            incoming = np.asarray(snapshot["arrays"][name][key])
            current = target.numpy()
            if incoming.shape != current.shape or incoming.dtype != current.dtype or not np.isfinite(incoming).all():
                raise ValueError(f"Invalid liquid checkpoint array {name}/{key}")
            writes.append((target, incoming))
    sampling = snapshot["sampling"]
    if sampling is not None:
        if set(sampling) != {"local", "body_ids", "angles", "coverage", "candidate_shapes", "rebuilds"}:
            raise ValueError("Invalid boundary sampling checkpoint")
        local, ids = np.asarray(sampling["local"]), np.asarray(sampling["body_ids"])
        if local.ndim != 2 or local.shape[1] != 3 or len(local) > s.maximum_points or ids.shape != (len(local),) or not np.isfinite(local).all():
            raise ValueError("Invalid boundary sample data")
        if not set(map(int, ids)) <= s.owner_bodies or type(sampling["rebuilds"]) is not int or sampling["rebuilds"] < 1:
            raise ValueError("Invalid boundary identities")
        shape = (s.region_count, 2, 3) if s.particle_regions is not None else (2, 3)
        if set(sampling["coverage"]) != {str(k) for k in s.groups} or any(np.asarray(v).shape != shape or not np.isfinite(v).all() or np.any(np.asarray(v).reshape(-1, 2, 3)[:, 0] > np.asarray(v).reshape(-1, 2, 3)[:, 1]) for v in sampling["coverage"].values()):
            raise ValueError("Invalid boundary coverage")
        angles = sampling["angles"]
        if angles is not None and (np.asarray(angles).shape != (len(local),) or not np.isfinite(angles).all() or np.any((angles <= 0) | (angles >= 180))):
            raise ValueError("Invalid boundary material angles")
        allowed = {shape.index for shapes in s.groups.values() for shape in shapes}
        if not set(sampling["candidate_shapes"]) <= allowed:
            raise ValueError("Invalid collision candidate shape")
    return writes


def restore(liquid, snapshot):
    writes = validate(liquid, snapshot)
    _, wp = require_newton()
    w, s = liquid.water, liquid.coupling.samples
    for target, incoming in writes:
        target.assign(incoming)
    sample = snapshot["sampling"]
    if sample is None:
        s.local, s.coverage, s.rebuilds = None, {}, 0
    else:
        s.local = wp.array(sample["local"], dtype=wp.vec3, device=w.config.device)
        s.body_ids = wp.array(sample["body_ids"], dtype=int, device=w.config.device)
        s.q, s.v = wp.empty_like(s.local), wp.empty_like(s.local)
        s.sample_contact_angles = None if sample["angles"] is None else np.array(sample["angles"], copy=True)
        s.coverage = {int(k): np.array(v, copy=True) for k, v in sample["coverage"].items()}
        s.rebuilds = sample["rebuilds"]
        s.candidate_shape_ids = list(sample["candidate_shapes"])
    liquid.coupling.contacts.generation = None
    liquid.coupling.contacts.broadphase.generation = None
    w.capillary._solid_generation = None
    w.tick, w.last_projection = snapshot["tick"], copy.deepcopy(snapshot["projection"])
    liquid.coupling.last_applied_body_forces = np.array(snapshot["last_rigid_loads"], copy=True)
    if liquid.filler:
        liquid.filler.restore(snapshot['filler'])

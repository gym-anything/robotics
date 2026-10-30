"""Serialized persistent apparatus session, shared by local and GAR controls.

This dry-apparatus component session is not the completed titration benchmark.
It exercises the real robot, rotary controls, free vessels, magnetic coupling,
camera clock and checkpoints while the liquid acceptance work remains separate.
No automatic retries, resets, restores, object pose setters or background ticks.
"""

from contextlib import contextmanager
import copy
import math
from pathlib import Path
import threading
import uuid
import numpy as np

from .mechanics import LaboratoryMechanics
from .mechanical_snapshot import capture, restore
from .magnetics import MagneticDrive


class ApparatusSession:
    def __init__(self, config=None, *, dt_s=.001, output_dir=None, renderer_options=None,
                 liquid_numerics=None, reagents=None):
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Positive finite timestep required")
        self._lock = threading.RLock()
        self.dt_s = float(dt_s)
        self.world = LaboratoryMechanics(config)
        self.config = self.world.lab.config
        support = self.world.lab.apparatus.bodies["stirrer"]
        self.magnet = MagneticDrive(self.world.lab.apparatus.bodies["stir_bar"],
            (0., 0., .087), phase=np.pi, support_body=support,
            support_com=tuple(self.world.model.body_com.numpy()[support]))
        self.output = Path(output_dir or Path("titrate/output/sessions")/uuid.uuid4().hex).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.renderer_options = dict(renderer_options or {})
        self._renderer = None
        self.closed = False
        self.generation = uuid.uuid4().hex
        self.command_id = None
        self.trace_callback = None
        self.trace_liquid_stride = 20
        self.liquid = None
        self.evaluator, self.task_readings = None, None
        if liquid_numerics is not None:
            if liquid_numerics.dt_s != self.dt_s or liquid_numerics.device != self.config.device:
                raise ValueError("Rigid/liquid timestep and device must agree exactly; no substep substitution")
            from .laboratory_liquid import LaboratoryLiquid
            self.liquid = LaboratoryLiquid(self.world, liquid_numerics, reagents)
            if self.liquid.reagents.aliquot_ml > 0:
                from .task import EndpointEvaluator
                from .task_readings import TaskReadings
                r = self.liquid.reagents
                self.evaluator = EndpointEvaluator(self.dt_s, r.aliquot_ml, r.acid_molar, r.base_molar)
                self.task_readings = TaskReadings(self.liquid)
                self.evaluator.update(0, **self.task_readings.read())
        self._initial = self.snapshot()

    @property
    def tick(self):
        return self.world.tick

    def _require_open(self):
        if self.closed:
            raise RuntimeError("Environment session is closed")

    @contextmanager
    def locked(self):
        with self._lock:
            self._require_open()
            yield

    def _step(self, assisted_joint_torques=None, body_wrenches=None):
        dial, _ = self.world.joint("stirrer_knob")
        fraction = float(np.clip(dial/(5*np.pi/3), 0., 1.))
        self.magnet.rpm = 0. if fraction < .01 else 100.+1400.*fraction
        try:
            if self.liquid is not None:
                fluid_forces = self.liquid.advance()
                body_wrenches = fluid_forces if body_wrenches is None else np.asarray(body_wrenches)+fluid_forces
            filler = self.liquid.filler if self.liquid else None
            self.world.step(self.dt_s, assisted_joint_torques=assisted_joint_torques,
                body_wrenches=body_wrenches,
                force_models=(self.magnet, filler.piston) if filler else (self.magnet,))
            if filler:
                filler.finish()
            if self.evaluator is not None:
                self.evaluator.update(self.tick, **self.task_readings.read())
        except Exception as exc:
            self.world.fault = f"Coupled step failed: {type(exc).__name__}: {exc}"
            raise
        if self.trace_callback is not None:
            try:
                self.trace_callback(self.trace_sample())
            except Exception as exc:
                # The step did happen. Latch its new tick/state, never report
                # that an action failed before mutation or secretly restore it.
                self.world.fault = f"Recording failure after tick {self.tick}: {exc}"
                raise

    def advance(self, ticks, *, targets=None):
        if type(ticks) is not int or ticks < 1:
            raise ValueError("Positive integer tick count required; no rounding")
        with self.locked():
            if self.world.fault:
                raise RuntimeError(self.world.fault)
            if targets:
                self.world.command_joints(targets)
            before = self.tick
            for _ in range(ticks):
                self._step()
            return {"ticks_executed": self.tick-before, "tick": self.tick}

    def assisted_step(self, *, joint_torques=None, body_wrenches=None):
        """Explicit component-fixture access; never a robot/policy API action."""
        with self.locked():
            self._step(joint_torques, body_wrenches)

    def trace_sample(self):
        import mujoco

        w = self.world
        data = w.solver.mj_data.contact
        forces = np.zeros((len(data), 6))
        for index in range(len(data)):
            mujoco.mj_contactForce(w.solver.mj_model, w.solver.mj_data, index, forces[index])
        contacts = {"geometry_ids": np.array(data.geom), "point_m": np.array(data.pos),
            "frame": np.array(data.frame), "distance_m": np.array(data.dist),
            "contact_frame_wrench": forces, "constraint_address": np.array(data.efc_address)}
        result = {"tick": self.tick, "time_s": w.time_s, "generation": self.generation,
            "command_id": self.command_id, "body_pose_xyzw": w.state.body_q.numpy(),
            "body_velocity_linear_angular": w.state.body_qd.numpy(),
            "joint_q": w.state.joint_q.numpy(), "joint_qd": w.state.joint_qd.numpy(),
            "joint_targets": w.control.joint_target_q.numpy(),
            "requested_joint_targets": w.actuator_targets.requested.copy(),
            "contacts": contacts, "contact_sample_step_start_tick": max(0, self.tick-1),
            "magnetic_drive": copy.deepcopy(self.magnet.last), "fault": w.fault}
        if self.liquid is not None:
            liquid, water = self.liquid, self.liquid.water
            result['liquid'] = {'tick': water.tick, 'projection': copy.deepcopy(water.last_projection),
                'pipette_filler': liquid.filler.state() if liquid.filler else None,
                'rigid_loads': liquid.coupling.last_applied_body_forces.copy(),
                'full_particle_sample': self.tick % self.trace_liquid_stride == 0,
                'particle_stride_ticks': self.trace_liquid_stride}
            if result['liquid']['full_particle_sample']:
                result['liquid'].update(positions_m=water.state.particle_q.numpy(),
                    velocities_m_s=water.state.particle_qd.numpy(), moles=liquid.chemistry.moles.numpy(),
                    reacted_mol=liquid.chemistry.reacted.numpy())
        if self.evaluator is not None:
            result['evaluation'] = copy.deepcopy(self.evaluator.result)
        return result

    def get_state(self):
        with self.locked():
            w = self.world
            q, v = w.state.body_q.numpy(), w.state.body_qd.numpy()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                w.fault = "Nonfinite mechanics state"
                raise FloatingPointError(w.fault)
            result = {"environment_id": "titration-apparatus-fixture", "tick": self.tick,
                "time_s": w.time_s, "generation": self.generation, "dt_s": self.dt_s,
                "ready": not bool(w.fault), "fault": w.fault,
                "benchmark_ready": False, "liquid_present": False,
                "body_pose_xyzw": q, "body_velocity_linear_angular": v,
                "joints": w.proprioception(),
                "apparatus_joints": {name: {
                    ("position_m" if name in ('pipette_plunger', 'pipette_vent') else "position_rad"): w.joint(name)[0],
                    ("velocity_m_s" if name in ('pipette_plunger', 'pipette_vent') else "velocity_rad_s"): w.joint(name)[1]}
                    for name in w.lab.apparatus.joints},
                "bodies": {name: {"body_index": index, "pose_xyzw": q[index].tolist(),
                    "velocity_linear_angular": v[index].tolist()}
                    for name, index in w.lab.apparatus.bodies.items()}}
            if self.liquid is not None:
                result.update(environment_id="titration-coupled-development", liquid_present=True,
                    liquid=self.liquid.state())
            if self.evaluator is not None:
                result['evaluation'] = copy.deepcopy(self.evaluator.result)
            return result

    def snapshot(self):
        with self.locked():
            result = {"schema": "titration.apparatus.session.v1", "dt_s": self.dt_s,
                "mechanics": capture(self.world), "magnet_phase": float(self.magnet.phase),
                "magnet_rpm": float(self.magnet.rpm), "magnet_last": copy.deepcopy(self.magnet.last)}
            if self.liquid is not None:
                from .liquid_snapshot import capture as capture_liquid
                result["liquid"] = capture_liquid(self.liquid)
            if self.evaluator is not None:
                result['evaluation'] = self.evaluator.snapshot()
            return result

    def restore(self, snapshot):
        with self.locked():
            expected = {"schema", "dt_s", "mechanics", "magnet_phase", "magnet_rpm", "magnet_last"}
            if self.liquid is not None:
                from .liquid_snapshot import validate as validate_liquid, restore as restore_liquid
                expected.add("liquid")
            if self.evaluator is not None:
                expected.add('evaluation')
            if set(snapshot) != expected or snapshot["schema"] != "titration.apparatus.session.v1" or snapshot["dt_s"] != self.dt_s:
                raise ValueError("Checkpoint session schema or timestep mismatch")
            if not np.isfinite([snapshot["magnet_phase"], snapshot["magnet_rpm"]]).all():
                raise ValueError("Invalid magnetic checkpoint")
            if self.liquid is not None:
                validate_liquid(self.liquid, snapshot["liquid"])
            if self.evaluator is not None:
                self.evaluator.validate(snapshot['evaluation'])
                if snapshot['evaluation']['last_tick'] != snapshot['liquid']['tick']:
                    raise ValueError('Checkpoint task/liquid clocks differ')
            restore(self.world, snapshot["mechanics"])
            if self.liquid is not None:
                restore_liquid(self.liquid, snapshot["liquid"])
            if self.evaluator is not None:
                self.evaluator.restore(snapshot['evaluation'])
            self.magnet.phase, self.magnet.rpm = snapshot["magnet_phase"], snapshot["magnet_rpm"]
            self.magnet.last = copy.deepcopy(snapshot["magnet_last"])
            self.generation = uuid.uuid4().hex
            return self.get_state()

    def reset(self):
        return self.restore(self._initial)

    def observe(self, cameras=("bench_rgb",)):
        names = tuple(cameras)
        if len(names) != len(set(names)) or not names or set(names)-{"bench_rgb", "laboratory", "apparatus", "flask", "valve", "dispensing", "gripper", "pipetting", "pipette_tip", "pipette_rack"}:
            raise ValueError("Choose distinct authored cameras")
        with self.locked():
            if self._renderer is None:
                from .usd_lab import export
                from .renderer import Renderer
                scene = export(self.world.lab, self.world.model, self.world.state, self.output/"scene")
                self._renderer = Renderer(scene, **self.renderer_options)
            surface = None
            if self.liquid is not None:
                from .surface import reconstruct_sparse
                water = self.liquid.water
                surface = reconstruct_sparse(water.state.particle_q.numpy(), water.volume_per_particle_m3,
                    3*water.config.spacing_m, water.config.spacing_m, device=self.config.device)
                from .liquid_optics import apply
                apply(surface, water.state.particle_q.numpy(), self.liquid.chemistry.moles.numpy(),
                    water.volume_per_particle_m3, 3*water.config.spacing_m, self.config.device)
            result = self._renderer.observe(self.tick, self.world.state.body_q.numpy(), names, liquid=surface)
            if surface is not None:
                result["surface"] = {"mesh_volume_m3": float(surface.volume_m3),
                    "material_volume_m3": float(surface.material_volume_m3),
                    "unresolved_parcels": surface.unresolved_particle_count,
                    "approximation": surface.approximation}
            return result

    def close(self):
        with self._lock:
            if not self.closed:
                if self._renderer is not None:
                    self._renderer.close()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

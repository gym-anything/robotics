"""GAR integration for explicitly configured titration development sessions.

Registry availability does not imply physical acceptance: callers must choose
a dry-apparatus fixture or explicit liquid numerics. No unvalidated liquid
configuration is silently chosen. HTTP authorization is shared with GAR.
"""

import base64
from collections import deque
from pathlib import Path
import uuid
from .base import RoboticsBackend, reject_unexecuted
from ..actions import NamedJointTarget
from ..errors import ActionRejectedError, BackendProtocolError, UnsupportedActionError, UnsupportedConfigurationError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation


class NewtonTitrationBackend(RoboticsBackend):
    def __init__(self, *, robot="panda", apparatus_fixture=False, config=None, dt_s=None,
                 cameras=("bench_rgb",), output_dir=None, renderer_options=None, recording=True,
                 liquid_numerics=None, reagents=None, liquid_trace_stride=20, variation_id=None, task='titration'):
        if robot != "panda":
            raise UnsupportedConfigurationError("Titration uses the manufacturer-description Franka Panda")
        if type(apparatus_fixture) is not bool or apparatus_fixture == (liquid_numerics is not None):
            raise UnsupportedConfigurationError("Choose exactly one: apparatus_fixture=True or an explicit liquid_numerics configuration")
        from titrate.newton15.sph import SphNumerics
        from titrate.newton15.liquid_charge import Reagents
        from titrate.newton15.lab_config import LabConfig
        from titrate.newton15.task_catalogue import task_name
        self.task = task_name(task)
        if task != 'titration' and apparatus_fixture:
            raise UnsupportedConfigurationError('The catalogue tasks other than titration require liquid')
        if variation_id is not None and (config is not None or reagents is not None):
            raise UnsupportedConfigurationError('Catalogue variation supplies layout/reagents; do not combine explicit overrides')
        if isinstance(liquid_numerics, dict):
            liquid_numerics = SphNumerics(**liquid_numerics)
        if isinstance(reagents, dict):
            reagents = Reagents(**reagents)
        if isinstance(config, dict):
            config = LabConfig(**config)
        self._base_config = config or LabConfig()
        self._base_reagents = reagents or Reagents()
        self.variation_id = None
        if variation_id is not None:
            from titrate.newton15.variations import resolve, index
            config, reagents = resolve(variation_id, spacing_m=liquid_numerics.spacing_m if liquid_numerics else None)
            self.variation_id = f'titration-n15-{index(variation_id):04d}'
        if type(liquid_trace_stride) is not int or liquid_trace_stride < 1:
            raise ValueError("Positive integer liquid trace stride required")
        effective_dt = (liquid_numerics.dt_s if liquid_numerics else .001) if dt_s is None else dt_s
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras)-{"bench_rgb"}:
            raise ValueError("Policy observations support bench_rgb; diagnostic views use render")
        from titrate.newton15.task_session import make_session
        from titrate.newton15.geometry_api import GeometryCatalog
        from titrate.newton15.session_trace import SessionTrace
        from whiteboard.storage import Store
        self.session = make_session(config, task=self.task, dt_s=effective_dt, output_dir=output_dir,
            renderer_options=renderer_options, liquid_numerics=liquid_numerics, reagents=reagents)
        self._liquid_numerics = liquid_numerics
        self._recording = recording
        self.session.trace_liquid_stride = liquid_trace_stride
        if not self.session.config.robot:
            self.session.close()
            raise UnsupportedConfigurationError("GAR apparatus fixture requires the physical Panda")
        self.geometry = GeometryCatalog(self.session.world)
        self.output = self.session.output
        self.trace = SessionTrace(self.session, self.output/"physics_trace.sqlite3") if recording else None
        self.checkpoints = Store(self.output/"checkpoints", budget_bytes=64*1024**2) if recording else None
        self.images = deque()
        options = renderer_options or {}
        width, height = options.get("width", 1280), options.get("height", 720)
        self.camera = CameraSpec("bench_rgb", "authored_external_stand", width, height, width, height,
            (.036, .036*height/width), (.028, .028))
        environment_id = "titration-apparatus-fixture" if apparatus_fixture else "titration-coupled-development"
        self.capabilities = Capabilities(environment_id, "newton", "panda",
            frozenset({"named_joint_target"}), frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, False, True, self.cameras, (self.camera,) if self.cameras else (),
            notes={"qualification": "Development fixture; not release acceptance",
                "liquid": "absent" if apparatus_fixture else "Newton-state repository SPH with local chemistry",
                "clock": "Explicit exact physics ticks; no background advancement",
                "joint_units": "panda_joint1..7: radians; panda_finger_joint1/2: metres",
                "camera": "Ideal calibrated external RGB; 36 mm virtual filmback, not a claim about measured sensor dimensions"})

    def get_state(self):
        try:
            return self.session.get_state()
        except (RuntimeError, FloatingPointError) as exc:
            raise BackendProtocolError(str(exc)) from exc

    def evaluate(self):
        result = super().evaluate()
        if result is not None and result.get('valid') is False:
            raise BackendProtocolError(result['invalid_reason'])
        return result

    def reset(self, *, seed=None, options=None):
        if options:
            if set(options)-{'variation_id', 'task'} or seed is not None:
                raise ValueError('Use variation_id and/or task; do not combine a seed override')
            from titrate.newton15.task_catalogue import task_name
            task = task_name(options.get('task', self.task))
            if 'variation_id' not in options:
                # Retain nominal charge/layout, never the finish preset's prepared charge.
                cfg = self.session.config
                from titrate.newton15.liquid_charge import Reagents
                nominal = getattr(self.session, 'task_setup', {}).get('nominal_reagents')
                reagents = Reagents(**nominal) if nominal else (
                    self.session.liquid.reagents if self.session.liquid else self._base_reagents)
                return self._replace_session(cfg, reagents, self.variation_id, task)
            from titrate.newton15.variations import resolve, index
            cfg, reagents = resolve(options['variation_id'], base_config=self._base_config,
                base_reagents=self._base_reagents,
                spacing_m=self._liquid_numerics.spacing_m if self._liquid_numerics else None)
            return self._replace_session(cfg, reagents, f'titration-n15-{index(options["variation_id"]):04d}', task)
        if seed is not None and seed != self.session.config.seed:
            raise ValueError("This component fixture has no seeded randomization; requested seed cannot be honored")
        return self.session.reset(), {"continuity": "reset", "seed": self.session.config.seed}

    def variation_contract(self):
        from titrate.newton15.variations import contract
        return {**contract(self.variation_id), 'current_task': self.task,
                'liquid_present': self.session.liquid is not None}

    def select_variation(self, request):
        from titrate.newton15.variations import select
        # A preview without a task selects the titration task.
        request = dict(request)
        result = select(request, base_config=self._base_config, base_reagents=self._base_reagents,
            spacing_m=self._liquid_numerics.spacing_m if self._liquid_numerics else None)
        if self.task != 'titration' and 'task' not in request:
            result['reset_request']['options']['task'] = 'titration'
        result['liquid_present'] = self.session.liquid is not None
        if self.session.liquid is None:
            result['task'] = 'Dry apparatus fixture: no liquid and no titration evaluation. '+result['task']
        return result

    def _replace_session(self, config, reagents, variation_id, task=None):
        from titrate.newton15.task_session import make_session
        task = self.task if task is None else task
        from titrate.newton15.geometry_api import GeometryCatalog
        from titrate.newton15.session_trace import SessionTrace
        candidate, trace = None, None
        try:
            candidate = make_session(config, task=task, dt_s=self.session.dt_s,
                output_dir=self.output/'episodes'/uuid.uuid4().hex,
                renderer_options=self.session.renderer_options,
                liquid_numerics=self._liquid_numerics, reagents=reagents)
            candidate.trace_liquid_stride = self.session.trace_liquid_stride
            geometry = GeometryCatalog(candidate.world)
            trace = SessionTrace(candidate, candidate.output/'physics_trace.sqlite3') if self._recording else None
            state = candidate.get_state()
        except BaseException:
            if trace is not None:
                trace.close()
            if candidate is not None:
                candidate.close()
            raise
        # Candidate construction/validation never advances or closes the current
        # environment. Only a successful candidate reaches this replacement.
        old, old_trace = self.session, self.trace
        self.session, self.trace, self.geometry = candidate, trace, geometry
        self.variation_id = variation_id
        self.task = task
        if old_trace is not None:
            old_trace.close()
        old.close()
        return state, {'continuity': 'new_variation', 'variation_id': variation_id,
            'seed': config.seed, 'previous_episode_preserved_during_validation': True}

    def policy_access(self):
        from titrate.newton15.task_catalogue import description
        joints = self.session.world.lab.robot_joints
        return PolicyAccess("panda", (self.camera,) if self.cameras else (),
            {("joints", name, field): None for name in joints for field in ("position", "velocity")},
            frozenset({"named_joint_target"}),
            lambda: ("Dry laboratory apparatus component fixture. No liquid or titration evaluation is present."
                if self.session.liquid is None else description(self.task, self.session.liquid.reagents)),
            {"dt_s": self.session.dt_s, "joints": {name: {**{k: v for k, v in spec.items() if k not in ("q", "dof")},
                "unit": "metre" if "finger" in name else "radian"} for name, spec in joints.items()}})

    def _render(self, cameras):
        with self.session.locked():
            from ..render_lifecycle import renderer_operation
            with renderer_operation(self.session, "_renderer"):
                result = self.session.observe(cameras)
            frames = {}
            for name, data in result["images"].items():
                path = self.output/f"{name}_{uuid.uuid4().hex}.png"
                path.write_bytes(base64.b64decode(data))
                self.images.append(path)
                frames[name] = path
                while len(self.images) > 64:
                    old = self.images.popleft()
                    if old.parent != self.output:
                        raise RuntimeError("Image retention escaped session output")
                    old.unlink(missing_ok=True)
            return frames

    def observation_from_state(self, state, *, sequence):
        with self.session.locked():
            if state["tick"] != self.session.tick or state["generation"] != self.session.generation:
                raise RuntimeError("Cannot combine stale state with a current camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], "panda", {"joints": state["joints"]},
                {name: ImageFrame(name, path, True, camera=self.camera) for name, path in frames.items()})

    def act(self, action):
        if not isinstance(action, NamedJointTarget):
            raise UnsupportedActionError("Use named Panda position targets and exact ticks; an empty target object holds current targets")
        try:
            info = reject_unexecuted(
                lambda: self.session.advance(action.ticks, targets=dict(action.targets_si)),
                lambda: (self.session.tick, self.session.world.actuator_targets.requested.copy()))
        except ValueError as exc:
            if isinstance(exc, ActionRejectedError) or self.session.world.fault is None:
                raise  # invalid requested input, rejected before physical stepping
            raise BackendProtocolError(f"Action stopped at tick {self.session.tick}: {exc}") from exc
        except (RuntimeError, FloatingPointError) as exc:
            raise BackendProtocolError(f"Action stopped at tick {self.session.tick}: {exc}") from exc
        state = self.get_state()
        result = state.get('evaluation')
        return BackendStep(state, info, result['score'] if result else None,
            result['success'] if result else False)

    def act_policy(self, action):
        # Simulator-only chemical state and the evaluator are teacher data.
        # Do not reveal them via reward or early success termination either.
        result = self.act(action)
        return BackendStep(result.state, result.info, None, False, result.truncated)

    def render(self, *, label="interactive"):
        name = "laboratory" if label in ("interactive", "http") else label
        return self._render((name,))[name]

    def get_geometry_index(self):
        return self.geometry.index()

    def get_geometry(self, geometry_id):
        return self.geometry.objects[str(geometry_id)]

    def get_geometry_asset(self, asset_id):
        return self.geometry.assets[str(asset_id)]

    def begin_action_trace(self, context):
        if self.trace is None:
            return {"available": False, "reason": "recording_disabled"}
        try:
            return self.trace.begin(context)
        except Exception as exc:
            # The shared gateway records a trace-start failure then invokes
            # act(). In this recording-required session that must not advance
            # unrecorded physics; latch the fault before the first step.
            self.session.world.fault = f"Trace could not start before tick {self.session.tick}: {exc}"
            raise

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def save_checkpoint(self, *, label):
        if self.checkpoints is None:
            raise RuntimeError("Checkpoint storage disabled")
        if not isinstance(label, str) or not 1 <= len(label) <= 64:
            raise ValueError("Checkpoint label must have 1..64 characters")
        with self.session.locked():
            self.checkpoints.put("checkpoint", label, self.session.snapshot())
            return {"name": label, "tick": self.session.tick, "durable": True}

    def restore_checkpoint(self, checkpoint):
        if self.checkpoints is None:
            raise RuntimeError("Checkpoint storage disabled")
        with self.session.locked():
            state = self.session.restore(self.checkpoints.get("checkpoint", checkpoint, arrays=True))
            return state, {"checkpoint": checkpoint, "continuity": "restored_discontinuity"}

    def list_checkpoints(self):
        return tuple({"name": name} for name in self.checkpoints.keys("checkpoint")) if self.checkpoints else ()

    def close(self):
        with self.session._lock:
            if self.trace:
                self.trace.close()
                self.trace = None
            if self.checkpoints:
                self.checkpoints.close()
                self.checkpoints = None
            self.session.close()

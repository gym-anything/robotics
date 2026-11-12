"""GAR's task session factory; no changes to physics-hashed source modules.

make_session builds a task's session with its setup.py (tasks/<task>/): titration's is the released
session.ApparatusSession, every other task's is catalogue_session below.
"""

from .task_catalogue import task_name, task_setup
from .task_presets import preset


def apply_rigid_start(world, setup):
    """Initialize both state buffers and free joints before liquid is charged."""
    from .runtime import require_newton
    newton, _ = require_newton()
    b = world.lab.builder
    names = world.lab.apparatus.bodies
    poses = world.state.body_q.numpy().copy()
    q = world.state.joint_q.numpy().copy()
    for name, position in setup['positions'].items():
        body = names[name]
        poses[body, :3] = position
        # add_body may have a free joint; never edit a mounted/revolute body.
        for j, child in enumerate(b.joint_child):
            if child == body:
                first = b.joint_q_start[j]
                if int(b.joint_type[j]) != int(newton.JointType.FREE):
                    raise ValueError(f'Task preset cannot move constrained body {name}')
                q[first:first+7] = poses[body]
    j = world.lab.apparatus.joints['stirrer_knob']
    q[b.joint_q_start[j]] = setup['dial_rad']
    for state in (world.state, world.next_state):
        state.body_q.assign(poses)
        state.joint_q.assign(q)
        newton.eval_fk(world.model, state.joint_q, world.state.joint_qd, state)
    # Same native initial-overlap guard as the existing session, after placement.
    world._validate_initial_contacts()


def make_session(config=None, *, task='titration', **kwargs):
    from .session import ApparatusSession  # noqa: F401  (a missing simulator fails before the task is checked)
    task_name(task)
    return task_setup(task).setup(config, **kwargs)


def catalogue_session(task, config=None, **kwargs):
    """The session of a task other than titration, from make_session's arguments: the dry apparatus with the task's
    preset start (task_presets.preset, its setup.py's arrange), then the liquid with the preset charge (prepared by
    its setup.py's precharge_chemistry, if it has one), the readings and the task's CatalogueEvaluator, which must
    start valid, unsuccessful and at score zero."""
    from .session import ApparatusSession
    from .lab_config import LabConfig
    from .liquid_charge import Reagents
    folder = task_setup(task)
    config = config or LabConfig()
    reagents = kwargs.get('reagents') or Reagents()
    numerics = kwargs.get('liquid_numerics')
    if numerics is None:
        raise ValueError('The new catalogue tasks require the explicit liquid configuration')
    setup = preset(task, config, reagents, numerics.spacing_m)
    session = ApparatusSession(config, **{**kwargs, 'liquid_numerics': None})
    try:
        apply_rigid_start(session.world, setup)
        from .laboratory_liquid import LaboratoryLiquid
        from .task_observations import CatalogueReadings
        from .task_evaluators import CatalogueEvaluator
        from .task_evaluators import motor_rpm
        session.magnet.rpm = motor_rpm(setup['dial_rad'])
        # Preserve the original exact-timestep check skipped by the dry constructor.
        if numerics.dt_s != session.dt_s or numerics.device != config.device:
            raise ValueError('Rigid/liquid timestep and device must agree exactly; no substep substitution')
        session.liquid = LaboratoryLiquid(session.world, numerics, Reagents(**setup['reagents']))
        precharge = getattr(folder, 'precharge_chemistry', None)
        if precharge is not None:          # a flask that starts part-titrated (finish-titration)
            precharge(session.liquid, setup)
        session.task_readings = CatalogueReadings(session.liquid, setup)
        initial = session.task_readings.read()
        # Rebase against the represented float state, so tick zero is exactly 0.
        setup['distance0_m'] = initial['flask_distance_m']
        rebase = getattr(folder, 'rebase', None)
        if rebase is not None:
            rebase(setup, initial)
        session.evaluator = CatalogueEvaluator(task, session.dt_s, reagents, setup)
        result = session.evaluator.update(0, **initial)
        if result['success'] or result['score'] != 0. or not result['valid']:
            raise ValueError('Task preset must be a valid, unsuccessful, zero-score starting state')
        session.task_setup = setup
        session._initial = session.snapshot()
        return session
    except BaseException:
        session.close()
        raise

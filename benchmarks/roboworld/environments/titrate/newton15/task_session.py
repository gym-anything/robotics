"""GAR's task session factory; no changes to physics-hashed source modules."""

from .task_catalogue import task_name
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


def precharge_chemistry(liquid, setup):
    """A prepared, mixed starting sample, not a reaction-time intervention.

    Volume moved from burette to flask was included in the ordinary lattice
    initialization. Preserve original aliquot, tracer, all sodium/chloride and
    analytical H/OH totals; initialize the already reacted amount explicitly.
    """
    import numpy as np
    base0 = setup['base0_ml']
    if not base0:
        return
    from .liquid_charge import Reagents
    r = Reagents(**setup['nominal_reagents'])
    v0 = r.aliquot_ml+r.dilution_ml
    volume = v0+base0
    first, last = liquid.initial_groups['flask']
    chem = liquid.chemistry
    chem.charge(slice(first, last), acid_m=r.acid_molar*r.aliquot_ml/volume,
                base_m=r.base_molar*base0/volume, indicator_m=r.indicator_molar*v0/volume)
    liquid.initial_totals = chem.observe()['totals_mol']
    n = chem.moles.numpy()
    reacted = np.minimum(n[:, 0], n[:, 1])
    n[:, 0] -= reacted
    n[:, 1] -= reacted
    chem.moles.assign(n)
    chem.next.assign(n)
    chem.reacted.assign(reacted)
    chem.next_reacted.assign(reacted)


def make_session(config=None, *, task='titration', **kwargs):
    from .session import ApparatusSession
    task_name(task)
    if task == 'titration':
        return ApparatusSession(config, **kwargs)
    from .lab_config import LabConfig
    from .liquid_charge import Reagents
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
        precharge_chemistry(session.liquid, setup)
        session.task_readings = CatalogueReadings(session.liquid, setup)
        initial = session.task_readings.read()
        # Rebase against the represented float state, so tick zero is exactly 0.
        setup['distance0_m'] = initial['flask_distance_m']
        if task == 'finish-titration':
            setup['base0_ml'] = initial['base_in_flask_ml']
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

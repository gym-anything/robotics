"""Native Newton non-convex contacts and MuJoCo Warp integration only."""

from .runtime import require_engine


def make_solver(model, c):
    """Use the same integration on construction and reset.

    The pinned full-implicit path needs the RNE bias derivative sign correction
    documented in implicit_solver.py. Contact damping is separate in scene.py.
    """
    from .implicit_solver import CorrectedImplicitSolver
    return CorrectedImplicitSolver(model, use_mujoco_contacts=False,
        integrator="implicit", cone="elliptic", solver="newton",
        iterations=c.solver_iterations, ls_iterations=30,
        nconmax=c.contact_capacity, njmax=c.contact_capacity*3)


def create(builder, c, joints=None):
    newton, wp = require_engine()
    from .nub_contacts import filter_pairs, NubPipeline
    pairs = filter_pairs(builder)
    with wp.ScopedDevice(c.device):
        model = builder.finalize(device=c.device)
        model.request_contact_attributes("force")
        # Builder defaults retain MuJoCo's acceleration-space convention. The
        # baseline k and d are forces per displacement/velocity, so opt into
        # Newton 1.5's explicit mass-aware FORCE_SPACE conversion.
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE
        model.mujoco.solref_mode.fill_(SOLREF_MODE_FORCE_SPACE)
        import numpy as np
        # The rail endpoint is a hard mechanical stop. Apply maximum native
        # constraint impedance only to the four Panda finger limits, activated
        # over the two-tick response horizon at their real 0.2 m/s speed limit.
        limit_impedance = model.mujoco.solimplimit.numpy()
        limit_margin = model.mujoco.limit_margin.numpy()
        for name, spec in (joints or {}).items():
            if name.endswith(("panda_finger_joint1", "panda_finger_joint2")):
                dof = spec["dof"]
                limit_impedance[dof, :2] = .9999
                limit_margin[dof] = 2.0 * spec["velocity_limit"] * c.dt
        model.mujoco.solimplimit.assign(limit_impedance)
        model.mujoco.limit_margin.assign(limit_margin)
        # A compliant nub/pad supplies pair compliance when its mate is rigid,
        # as in PhysX. Arithmetic averaging with the hard shell would erase it.
        priorities = np.zeros(model.shape_count, np.int32)
        for i, label in enumerate(builder.shape_label):
            body = builder.shape_body[i]
            if "cap_nub" in label:
                priorities[i] = 10
            elif body >= 0 and "finger" in builder.body_label[body]:
                priorities[i] = 5
        model.mujoco.geom_priority.assign(priorities)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=c.contact_capacity,
            broad_phase="sap", reduce_contacts=True, contact_matching="latest", contact_report=True)
        pipeline = NubPipeline(pipeline, model, pairs)
        solver = make_solver(model, c)
        a, b = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, a)
        b.assign(a)
    return model, pipeline, solver, a, b, model.control(), pipeline.contacts()

"""Declared numerical rigid-contact compliance, independent of water physics.

Newton's builder ke/kd defaults are not laboratory glass/PTFE material data.
The CPU MuJoCo backend interprets its direct solref in acceleration space.
Use explicit critically damped rigid-limit constraints, then measure their
penetration under the actual masses and loads in the component fixtures.
These are numerical contact parameters, NOT measured elastic moduli.
"""

import numpy as np
from .runtime import require_newton

CONTACT_TIME_CONSTANT_S = .002
CONTACT_DAMPING_RATIO = 1.0
CCD_TOLERANCE = 1.0e-10
CCD_ITERATIONS = 100
CONTACT_SEARCH_GAP_M = .0001


def register(builder):
    newton, _ = require_newton()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)


def create_solver(model):
    """One declared Newton CPU rigid configuration for probes and runtime."""
    newton, _ = require_newton()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True,
        use_mujoco_contacts=True, ccd_tolerance=CCD_TOLERANCE,
        ccd_iterations=CCD_ITERATIONS, enable_multiccd=True)
    # Newton 1.5's post-compile inertial-axis synchronization leaves CPU
    # collision BVHs in their former frame (upstream issue #3805). Reuse the
    # existing whiteboard geometric refit; shapes, masses and contacts remain
    # unchanged. It restores the accelerator to the actual geometry.
    from whiteboard.cpu_collision import refit_cpu_bvh

    solver.titration_collision_refit = refit_cpu_bvh(solver.mj_model)
    return solver


def configure(model):
    _, wp = require_newton()
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW

    # Explicit ShapeConfig(...) does not inherit builder.default_shape_cfg.
    # Its default gap is 0.1 m, inappropriate for millimetre glass wall cells:
    # it creates enormous inactive-contact lists. This is a contact-search
    # distance, not an enlarged/softened collision shape or penetration limit.
    model.shape_gap.fill_(CONTACT_SEARCH_GAP_M)
    model.mujoco.solref.fill_(wp.vec2(CONTACT_TIME_CONSTANT_S, CONTACT_DAMPING_RATIO))
    model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
    values = np.tile([.99, .999, .0001, .5, 2.], (model.shape_count, 1)).astype(np.float32)
    model.mujoco.geom_solimp.assign(values)

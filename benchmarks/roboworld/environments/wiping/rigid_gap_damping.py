# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton 1.5 rigid contact kernels with explicit compression-rate damping.

Three functions derive from Newton with a marked compression-rate correction.
For hard contacts, initial separation is not stabilized as a penetration error.
Friction, levers, buffers and scheduling remain native. Both the solver and
force-export kernels call the same corrected law.
Opt-in only; no installed-package edits or global monkeypatching.
Native source SHA256: 0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f
"""
import hashlib
from pathlib import Path
from types import FunctionType

import warp as wp
from newton.solvers import SolverVBD
from newton._src.sim.contacts import contact_surface_point, contact_surface_separation
from newton._src.solvers.vbd import rigid_vbd_kernels as native
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _SMALL_LENGTH_EPS, _NUM_CONTACT_THREADS_PER_BODY,
    compute_projected_isotropic_friction,
)

wp.set_module_options({"enable_backward": False})
NATIVE_SHA256 = "0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f"

@wp.func
def evaluate_rigid_contact_from_collision(
    body_a_index: int,
    body_b_index: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    contact_point_a_local: wp.vec3,
    contact_point_b_local: wp.vec3,
    contact_offset_a_local: wp.vec3,
    contact_offset_b_local: wp.vec3,
    contact_normal: wp.vec3,
    penetration_depth: float,
    geometric_depth: float,
    contact_ke: float,
    contact_ke_t: float,
    contact_kd: float,
    contact_lam: wp.vec3,
    friction_mu: float,
    friction_epsilon: float,
    hard_contact: int,
    dt: float,
    friction_c0: wp.vec3,
):
    """Compute augmented-Lagrangian contact forces and 3x3 Hessian blocks for a rigid contact pair.

    Hard contacts: ALM normal + displacement-based tangential friction with Coulomb cone clamping.
    The tangential constraint is the relative tangential displacement from body_q_prev to body_q,
    which correctly captures kinematic body motion.
    Soft contacts: velocity-based IPC friction with scalar penalty.

    Returns:
        10-tuple: (force_a, torque_a, H_ll_a, H_al_a, H_aa_a,
                   force_b, torque_b, H_ll_b, H_al_b, H_aa_b)
    """
    lam_n = wp.dot(contact_lam, contact_normal)

    if penetration_depth <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    f_n = contact_ke * penetration_depth + lam_n
    if contact_ke <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)
    f_n = wp.max(f_n, 0.0)

    if f_n == 0.0 and hard_contact == 0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    if body_a_index < 0:
        X_wa = wp.transform_identity()
        X_wa_prev = wp.transform_identity()
        body_a_com_local = wp.vec3(0.0)
    else:
        X_wa = body_q[body_a_index]
        X_wa_prev = body_q_prev[body_a_index]
        body_a_com_local = body_com[body_a_index]

    if body_b_index < 0:
        X_wb = wp.transform_identity()
        X_wb_prev = wp.transform_identity()
        body_b_com_local = wp.vec3(0.0)
    else:
        X_wb = body_q[body_b_index]
        X_wb_prev = body_q_prev[body_b_index]
        body_b_com_local = body_com[body_b_index]

    x_com_a_now = wp.transform_point(X_wa, body_a_com_local)
    x_com_b_now = wp.transform_point(X_wb, body_b_com_local)

    # Normal response uses geometric (skeleton) points; friction uses the surface anchor.
    x_s_a_now = wp.transform_point(X_wa, contact_point_a_local)
    x_s_b_now = wp.transform_point(X_wb, contact_point_b_local)
    x_s_a_prev = wp.transform_point(X_wa_prev, contact_point_a_local)
    x_s_b_prev = wp.transform_point(X_wb_prev, contact_point_b_local)

    x_c_a_now = contact_surface_point(X_wa, contact_point_a_local, contact_offset_a_local)
    x_c_b_now = contact_surface_point(X_wb, contact_point_b_local, contact_offset_b_local)
    x_c_a_prev = contact_surface_point(X_wa_prev, contact_point_a_local, contact_offset_a_local)
    x_c_b_prev = contact_surface_point(X_wb_prev, contact_point_b_local, contact_offset_b_local)

    n_outer = wp.outer(contact_normal, contact_normal)
    I3 = wp.identity(n=3, dtype=float)

    # Normal approach rate from the geometric points (not the rotating anchor).
    v_rel_n = (x_s_b_now - x_s_b_prev - x_s_a_now + x_s_a_prev) / dt
    v_dot_n = wp.dot(contact_normal, v_rel_n)

    # Backward difference of actual *compression*, not flight across an air
    # gap and not the stabilized AL residual. With a previously separated
    # contact this ramps continuously from zero, rather than switching on a
    # kd*flight_speed impulse at infinitesimal penetration.
    previous_depth = geometric_depth + dt * v_dot_n
    v_dot_n = (wp.max(previous_depth, 0.0) - wp.max(geometric_depth, 0.0)) / dt

    # Tangential slip from the surface anchor (required for finite-radius friction).
    v_rel_t = (x_c_b_now - x_c_b_prev - x_c_a_now + x_c_a_prev) / dt
    v_t = v_rel_t - contact_normal * wp.dot(contact_normal, v_rel_t)

    # Normal block (force + optional approach damping), applied at the geometric lever.
    f_n_vec = contact_normal * f_n
    K_n = contact_ke * n_outer

    # Tangential friction block, applied at the surface-anchor lever.
    f_t_vec = wp.vec3(0.0)
    K_t = wp.mat33(0.0)

    if hard_contact == 1:
        if friction_mu > 0.0 and f_n > 0.0:
            # ALM tangential friction with Coulomb cone clamping.
            # Tangential constraint: rel_disp + friction_c0
            # (friction_c0 = (1 - alpha) * C0_t, pre-scaled by the caller).
            tangential_disp = -(v_t * dt)
            lam_t = contact_lam - contact_normal * lam_n
            f_t_vec = contact_ke_t * (tangential_disp + friction_c0) + lam_t
            f_t_len = wp.length(f_t_vec)
            cone_limit = friction_mu * f_n
            if f_t_len > cone_limit and f_t_len > 0.0:
                cone_ratio = cone_limit / f_t_len
                f_t_vec = f_t_vec * cone_ratio
            K_t = contact_ke_t * (I3 - n_outer)
    else:
        # Soft contact: IPC velocity-based friction.
        if friction_mu > 0.0 and f_n > 0.0:
            f_friction, K_friction = compute_projected_isotropic_friction(
                friction_mu, f_n, contact_normal, v_t * dt, friction_epsilon * dt
            )
            f_t_vec = f_friction
            K_t = K_friction

    if contact_kd > 0.0 and v_dot_n < 0.0 and f_n > 0.0:
        f_n_vec = f_n_vec - contact_kd * v_dot_n * contact_normal
        K_n = K_n + (contact_kd / dt) * n_outer

    f_total = f_n_vec + f_t_vec
    K_total = K_n + K_t

    # Geometric lever for the normal block, surface-anchor lever for friction.
    r_s_a = x_s_a_now - x_com_a_now
    r_c_a = x_c_a_now - x_com_a_now
    r_s_b = x_s_b_now - x_com_b_now
    r_c_b = x_c_b_now - x_com_b_now

    r_s_a_skew_T = wp.transpose(wp.skew(r_s_a))
    r_c_a_skew_T = wp.transpose(wp.skew(r_c_a))
    h_al_a = -(r_s_a_skew_T * K_n + r_c_a_skew_T * K_t)
    h_aa_a = r_s_a_skew_T * K_n * wp.skew(r_s_a) + r_c_a_skew_T * K_t * wp.skew(r_c_a)

    r_s_b_skew_T = wp.transpose(wp.skew(r_s_b))
    r_c_b_skew_T = wp.transpose(wp.skew(r_c_b))
    h_al_b = -(r_s_b_skew_T * K_n + r_c_b_skew_T * K_t)
    h_aa_b = r_s_b_skew_T * K_n * wp.skew(r_s_b) + r_c_b_skew_T * K_t * wp.skew(r_c_b)

    torque_a = wp.cross(r_s_a, -f_n_vec) + wp.cross(r_c_a, -f_t_vec)
    torque_b = wp.cross(r_s_b, f_n_vec) + wp.cross(r_c_b, f_t_vec)

    return (
        -f_total,
        torque_a,
        K_total,
        h_al_a,
        h_aa_a,
        f_total,
        torque_b,
        K_total,
        h_al_b,
        h_aa_b,
    )

@wp.kernel
def accumulate_body_body_contacts_per_body(
    dt: float,
    color_group: wp.array[wp.int32],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    friction_epsilon: float,
    contact_penalty_k: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    avbd_alpha: float,
    hard_contacts: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body augmented-Lagrangian contact accumulation with _NUM_CONTACT_THREADS_PER_BODY strided threads.
    """
    tid = wp.tid()
    body_idx_in_group = tid // _NUM_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return

    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return

    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_buffer_pre_alloc:
        num_contacts = body_contact_buffer_pre_alloc

    contact_count = rigid_contact_count[0]

    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)

    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_buffer_pre_alloc + i]
        if contact_idx >= contact_count:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1

        if b0 != body_id and b1 != body_id:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        cp0_offset_local = rigid_contact_offset0[contact_idx]
        cp1_offset_local = rigid_contact_offset1[contact_idx]
        contact_normal = rigid_contact_normal[contact_idx]
        # Normal C_n uses the unprojected (skeleton) points: ``thickness`` already accounts
        # for the radial extent, so adding the offset here would double-count it.
        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        C_n = -contact_surface_separation(
            cp0_world, cp1_world, contact_normal, rigid_contact_margin0[contact_idx], rigid_contact_margin1[contact_idx]
        )

        lam_n = float(0.0)
        C_eff = C_n
        lam_vec = wp.vec3(0.0)
        k = contact_penalty_k[contact_idx]
        friction_c0 = wp.vec3(0.0)

        if hard_contacts == 1:
            lam_vec = contact_lambda[contact_idx]
            lam_n = wp.dot(lam_vec, contact_normal)
            C0_vec = contact_C0[contact_idx]
            C0_n = wp.dot(contact_normal, C0_vec)
            # Hard-contact stabilization: normal uses C_n - alpha*C0_n; tangent caches
            # (1 - alpha)*C0_t for the later tangential update.
            C_eff = C_n - avbd_alpha * C0_n
            friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)

        if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        f_n_check = k * C_eff + lam_n
        if f_n_check <= 0.0 and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]

        (
            force_0,
            torque_0,
            h_ll_0,
            h_al_0,
            h_aa_0,
            force_1,
            torque_1,
            h_ll_1,
            h_al_1,
            h_aa_1,
        ) = evaluate_rigid_contact_from_collision(
            b0,
            b1,
            body_q,
            body_q_prev,
            body_com,
            cp0_local,
            cp1_local,
            cp0_offset_local,
            cp1_offset_local,
            contact_normal,
            C_eff,
            C_n,
            k,
            k,
            contact_kd,
            lam_vec,
            contact_mu,
            friction_epsilon,
            hard_contacts,
            dt,
            friction_c0,
        )

        if body_id == b0:
            force_acc += force_0
            torque_acc += torque_0
            h_ll_acc += h_ll_0
            h_al_acc += h_al_0
            h_aa_acc += h_aa_0
        else:
            force_acc += force_1
            torque_acc += torque_1
            h_ll_acc += h_ll_1
            h_al_acc += h_al_1
            h_aa_acc += h_aa_1

        i += _NUM_CONTACT_THREADS_PER_BODY

    wp.atomic_add(body_forces, body_id, force_acc)
    wp.atomic_add(body_torques, body_id, torque_acc)
    wp.atomic_add(body_hessian_ll, body_id, h_ll_acc)
    wp.atomic_add(body_hessian_al, body_id, h_al_acc)
    wp.atomic_add(body_hessian_aa, body_id, h_aa_acc)

@wp.kernel
def compute_rigid_contact_forces(
    dt: float,
    # Contact data
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    # Model/state
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # Contact material properties (per-contact)
    contact_penalty_k: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    avbd_alpha: float,
    hard_contacts: int,
    friction_epsilon: float,
    # Outputs (length = rigid_contact_max)
    out_body0: wp.array[wp.int32],
    out_body1: wp.array[wp.int32],
    out_point0_world: wp.array[wp.vec3],
    out_point1_world: wp.array[wp.vec3],
    out_force_on_body1: wp.array[wp.vec3],
):
    """Compute per-contact forces in world space (hard: ALM, soft: penalty)."""
    contact_idx = wp.tid()

    rc = rigid_contact_count[0]
    if contact_idx >= rc:
        # Fill sentinel values for inactive entries (useful when launching with rigid_contact_max)
        out_body0[contact_idx] = wp.int32(-1)
        out_body1[contact_idx] = wp.int32(-1)
        out_point0_world[contact_idx] = wp.vec3(0.0)
        out_point1_world[contact_idx] = wp.vec3(0.0)
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    s0 = rigid_contact_shape0[contact_idx]
    s1 = rigid_contact_shape1[contact_idx]
    if s0 < 0 or s1 < 0:
        out_body0[contact_idx] = wp.int32(-1)
        out_body1[contact_idx] = wp.int32(-1)
        out_point0_world[contact_idx] = wp.vec3(0.0)
        out_point1_world[contact_idx] = wp.vec3(0.0)
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    out_body0[contact_idx] = b0
    out_body1[contact_idx] = b1

    cp0_local = rigid_contact_point0[contact_idx]
    cp1_local = rigid_contact_point1[contact_idx]
    cp0_offset_local = rigid_contact_offset0[contact_idx]
    cp1_offset_local = rigid_contact_offset1[contact_idx]
    contact_normal = rigid_contact_normal[contact_idx]

    # Normal C_n uses the unprojected (skeleton) points: ``thickness`` already accounts
    # for the radial extent, so adding the offset here would double-count it.
    cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
    cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
    out_point0_world[contact_idx] = (
        wp.transform_point(body_q[b0], cp0_local + cp0_offset_local) if b0 >= 0 else cp0_local + cp0_offset_local
    )
    out_point1_world[contact_idx] = (
        wp.transform_point(body_q[b1], cp1_local + cp1_offset_local) if b1 >= 0 else cp1_local + cp1_offset_local
    )

    C_n = -contact_surface_separation(
        cp0_world, cp1_world, contact_normal, rigid_contact_margin0[contact_idx], rigid_contact_margin1[contact_idx]
    )

    lam_n = float(0.0)
    C_eff = C_n
    lam_vec = wp.vec3(0.0)
    k = contact_penalty_k[contact_idx]
    friction_c0 = wp.vec3(0.0)

    if hard_contacts == 1:
        lam_vec = contact_lambda[contact_idx]
        lam_n = wp.dot(lam_vec, contact_normal)
        C0_vec = contact_C0[contact_idx]
        C0_n = wp.dot(contact_normal, C0_vec)
        # Hard-contact stabilization: normal uses C_n - alpha*C0_n; tangent caches
        # (1 - alpha)*C0_t for the later tangential update.
        C_eff = C_n - avbd_alpha * C0_n
        friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)

    f_n_check = k * C_eff + lam_n
    if (C_n <= _SMALL_LENGTH_EPS or f_n_check <= 0.0) and lam_n <= 0.0:
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    contact_kd = contact_material_kd[contact_idx]
    contact_mu = contact_material_mu[contact_idx]

    (
        _force_0,
        _torque_0,
        _h_ll_0,
        _h_al_0,
        _h_aa_0,
        force_1,
        _torque_1,
        _h_ll_1,
        _h_al_1,
        _h_aa_1,
    ) = evaluate_rigid_contact_from_collision(
        int(b0),
        int(b1),
        body_q,
        body_q_prev,
        body_com,
        cp0_local,
        cp1_local,
        cp0_offset_local,
        cp1_offset_local,
        contact_normal,
        C_eff,
        C_n,
        k,
        k,
        contact_kd,
        lam_vec,
        contact_mu,
        friction_epsilon,
        hard_contacts,
        dt,
        friction_c0,
    )

    out_force_on_body1[contact_idx] = force_1


@wp.kernel
def remove_initial_air_gap(count: wp.array[int], normal: wp.array[wp.vec3],
                           initial: wp.array[wp.vec3]):
    i = wp.tid()
    if i >= count[0]:return
    n = normal[i]
    d = wp.dot(initial[i], n)
    if d < 0.0:initial[i] = initial[i] - d*n


def redirect_rigid_method(method):
    """Compose with the independent cloth compression law, if selected."""
    fn = FunctionType(method.__code__, method.__globals__ | {
        'accumulate_body_body_contacts_per_body': accumulate_body_body_contacts_per_body},
        method.__name__, method.__defaults__, method.__closure__)
    fn.__kwdefaults__ = method.__kwdefaults__
    return fn


def redirected_methods():
    """Bind private Python launch copies; never change native module globals."""
    actual = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if actual != NATIVE_SHA256:
        raise RuntimeError("Gap-aware damping requires the audited Newton 1.5 kernel source")
    def redirect(method, replacements):
        fn = FunctionType(method.__code__, method.__globals__ | replacements,
                          method.__name__, method.__defaults__, method.__closure__)
        fn.__kwdefaults__ = method.__kwdefaults__
        return fn
    return (
        redirect(SolverVBD._solve_rigid_body_iteration,
                 {"accumulate_body_body_contacts_per_body": accumulate_body_body_contacts_per_body}),
        redirect(SolverVBD.collect_rigid_contact_forces,
                 {"compute_rigid_contact_forces": compute_rigid_contact_forces}),
    )

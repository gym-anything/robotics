"""Internal WSG belt constraint integrated into native VBD; no object attachment.

rho is a numerical iteration metric, not a physical spring. See WSG_NUMERICS.md.
"""

import ast
import inspect
import textwrap
import types
import numpy as np
import warp as wp
from .coordinates import coordinate, joint_pairs


wp.set_module_options({"enable_backward": False})


@wp.func
def contribution(
    body: int,
    pairs: wp.array2d[int],
    lam: wp.array[float],
    rho: float,
    q: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    parents: wp.array[int],
    children: wp.array[int],
    xp: wp.array[wp.transform],
    xc: wp.array[wp.transform],
    starts: wp.array[int],
    axes: wp.array[wp.vec3],
):
    f = wp.vec3(0.0)
    t = wp.vec3(0.0)
    hll = wp.mat33(0.0)
    hal = wp.mat33(0.0)
    haa = wp.mat33(0.0)
    for k in range(pairs.shape[0]):
        j0 = pairs[k, 0]
        j1 = pairs[k, 1]
        if body == parents[j0] or body == children[j0] or body == children[j1]:
            c0, gp0, gc0 = coordinate(
                j0, q, com, parents, children, xp, xc, starts, axes
            )
            c1, gp1, gc1 = coordinate(
                j1, q, com, parents, children, xp, xc, starts, axes
            )
            g = gp0 - gp1
            if body == children[j0]:
                g = gc0
            elif body == children[j1]:
                g = -gc1
            gl = wp.spatial_top(g)
            ga = wp.spatial_bottom(g)
            scalar = lam[k] + rho * (c0 - c1)
            f -= scalar * gl
            t -= scalar * ga
            hll += rho * wp.outer(gl, gl)
            hal += rho * wp.outer(ga, gl)
            haa += rho * wp.outer(ga, ga)
    return f, t, hll, hal, haa


@wp.kernel
def dual(
    pairs: wp.array2d[int],
    lam: wp.array[float],
    rho: float,
    q: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    parents: wp.array[int],
    children: wp.array[int],
    xp: wp.array[wp.transform],
    xc: wp.array[wp.transform],
    starts: wp.array[int],
    axes: wp.array[wp.vec3],
    error: wp.array[float],
):
    k = wp.tid()
    c0, gp0, gc0 = coordinate(
        pairs[k, 0], q, com, parents, children, xp, xc, starts, axes
    )
    c1, gp1, gc1 = coordinate(
        pairs[k, 1], q, com, parents, children, xp, xc, starts, axes
    )
    error[k] = c0 - c1
    lam[k] += rho * (c0 - c1)


class Mechanism:
    def __init__(
        self, solver, pairs=None, rho=10000.0, *, compile_tree, schedule_plan=None
    ):
        self.compile_tree = compile_tree
        self.schedule_plan = schedule_plan
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError("Positive ALM metric required")
        self.s = solver
        self.pairs = joint_pairs(solver.model, pairs)
        self.rho = float(rho)
        self.lam = wp.zeros(self.pairs.shape[0], dtype=float, device=solver.device)
        self.error = wp.zeros_like(self.lam)

    def update(self, state):
        m = self.s.model
        wp.launch(
            dual,
            self.pairs.shape[0],
            [
                self.pairs,
                self.lam,
                self.rho,
                state.body_q,
                m.body_com,
                m.joint_parent,
                m.joint_child,
                m.joint_X_p,
                m.joint_X_c,
                m.joint_qd_start,
                m.joint_axis,
                self.error,
            ],
            device=m.device,
        )

    def install(self):
        s = self.s
        compile_tree = self.compile_tree
        if hasattr(s, "wsg_synchronizer"):
            raise RuntimeError("WSG synchronizer already installed")
        method = s._solve_rigid_body_iteration.__func__
        if hasattr(s, "_rigid_schedule"):
            # The internal belt creates an extra read dependency between jaws.
            # Add it to scheduling only; do not add a fictitious physics joint.
            plan = self.schedule_plan
            if plan is None:
                raise ValueError(
                    "Dependency schedule requires an explicit schedule kernel"
                )
            m = s.model
            schedule = s._rigid_schedule
            pairs = self.pairs.numpy()
            child = m.joint_child.numpy()
            self.schedule_parent = wp.array(
                np.r_[m.joint_parent.numpy(), child[pairs[:, 0]]],
                dtype=int,
                device=m.device,
            )
            self.schedule_child = wp.array(
                np.r_[m.joint_child.numpy(), child[pairs[:, 1]]],
                dtype=int,
                device=m.device,
            )
            self.schedule_enabled = wp.array(
                np.r_[m.joint_enabled.numpy(), np.ones(len(pairs), bool)],
                dtype=bool,
                device=m.device,
            )

            def prepare(owner, contacts):
                wp.launch(
                    plan,
                    1,
                    [
                        m.body_count,
                        s.body_body_contact_counts,
                        s.body_body_contact_indices,
                        s.body_body_contact_buffer_pre_alloc,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        m.shape_body,
                        self.schedule_parent,
                        self.schedule_child,
                        self.schedule_enabled,
                        owner.levels,
                        owner.maximum,
                        owner.current,
                        owner.condition,
                    ],
                    device=m.device,
                )

            schedule.prepare = types.MethodType(prepare, schedule)
        elif any(len(group) > 1 for group in s.model.body_color_groups):
            raise ValueError(
                "Use serial singleton body groups or dependency-aware schedule for belt"
            )
        kernel = method.__globals__["solve_rigid_body"]
        tree = ast.parse(textwrap.dedent(inspect.getsource(kernel.func)))
        fn = tree.body[0]
        fn.name = "solve_rigid_body_with_belt_alm"
        args = (
            ast.parse(
                "def p(belt_pairs:wp.array2d[int],belt_lam:wp.array[float],belt_rho:float):pass"
            )
            .body[0]
            .args.args
        )
        fn.args.args[-1:-1] = args
        indices = [
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "trA"
        ]
        if len(indices) != 1:
            raise RuntimeError("Unexpected native rigid solve structure")
        fn.body[indices[0] : indices[0]] = (
            ast.parse("""belt_f,belt_t,belt_hll,belt_hal,belt_haa=contribution(body_index,belt_pairs,belt_lam,belt_rho,body_q,body_com,joint_parent,joint_child,joint_X_p,joint_X_c,joint_qd_start,joint_axis)
f_force+=belt_f
f_torque+=belt_t
h_ll+=belt_hll
h_al+=belt_hal
h_aa+=belt_haa
""").body
        )
        ns, kernel_source = compile_tree(
            tree,
            dict(kernel.func.__globals__, contribution=contribution),
            "belt-alm-kernel",
        )
        replacement = ns[fn.name]
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keys = {k.arg: k.value for k in node.keywords}
            if "kernel" in keys and ast.unparse(keys["kernel"]) == "solve_rigid_body":
                keys["inputs"].elts += ast.parse(
                    "[self.wsg_synchronizer.pairs,self.wsg_synchronizer.lam,self.wsg_synchronizer.rho]",
                    mode="eval",
                ).body.elts
                count += 1
        if count != 1:
            raise RuntimeError("Expected native or scheduled, not fused, rigid launch")
        ns, method_source = compile_tree(
            tree,
            dict(method.__globals__, solve_rigid_body=replacement),
            "belt-alm-method",
        )
        original = types.MethodType(ns[tree.body[0].name], s)
        initialize = s._initialize_rigid_bodies

        def init(owner, *args, **kwargs):
            self.lam.zero_()
            return initialize(*args, **kwargs)

        def iteration(owner, state_in, state_out, control, contacts, dt):
            original(state_in, state_out, control, contacts, dt)
            self.update(state_in)

        s.wsg_synchronizer = self
        s._initialize_rigid_bodies = types.MethodType(init, s)
        s._solve_rigid_body_iteration = types.MethodType(iteration, s)
        self.sources = {"kernel": kernel_source, "method": method_source}
        return self

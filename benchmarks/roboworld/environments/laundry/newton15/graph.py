"""Capture the unchanged native PCG solve, not a different physics algorithm.

CUDA capture records the original kernel/reduction/copy sequence exactly once.
No warm-up simulation step, iteration reduction, convergence change or skipping.
Arrays are retained with each graph; changing an input pointer builds a new graph.
"""
import warp as wp


class CapturedPCG:
    def __init__(self, native):
        self.native, self.graphs = native, {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def solve(self, A, diagonal, x0, b, inv_M, x1, iterations, additional_multiplier=None):
        device = wp.get_device(self.native.device)
        if not device.is_cuda or device.is_capturing:
            return self.native.solve(A, diagonal, x0, b, inv_M, x1, iterations, additional_multiplier)
        arrays = (A.num_nz, A.nz_ell, diagonal, x0, b, inv_M, x1)
        callback = getattr(additional_multiplier, "__self__", additional_multiplier)
        key = (*[None if a is None else a.ptr for a in arrays], int(iterations), id(callback))
        if key not in self.graphs:
            with wp.ScopedCapture(device=self.native.device) as capture:
                self.native.solve(A, diagonal, x0, b, inv_M, x1, iterations, additional_multiplier)
            self.graphs[key] = (capture.graph, arrays, callback)
        wp.capture_launch(self.graphs[key][0])


class CapturedClothStep:
    """Capture the native cloth integration; retain exact kernel order/counts."""
    def __init__(self, native):
        self.native, self.graphs = native, {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def record_graph(self, state_in, state_out, control, contacts, dt):
        """Capture the graph for these arguments once, without launching it."""
        key = (id(state_in), id(state_out), id(contacts), float(dt))
        if key not in self.graphs:
            with wp.ScopedCapture(device=self.native.device) as capture:
                self.native.step(state_in, state_out, control, contacts, dt)
            self.graphs[key] = capture.graph
        return self.graphs[key]

    def step(self, state_in, state_out, control, contacts, dt):
        if not wp.get_device(self.native.device).is_cuda:
            return self.native.step(state_in, state_out, control, contacts, dt)
        wp.capture_launch(self.record_graph(state_in, state_out, control, contacts, dt))


class CapturedRigidStep:
    """Native MuJoCo-Warp integration, with its Python step counter preserved."""
    def __init__(self, native):
        if native.update_data_interval != 1:
            raise ValueError("Rigid graph requires native state synchronization every step")
        self.native, self.graphs = native, {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def step(self, state_in, state_out, control, contacts, dt):
        if self.native.use_mujoco_cpu:
            return self.native.step(state_in, state_out, control, contacts, dt)
        key = (id(state_in), id(state_out), id(contacts), float(dt))
        if key not in self.graphs:
            previous = self.native._step
            with wp.ScopedCapture(device=self.native.model.device) as capture:
                self.native.step(state_in, state_out, control, contacts, dt)
            self.native._step = previous
            self.graphs[key] = capture.graph
        wp.capture_launch(self.graphs[key])
        self.native._step += 1


class CapturedCollision:
    """Capture the native fixed-inventory pipeline, including GPU generations."""
    def __init__(self, native):
        if native._contact_matcher is not None:
            raise ValueError("Captured collision does not support a Python-swapped contact matcher")
        self.native, self.graphs = native, {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def record_graph(self, state, contacts, **kwargs):
        """Capture the graph for these arguments once, without launching it."""
        key = (id(state), id(contacts), tuple(sorted(kwargs.items())))
        if key not in self.graphs:
            with wp.ScopedCapture(device=self.native.device) as capture:
                self.native.collide(state, contacts, **kwargs)
            self.graphs[key] = capture.graph
        return self.graphs[key]

    def collide(self, state, contacts, **kwargs):
        if not self.native.device.is_cuda:
            return self.native.collide(state, contacts, **kwargs)
        wp.capture_launch(self.record_graph(state, contacts, **kwargs))

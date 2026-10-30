"""Small GPU reductions for broad phase; no material-point subsampling."""

import numpy as np
import warp as wp


@wp.func
def component_x(p: wp.vec3):
    return p[0]


@wp.func
def component_y(p: wp.vec3):
    return p[1]


@wp.func
def component_z(p: wp.vec3):
    return p[2]


@wp.func
def minimum_valid(index: int, value: float, count: int, offset: int):
    return wp.where(index+offset < count, value, 1.e30)


@wp.func
def maximum_valid(index: int, value: float, count: int, offset: int):
    return wp.where(index+offset < count, value, -1.e30)


@wp.kernel
def bounds_blocks(q: wp.array[wp.vec3], output: wp.array2d[float]):
    block, lane = wp.tid()
    points = wp.tile_load(q, shape=256, offset=block*256)
    # Padded entries are excluded explicitly, not interpreted as the origin.
    idx = wp.tile_arange(256, dtype=int)
    x = wp.tile_map(component_x, points)
    y = wp.tile_map(component_y, points)
    z = wp.tile_map(component_z, points)
    lx = wp.tile_min(wp.tile_map(minimum_valid, idx, x, q.shape[0], block*256))
    ly = wp.tile_min(wp.tile_map(minimum_valid, idx, y, q.shape[0], block*256))
    lz = wp.tile_min(wp.tile_map(minimum_valid, idx, z, q.shape[0], block*256))
    hx = wp.tile_max(wp.tile_map(maximum_valid, idx, x, q.shape[0], block*256))
    hy = wp.tile_max(wp.tile_map(maximum_valid, idx, y, q.shape[0], block*256))
    hz = wp.tile_max(wp.tile_map(maximum_valid, idx, z, q.shape[0], block*256))
    if lane == 0:
        output[block, 0] = lx[0]
        output[block, 1] = ly[0]
        output[block, 2] = lz[0]
        output[block, 3] = hx[0]
        output[block, 4] = hy[0]
        output[block, 5] = hz[0]


class ParticleBounds:
    def __init__(self, count, device):
        self.blocks = wp.empty(((count+255)//256, 6), dtype=float, device=device)
        self.device = device
        self.total = wp.empty(6, dtype=float, device=device)

    def on_device(self, positions):
        wp.launch(bounds_blocks, dim=(len(self.blocks), 256), block_dim=256,
            inputs=[positions], outputs=[self.blocks], device=self.device)
        wp.launch(finish_bounds, dim=6, inputs=[self.blocks], outputs=[self.total], device=self.device)
        return self.total

    def __call__(self, positions):
        values = self.on_device(positions).numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite liquid broad-phase bounds")
        return values.reshape(2, 3)


@wp.kernel
def finish_bounds(blocks: wp.array2d[float], result: wp.array[float]):
    k = wp.tid()
    value = float(1.e30) if k < 3 else float(-1.e30)
    for i in range(blocks.shape[0]):
        value = wp.min(value, blocks[i, k]) if k < 3 else wp.max(value, blocks[i, k])
    result[k] = value

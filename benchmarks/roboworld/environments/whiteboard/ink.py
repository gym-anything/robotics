"""Sparse board-local pigment. Simulation time only; no screen-space drawing.

The optical-density mixture and deposition/erasure laws are explicit reduced
models, not a claim to reproduce the proprietary ink chemistry.
"""

import math
import numpy as np
from .config import COLORS


class InkSurface:
    tile_size = 128
    subcells = 4  # 4x4 persistent material samples per output texel, not AA.

    def __init__(self, board, width, height, materials):
        self.board, self.width, self.height, self.materials = (
            board,
            width,
            height,
            materials,
        )
        self.tiles = {}  # tile -> (H,W,16,6 pigment, H,W,16 deposition time)
        self.dirty = set()
        self.reservoir = {name: materials.ink_reservoir_ul for name in COLORS}
        self.eraser_load = np.zeros(6, dtype=np.float64)
        self.total_deposited_ul = 0.0
        self.total_removed_ul = 0.0
        self.revision = 0

    def uv(self, x, z):
        b = self.board
        return (
            (x / b.writing_width + 0.5) * self.width,
            (0.5 - (z - b.center_z) / b.writing_height) * self.height,
        )

    def _patch(self, center, axes, time):
        """Yield tiles and smooth elliptical footprint, clipped to porcelain only."""
        u, v = self.uv(*center)
        matrix = np.asarray(axes, float).copy()
        matrix[0] *= self.width / self.board.writing_width
        matrix[1] *= -self.height / self.board.writing_height
        if abs(np.linalg.det(matrix)) < 1e-8:
            return
        radius = np.sqrt((matrix * matrix).sum(axis=1)) + 1
        x0, x1 = (
            max(0, math.floor(u - radius[0])),
            min(self.width, math.ceil(u + radius[0])),
        )
        y0, y1 = (
            max(0, math.floor(v - radius[1])),
            min(self.height, math.ceil(v + radius[1])),
        )
        inv = np.linalg.inv(matrix)
        for ty in range(y0 // 128, (y1 + 127) // 128):
            for tx in range(x0 // 128, (x1 + 127) // 128):
                xa, xb = max(x0, tx * 128), min(x1, (tx + 1) * 128)
                ya, yb = max(y0, ty * 128), min(y1, (ty + 1) * 128)
                if xa >= xb or ya >= yb:
                    continue
                yy, xx = np.mgrid[ya:yb, xa:xb]
                # Nonlinear saturation must be integrated at persistent
                # material samples, NOT by averaging pressure before applying
                # the exponential update (those operations do not commute).
                coverage = []
                for oy in (0.125, 0.375, 0.625, 0.875):
                    for ox in (0.125, 0.375, 0.625, 0.875):
                        local = inv @ np.stack((xx.ravel() + ox - u, yy.ravel() + oy - v))
                        r = np.sqrt((local * local).sum(axis=0)).reshape(xx.shape)
                        coverage.append(np.clip((1 - r) * 4, 0, 1))
                coverage = np.stack(coverage, axis=2)
                if not coverage.any():
                    continue
                key = tx, ty
                if key not in self.tiles:
                    self.tiles[key] = (
                        np.zeros((128, 128, self.subcells**2, 6), np.float32),
                        np.full((128, 128, self.subcells**2), -1e6, np.float32),
                    )
                sl = np.s_[ya - ty * 128 : yb - ty * 128, xa - tx * 128 : xb - tx * 128]
                yield key, sl, coverage

    @property
    def pixel_volume_ul(self):
        area = self.board.writing_width * self.board.writing_height / (self.width * self.height)
        return area * self.materials.ink_film_um * 1e-6 * 1e9 / self.subcells**2

    def deposit(self, color, center, axes, force_n, dt, time):
        if color not in self.reservoir or self.reservoir[color] <= 0 or force_n <= 0 or dt <= 0:
            return 0.0
        channel = tuple(COLORS).index(color)
        deposited = 0.0
        for key, sl, mask in self._patch(center, axes, time):
            pigment, wet = self.tiles[key]
            target = pigment[sl][..., channel]
            # Saturating transfer from a finite reservoir. Resting does not flood indefinitely.
            amount = np.maximum(0, 1 - target) * (1 - np.exp(-dt * 350 * min(force_n, 1) * mask))
            volume = float(amount.sum()) * self.pixel_volume_ul
            if volume > self.reservoir[color]:
                amount *= self.reservoir[color] / volume
                volume = self.reservoir[color]
            if volume <= 0:
                continue
            target += amount
            wet[sl][amount > 0] = time
            self.reservoir[color] -= volume
            deposited += volume
            self.dirty.add(key)
        self.total_deposited_ul += deposited
        if deposited:
            self.revision += 1
        return deposited

    def erase(self, center, axes, force_n, travel_m, time):
        if force_n <= 0 or travel_m <= 0:
            return 0.0
        removed = 0.0
        for key, sl, mask in self._patch(center, axes, time):
            pigment, stamp = self.tiles[key]
            area = pigment[sl]
            wet = np.exp(-np.maximum(0, time - stamp[sl]) / self.materials.ink_dry_time_s)
            saturation = 1 / (1 + float(self.eraser_load.sum()) / 120)
            rate = travel_m / self.materials.erasure_distance_m * min(force_n / 2, 2) * saturation
            fraction = 1 - np.exp(-rate * (1 + 0.5 * wet) * mask)
            picked = area * fraction[..., None]
            area -= picked
            volume_by_color = picked.sum(axis=(0, 1, 2)) * self.pixel_volume_ul
            self.eraser_load += volume_by_color
            volume = float(volume_by_color.sum())
            removed += volume
            if volume:
                self.dirty.add(key)
        self.total_removed_ul += removed
        if removed:
            self.revision += 1
        return removed

    def rgb(self):
        image = np.full((self.height, self.width, 3), 246, np.uint8)
        pigments = np.asarray(list(COLORS.values()))
        absorption = -np.log(np.maximum(pigments, 0.012))
        for (tx, ty), (density, _) in self.tiles.items():
            h, w = min(128, self.height - ty * 128), min(128, self.width - tx * 128)
            rgb = (0.965 * np.exp(-np.einsum("hwqc,ck->hwqk", density[:h, :w], absorption))).mean(
                axis=2
            )
            image[ty * 128 : ty * 128 + h, tx * 128 : tx * 128 + w] = np.clip(
                rgb * 255, 0, 255
            ).astype(np.uint8)
        return image

    def pigment_volume_ul(self):
        return sum(
            float(a.sum(dtype=np.float64)) * self.pixel_volume_ul for a, _ in self.tiles.values()
        )

    def metadata(self):
        return {
            "revision": self.revision,
            "material_samples_per_texel": self.subcells**2,
            "material_sample_spacing_m": [
                self.board.writing_width / self.width / self.subcells,
                self.board.writing_height / self.height / self.subcells,
            ],
            "tiles": len(self.tiles),
            "allocated_bytes": sum(a.nbytes + t.nbytes for a, t in self.tiles.values()),
            "reservoir_ul": dict(self.reservoir),
            "eraser_load_ul": self.eraser_load.tolist(),
            "deposited_ul": self.total_deposited_ul,
            "removed_ul": self.total_removed_ul,
        }

"""Bounded presentation readback; callbacks must not advance simulation physics."""

import numpy as np


def capture_rgb(step_render, read_pixels, width, height, *, minimum_std=0.0, flush_attempts=2):
    if type(flush_attempts) is not int or not 1 <= flush_attempts <= 8:
        raise ValueError('Readback flush attempts must be in 1..8')
    invalid = 0
    last_shape = None
    for attempt in range(8):
        step_render()
        raw = np.asarray(read_pixels())
        last_shape = raw.shape
        valid = raw.ndim == 3 and raw.shape[:2] == (height, width) and raw.shape[2] >= 3
        if valid:
            rgb = raw[:, :, :3]
            valid = bool(rgb.max() > 0 and rgb.std() >= minimum_std)
        if not valid:
            invalid += 1
        elif attempt+1 >= flush_attempts:
            return rgb.copy(), invalid
    raise RuntimeError(f"No valid RGB readback after 8 render-only attempts; shape={last_shape}")

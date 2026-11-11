"""Constructed folds and rolls for the `laundry-tasks-v2` tests; never import a simulator.

Recorded geometry (`recorded`) and the v1 folds (`half_fold`, `thirds_fold`) come from
`task_set_fixtures`, unchanged. These are constructed shapes only: no reachability,
crossing or physical-retention claim.
"""
import numpy as np


def accordion(s, bends, heading=-1.):
    """Length-preserving centreline across a strip: flat plies joined at creases.

    s is arclength from the first hem; bends are (start, radius, +1 up / -1 down,
    rise). A positive radius is a semicircular U-bend; radius 0 is a sharp crease
    stepping the height by rise. Returns plan position and height along the fold axis.
    """
    u, z = np.zeros_like(s), np.zeros_like(s)
    u_b = z_b = s_b = 0.
    h = heading
    for start, radius, turn, rise in (*bends, (np.inf, 0., 0., 0.)):
        run = (s >= s_b) & (s < start)
        u[run], z[run] = u_b + h * (s[run] - s_b), z_b
        if not np.isfinite(start):
            break
        u_b += h * (start - s_b)
        if radius > 0:
            bend = (s >= start) & (s < start + np.pi * radius)
            t = (s[bend] - start) / radius
            u[bend], z[bend] = u_b + h * radius * np.sin(t), z_b + turn * radius * (1 - np.cos(t))
        z_b += turn * (2 * radius if radius > 0 else rise)
        s_b, h = start + np.pi * radius, -h
    return u, z


def letter_fold(x, widths=None, *, gap=.005, z_fold=False, rounded=True, table=.790):
    """Towel folded in thirds across its width (material x), plies `gap` apart.

    widths: plan widths of (left flap, middle, right flap); None is the equal letter
    fold. C: both flaps over the middle, the right one outermost. Z: left flap over,
    right flap under the middle. Constructed shapes only; no reachability or crossing claim.
    """
    ra, rb = ((gap / 2, gap / 2) if z_fold else (gap / 2, gap)) if rounded else (0., 0.)
    turns, rises = ((-1, -1), (gap, gap)) if z_fold else ((-1, 1), (gap, 2 * gap))
    s = x[:, 0] - x[:, 0].min()
    if widths is None:
        widths = ((s.max() - np.pi * (ra + rb)) / 3,) * 3
    left, middle, _ = widths
    u, z = accordion(s, ((left, ra, turns[0], rises[0]),
                         (left + np.pi * ra + middle, rb, turns[1], rises[1])))
    y = x.copy()
    y[:, 0], y[:, 2] = u - (u.min() + u.max()) / 2, z
    y[:, 2] += table - y[:, 2].min()
    return y


def quarter_fold(x, *, gap=.005, table=.790):
    """Half fold folded in half again: four flat plies, area a quarter."""
    s = x[:, 0] - x[:, 0].min()
    q = s.max() / 4
    k = np.minimum((s // q).astype(int), 3)
    r = s - k * q
    y = x.copy()
    y[:, 0] = np.where(k % 2 == 0, r, q - r) - q / 2
    y[:, 2] = table + np.array([1, 0, 3, 2])[k] * gap
    return y


def rolled(x, *, inner_radius=.030, pitch=.015, from_top=True, length=None, table=.790):
    """Towel rolled end to end on an Archimedean spiral (r = inner_radius + pitch * phi / 2 pi),
    length-preserving along material y, axis along x, the starting hem innermost. `length`
    rolls only that much; the rest lies flat, leaving the roll's outer layer at its bottom.
    Constructed shapes only; no reachability or crossing claim."""
    s = (x[:, 1].max() - x[:, 1]) if from_top else (x[:, 1] - x[:, 1].min())
    total = s.max() if length is None else length
    a, b = inner_radius, pitch / (2 * np.pi)
    phi = np.linspace(0., 400., 400001)
    arc = a * phi + b * phi ** 2 / 2
    end = np.interp(total, arc, phi)
    turn = np.interp(np.minimum(s, total), arc, phi) - end      # 0 at the roll's bottom
    r = a + b * (turn + end)
    outer = a + b * end
    y = x.copy()
    y[:, 1] = np.where(s <= total, r * np.sin(turn), s - total)
    y[:, 2] = np.where(s <= total, table + outer - r * np.cos(turn), table)
    return y


def tee_half_fold(x, *, left_over=True, gap=.006, crosswise=False):
    """Mirror one half of the tee onto the other about its centre line (x = 0), or for
    crosswise=True the neck half onto the hem half about y = mid. The moved half flips:
    its layer order reverses and it lies `gap` above the half that stays."""
    axis = 1 if crosswise else 0
    centre = (x[:, 1].min() + x[:, 1].max()) / 2 if crosswise else 0.
    moving = x[:, axis] < centre if left_over else x[:, axis] > centre
    base = x[:, 2] - x[:, 2].min()
    y = x.copy()
    y[moving, axis] = 2 * centre - x[moving, axis]
    y[moving, 2] = x[:, 2].min() + base.max() + gap + (base.max() - base[moving])
    return y

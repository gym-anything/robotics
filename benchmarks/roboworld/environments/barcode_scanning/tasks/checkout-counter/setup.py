"""Start of checkout-counter: the customer's five to seven items set out on the counter left of the scanner, standing
as on a shelf at random spots and turns, the empty bagging tote on the right and the scanner in its cradle.

setup(scene) stocks the scene (scene.Scene calls it after the counter's fixtures) and returns nothing;
_counter_items(scene, z, by) sets the items out on the counter top at height z, the tote centred at y = by."""
import math

import numpy as np

from barcode_scanning import stock
from barcode_scanning.catalog import SKUS
from barcode_scanning.stock import _open_box

COUNTER_SPREAD = dict(x=(-.66, -.12), y=(.06, .36), gap=.055)   # m: where the customer sets the items out, and their spacing


def _counter_items(scene, z, by):
    """checkout-counter: the customer's items set out on the counter, standing as they stand on a shelf, at random
    spots and turns left of the scanner, at least COUNTER_SPREAD["gap"] apart (the group's checkout row 087: "merchandise
    resting on counter unscanned"; in the basket no offered robot has a grasp). The same
    items as the basket tasks; the bagging tote as in checkout-present."""
    from barcode_scanning.scene import quat_axis_angle
    rng = scene.rng
    _open_box(scene, "bag_tote", (.34, by, z), (.40, .30, .18), "tote_grey", wall=.004, mass=.9)
    items = ["K03", "K01", "K02", "K04", "K05", "K09", "K10"]
    rng.shuffle(items)
    items = items[:5 + scene.c.seed % 3]
    S = COUNTER_SPREAD

    def outline(sku_id, x, y, yaw):
        sku = SKUS[sku_id]
        if sku.shape == "box":
            w, d, _ = sku.dims
            u = np.linspace(-.5, .5, 9)
            loc = np.array([(a*w, b*d) for a in u for b in (-.5, .5)] + [(a*w, b*d) for a in (-.5, .5) for b in u])
        else:
            r = sku.dims[0]
            a = np.linspace(0, 2*math.pi, 24, endpoint=False)
            loc = np.c_[r*np.cos(a), r*np.sin(a)]
        c, s = math.cos(yaw), math.sin(yaw)
        return loc @ np.array([[c, s], [-s, c]]) + (x, y)
    def layout():
        placed, poses = [], []
        for sku_id in items:
            for _ in range(600):
                x, y, yaw = rng.uniform(*S["x"]), rng.uniform(*S["y"]), rng.uniform(-math.pi, math.pi)
                pts = outline(sku_id, x, y, yaw)
                if pts[:, 0].min() < S["x"][0] - .06 or pts[:, 0].max() > S["x"][1] or pts[:, 1].min() < S["y"][0] - .04 \
                        or pts[:, 1].max() > S["y"][1] + .04:
                    continue
                if all(np.min(np.linalg.norm(pts[:, None, :] - q[None, :, :], axis=2)) >= S["gap"] for q in placed):
                    placed.append(pts); poses.append((x, y, yaw))
                    break
            else:
                return None                                       # no room left: lay the whole counter out again
        return poses
    for _ in range(40):
        poses = layout()
        if poses:
            break
    else:
        raise RuntimeError(f"no counter layout for {items} (seed {scene.c.seed})")
    for k, (sku_id, (x, y, yaw)) in enumerate(zip(items, poses)):
        sku = SKUS[sku_id]
        h = sku.dims[2] if sku.shape == "box" else sku.dims[1]
        name = f"counter_{sku_id}_{k}"
        scene.add_item(name, sku_id, (x, y, z + h/2 + .0005), rot=quat_axis_angle((0, 0, 1), yaw))
        scene.items[name].update(container="counter")
    scene.task_meta = dict(truth={it["sku"]: sum(1 for j in scene.items.values() if j.get("sku") == it["sku"])
                                  for it in scene.items.values() if it.get("sku")})


def setup(scene):
    stock.stock_counter(scene, set_out=_counter_items)

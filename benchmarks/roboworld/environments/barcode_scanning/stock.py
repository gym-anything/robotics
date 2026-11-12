"""Seeded stock layouts (planograms) for each preset.

The stockroom shelf holds retail overstock and store supplies faced the way a stocker leaves them: fronts to
the aisle, so GS1's back-panel barcodes face away. Ground truth for every task (which
units are in the target slot, their GTINs, which symbol belongs to which unit) comes from here and the
scene's label registry, never from rendered pixels.

Each task stocks its scene in its folder (tasks/<task>/setup.py): its setup(scene), which the scene calls after the
preset's fixtures, calls stock_stockroom or stock_counter here with its own parts, or stocks the library desk itself.
"""

import math
import numpy as np
from .catalog import SKUS
from .config import task_setup

# reachable count locations: deck index 1..3 (0.57, 1.03, 1.45 m) of the centre bay
TARGET_SLOTS = ("A03-3-B", "A03-3-A", "A03-3-C", "A03-2-B", "A03-4-B")

DECK_ASSORTMENT = {
    1: [("K08", 1, 1), ("K07", 1, 2)],                                 # 0.57 m: supplies (reams, glove boxes)
    2: [("K03", 1, 3), ("K01", 3, 2), ("K06", 1, 2), ("K02", 4, 2)],   # 1.03 m: grocery
    3: [("K05", 3, 2), ("K04", 3, 2)],                                  # 1.45 m: bottles and jars
}


def target_slot(config):
    """The slot the instruction names, from the seed; None for a task whose setup.py sets NAMES_SLOT = False."""
    if not getattr(task_setup(config.task), "NAMES_SLOT", True):
        return None
    return TARGET_SLOTS[config.seed % len(TARGET_SLOTS)] if config.seed else "A03-3-B"


def stock_stockroom(scene, counted=None):
    """The stockroom shelf, for a stockroom task's setup.py: shipping cases in the neighbouring bays, the centre bay's
    slots from DECK_ASSORTMENT; counted(deck, rng), from a counting task's setup.py, gives the counted slot's stock."""
    rng = scene.rng
    c = scene.c
    target = target_slot(c)
    scene.task_meta = dict(target_slot=target)
    for code, s in scene.slots.items():
        if s["deck"] not in (1, 2, 3):
            continue
        if s["bay"] != 0:
            # neighbouring bays: corrugated shipping cases, one per slot (dynamic, reachable at the bay edges)
            name = f"case_{code}"
            w, d, h = .26, .40, .24 if s["deck"] < 3 else .20
            cx = (s["x"][0] + s["x"][1])/2
            from .scene import quat_axis_angle
            body = scene.free_body(name, (cx, .23, s["z"] + h/2 + .0005), quat_axis_angle((0, 0, 1), float(rng.normal(0, .03))))
            scene.box(f"{name}_hull", (0., 0., 0.), (w/2, d/2, h/2), "carton", body=body, density=5.0/(w*d*h), mu=.45)
            scene.set_mass(body, 5.0)
            scene.items[name] = dict(body=body, sku=None, name="shipping case", slot=code)
            continue
        options = DECK_ASSORTMENT[s["deck"]]
        if code == target and counted is not None:
            entries = counted(s["deck"], rng)
        else:
            k = s["slot"] % len(options) if len(options) > 1 else 0
            sku, facings, depth = options[(k + int(rng.integers(0, len(options)))) % len(options)]
            entries = [(sku, facings, depth)]
        scene.planogram(code, entries)
    if target:
        truth = {}
        for name, it in scene.items.items():
            if it.get("slot") == target and it.get("sku"):
                truth[it["sku"]] = truth.get(it["sku"], 0) + 1
        scene.task_meta.update(truth=truth)


def _target_entries(deck, rng):
    """Stock drawn for the counted slot: two or three SKUs, several units deep (the back rows stay hidden).
    Scene.planogram places them left to right and skips what no longer fits (and everything after it), so a counted
    slot holds one or two of them: on deck 2 cereal and soup, cereal only or soup only (the 6-pack and the tomato paste
    never fit), on deck 1 the glove boxes, on deck 3 water and sauce."""
    if deck == 2:
        pool = [("K03", 1, 2), ("K01", 2, 2), ("K06", 1, 1), ("K02", 2, 2)]
    elif deck == 1:
        pool = [("K07", 1, 3), ("K08", 1, 1)]
    else:
        pool = [("K05", 2, 2), ("K04", 2, 2)]
    idx = rng.permutation(len(pool))[:min(len(pool), 3 if deck == 2 else 2)]
    chosen = [pool[i] for i in sorted(idx)]
    return chosen


def stock_counter(scene, set_out=None, extra=None):
    """Customer's hand basket on the counter's left, bagging tote on the right, for a checkout task's setup.py:
    set_out(scene, z, by) sets the items out on the counter instead (checkout-counter), extra(scene, z) adds to the
    counter after the basket (checkout-bulky's case); z is the counter top's height."""
    from .scene import quat_axis_angle
    rng = scene.rng
    z = scene.fixtures["counter"]["top"]
    # within reach of the two FR3s (bases 0.40 m behind the counter edge, 0.50 m apart): the basket in front of the
    # left arm, the bagging tote in front of the right one, the scanner's cradle between them (a layout at
    # x = +-0.50 m, 0.30 m in puts the basket ~1 m from the right arm and the tote as far from the left)
    bx, by = -.34, .24
    if set_out is not None:
        return set_out(scene, z, by)
    basket = _open_box(scene, "basket", (bx, by, z), (.43, .30, .20), "basket_blue", wall=.004, mass=.6)
    tote = _open_box(scene, "bag_tote", (.34, by, z), (.40, .30, .18), "tote_grey", wall=.004, mass=.9)
    items = ["K03", "K01", "K02", "K04", "K05", "K09", "K10"]
    rng.shuffle(items)
    items = items[:5 + scene.c.seed % 3]
    # the items go in one by one, the way a customer drops a few items into a basket: on the floor wherever there is
    # room, at a random spot, keeping the largest gap from what is already there and from the walls that the floor
    # allows (30 mm, else 20, 10, 3 mm), stacking only when the floor is full (a 5 mm heightmap of the basket floor
    # and its contents). Boxes lie flat on their back or front, cans and bottles on their side, long axis along x or
    # y. A 12 x 16 cm grid for items up to 29 cm long spawned them interpenetrating (a 20 cm pop with rigid contacts);
    # packing them 3 mm apart into the back corner, which no customer does, left no room for any finger (every basket
    # item was ungraspable)
    floor = z + .004 + .0005
    inner = (.43 - .008 - .006, .30 - .008 - .006)                     # interior less a 3 mm clearance to each wall
    cell = .005
    nx, ny = int(inner[0]/cell), int(inner[1]/cell)
    hmap = np.zeros((nx, ny))
    for k, sku_id in enumerate(items):
        sku = SKUS[sku_id]
        if sku.shape == "box":
            w, d, h = sku.dims
            flip = math.pi/2 if rng.random() < .5 else -math.pi/2
            shapes = [((w, h), d, quat_axis_angle((1, 0, 0), flip))]
            shapes.append(((h, w), d, _qmul(quat_axis_angle((0, 0, 1), math.pi/2), shapes[0][2])))
        else:
            r, h = sku.dims
            lying = quat_axis_angle((0, 1, 0), math.pi/2)                # axis along x
            shapes = [((h, 2*r), 2*r, lying), ((2*r, h), 2*r, _qmul(quat_axis_angle((0, 0, 1), math.pi/2), lying))]
        best = None
        occupied = hmap > 0
        for gap_m in (.030, .020, .010, 0.):
            g = int(math.ceil(gap_m/cell))
            options = []
            for (fx, fy), thick, q in shapes:
                cx, cy = int(math.ceil((fx + .006)/cell)), int(math.ceil((fy + .006)/cell))
                if cx + 2*g > nx or cy + 2*g > ny:
                    continue
                for i in range(g, nx - cx - g + 1):
                    for j in range(g, ny - cy - g + 1):
                        if occupied[max(0, i - g):i + cx + g, max(0, j - g):j + cy + g].any():
                            continue
                        options.append((i, j, cx, cy, thick, q, fx, fy))
            if options:
                best = options[int(rng.integers(0, len(options)))]
                break
        if best is None:                                             # the floor is full: lowest spot, stacked
            for (fx, fy), thick, q in shapes:
                cx, cy = int(math.ceil((fx + .006)/cell)), int(math.ceil((fy + .006)/cell))
                if cx > nx or cy > ny:
                    continue
                for i in range(0, nx - cx + 1, 2):
                    for j in range(0, ny - cy + 1, 2):
                        base = float(hmap[i:i + cx, j:j + cy].max())
                        if best is None or base < best[0]:
                            best = (base, (i, j, cx, cy, thick, q, fx, fy))
            best = best[1]
        i, j, cx, cy, thick, q, fx, fy = best
        base = float(hmap[i:i + cx, j:j + cy].max())
        hmap[i:i + cx, j:j + cy] = base + thick
        x = bx - inner[0]/2 + (i + cx/2)*cell
        y = by - inner[1]/2 + (j + cy/2)*cell
        pos = (x, y, floor + base + thick/2 + .002)
        name = f"basket_{sku_id}_{k}"
        scene.add_item(name, sku_id, pos, rot=q)
        scene.items[name].update(container="basket")
    if extra is not None:
        extra(scene, z)
    scene.task_meta = dict(truth={it["sku"]: sum(1 for j in scene.items.values() if j.get("sku") == it["sku"])
                                  for it in scene.items.values() if it.get("sku")})


def _qmul(a, b):
    from .scene import quat_mul
    return quat_mul(a, b)


def _open_box(scene, name, centre, size, material, wall, mass):
    """A rigid open-top container (hand basket / tote) as a free body of five walls."""
    x, y, z = centre
    L, W, H = size
    body = scene.free_body(name, (x, y, z + H/2 + .0005))
    parts = [((0, 0, -H/2 + wall/2), (L/2, W/2, wall/2)),
             ((L/2 - wall/2, 0, 0), (wall/2, W/2, H/2)), ((-L/2 + wall/2, 0, 0), (wall/2, W/2, H/2)),
             ((0, W/2 - wall/2, 0), (L/2, wall/2, H/2)), ((0, -W/2 + wall/2, 0), (L/2, wall/2, H/2))]
    vol = sum(8*h[0]*h[1]*h[2] for _, h in parts)
    for k, (p, h) in enumerate(parts):
        scene.box(f"{name}_wall{k}", p, h, material, body=body, density=mass/vol, mu=.4)
    scene.set_mass(body, mass)
    scene.items[name] = dict(body=body, sku=None, name=name, container_size=size)
    return body


def library_item_number(library, sequence):
    """14-digit library item number: type digit 3, 4-digit library code, 8-digit sequence, mod-10 check digit
    (Luhn, the scheme library barcode vendors use)."""
    body = f"3{library:04d}{sequence:08d}"
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return body + str((10 - total % 10) % 10)

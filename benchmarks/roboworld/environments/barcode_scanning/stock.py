"""Seeded stock layouts (planograms) for each preset.

The stockroom shelf holds retail overstock and store supplies faced the way a stocker leaves them: fronts to
the aisle, so GS1's back-panel barcodes face away. Ground truth for every task (which
units are in the target slot, their GTINs, which symbol belongs to which unit) comes from here and the
scene's label registry, never from rendered pixels.
"""

import math
import numpy as np
from .catalog import SKUS
from . import params as P

# reachable count locations: deck index 1..3 (0.57, 1.03, 1.45 m) of the centre bay
TARGET_SLOTS = ("A03-3-B", "A03-3-A", "A03-3-C", "A03-2-B", "A03-4-B")

DECK_ASSORTMENT = {
    1: [("K08", 1, 1), ("K07", 1, 2)],                                 # 0.57 m: supplies (reams, glove boxes)
    2: [("K03", 1, 3), ("K01", 3, 2), ("K06", 1, 2), ("K02", 4, 2)],   # 1.03 m: grocery
    3: [("K05", 3, 2), ("K04", 3, 2)],                                  # 1.45 m: bottles and jars
}


def target_slot(config):
    if config.task in ("hold-scanner",):
        return None
    return TARGET_SLOTS[config.seed % len(TARGET_SLOTS)] if config.seed else "A03-3-B"


def stock_stockroom(scene):
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
        if code == target and c.task in ("count-by-scanning", "count-key-quantity"):
            entries = _target_entries(s["deck"], rng, c.task)
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


def _target_entries(deck, rng, task):
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
    if task == "count-key-quantity":
        chosen = [(s, f, max(d, 3)) for s, f, d in chosen]
    return chosen


def stock_counter(scene):
    """Customer's hand basket on the counter's left, bagging tote on the right."""
    from .scene import quat_axis_angle
    rng = scene.rng
    z = scene.fixtures["counter"]["top"]
    # within reach of the two FR3s (bases 0.40 m behind the counter edge, 0.50 m apart): the basket in front of the
    # left arm, the bagging tote in front of the right one, the scanner's cradle between them (a layout at
    # x = +-0.50 m, 0.30 m in puts the basket ~1 m from the right arm and the tote as far from the left)
    bx, by = -.34, .24
    if scene.c.task == "checkout-counter":
        return _counter_items(scene, z, by)
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
    if scene.c.task == "checkout-bulky":
        # the 24-pack at the counter's end beyond the tote, its barcode side (catalogue "left") turned to the robots;
        # footprint 279 x 406 mm, 20 mm clear of the tote (0.14..0.54 m). The first layout put it at (0.35, 0.44), on
        # top of the tote: the settle threw the tote 1.2 m onto the floor (adversarial evaluator tests)
        name = "water_case"
        scene.add_item(name, "K11", (.70, .30, z + SKUS["K11"].dims[2]/2 + .0005), yaw=math.pi/2 + float(rng.normal(0, .02)))
        scene.items[name].update(container="counter")
    scene.task_meta = dict(truth={it["sku"]: sum(1 for j in scene.items.values() if j.get("sku") == it["sku"])
                                  for it in scene.items.values() if it.get("sku")})


COUNTER_SPREAD = dict(x=(-.66, -.12), y=(.06, .36), gap=.055)   # m: where the customer sets the items out, and their spacing


def _counter_items(scene, z, by):
    """checkout-counter: the customer's items set out on the counter, standing as they stand on a shelf, at random
    spots and turns left of the scanner, at least COUNTER_SPREAD["gap"] apart (the group's checkout row 087: "merchandise
    resting on counter unscanned"; in the basket no offered robot has a grasp). The same
    items as the basket tasks; the bagging tote as in checkout-present."""
    from .scene import quat_axis_angle
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


def stock_library(scene):
    from .scene import quat_axis_angle
    rng = scene.rng
    z = scene.fixtures["counter"]["top"]
    _open_box(scene, "returns_bin", (.34, .24, z), (.40, .32, .16), "tote_grey", wall=.004, mass=.9)
    books = ["K12", "K13", "K12", "K13", "K12"][:3 + scene.c.seed % 3]
    zz = z
    from .artwork import Symbol, render_symbol
    for k, sku_id in enumerate(books):
        s = SKUS[sku_id]
        w, d, h = s.dims
        # where the library put its item label (seeded): on the back cover near the top, or inside the front cover;
        # returned books come back in either orientation -- the ones labelled inside lie
        # front cover up here, so the cover has to be opened to reach the label
        inside = (k + scene.c.seed) % 3 == 1
        yaw = float(rng.normal(0, .08)) + (math.pi if k % 2 else 0.)
        flip = -math.pi/2 if inside else math.pi/2          # -90: front (-y) up; +90: back (+y) up
        q = _qmul(quat_axis_angle((0, 0, 1), yaw), quat_axis_angle((1, 0, 0), flip))
        name = f"book_{k}"
        body, cover = scene.add_book(name, sku_id, (-.30 + float(rng.normal(0, .01)), .24, zz + d/2 + .0005), q)
        item_no = library_item_number(1234, 5000 + 37*k + scene.c.seed)
        scene.items[name].update(container="stack", library_id=item_no, label_inside_cover=inside)
        sym = Symbol("Codabar", f"A{item_no}B", .25, bar_height_mm=8.5, hri=True)
        _, w_mm, h_mm = render_symbol(sym, 2)
        size = (w_mm/1e3 + .004, h_mm/1e3 + .0055)
        if inside:
            # inner face of the front board (cover frame: hinge at the origin, board along +x, inner face +y)
            centre = (w - .012 - w_mm/2e3, P.book_cover(SKUS[sku_id])[0]/2, h/2 - .018 - h_mm/2e3)
            scene.label(f"{name}_library", cover, sym, centre, (0, 1, 0), (0, 0, 1), size, "matte", kind="library",
                        item=name, text=item_no)
        else:
            centre = (w/2 - .012 - w_mm/2e3, d/2, h/2 - .018 - h_mm/2e3)
            scene.label(f"{name}_library", body, sym, centre, (0, 1, 0), (0, 0, 1), size, "film", kind="library",
                        item=name, text=item_no)
        zz += d + .001
    scene.task_meta = dict(truth={})


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

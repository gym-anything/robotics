"""Start of part-id-match: the receiving bench (scene.part) with three stainless-steel mounting plates from one delivery
lying on it, each laser-marked with its part number (plates: confusable part numbers, which slot holds the labelled
plate, plate poses within +-8 mm and +-4 deg, marking corners), and the thermal-transfer printer presenting the 2 x 4 in
polyester ID label it has just printed for one of them, carrying that plate's part number and the seed's serial number
(payload).

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). part-id-match starts from the printer's presented label as the scene
builds it: setup constructs nothing and returns {}. place_items(scene, z) lays the plates on the bench and names the
target plate (scene.part calls it while building the scene, z the bench top); payload(seed) is the barcode payload
printed for the seed (layouts.payload calls it): PAYLOAD at seed 0. WRIST_CAMERAS: the backend turns the wrist cameras
on by default."""
import math
from label_application.layouts import _rng, _serial

# seed 0 (payload draws it: plates(0)'s target part number and serial 318)
PAYLOAD = "NG-7715-0042-000318"
# wrist cameras on by default (backend): the plates' laser-marked part numbers and the label's printed one (~3 px per
# character overhead)
WRIST_CAMERAS = True
PLATE_SIZE = (.16, .12, .006)          # the stainless plate of the retired part-id-steel task, unchanged
PLATE_SLOTS = (-.03, .17, .37)         # x of the three plates on the receiving bench (left of the tool tray)
PLATE_Y = -.03
MARKING_SIZE = (.064, .014)            # laser-marked "P/N ...." block on the plate face (u, v extents)


def payload(seed):
    p = plates(seed)
    return f"NG-{p['part_numbers'][p['target']]}-{_serial(seed, 7):06d}"


def plates(seed):
    """Three stainless mounting plates from one delivery, each laser-marked with its part number; the printed ID
    label belongs to one of them. Part numbers are confusable (same family, one digit or one place apart), as
    left/right-hand and revision variants are. Returns slot poses (x, y, yaw), part numbers per slot, marking
    rectangles in plate coordinates and the target slot."""
    r = _rng(seed, 8)
    if seed == 0:
        a, b = 7715, 42
    else:
        a, b = int(r.integers(1100, 9800)), int(r.integers(10, 980))
    family = [f"{a:04d}-{b:04d}", f"{a:04d}-{b + 1:04d}", f"{a + 10:04d}-{b:04d}"]
    order = [0, 1, 2] if seed == 0 else [int(k) for k in r.permutation(3)]
    numbers = [family[k] for k in order]
    target = order.index(0)                       # the label is printed for family[0]
    poses, markings = [], []
    for k, x in enumerate(PLATE_SLOTS):
        if seed == 0:
            dx, dy, yaw = 0., 0., 0.
        else:
            dx, dy = (float(v) for v in r.uniform(-.008, .008, 2))
            yaw = float(math.radians(r.uniform(-4., 4.)))
        poses.append((x + dx, PLATE_Y + dy, yaw))
        su = 1 if (seed == 0 and k != 1) or (seed and r.random() < .5) else -1
        sv = 1 if (seed == 0) or r.random() < .5 else -1
        mu = su*.040
        mv = sv*(PLATE_SIZE[1]/2 - .009 - MARKING_SIZE[1]/2)
        markings.append(dict(u=(mu - MARKING_SIZE[0]/2, mu + MARKING_SIZE[0]/2),
                             v=(mv - MARKING_SIZE[1]/2, mv + MARKING_SIZE[1]/2)))
    return dict(poses=poses, part_numbers=numbers, markings=markings, target=target)


def place_items(scene, z):
    """The three plates on the bench, each with its part number and marking, and the target plate (scene.part)."""
    from label_application import materials as M
    from label_application.labelsim import Collider
    # three stainless mounting plates from one delivery (the retired part-id-steel plate, unchanged: same size,
    # material, density, friction and label surface), each laser-marked with its part number (visual only:
    # annealed marking is flush with the surface); the label belongs to one of them (plates)
    lay = plates(scene.c.seed)
    sx, sy, sz = PLATE_SIZE
    for k, ((x, y, yaw), number, mark) in enumerate(zip(lay["poses"], lay["part_numbers"], lay["markings"])):
        name = f"plate_{k + 1}"
        q = scene.wp.quat_from_axis_angle(scene.wp.vec3(0, 0, 1), float(yaw))
        body = scene.free_body(name, (x, y, z + sz/2), q)
        scene.box(f"{name}_steel", (0, 0, 0), (sx/2, sy/2, sz/2), "stainless", body=body, mu=.3, density=7900.)
        scene.colliders.append(Collider(name, "box", (sx/2, sy/2, sz/2), M.SURFACES["steel"], body=body))
        scene.items[name] = dict(body=body, kind="steel_plate", size=PLATE_SIZE, part_number=number,
                                 marking=mark)
    scene.target = f"plate_{lay['target'] + 1}"
    scene.task_geometry["plates"] = lay


def setup(world):
    return {}

"""Start of finish-securing: the shipping pack bench with the sealed carton at its seeded place and turn, and the 4 x 6
in shipping label stuck down on the carton's top by a co-worker except for two to four areas that did not take and have
curled up off the carton; the robot waits in its ready pose with the hand above the label (ladder.build_start). The
label lies on either flap half (centre +-30 mm along, +-5 mm across, turned up to 3 deg), either way round; the lifted
areas are among the four corners (legs 28-45 / 24-40 mm), the two ends (14-40 mm deep, the front up to 12 mm out of
square) and the two long sides (12-17 mm deep), curled 60-90 deg (at most 100/m), apart from each other and clear of the
barcode, together 12-35 % of the label.

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). setup builds the partly applied label and the ready pose on the world
(ladder.build_start) and returns what that returns. laid_label(seed, carton_W) is where the label was stuck down and
which areas of it are lifted (layouts.partly_applied, which draws them with this file's LAYOUT_TAG, label_side, YAW_DEG
and lifted_areas), kept by scene.parcel as task_geometry["laid_label"]. payload(seed) is the barcode payload printed for
the seed (layouts.payload calls it): a carrier tracking number, as on press-down, whose bench this task shares.
WRIST_CAMERAS: the backend turns the wrist cameras on by default."""
from label_application.layouts import _flap, _flaps_ok, _rng

TASK = "finish-securing"
LAYOUT_TAG = 10     # layouts.partly_applied's generator: _rng(seed, LAYOUT_TAG)
YAW_DEG = 3.        # the label is turned up to +-YAW_DEG on the carton
# wrist cameras on by default (backend): the close view of the lifted areas from the ready pose's hand above the
# label (a 4 x 6 in label is ~90 x 60 px overhead)
WRIST_CAMERAS = True


def payload(seed):
    return "9612019" + "".join(str(d) for d in _rng(seed, 12).integers(0, 10, 11))


def label_side(r):
    """The flap half the label was stuck on: +1 toward the robots, -1 the far half."""
    return 1 if r.random() < .5 else -1


def lifted_areas(r, corners, L, W, seed):
    """The lifted areas (layouts._flap) drawn from r after the label's place; corners: the label's four material
    corners, L x W its size."""
    for _ in range(1000):
        k = int(r.choice([2, 2, 3, 3, 4]))
        pool = [("corner", c) for c in corners] + [("end", 0.), ("end", L), ("side", 0.), ("side", W)]
        pick = [pool[i] for i in r.permutation(len(pool))[:k]]
        flaps = []
        for kind, c in pick:
            tip = r.uniform(60., 90.)
            if kind == "corner":
                flaps.append(_flap(kind, r.uniform(.028, .045), r.uniform(.024, .040), c, tip))
            elif kind == "end":
                w, skew = r.uniform(.014, .040), r.uniform(-.006, .006)
                flaps.append(_flap(kind, w + skew, w - skew, c, tip))
            else:
                w, skew = r.uniform(.012, .017), r.uniform(-.002, .002)
                flaps.append(_flap(kind, w + skew, w - skew, c, tip))
        if _flaps_ok(flaps, (.12, .35)):
            break
    else:                                   # pragma: no cover - the draw always succeeds within a few tries
        raise RuntimeError(f"no lifted-area layout for seed {seed}")
    return flaps


def laid_label(seed, carton_W):
    from label_application import layouts
    return layouts.partly_applied(TASK, seed, carton_W=carton_W)


def setup(world):
    from label_application import ladder
    return ladder.build_start(world)

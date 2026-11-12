"""Start of smooth-corner: the shipping pack bench with the sealed carton at its seeded place and turn, and the 4 x 6 in
shipping label stuck down on the carton's top by a co-worker except for one corner that did not take and has curled up
off the carton; the robot waits in its ready pose with the hand above the label (ladder.build_start). The label lies on
the flap half toward the robots like press-down's laid label (centre +-30 mm along, +-5 mm across the zone's middle,
turned up to 2 deg), either way round; the lifted corner is any of the four, legs 28-38 mm along the label and 22-32 mm
across, its tip curled 65-90 deg off the carton.

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). setup builds the partly applied label and the ready pose on the world
(ladder.build_start) and returns what that returns. laid_label(seed, carton_W) is where the label was stuck down and
which areas of it are lifted (layouts.partly_applied, which draws them with this file's LAYOUT_TAG, label_side, YAW_DEG
and lifted_areas), kept by scene.parcel as task_geometry["laid_label"]. payload(seed) is the barcode payload printed for
the seed (layouts.payload calls it): a carrier tracking number, as on press-down, whose bench this task shares.
WRIST_CAMERAS: the backend turns the wrist cameras on by default."""
from label_application.layouts import _flap, _flaps_ok, _rng

TASK = "smooth-corner"
LAYOUT_TAG = 9      # layouts.partly_applied's generator: _rng(seed, LAYOUT_TAG)
YAW_DEG = 2.        # the label is turned up to +-YAW_DEG on the carton
# wrist cameras on by default (backend): the close view of the lifted areas from the ready pose's hand above the
# label (a 4 x 6 in label is ~90 x 60 px overhead)
WRIST_CAMERAS = True


def payload(seed):
    return "9612019" + "".join(str(d) for d in _rng(seed, 11).integers(0, 10, 11))


def label_side(r):
    """The flap half the label was stuck on: +1 toward the robots, -1 the far half."""
    return 1


def lifted_areas(r, corners, L, W, seed):
    """The lifted areas (layouts._flap) drawn from r after the label's place; corners: the label's four material
    corners, L x W its size."""
    c = corners[int(r.integers(0, 4))]
    flaps = [_flap("corner", r.uniform(.028, .038), r.uniform(.022, .032), c, r.uniform(65., 90.))]
    if not _flaps_ok(flaps):
        raise RuntimeError(f"smooth-corner seed {seed}: corner layout invalid")
    return flaps


def laid_label(seed, carton_W):
    from label_application import layouts
    return layouts.partly_applied(TASK, seed, carton_W=carton_W)


def setup(world):
    from label_application import ladder
    return ladder.build_start(world)

"""gift-bags-tasks-v2 score arithmetic: task_set_scoring.py with the render-check fixes (closed-flap creep margin,
put-lid-on engagement from the skirt). Pure. All distances are mm and angles degrees unless stated otherwise."""
import math

FAILED = .9
# Closed flaps creep open by themselves once released: untaped, 0.6 deg after 0.1 s and 1.2 deg after 4 s (REPORT
# 6.11 hinge bench); the render-check probe measured the open-flaps start's majors at 0.61, 0.79, 0.95
# and 1.09 deg over 3 s hands-off, which paid an idle robot 0.0025 and rising. Only opening beyond this margin over
# the start angle counts. Never more than the spec's angle/90 per flap.
FLAP_SPRINGBACK_DEG = 10.


def clip(x):
    if not math.isfinite(float(x)):
        raise ValueError("Non-finite task metric")
    return max(0., min(1., float(x)))


def ramp(x, a, b):
    return clip((x - a)/(b - a))


def improved(value, start, target=1.):
    """No credit for the given start; no tick/action/order dependence."""
    return ramp(value, start, target) if target > start + 1e-12 else 0.


def score(success, progress):
    return 1. if success else FAILED*clip(progress)


def base_factor(lift, tilt, lift_limit=5., tilt_limit=10.):
    return (1. - ramp(lift, lift_limit, 25.))*(1. - ramp(tilt, tilt_limit, 45.))


def lid_off(m, initial, frame):
    vertical = clip((initial["lid_overlap_mm"] - m["lid_overlap_mm"])/max(initial["lid_overlap_mm"], 1e-9))
    off = max(vertical, improved(1. - m["lid_plan_fraction"], 1. - initial["lid_plan_fraction"]))
    down = 1. - ramp(m["lid_lowest_above_table_mm"], 5., 100.)
    return off*(.5 + .5*down)*base_factor(m["episode_base_lift_mm"], m["episode_base_tilt_deg"])*frame


def lid_on(m, initial, frame):
    # The spec's engaged = clip(overlap / 20 mm) reads the skirt's engagement on the base (lid_skirt_engagement_mm),
    # not the vertical overlap alone: that is already full for a lid beside the box, so a lid leaning half over the
    # base would earn the whole engagement term without being on it.
    raw = m["lid_plan_fraction"]*(.5 + .5*clip(m["lid_skirt_engagement_mm"]/20.))
    start = initial["lid_plan_fraction"]*(.5 + .5*clip(initial["lid_skirt_engagement_mm"]/20.))
    return improved(raw, start)*base_factor(m["base_lowest_above_table_mm"], m["base_tilt_signed_deg"])*frame


def unwrap(m, initial):
    bare = improved(1. - m["box_coverage"], 1. - initial["box_coverage"], .95)
    return bare*clip(m["lid_overlap_mm"]/20.)*(1. - ramp(m["box_tilt_deg"], 15., 60.))


def carton(m, initial, flaps, taped):
    # Remove the closed flaps' own springback and creep (FLAP_SPRINGBACK_DEG): an untouched carton earns zero.
    opened = sum(improved(m[f"{f}_angle_deg"], initial[f"{f}_angle_deg"] + FLAP_SPRINGBACK_DEG, 90.)
                 for f in flaps)/len(flaps)
    tape = improved(1. - m.get("seam_joined_fraction", 1.), 1. - initial.get("seam_joined_fraction", 1.), .90)
    work = .4*tape + .6*opened if taped else opened
    place = (1. - ramp(m["carton_move_mm"], 30., 100.))*(1. - ramp(m["carton_tilt_signed_deg"], 10., 45.))
    # Continuous distance from the allowed contents volume, not a count of passed containment checks.
    contents = sum(1. - ramp(d, 0., 100.) for d in m["contents_outside_distance_mm"])/len(m["contents_outside_distance_mm"])
    return work*place*contents


def bag(m, initial, purchase):
    r = m["reward"]
    if purchase:
        # Given the bag and board: only improvement of the candle earns credit.
        c = improved(r["candle"], initial["reward"]["candle"])
        return c*r["opened"]*r["board"]*r["integrity"]
    work = (r["bag"] + r["board"])/2.
    start = (initial["reward"]["bag"] + initial["reward"]["board"])/2.
    return improved(work, start)*r["integrity"]

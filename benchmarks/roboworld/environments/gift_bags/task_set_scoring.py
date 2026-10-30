"""Pure score arithmetic. All distances are mm and angles degrees unless stated otherwise."""
import math

FAILED = .9


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
    raw = m["lid_plan_fraction"]*(.5 + .5*clip(m["lid_overlap_mm"]/20.))
    start = initial["lid_plan_fraction"]*(.5 + .5*clip(initial["lid_overlap_mm"]/20.))
    return improved(raw, start)*base_factor(m["base_lowest_above_table_mm"], m["base_tilt_signed_deg"])*frame


def unwrap(m, initial):
    bare = improved(1. - m["box_coverage"], 1. - initial["box_coverage"], .95)
    return bare*clip(m["lid_overlap_mm"]/20.)*(1. - ramp(m["box_tilt_deg"], 15., 60.))


def carton(m, initial, flaps, taped):
    # Remove the tiny closed-state springback from each angle; an untouched untaped carton earns zero.
    opened = sum(improved(m[f"{f}_angle_deg"], initial[f"{f}_angle_deg"], 90.) for f in flaps)/len(flaps)
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

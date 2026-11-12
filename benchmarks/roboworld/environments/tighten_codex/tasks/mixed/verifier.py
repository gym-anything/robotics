"""Outcome of mixed: all four screws seated by their own preset drivers (green 0.35 N m on the M3s, yellow 0.60 N m on
the M4s) until each clicks and holding their load hands-off, the M3 bit moment never above 0.4 N m, the rail level, the
reference flag back at its home with the sensor detecting it there, and nothing robot-held touching the assembly
(TightenEnv.MIXED_SCOPE has the full rule). The score is 1 on success, else 0.95 x the mean screw credit (score).

verify(env, metrics) gets the TightenEnv and its per-screw metrics (TightenEnv._evaluate) and returns the evaluation
dict: TightenEnv._evaluate_released under tighten-mixed-v1, the only contract that selects mixed, which asks for every
screw and calls score below. HANDS_OFF_RULES puts this task under mixed's hands-off rules (task.hands_off_rules);
DETECTION and SPOT_CENTRE_DETECTION are read by the graders."""
import numpy as np

# mixed partial credit only: a flag short of home keeps a share of the
# detection grade that falls linearly from the band edge to 0 at 10 mm
# (the detected-cue LED is lit to ~7.2 mm). Success still needs home.
MIXED_FLAG_RETURN_GRADE_M = .010

HANDS_OFF_RULES = True
DETECTION = True
SPOT_CENTRE_DETECTION = True


def verify(env, metrics):
    return env._evaluate_released(metrics)


def score(env, metrics, optical, success, flag_offset):
    """Per-deliverable partial credit from hands-off readings (see
    TightenEnv.MIXED_SCOPE); never 1.0 without success. No term rewards motion,
    contact, tool pickup, flag retraction, support or work order."""
    tol=env.config.rail_level_tolerance_rad
    level=float(np.clip((env.FREE_DROOP_RAD-abs(env._free_pitch))/(env.FREE_DROOP_RAD-tol),0.,1.))
    coverage=env._spot_coverage(optical)
    # Detection grade = aim x return. The aim is judged against a flag at
    # its HOME x (1 when the spot centre would be on it, else twice the
    # spot share there), so moving the flag can never raise it; the return
    # share is 1 inside the home band and falls linearly to 0 at 10 mm.
    hit=optical.get("intersection_m")
    if hit is None:
        aim=float(optical["detected"])
    else:
        home_x=float(env.state.body_q.numpy()[env.items["flag"],0])-flag_offset
        on_face=(.04<=optical["range_m"]<=.14 and abs(hit[0]-home_x)<=.005 and
                 abs(hit[2]-env.items["flag_center_z_m"])<=.022)
        aim=1. if on_face else min(1.,2*env._spot_coverage(optical,home_x))
    band,reach=env.MIXED_FLAG_HOME_BAND_M,MIXED_FLAG_RETURN_GRADE_M
    returned=1. if abs(flag_offset)<=band else max(0.,(reach-abs(flag_offset))/(reach-band))
    seen=aim*returned
    credit={}
    for name,m in metrics.items():
        m3=name.startswith("sensor_")
        floor=m["minimum_retained_load_n"]
        if m3 and m["peak_bit_socket_moment_2ms_nm"]>.4:
            seat=0.
        elif (m["driver_released_on_screw"] and m["hands_off_load_n"]>=floor and
              m["advance_m"]>=(.0004 if m3 else .00045)):
            seat=1.
        else:
            seat=.9*min(1.,m["hands_off_load_n"]*.8/floor)
        # Only the upward M4 sets the rail's attitude; the forward M4
        # clamps the (non-rotating) bracket to the post.
        credit[name]=seat*((.75+.25*seen) if m3 else level if name=="frame_top" else 1.)
    partial=sum(credit.values())/len(credit)
    return (1. if success else .95*partial),{
        "score_terms":{"screw_credit":credit,"rail_level_grade":level,
                       "hands_off_rail_pitch_rad":env._free_pitch,
                       "spot_on_flag_fraction":coverage,"aim_at_home_grade":aim,
                       "flag_return_grade":returned,"detection_grade":seen,
                       "partial_mean":partial}}

"""Outcome of set-up-gun: the source at the spray procedure (12.7 m/min wire feed within 0.3, 29 V within 0.5) and
1/2 in of wire out of the contact tip (within 1/16 in), with no arc, the gun's hardware touching nothing and the gas
valve closed at the end.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle changes nothing here) and
returns the evaluation: success, score, checks, progress and reward (Evaluator.evaluate_checks with _eval_set_up_gun
and _progress_set_up_gun below). WELD: the task is not judged on the weld's sections."""
from welding import params as PR
from welding.task import FAULT_FACTOR, SETUP_PROCEDURE, SETUP_VOLTS_TOL, SETUP_WFS_TOL, _mean, _ok, net_of_start

WELD = False
# the procedure's electrical stick-out = contact tip to work - 1/4 in = 19 - 6.35 = 12.7 mm, Lincoln L-56 [read], the
# basis of the variations' stick-out range; tolerance [authored]: 1/16 in
SETUP_STICKOUT = 12.7e-3
SETUP_STICKOUT_TOL = 1.5e-3


def verify(e, settle=False):
    return e.evaluate_checks(lambda: _eval_set_up_gun(e), lambda: _progress_set_up_gun(e))


def setup_progress(wfs_m_min, volts, stickout_m, start, faulted):
    """set-up-gun's progress: the mean of the wire feed, voltage and stick-out terms, each net of the start state
    (net_of_start; a term already in its band at the start is left out); halved for gun contact or an arc."""
    target = PR.v(PR.SCHEDULES)[SETUP_PROCEDURE]
    terms = dict(
        wire_feed=net_of_start(abs(wfs_m_min - target["wfs_m_min"]), abs(start["wfs_m_min"] - target["wfs_m_min"]),
                               SETUP_WFS_TOL),
        voltage=net_of_start(abs(volts - target["voltage_v"]), abs(start["voltage_v"] - target["voltage_v"]),
                             SETUP_VOLTS_TOL),
        stickout=net_of_start(abs(stickout_m - SETUP_STICKOUT), abs(start["stickout_m"] - SETUP_STICKOUT),
                              SETUP_STICKOUT_TOL))
    p = _mean([v for v in terms.values() if v is not None])*(FAULT_FACTOR if faulted else 1.)
    return dict(progress=float(p), terms=terms, fault_factor=FAULT_FACTOR if faulted else 1.)


def _eval_set_up_gun(e):
    """The source's set points as its state holds them (what the tool_io outputs set, not a command log), the wire's
    stick-out, no arc, the gun's hardware clear of everything (the wire may touch: touch sensing measures the
    stick-out against the plate), the gas valve closed at the end (a purge during the episode is allowed)."""
    env, w = e.env, e.env.world
    st, s = w.process.src.st, w.process.src.s
    target = PR.v(PR.SCHEDULES)[SETUP_PROCEDURE]
    contacts = e._gun_contacts()
    h = env.history
    e._measured = dict(wfs=float(s.wfs_m_min), volts=float(s.voltage_v), stickout=float(st.l_s),
                       faulted=bool(h["gun_contact_ticks"] or h["arc_s"] or contacts))
    return {
        "wire_feed": dict(value_m_min=float(s.wfs_m_min), target_m_min=target["wfs_m_min"],
                          tolerance_m_min=SETUP_WFS_TOL,
                          ok=_ok(abs(s.wfs_m_min - target["wfs_m_min"]) <= SETUP_WFS_TOL + 1e-9)),
        "voltage": dict(value_v=float(s.voltage_v), target_v=target["voltage_v"], tolerance_v=SETUP_VOLTS_TOL,
                        ok=_ok(abs(s.voltage_v - target["voltage_v"]) <= SETUP_VOLTS_TOL + 1e-9)),
        "stickout": dict(value_mm=float(st.l_s*1e3), target_mm=SETUP_STICKOUT*1e3,
                         tolerance_mm=SETUP_STICKOUT_TOL*1e3,
                         ok=_ok(abs(st.l_s - SETUP_STICKOUT) <= SETUP_STICKOUT_TOL + 1e-9)),
        "no_arc": dict(arc_s=h["arc_s"], ok=_ok(h["arc_s"] == 0.)),
        "no_contact": dict(contacts=contacts, wire_touched=bool(h["wire_touch"]),
                           gun_contact_ticks=h["gun_contact_ticks"],
                           ok=_ok(not contacts and h["gun_contact_ticks"] == 0)),
        "gas_off": dict(gas_on=bool(st.gas_on), ok=_ok(not st.gas_on)),
    }


def _progress_set_up_gun(e):
    m = e._measured
    return setup_progress(m["wfs"], m["volts"], m["stickout"], e._start_measures(), m["faulted"])

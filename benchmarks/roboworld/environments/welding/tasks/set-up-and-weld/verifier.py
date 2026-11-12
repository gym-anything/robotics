"""Outcome of set-up-and-weld: fillet-2f's weld, judged unchanged (a continuous 1/4 in fillet along the near side,
end to end, crater filled), and the source at the spray procedure at the end (task.SETUP_PROCEDURE within
SETUP_WFS_TOL / SETUP_VOLTS_TOL, as set-up-gun judges it).

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle=True, the on-demand
evaluation, judges a weld still freezing with the arc off as it freezes hands-off) and returns the evaluation:
success, score, checks, progress, reward, judged, flags and diagnostics (judge_set_up_and_weld below on
Evaluator._final_rows and the source's set points). WELD: the task is judged on the weld's sections (the evaluator
reads the start state's raw progress, Evaluator.start_raw_progress)."""
from welding import params as PR
from welding.task import SETUP_PROCEDURE, SETUP_VOLTS_TOL, SETUP_WFS_TOL, _ok, fillet_progress

WELD = True


def verify(e, settle=False):
    rows, judged, flags = e._final_rows(settle)
    s = e.env.world.process.src.s
    return judge_set_up_and_weld(e, rows, judged, flags, wfs_m_min=float(s.wfs_m_min), voltage_v=float(s.voltage_v))


def judge_set_up_and_weld(e, rows, judged=None, flags=None, wfs_m_min=0., voltage_v=0.):
    """set-up-and-weld (A14): every fillet-2f check on the weld, unchanged, and the source at the procedure at the
    end (12.7 +- 0.3 m/min, 29 +- 0.5 V: set-up-gun's tolerances). Progress: fillet_progress net of the tacked start,
    as fillet-2f's (the weld is the job; a weld laid at the short-circuit job's settings is judged as what it is)."""
    target = PR.v(PR.SCHEDULES)[SETUP_PROCEDURE]
    checks = e._eval_fillet_2f(rows)
    diagnostics = dict(e.diagnostics)
    checks["source_at_procedure"] = dict(
        wire_feed_m_min=wfs_m_min, voltage_v=voltage_v, target=dict(wire_feed_m_min=target["wfs_m_min"],
                                                                    voltage_v=target["voltage_v"]),
        tolerance=dict(wire_feed_m_min=SETUP_WFS_TOL, voltage_v=SETUP_VOLTS_TOL),
        ok=_ok(abs(wfs_m_min - target["wfs_m_min"]) <= SETUP_WFS_TOL + 1e-9
               and abs(voltage_v - target["voltage_v"]) <= SETUP_VOLTS_TOL + 1e-9))
    checks = e._named(checks)
    reward = fillet_progress(rows, e.start_raw_progress, e._last_craters)
    return e._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                     flags=flags or {}, diagnostics=diagnostics)

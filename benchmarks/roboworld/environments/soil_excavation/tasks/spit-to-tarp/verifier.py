"""Outcome of spit-to-tarp: at least 1.5 kg of the bed's soil on the tarp and no more than 15 % of the soil lifted out
spilled elsewhere (released: with the spade still in both hands).

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
from soil_excavation.task import ORIGIN_BLOCK, crit, diag, tool_held
from soil_excavation.task_set_v2 import clip, ramp


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _spit_to_tarp(e, lg, final)
    return read(e, lg)


def _spit_to_tarp(e, lg, final):
    bed = lg.origin == ORIGIN_BLOCK
    tarp = lg.on_tarp() & bed
    m_tarp = float(lg.mass[tarp].sum())
    out = lg.out_of_block() & bed & ~tarp
    m_spill = float(lg.mass[out].sum())
    frac = m_spill/max(m_spill + m_tarp, 1e-9)
    held = tool_held(e.env, "spade")
    c = [crit("soil_on_tarp", m_tarp, m_tarp >= 1.5, ">= 1.5 kg of bed soil on the tarp (a half spit)", "kg"),
         crit("spill_fraction", frac, frac <= 0.15 or m_tarp + m_spill < 1e-6, "<= 15 % of the soil lifted out", "-"),
         crit("spade_held", held, all(held.values()), "both hands on the spade")]
    return c, [min(1., m_tarp/1.5), max(0., 1 - frac/0.5), float(all(held.values()))]


# soil-tasks-v1's and soil-tasks-v2's reading (the same in task_set.py and task_set_v2.py)
def read(e, lg):
    c, _ = _spit_to_tarp(e, lg, False)
    bed, tarp = lg.origin == ORIGIN_BLOCK, lg.on_tarp()
    mass = float(lg.mass[bed & tarp].sum())
    spill = float(lg.mass[bed & ~tarp & lg.out_of_block()].sum())
    frac = spill/max(mass + spill, 1e-9)
    held = c.pop()
    return c, clip(mass/1.5)*ramp(frac, .15, .50), [diag("spade_held", held["value"], "reported only")]

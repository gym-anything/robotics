"""Outcome of titration, re-evaluated at every physics tick and never latched: the flask's sodium has reached its own
chloride and exceeds it by at most 0.05 mL of base (titrant on its way counted as landed), the pink has persisted
throughout the solution for 30 s, at most 0.05 mL of stray liquid and at most 0.05 mL of the aliquot lost.

verify(session) receives the released session (session.ApparatusSession, built by setup.py) once its liquid is charged
with an acid aliquot, gives it its evaluator, task.EndpointEvaluator (the checks and the progress score above) over
task_readings.TaskReadings, evaluates tick zero and returns nothing; the session updates the evaluator at every
physics tick and reports its result. The evaluator stays in task.py: the seven other tasks compose it
(task_evaluators.CatalogueEvaluator), and the liquid checkpoint hashes that file (liquid_snapshot.source_identity)."""


def verify(session):
    from titrate.newton15.task import EndpointEvaluator
    from titrate.newton15.task_readings import TaskReadings
    r = session.liquid.reagents
    session.evaluator = EndpointEvaluator(session.dt_s, r.aliquot_ml, r.acid_molar, r.base_molar)
    session.task_readings = TaskReadings(session.liquid)
    session.evaluator.update(0, **session.task_readings.read())

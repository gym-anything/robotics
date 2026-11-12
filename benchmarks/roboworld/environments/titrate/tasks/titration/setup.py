"""Start of titration: the selected variation's bench and charge as they are, nothing moved or run in beforehand: the
acid aliquot, diluted with water and phenolphthalein, in the flask on the stirrer under the burette, and the burette
charged with sodium hydroxide.

setup(config, **kwargs) receives task_session.make_session's arguments and returns the released session
(session.ApparatusSession), which charges the liquid itself and attaches the evaluator (verifier.py)."""


def setup(config=None, **kwargs):
    from titrate.newton15.session import ApparatusSession
    return ApparatusSession(config, **kwargs)

"""sewing/params.py's ledger: every entry has a known label and a source, the solver's defaults are its values, and
every named constant is read or its entry says why not."""


from sewing import params as P
from tests.packages import PACKAGES

LABELS = {"read", "abstract", "search", "literature", "repo", "measured", "derived", "fitted", "estimate", "authored", "open",
          "prior"}


def test_every_entry_is_labelled():
    for key, v in P.LEDGER.items():
        assert all(part in LABELS for part in v["label"].split("/")), (key, v["label"])
        assert v["source"].strip(), key


def test_solver_defaults_match_the_ledger():
    """cloth.py's compression defaults are the ledger's values (the solver cannot import the ledger's labels)."""
    from sewing import cloth as C
    assert abs(C.KES_PM - P.KES_PM) < 1e-9 and abs(C.COMP_LC - P.COMP_LC) < 1e-12
    s = C.ClothSolver.__new__(C.ClothSolver)
    C.ClothSolver.__init__(s, device="cpu")
    assert abs(s.set_recovery_tau - P.SET_RECOVERY_TAU) < 1e-12 and abs(s.set_recovery_pa - P.SET_RECOVERY_PA) < 1e-9


def test_every_ledger_constant_is_consumed_or_says_why():
    """Each named ledger constant is read by the model (outside params.py, or by another ledger entry or fabric), or its
    entry says why it is not (an unread parameter is an unwired formula or an unbuilt mechanism)."""
    import re
    here = PACKAGES / "sewing"
    src = (here/"params.py").read_text()
    code = "\n".join(p.read_text() for p in here.rglob("*.py") if p.name != "params.py" and "tests" not in p.parts)
    missing = []
    for name in re.findall(r"^([A-Z][A-Z0-9_]*) = P\(", src, re.M):
        used = re.search(r"\b" + name + r"\b", code) or len(re.findall(r"\b" + name + r"\b", src)) > 1
        key = re.search(name + r' = P\("([^"]+)"', src).group(1)
        note = P.LEDGER[key].get("note", "")
        src_text = P.LEDGER[key]["source"]
        if not used and "not consumed" not in note and "rejected" not in src_text:
            missing.append(name)
    assert not missing, missing

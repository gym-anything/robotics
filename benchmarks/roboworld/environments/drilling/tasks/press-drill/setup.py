"""Start of press-drill: the steel flat bar lies loose on the bench beside the drill press (its yaw within 6 degrees),
its one mark already centre-punched; the press holds a 9 mm drill and its vise stands open.

setup(s) receives layout.BenchStart (layout.bench_layout calls it), appends the bar to s.parts and puts its mark in
s.L["marks"]; it returns nothing."""
from drilling.layout import flat_bar


def setup(s):
    s.L["marks"] = flat_bar(s.parts, s.rng, punched=True)

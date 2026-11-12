"""Start of open-carton: a corrugated supply carton sealed with packing tape (a folded-back tab at one end) standing on
the stainless receiving table, its contents inside (scene.receiving, carton.settled); seeds move the carton
(variants.placement).

setup(spec, contract) gets the receiving preset's scene spec (scene.SceneSpec) before the contact world is built;
scene.build calls it with contract None for every selection of open-carton (the task-set contracts start it as
released). Nothing in the start is open-carton's alone (open-flaps builds on the same world), so it changes nothing."""


def setup(spec, contract):
    pass

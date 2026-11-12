"""Start of lift-lid: a two-piece gift box closed on the dining table, the photo frame inside its base (scene.home);
seeds draw the box, its lid clearance (variants.home_variant) and its pose on the table (variants.placement).

setup(spec, contract) gets the home preset's scene spec (scene.SceneSpec) before the contact world is built;
scene.build calls it with contract None for every selection of lift-lid (the task-set contracts start it as released).
Nothing in the start is lift-lid's alone (put-lid-on builds on the same world), so it changes nothing."""


def setup(spec, contract):
    pass

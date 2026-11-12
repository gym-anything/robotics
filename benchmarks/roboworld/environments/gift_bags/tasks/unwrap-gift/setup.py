"""Start of unwrap-gift: the gift box wrapped in taped paper on the dining table, as a person leaves a present
(scene.home's wrapped present, wrapmap.relaxed_gift, which open-present starts from too); seeds draw the box, its lid
clearance, the wrap stock (variants.home_variant) and the present's pose on the table (variants.placement).

setup(spec, contract) gets the home preset's scene spec (scene.SceneSpec) before the contact world is built;
scene.build calls it with contract None for every selection of unwrap-gift (the task-set contracts start it as
released). Nothing in the start is unwrap-gift's alone, so it changes nothing."""


def setup(spec, contract):
    pass

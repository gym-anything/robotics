"""Start of empty_tub: the bussing tub on the counter holds 16 empty cans lying in two layers; the bag already holds
48 cans standing in rows and the reusable cups stand on their tray.

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans) and leaves it as it is: the targets lie in the
tub (assembly.can_inventory) and the cups stand on their tray (props.build)."""

INVENTORY=dict(target_count=16,tub_count=16,preload_count=48)

def setup(s):
    pass

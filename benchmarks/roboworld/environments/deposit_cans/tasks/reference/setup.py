"""Start of reference: 24 empty cans, 16 lying in two layers in the bussing tub and 8 standing in two rows on the
return counter; the bag already holds 48 cans standing in rows and the reusable cups stand on their tray.

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans) and leaves it as it is: the targets start in the
tub and on the counter (assembly.can_inventory) and the cups on their tray (props.build)."""

INVENTORY=dict(target_count=24,tub_count=16,preload_count=48)

def setup(s):
    pass

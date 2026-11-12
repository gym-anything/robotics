"""Start of sort_cups: eight empty cans stand in two rows on the return counter with the three reusable cups among
them on the right-hand row; the tray stays empty in its place and the bag already holds 48 cans standing in rows.

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans) and sets s.cup_starts, the cups' bases
where props.build stands them."""

INVENTORY=dict(target_count=8,tub_count=0,preload_count=48)
# Task-specific start placements, world metres (z of a cup is its base). Counter cans stand at
# x = -0.60 + 0.15 (j % 4), y = -0.53 (j < 4) or +0.53 (assembly.can_inventory); arm bases at (0.12, +-0.65, 1.02).
# sort_cups: the three reusable cups stand midway between the counter cans of the right-hand row (4.5 mm from a
# can at the cup rim), all on the right arm's side, which also reaches the tray at (-0.43, 0.375): no hand-off.
SORT_CUPS_ON_COUNTER=((-.525,.53,.903),(-.375,.53,.903),(-.225,.53,.903))

def setup(s):
    s.cup_starts=SORT_CUPS_ON_COUNTER

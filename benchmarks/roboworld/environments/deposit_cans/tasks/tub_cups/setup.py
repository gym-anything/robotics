"""Start of tub_cups: the bussing tub holds eight empty cans lying in its bottom rows and the three reusable cups
standing on its floor; the tray stays empty in its place and the bag already holds 48 cans standing in rows.

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans) and sets s.cup_starts, the cups' bases
where props.build stands them."""

INVENTORY=dict(target_count=8,tub_count=8,preload_count=48)
# Task-specific start placements, world metres (z of a cup is its base). Counter cans stand at
# x = -0.60 + 0.15 (j % 4), y = -0.53 (j < 4) or +0.53 (assembly.can_inventory); arm bases at (0.12, +-0.65, 1.02).
# tub_cups: the three reusable cups stand upright on the tub floor's flat +y end (tub-frame x -0.10, 0, 0.10;
# y 0.15, beyond the floor ribs at |y| < 0.095), 4 mm above it; the tub's eight cans are its bottom rows 0-3.
TUB_CUPS=((-.47,.15,.909),(-.37,.15,.909),(-.27,.15,.909))

def setup(s):
    s.cup_starts=TUB_CUPS

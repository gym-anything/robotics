"""Start of sort_waste: six empty cans and two used paper coffee cups stand on the return counter, the paper cups in
the right-hand row; the bag already holds 48 cans standing in rows and the reusable cups stand on their tray.

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans) and sets s.paper_cups, the paper cups'
bases where props.build stands them."""

INVENTORY=dict(target_count=6,tub_count=0,preload_count=48)
# Task-specific start placements, world metres (z of a cup is its base). Counter cans stand at
# x = -0.60 + 0.15 (j % 4), y = -0.53 (j < 4) or +0.53 (assembly.can_inventory); arm bases at (0.12, +-0.65, 1.02).
# sort_waste: two single-use paper cups on the counter's right-hand row, in the two slots the six cans leave free.
PAPER_CUPS=((-.30,.53,.903),(-.15,.53,.903))

def setup(s):
    s.paper_cups=PAPER_CUPS

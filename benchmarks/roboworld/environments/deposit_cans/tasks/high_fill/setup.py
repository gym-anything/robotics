"""Start of high_fill: the reference start (16 cans lying in the tub, 8 standing on the counter) with 120 cans
already in the bag, lying in the authored resting pile of bag_pile.py (scene revisions before 7 stand them in rows).

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans). From scene revision 7 it sets s.place_preload
(bag_pile.pose), with which assembly.can_inventory places each pre-existing can: it receives the can's index and its
seeded yaw, and returns the can's centre and rotation."""

INVENTORY=dict(target_count=24,tub_count=16,preload_count=120)

def setup(s):
    # high_fill starts with its pre-existing cans lying in the bag (bag_pile.py), not as
    # the floating upright grid of assembly.can_inventory, which the reference preset keeps.
    if s.config.scene_revision>=7:
        from deposit_cans.bag_pile import pose as pile_pose
        s.place_preload=pile_pose

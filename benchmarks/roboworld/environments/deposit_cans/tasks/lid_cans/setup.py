"""Start of lid_cans: three empty cans stand upright on the recycling station lid's flat roof; the bag already holds
48 cans standing in rows and the reusable cups stand on their tray.

INVENTORY is the variant's catalogue inventory (variants.PRESETS): its target cans, how many of them start in the
tub and the cans already in the bag. setup(s) receives the assembly.Start the Assembly is built on (Assembly.__init__
calls it before the props and cans) and sets s.place_target, with which
assembly.can_inventory places each target can: it receives the configuration, the can's index, its height and the
seeded generator, and returns the can's centre, rotation and origin."""
import numpy as np
from scipy.spatial.transform import Rotation

INVENTORY=dict(target_count=3,tub_count=0,preload_count=48)
# Task-specific start placements, world metres (z of a cup is its base). Counter cans stand at
# x = -0.60 + 0.15 (j % 4), y = -0.53 (j < 4) or +0.53 (assembly.can_inventory); arm bases at (0.12, +-0.65, 1.02).
# lid_cans: the three target cans stand upright on the station lid's flat roof (station-frame xy; the roof is
# flat at z 1.016 between the slot edge |y| 0.05715 and the roof edge |y| 0.2276, station.lid_mesh revision 3).
STATION_XY=(.49,0.)
LID_ROOF_Z=1.016
LID_CANS_XY=((-.12,-.14),(.12,-.14),(0.,.14))

def place_target(c,i,h,rng):
    # lid_cans: the targets stand upright on the station lid's flat roof, 5 mm above it (the counter cans' clearance).
    x,y=LID_CANS_XY[i-c.tub_count]
    pos=np.array([STATION_XY[0]+x,STATION_XY[1]+y,LID_ROOF_Z+.002+h/2+.003])
    rot=Rotation.from_euler('z',rng.uniform(-np.pi,np.pi));origin='lid'
    return pos,rot,origin

def setup(s):
    s.place_target=place_target

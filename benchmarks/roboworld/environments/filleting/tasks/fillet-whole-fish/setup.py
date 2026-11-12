"""Start of fillet-whole-fish: the gutted trout on its side on the board, the fillet knife in the right hand's insert
grasp, the perforated fillet tray on the holding-arm side, the empty frame pan at the front left of the bench and the
stock ice pan right of the board; seeds vary the fish's length, its rib and pin-bone counts and its pose.

setup() gives the choices this start makes (layout.task_start): collision_profile, the robots' collision thresholds
when the selection names none (libfranka's contact-task thresholds, as for every cutting task:
variants.collision_profile); bench, where this scene puts the fillet tray and the ice pan (layout.fillet_tray,
ice_pan_center); props, the bench props only this scene has (the frame pan, layout.FRAME_PAN)."""

# The perforated fillet tray stands on the holding-arm side, where the stock ice pan stands on the common bench, 325 mm
# along x, so the only hand that can pinch a fillet (the left: the right fingers carry the knife-handle inserts) can lay
# both fillets flat side by side. At the common position no side-by-side set-down point solves for the left arm; here
# every point does (kinematics only). The stock ice pan takes the tray's common place, right of the board. Their
# geometry is unchanged.
BENCH = {"FILLET_TRAY": (-0.505, 0.42), "ICE_PAN": (0.605, 0.30)}


def setup():
    return dict(collision_profile="contact", bench=BENCH, props=("FRAME_PAN",))

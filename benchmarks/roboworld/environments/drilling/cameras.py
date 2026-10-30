"""Camera placement for both presets (policy and diagnostic views). Cameras carry no bodies: moving a camera never
changes the physics. Every placement was checked on renders of the start states.

Field of view is horizontal (render._cameras: 36 mm aperture).
"""
import math


def wall_cameras(x_c, z_marks):
    """x_c: the centre of the marked field and of the cart (layout.wall_layout); z_marks: the marks' mean height. The
    arms stand at x_c +-0.28 m with their hands 0.27 m in front of the wall at z 1.25-1.41 m; the marks lie between
    them at z 1.0-1.5 m."""
    return dict(
        # a fixed job-site camera high on a mast at the cart's rear, between and behind the arm bases, 1.7 m above
        # the deck (its lens 1.4 m from either shoulder: outside the FR3's 0.855 m reach), looking down at the marked
        # field from just right of centre. Chosen by rendering candidates at recorded frames of the robot wall-drill run
        # (start, hammer at the mark, drilling at 25 mm and 62 mm): it sees all four marks in the ready pose, and while
        # drilling it sees the bit enter its mark obliquely from above - a bit off square in yaw or pitch changes the
        # bit's direction in the image - with the other marks clear. Rejected (what blocked them, looked at in the
        # renders):
        #   * a tripod left of the left arm, 0.38 m off the wall (eye x_c - 0.80, 2.62, 1.48): the left
        #     arm's shoulder and upper arm filled about 45 % of the frame in the ready pose and the hammer and arm
        #     the right half while drilling;
        #   * high beside either arm (x_c -+ 1.05, 2.45, 2.05) and high or mid-height close to the wall plane
        #     (x_c -+ 1.2, 2.70-2.75, 1.95-2.35; x_c - 1.35, 2.55, 2.45): at the grazing angle the marks lie behind the
        #     near arm, which fills half the frame;
        #   * straight above between the arms (x_c, 2.30, 2.45) and further back (a tripod behind the cart,
        #     x_c, 1.50, 2.55; a mast at the cart's rear edge, x_c, 1.85, 2.35): all marks in the ready pose, but the
        #     bit is seen end-on (pitch unreadable) and the left elbow covers the drilled mark;
        #   * high to the right (x_c + 0.55, 2.25, 2.45; x_c + 0.85, 2.05, 2.55): squareness reads well, but the left
        #     arm hides one or two marks in the ready pose.
        station_rgb=dict(eye=[x_c + 0.10, 2.20, 2.50], target=[x_c - 0.05, 3.0, 1.28], up=[0, 0, 1], fov=55.,
                         role="policy", width=960, height=540),
        overview_rgb=dict(eye=[x_c + 2.2, 0.6, 2.75], target=[x_c, 2.55, 1.05], up=[0, 0, 1], fov=55., role="policy",
                          width=960, height=540),
        hero=dict(eye=[x_c + 1.6, 1.0, 1.75], target=[x_c, 2.7, 1.1], up=[0, 0, 1], fov=50., role="diagnostic"),
        # between the arms, 0.40 m off the wall and 0.42 m above the marks: every mark, hole and anchor of the field
        # (from the hands' height it saw one mark; from below a ledger hid the wall)
        wall_macro=dict(eye=[x_c, 2.60, z_marks + 0.42], target=[x_c, 3.0, z_marks], up=[0, 0, 1], fov=70.,
                        role="diagnostic"),
        # establishing view from the room's south-east quarter: the bay, its cart and the workshop around it (visual
        # review of the room dressing; not a policy camera)
        room=dict(eye=[3.1, -2.1, 2.45], target=[-0.3, 2.3, 1.05], up=[0, 0, 1], fov=72., role="diagnostic"),
    )


def bench_cameras(sx, sy, focus):
    """(sx, sy): the drill-press spindle; focus: the task's first workpiece or hand tool (bench_macro looks at it)."""
    fx, fy, fz = focus
    # a workpiece under the spindle (in the vise) is seen low from the front, under the press head: from the usual
    # high offset the camera sat inside the right arm's joint cap (a black wedge over half the frame)
    under = math.hypot(fx - sx, fy - sy) < 0.05
    # elsewhere it looks steeply down on the marked face (62 degrees), as one inspects a layout, from the side on which
    # the arms' ready poses leave the most room over every bench task's workpiece (0.21 m to the nearest link origin;
    # 0.11 m straight above the focus the camera sat inside the left hand and press-drill's view was black). From a
    # lower angle (41 degrees, from the other side) the upright leg of match-bolt's bracket hid the marks on
    # its flange
    macro_eye = [fx - 0.30, fy + 0.22, fz + 0.12] if under else [fx + 0.156, fy - 0.09, fz + 0.34]
    return dict(
        station_rgb=dict(eye=[sx + 0.36, sy + 0.30, 1.30], target=[sx, sy, 1.02], up=[0, 0, 1], fov=55., role="policy",
                         width=960, height=540),
        overview_rgb=dict(eye=[1.35, 0.9, 2.7], target=[3.45, -0.75, 0.95], up=[0, 0, 1], fov=55., role="policy",
                          width=960, height=540),
        hero=dict(eye=[2.15, 0.85, 1.85], target=[3.45, -1.0, 1.05], up=[0, 0, 1], fov=55., role="diagnostic"),
        press_side=dict(eye=[sx - 0.05, sy - 0.55, 1.15], target=[sx, sy, 1.05], up=[0, 0, 1], fov=45., role="diagnostic"),
        bench_macro=dict(eye=macro_eye, target=[fx, fy, fz], up=[0, 0, 1], fov=45.,
                         role="diagnostic"),
        # establishing view from the room's north-west quarter: bench, press, cart and the workshop around them
        room=dict(eye=[-1.3, 2.3, 2.35], target=[3.4, -0.8, 1.0], up=[0, 0, 1], fov=72., role="diagnostic"),
    )

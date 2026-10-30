"""Watertight solid meshes for the lock, built by boolean subtraction.

A pin tumbler plug is a cylinder with vertical chambers bored through it, and
the shell is a body with a horizontal bore for the plug and vertical chambers
above. Neither is convex, so a collider needs either a signed-distance-field or
a convex decomposition, and both need a real watertight mesh first.
trimesh with the manifold3d backend produces those meshes exactly; this module
builds them and hands back (vertices, faces) plus a helper that authors a
UsdGeom.Mesh with SDF collision.

Frame: X = plug axis, Z = up, chambers vertical. The plug centre line is the
X axis at z = 0, so the shear plane (top of the plug) is z = +PLUG_R.
"""
import numpy as np
import trimesh

import lock_specs as LK


def _cyl_x(radius, length, cx=0.0, sections=96):
    """a cylinder whose axis is X, centred at (cx, 0, 0)."""
    c = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    # trimesh cylinders are built along Z; rotate Z->X
    c.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2.0, [0, 1, 0]))
    c.apply_translation([cx, 0, 0])
    return c


def _cyl_z(radius, length, cx=0.0, cz=0.0, sections=48):
    """a cylinder whose axis is Z (a vertical bore), centred at (cx, 0, cz)."""
    c = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    c.apply_translation([cx, 0, cz])
    return c


def _countersink(cx, z_mouth, bore_r, depth, opening, sections=48,
                 margin=0.6e-3):
    """A conical countersink to subtract at a chamber mouth. `z_mouth` is the
    shear plane. The cone passes through radius bore_r+depth at the mouth and
    tapers to a point `depth` into the chamber, and its wide end pokes `margin`
    PAST the mouth so the boolean never leaves a face coplanar with the plug
    or shell surface (which makes the result non-watertight).
    `opening=-1`: widens below the mouth (plug chamber, mouth faces up).
    `opening=+1`: widens above the mouth (shell chamber, mouth faces down).
    The removed region relieves the shear edge so a pin near shear cams clear
    instead of hard-binding -- a real countersunk chamber mouth."""
    # a true 45-degree bevel of the top bore edge: the cone rises at slope 1
    # from radius 0 at the apex to bore_r+depth at the mouth, so the apex sits
    # (bore_r+depth) away from the mouth. It pokes `margin` past the mouth.
    H = bore_r + depth + margin
    Rb = H                                   # 45 degrees -> radius == height
    c = trimesh.creation.cone(radius=Rb, height=H, sections=sections)
    # cone: base (r=Rb) at local z=0, apex (r=0) at local z=H
    if opening < 0:
        # apex below the mouth by (bore_r+depth); base pokes `margin` above
        c.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0]))
        # after flip: base at z=0, apex at z=-H; place apex at z_mouth-(bore_r+depth)
        c.apply_translation([cx, 0, z_mouth - (bore_r + depth) + H])
    else:
        # apex above the mouth by (bore_r+depth); base pokes `margin` below
        c.apply_translation([cx, 0, z_mouth + (bore_r + depth) - H])
    return c


def make_plug(length, chambers_x, bore_r, keyway_hw=None, keyway_h=None,
              ledge_z=None, mouth_chamfer=None):
    """Plug = solid cylinder (axis X) minus a vertical blind chamber at each
    chamber X, optionally minus a horizontal keyway slot along the axis.
    `chambers_x` are the chamber centre X positions (metres, negative into
    the plug). Returns a watertight trimesh.

    `ledge_z`, if given, is a list of key-surface heights (one per chamber):
    a small ledge is UNIONED into the chamber at that height so the key pin
    rests on it. Because the ledge is part of the plug solid, it rotates with
    the plug -- the faithful stand-in for a key cut that turns with the key,
    without a separate kinematic body (which would not follow the rotation)."""
    plug = _cyl_x(LK.PLUG_R, length)
    bore_len = LK.PLUG_R + 2.0e-3
    for cx in chambers_x:
        bore = _cyl_z(bore_r, bore_len, cx=cx, cz=0.5 * bore_len - 0.2e-3)
        plug = plug.difference(bore)
    # unions first (they are the fragile boolean op in the manifold backend),
    # then all subtractions last so a difference is never fed into a union
    if ledge_z is not None:
        led_r = bore_r - 0.02e-3
        for cx, zt in zip(chambers_x, ledge_z):
            # a short cylindrical plug of material capping the chamber from
            # below up to height zt (its top face = the key surface)
            h = zt - (-LK.PLUG_R)          # from well below up to the surface
            led = _cyl_z(led_r, h, cx=cx, cz=zt - h * 0.5)
            plug = plug.union(led)
    if keyway_hw is not None:
        slot = trimesh.creation.box(
            extents=[length + 2.0e-3, 2.0 * keyway_hw, keyway_h])
        slot.apply_translation([0.0, 0.0, -keyway_h * 0.5 + 0.3e-3])
        plug = plug.difference(slot)
    if mouth_chamfer:
        for cx in chambers_x:
            # countersink the chamber mouth at the shear plane (plug top) so a
            # pin near shear cams clear -- the real relieved shear edge
            cs = _countersink(cx, LK.PLUG_R, bore_r, mouth_chamfer, opening=-1)
            plug = plug.difference(cs)
    return plug


def _round_box(dx, dy, dz, r=0.003, sections=16):
    """a box with rounded vertical (Z) edges, centred at origin."""
    import shapely.geometry as sg
    rect = sg.box(-dx / 2, -dy / 2, dx / 2, dy / 2).buffer(
        r, join_style=1, resolution=sections // 4)
    body = trimesh.creation.extrude_polygon(rect, height=dz)
    body.apply_translation([0, 0, -dz / 2])
    return body


def make_padlock_body(plug_len, chambers_x, bore_r, keyway_hw, keyway_z,
                      keyway_h, *, align_shackle_bores=False):
    """The padlock body: a rounded brass block with a horizontal cylindrical
    bore for the plug (axis X, keyway on the +X front face), five vertical
    chambers above the bore for the driver stacks, a keyway slot cut into the
    front, and two vertical holes in the top for the shackle legs. This IS the
    shell of the cylinder -- exactly as a real padlock body is. Returns a
    watertight trimesh plus the shackle-leg X/Y positions."""
    xf = plug_len / 2 + 0.003            # front face (keyway) X
    xb = -plug_len / 2 - 0.006           # back X
    W = 0.030                             # body width (Y)
    z_bot = -0.014
    z_top = 0.040
    body = _round_box(xf - xb, W, z_top - z_bot, r=0.004)
    body.apply_translation([0.5 * (xf + xb), 0, 0.5 * (z_top + z_bot)])
    # plug bore (through, along X)
    plug_bore = _cyl_x(LK.PLUG_R + 0.06e-3, (xf - xb) + 0.004,
                       cx=0.5 * (xf + xb))
    body = body.difference(plug_bore)
    # chambers up from the bore, and the shackle-side clearance
    for cx in chambers_x:
        ch = _cyl_z(bore_r, 0.030, cx=cx, cz=0.010)
        body = body.difference(ch)
    # keyway slot in the front face into the plug bore
    slot = trimesh.creation.box(extents=[0.008, 2 * keyway_hw, keyway_h])
    slot.apply_translation([xf - 0.002, 0, keyway_z])
    body = body.difference(slot)
    # shackle legs: two vertical holes in the top (one open long side to slide)
    leg_y = 0.010
    holes = []
    for sy, depth in ((+leg_y, 0.050), (-leg_y, 0.030)):
        h = _cyl_z(0.0032, depth, cx=0.5 * (xf + xb), cz=z_top - depth / 2 + 1e-4)
        if align_shackle_bores:
            # _cyl_z always centres Y at zero. The retained legacy default is
            # unchanged; corrected assemblies must offset EACH leg's cutter.
            h.apply_translation([0., sy, 0.])
        holes.append(h)
    for h in holes:
        body = body.difference(h)
    return body, (0.5 * (xf + xb), leg_y)


def make_shackle(cx, leg_y, z_top=0.040, dia=0.006, span=None, height=0.024):
    """A steel U shackle: two vertical legs at x=cx, y=+-leg_y, joined by a
    semicircular arch above the body. Returns a trimesh."""
    r = dia / 2
    if span is None:
        span = 2 * leg_y
    parts = []
    for sy in (+leg_y, -leg_y):
        leg = trimesh.creation.cylinder(radius=r, height=height + 0.012)
        leg.apply_translation([cx, sy, z_top + (height + 0.012) / 2 - 0.010])
        parts.append(leg)
    # arch: a torus half joining the leg tops
    arch = trimesh.creation.torus(major_radius=leg_y, minor_radius=r,
                                  major_sections=48, minor_sections=12)
    arch.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]))
    arch.apply_translation([cx, 0, z_top + height + 0.002])
    parts.append(arch)
    return trimesh.util.concatenate(parts)


def make_plug_convex(length, chambers_x, bore_r, ledge_z, sections=96):
    """The plug as an EXPLICIT convex decomposition (no SDF). `chambers_x` and
    `ledge_z` are lists (one per chamber). The cylinder is split by the planes
    y=+-bore_r into two side pieces, the middle strip |y|<bore_r is divided along
    X into the solid segments BETWEEN the chambers, and each chamber gets a ledge
    box filling it below its ledge height. Every piece is convex (a cylinder cut
    by planes, or a box), so PhysX collides them robustly -- the fix for the SDF
    rotating-contact pathologies. Returns a list of trimeshes to attach as
    convexHull child colliders of one plug body."""
    if not isinstance(chambers_x, (list, tuple)):
        chambers_x, ledge_z = [chambers_x], [ledge_z]
    order = sorted(range(len(chambers_x)), key=lambda i: chambers_x[i])
    xs = [chambers_x[i] for i in order]
    lz = [ledge_z[i] for i in order]
    cyl = _cyl_x(LK.PLUG_R, length, sections=sections)
    BIG = LK.PLUG_R * 4
    hw = bore_r

    def region(xlo, xhi, ylo, yhi):
        if xhi - xlo < 1e-6:
            return None
        b = trimesh.creation.box(extents=[xhi - xlo, yhi - ylo, 4 * LK.PLUG_R])
        b.apply_translation([0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.0])
        return cyl.intersection(b)

    pieces = [region(-BIG, BIG, -BIG, -hw), region(-BIG, BIG, hw, BIG)]
    # solid segments of the middle strip between the chambers
    edges = [-BIG]
    for x in xs:
        edges += [x - hw, x + hw]
    edges += [BIG]
    for i in range(0, len(edges), 2):
        pieces.append(region(edges[i], edges[i + 1], -hw, hw))
    # ledge box per chamber
    for x, z in zip(xs, lz):
        led = trimesh.creation.box(extents=[2 * hw, 2 * hw, (z + LK.PLUG_R)])
        led.apply_translation([x, 0.0, 0.5 * (z - LK.PLUG_R)])
        pieces.append(cyl.intersection(led))
    return [p for p in pieces if p is not None and len(p.vertices) >= 4]


def make_keyway_cut(length, keyway_hw, keyway_h, keyway_z, grooves,
                    clearance=0.0):
    """The negative to subtract from the plug to form a WARDED keyway: a slot
    (width 2*keyway_hw in Y, height keyway_h in Z, along X at z=keyway_z) MINUS
    the ward-ridge boxes, so the ridges remain as plug material protruding into
    the slot. `grooves` is a list of (face, z_centre, width, depth): face +1 =
    +Y wall, -1 = -Y wall. A matching key has grooves that clear the ridges; a
    foreign blank collides with them. Returns a trimesh."""
    slot = trimesh.creation.box(extents=[length, 2 * keyway_hw, keyway_h])
    slot.apply_translation([0.0, 0.0, keyway_z])
    ridges = []
    for (face, z_c, w, depth) in grooves:
        # ridge slightly undersized by clearance so the true key slides
        d = max(depth - clearance, 1e-4)
        ww = max(w - 2 * clearance, 1e-4)
        if face > 0:
            rb = trimesh.creation.box(extents=[length + 2e-3, 2 * d, ww])
            rb.apply_translation([0.0, keyway_hw - d, z_c])
        else:
            rb = trimesh.creation.box(extents=[length + 2e-3, 2 * d, ww])
            rb.apply_translation([0.0, -(keyway_hw - d), z_c])
        ridges.append(rb)
    for rb in ridges:
        slot = slot.difference(rb)
    return slot


def make_shell(length, chambers_x, bore_r, plug_clear,
               outer_hw=None, top_z=None, mouth_chamfer=None):
    """Shell = solid block minus a horizontal bore (for the plug) minus a
    vertical chamber above each pin position. The plug bore radius is
    PLUG_R + plug_clear. Returns a watertight trimesh."""
    if outer_hw is None:
        outer_hw = LK.PLUG_R + 4.0e-3
    if top_z is None:
        top_z = LK.PLUG_R + 8.0e-3
    bottom_z = -outer_hw
    block_h = top_z - bottom_z
    block = trimesh.creation.box(extents=[length, 2 * outer_hw, block_h])
    block.apply_translation([0.0, 0.0, 0.5 * (top_z + bottom_z)])
    # horizontal bore for the plug
    plug_bore = _cyl_x(LK.PLUG_R + plug_clear, length + 2.0e-3)
    shell = block.difference(plug_bore)
    # vertical chambers above the plug: from the shear plane up through the top
    ch_len = top_z          # spans z=0..top_z with margin
    for cx in chambers_x:
        bore = _cyl_z(bore_r, ch_len + 2.0e-3, cx=cx, cz=0.5 * ch_len)
        shell = shell.difference(bore)
        if mouth_chamfer:
            # countersink the shell chamber mouth (its underside, at the shear
            # plane) so the driver cams up instead of hard-binding near shear
            cs = _countersink(cx, LK.PLUG_R, bore_r, mouth_chamfer, opening=1)
            shell = shell.difference(cs)
    return shell


def make_pin_mesh(length, r=None, tip_top=0.0, tip_bot=0.0, sections=32):
    """A pin cylinder of the given length (axis Z, centred at origin), with an
    optional conical chamfer of height tip_top on the top end and tip_bot on
    the bottom end. Real pin tumbler pins are tapered on both ends; the taper
    cams a slightly-misaligned pin clear of the shear line instead of hard-
    binding, which is the physical source of the lock's shear tolerance. Built
    by revolving a 2D (r,z) profile so it is watertight. Returns a trimesh."""
    if r is None:
        r = LK.PIN_R
    hz = length / 2.0
    # 2D profile in (r, z): a FLAT central face at each end (stable contact
    # with the neighbouring pin) plus a 45-degree BEVELLED outer edge (which
    # cams on the chamber edge). Trace from the bottom-centre outward, up the
    # side, and back in along the top.
    prof = [(0.0, -hz)]
    if tip_bot > 0:
        prof += [(r - tip_bot, -hz), (r, -hz + tip_bot)]
    else:
        prof += [(r, -hz)]
    if tip_top > 0:
        prof += [(r, hz - tip_top), (r - tip_top, hz), (0.0, hz)]
    else:
        prof += [(r, hz), (0.0, hz)]
    return trimesh.creation.revolve(prof, sections=sections)


def key_top_profile(bitting, x_samples, flat_root=None, root_tare=0.0):
    """The milled top edge of a KW1 blade, sampled at x_samples (metres from
    the shoulder). Each cut is a 90-degree V with a flat root of width
    CUT_ROOT_FLAT plunged to ROOT_DEPTH[depth]; the blade top is the lower
    envelope of those cutters, clamped to the full blade height. Returns z
    heights (metres from the spine). `flat_root` overrides the flat width (used
    to seat a full-width pin cleanly -- a disclosed simplification). `root_tare`
    deepens every cut by that amount, seating each break that far below the
    shear plane WITHOUT sinking the blade spine (a spine tare pushes the blade
    bottom into the shell bore floor -- the original insertion jam); it stands
    in for the dome of real pin ends (see p3_slidekey BREAK_TARE comment)."""
    flat = (LK.CUT_ROOT_FLAT if flat_root is None else flat_root) * 0.5
    z = np.full_like(x_samples, LK.BLADE_HEIGHT, dtype=float)
    for i, d in enumerate(bitting):
        xi = LK.FIRST_CUT + i * LK.CUT_SPACING
        root = LK.ROOT_DEPTH[d] - root_tare
        # distance outside the flat root, ramped up at 45 degrees
        dx = np.maximum(np.abs(x_samples - xi) - flat, 0.0)
        z = np.minimum(z, root + dx)
    return np.clip(z, 0.0, LK.BLADE_HEIGHT)


def _blade_slab(dx, width, h, corner_bevel):
    """one blade slab: dx long (X), `width` wide (Y), `h` tall (Z from 0), with
    the BOTTOM corners bevelled 45 degrees by `corner_bevel`. A rectangular
    blade's bottom corners at (+-w/2, spine) lie sqrt((w/2)^2+spine^2) from the
    plug axis -- OUTSIDE the shell bore (6.453 vs 6.40 mm for the KW1 numbers),
    so a sharp-cornered blade scrapes the shell its whole travel and a real
    13 N hand stalls. Real keys relieve exactly these corners with the warding
    grooves; the bevel is that relief. Returns a convex trimesh."""
    import shapely.geometry as sg
    w2 = width / 2.0
    c = min(corner_bevel, w2 * 0.9)
    if c <= 1e-6:
        poly = sg.Polygon([(-w2, 0.0), (w2, 0.0), (w2, h), (-w2, h)])
    elif h > c:
        poly = sg.Polygon([(-w2 + c, 0.0), (w2 - c, 0.0), (w2, c),
                           (w2, h), (-w2, h), (-w2, c)])
    else:
        # a slab shorter than the bevel: keep the narrow bottom the SAME width
        # (the bore clearance must not depend on slab height) and let the 45-deg
        # sides rise only as far as h allows
        wt = min(w2, w2 - c + h)
        poly = sg.Polygon([(-w2 + c, 0.0), (w2 - c, 0.0), (wt, h), (-wt, h)])
    prism = trimesh.creation.extrude_polygon(poly, height=dx)
    # extrude gives (y, z, x'); permute to (x, y, z)
    T = np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    prism.apply_transform(T)
    prism.apply_translation([-dx / 2.0, 0.0, 0.0])
    return prism


def make_key_blade_convex(bitting, length=0.030, spine_z=None, n_slabs=60,
                          shoulder=0.004, width=None, flat_root=None,
                          tip_ramp=0.0, root_tare=0.0, corner_bevel=0.6e-3,
                          min_slab_h=1.0e-3):
    """The KW1 blade cut to `bitting`, as a list of CONVEX box slabs (one per X
    sample) whose top follows the milled bitting profile. Convex-vs-convex
    against the pins is robust (the same reason the plug is convex). `spine_z`
    is the absolute Z of the blade spine; the slab tops are spine_z plus the
    milled height so that, fully inserted, the cut under each chamber sits at
    the correct key surface. `tip_ramp`, if >0, tapers the leading `tip_ramp`
    metres of the blade top from the profile height down to the spine at the very
    tip -- the chamfered key tip that cams a resting pin up as it slides in
    instead of ramming its side. Returns (slabs, x_positions) where x is the
    slab centre along the blade (0 at the shoulder, +length at the tip)."""
    if spine_z is None:
        spine_z = LK.PLUG_R - LK.EDP_IN * LK.IN    # so a cut lifts its pin to shear
    if width is None:
        width = LK.BLADE_THICK
    xs = np.linspace(0.0, length, n_slabs)
    dx = length / (n_slabs - 1)
    tops = key_top_profile(bitting, xs, flat_root=flat_root, root_tare=root_tare)
    if tip_ramp and tip_ramp > 0:
        # linearly taper the leading tip_ramp to a point (a 45-ish deg lead-in)
        over = np.maximum(xs - (length - tip_ramp), 0.0) / tip_ramp
        tops = np.minimum(tops, tops * (1.0 - over))
    slabs = []
    for x, t in zip(xs, tops):
        # PhysX cooking corrupts hulls thinner than ~1 mm (measured law), so no
        # slab may be shorter than min_slab_h: the tip taper ends in a 1 mm
        # blunt nose, which still slides UNDER the resting pins (their bottoms
        # sit on the -2 mm ledge, the nose top is at spine+1 = -5.4 mm).
        h = max(t, min_slab_h)
        b = _blade_slab(dx * 1.02, width, h, corner_bevel)
        b.apply_translation([x, 0.0, spine_z])
        slabs.append(b)
    # a shoulder block behind the cuts (the key stop / grip root)
    sh = _blade_slab(shoulder, width, LK.BLADE_HEIGHT, corner_bevel)
    sh.apply_translation([-shoulder * 0.5, 0.0, spine_z])
    slabs.append(sh)
    return slabs, xs


def make_key_blade_prisms(bitting, length=0.030, spine_z=None, shoulder=0.004,
                          width=None, flat_root=None, root_tare=0.0,
                          nose_h=1.0e-3, corner_bevel=0.6e-3, ext=0.6e-3):
    """The KW1 blade as a few LARGE convex prisms with EXACT sloped top faces.

    Replaces the thin-slab sampling, which cannot work: PhysX cooking corrupts
    hulls under ~1 mm (measured: a 0.51 mm hull face inflates +0.345 mm), and
    1 mm-wide slabs turn the 45-degree ramps into 1 mm vertical risers a pin
    cannot cam up (measured: insertion stalls with the pin hard against a
    riser). Instead the milled profile (piecewise linear: flats, 45-degree
    ramps, the blade-height plateaus, and a 45-degree tip lead-in that ends in
    a `nose_h` blunt nose) is split at its CONVEX corners into maximal concave
    runs; the solid under each run is one convex prism. Each prism is extended
    `ext` past both ends along its end slopes -- that extension lies BELOW the
    neighbouring prism's surface, so the union's contact envelope is exactly
    the profile while every hull stays >1 mm thick. Bottom corners carry the
    warding-groove bevel (see _blade_slab). Returns (pieces, None)."""
    import shapely.geometry as sg
    if spine_z is None:
        spine_z = LK.PLUG_R - LK.EDP_IN * LK.IN
    if width is None:
        width = LK.BLADE_THICK

    # fine-sample the profile, apply the 45-degree tip line + nose clamp
    n = 3001
    xs = np.linspace(0.0, length, n)
    f = key_top_profile(bitting, xs, flat_root=flat_root, root_tare=root_tare)
    tip_line = np.maximum(length - xs, 0.0) + nose_h   # 45 deg down to the nose
    f = np.minimum(f, np.maximum(tip_line, nose_h))
    f = np.maximum(f, nose_h)

    # polyline vertices where the slope changes; split at CONVEX corners
    dx = xs[1] - xs[0]
    slope = np.round(np.diff(f) / dx, 3)
    runs = []
    cur = [(xs[0], f[0])]
    for i in range(1, n - 1):
        if slope[i] != slope[i - 1]:
            cur.append((xs[i], f[i]))
            if slope[i] > slope[i - 1] + 1e-6:      # convex corner -> split
                runs.append(cur)
                cur = [(xs[i], f[i])]
    cur.append((xs[-1], f[-1]))
    runs.append(cur)

    def prism_from_run(pts):
        (x0, z0), (x1, z1) = pts[0], pts[-1]
        s0 = (pts[1][1] - pts[0][1]) / max(pts[1][0] - pts[0][0], 1e-9)
        s1 = (pts[-1][1] - pts[-2][1]) / max(pts[-1][0] - pts[-2][0], 1e-9)
        top = ([(x0 - ext, max(z0 - s0 * ext, 0.2e-3))] + list(pts)
               + [(x1 + ext, max(z1 + s1 * ext, 0.2e-3))])
        poly = sg.Polygon([(top[0][0], 0.0), (top[-1][0], 0.0)]
                          + [(px, pz) for px, pz in reversed(top)])
        solid = trimesh.creation.extrude_polygon(poly, height=width)
        # extrude is along +z; rotate so the section lies in x-z, width along y
        solid.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2.0, [1, 0, 0]))
        solid.apply_translation([0.0, width / 2.0, 0.0])
        # bevel prism: hexagonal y-z section over the piece's x-range
        hmax = max(pz for px, pz in top) + 1e-3
        bev = _blade_slab((top[-1][0] - top[0][0]) + 2e-3, width, hmax,
                          corner_bevel)
        bev.apply_translation([0.5 * (top[0][0] + top[-1][0]), 0.0, 0.0])
        cut = solid.intersection(bev)
        return cut if cut is not None and len(cut.vertices) >= 4 else None

    pieces = []
    for r in runs:
        p = prism_from_run(r)
        if p is not None:
            p.apply_translation([0.0, 0.0, spine_z])
            pieces.append(p)
    sh = _blade_slab(shoulder, width, LK.BLADE_HEIGHT, corner_bevel)
    sh.apply_translation([-shoulder * 0.5, 0.0, spine_z])
    pieces.append(sh)
    return pieces, None


def make_plug_keyway_convex(length, chambers_x, bore_r, keyway_hw, keyway_top,
                            sections=96):
    """Convex plug with chambers open to a bottom keyway channel (NO ledges --
    a sliding key provides the pin surface). Like make_plug_convex but the
    middle-strip pieces stop at `keyway_top` so a blade can slide along X in the
    channel below, and the pins drop through the chambers onto the blade."""
    if not isinstance(chambers_x, (list, tuple)):
        chambers_x = [chambers_x]
    xs = sorted(chambers_x)
    cyl = _cyl_x(LK.PLUG_R, length, sections=sections)
    BIG = LK.PLUG_R * 4
    hw = bore_r

    def region(xlo, xhi, ylo, yhi, zlo, zhi):
        if xhi - xlo < 1e-6 or yhi - ylo < 1e-6 or zhi - zlo < 1e-6:
            return None
        b = trimesh.creation.box(extents=[xhi - xlo, yhi - ylo, zhi - zlo])
        b.apply_translation([0.5 * (xlo + xhi), 0.5 * (ylo + yhi),
                             0.5 * (zlo + zhi)])
        return cyl.intersection(b)

    # side pieces: full height EXCEPT the keyway channel (|y|<keyway_hw) is open
    pieces = []
    pieces.append(region(-BIG, BIG, -BIG, -hw, -BIG, BIG))          # y<-bore
    pieces.append(region(-BIG, BIG, hw, BIG, -BIG, BIG))            # y>+bore
    # the keyway channel is |y|<keyway_hw; between keyway_hw and bore_r the side
    # walls run full height too
    pieces.append(region(-BIG, BIG, keyway_hw, hw, -BIG, BIG))      # bore>y>kw
    pieces.append(region(-BIG, BIG, -hw, -keyway_hw, -BIG, BIG))    # -bore<y<-kw
    # middle strip |y|<keyway_hw: solid ABOVE keyway_top, chambers left void
    edges = [-BIG]
    for x in xs:
        edges += [x - hw, x + hw]
    edges += [BIG]
    for i in range(0, len(edges), 2):
        pieces.append(region(edges[i], edges[i + 1], -keyway_hw, keyway_hw,
                             keyway_top, BIG))
    return [p for p in pieces if p is not None and len(p.vertices) >= 4]


def make_plug_ledgekeyway_convex(length, chambers_x, bore_r, keyway_hw, ledge_z,
                                 keyway_top, sections=96):
    """Convex plug with a REAL narrow keyway. The keyway slot (|y|<keyway_hw,
    which is NARROWER than a pin so a pin cannot fall through it) runs the full
    length at the bottom for the key blade. Above it each chamber is a bore-wide
    shaft for the pin. At z=ledge_z the chamber meets the slot on a SHOULDER
    (the bands keyway_hw<|y|<bore_r): when no key is under a chamber the key pin
    drops only until its rim rests on that shoulder, its tip protruding into the
    slot, so the sliding key's ramped tip cams it up -- the physical mechanism,
    replacing the old chamber-wide keyway the pins fell straight through.

    Returns (collide_pieces, visual_pieces). MEASURED ENGINE LAW: PhysX convex
    cooking corrupts hulls thinner than ~1 mm (a 0.51 mm wall's face inflates
    +0.345 mm; a 0.30 mm wall is unstable; >=1.0 mm is exact to a micron), so
    every returned COLLIDE piece is >=1 mm thick in all dimensions: the slot
    bands extend outward past keyway_hw+1.5 mm (overlapping the side pieces --
    same body, harmless) so their slot faces at +-keyway_hw stay exact; the
    0.69 mm between-chamber walls and slot caps go in VISUAL pieces only --
    with pins/drivers guided by joints they are collision-redundant (a bound
    driver binds against the thick outer side pieces)."""
    if not isinstance(chambers_x, (list, tuple)):
        chambers_x = [chambers_x]
    xs = sorted(chambers_x)
    cyl = _cyl_x(LK.PLUG_R, length, sections=sections)
    BIG = LK.PLUG_R * 4
    hw = bore_r

    def region(xlo, xhi, ylo, yhi, zlo, zhi):
        if xhi - xlo < 1e-6 or yhi - ylo < 1e-6 or zhi - zlo < 1e-6:
            return None
        b = trimesh.creation.box(extents=[xhi - xlo, yhi - ylo, zhi - zlo])
        b.apply_translation([0.5 * (xlo + xhi), 0.5 * (ylo + yhi),
                             0.5 * (zlo + zhi)])
        return cyl.intersection(b)

    collide = []
    visual = []
    # outer sides beyond the chamber bore (full height, thick)
    collide.append(region(-BIG, BIG, -BIG, -hw, -BIG, BIG))
    collide.append(region(-BIG, BIG, hw, BIG, -BIG, BIG))
    # shoulders + slot side walls BELOW the ledge, thickened OUTWARD past the
    # bore radius so the hull is >=1 mm; the slot face (+-keyway_hw) and the
    # shelf top (ledge_z) are the exact functional faces.
    band_out = keyway_hw + 1.5e-3
    collide.append(region(-BIG, BIG, keyway_hw, band_out, -BIG, ledge_z))
    collide.append(region(-BIG, BIG, -band_out, -keyway_hw, -BIG, ledge_z))
    edges = [-BIG]
    for x in xs:
        edges += [x - hw, x + hw]
    edges += [BIG]
    for i in range(0, len(edges), 2):
        xlo, xhi = edges[i], edges[i + 1]
        # between-chamber walls and slot caps: sub-mm in X, and the refined
        # measured law is that a hull with ANY sub-mm dimension bleeds on ALL
        # faces (the widened 0.69 mm bands poked 0.16 mm into the slot at
        # their ledge corner and froze the insertion) -> VISUAL ONLY.
        visual.append(region(xlo, xhi, keyway_hw, hw, ledge_z, BIG))
        visual.append(region(xlo, xhi, -hw, -keyway_hw, ledge_z, BIG))
        visual.append(region(xlo, xhi, -keyway_hw, keyway_hw, keyway_top, BIG))
    # CORNER BANDS: the real chamber is ROUND, so at |y|>=1.2 mm the material
    # between chambers is (3.81 - 2*sqrt(r_ch^2 - y^2)) ~ 1.81 mm wide -- wide
    # enough for a trustworthy hull. These bands (y in [1.2, 2.7], set back
    # 1.0 mm from each chamber centre, matching the round wall at y=1.2) are
    # what a lifted driver's rim RIDES as the plug turns (rim lobes reach
    # |y|=1.46), while pins/drivers (lobes <=0.83 at y=1.2) and the blade
    # (|y|<=1.0) never reach them.
    CY = 1.2e-3
    SB = 1.0e-3
    gaps = [(-BIG, xs[0] - SB)]
    for j in range(len(xs) - 1):
        gaps.append((xs[j] + SB, xs[j + 1] - SB))
    gaps.append((xs[-1] + SB, BIG))
    for xlo, xhi in gaps:
        collide.append(region(xlo, xhi, CY, CY + 1.5e-3, ledge_z, BIG))
        collide.append(region(xlo, xhi, -(CY + 1.5e-3), -CY, ledge_z, BIG))
    collide = [p for p in collide if p is not None and len(p.vertices) >= 4]
    visual = [p for p in visual if p is not None and len(p.vertices) >= 4]
    return collide, visual


def make_key_blade(bitting, length=0.030, grooves=None, extra_shoulder=0.004):
    """A KW1 key blade cut to `bitting` (list of 5 depths 1..7), as a watertight
    trimesh. The blade envelope is BLADE_THICK x BLADE_HEIGHT; the top edge
    carries the milled bitting so pins ride it. `grooves` is a list of
    (z_centre, width, depth, face) longitudinal warding channels; face is +1
    for +Y, -1 for -Y. Extruded along Y (thickness)."""
    import shapely.geometry as sg

    # sample the side profile densely enough to capture the 45-degree ramps
    xs = np.linspace(0.0, length, 600)
    zt = key_top_profile(bitting, xs)
    # 2D side polygon (X-Z): along the top edge, back along the spine.
    pts = [(float(x), float(z)) for x, z in zip(xs, zt)]
    pts += [(length, 0.0), (0.0, 0.0)]
    poly = sg.Polygon(pts)
    blade = trimesh.creation.extrude_polygon(poly, height=LK.BLADE_THICK)
    # extrude_polygon extrudes along +Z; we want the blade thickness along Y.
    blade.apply_transform(trimesh.transformations.rotation_matrix(
        -np.pi / 2.0, [1, 0, 0]))
    blade.apply_translation([0.0, LK.BLADE_THICK * 0.5, 0.0])
    # a shoulder stop block behind the cuts (where the key seats against the face)
    if extra_shoulder > 0:
        sh = trimesh.creation.box(extents=[extra_shoulder, LK.BLADE_THICK,
                                           LK.BLADE_HEIGHT])
        sh.apply_translation([-extra_shoulder * 0.5, 0.0, LK.BLADE_HEIGHT * 0.5])
        blade = blade.union(sh)
    if grooves:
        for (zc, w, dep, face) in grooves:
            g = trimesh.creation.box(extents=[length + extra_shoulder + 2e-3,
                                              dep * 2.0, w])
            g.apply_translation([0.0, face * (LK.BLADE_THICK * 0.5), zc])
            blade = blade.difference(g)
    return blade


def to_usd_mesh(stage, path, tm, up_axis_fix=False):
    """Author a UsdGeom.Mesh prim from a trimesh (metres). Returns the prim."""
    from pxr import Gf, UsdGeom, Vt
    m = UsdGeom.Mesh.Define(stage, path)
    verts = tm.vertices.astype(np.float32)
    faces = tm.faces.astype(np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(len(faces), 3, dtype=np.int32)))
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    m.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    return m.GetPrim()


if __name__ == "__main__":
    # sanity: build a one-chamber plug and shell, report watertightness/volume
    ch = [LK.chamber_x(0)]
    plug = make_plug(0.020, ch, LK.PIN_R + 0.10e-3)
    shell = make_shell(0.020, ch, LK.PIN_R + 0.10e-3, 0.05e-3)
    for nm, tm in (("plug", plug), ("shell", shell)):
        print(f"{nm}: watertight={tm.is_watertight}  volume={tm.volume*1e9:.1f} "
              f"mm^3  verts={len(tm.vertices)}  faces={len(tm.faces)}", flush=True)

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton1.5 analytic boxes with float32-scale face/edge tie breaking.

Only the marked axis-selection comparison differs from the native routine.
For nearly coplanar support, the native edge branch can turn a tangential
separation into a large negative distance. Reproduced on CPU and CUDA with
recorded-pose perturbations in box_witness_replay_332.json. This does not
change actual separating-axis rejection, object geometry or constitutive law.
"""
import warp as wp
from newton._src.geometry.collision_primitive import (
    _vec8f, _mat83f, _mat43f, _compute_rotmore, MAXVAL, MINVAL, safe_div,
)

@wp.func
def collide_box_box_stable(
    # In:
    box1_pos: wp.vec3,
    box1_rot: wp.mat33,
    box1_size: wp.vec3,
    box2_pos: wp.vec3,
    box2_rot: wp.mat33,
    box2_size: wp.vec3,
    margin: float = 0.0,
) -> tuple[wp.types.vector(8, wp.float32), wp.types.matrix((8, 3), wp.float32), wp.types.matrix((8, 3), wp.float32)]:
    """Core contact geometry calculation for box-box collision.

    Args:
      box1_pos: Center position of the first box
      box1_rot: Rotation matrix of the first box
      box1_size: Half-extents of the first box along each axis
      box2_pos: Center position of the second box
      box2_rot: Rotation matrix of the second box
      box2_size: Half-extents of the second box along each axis
      margin: Distance threshold for early contact generation (default: 0.0).
              When positive, contacts are generated before boxes overlap.

    Returns:
      Tuple containing:
        contact_dist: Vector of contact distances (MAXVAL for unpopulated contacts)
        contact_pos: Matrix of contact positions (one per row)
        contact_normals: Matrix of contact normal vectors (one per row)
    """

    # Initialize output matrices
    contact_dist = _vec8f()
    for i in range(8):
        contact_dist[i] = MAXVAL
    contact_pos = _mat83f()
    contact_normals = _mat83f()
    contact_count = 0

    # Compute transforms between box's frames
    pos21 = wp.transpose(box1_rot) @ (box2_pos - box1_pos)
    pos12 = wp.transpose(box2_rot) @ (box1_pos - box2_pos)

    rot21 = wp.transpose(box1_rot) @ box2_rot
    rot12 = wp.transpose(rot21)

    rot21abs = wp.matrix_from_rows(wp.abs(rot21[0]), wp.abs(rot21[1]), wp.abs(rot21[2]))
    rot12abs = wp.transpose(rot21abs)

    plen2 = rot21abs @ box2_size
    plen1 = rot12abs @ box1_size

    # Compute axis of maximum separation
    s_sum_3 = 3.0 * (box1_size + box2_size)
    separation = wp.float32(margin + s_sum_3[0] + s_sum_3[1] + s_sum_3[2])
    axis_code = wp.int32(-1)

    # First test: consider boxes' face normals
    for i in range(3):
        c1 = -wp.abs(pos21[i]) + box1_size[i] + plen2[i]

        c2 = -wp.abs(pos12[i]) + box2_size[i] + plen1[i]

        if c1 < -margin or c2 < -margin:
            return contact_dist, contact_pos, contact_normals

        if c1 < separation:
            separation = c1
            axis_code = i + 3 * wp.int32(pos21[i] < 0) + 0  # Face of box1
        if c2 < separation:
            separation = c2
            axis_code = i + 3 * wp.int32(pos12[i] < 0) + 6  # Face of box2

    clnorm = wp.vec3(0.0)
    inv = wp.bool(False)
    cle1 = wp.int32(0)
    cle2 = wp.int32(0)

    # Second test: consider cross products of boxes' edges
    for i in range(3):
        for j in range(3):
            # Compute cross product of box edges (potential separating axis)
            if i == 0:
                cross_axis = wp.vec3(0.0, -rot12[j, 2], rot12[j, 1])
            elif i == 1:
                cross_axis = wp.vec3(rot12[j, 2], 0.0, -rot12[j, 0])
            else:
                cross_axis = wp.vec3(-rot12[j, 1], rot12[j, 0], 0.0)

            cross_length = wp.length(cross_axis)
            if cross_length < MINVAL:
                continue

            cross_axis /= cross_length

            box_dist = wp.dot(pos21, cross_axis)
            c3 = wp.float32(0.0)

            # Project box half-sizes onto the potential separating axis
            for k in range(3):
                if k != i:
                    c3 += box1_size[k] * wp.abs(cross_axis[k])
                if k != j:
                    c3 += box2_size[k] * rot21abs[i, 3 - k - j] / cross_length

            c3 -= wp.abs(box_dist)

            # Early exit: no collision if separated along this axis
            if c3 < -margin:
                return contact_dist, contact_pos, contact_normals

            # Track minimum separation and which edge-edge pair it occurs on
            # Local geometry scale, not world origin. Prefer a face manifold
            # when the edge-axis improvement is at float32 roundoff scale.
            # All separating-axis rejection tests above remain unchanged.
            face_tolerance = 8.0 * 1.1920929e-7 * (
                wp.length(box1_size) + wp.length(box2_size) + wp.length(pos21)
            )
            tolerance = wp.where(axis_code < 12, face_tolerance, 0.0)
            if c3 < separation - tolerance:
                separation = c3
                # Determine which corners/edges are closest
                cle1 = 0
                cle2 = 0

                for k in range(3):
                    if k != i and (int(cross_axis[k] > 0) ^ int(box_dist < 0)):
                        cle1 += 1 << k
                    if k != j:
                        if int(rot21[i, 3 - k - j] > 0) ^ int(box_dist < 0) ^ int((k - j + 3) % 3 == 1):
                            cle2 += 1 << k

                axis_code = 12 + i * 3 + j
                clnorm = cross_axis
                inv = box_dist < 0

    # No axis with separation < margin found
    if axis_code == -1:
        return contact_dist, contact_pos, contact_normals

    points = _mat83f()
    depth = _vec8f()
    max_con_pair = 8
    # 8 contacts should suffice for most configurations

    if axis_code < 12:
        # Handle face-vertex collision
        face_idx = axis_code % 6
        box_idx = axis_code // 6
        rotmore = _compute_rotmore(face_idx)

        r = rotmore @ wp.where(box_idx, rot12, rot21)
        p = rotmore @ wp.where(box_idx, pos12, pos21)
        ss = wp.abs(rotmore @ wp.where(box_idx, box2_size, box1_size))
        s = wp.where(box_idx, box1_size, box2_size)
        rt = wp.transpose(r)

        lx, ly, hz = ss[0], ss[1], ss[2]
        p[2] -= hz

        clcorner = wp.int32(0)  # corner of non-face box with least axis separation

        for i in range(3):
            if r[2, i] < 0:
                clcorner += 1 << i

        lp = p
        for i in range(wp.static(3)):
            lp += rt[i] * s[i] * wp.where(clcorner & 1 << i, 1.0, -1.0)

        m = wp.int32(1)
        dirs = wp.int32(0)

        cn1 = wp.vec3(0.0)
        cn2 = wp.vec3(0.0)

        for i in range(3):
            if wp.abs(r[2, i]) < 0.5:
                if not dirs:
                    cn1 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)
                else:
                    cn2 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)

                dirs += 1

        k = dirs * dirs

        # Find potential contact points

        n = wp.int32(0)

        for i in range(k):
            for q in range(2):
                # lines_a and lines_b (lines between corners) computed on the fly
                lav = lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, cn1, cn2))
                lbv = wp.where(i == 0 or i == 3, cn1, cn2)

                if wp.abs(lbv[q]) > MINVAL:
                    br = 1.0 / lbv[q]
                    for j in range(-1, 2, 2):
                        l = ss[q] * wp.float32(j)
                        c1 = (l - lav[q]) * br
                        if c1 < 0 or c1 > 1:
                            continue
                        c2 = lav[1 - q] + lbv[1 - q] * c1
                        if wp.abs(c2) > ss[1 - q]:
                            continue

                        points[n] = lav + c1 * lbv
                        n += 1

        if dirs == 2:
            ax = cn1[0]
            bx = cn2[0]
            ay = cn1[1]
            by = cn2[1]
            C = safe_div(1.0, ax * by - bx * ay)

            for i in range(4):
                llx = wp.where(i // 2, lx, -lx)
                lly = wp.where(i % 2, ly, -ly)

                x = llx - lp[0]
                y = lly - lp[1]

                u = (x * by - y * bx) * C
                v = (y * ax - x * ay) * C

                if u > 0 and v > 0 and u < 1 and v < 1:
                    points[n] = wp.vec3(llx, lly, lp[2] + u * cn1[2] + v * cn2[2])
                    n += 1

        for i in range(1 << dirs):
            tmpv = lp + wp.float32(i & 1) * cn1 + wp.float32((i & 2) != 0) * cn2
            if tmpv[0] > -lx and tmpv[0] < lx and tmpv[1] > -ly and tmpv[1] < ly:
                points[n] = tmpv
                n += 1

        m = n
        n = wp.int32(0)

        for i in range(m):
            if points[i][2] > margin:
                continue
            if i != n:
                points[n] = points[i]

            points[n, 2] *= 0.5
            depth[n] = points[n, 2] * 2.0
            n += 1

        # Set up contact frame
        rw = wp.where(box_idx, box2_rot, box1_rot) @ wp.transpose(rotmore)
        pw = wp.where(box_idx, box2_pos, box1_pos)
        normal = wp.where(box_idx, -1.0, 1.0) * wp.transpose(rw)[2]

    else:
        # Handle edge-edge collision
        edge1 = (axis_code - 12) // 3
        edge2 = (axis_code - 12) % 3

        # Set up non-contacting edges ax1, ax2 for box2 and pax1, pax2 for box 1
        ax1 = wp.int32(1 - (edge2 & 1))
        ax2 = wp.int32(2 - (edge2 & 2))

        pax1 = wp.int32(1 - (edge1 & 1))
        pax2 = wp.int32(2 - (edge1 & 2))

        if rot21abs[edge1, ax1] < rot21abs[edge1, ax2]:
            ax1, ax2 = ax2, ax1

        if rot12abs[edge2, pax1] < rot12abs[edge2, pax2]:
            pax1, pax2 = pax2, pax1

        rotmore = _compute_rotmore(wp.where(cle1 & (1 << pax2), pax2, pax2 + 3))

        # Transform coordinates for edge-edge contact calculation
        p = rotmore @ pos21
        rnorm = rotmore @ clnorm
        r = rotmore @ rot21
        rt = wp.transpose(r)
        s = wp.abs(wp.transpose(rotmore) @ box1_size)

        lx, ly, hz = s[0], s[1], s[2]
        p[2] -= hz

        # Calculate closest box2 face

        points[0] = (
            p
            + rt[ax1] * box2_size[ax1] * wp.where(cle2 & (1 << ax1), 1.0, -1.0)
            + rt[ax2] * box2_size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
        )
        points[1] = points[0] - rt[edge2] * box2_size[edge2]
        points[0] += rt[edge2] * box2_size[edge2]

        points[2] = (
            p
            + rt[ax1] * box2_size[ax1] * wp.where(cle2 & (1 << ax1), -1.0, 1.0)
            + rt[ax2] * box2_size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
        )

        points[3] = points[2] - rt[edge2] * box2_size[edge2]
        points[2] += rt[edge2] * box2_size[edge2]

        n = 4

        # Set up coordinate axes for contact face of box2
        axi_lp = points[0]
        axi_cn1 = points[1] - points[0]
        axi_cn2 = points[2] - points[0]

        # Check if contact normal is valid
        if wp.abs(rnorm[2]) < MINVAL:
            return contact_dist, contact_pos, contact_normals  # Shouldn't happen

        # Calculate inverse normal for projection
        innorm = wp.where(inv, -1.0, 1.0) / rnorm[2]

        pu = _mat43f()

        # Project points onto contact plane
        for i in range(4):
            pu[i] = points[i]
            c_scl = points[i, 2] * wp.where(inv, -1.0, 1.0) * innorm
            points[i] -= rnorm * c_scl

        pts_lp = points[0]
        pts_cn1 = points[1] - points[0]
        pts_cn2 = points[2] - points[0]

        n = wp.int32(0)

        for i in range(4):
            for q in range(2):
                la = pts_lp[q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[q], pts_cn2[q]))
                lb = wp.where(i == 0 or i == 3, pts_cn1[q], pts_cn2[q])
                lc = pts_lp[1 - q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[1 - q], pts_cn2[1 - q]))
                ld = wp.where(i == 0 or i == 3, pts_cn1[1 - q], pts_cn2[1 - q])

                # linesu_a and linesu_b (lines between corners) computed on the fly
                lua = axi_lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, axi_cn1, axi_cn2))
                lub = wp.where(i == 0 or i == 3, axi_cn1, axi_cn2)

                if wp.abs(lb) > MINVAL:
                    br = 1.0 / lb
                    for j in range(-1, 2, 2):
                        if n == max_con_pair:
                            break
                        l = s[q] * wp.float32(j)
                        c1 = (l - la) * br
                        if c1 < 0 or c1 > 1:
                            continue
                        c2 = lc + ld * c1
                        if wp.abs(c2) > s[1 - q]:
                            continue
                        if (lua[2] + lub[2] * c1) * innorm > margin:
                            continue

                        points[n] = lua * 0.5 + c1 * lub * 0.5
                        points[n, q] += 0.5 * l
                        points[n, 1 - q] += 0.5 * c2
                        depth[n] = points[n, 2] * innorm * 2.0
                        n += 1

        nl = n

        ax = pts_cn1[0]
        bx = pts_cn2[0]
        ay = pts_cn1[1]
        by = pts_cn2[1]
        C = safe_div(1.0, ax * by - bx * ay)

        for i in range(4):
            if n == max_con_pair:
                break
            llx = wp.where(i // 2, lx, -lx)
            lly = wp.where(i % 2, ly, -ly)

            x = llx - pts_lp[0]
            y = lly - pts_lp[1]

            u = (x * by - y * bx) * C
            v = (y * ax - x * ay) * C

            if nl == 0:
                if (u < 0 or u > 1) and (v < 0 or v > 1):
                    continue
            elif u < 0 or v < 0 or u > 1 or v > 1:
                continue

            u = wp.clamp(u, 0.0, 1.0)
            v = wp.clamp(v, 0.0, 1.0)
            w = 1.0 - u - v
            vtmp = pu[0] * w + pu[1] * u + pu[2] * v

            points[n] = wp.vec3(llx, lly, 0.0)

            vtmp2 = points[n] - vtmp
            tc1 = wp.length_sq(vtmp2)
            if vtmp[2] > 0 and tc1 > margin * margin:
                continue

            points[n] = 0.5 * (points[n] + vtmp)

            depth[n] = wp.sqrt(tc1) * wp.where(vtmp[2] < 0, -1.0, 1.0)
            n += 1

        nf = n

        for i in range(4):
            if n >= max_con_pair:
                break
            x = pu[i, 0]
            y = pu[i, 1]
            if nl == 0 and nf != 0:
                if (x < -lx or x > lx) and (y < -ly or y > ly):
                    continue
            elif x < -lx or x > lx or y < -ly or y > ly:
                continue

            c1 = wp.float32(0)

            for j in range(2):
                if pu[i, j] < -s[j]:
                    c1 += (pu[i, j] + s[j]) * (pu[i, j] + s[j])
                elif pu[i, j] > s[j]:
                    c1 += (pu[i, j] - s[j]) * (pu[i, j] - s[j])

            c1 += pu[i, 2] * innorm * pu[i, 2] * innorm

            if pu[i, 2] > 0 and c1 > margin * margin:
                continue

            tmp_p = wp.vec3(pu[i, 0], pu[i, 1], 0.0)

            for j in range(2):
                if pu[i, j] < -s[j]:
                    tmp_p[j] = -s[j] * 0.5
                elif pu[i, j] > s[j]:
                    tmp_p[j] = +s[j] * 0.5

            tmp_p += pu[i]
            points[n] = tmp_p * 0.5

            depth[n] = wp.sqrt(c1) * wp.where(pu[i, 2] < 0, -1.0, 1.0)
            n += 1

        # Set up contact data for all points
        rw = box1_rot @ wp.transpose(rotmore)
        pw = box1_pos
        normal = wp.where(inv, -1.0, 1.0) * rw @ rnorm

    contact_count = n

    # Copy contact data to output matrices
    for i in range(contact_count):
        points[i, 2] += hz
        pos = rw @ points[i] + pw
        contact_dist[i] = depth[i]
        contact_pos[i] = pos
        contact_normals[i] = normal

    return contact_dist, contact_pos, contact_normals

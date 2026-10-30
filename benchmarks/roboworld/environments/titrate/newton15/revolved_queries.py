"""Exact interior test for the authored polygonal solids of revolution.

This uses the same meridian and angular facets as geometry.revolve, not an
ideal smooth cylinder with different dimensions. Generic meshes still use
winding queries. Closest points still come from the actual triangulated mesh.
"""

import warp as wp


@wp.func
def inside(p: wp.vec3, profile: wp.array[wp.vec2], start: int, count: int, segments: int):
    angle = wp.atan2(p[1], p[0])
    width = 2.*wp.pi/float(segments)
    middle = (wp.floor(angle/width)+.5)*width
    # Radial coordinate on the actual regular-polygon facet, whose apothem
    # is r*cos(pi/N). The 2D meridian then gives an exact inside/outside test.
    radial = (p[0]*wp.cos(middle)+p[1]*wp.sin(middle))/wp.cos(.5*width)
    hit = int(0)
    for i in range(count):
        a, b = profile[start+i], profile[start+(i+1)%count]
        if (a[1] > p[2]) != (b[1] > p[2]):
            crossing = a[0]+(p[2]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if radial < crossing:
                hit = 1-hit
    return hit


@wp.kernel
def inside_shape(points: wp.array[wp.vec3], transform: wp.transform, scale: wp.vec3,
                 profile: wp.array[wp.vec2], start: int, count: int, segments: int,
                 result: wp.array[int]):
    i = wp.tid()
    p = wp.cw_div(wp.transform_point(wp.transform_inverse(transform), points[i]), scale)
    if inside(p, profile, start, count, segments):
        result[i] = 1

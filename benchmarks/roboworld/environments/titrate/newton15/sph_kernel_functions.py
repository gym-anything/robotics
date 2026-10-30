"""Normalized cubic-spline SPH kernel and exact planar volume integrals.

Support h is the full support radius (twice the conventional spline length).
The pressure kernel is selected explicitly; existing Wendland experiments keep
their original operator. All derivatives here are with respect to displacement.
"""

import warp as wp


@wp.func
def cubic_w(r: wp.vec3, h: float):
    u = wp.length(r)/h
    f = float(0.)
    if u < .5:
        f = 1.-6.*u*u+6.*u*u*u
    elif u < 1.:
        f = 2.*wp.pow(1.-u, 3.)
    return 8./(wp.pi*h*h*h)*f


@wp.func
def cubic_grad(r: wp.vec3, h: float):
    length = wp.length(r)
    u = length/h
    f = float(0.)
    if length > 1.e-12:
        if u < .5:
            f = -12.*u+18.*u*u
        elif u < 1.:
            f = -6.*(1.-u)*(1.-u)
        return 8./(wp.pi*h*h*h*h)*f*r/length
    return wp.vec3(0.)


@wp.func
def primitive(u: float, power: int, lower: int):
    a, b, c, d = float(2.), float(-6.), float(6.), float(-2.)
    if lower != 0:
        a, b, c, d = 1., 0., -6., 6.
    return (a*wp.pow(u, float(power+1))/float(power+1)
            +b*wp.pow(u, float(power+2))/float(power+2)
            +c*wp.pow(u, float(power+3))/float(power+3)
            +d*wp.pow(u, float(power+4))/float(power+4))


@wp.func
def tail(s: float, power: int):
    result = float(0.)
    if s < 1.:
        result = primitive(1., power, 0)-primitive(wp.max(s, .5), power, 0)
        if s < .5:
            result += primitive(.5, power, 1)-primitive(s, power, 1)
    return result


@wp.func
def cubic_plane_fraction(z: float, h: float):
    s = wp.min(wp.abs(z/h), 1.)
    result = 16.*(tail(s, 2)-s*tail(s, 1))
    if z < 0.:
        result = 1.-result
    return result


@wp.func
def cubic_plane_gradient(z: float, h: float):
    s = wp.min(wp.abs(z/h), 1.)
    return wp.vec3(0., 0., -16./h*tail(s, 1))

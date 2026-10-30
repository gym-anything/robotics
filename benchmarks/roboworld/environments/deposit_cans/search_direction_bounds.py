"""Per-shell numerical search normalization inside existing discovery bounds."""
import warp as wp
from .relative_step import double

@wp.kernel
def group_bounds(reference:wp.array[wp.vec3],current:wp.array[wp.vec3],delta:wp.array[wp.vec3],
                 radius:float,groups:wp.array[int],scales:wp.array[float]):
    i=wp.tid();g=groups[i]
    if g<0:return
    y=double(current[i])-double(reference[i]);d=double(delta[i])
    a=wp.dot(d,d);b=wp.dot(y,d);c=wp.dot(y,y)-wp.float64(radius)*wp.float64(radius)
    bound=wp.float64(1.)
    if c>wp.float64(1e-12):bound=wp.float64(0.)
    elif a>wp.float64(1e-30):bound=wp.clamp((-b+wp.sqrt(wp.max(b*b-a*c,wp.float64(0.))))/a,wp.float64(0.),wp.float64(1.))
    wp.atomic_min(scales,g,float(bound))

@wp.kernel
def scale_direction(groups:wp.array[int],scales:wp.array[float],direction:wp.array[wp.vec3]):
    i=wp.tid();g=groups[i]
    if g>=0:direction[i]*=scales[g]

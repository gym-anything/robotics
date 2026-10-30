"""Opt-in authored resting-stack candidate; no simulation or state projection.

Twelve horizontal cans occupy six bottom rows. Four further cans sit above
two-can pockets, rather than balancing over the crown of one lower cylinder.
Support heights use the actual initial convex envelopes, not nominal radii.
This is a geometric initialization, NOT proof of dynamic equilibrium. Ordinary
contact, settling and unchanged stationary qualification remain required.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from deposit_cans.geometry import can_mesh


def support_planes(lower, upper):
    """Minkowski outer envelope for relative upper-centre translation."""
    l=lower[ConvexHull(lower).vertices]
    u=upper[ConvexHull(upper).vertices]
    hull=ConvexHull((l[:,None,:]-u[None,:,:]).reshape(-1,3))
    return hull.equations[hull.equations[:,2]>1e-10]


def height(planes,x,y,clearance=0.):
    return float(np.min(-(planes[:,0]*x+planes[:,1]*y+planes[:,3]-clearance)/planes[:,2]))


@lru_cache(maxsize=4)
def layout(segments,normal_clearance=False):
    rot=Rotation.from_euler('y',np.pi/2)
    vertices=[rot.apply(can_mesh(i%3,segments,.0015 if i%7==3 else 0.,quality_end=True)[0])
              for i in range(16)]
    centres=[];supports=[]
    # The floor skin is1.7mm with3mm molded ribs. Keep1mm surface gap
    # above its highest feature; any additional settling remains recorded.
    for i in range(12):
        centres.append(np.array([-.37+(-.085 if i%2==0 else .085),
            -.185+.074*(i//2),.903+.0017+.003+.001-vertices[i][:,2].min()]))
    for i,(a,b) in enumerate(((2,4),(3,5),(6,8),(7,9)),12):
        planes=[support_planes(vertices[j],vertices[i])for j in (a,b)]
        x=float(centres[a][0])
        def support(y):
            return max(centres[j][2]+height(p,x-centres[j][0],y-centres[j][1],.0011 if normal_clearance else 0.)
                       for j,p in zip((a,b),planes))
        fit=minimize_scalar(support,bounds=(centres[a][1]+.001,centres[b][1]-.001),
                            method='bounded',options={'xatol':1e-13})
        if not fit.success:raise ValueError('Pocket geometry minimization failed')
        y=float(fit.x);z=support(y)
        nearby=[support(y+d)-z for d in (-.001,.001)]
        if min(nearby)<=0:raise ValueError('Candidate is not a two-sided support minimum')
        # Surface-to-surface clearance for two.5mm particle radii, plus.1mm.
        # Revision6 added clearance vertically. On sloped supporting faces the
        # normal gap was only0.873mm, inside the1mm contact envelope. Revision7
        # offsets unit-normal support planes instead, preserving real clearance.
        centres.append(np.array([x,y,z+(0. if normal_clearance else .0011)]))
        supports.append(dict(upper=i,lower=[a,b],centre_m=centres[-1].tolist(),
            support_height_m=z,one_mm_perturbation_height_increase_m=nearby))
    return np.asarray(centres),supports


def pose(index,segments,normal_clearance=False):
    if not 0<=index<16:raise ValueError('The authored tub layout supports at most16 cans')
    centres,_=layout(segments,normal_clearance)
    return centres[index].copy(),Rotation.from_euler('y',np.pi/2)

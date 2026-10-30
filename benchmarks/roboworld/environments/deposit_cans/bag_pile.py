"""Authored lying pile for the high_fill preset's 120 cans already in the bag.

Task-setting initialization, not a physics change. A floating 6x4x5 grid of upright cans
(0.175 m layer pitch, top 0.897 m) would collapse by itself during the first ~4 s. Here every
pre-existing can lies on the bag floor or on cans already placed, like returned cans tipped into a bag:

* Each can is enclosed by its cylinder envelope (radius r, length h from CAN_SPECS; every
  can_mesh vertex lies inside it), inflated by CLEARANCE in every direction.
* Cans are dropped one at a time, in inventory order, onto a height map of the pile top
  (bag floor level FLOOR_Z, then the inflated top surfaces of cans already placed). Of
  TRIALS random lying poses inside the inset bag walls, the lowest resting one is kept and
  then slid/turned down-hill until no move lowers it, so cans settle into grooves.
* Axis yaw is drawn around the bag's short (y) side, AXIS_YAW_SPREAD standard deviation:
  long cans lie across the narrow bag rather than at every angle. Free yaw packs so loosely
  (p90 top 0.72 m) that the pile would sink far by itself; this choice gives p90 0.61 m and
  highest 0.67 m in the unloaded bag, against the published 30 s settled pile's 0.49/0.66 m
  in a bag that had bulged under that load (experiments/high_runtime.json.gz).
* Footprints stay WALL_MARGIN inside the bag's narrowest (bottom) rectangle.

This is a geometric initialization, NOT a settled state: the bag keeps its unloaded
authored shape, so under the unchanged physics its floor sags ~25 mm onto the room floor,
its walls bulge and the pile slumps in its first ~0.3 s (measured: mean can drop ~9.5 cm).
The arrangement is precomputed into high_fill_pile.json so construction does not repeat
the search. Regenerate with:  python -m deposit_cans.bag_pile --write
"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import CAN_SPECS

DATA=Path(__file__).with_name('high_fill_pile.json')
GENERATOR='lowest-resting-drop-v1'
CENTRE=(.49,0.)
# bag_mesh: bottom ring is rectangle_ring(.288*.92,.197*.92) with +-1.5 mm radial slack.
HALF=(.288*.92-.0015,.197*.92-.0015)
WALL_MARGIN=.004
FLOOR_Z=.030      # lowest can point; the bag's pleated floor peaks at 0.025 m
CLEARANCE=.0015   # per can, so >=3 mm nominal surface gap between envelopes
CELL=.001
TRIALS=512
AXIS_YAW_SPREAD=np.deg2rad(15.)
SEED=20261022


def kind_of(index):return index%3


def dent_of(index):return .0015 if index%7==3 else 0.


def lying(psi,spin=0.):
    """Can axis horizontal along (cos psi, sin psi, 0), spun by `spin` about its own axis."""
    return Rotation.from_euler('zyz',[spin,np.pi/2,psi])


class _Pile:
    def __init__(self):
        self.x=np.arange(CENTRE[0]-HALF[0],CENTRE[0]+HALF[0]+CELL/2,CELL)
        self.y=np.arange(CENTRE[1]-HALF[1],CENTRE[1]+HALF[1]+CELL/2,CELL)
        self.top=np.full((len(self.x),len(self.y)),FLOOR_Z)

    def _window(self,x,y,psi,r,h):
        c,s=np.cos(psi),np.sin(psi);a,b=h/2+CLEARANCE,r+CLEARANCE
        ex,ey=a*abs(c)+b*abs(s),a*abs(s)+b*abs(c)
        i=slice(*np.searchsorted(self.x,[x-ex,x+ex]));j=slice(*np.searchsorted(self.y,[y-ey,y+ey]))
        X,Y=np.meshgrid(self.x[i],self.y[j],indexing='ij')
        t=(X-x)*c+(Y-y)*s;d=-(X-x)*s+(Y-y)*c
        mask=(np.abs(t)<=a)&(np.abs(d)<b)
        return i,j,mask,np.sqrt(np.maximum(b*b-d*d,0.))

    def inside(self,x,y,psi,r,h):
        c,s=np.cos(psi),np.sin(psi);a,b=h/2+CLEARANCE,r+CLEARANCE
        ex,ey=a*abs(c)+b*abs(s),a*abs(s)+b*abs(c)
        return abs(x-CENTRE[0])+ex<=HALF[0]-WALL_MARGIN and abs(y-CENTRE[1])+ey<=HALF[1]-WALL_MARGIN

    def rest(self,x,y,psi,r,h):
        """Centre height at which the inflated envelope rests on the pile top (inf if outside)."""
        if not self.inside(x,y,psi,r,h):return np.inf
        i,j,mask,half=self._window(x,y,psi,r,h)
        return float(max(FLOOR_Z+r,np.max(np.where(mask,self.top[i,j]+half,-np.inf))))

    def add(self,x,y,z,psi,r,h):
        i,j,mask,half=self._window(x,y,psi,r,h)
        self.top[i,j]=np.where(mask,np.maximum(self.top[i,j],z+half),self.top[i,j])


def generate(indices,*,seed=SEED,pile=None,yaw=None,trials=None):
    """Drop cans `indices` (inventory numbers) in order; return placements and the pile."""
    rng=np.random.default_rng(seed);pile=pile or _Pile();rows=[]
    yaw=yaw or (lambda g:np.pi/2+AXIS_YAW_SPREAD*g.standard_normal())
    for index in indices:
        d,h,_=CAN_SPECS[kind_of(index)];r=d/2
        best=(np.inf,None)
        for _ in range(trials or TRIALS):
            psi=yaw(rng)
            c,s=np.cos(psi),np.sin(psi);a,b=h/2+CLEARANCE,r+CLEARANCE
            ex=HALF[0]-WALL_MARGIN-(a*abs(c)+b*abs(s));ey=HALF[1]-WALL_MARGIN-(a*abs(s)+b*abs(c))
            if ex<0 or ey<0:continue
            x=CENTRE[0]+rng.uniform(-ex,ex);y=CENTRE[1]+rng.uniform(-ey,ey)
            z=pile.rest(x,y,psi,r,h)
            if z<best[0]:best=(z,(x,y,psi))
        z,(x,y,psi)=best
        # Down-hill slide/turn: the can rolls into the groove it landed beside.
        for step,turn in ((.008,.12),(.004,.06),(.002,.03),(.001,.015)):
            moved=True
            while moved:
                moved=False
                for dx,dy,dpsi in ((step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,turn),(0,0,-turn)):
                    candidate=pile.rest(x+dx,y+dy,psi+dpsi,r,h)
                    if candidate<z-1e-6:z,x,y,psi,moved=candidate,x+dx,y+dy,psi+dpsi,True
        pile.add(x,y,z,psi,r,h)
        rows.append(dict(index=int(index),kind=kind_of(index),centre_m=[float(x),float(y),float(z)],
            axis_yaw_rad=float((psi+np.pi)%(2*np.pi)-np.pi),top_m=float(z+r)))
    return rows,pile


def statistics(rows):
    tops=np.array([row['top_m'] for row in rows])
    return dict(count=len(rows),highest_top_m=float(tops.max()),p90_top_m=float(np.percentile(tops,90)),
        p95_top_m=float(np.percentile(tops,95)),median_top_m=float(np.median(tops)))


def build(first=24,count=120):
    rows,_=generate(range(first,first+count))
    return dict(schema='deposit_cans.high_fill_pile.v1',generator=GENERATOR,seed=SEED,
        parameters=dict(cell_m=CELL,trials=TRIALS,clearance_m=CLEARANCE,wall_margin_m=WALL_MARGIN,
            floor_z_m=FLOOR_Z,interior_half_m=list(HALF),centre_m=list(CENTRE),
            axis_yaw_mean_rad=float(np.pi/2),axis_yaw_spread_rad=float(AXIS_YAW_SPREAD)),
        note='Lying cans in inventory order can_024..can_143; axis yaw and centre fixed, spin about the can axis is the seeded yaw draw.',
        statistics=statistics(rows),cans=rows)


@lru_cache(maxsize=1)
def table():
    data=json.loads(DATA.read_text(encoding='utf-8'))
    if data.get('schema')!='deposit_cans.high_fill_pile.v1' or data.get('generator')!=GENERATOR:
        raise ValueError('Unexpected high_fill pile data')
    return {row['index']:row for row in data['cans']}


def pose(index,spin):
    """Centre and rotation of pre-existing can `index` in the authored high_fill pile."""
    row=table().get(index)
    if row is None or row['kind']!=kind_of(index):raise ValueError('No authored high_fill pile pose for can '+str(index))
    return np.array(row['centre_m']),lying(row['axis_yaw_rad'],spin)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    data=build()
    print(json.dumps(data['statistics']))
    if args.write:DATA.write_text(json.dumps(data,indent=1)+'\n',encoding='utf-8')

"""Contact-dependent conductor network, separate from mechanical retention.

Only registered metal electrodes can form transient edges. Geometric proximity,
connector names and a commanded pose never create a connection. A 1 mN numerical
contact-force floor rejects solver noise; it is not a product calibration claim.
Fixed edges represent physically manufactured conductors in an authored cable.
This lumped continuity model does not simulate Ethernet waveforms or arcing.
"""
from dataclasses import dataclass
import heapq
import math
import numpy as np

@dataclass(frozen=True)
class Edge:
    a:str
    b:str
    resistance_ohm:float
    kind:str="manufactured_conductor"
    normal_force_n:float|None=None
    def __post_init__(self):
        if not self.a or not self.b or self.a==self.b:raise ValueError("Distinct named electrical nodes required")
        if not math.isfinite(self.resistance_ohm) or self.resistance_ohm<=0:raise ValueError("Positive finite resistance required")
        if self.normal_force_n is not None and (not math.isfinite(self.normal_force_n) or self.normal_force_n<0):raise ValueError("Invalid contact force")

def metal_contact_edges(physics,previous_q,*,contact_resistance_ohm=.05,min_normal_force_n=.001):
    """Read the native post-solve force law; do not collide or advance a tick.

    previous_q must be the physical pre-step pose. Native post-solve contact
    forces are not claimed to be separately measured impulses.
    """
    if not math.isfinite(contact_resistance_ohm) or contact_resistance_ohm<=0:raise ValueError("Invalid contact resistance prior")
    if not math.isfinite(min_normal_force_n) or min_normal_force_n<0:raise ValueError("Invalid numerical contact floor")
    electrodes=physics.assembly.electrodes
    from .force_observer import vbd_contact_outputs
    contacts,values,_=vbd_contact_outputs(physics,previous_q)
    count=int(physics.read_array(values[-1])[0])
    force=physics.read_array(values[4])[:count]
    normal=physics.read_array(contacts.rigid_contact_normal)[:count]
    if not np.isfinite(force).all() or not np.isfinite(normal).all():raise FloatingPointError("Non-finite electrical contact observation")
    ids=[physics.read_array(getattr(contacts,"rigid_contact_shape"+str(i)))[:count] for i in (0,1)]
    grouped={}
    for i in range(count):
        s0,s1=int(ids[0][i]),int(ids[1][i])
        if s0 not in electrodes or s1 not in electrodes:continue
        if physics.builder.shape_body[s0]==physics.builder.shape_body[s1]:continue
        f=float(np.dot(force[i],normal[i]))
        if f<min_normal_force_n:continue
        key=tuple(sorted((electrodes[s0],electrodes[s1])))
        if key[0]==key[1]:continue
        grouped[key]=grouped.get(key,0.)+f
    return [Edge(a,b,contact_resistance_ohm,"metal_contact",f) for (a,b),f in sorted(grouped.items())]

class Network:
    def __init__(self,fixed_edges=()):
        self.fixed=tuple(fixed_edges)
        if any(e.kind!="manufactured_conductor" for e in self.fixed):raise ValueError("Fixed edges must be manufactured conductors")
        self.update(())

    def update(self,contacts):
        contacts=tuple(contacts)
        if any(e.kind!="metal_contact" or e.normal_force_n is None for e in contacts):raise ValueError("Measured metal contact edges required")
        old={tuple(sorted((e.a,e.b))) for e in getattr(self,"contacts",())}
        new={tuple(sorted((e.a,e.b))) for e in contacts}
        self.contacts=contacts;self.adjacency={}
        for edge in (*self.fixed,*contacts):
            self.adjacency.setdefault(edge.a,[]).append((edge.b,edge.resistance_ohm))
            self.adjacency.setdefault(edge.b,[]).append((edge.a,edge.resistance_ohm))
        self.transitions={"contact_made":sorted(new-old),"contact_lost":sorted(old-new)}
        return self.transitions

    def path_resistance(self,a,b):
        """Resistance of the lowest-series-resistance path, not parallel equivalent."""
        if a not in self.adjacency or b not in self.adjacency:return None
        todo=[(0.,a)];best={a:0.}
        while todo:
            resistance,node=heapq.heappop(todo)
            if node==b:return resistance
            if resistance>best[node]:continue
            for nxt,r in self.adjacency[node]:
                candidate=resistance+r
                if candidate<best.get(nxt,math.inf):
                    best[nxt]=candidate;heapq.heappush(todo,(candidate,nxt))
        return None

    def connected(self,a,b):return self.path_resistance(a,b) is not None

    def pin_map(self,left,right,pins):
        correct=[self.connected(f"{left}.{i}",f"{right}.{i}") for i in range(1,pins+1)]
        shorts=[(i,j) for i in range(1,pins+1) for j in range(1,pins+1)
                if i!=j and self.connected(f"{left}.{i}",f"{right}.{j}")]
        return dict(all_pins_connected=all(correct),per_pin=correct,wrong_pin_paths=shorts,
                    correct=all(correct) and not shorts)

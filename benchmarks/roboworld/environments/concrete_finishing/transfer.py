"""Conservative material transport for a column-averaged paste model.

Packets carry volume, the actual constitutive material, age and mixture history.
Intensive histories mix by paste volume; surface water and the finite convertible
paste reservoir travel as extensive volumes. Wall adhesion belongs to the receiving
substrate, not the packet. This is a mixed-column approximation, not stratigraphy.
"""
import copy
import numpy as np

HISTORY = ("E", "D", "S", "dwc", "W", "a", "th", "wet_worked", "compact", "tau_w", "late", "worked", "seal_age", "skin_torn")
EXTENSIVE = ("ts", "reserve", "w", "q")


class Packet:
    def __init__(self, volume, material, values=None):
        self.volume = float(volume)
        self.material = material
        self.values = dict(values or {})

    def clone(self, volume=None):
        return Packet(self.volume if volume is None else volume, self.material, self.values)

    def mix(self, other):
        if other.volume <= 0:
            return
        if self.volume <= 1e-18:
            self.material, self.values, self.volume = other.material, dict(other.values), other.volume
            return
        if self.material.name != other.material.name:
            raise ValueError("Different paste species cannot be silently converted")
        total = self.volume + other.volume
        self.values = {k: (self.values.get(k, 0.)*self.volume + other.values.get(k, 0.)*other.volume)/total
                       for k in self.values.keys() | other.values.keys()}
        self.volume = total

    def snapshot(self):
        # Material objects are small, deterministic constitutive configurations.
        return dict(volume=self.volume, material=copy.deepcopy(self.material), values=dict(self.values))

    @classmethod
    def restore(cls, s):
        return cls(s["volume"], copy.deepcopy(s["material"]), s["values"])


def sample(reg, sl, thickness):
    """History of material about to leave these columns, without changing them."""
    amount = np.asarray(thickness, float)
    total = float(amount.sum())
    if total <= 0:
        return Packet(0., reg.material)
    t = reg.t_at(sl)
    values = {f: float((getattr(reg, f)[sl]*amount).sum()/total) for f in HISTORY}
    values["age"] = float((reg.age(sl)*amount).sum()/total)
    # Surface layer is removed first. Convertible fines reside immediately below it.
    surface = np.minimum(amount, reg.ts[sl])
    reserve = np.minimum(np.maximum(amount-surface, 0.), reg.reserve[sl])
    values["ts"] = float(surface.sum()/total)
    values["reserve"] = float(reserve.sum()/total)
    fraction = np.divide(amount, t, out=np.zeros_like(amount), where=t > 1e-15)
    for f in ("w", "q"):
        values[f] = float((getattr(reg, f)[sl]*fraction).sum()/total)
    return Packet(total*reg.cell**2, reg.material, values)


def remove(reg, sl, thickness):
    packet = sample(reg, sl, thickness)
    t = reg.t_at(sl)
    fraction = np.divide(thickness, t, out=np.zeros_like(t), where=t > 1e-15)
    surface = np.minimum(thickness, reg.ts[sl])
    bulk = np.minimum(thickness-surface, reg.tb[sl])
    reg.ts[sl] -= surface
    reg.tb[sl] -= bulk
    reg.reserve[sl] -= np.minimum(bulk, reg.reserve[sl])
    for f in ("w", "q"):
        getattr(reg, f)[sl] *= 1.-fraction
    reg.dirty = True
    return packet


def deposit(reg, sl, thickness, packet):
    """Mix incoming paste into receiving columns, replacing stale empty-cell history."""
    if packet.material.name != reg.material.name:
        raise ValueError("Incompatible receiving material")
    add = np.asarray(thickness, float)
    old = reg.t_at(sl)
    total = old+add
    mask = add > 0
    denom = np.maximum(total, 1e-30)
    for f in HISTORY:
        field = getattr(reg, f)[sl]
        value = packet.values.get(f, -1e9 if f == "seal_age" else 0.)
        mixed = (field*old+value*add)/denom
        field[mask] = mixed[mask]
        getattr(reg, f)[sl] = field
    age = (reg.age(sl)*old + packet.values.get("age", 0.)*add)/denom
    reg.t0[sl] = np.where(mask, reg.clock-age, reg.t0[sl])
    reg.seal_t[sl] = reg.t0[sl]+reg.seal_age[sl]
    surface = add*np.clip(packet.values.get("ts", 0.), 0., 1.)
    reg.tb[sl] += add-surface
    reg.ts[sl] += surface
    for f in ("reserve", "w", "q"):
        getattr(reg, f)[sl] += add*max(packet.values.get(f, 0.), 0.)
    reg.dirty = True


def all_regions(regions):
    for r in regions.values():
        yield r
        yield from r.catches.values()


def catch_deposit(root, indices, thickness, packet, *, floor=False):
    """Sparse 128-cell catch tiles at the actual adjacent wall/slab or floor.

    Coordinates use the parent's infinite lattice, or room XY for a falling load.
    Only allocated cells carry material; empty tiles cost no stepping work. Root
    overlap is excluded so a tool never receives two competing support surfaces.
    """
    from .paste import Region
    ii, jj = (np.asarray(x, int) for x in indices)
    values = np.asarray(thickness, float)
    if not len(ii):
        return
    size = 128
    tags = np.stack([ii//size, jj//size], 1)
    for ti, tj in np.unique(tags, axis=0):
        select = (tags[:, 0] == ti) & (tags[:, 1] == tj)
        key = f'{"floor" if floor else "edge"}_{ti}_{tj}'.replace('-', 'm')
        r = root.catches.get(key)
        if r is None:
            i0, j0 = ti*size, tj*size
            if floor:
                o, U, V = np.array([i0*root.cell, j0*root.cell, 0.]), (1, 0, 0), (0, 1, 0)
            else:
                o, U, V = root.o + i0*root.cell*root.U + j0*root.cell*root.V, root.U, root.V
            r = Region(root.name+'__'+key, root.material, o, U, V, root.cell, (size, size), air=root.air, temp_c=root.temp)
            r.clock = root.clock
            r.ref_mask[:] = True
            r.paste_mask[:] = True
            r.catch_root, r.catch_offset, r.catch_floor = root, (int(i0), int(j0)), floor
            if not floor:
                # Outside a form the support is the room slab, not the form top.
                r.b[:] = -root.o[2] if root.N[2] > .9 else 0.
                I, J = np.meshgrid(np.arange(i0, i0+size), np.arange(j0, j0+size), indexing='ij')
                r.active[:] = ~((I >= 0) & (I < root.nu) & (J >= 0) & (J < root.nv))
            root.catches[key] = r
        ai, aj = ii[select]-ti*size, jj[select]-tj*size
        deposit(r, (ai, aj), values[select], packet)


def snapshot_catches(root):
    return {key: dict(o=r.o.copy(), U=r.U.copy(), V=r.V.copy(), cell=r.cell, shape=(r.nu, r.nv),
                     active=r.active.copy(), offset=r.catch_offset, floor=r.catch_floor, state=r.snapshot())
            for key, r in root.catches.items()}


def restore_catches(root, saved):
    from .paste import Region
    root.catches = {}
    for key, s in saved.items():
        r = Region(root.name+'__'+key, root.material, s['o'], s['U'], s['V'], s['cell'], s['shape'], air=root.air, temp_c=root.temp)
        r.catch_root, r.catch_offset, r.catch_floor = root, s['offset'], s['floor']
        r.active[:] = s['active']; r.ref_mask[:] = True
        r.restore(s['state'])
        root.catches[key] = r


class ContactWindow:
    """A bounded contiguous view across a root and its adjacent catch tiles.

    Every cell has one owner. Constitutive/contact operations use the combined
    footprint once, then scatter all histories back. Empty adjacent cells use the
    same physical wall/slab plane as catches and are allocated only if deposited
    upon. No new material law or cross-tile gravity flow is introduced.
    """
    def __new__(cls, anchor, c, p, R):
        from .paste import Region
        root = getattr(anchor, 'catch_root', anchor)
        floor = getattr(anchor, 'catch_floor', False)
        family = ([root] if not floor else []) + [r for r in root.catches.values() if r.catch_floor == floor]
        origin = np.zeros(3) if floor else root.o
        basis = np.eye(3) if floor else root.R
        centre = (p-origin)@basis
        rr = basis.T@R
        extent = np.abs(rr[:2, 0])*c.sole.hx + np.abs(rr[:2, 1])*c.sole.hy
        # Heave (60 mm), local overflow (20 mm), and the complete roll prism.
        margin = max(.065, np.sqrt(2*max(c.roll,0.)/max(2*min(c.sole.hx,c.sole.hy),root.cell))+.01)
        lo = np.floor((centre[:2]-extent-margin)/root.cell).astype(int)-2
        hi = np.ceil((centre[:2]+extent+margin)/root.cell).astype(int)+2
        # Preserve the original fast path if the whole operation is on the root.
        if not floor and np.all(lo >= 0) and np.all(hi <= [root.nu,root.nv]):
            return root
        def intersects(r):
            off=np.array(getattr(r,'catch_offset',(0,0)))
            return np.all(off < hi) and np.all(off+[r.nu,r.nv] > lo)
        if not any(intersects(r) for r in family if r is not root):
            return anchor
        obj=object.__new__(cls)
        obj.surface=Region(root.name+('__floor' if floor else ''),root.material,origin+basis[:,:2]@(lo*root.cell),
                           basis[:,0],basis[:,1],root.cell,tuple(hi-lo),air=root.air,temp_c=root.temp)
        v=obj.surface
        v.clock=root.clock;v.catch_root=root;v.catch_floor=floor;v.catch_offset=tuple(lo);v.contact_window=obj
        v.b[:]=0. if floor or root.N[2]<.9 else -root.o[2]
        v.ref_mask[:]=True
        obj.root=root;obj.floor=floor;obj.lo=lo;obj.hi=hi
        obj.fields=Region.FIELDS+('active','paste_mask','ref_mask','kind')
        # Optional geometry masks keep native mesh/joint behavior at a root edge.
        for name in ('mesh','holes','edge_l','edge_r','course'):
            if hasattr(root,name):
                setattr(v,name,np.zeros((v.nu,v.nv),dtype=getattr(root,name).dtype))
                obj.fields += (name,)
        for r, dst, src in obj.owners():
            mask=r.active[src]
            for name in obj.fields:
                if hasattr(r,name):getattr(v,name)[dst][mask]=getattr(r,name)[src][mask]
        return v

    def owners(self):
        family=([self.root] if not self.floor else [])+[r for r in self.root.catches.values() if r.catch_floor==self.floor]
        for r in family:
            off=np.array(getattr(r,'catch_offset',(0,0)))
            lo=np.maximum(self.lo,off);hi=np.minimum(self.hi,off+[r.nu,r.nv])
            if np.any(lo>=hi):continue
            dst=tuple(slice(int(a),int(b)) for a,b in zip(lo-self.lo,hi-self.lo))
            src=tuple(slice(int(a),int(b)) for a,b in zip(lo-off,hi-off))
            yield r,dst,src

    def flush(self):
        v=self.surface
        owned=np.zeros((v.nu,v.nv),bool)
        for r,dst,src in self.owners():owned[dst] |= r.active[src]
        ii,jj=np.nonzero((v.t>0)&~owned)
        if len(ii):
            # Allocate ownership without depositing a second copy of the material.
            catch_deposit(self.root,(ii+self.lo[0],jj+self.lo[1]),np.zeros(len(ii)),
                          Packet(0.,v.material),floor=self.floor)
        for r,dst,src in self.owners():
            mask=r.active[src]
            for name in self.fields:
                if hasattr(r,name):getattr(r,name)[src][mask]=getattr(v,name)[dst][mask]
            r.dirty=True

"""Conservative contact-conditioned surface film / porous-cloth transport.

This is a reduced model, NOT resolved Navier-Stokes or microbiology. All masses
are kg. Contact witnesses and forces come from the native cloth contact solve.
No image coverage, arm pose, task progress or success flag removes material.
The same exchange law applies to clean and already dirty surfaces.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

SPECIES = ('water','food_solids','oil')


def _scatter_sum(indices, values, size):
    """Zero-based float64 sums, retaining contribution order within each bin."""
    if values.ndim == 1:
        return np.bincount(indices, weights=values, minlength=size).astype(np.float64, copy=False)
    result = np.empty((size, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        result[:, k] = np.bincount(indices, weights=values[:, k], minlength=size)
    return result


@dataclass
class Surface:
    name: str
    shape: int
    body: int
    origin: tuple
    u: tuple
    v: tuple
    size: tuple
    resolution: float = .005
    protected: bool = False
    material: str = 'steel'
    triangle_uv: tuple | None = None

    def __post_init__(self):
        self.nx,self.ny = np.ceil(np.asarray(self.size)/self.resolution).astype(int)
        self.du,self.dv = self.size[0]/self.nx,self.size[1]/self.ny
        self.cell_area = self.du*self.dv
        if self.triangle_uv is not None:
            self.nx=self.ny=1;self.du,self.dv=self.size
            self.triangle_uv=np.asarray(self.triangle_uv,float)
            a,b=self.triangle_uv[1:]-self.triangle_uv[0]
            self.cell_area=abs(float(a[0]*b[1]-a[1]*b[0]))/2
            if self.cell_area<=0:raise ValueError('Degenerate material triangle')
        self.u,self.v,self.origin = (np.asarray(a,float) for a in (self.u,self.v,self.origin))
        self.normal = np.cross(self.u,self.v)
        if not np.allclose([np.linalg.norm(self.u),np.linalg.norm(self.v),
                            np.linalg.norm(self.normal)],[1.,1.,1.]):
            raise ValueError('Surface frame must be orthonormal')

    def contains(self,uv,tolerance=0.):
        uv=np.asarray(uv)
        if self.triangle_uv is None:
            return (uv[:,0]>=-tolerance)&(uv[:,1]>=-tolerance)&(uv[:,0]<self.size[0]+tolerance)&(uv[:,1]<self.size[1]+tolerance)
        inside=np.ones(len(uv),bool)
        for p,q in zip(self.triangle_uv,np.roll(self.triangle_uv,-1,axis=0)):
            edge=q-p;r=uv-p
            inside&=(edge[0]*r[:,1]-edge[1]*r[:,0]>=-tolerance*np.linalg.norm(edge))
        return inside

    @classmethod
    def from_triangle(cls,name,shape,body,points,**kwargs):
        points=np.asarray(points,float);u=points[1]-points[0];u/=np.linalg.norm(u)
        normal=np.cross(u,points[2]-points[0]);normal/=np.linalg.norm(normal)
        v=np.cross(normal,u);uv=np.column_stack(((points-points[0])@u,(points-points[0])@v))
        low=uv.min(0);origin=points[0]+low[0]*u+low[1]*v;uv-=low
        return cls(name,shape,body,tuple(origin),tuple(u),tuple(v),tuple(uv.max(0)),triangle_uv=tuple(map(tuple,uv)),**kwargs)


class Transport:
    def __init__(self,sim):
        c,a = sim.config,sim.assembly
        self.sim=sim
        self.surfaces=[
            Surface('worktop',a.surface_shapes['worktop'],-1,
                (-c.table_depth/2,-c.table_width/2,c.table_height),
                (1,0,0),(0,1,0),(c.table_depth,c.table_width)),
            Surface('backsplash',a.surface_shapes['backsplash'],-1,
                (c.table_depth/2,-c.table_width/2,c.table_height),
                (0,0,1),(0,1,0),(c.backsplash_height,c.table_width)),
            Surface('front_edge',a.surface_shapes['worktop'],-1,
                (-c.table_depth/2,-c.table_width/2,c.table_height-.04),
                (0,0,1),(0,1,0),(.04,c.table_width))]
        if c.scene=='kitchen':
            from .surface_inventory import kitchen_faces
            self.surfaces.extend(kitchen_faces(a,Surface))
        self.offsets=np.cumsum([0]+[s.nx*s.ny for s in self.surfaces])
        self._static_frames=[(s.origin,s.u,s.v,s.normal) for s in self.surfaces]
        self._dynamic_frames=[]
        for body in sorted({s.body for s in self.surfaces if s.body>=0}):
            ids=[i for i,s in enumerate(self.surfaces) if s.body==body]
            self._dynamic_frames.append((body,ids,np.asarray([self._static_frames[i] for i in ids])))
        # Species-contiguous storage avoids repeatedly striding the entire
        # kitchen ledger for exact species totals. Logical indexing is unchanged.
        self.surface=np.zeros((self.offsets[-1],3),np.float64,order='F')
        self.cloth=np.zeros((sim.model.particle_count,3),np.float64)
        self.cloth[:,0]=sim.water
        self.capacity=sim.dry_mass.astype(float)*c.water_capacity_fraction
        self.area=sim.dry_mass/(c.cloth_dry_mass/c.cloth_side**2)
        self.external=np.zeros((3,),float)  # exported waste/vapor only by explicit named operation
        self.pickup_total=np.zeros(3)
        self.release_total=np.zeros(3)
        self.contact_work_j=0.
        self.initial=None
        self.step_metrics={}
        self.motion=np.zeros((len(self.surface),3),float)
        self.source_velocity=np.empty((0,3))
        self.capture_momentum=np.zeros_like(self.cloth)
        self.flow=None
        if c.film_transport:
            from .film_flow import FilmFlow
            self.flow=FilmFlow(self)

    def frames(self):
        q=self.sim.state_0.body_q.numpy() if self.sim.state_0.body_q is not None else np.empty((0,7))
        result=list(self._static_frames)
        for body,ids,local in self._dynamic_frames:
            world=Rotation.from_quat(q[body,3:]).apply(local.reshape(-1,3)).reshape(local.shape)
            world[:,0,:]+=q[body,:3]
            for index,frame in zip(ids,world):result[index]=tuple(frame)
        return result

    def add_patch(self,name,uv,radii,masses):
        """Episode construction only. Exact specified mass on a Gaussian patch."""
        index=next(i for i,s in enumerate(self.surfaces) if s.name==name)
        s=self.surfaces[index]
        x,y=np.meshgrid((np.arange(s.nx)+.5)*s.du,(np.arange(s.ny)+.5)*s.dv,indexing='ij')
        w=np.exp(-.5*((x-uv[0])/radii[0])**2-.5*((y-uv[1])/radii[1])**2)
        w[w<.01]=0
        if not w.sum()>0 or np.any(np.asarray(masses)<0):
            raise ValueError('Invalid soil patch')
        self.surface[self.offsets[index]:self.offsets[index+1]] += w.reshape(-1,1)/w.sum()*np.asarray(masses)

    def freeze_initial(self):
        self.initial=self.total().copy()

    def add_spill(self,name,uv,radii,masses,rng):
        """Authored asymmetric spill lobes, preserving exact total material mass."""
        center=np.asarray(uv);radii=np.asarray(radii);masses=np.asarray(masses)
        self.add_patch(name,center,radii*.65,masses*.55)
        weights=rng.uniform(.5,1.5,8);weights*=.45/weights.sum()
        for weight in weights:
            offset=rng.uniform(-1.,1.,2)*radii*1.5
            size=radii*rng.uniform(.12,.45,2)
            self.add_patch(name,center+offset,size,masses*weight)

    def total(self):
        return self.surface.sum(0)+self.cloth.sum(0)+self.external+(self.flow.total() if self.flow else 0.)

    def witnesses(self):
        """Return only actual positive-load native contacts, mapped to exposed faces."""
        sim=self.sim
        n=int(sim.contacts.soft_contact_count.numpy()[0])
        if n>sim.contacts.soft_contact_max:
            raise RuntimeError(f'Native soft-contact buffer overflow at tick {sim.tick}: '
                               f'{n} > {sim.contacts.soft_contact_max}; material exchange stopped')
        if n==0:
            self.motion.fill(0.);self.source_velocity=np.empty((0,3))
            return np.zeros(0,int),np.zeros(0,int),np.zeros(0),np.zeros(0),np.zeros(0)
        shape=sim.contacts.soft_contact_shape.numpy()[:n]
        corners=sim.contacts.soft_contact_indices.numpy()[:n]
        bary=sim.contacts.soft_contact_barycentric.numpy()[:n]
        point=sim.contacts.soft_contact_body_pos.numpy()[:n]
        normal=sim.contacts.soft_contact_normal.numpy()[:n]
        force=sim.solver.collision.forces.numpy()[:n]
        load=np.maximum(0,np.sum(force*normal,axis=1))
        active=load>1e-8
        shape,corners,bary,point,normal,load=(x[active] for x in (shape,corners,bary,point,normal,load))
        qd=sim.state_0.particle_qd.numpy()
        velocity=(qd[np.maximum(corners,0)]*bary[:,:,None]).sum(1)
        body_q=sim.state_0.body_q.numpy() if sim.state_0.body_q is not None else np.empty((0,7))
        body_qd=sim.state_0.body_qd.numpy() if sim.state_0.body_qd is not None else np.empty((0,6))
        com=sim.model.body_com.numpy()
        body_of_shape=np.asarray(sim.assembly.builder.shape_body)
        local_normals=normal.astype(float);surface_velocities=np.zeros(velocity.shape,dtype=float)
        for body in np.unique(body_of_shape[shape]):
            if body<0:continue
            ids=np.flatnonzero(body_of_shape[shape]==body)
            rot=Rotation.from_quat(body_q[body,3:])
            local_normals[ids]=rot.inv().apply(normal[ids])
            arm=rot.apply(point[ids]-com[body])
            surface_velocities[ids]=body_qd[body,:3]+np.cross(body_qd[body,3:],arm)
        relative_velocities=velocity-surface_velocities
        relative_velocities-=normal*np.sum(relative_velocities*normal,axis=1)[:,None]
        local_speeds=np.linalg.norm(relative_velocities,axis=1)
        motion=np.zeros_like(self.motion);pressure_weight=np.zeros(len(self.surface))
        cells=[];particles=[];loads=[];speeds=[];weights=[];source_velocity=[]
        by_shape={sh:np.flatnonzero(shape==sh) for sh in np.unique(shape)}
        # One physical witness must select one exposed face, including when a
        # formed pan has several nearly parallel patches on ONE mesh shape.
        # Normal similarity alone could otherwise double-count interior/exterior
        # strips. Positions/normals here are in the body's local frame.
        chosen=np.full(len(shape),-1,int);nearest=np.full(len(shape),np.inf)
        for i,s in enumerate(self.surfaces):
            if s.shape not in by_shape:continue
            ids=by_shape[s.shape];relative=point[ids]-s.origin
            uv=np.column_stack((relative@s.u,relative@s.v))
            distance=abs(relative@s.normal)
            valid=(local_normals[ids]@s.normal>.8)&s.contains(uv,tolerance=1e-7)
            valid&=(distance<=.003)&(distance<nearest[ids])
            chosen[ids[valid]]=i;nearest[ids[valid]]=distance[valid]
        for i,s in enumerate(self.surfaces):
            if s.shape not in by_shape:continue
            ids=np.flatnonzero(chosen==i)
            if not len(ids):continue
            local_normal=local_normals[ids];surface_velocity=surface_velocities[ids]
            relative=point[ids]-s.origin
            uv=np.column_stack((relative@s.u,relative@s.v))
            valid=(np.sum(local_normal*s.normal,axis=1)>.8)&s.contains(uv,tolerance=1e-7)
            surface_velocity=surface_velocity[valid]
            ids=ids[valid];uv=uv[valid]
            relative_velocity=relative_velocities[ids];local_speed=local_speeds[ids]
            cell=self.offsets[i]+np.clip((uv[:,0]/s.du).astype(int),0,s.nx-1)*s.ny+np.clip((uv[:,1]/s.dv).astype(int),0,s.ny-1)
            np.add.at(motion,cell,relative_velocity*load[ids,None])
            np.add.at(pressure_weight,cell,load[ids])
            for k in range(3):
                active=(corners[ids,k]>=0)&(bary[ids,k]>1e-8)
                hit=ids[active];w=bary[hit,k]
                cells.append(cell[active]);particles.append(corners[hit,k])
                loads.append(load[hit]*w);speeds.append(local_speed[active]);weights.append(w)
                source_velocity.append(surface_velocity[active])
        self.motion[:]=motion/np.maximum(pressure_weight[:,None],1e-30)
        self.source_velocity=np.concatenate(source_velocity) if source_velocity else np.empty((0,3))
        return tuple(np.concatenate(a) if a else np.empty(0,dtype=d) for a,d in
                     ((cells,int),(particles,int),(loads,float),(speeds,float),(weights,float)))

    def exchange(self,cell,particle,load,speed,dt,weights=None,source_velocity=None):
        """Bounded paired transfers; shared witnesses cannot overdraw source bins.

        Mass flows are first requested, then scaled by BOTH source availability
        and receiving pore capacity. Deposits remain on the surface and can be
        picked up on later steps. Array round-off is audited, never 'repaired'
        by deleting negative mass. Rates/pressure scales are documented estimates.
        """
        if not len(cell):
            self.capture_momentum.fill(0.)
            self.step_metrics=dict(contacts=0,picked_up_kg=[0.]*3,released_kg=[0.]*3)
            return np.zeros_like(self.cloth),np.zeros_like(self.cloth)
        if dt<=0 or np.any(load<0) or np.any(speed<0):
            raise ValueError('Nonphysical exchange input')
        if weights is None:weights=np.ones(len(cell))
        area=self.area[particle]*weights
        pressure=load/area
        pressure_factor=pressure/(pressure+300.)
        motion=.10+np.minimum(speed/.05,20.)
        effective_capacity=self.capacity[particle]*(1-.65*pressure/(pressure+1500.))
        saturation=np.minimum(self.cloth[particle].sum(1)/np.maximum(effective_capacity,1e-15),2.)
        # Water wicks fastest; viscous food and oil transfer more slowly.
        uptake_hz=np.asarray((5.,1.5,.8))
        release_hz=np.asarray((1.5,.4,.25))
        native_touch=(load>1e-8).astype(float)
        pickup=self.surface[cell]*(-np.expm1(-dt*motion[:,None]*(.3+pressure_factor[:,None])*
                                uptake_hz))*native_touch[:,None]*weights[:,None]
        deposit=self.cloth[particle]*(-np.expm1(-dt*motion[:,None]*pressure_factor[:,None]*
                                saturation[:,None]**2*release_hz))*native_touch[:,None]*weights[:,None]
        touched,local_cell=np.unique(cell,return_inverse=True)
        requested_surface=_scatter_sum(local_cell,pickup,len(touched))
        requested_pickup=_scatter_sum(particle,pickup.sum(1),len(self.cloth))
        requested_deposit=_scatter_sum(particle,deposit,len(self.cloth))
        free=np.maximum(0,self.capacity-self.cloth.sum(1))
        receive_scale=np.minimum(1,free/np.maximum(requested_pickup,1e-30))
        source_scale=np.minimum(1,self.surface[touched]/np.maximum(requested_surface,1e-30))
        pickup*=source_scale[local_cell]*receive_scale[particle,None]
        deposit*=np.minimum(1,self.cloth/np.maximum(requested_deposit,1e-30))[particle]
        observer=getattr(self,'exchange_observer',None)
        if observer is not None:observer(cell,pickup,deposit)
        self.capture_momentum.fill(0.)
        if source_velocity is not None:
            self.capture_momentum[:]=_scatter_sum(particle,pickup.sum(1)[:,None]*source_velocity,len(self.cloth))
        picked=_scatter_sum(particle,pickup,len(self.cloth))
        released=_scatter_sum(particle,deposit,len(self.cloth))
        removed_surface=_scatter_sum(local_cell,pickup,len(touched))
        added_surface=_scatter_sum(local_cell,deposit,len(touched))
        # Positivity-preserving source depletion. Subtracting many separately
        # rounded requests from an exhausted cell can leave a -1e-19 kg residue;
        # using its bounded remaining fraction avoids feeding that sign into
        # the next step's availability ratios. No arbitrary mass cleanup.
        self.cloth*=1-np.minimum(1,released/np.maximum(self.cloth,1e-30))
        self.cloth+=picked
        self.surface[touched]*=1-np.minimum(1,removed_surface/np.maximum(self.surface[touched],1e-30))
        self.surface[touched]+=added_surface
        if self.surface.min() < -1e-14 or self.cloth.min() < -1e-14:
            raise FloatingPointError('Negative material mass')
        self.pickup_total+=pickup.sum(0)
        self.release_total+=deposit.sum(0)
        self.contact_work_j+=float(np.sum(load*speed)*dt)
        self.step_metrics=dict(contacts=len(cell),picked_up_kg=pickup.sum(0).tolist(),
                              released_kg=deposit.sum(0).tolist(),max_pressure_pa=float(pressure.max()))
        return picked,released

    def step(self):
        old_mass=self.sim.dry_mass.astype(float)+self.cloth.sum(1)
        cell,particle,load,speed,weights=self.witnesses()
        picked,released=self.exchange(cell,particle,load,speed,self.sim.config.dt,weights,self.source_velocity)
        new_mass=self.sim.dry_mass+self.cloth.sum(1)
        # Capturing initially stationary surface film dilutes particle momentum;
        # released material leaves at cloth velocity. Static bench takes impulse.
        v=self.sim.state_0.particle_qd.numpy()
        scale=(old_mass-released.sum(1))/new_mass
        v*=scale[:,None]
        v+=self.capture_momentum/new_mass[:,None]
        self.sim.state_0.particle_qd.assign(v.astype(np.float32))
        self.sim.water=self.cloth[:,0].astype(np.float32)
        self.sim.model.particle_mass.assign(new_mass.astype(np.float32))
        self.sim.model.particle_inv_mass.assign((1/new_mass).astype(np.float32))
        c=self.sim.config
        if self.flow and self.sim.tick%round(1/(c.film_hz*c.dt))==0:
            self.flow.advance(1/c.film_hz)
        residual=self.total()-self.initial
        self.last_residual=residual.copy()
        if not np.isfinite(residual).all() or np.max(abs(residual))>1e-10:
            raise FloatingPointError(f'Material ledger residual {residual}')

    def ledger(self):
        total=self.total()
        residual=total-self.initial if self.initial is not None else np.zeros(3)
        return dict(species=list(SPECIES),surface_kg=self.surface.sum(0).tolist(),
            cloth_kg=self.cloth.sum(0).tolist(),external_kg=self.external.tolist(),
            total_kg=total.tolist(),initial_kg=None if self.initial is None else self.initial.tolist(),
            residual_kg=residual.tolist(),cumulative_pickup_kg=self.pickup_total.tolist(),
            cumulative_release_kg=self.release_total.tolist(),last_exchange=self.step_metrics,
            runoff=None if self.flow is None else self.flow.ledger(),
            by_surface={s.name:self.surface[self.offsets[i]:self.offsets[i+1]].sum(0).tolist()
                        for i,s in enumerate(self.surfaces)})

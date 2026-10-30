"""Conservative reduced thin-film advection and ray-collided ballistic runoff.

SI units. Laminar depth-averaged gravity/height-gradient flow and cloth-driven
Couette entrainment; this is not resolved CFD or a capillary contact-line model.
No state is removed on leaving a face. Runoff is recorded until first physical
rigid-shape impact, deposition, or explicitly accounted escape from the workcell.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from newton._src.geometry.raycast import intersect_ray

DENSITY=np.asarray((1000.,1250.,920.))
# Finite-volume numerical tails are retained in their donor boundary cell until
# a boundary transfer reaches 1 nanogram. No mass is deleted or put off ledger.
# This is an emission quadrature cutoff, NOT a droplet-size/absorption model.
MIN_EMISSION_KG=1e-12


def coalesce_identical_packets(positions,velocities,mass):
    """Combine only EXACTLY identical ballistic phase-space quadrature points.

    CFL substeps can emit many rows at the same boundary position and velocity.
    They then share every future ray/trajectory. Sum their species weights;
    do not merge nearby positions, average velocities, delete mass, increase
    the emission threshold or weaken the distinct-packet capacity guard.
    """
    if len(mass)<4096:return positions,velocities,mass
    _,first,inverse=np.unique(np.column_stack((positions,velocities)),axis=0,
        return_index=True,return_inverse=True)
    if len(first)==len(mass):return positions,velocities,mass
    merged=np.zeros((len(first),3),dtype=mass.dtype)
    np.add.at(merged,inverse,mass)
    order=np.argsort(first)
    return positions[first[order]],velocities[first[order]],merged[order]


class FilmFlow:
    def __init__(self,transport):
        self.t=transport
        self.positions=np.empty((0,3),float)
        self.velocities=np.empty((0,3),float)
        self.mass=np.empty((0,3),float)
        self.unmapped=np.zeros((transport.sim.model.shape_count,3),float)
        self.escaped=np.zeros(3)
        self.emitted=np.zeros(3)
        self.landed=np.zeros(3)
        self.steps=0
        self.impact_local=np.empty((0,3),float)
        self.impact_body=np.empty(0,int)
        self.impact_mass=np.empty((0,3),float)
        from .mesh_surface_flow import edge_network
        self.triangle_edges=edge_network(transport.surfaces,transport.offsets)

    def total(self):return self.mass.sum(0)+self.unmapped.sum(0)+self.escaped

    def advance(self,dt):
        t=self.t;frames=t.frames();emit_p=[];emit_v=[];emit_m=[];mesh_deposits=[]
        gravity=np.asarray((0.,0.,-9.80665))
        body_v=t.sim.state_0.body_qd.numpy() if t.sim.state_0.body_qd is not None else np.empty((0,6))
        body_q=t.sim.state_0.body_q.numpy() if t.sim.state_0.body_q is not None else np.empty((0,7))
        com=t.sim.model.body_com.numpy()
        # Exact sparse sweep: omit only identically empty cells/faces, without
        # a material cutoff. Ordered nonempty faces retain the original sweep
        # order; mesh deposits are still applied after the complete sweep.
        live_cells=np.flatnonzero(np.any(t.surface!=0.,axis=1))
        live_faces=np.unique(np.searchsorted(t.offsets[1:],live_cells,side='right'))
        for i in live_faces:
            s=t.surfaces[i]
            start,end=t.offsets[i:i+2]
            current=t.surface[start:end]
            if not np.any(current):continue
            mass=current.reshape(s.nx,s.ny,3)
            origin,u,v,normal=frames[i]
            if s.triangle_uv is not None:
                from .mesh_surface_flow import advance_cell
                remaining,deposits=advance_cell(s,current[0].copy(),np.array((gravity@u,gravity@v)),
                    .5*np.array((t.motion[start]@u,t.motion[start]@v)),self.triangle_edges[i],dt,DENSITY)
                current[0]=remaining;mesh_deposits.extend(deposits)
                continue
            height=(mass/DENSITY).sum(2)/s.cell_area
            # A dry sticky residue resists flow; water reduces its effective
            # viscosity/yield stress. These authored constitutive estimates are
            # exposed in the parameter ledger, not manufacturer measurements.
            wet_ratio=mass[:,:,0]/np.maximum(mass[:,:,1],1e-12)
            viscosity=np.broadcast_to((.001,.5,.08),mass.shape).copy()
            viscosity[:,:,1]=np.maximum(.005,.5/(1+10*wet_ratio))
            yield_stress=np.zeros_like(mass);yield_stress[:,:,1]=3/(1+wet_ratio)
            drive=np.zeros((*height.shape,2))
            gn=max(0.,-float(gravity@normal))
            for axis,(direction,spacing) in enumerate(((u,s.du),(v,s.dv))):
                gradient=np.gradient(height,spacing,axis=axis) if height.shape[axis]>1 else np.zeros_like(height)
                drive[:,:,axis]=gravity@direction-gn*gradient
            tau=np.linalg.norm(drive,axis=2)[:,:,None]*height[:,:,None]*DENSITY
            yielded=np.maximum(0.,1-yield_stress/np.maximum(tau,1e-20))
            speed=drive[:,:,None,:]*(DENSITY*height[:,:,None]**2/(3*viscosity)*yielded)[:,:,:,None]
            motion=t.motion[start:end].reshape(s.nx,s.ny,3)
            # Actual positive-load cloth witnesses supply this velocity; no
            # commanded arm path or task completion state enters this law.
            drag=.5*np.stack((motion@u,motion@v),axis=-1)
            speed+=drag[:,:,None,:]
            cfl=float(np.max((abs(speed[:,:,:,0])/s.du+abs(speed[:,:,:,1])/s.dv)*dt))
            substeps=max(1,int(np.ceil(cfl/.45)))
            if substeps>1024:raise FloatingPointError(f'Thin-film CFL on {s.name} requires {substeps} substeps; '
                f'peak thickness {height.max()} m, speed {abs(speed).max()} m/s; unsupported deep/fast flow')
            # The transport velocity is frozen over these CFL substeps in the
            # existing discretization. Hoist only invariant arithmetic: no
            # changed substep count, flux ordering, threshold or mass update.
            directions=[]
            for axis,spacing in enumerate((s.du,s.dv)):
                for sign in (-1,1):
                    fraction=np.maximum(0.,sign*speed[:,:,:,axis])*(dt/substeps)/spacing
                    if axis==0:
                        coords=np.column_stack((np.full(s.ny,s.size[0]+1e-6 if sign==1 else -1e-6),(np.arange(s.ny)+.5)*s.dv))
                        h=height[-1 if sign==1 else 0];vs=speed[-1 if sign==1 else 0]
                    else:
                        coords=np.column_stack(((np.arange(s.nx)+.5)*s.du,np.full(s.nx,s.size[1]+1e-6 if sign==1 else -1e-6)))
                        h=height[:,-1 if sign==1 else 0];vs=speed[:,-1 if sign==1 else 0]
                    directions.append((axis,sign,fraction,coords,h,vs))
            for _ in range(substeps):
                out=np.zeros_like(mass);incoming=np.zeros_like(mass)
                for axis,sign,fraction,coords,h,vs in directions:
                    flux=mass*fraction
                    edge=flux[-1 if sign==1 else 0] if axis==0 else flux[:,-1 if sign==1 else 0]
                    edge[edge.sum(1)<MIN_EMISSION_KG]=0.
                    out+=flux
                    if axis==0:
                        if sign==1:incoming[1:]+=flux[:-1];edge=flux[-1]
                        else:incoming[:-1]+=flux[1:];edge=flux[0]
                    else:
                        if sign==1:incoming[:,1:]+=flux[:,:-1];edge=flux[:,-1]
                        else:incoming[:,:-1]+=flux[:,1:];edge=flux[:,0]
                    live=edge.sum(1)>0
                    if np.any(live):
                        m=edge[live];uv=coords[live]
                        p=origin+uv[:,0,None]*u+uv[:,1,None]*v+(h[live,None]+1e-5)*normal
                        velocity2=(vs[live]*m[:,:,None]).sum(1)/m.sum(1)[:,None]
                        velocity=velocity2[:,0,None]*u+velocity2[:,1,None]*v
                        if s.body>=0:
                            center=Rotation.from_quat(body_q[s.body,3:]).apply(com[s.body])+body_q[s.body,:3]
                            velocity+=body_v[s.body,:3]+np.cross(body_v[s.body,3:],p-center)
                        emit_p.append(p);emit_v.append(velocity);emit_m.append(m)
                mass[:]=mass*(1-np.minimum(1,out/np.maximum(mass,1e-30)))+incoming
        # Apply once after the sweep: no order-dependent repeated traversal of
        # several corner cells in a single material step.
        for cell,mass in mesh_deposits:t.surface[cell]+=mass
        if emit_m:
            mass=np.concatenate(emit_m);self.emitted+=mass.sum(0)
            self.positions=np.concatenate((self.positions,*emit_p))
            self.velocities=np.concatenate((self.velocities,*emit_v))
            self.mass=np.concatenate((self.mass,mass))
        self.flight(dt,frames)
        self.steps+=1

    def flight(self,dt,frames):
        if not len(self.mass):return
        self.positions,self.velocities,self.mass=coalesce_identical_packets(
            self.positions,self.velocities,self.mass)
        sim=self.t.sim;m=sim.model;n=len(self.mass)
        if n>100000:raise RuntimeError('Runoff packet capacity exceeded; no packets discarded')
        delta=self.velocities*dt+np.asarray((0,0,-.5*9.80665*dt*dt))
        length=np.linalg.norm(delta,axis=1)
        direction=delta/np.maximum(length[:,None],1e-30)
        origins=wp.array(self.positions,dtype=wp.vec3,device=m.device)
        directions=wp.array(direction,dtype=wp.vec3,device=m.device)
        worlds=wp.full(n,-1,dtype=wp.int32,device=m.device)
        distance=wp.empty(n,dtype=float,device=m.device);shape=wp.empty(n,dtype=wp.int32,device=m.device)
        normal=wp.empty(n,dtype=wp.vec3,device=m.device)
        m.bvh_refit_shapes(sim.state_0)
        intersect_ray(m,ray_origins=origins,ray_directions=directions,ray_worlds=worlds,
                      out_dist=distance,out_shape_id=shape,out_normal=normal)
        d,sh,no=distance.numpy(),shape.numpy(),normal.numpy()
        hit=(sh>=0)&(d>=0)&(d<=length+1e-7)
        point=self.positions+direction*d[:,None]
        unmatched=hit.copy()
        for i,s in enumerate(self.t.surfaces):
            ids=np.flatnonzero(unmatched&(sh==s.shape))
            if not len(ids):continue
            origin,u,v,norm=frames[i];r=point[ids]-origin
            uv=np.column_stack((r@u,r@v))
            ok=(no[ids]@norm>.8)&(abs(r@norm)<1e-5)&s.contains(uv,tolerance=1e-5)
            ids=ids[ok];uv=uv[ok]
            ix=np.clip((uv[:,0]/s.du).astype(int),0,s.nx-1)
            iy=np.clip((uv[:,1]/s.dv).astype(int),0,s.ny-1)
            cells=self.t.offsets[i]+ix*s.ny+iy
            np.add.at(self.t.surface,cells,self.mass[ids]);unmatched[ids]=False
        if np.any(unmatched):
            np.add.at(self.unmapped,sh[unmatched],self.mass[unmatched])
            points=point[unmatched].copy();bodies=m.shape_body.numpy()[sh[unmatched]]
            dynamic=bodies>=0
            if dynamic.any():
                poses=sim.state_0.body_q.numpy()[bodies[dynamic]]
                points[dynamic]=Rotation.from_quat(poses[:,3:]).inv().apply(points[dynamic]-poses[:,:3])
            self.impact_local=np.concatenate((self.impact_local,points))
            self.impact_body=np.r_[self.impact_body,bodies]
            self.impact_mass=np.concatenate((self.impact_mass,self.mass[unmatched]))
        self.landed+=self.mass[hit].sum(0)
        self.positions+=delta;self.velocities[:,2]-=9.80665*dt
        escaped=(~hit)&((np.abs(self.positions[:,:2]).max(1)>3.5)|(self.positions[:,2]<-.1))
        self.escaped+=self.mass[escaped].sum(0)
        keep=~(hit|escaped)
        self.positions,self.velocities,self.mass=self.positions[keep],self.velocities[keep],self.mass[keep]

    def ledger(self):
        return dict(in_flight_kg=self.mass.sum(0).tolist(),unmapped_shape_kg=self.unmapped.sum(0).tolist(),
                    escaped_kg=self.escaped.tolist(),emitted_kg=self.emitted.tolist(),
                    landed_kg=self.landed.tolist(),packets=len(self.mass),material_steps=self.steps,
                    packet_representation='Exact equal-position/equal-velocity rows share summed species weights; no proximity merging')

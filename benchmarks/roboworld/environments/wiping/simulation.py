"""Finite-effort robot/deformable cloth with explicit solver configuration.

Native coupled contact is under qualification; legacy candidates are retained
for provenance. Material exchange observes actual native contact witnesses.
"""
import importlib.metadata
import time
from types import SimpleNamespace
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo,SolverStyle3D
from laundry.newton15.coupling import ReactionCollision
from laundry.newton15.graph import CapturedClothStep,CapturedRigidStep,CapturedCollision,CapturedPCG
from laundry.newton_style3d_topology_contact import TopologicallyFilteredStyle3DCollision
from .assembly import Assembly
from .config import Config


class Simulation:
    def __init__(self,c=Config(),*,robots=True,folded=True):
        if importlib.metadata.version("newton") != "1.5.0":
            raise RuntimeError("Requires Newton exactly 1.5.0")
        self.config=c
        self.assembly=Assembly(c,robots=robots,folded=folded)
        self.assembly.builder.validate_inertia_detailed=True
        self.model=self.assembly.builder.finalize(device=c.device)
        m=self.model
        m.soft_contact_ke=c.contact_ke
        m.soft_contact_kd=c.contact_kd
        m.soft_contact_mu=c.contact_mu
        m.soft_contact_radius=c.cloth_radius
        m.soft_contact_margin=c.cloth_contact_margin
        self.dry_mass=m.particle_mass.numpy().copy()
        self.water=np.zeros(m.particle_count,np.float32)
        for cloth in self.assembly.cloths:
            s=slice(cloth['first'],cloth['first']+cloth['count'])
            self.water[s]=self.dry_mass[s]*cloth['initial_water_fraction']
        self.update_wet_mass()
        self.state_0,self.state_1=m.state(),m.state()
        newton.eval_fk(m,m.joint_q,m.joint_qd,self.state_0)
        self.state_1.assign(self.state_0)
        self.control=m.control()
        self.targets=self.control.joint_target_q.numpy().copy() if self.control.joint_target_q is not None else np.zeros(0,np.float32)
        self.rates=np.zeros(len(self.targets),np.float32)
        for spec in self.assembly.robot_joints.values():
            self.rates[spec['q']]=spec['velocity_limit']
        self.pipeline=newton.CollisionPipeline(m,broad_phase='nxn',
            soft_contact_margin=c.cloth_contact_margin,enable_rigid_soft_full_surface_contact=c.cloth_solver=='vbd',
            soft_contact_max=c.contact_capacity)
        self.contacts=self.pipeline.contacts()
        if c.native_analytic_box_contacts:
            from .box_contacts import install
            install(self.pipeline,m,stable=c.native_stable_box_contacts)
        if c.native_shape_grouped_contacts:
            from .contact_order import group_soft_pairs
            group_soft_pairs(self.pipeline)
        # Native coupling generates its OWN full-surface cloth contact set after
        # synchronizing finite-inertia robot proxies. The outer set is consumed
        # only by the particle-free MuJoCo entry; its soft contacts were generated
        # and immediately replaced unused. Keep ALL rigid contacts here and ALL
        # cloth contacts in the native proxy pipeline, without duplicate work.
        self._rigid_collision_states={id(s):SimpleNamespace(body_q=s.body_q,particle_q=None)
                                      for s in (self.state_0,self.state_1)} if c.coupling=='native' else None
        if c.coupling=='native':
            from .native_coupling import NativeCoupled
            self.solver=NativeCoupled(self)
        elif c.cloth_solver=='vbd':
            from .vbd_contact import FullSurfaceCloth
            self.solver=FullSurfaceCloth(self)
        else:
            self.solver=SolverStyle3D(m,iterations=c.nonlinear_iterations,linear_iterations=c.linear_iterations,
                                    enable_mouse_dragging=False)
            self.solver.collision=TopologicallyFilteredStyle3DCollision(m,m.tri_indices.numpy(),graph_hops=1)
            self.solver.collision.radius=c.cloth_radius
            self.solver.collision.integrate_with_external_rigid_solver=True
            self.solver.collision=ReactionCollision(self.solver.collision,self)
        self.rigid_solver=self.solver.rigid if c.coupling=='native' else SolverMuJoCo(m,use_mujoco_contacts=False,integrator='implicitfast',
            cone='elliptic',solver='newton',iterations=c.rigid_iterations,ls_iterations=20,
            nconmax=c.contact_capacity,njmax=c.contact_capacity*3) if m.body_count else None
        if c.capture:
            if c.cloth_solver=='style3d':
                self.solver.linear_solver=CapturedPCG(self.solver.linear_solver)
            self.solver=CapturedClothStep(self.solver)
            if self.rigid_solver is not None and c.coupling!='native':
                self.rigid_solver=CapturedRigidStep(self.rigid_solver)
            self.pipeline=CapturedCollision(self.pipeline)
        self.tick=0
        self.fault=None
        self.transport=None
        if c.soil:
            from .transport import Transport
            self.transport=Transport(self)
            if c.scene=='kitchen':
                rng=np.random.default_rng(c.seed)
                food,oil={'reference':(1.,1.),'oil-heavy':(.7,2.),'sticky-food':(2.,.5)}[c.soil_preset]
                jitter=rng.uniform(-.01,.01,(2,2))
                self.transport.add_spill('worktop',np.asarray((.311,.2896))+jitter[0],(.045,.065),(.008,.001*food,0.),rng)
                self.transport.add_spill('worktop',np.asarray((.45,.60))+jitter[1],(.05,.065),(.010,.0008*food,.001*oil),rng)
                self.transport.add_spill('backsplash',(.06,.56),(.02,.045),(.002,.0003*food,0.),rng)
            else:
                self.transport.add_patch('worktop',(.36,.37),(.045,.065),(.004,.001,0.))
                self.transport.add_patch('worktop',(.54,.70),(.04,.05),(.003,0.,.001))
                self.transport.add_patch('backsplash',(.06,.56),(.02,.045),(.001,.0003,0.))
            self.transport.freeze_initial()

    def update_wet_mass(self):
        mass=self.dry_mass+self.water
        self.model.particle_mass.assign(mass)
        self.model.particle_inv_mass.assign(1./mass)

    def step(self,count=1,*,fixture_forces=None,fixture_body_forces=None,callback=None):
        if self.fault:
            raise RuntimeError(self.fault)
        if not isinstance(count,int) or count<1:
            raise ValueError("Positive integer tick count required")
        c=self.config
        for _ in range(count):
            self.state_0.clear_forces()
            if fixture_forces is not None:
                # Explicit diagnostic fixture only. Never exposed by policy API.
                self.state_0.particle_f.assign(fixture_forces)
            if self.state_0.body_f is not None and c.coupling!='native':
                self.state_0.body_f.assign(self.solver.collision.wrenches)
            if fixture_body_forces is not None:
                if c.coupling!='native':raise ValueError('Rigid diagnostic fixture requires native coupling')
                # Explicit digital qualification aid, never a policy action.
                self.state_0.body_f.assign(fixture_body_forces)
            if self.control.joint_target_q is not None:
                current=self.control.joint_target_q.numpy()
                current+=np.clip(self.targets-current,-self.rates*c.dt,self.rates*c.dt)
                self.control.joint_target_q.assign(current)
            collision_state=self._rigid_collision_states[id(self.state_0)] if self._rigid_collision_states else self.state_0
            self.pipeline.collide(collision_state,self.contacts)
            if self.rigid_solver is not None and c.coupling!='native':
                self.rigid_solver.step(self.state_0,self.state_1,self.control,self.contacts,c.dt)
            self.solver.step(self.state_0,self.state_1,self.control,self.contacts,c.dt)
            if c.coupling=='native':self.rigid_solver._step=self.tick+1
            self.state_0,self.state_1=self.state_1,self.state_0
            self.tick+=1
            if self.transport is not None:
                try:self.transport.step()
                except Exception as exc:
                    self.fault=f'Material stepping failed at tick {self.tick}: {type(exc).__name__}: {exc}'
                    raise
            self.check_finite()
            if callback is not None:
                callback(self)
        self.check_finite()

    def check_finite(self):
        if self.config.cloth_solver=='vbd':
            cloth=self.solver.native if hasattr(self.solver,'graphs') else self.solver
            if self.config.coupling=='native':cloth.check_capacity()
            flags=cloth.native.trimesh_collision_detector.resize_flags.numpy()
            if np.any(flags):
                self.fault=f'Native cloth self-contact capacity overflow at tick {self.tick}: {flags.tolist()}'
                raise OverflowError(self.fault)
        n=int(self.contacts.soft_contact_count.numpy()[0])
        if n>self.contacts.soft_contact_max:
            self.fault=f'Native soft-contact capacity overflow at tick {self.tick}: {n}'
            raise OverflowError(self.fault)
        for key in ('particle_q','particle_qd','body_q','body_qd','joint_q','joint_qd'):
            a=getattr(self.state_0,key)
            value=None if a is None else a.numpy()
            if value is not None and not np.isfinite(value).all():
                self.fault=f"Non-finite {key} at tick {self.tick}; actions stopped"
                raise FloatingPointError(self.fault)
            if value is not None and key in ('body_q','particle_q','body_qd','particle_qd'):
                # A finite value can already be a catastrophic numerical state.
                # This stops and reports it; it never clamps, teleports or hides it.
                is_position=key.endswith('_q')
                magnitude=np.max(abs(value[:,:3]),axis=1) if is_position else np.linalg.norm(value[:,:3],axis=1)
                limit=10. if is_position else 50.
                if len(magnitude) and magnitude.max()>limit:
                    index=int(magnitude.argmax())
                    label=self.model.body_label[index] if key.startswith('body') else f'particle_{index}'
                    self.fault=f'Numerical workcell envelope exceeded: {label} {key}={value[index].tolist()} at tick {self.tick}; no clamp applied'
                    raise FloatingPointError(self.fault)

    def summary(self):
        p=self.state_0.particle_q.numpy()
        v=self.state_0.particle_qd.numpy()
        body_speed=np.linalg.norm(self.state_0.body_qd.numpy()[:,:3],axis=1)
        q=self.state_0.body_q.numpy();crumbs=self.assembly.objects.get('crumbs',[])
        return dict(tick=self.tick,time_s=self.tick*self.config.dt,particles=len(p),
            max_body_speed_m_s=float(body_speed.max()) if len(body_speed) else 0.,
            fastest_body=self.model.body_label[int(body_speed.argmax())] if len(body_speed) else None,
            crumb_bottom_min_m=float(min(q[a['body'],2]-a['radius'] for a in crumbs)) if crumbs else None,
            material=None if self.transport is None else self.transport.ledger(),
            cloth_bounds=[dict(name=a['name'],min=p[a['first']:a['first']+a['count']].min(0).tolist(),
                              max=p[a['first']:a['first']+a['count']].max(0).tolist()) for a in self.assembly.cloths],
            max_particle_speed_m_s=float(np.linalg.norm(v,axis=1).max()),
            dry_mass_kg=float(self.dry_mass.sum()),water_mass_kg=float(self.water.sum()),
            contact_count=int(self.contacts.soft_contact_count.numpy()[0]),
            max_cloth_contact_force_n=float(np.linalg.norm(self.solver.collision.forces.numpy(),axis=1).max()),
            fault=self.fault)

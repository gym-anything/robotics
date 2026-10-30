"""Metric load-bearing event return station, assembled only at episode creation."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
import newton
import warp as wp
from newton.solvers import SolverMuJoCo,SolverVBD
from .geometry import can_mesh,bag_mesh,area,validate_surface,CAN_SPECS
from .robot import add_robot

class Assembly:
    def __init__(self,c,*,robots=True,fixture=None,robot_setup=None):
        self.config=c;self.sources={};self.materials={};self.objects={};self.surfaces=[]
        self.robot_joints={};self.robot_bases={};self.robot_yaws={}
        from robots.deployment import deposit_setup
        self.robot_setup=deposit_setup(robot_setup)
        self.robot_selection=self.robot_setup.to_dict()['instances']
        self.plastic_yield=[];self.pins=[];self.geometry_checks={}
        newton.use_coord_layout_targets=True
        b=self.builder=newton.ModelBuilder(gravity=(0,0,-9.80665))
        SolverMuJoCo.register_custom_attributes(b);SolverVBD.register_custom_attributes(b)
        b.default_shape_cfg=self.shape_config()
        self.box('floor',(0,0,-.05),(3.,3.,.05),'floor')
        self.box('return_counter',(-.38,0,.875),(.32,.69,.025),'counter')
        for x in (-.64,-.12):
            for y in (-.61,.61):self.box(f'counter_leg_{x}_{y}',(x,y,.425),(.023,.023,.425),'steel')
        self.frame=self.frame_station((.49,0,0))
        self.bag()
        tub_pose=(fixture or {}).get('tub_pose')
        self.tub=self.make_tub(tub_pose['position_m'] if tub_pose else(-.37,0,.903),
            rotation_xyz=tub_pose.get('euler_xyz',[0.,0.,0.])if tub_pose else None)
        from .props import build as build_props
        from .room import build as build_room
        if (fixture is None or fixture.get('keep_room',False)) and robots:
            build_props(self);build_room(self)
        if robots:
            for name,y,yaw in (('left',-.65,math.pi/2),('right',.65,-math.pi/2)):
                self.box(name+'_pedestal',(.12,y,.492),(.12,.12,.492),'charcoal')
                self.box(name+'_mount',(.12,y,1.002),(.16,.16,.018),'steel')
                add_robot(self,name,(.12,y,1.02),yaw)
            from .cameras import install
            install(self)
        self.can_inventory(c,fixture)
        from .initial_geometry import audit as audit_initial_geometry
        audit_initial_geometry(self)
        self.plastic_yield=np.asarray(self.plastic_yield,np.float32)
        b.color()

    def shape_config(self,**options):
        values=dict(ke=self.config.contact_ke,kd=self.config.contact_kd,mu=self.config.friction,
            density=1000.,gap=self.config.contact_margin,margin=0.,mu_torsional=0.,mu_rolling=0.)
        values.update(options);return newton.ModelBuilder.ShapeConfig(**values)

    def box(self,name,center,half,material='steel',body=-1,cfg=None):
        shape=self.builder.add_shape_box(body,xform=wp.transform(center,wp.quat_identity()),hx=half[0],hy=half[1],hz=half[2],cfg=cfg or self.shape_config(),label=name)
        self.materials[shape]=material;return shape

    def tube(self,name,a,z,r,body,material='steel'):
        a,z=np.asarray(a),np.asarray(z);delta=z-a;length=np.linalg.norm(delta)
        quat=Rotation.align_vectors([delta/length],[[0,0,1]])[0].as_quat()
        shape=self.builder.add_shape_cylinder(body,xform=wp.transform((a+z)/2,wp.quat(*quat)),radius=r,half_height=length/2,cfg=self.shape_config(density=7850.),label=name)
        self.materials[shape]=material

    def frame_station(self,center):
        if self.config.scene_revision>=2:
            from .station import build
            return build(self,center)
        b=self.builder;body=b.add_body(xform=wp.transform(center,wp.quat_identity()),label='recycling_frame')
        for i,x in enumerate((-.3015,.3015)):
            self.tube(f'frame_{i}_cross',(x,-.2126,.992),(x,.2126,.992),.0047625,body)
            for j,y in enumerate((-.2126,.2126)):
                self.tube(f'frame_{i}_{j}',(x,y,.005),(x,y,.992),.0047625,body)
                self.box(f'foot_{i}_{j}',(x,y,.006),(.016,.016,.006),'rubber',body,cfg=self.shape_config(density=1100.,mu=.65))
        for y in (-.2126,.2126):self.tube('lower_brace'+str(y),(-.3015,y,.12),(.3015,y,.12),.0047625,body)
        # Published 8lb system /1.75lb lid: preserve both contributions to the
        # COM/inertia, not just total mass of an implausible solid plastic slab.
        self.set_mass(body,3.62874-.79378665)
        from wiping.room_fixtures import rounded_ring
        rings=[rounded_ring(hx,hy,r,z) for hx,hy,r,z in
            ((.3175,.2286,.025,1.016),(.2032,.05715,.015,1.016),
             (.2032,.05715,.015,1.013),(.3175,.2286,.025,1.013))]
        n=len(rings[0]);vertices=np.concatenate(rings);faces=[]
        for a,d in ((0,1),(1,2),(2,3),(3,0)):
            for i in range(n):
                j=(i+1)%n;faces.extend(((a*n+i,a*n+j,d*n+j),(a*n+i,d*n+j,d*n+i)))
        faces=np.asarray(faces,np.int32)
        volume=float(np.einsum('ij,ij->i',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6)
        if volume<=0:raise ValueError('Invalid lid winding')
        lid=newton.Mesh(vertices,faces.ravel());lid.build_sdf(device=self.config.device,max_resolution=512)
        shape=b.add_shape_mesh(body,mesh=lid,cfg=self.shape_config(density=.79378665/volume),label='rounded_slot_lid')
        self.materials[shape]='blue_plastic'
        self.objects['station']=dict(body=body,slot_m=[.4064,.1143],height_m=1.016,mass_kg=3.62874,
            lid_mass_kg=.79378665,lid_wall_m=.003,slot_corner_radius_m=.015,
            construction='Assembled locked lid/frame; wall and fillets authored estimates')
        return body

    def set_mass(self,body,mass):
        b=self.builder;old=b.body_mass[body]
        if old<=0:raise ValueError('Invalid density-derived inertia')
        b.body_inertia[body]=wp.mat33(*(np.asarray(b.body_inertia[body]).reshape(-1)*mass/old).tolist());b.body_mass[body]=float(mass)

    def make_tub(self,center,*,rotation_xyz=None):
        from wiping.formed_pan import geometry
        rotation=wp.quat_identity()if rotation_xyz is None else wp.quat(*Rotation.from_euler('xyz',rotation_xyz).as_quat())
        b=self.builder;body=b.add_body(xform=wp.transform(center,rotation),label='bussing_tub')
        if self.config.scene_revision>=4:
            from .tub_geometry import geometry as molded
            v,f=molded();f=f.ravel()
        else:v,f,_=geometry((.389,.506,.125))
        mesh=newton.Mesh(v,f)
        self.tub_surface=(np.asarray(v,dtype=np.float32),np.asarray(f,dtype=np.int32).reshape(-1,3))
        # Resolve the reconstructed1.7mm skin instead of using a2mm voxel.
        mesh.build_sdf(device=self.config.device,max_resolution=512 if self.config.scene_revision>=4 else 256)
        shape=b.add_shape_mesh(body,mesh=mesh,cfg=self.shape_config(density=950.),label='formed_bus_tub')
        self.materials[shape]='gray_plastic';self.set_mass(body,.70)
        self.objects['tub']=dict(body=body,mass_kg=.70,size_m=[.389,.506,.125],wall_m=.003)
        if self.config.scene_revision>=4:
            self.objects['tub']['wall_m']=.0017
            self.objects['tub']['construction']='Authored molded rim grips, return apron and bottom ribs; shared physical/optical surface. Feature dimensions are estimates.'
        return body

    def add_surface(self,name,v,f,thickness,E,nu,density,*,flat=False,plastic=False,kind='surface'):
        b=self.builder;c=self.config;self.geometry_checks[name]=validate_surface(v,f)
        p0,t0,e0=b.particle_count,b.tri_count,b.edge_count
        thick=np.broadcast_to(np.asarray(thickness),len(f));mu=E/(2*(1+nu));lam=E*nu/(1-nu*nu)
        b.add_cloth_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0,0,0),vertices=v.tolist(),indices=f.ravel().tolist(),density=density*float(thick.mean()),
            tri_ke=mu*float(thick.mean()),tri_ka=lam*float(thick.mean()),tri_kd=.001,edge_ke=0.,edge_kd=0.,particle_radius=c.contact_radius,validate_mesh=True,label=name)
        weights=area(v,f)*thick*density;mass=np.zeros(len(v))
        for j in range(3):np.add.at(mass,f[:,j],weights/3)
        b.particle_mass[p0:p0+len(v)]=mass.tolist()
        b.tri_materials[t0:t0+len(f)]=[(mu*t,lam*t,.001,0.,0.) for t in thick]
        # Finalize stores vec3<float32>. Its metric must describe those SAME
        # coordinates, not the pre-rounding Python vertices. Otherwise the
        # nominal rest can starts with up to10N of unintended vertex stress.
        all_positions=np.asarray(b.particle_q,dtype=np.float32).astype(np.float64)
        points=all_positions[p0+f];ab=points[:,1]-points[:,0];ac=points[:,2]-points[:,0]
        lengths=np.linalg.norm(ab,axis=1);along=np.einsum('ij,ij->i',ab,ac)/lengths
        height=np.linalg.norm(np.cross(ab,ac),axis=1)/lengths
        poses=np.zeros((len(f),2,2));poses[:,0,0]=1/lengths
        poses[:,0,1]=-along/(lengths*height);poses[:,1,1]=1/height
        b.tri_poses[t0:t0+len(f)]=poses.tolist()
        incident={frozenset((row+p0).tolist()):i for i,row in enumerate(f)}
        face_area=area(v,f)
        rigidity=E*thick**3/(12*(1-nu*nu))
        for edge in range(e0,b.edge_count):
            a,d,e,g=b.edge_indices[edge];yield_angle=0.
            if min(a,d)>=0:
                pa,pd,pe,pg=all_positions[[a,d,e,g]]
                L=np.linalg.norm(pe-pg)
                i=incident[frozenset((a,e,g))];j=incident[frozenset((d,e,g))]
                # Discrete-shell diamond energy, with the two material-side
                # compliances in series. Actual end thickness is not averaged
                # with the thin walls across the entire can.
                stiffness=3*L/(face_area[i]/rigidity[i]+face_area[j]/rigidity[j])
                b.edge_bending_properties[edge]=(stiffness,stiffness*.002)
                if flat:b.edge_rest_angle[edge]=0.
                if plastic:yield_angle=c.aluminum_yield*min(thick[i],thick[j])**2/(6*stiffness)
            self.plastic_yield.append(yield_angle)
        item=dict(name=name,first=p0,count=len(v),first_triangle=t0,triangles=f,rest_vertices=v.copy(),kind=kind,mass_kg=float(mass.sum()))
        self.surfaces.append(item);return item

    def bag(self):
        b=self.builder
        v,f,mouth,pins=bag_mesh(self.config.bag_segments,self.config.bag_rows,
            cuff=self.config.scene_revision>=3,star_bottom=self.config.scene_revision>=5)
        v+=np.array([.49,0,0]);c=self.config
        item=self.add_surface('collection_bag',v,f,c.film_thickness,c.film_young,c.film_poisson,c.film_density,flat=True,kind='bag')
        item['bottom_construction']=('Regularized opened star-seal:6mm gather and radial pleats, physical shell; no individual welded plies or puncture.'
            if c.scene_revision>=5 else 'Planar-bottom development approximation')
        item['mouth']=mouth;self.bag_spec=item
        if c.scene_revision>=3:
            # Containment closes the inner rim, not the lower outside cuff.
            n=c.bag_segments;cut=(c.bag_rows+3)*n;bottom_start=(c.bag_rows+8)*n
            item['physical_boundary']=mouth.copy()
            item['mouth']=np.arange((c.bag_rows+2)*n,cut,dtype=np.int32)
            item['containment_triangles']=f[np.all((f<cut)|(f>=bottom_start),axis=1)]
        self.pins=[(int(item['first']+i),self.frame,(v[i]-np.array([.49,0,0])).tolist()) for i in pins]
        self.anchor_ids=[]
        for particle,body,local in self.pins:
            anchor=b.particle_count
            b.add_particle(pos=wp.vec3(*b.particle_q[particle]),vel=wp.vec3(0,0,0),mass=0.,radius=0.)
            b.add_spring(particle,anchor,ke=1/c.retention_compliance,kd=0.,control=0.)
            self.anchor_ids.append(anchor)
        # Anchors follow the finite-mass frame, not the world. The solver applies
        # the equal-and-opposite assembly spring force to that frame each step.

    def can_inventory(self,c,fixture):
        rng=np.random.default_rng(c.seed)
        count=c.target_count+c.preload_count
        from .variants import preset
        # high_fill starts with its pre-existing cans lying in the bag (bag_pile.py), not as
        # the floating upright grid below, which the reference preset keeps.
        lying_pile=preset(c)=='high_fill' and c.scene_revision>=7
        # lid_cans: the targets stand upright on the station lid's flat roof, 5 mm above it (the counter cans' clearance).
        on_lid=preset(c)=='lid_cans'
        for i in range(count):
            kind=i%3;diameter,h,_=CAN_SPECS[kind]
            v,f,t=can_mesh(kind,c.can_segments,.0015 if i%7==3 else 0.,quality_end=c.scene_revision>=2)
            label_uv=np.stack((np.arctan2(v[:,1],v[:,0])/(2*np.pi)+.5,(v[:,2]+h/2)/h),axis=1)[f]
            seam=np.ptp(label_uv[:,:,0],axis=1)>.5
            label_uv[seam,:,0]+=label_uv[seam,:,0]<.5
            # Follow complete circumferential rings; a centroid threshold
            # incorrectly split each quad into silver/painted sawteeth.
            label_faces=np.flatnonzero((v[f,2].min(1)>=-h/2+.010-1e-9)&(v[f,2].max(1)<=h/2-.008+1e-9))
            if i<c.tub_count:
                # Two horizontal layers. Initial spacings exceed physical can diameter.
                j=i%8;layer=i//8
                half_spacing=.091 if c.scene_revision>=3 else .092
                pos=np.array([-.37+(-half_spacing if j%2==0 else half_spacing),-.12+.080*(j//2),.910+diameter/2+layer*.070])
                rot=Rotation.from_euler('y',np.pi/2)
                if c.scene_revision>=6:
                    from .tub_layout import pose
                    pos,rot=pose(i,c.can_segments,normal_clearance=c.scene_revision>=7)
                origin='tub'
            elif i<c.target_count and on_lid:
                from .variants import STATION_XY,LID_ROOF_Z,LID_CANS_XY
                x,y=LID_CANS_XY[i-c.tub_count]
                pos=np.array([STATION_XY[0]+x,STATION_XY[1]+y,LID_ROOF_Z+.002+h/2+.003])
                rot=Rotation.from_euler('z',rng.uniform(-np.pi,np.pi));origin='lid'
            elif i<c.target_count:
                j=i-c.tub_count
                pos=np.array([-.60+.15*(j%4),(-.53 if j<4 else .53),.902+h/2+.003])
                rot=Rotation.from_euler('z',rng.uniform(-np.pi,np.pi));origin='counter'
            else:
                j=i-c.target_count;layer=j//24;k=j%24
                # Centre the initial rows; the old leftmost four cans pierced
                # the tapered bag by up to3.69mm before any integration.
                pos=np.array([.49+(-.225+.090*(k%6) if c.scene_revision>=3 else -.235+.092*(k%6)),-.13+.085*(k//6),.030+h/2+layer*.175])
                yaw=rng.uniform(-np.pi,np.pi);rot=Rotation.from_euler('z',yaw);origin='preload'
                if lying_pile:
                    # Same seeded draw, used as the spin of the lying can about its own axis.
                    from .bag_pile import pose as pile_pose
                    pos,rot=pile_pose(i,yaw)
            if fixture is not None and i==0:
                # A tub/room fixture is not an implicit single-can pose override.
                # In particular, replacing the horizontal tub-can orientation
                # with identity at its old centre starts it THROUGH the floor.
                if 'can_position' in fixture:pos=np.array(fixture['can_position'])
                if 'can_euler' in fixture:rot=Rotation.from_euler('xyz',fixture['can_euler'])
            v=rot.apply(v)+pos
            if fixture is not None and origin=='tub' and fixture.get('tub_pose'):
                # One labelled INITIAL rigid placement of tub and contents,
                # before solver/model state creation. Never a runtime action.
                pose=fixture['tub_pose'];R=Rotation.from_euler('xyz',pose.get('euler_xyz',[0.,0.,0.]))
                base=np.asarray(pose['position_m']);old=np.array([-.37,0.,.903])
                v=R.apply(v-old)+base;pos=R.apply(pos-old)+base
            item=self.add_surface(f'can_{i:03d}',v,f,t,c.aluminum_young,c.aluminum_poisson,2700.,plastic=True,kind='can')
            item.update(target=i<c.target_count,origin=origin,can_type=kind,initial_center_m=pos.tolist(),label_uv=label_uv,label_faces=label_faces)

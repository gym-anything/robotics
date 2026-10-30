"""Physical kitchen inventory. No task object is just a rendered decoration."""
import math
import numpy as np
import warp as wp
import newton


def jug_geometry(*,hollow=False):
    """Same12-sided tapered external envelope for collision and optical shell.

    The fixed lid seals this off-task empty vessel; the rigid collider encloses
    its inaccessible interior. Optical3mm walls expose no traversable volume.
    """
    n=12;theta=np.arange(n)*2*np.pi/n
    layers=[(.051,.228),(.080,.444)]
    if hollow:layers += [(.077,.444),(.048,.231)]
    vertices=np.array([(r*np.cos(t),r*np.sin(t),z) for r,z in layers for t in theta],np.float32)
    faces=[]
    for ring in range(len(layers)-1):
        for i in range(n):
            j=(i+1)%n;a=ring*n+i;b=ring*n+j;c=(ring+1)*n+j;d=(ring+1)*n+i
            faces.extend(((a,b,c),(a,c,d)))
    # Bottom closures: outer bottom reversed, inner bottom upward.
    for i in range(1,n-1):faces.append((0,i+1,i))
    last=(len(layers)-1)*n
    for i in range(1,n-1):faces.append((last,last+i,last+i+1))
    return vertices,np.array(faces,np.int32).ravel()


def set_mass(builder,body,mass):
    previous=float(builder.body_mass[body])
    if previous<=0:raise ValueError('Shape-derived inertia required for a free object')
    builder.body_inertia[body]=builder.body_inertia[body]*(mass/previous)
    builder.body_mass[body]=mass
    builder.body_inv_mass[body]=1./mass
    builder.body_inv_inertia[body]=wp.inverse(builder.body_inertia[body])


def pan(a,name,center,size,mass,material='pan_white'):
    """Open shell with contact-bearing molded flanges, not a closed collider.

    Manufacturer outer dimensions are preserved. The4mm flange lies inward,
    with a2mm section; all these faces enter the same material ledger.
    """
    if a.config.scene_revision>=3 and name in ('prep_tray','clean_staging_tray'):
        from .formed_pan import build
        return build(a,name,center,size,mass,material)
    b=a.builder;x,y,h=size;t=.002
    body=b.add_body(xform=wp.transform(center,wp.quat_identity()),label=name)
    bottom=a.box(name+'_bottom',(0,0,t/2),(x/2,y/2,t/2),material,body)
    for sign in (-1,1):
        a.box(name+f'_x{sign}',(sign*(x-t)/2,0,h/2),(t/2,y/2,h/2),material,body)
        a.box(name+f'_y{sign}',(0,sign*(y-t)/2,h/2),((x-2*t)/2,t/2,h/2),material,body)
        if a.config.scene_revision>=1:
            lip=.004
            a.box(name+f'_lip_x{sign}',(sign*(x-lip)/2,0,h-t/2),(lip/2,y/2,t/2),material,body)
            a.box(name+f'_lip_y{sign}',(0,sign*(y-lip)/2,h-t/2),((x-2*lip)/2,lip/2,t/2),material,body)
    set_mass(b,body,mass)
    a.objects[name]=dict(body=body,bottom_shape=bottom,size=size,wall=t,mass=mass)
    return body


def bin_box(a,name,center,size,material):
    x,y,h=size;t=.008;z=center[2]
    base=a.box(name+'_bottom',(center[0],center[1],z+t/2),(x/2,y/2,t/2),material)
    for sign in (-1,1):
        a.box(name+f'_x{sign}',(center[0]+sign*(x-t)/2,center[1],z+h/2),(t/2,y/2,h/2),material)
        a.box(name+f'_y{sign}',(center[0],center[1]+sign*(y-t)/2,z+h/2),((x-2*t)/2,t/2,h/2),material)
    a.objects[name]=dict(body=-1,bottom_shape=base,center=center,size=size,wall=t,
                         fixed=True,note='floor-mounted research collection fixture')


def scraper(a,center):
    b=a.builder
    body=b.add_body(xform=wp.transform(center,wp.quat_identity()),label='scraper')
    # Two contact-bearing parts retain a wide 3.3 mm blade and thicker handle.
    blade=a.box('scraper_blade',(-.045,0,.00165),(.06,.051,.00165),'scraper_blue',body)
    a.box('scraper_handle',(.06,0,.009),(.045,.020,.009),'scraper_blue',body)
    set_mass(b,body,.060)
    a.objects['scraper']=dict(body=body,blade_shape=blade,mass=.060,
        dimensions=(.210,.102,.018),blade_thickness=.0033,
        geometry='two-part rigid approximation; manufacturer outer dimensions')


def blender(a):
    b=a.builder;z=a.config.table_height
    body=b.add_body(xform=wp.transform((.25,.40,z+(0. if a.config.scene_revision>=4 else .001)),wp.quat_identity()),label='blender')
    if a.config.scene_revision>=1:
        a.box('blender_motor',(0,0,.105),(.110998,.095504,.095),'charcoal',body)
        for x in (-.087,.087):
            for y in (-.072,.072):
                a.box(f'blender_foot_{x}_{y}',(x,y,.006),(.013,.013,.006),'charcoal',body)
        # Physical drive pedestal bridges the former28mm visual air gap.
        drive=b.add_shape_cylinder(body,xform=wp.transform((0,0,.216),wp.quat_identity()),
            radius=.065,half_height=.02,cfg=a.shape_config(density=600.,mu=.5),label='blender_drive')
        a.materials[drive]='charcoal'
    else:a.box('blender_motor',(0,0,.10),(.110998,.095504,.10),'charcoal',body)
    # The closed, empty jug is not part of the wiping target; its collision
    # envelope remains present so the robot cannot pass through it.
    if a.config.scene_revision>=1:
        v,f=jug_geometry();cfg=a.shape_config(density=350.,mu=.5)
        mesh=newton.Mesh(v,f)
        if not a.render_only:mesh.build_sdf(device=a.config.device,max_resolution=128)
        shape=b.add_shape_mesh(body,mesh=mesh,cfg=cfg,label='blender_jug')
    else:
        shape=b.add_shape_cylinder(body,xform=wp.transform((0,0,.336),wp.quat_identity()),
            radius=.080,half_height=.108,cfg=a.shape_config(density=350.,mu=.5),label='blender_jug')
    a.materials[shape]='jug_frosted'
    a.box('blender_lid',(0,0,.464),(.086,.084,.020),'charcoal',body)
    a.box('blender_lid_plug',(0,0,.496),(.025,.025,.012),'charcoal',body)
    if a.config.scene_revision>=1:
        # Closed handle loop has a real graspable opening, not an opaque block.
        a.box('blender_handle_spine',(0,-.126,.336),(.012,.009,.083),'charcoal',body)
        for zlink in (.261,.411):
            a.box(f'blender_handle_link_{zlink}',(0,-.102,zlink),(.012,.033,.009),'charcoal',body)
    else:a.box('blender_handle',(0,-.104,.336),(.022,.015,.085),'charcoal',body)
    set_mass(b,body,5.533828)
    a.objects['blender']=dict(body=body,mass=5.533828,footprint=(.221996,.191008),
        center=(.25,.40,z),height=.508,task_exclusion='permanent appliance footprint only',powered=False)


def crumbs(a):
    b=a.builder;rng=np.random.default_rng(a.config.seed)
    a.objects['crumbs']=[]
    for i in range(18):
        radius=float(rng.uniform(.0025,.004))
        for attempt in range(100):
            position=(float(rng.uniform(-.20,.20)),float(rng.uniform(-.14,.20)),a.config.table_height+radius+(0. if a.config.scene_revision>=4 else .0005))
            if all(np.linalg.norm(np.asarray(position[:2])-np.asarray(old['initial_position'][:2]))>
                   radius+old['radius']+.001 for old in a.objects['crumbs']):break
        else:raise ValueError('Cannot place separated crumbs for this seed')
        body=b.add_body(xform=wp.transform(position,wp.quat_identity()),label=f'crumb_{i:02d}')
        shape=b.add_shape_sphere(body,radius=radius,cfg=a.shape_config(density=450.,mu=.60),label=f'crumb_{i:02d}')
        b.shape_gap[shape]=a.config.grain_contact_gap
        a.materials[shape]='crumb'
        a.objects['crumbs'].append(dict(body=body,shape=shape,radius=radius,mass=float(b.body_mass[body]),initial_position=position))


def build(a):
    """Authored commercial prep layout; qualification records audit reach/contact."""
    z=a.config.table_height
    supported=a.config.scene_revision>=4
    # Lower clean-tool trolley adjacent to, but not intersecting, the workbench.
    a.surface_shapes['tool_trolley']=a.box('tool_trolley_top',(-.65,.05,.825),(.25,.69,.015))
    for x in (-.85,-.45):
        for y in (-.58,.68):
            a.box(f'trolley_leg_{x}_{y}',(x,y,.41),(.018,.018,.40))
    a.box('trolley_shelf',(-.65,.05,.22),(.24,.68,.012))
    for i,(x,y) in enumerate(((-.51,-.27),(-.51,.23),(-.765,.52))):
        pan(a,f'clean_cloth_tray_{i}',(x,y,.840 if supported else .841),(.22,.44,.022),.18,'pan_gray')
    pan(a,'prep_tray',(-.07,-.32,z+(0. if supported else .001)),(.265,.325,.065),.42)
    pan(a,'clean_staging_tray',(-.10,.38,z+(0. if supported else .001)),(.176,.325,.065),.30,'pan_green')
    pan(a,'collection_pan',(-.75,-.37,.840 if supported else .842),(.25,.22,.035),.20,'pan_gray')
    scraper(a,(-.74,-.585,.840 if supported else .844))
    blender(a)
    bin_box(a,'used_cloth_bin',(-.55,-1.0,0),(.30,.28,.55),'bin_blue')
    bin_box(a,'waste_bin',(-.55,1.02,0),(.30,.28,.55),'bin_gray')
    crumbs(a)
    # Room shell: behind the bench, leaving the operational front open.
    a.box('back_wall',(1.45,0,1.50),(.075,2.4,1.5),'tile_wall')
    a.box('side_wall',(0,2.40,1.50),(1.5,.075,1.5),'tile_wall')
    a.box('service_counter',(1.13,-.55,.44),(.24,.90,.44),'steel')
    a.box('service_counter_top',(1.13,-.55,.91),(.27,.93,.025),'steel')
    if a.config.scene_revision>=2:
        from .room_fixtures import build as fixtures
        fixtures(a)
    # Revision4 starts resting objects on their actual support. The folded
    # cloth bottom is its center surface minus the physical pile radius; its
    # holder's2mm floor rests on the trolley's840mm upper surface. Historical
    # revisions preserve their0.5..5.5mm drop clearances. No runtime pose writes.
    cloth_z=.840+.002+a.config.cloth_radius if supported else .850
    return [(-.51,-.27,cloth_z),(-.51,.23,cloth_z),(-.765,.52,cloth_z)]

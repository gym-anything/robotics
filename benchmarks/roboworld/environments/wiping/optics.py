"""Visualize physical kg/area and cloth load, independent of task evaluation."""
from pathlib import Path
import numpy as np


def vertex_material(stage,material,*,transparent=False,textile=False,film=False):
    from pxr import Gf,Sdf,UsdShade
    path=str(material.GetPath())
    shader=UsdShade.Shader.Get(stage,path+'/surface')
    color=UsdShade.Shader.Define(stage,path+'/color')
    color.CreateIdAttr('UsdPrimvarReader_float3')
    color.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
    color.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(color.ConnectableAPI(),'result')
    if film:
        for field in ('roughness','metallic'):
            reader=UsdShade.Shader.Define(stage,path+'/'+field)
            reader.CreateIdAttr('UsdPrimvarReader_float')
            reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set(field)
            reader.CreateOutput('result',Sdf.ValueTypeNames.Float)
            shader.GetInput(field).ConnectToSource(reader.ConnectableAPI(),'result')
    if transparent:
        opacity=UsdShade.Shader.Define(stage,path+'/alpha');opacity.CreateIdAttr('UsdPrimvarReader_float')
        opacity.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayOpacity')
        opacity.CreateOutput('result',Sdf.ValueTypeNames.Float)
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(opacity.ConnectableAPI(),'result')
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.333)
    if textile:
        st=UsdShade.Shader.Define(stage,path+'/uv');st.CreateIdAttr('UsdPrimvarReader_float2')
        st.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        st.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        transform=UsdShade.Shader.Define(stage,path+'/repeat');transform.CreateIdAttr('UsdTransform2d')
        transform.CreateInput('in',Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(),'result')
        transform.CreateInput('scale',Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(16.,16.))
        transform.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        normal=UsdShade.Shader.Define(stage,path+'/normal');normal.CreateIdAttr('UsdUVTexture')
        normal.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(
            Path(__file__).resolve().parents[1]/'laundry/assets/woven_cotton_normal.png')))
        normal.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
        normal.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(),'result')
        normal.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
        normal.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
        normal.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.,2.,2.,1.))
        normal.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.,-1.,-1.,0.))
        normal.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
        shader.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(),'rgb')


def cloth_color(sim,cloth,dye):
    first,count=cloth['first'],cloth['count'];ids=slice(first,first+count)
    if sim.transport is None:
        load=np.zeros((count,3));load[:,0]=sim.water[ids]
    else:load=sim.transport.cloth[ids]
    dry=sim.dry_mass[ids].astype(float)
    damp=np.minimum(load[:,0]/dry,2.)/2
    soil=1-np.exp(-(load[:,1]+load[:,2])/(dry*.08))
    clean=np.tile(np.asarray(dye),(count,1))*(1-.22*damp[:,None])
    dirty=np.tile((.12,.041,.008),(count,1))
    return (clean*(1-soil[:,None])+dirty*soil[:,None]).astype(np.float32)


def film_arrays(transport,index):
    import os
    layered=os.environ.get('WIPE_LAYERED_COAT')=='1'
    species_optics=os.environ.get('WIPE_SPECIES_OPTICS')=='1' or layered
    s=transport.surfaces[index]
    mass=transport.surface[transport.offsets[index]:transport.offsets[index+1]]
    if s.triangle_uv is not None:
        from .render import COLORS
        rgb,base_rough,base_metal=COLORS[s.material]
        visible=1-np.exp(-(mass[0,1]/s.cell_area/.0015+mass[0,2]/s.cell_area/.025))
        water=min(1.,mass[0,0]/(s.cell_area*.15))*.20;alpha=max(visible,water)
        height=float((mass[0]/np.array((1000.,1250.,920.))).sum()/s.cell_area)
        uv=s.triangle_uv
        points=s.origin+uv[:,0,None]*s.u+uv[:,1,None]*s.v+(height+.00008)*s.normal
        if mass.sum()<1e-12:points[:]=points[:1]
        result=dict(points=points.astype(np.float32),normals=np.tile(s.normal,(3,1)).astype(np.float32),
            triangles=np.array(((0,1,2),),np.int32),
            color=np.tile(np.asarray(rgb)*(1-visible)+np.asarray((.20,.047,.008))*visible,(3,1)).astype(np.float32),
            roughness=np.full(3,base_rough*(1-alpha)+(.12+.18*visible)*alpha,np.float32),
            metallic=np.full(3,base_metal*(1-visible),np.float32))
        if species_optics:
            from .species_optics import properties
            color,rough,metal=properties(mass[0],s.cell_area,COLORS[s.material],layered=layered)
            result.update(color=np.tile(color,(3,1)).astype(np.float32),
                roughness=np.full(3,rough,np.float32),metallic=np.full(3,metal,np.float32))
        if layered:
            from .species_optics import coating
            result.update(coat_weight=np.full(3,coating(mass[0],s.cell_area),np.float32))
        if os.environ.get('WIPE_RESIDUE_DETAIL')=='1':
            result['food_coverage']=np.full(3,-np.expm1(-mass[0,1]/s.cell_area/.015),np.float32)
        return result
    x,y=np.meshgrid(np.arange(s.nx)*s.du,np.arange(s.ny)*s.dv,indexing='ij')
    uv=np.stack((x,y),axis=-1).reshape(-1,2)
    # Shared corner reconstruction removes visible 5 mm stair steps. It changes
    # only optical interpolation, never the cell masses used by physics/scoring.
    cells=mass.reshape(s.nx,s.ny,3)
    padded=np.pad(cells,((1,1),(1,1),(0,0)))
    vertices=(padded[:-1,:-1]+padded[1:,:-1]+padded[:-1,1:]+padded[1:,1:])*.25
    corner_mass=np.stack((vertices[:-1,:-1],vertices[1:,:-1],vertices[1:,1:],vertices[:-1,1:]),axis=2).reshape(-1,4,3)
    # Normals must follow the SAME mass-derived height field as the vertices.
    # Constant substrate normals hid film relief and gave flat, decal-like
    # highlights. No invented displacement, color noise or mass change.
    vertex_height=(vertices/np.asarray((1000.,1250.,920.))).sum(2)/s.cell_area
    slope_u,slope_v=np.gradient(vertex_height,s.du,s.dv)
    vertex_normal=s.normal-slope_u[:,:,None]*s.u-slope_v[:,:,None]*s.v
    vertex_normal/=np.linalg.norm(vertex_normal,axis=2)[:,:,None]
    corner_normal=np.stack((vertex_normal[:-1,:-1],vertex_normal[1:,:-1],
        vertex_normal[1:,1:],vertex_normal[:-1,1:]),axis=2).reshape(-1,4,3)
    # Explicit render candidate: stable per-surface capacity avoids rebuilding
    # changing-size GPU geometry buffers for every recorded state. Empty-cell
    # triangles remain degenerate below. Optical interpolation/mass is identical.
    import os
    occupied=np.flatnonzero(corner_mass.sum((1,2))>=1e-12)
    fixed=getattr(transport,'recorded_film_union',None)
    if fixed is not None:
        active=fixed[index]
        if not np.isin(occupied,active).all():raise ValueError('Recorded film exceeds exact replay allocation')
    else:
        active=(np.arange(len(corner_mass)) if os.environ.get('WIPE_STABLE_FILM_TOPOLOGY')=='1' else occupied)
    if not len(active):active=np.asarray([0])
    uv=uv[active];mass=corner_mass[active]
    corners=uv[:,None,:]+np.asarray(((0,0),(s.du,0),(s.du,s.dv),(0,s.dv)))[None,:,:]
    height=(mass/np.asarray((1000.,1250.,920.))).sum(2)/s.cell_area
    points=s.origin+corners[:,:,0,None]*s.u+corners[:,:,1,None]*s.v+(height[:,:,None]+.00008)*s.normal
    food=mass[:,:,1]/s.cell_area;oil=mass[:,:,2]/s.cell_area
    visible=1-np.exp(-(food/.0015+oil/.025))
    water=np.minimum(1,mass[:,:,0]/(s.cell_area*.15))*.20
    # Effective thin coating atop the substrate, not an opaque puddle or a
    # free-standing transmissive sheet. The substrate remains visible through
    # low optical-density residue. Water primarily changes surface roughness.
    alpha=np.maximum(visible,water)
    from .render import COLORS
    rgb,base_rough,base_metal=COLORS[s.material]
    color=np.asarray(rgb)*(1-visible[:,:,None])+np.asarray((.20,.047,.008))*visible[:,:,None]
    rough=base_rough*(1-alpha)+(.12+.18*visible)*alpha
    metallic=base_metal*(1-visible)
    if species_optics:
        from .species_optics import properties
        color,rough,metallic=properties(mass,s.cell_area,COLORS[s.material],layered=layered)
    # Empty cells must not form a transparent full-worktop optical sheet.
    # Degenerate only their render triangles; retained physical mass is unchanged.
    empty=mass.sum((1,2))<1e-12
    points[empty]=points[empty,:1,:]
    p=points.reshape(-1,3).astype(np.float32)
    ids=np.arange(len(mass)*4).reshape(-1,4)
    triangles=np.concatenate((ids[:,[0,1,2]],ids[:,[0,2,3]]))
    result=dict(points=p,normals=corner_normal[active].reshape(-1,3).astype(np.float32),triangles=triangles,
                color=color.reshape(-1,3).astype(np.float32),
                roughness=rough.ravel().astype(np.float32),metallic=metallic.ravel().astype(np.float32))
    if layered:
        from .species_optics import coating
        result.update(coat_weight=coating(mass,s.cell_area).ravel().astype(np.float32))
    if os.environ.get('WIPE_RESIDUE_DETAIL')=='1':
        result['food_coverage']=(-np.expm1(-mass[:,:,1]/s.cell_area/.015)).ravel().astype(np.float32)
    return result


def allocate_recorded_film_union(transport,surface_frames):
    """Fixed replay buffers covering EVERY positive recorded cell and its halo.

    Shared corner interpolation reaches at most one neighboring cell in each
    direction. The nine-cell halo therefore includes every drawable triangle
    for every saved state. Never uses a selected/keyframe subset. Empty allocated
    cells remain degenerate in film_arrays; mass and occupied optics are unchanged.
    This is offline allocation only, not predictive state provided to a policy.
    """
    import hashlib
    transport.recorded_film_union={};receipt=[]
    for i,s in enumerate(transport.surfaces):
        if s.triangle_uv is not None:continue
        first,last=transport.offsets[i:i+2]
        used=np.any(surface_frames[:,first:last,:]>0,axis=(0,2)).reshape(s.nx,s.ny)
        padded=np.pad(used,1);halo=np.zeros_like(used)
        for du in range(3):
            for dv in range(3):halo|=padded[du:du+s.nx,dv:dv+s.ny]
        active=np.flatnonzero(halo.ravel())
        if not len(active):active=np.asarray([0])
        transport.recorded_film_union[i]=active
        receipt.append(dict(surface=s.name,allocated_cells=len(active),all_cells=int(s.nx*s.ny),
            index_sha256=hashlib.sha256(active.astype('<i8').tobytes()).hexdigest()))
    return dict(source_frames=len(surface_frames),all_frames_used=True,surfaces=receipt,
        scope='Render allocation only: no material state, samples, resolution or physical steps changed.')


def runoff_arrays(sim):
    from scipy.spatial.transform import Rotation
    flow=sim.transport.flow
    points=flow.impact_local.copy()
    dynamic=flow.impact_body>=0
    if dynamic.any():
        poses=sim.state_0.body_q.numpy()[flow.impact_body[dynamic]]
        points[dynamic]=poses[:,:3]+Rotation.from_quat(poses[:,3:]).apply(points[dynamic])
    points=np.concatenate((flow.positions,points))
    mass=np.concatenate((flow.mass,flow.impact_mass))
    # Optical culling only; the exact mass remains in the transport ledger.
    active=mass.sum(1)>=1e-10;points=points[active];mass=mass[active]
    volume=(mass/np.asarray((1000.,1250.,920.))).sum(1)
    widths=2*np.cbrt(3*volume/(4*np.pi))
    fraction=mass[:,1:].sum(1)/np.maximum(mass.sum(1),1e-30)
    color=np.asarray((.35,.42,.44))*(1-fraction[:,None])+np.asarray((.18,.04,.005))*fraction[:,None]
    if not len(points):points=np.zeros((1,3));widths=np.zeros(1);color=np.zeros((1,3))
    return dict(points=points.astype(np.float32),widths=widths.astype(np.float32),color=color.astype(np.float32))

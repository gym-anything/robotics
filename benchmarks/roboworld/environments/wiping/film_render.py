"""Render-only film batching with exact visible triangle attributes.

Group triangles by their existing rigid parent, not physical reservoirs.
Fixed256-triangle pages avoid reallocating changing topology and thousands of
empty surface prims. No future state/recording is needed for live observations.
The model, film discretization, mass, normals and visible geometry are unchanged.
"""
import os
import numpy as np

FIELDS=('points','normals','color','roughness','metallic')
if os.environ.get('WIPE_LAYERED_COAT')=='1':FIELDS+=('coat_weight',)
if os.environ.get('WIPE_RESIDUE_DETAIL')=='1':FIELDS+=('food_coverage',)
PAGE_TRIANGLES=256
PAGE_VERTICES=PAGE_TRIANGLES*3


def groups(transport):
    from .optics import film_arrays
    collected={}
    for i,s in enumerate(transport.surfaces):
        if not np.any(transport.surface[transport.offsets[i]:transport.offsets[i+1]]):continue
        data=film_arrays(transport,i);tri=data['triangles'];p=data['points']
        area=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
        ids=tri[np.any(area!=0,axis=1)].ravel()
        if not len(ids):continue
        group=collected.setdefault(str(s.body),{k:[] for k in FIELDS})
        for key in FIELDS:group[key].append(data[key][ids])
    return {body:{key:np.concatenate(value) for key,value in data.items()} for body,data in collected.items()}


def pages(data,minimum_pages=0):
    n=len(data['points'])
    if n%3 or set(data)!=set(FIELDS) or any(len(v)!=n for v in data.values()):raise ValueError('Invalid film triangle fields')
    number=max(minimum_pages,(n+PAGE_VERTICES-1)//PAGE_VERTICES)
    for i in range(number):
        start=i*PAGE_VERTICES;count=max(0,min(n-start,PAGE_VERTICES))
        page={key:np.zeros((PAGE_VERTICES,*np.asarray(value).shape[1:]),np.float32) for key,value in data.items()}
        for key in FIELDS:page[key][:count]=data[key][start:start+count]
        yield page


def empty():
    return {key:np.zeros((0,3) if key in ('points','normals','color') else (0,),np.float32) for key in FIELDS}


def prepare(stage,grouped):
    from pxr import UsdGeom,UsdShade,Sdf,Vt
    material=UsdShade.Material(stage.GetPrimAtPath('/Materials/film'))
    if not material:raise ValueError('Missing film material')
    # Reuse emptied pages with degenerate triangles, never leave stale film.
    existing={}
    for prim in stage.GetPrimAtPath('/Lab').GetChildren():
        if prim.GetName()=='film_pages':existing['-1']=prim
        elif prim.GetName().startswith('body_'):
            child=prim.GetChild('film_pages')
            if child:existing[prim.GetName()[5:]]=child
    updates=[]
    for body in sorted(set(grouped)|set(existing),key=int):
        if str(int(body))!=body or int(body)<-1:raise ValueError('Invalid film parent')
        parent='/Lab' if body=='-1' else '/Lab/body_'+body
        if not stage.GetPrimAtPath(parent):raise ValueError('Missing film parent')
        root=UsdGeom.Xform.Define(stage,parent+'/film_pages').GetPrim()
        minimum=len(root.GetChildren());data=grouped.get(body,empty())
        for i,page in enumerate(pages(data,minimum)):
            path=f'{root.GetPath()}/page_{i:06d}';g=UsdGeom.Mesh.Get(stage,path)
            if not g:
                g=UsdGeom.Mesh.Define(stage,path);g.CreateSubdivisionSchemeAttr('none')
                g.CreatePointsAttr();g.CreateNormalsAttr();g.SetNormalsInterpolation('vertex');g.CreateExtentAttr()
                g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(PAGE_TRIANGLES,3,np.int32)))
                g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(PAGE_VERTICES,dtype=np.int32)))
                g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
                for key in FIELDS[3:]:
                    UsdGeom.PrimvarsAPI(g).CreatePrimvar(key,Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex)
                UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
            for key,value in page.items():
                name=('primvars:displayColor' if key=='color' else
                      'primvars:'+key if key in FIELDS[3:] else key)
                typed=Vt.Vec3fArray.FromNumpy(value) if value.ndim==2 else Vt.FloatArray.FromNumpy(value)
                updates.append((stage.GetRootLayer().GetAttributeAtPath(path+'.'+name),typed))
            points=page['points'];extent=np.asarray([points.min(0),points.max(0)],np.float32)
            updates.append((stage.GetRootLayer().GetAttributeAtPath(path+'.extent'),Vt.Vec3fArray.FromNumpy(extent)))
    return updates


def author(stage,grouped):
    from pxr import Sdf
    updates=prepare(stage,grouped)
    with Sdf.ChangeBlock():
        for spec,value in updates:spec.default=value

"""Fixed-size render-only point pages; no change to physical runoff packets.

Hydra may synchronize point geometry and primvars separately even after a
coalesced Sdf notice. Growing an existing Points prim produced insufficient
width/color warnings. Each page therefore always has256 points; unused slots
have zero width. New pages have all attributes before their first render.
There is no total-page limit, particle merge, resampling or physical mutation.
"""
import numpy as np

PAGE_SIZE=256


def pages(data,minimum_pages=1):
    n=len(data['points'])
    if len(data['widths'])!=n or len(data['color'])!=n:raise ValueError('Runoff array sizes differ')
    for start in range(0,max(PAGE_SIZE*minimum_pages,((n+PAGE_SIZE-1)//PAGE_SIZE)*PAGE_SIZE),PAGE_SIZE):
        count=max(0,min(n-start,PAGE_SIZE))
        item={k:np.zeros((PAGE_SIZE,*np.asarray(v).shape[1:]),np.float32) for k,v in data.items()}
        for key in item:item[key][:count]=data[key][start:start+count]
        yield item


def prepare(stage,data):
    """Return complete attribute writes, creating only previously unseen pages."""
    from pxr import UsdGeom,UsdShade,Vt
    root=UsdGeom.Xform.Define(stage,'/Lab/runoff').GetPrim()
    minimum=max(1,len(root.GetChildren()));updates=[]
    material=UsdShade.Material(stage.GetPrimAtPath('/Materials/runoff'))
    if not material:raise ValueError('Missing runoff material')
    for i,item in enumerate(pages(data,minimum)):
        path=f'/Lab/runoff/page_{i:06d}';g=UsdGeom.Points.Get(stage,path)
        if not g:
            g=UsdGeom.Points.Define(stage,path)
            g.CreatePointsAttr();g.CreateWidthsAttr();g.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex);g.CreateExtentAttr()
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
        points=item['points']
        for name,value in (('points',Vt.Vec3fArray.FromNumpy(points)),
            ('widths',Vt.FloatArray.FromNumpy(item['widths'])),
            ('primvars:displayColor',Vt.Vec3fArray.FromNumpy(item['color'])),
            ('extent',Vt.Vec3fArray.FromNumpy(np.asarray([points.min(0),points.max(0)],np.float32)))):
            updates.append((stage.GetRootLayer().GetAttributeAtPath(path+'.'+name),value))
    return updates


def author(stage,data):
    from pxr import Sdf
    updates=prepare(stage,data)
    with Sdf.ChangeBlock():
        for spec,value in updates:spec.default=value

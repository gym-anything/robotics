"""Visual apertures matching the four native wallboard depth-field holes.

The general wall collider stays a coarse box; nominated tool soles use the native
material/base field. This fixes occlusion of those holes in RGB, not robot-wide
wall penetration. Derivation changes only the partition mesh, never saved states.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from . import params as P
from .recording import canonical,read_recording,sha


def geometry(c,h):
    c=np.asarray(c);h=np.asarray(h);lo=c-h;hi=c+h;r=P.DW_HOLE_D/2
    holes=[(x,z) for x,z in P.DW_HOLES if lo[0]<x<hi[0] and lo[2]<z<hi[2]]
    xs=sorted({lo[0],hi[0],*(x+d for x,z in holes for d in (-r,r))})
    zs=sorted({lo[2],hi[2],*(z+d for x,z in holes for d in (-r,r))})
    vertices=[];faces=[]
    def quad(a,b,c,d):
        k=len(vertices);vertices.extend([a,b,c,d]);faces.extend([(k,k+1,k+2),(k,k+2,k+3)])
    y=hi[1]
    for x0,x1 in zip(xs[:-1],xs[1:]):
        for z0,z1 in zip(zs[:-1],zs[1:]):
            if any(abs((x0+x1)/2-x)<r and abs((z0+z1)/2-z)<r for x,z in holes):continue
            quad((x0,y,z0),(x0,y,z1),(x1,y,z1),(x1,y,z0))
    for x,z in holes:
        angles=np.linspace(0,2*np.pi,33);v=np.column_stack([np.cos(angles),np.sin(angles)])
        circle=v*r;square=v*(r/np.max(abs(v),axis=1))[:,None]
        def point(v,depth=0):return (x+v[0],y-depth,z+v[1])
        for k in range(32):
            quad(point(square[k]),point(square[k+1]),point(circle[k+1]),point(circle[k]))
            quad(point(circle[k]),point(circle[k+1]),point(circle[k+1],P.BOARD_T),point(circle[k],P.BOARD_T))
            n=len(vertices);vertices.extend([(x,y-P.BOARD_T,z),point(circle[k],P.BOARD_T),point(circle[k+1],P.BOARD_T)]);faces.append((n,n+1,n+2))
    # Back and outer edges keep the original closed wall envelope.
    x0,y0,z0=lo;x1,y1,z1=hi
    for a,b,c,d in [((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)),
       ((x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)),
       ((x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)),
       ((x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)),
       ((x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1))]:quad(a,b,c,d)
    v=np.asarray(vertices,np.float32);f=np.asarray(faces,np.int32)
    valid=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)>1e-14
    return v,f[valid]


def replace(stage):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,Vt
    path='/Room/partition_w';prim=stage.GetPrimAtPath(path)
    if not prim:raise ValueError('partition_w missing')
    material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    old=UsdGeom.Mesh(prim);points=np.asarray(old.GetPointsAttr().Get());matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    world=np.c_[points,np.ones(len(points))]@matrix;lo=world[:,:3].min(0);hi=world[:,:3].max(0)
    v,f=geometry((lo+hi)/2,(hi-lo)/2);g=old
    # Preserve child ordering so reversing this one prim is byte-identical.
    for prop in ('xformOp:translate','xformOp:orient'):
        prim.RemoveProperty(prop)
    g.GetXformOpOrderAttr().Set([])
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v));g.CreateFaceVertexCountsAttr([3]*len(f));g.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist());g.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);length=np.linalg.norm(normals,axis=1)
    assert np.isfinite(v).all() and (length>1e-14).all()
    normals/=length[:,None];g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)));g.SetNormalsInterpolation('uniform')
    uv=[]
    for face,n in zip(f,normals):uv.extend(v[face][:,[i for i in range(3) if i!=np.argmax(abs(n))]])
    UsdGeom.PrimvarsAPI(g).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv,np.float32)))
    return dict(path=path,vertices=len(v),triangles=len(f),hole_diameter_m=P.DW_HOLE_D,depth_m=P.BOARD_T)


def derive(source,output):
    from pxr import Sdf,Usd
    source,output=Path(source),Path(output);doc,frames=read_recording(source)
    if output.exists():raise FileExistsError(output)
    shutil.copytree(source,output);stage=Usd.Stage.Open(str(output/doc['scene']));layer=stage.GetRootLayer();before=layer.ExportToString()
    backup=Sdf.Layer.CreateAnonymous();Sdf.CopySpec(layer,'/Room/partition_w',backup,'/partition_w')
    info=replace(stage);layer.Save()
    check=Sdf.Layer.CreateAnonymous();check.TransferContent(layer);Sdf.CopySpec(backup,'/partition_w',check,'/Room/partition_w')
    assert check.ExportToString()==before,'unrelated scene edit'
    doc['assets'][doc['scene']]=sha((output/doc['scene']).read_bytes());doc['derived_visual']=dict(original_recording_sha256=sha((source/'recording.json').read_bytes()),physical_frames_unchanged=True,
       only_partition_mesh_changed=True,reason='Remove visual wall occlusion over native anchor holes; geometry matches their diameter/depth; no material state edits',source_sha256=sha(Path(__file__).read_bytes()),**info)
    (output/'recording.json').write_bytes(canonical(doc));read_recording(output);return doc['derived_visual']
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('--output',required=True);a=ap.parse_args();print(json.dumps(derive(a.source,a.output),indent=2))

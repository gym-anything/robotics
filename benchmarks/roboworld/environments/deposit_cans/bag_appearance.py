"""Authored recycling print attached to actual material vertices.

Reference product has blue printing (S4); artwork here is original, not a copy
of its logo. Roughness is an optical estimate. No geometry, physics or contents
are changed; historical recordings opt in and retain their original scene.
"""
from pathlib import Path
import shutil
import numpy as np

def bind(stage,folder,vertices,faces,*,segments,rows,cuff):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import FancyArrowPatch
    from pxr import UsdGeom,UsdShade,Sdf,Vt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig=Figure(figsize=(8,8),dpi=128);FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    color='#124fad'
    for angle in (90,210,330):
        a=np.deg2rad(angle);b=np.deg2rad(angle+105)
        start=np.array([.5,.62])+.17*np.array([np.cos(a),np.sin(a)])
        end=np.array([.5,.62])+.17*np.array([np.cos(b),np.sin(b)])
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=52,
            linewidth=12,color=color,connectionstyle='arc3,rad=.16'))
    ax.text(.5,.32,'RECYCLING',ha='center',color=color,fontsize=29,fontweight='bold')
    ax.text(.5,.23,'ONLY',ha='center',color=color,fontsize=29,fontweight='bold')
    ax.text(.5,.105,'EMPTY CANS / NO GENERAL WASTE',ha='center',color=color,fontsize=11)
    fig.savefig(folder/'recycling_print.svg',transparent=True)
    image=folder/'recycling_print.png';fig.savefig(image,transparent=True);fig.clear()
    module=folder/'printed_film.mdl'
    shutil.copy2(Path(__file__).with_name('printed_film.mdl'),module)
    material=UsdShade.Material.Define(stage,'/Materials/printed_film')
    shader=UsdShade.Shader.Define(stage,'/Materials/printed_film/surface')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(module.resolve())),'mdl')
    shader.SetSourceAssetSubIdentifier('printed_film','mdl')
    shader.CreateInput('print_texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image.resolve())))
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    g=UsdGeom.Mesh.Get(stage,'/Lab/collection_bag')
    q=np.asarray(vertices);f=np.asarray(faces).reshape(-1,3)
    uv=np.zeros((len(q),2),np.float32)
    side_count=(rows+(8 if cuff else 0))*segments
    # Two printed panels around the installed film; coordinates follow the
    # material ring index, not a world projection that slides over wrinkles.
    uv[:side_count,0]=(np.arange(side_count)%segments)/segments*2.
    uv[:side_count,1]=(q[:side_count,2]-.12)/.77
    per_face=uv[f].copy()
    seam=np.ptp(per_face[:,:,0],axis=1)>1.
    per_face[seam,:,0]+=2.*(per_face[seam,:,0]<1.)
    UsdGeom.PrimvarsAPI(g).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(
        Vt.Vec2fArray.FromNumpy(per_face.reshape(-1,2)))
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)

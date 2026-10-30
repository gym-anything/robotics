"""Authored vector packaging/signage and crease-aware optical normals.

Fictional branding; no downloaded product artwork. Labels carry no hidden
instance identities or task progress. PNGs are render textures rasterized from
the same vector design saved as SVG, not generated simulation evidence.
"""
from pathlib import Path
import numpy as np

def artwork(folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle,Circle,FancyArrowPatch
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);paths={}
    designs=[('can_0','#145648','SPARKLING','WATER','330 mL'),
        ('can_1','#164c79','AFTERHOURS','SODA','SLIM / ALUMINIUM'),
        ('can_2','#ad551c','CITRUS','SPARKLING DRINK','500 mL'),
        ('recycling_sign','#12694f','RECYCLING','EMPTY BEVERAGE CANS','No cups / no general waste'),
        ('landfill_sign','#323936','GENERAL WASTE','LANDFILL','Keep recyclables separate'),
        ('lid_label','#164780','CANS ONLY','EMPTY AND DRAINED','')]
    for name,color,title,subtitle,small in designs:
        can=name.startswith('can_');fig=Figure(figsize=(10,6),dpi=150);FigureCanvasAgg(fig)
        ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1000),ylim=(0,600));ax.axis('off')
        fig.patch.set_facecolor(color);ax.add_patch(Rectangle((0,0),1000,600,color=color))
        if can:
            for x,y,r in ((80,470,65),(640,100,95),(390,450,18),(460,125,35)):
                ax.add_patch(Circle((x,y),r,fill=False,edgecolor='#cde0ca',linewidth=2,alpha=.5))
            ax.add_patch(Rectangle((650,65),295,470,facecolor='#f2f0df',edgecolor='none'))
            ax.text(670,505,'INGREDIENTS',fontsize=14,fontweight='bold',color='#24352d')
            ax.text(670,470,'Carbonated water',fontsize=11,color='#24352d')
            ax.text(670,445,'Natural flavour',fontsize=11,color='#24352d')
            for y in range(400,195,-35):ax.plot([670,925],[y,y],color='#35483d',linewidth=.6)
            ax.text(670,160,'PLEASE RECYCLE',fontsize=12,fontweight='bold',color='#24352d')
            for i in range(45):
                width=1+(i*7%3);x=676+i*5
                ax.add_patch(Rectangle((x,85),width,50,color='#24352d'))
            center=320
            ax.text(center,390,'VENUE / ORIGINAL',ha='center',fontsize=14,color='#eceddf')
            ax.text(center,300,title,ha='center',fontsize=34,fontweight='bold',color='#fffdeb')
            ax.text(center,236,subtitle,ha='center',fontsize=18,color='#eceddf')
            ax.text(center,95,small,ha='center',fontsize=17,color='#fffdeb')
        else:
            ax.text(500,390,title,ha='center',fontsize=37,fontweight='bold',color='white')
            ax.text(500,265,subtitle,ha='center',fontsize=23,fontweight='bold',color='white')
            ax.text(500,115,small,ha='center',fontsize=18,color='white')
            ax.plot([80,920],[330,330],color='white',linewidth=2)
        fig.savefig(folder/(name+'.svg'),format='svg',facecolor=color)
        path=folder/(name+'.png');fig.savefig(path,format='png',facecolor=color)
        fig.clear();paths[name]=path
    return paths

def texture(stage,material,path):
    from pxr import Sdf,UsdShade,Gf
    prefix=str(material.GetPath());reader=UsdShade.Shader.Define(stage,prefix+'/uv')
    reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex=UsdShade.Shader.Define(stage,prefix+'/print');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(path).resolve())))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    for axis in ('S','T'):tex.CreateInput('wrap'+axis,Sdf.ValueTypeNames.Token).Set('repeat')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    surface=UsdShade.Shader.Get(stage,prefix+'/surface')
    surface.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
    mdl=UsdShade.Shader.Define(stage,prefix+'/rtx_print')
    mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    mdl.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'),'mdl');mdl.SetSourceAssetSubIdentifier('OmniPBR','mdl')
    mdl.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.))
    mdl.CreateInput('diffuse_texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(path).resolve())))
    mdl.CreateInput('reflection_roughness_constant',Sdf.ValueTypeNames.Float).Set(.3)
    mdl.CreateInput('metallic_constant',Sdf.ValueTypeNames.Float).Set(.15)
    mdl.CreateInput('project_uvw',Sdf.ValueTypeNames.Bool).Set(False)
    material.CreateSurfaceOutput('mdl').ConnectToSource(mdl.ConnectableAPI(),'out')

def crease_normals(g,vertices,faces,angle=35.):
    from pxr import Vt
    v=np.asarray(vertices);f=np.asarray(faces).reshape(-1,3)
    raw=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    fn=raw/np.maximum(np.linalg.norm(raw,axis=1)[:,None],1e-20)
    adjacency=[[] for _ in v]
    for i,row in enumerate(f):
        for vertex in row:adjacency[int(vertex)].append(i)
    result=[];threshold=np.cos(np.deg2rad(angle))
    for i,row in enumerate(f):
        for vertex in row:
            neighbors=np.asarray(adjacency[int(vertex)])
            selected=neighbors[(fn[neighbors]@fn[i])>=threshold]
            normal=raw[selected].sum(0);normal/=max(np.linalg.norm(normal),1e-20);result.append(normal)
    g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(result,np.float32)))
    g.SetNormalsInterpolation('faceVarying')

def printed_quad(stage,path,points,material):
    from pxr import UsdGeom,UsdShade,Sdf,Vt
    g=UsdGeom.Mesh.Define(stage,path);g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points,np.float32)))
    g.CreateFaceVertexCountsAttr([4]);g.CreateFaceVertexIndicesAttr([0,1,2,3]);g.CreateSubdivisionSchemeAttr('none')
    g.CreateDoubleSidedAttr(False)
    UsdGeom.PrimvarsAPI(g).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    return g


class CreaseNormals:
    """Fixed-topology, vectorized optical normals; never changes vertices."""
    def __init__(self,faces,count,angle=35.):
        self.faces=np.asarray(faces,np.int32).reshape(-1,3)
        adjacency=[[]for _ in range(count)]
        for i,row in enumerate(self.faces):
            for v in row:adjacency[int(v)].append(i)
        width=max(map(len,adjacency))
        table=np.full((count,width),-1,np.int32)
        for i,neighbors in enumerate(adjacency):table[i,:len(neighbors)]=neighbors
        self.neighbors=table[self.faces.ravel()]
        self.owner=np.repeat(np.arange(len(self.faces)),3)
        self.threshold=np.cos(np.deg2rad(angle))
    def __call__(self,vertices):
        p=np.asarray(vertices)[self.faces]
        raw=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        normal=raw/np.maximum(np.linalg.norm(raw,axis=1)[:,None],1e-20)
        neighbors=np.maximum(self.neighbors,0)
        cosine=np.einsum('nki,ni->nk',normal[neighbors],normal[self.owner])
        use=(self.neighbors>=0)&(cosine>=self.threshold)
        result=np.sum(raw[neighbors]*use[:,:,None],axis=1)
        result/=np.maximum(np.linalg.norm(result,axis=1)[:,None],1e-20)
        return result.astype(np.float32)

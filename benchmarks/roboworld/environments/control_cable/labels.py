"""Legible physical equipment labels exported as USD surface textures."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pxr import Gf,Sdf,UsdGeom,UsdShade,Vt
def export_labels(stage,labels,output):
    folder=Path(output)/"labels";folder.mkdir(parents=True,exist_ok=True)
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for i,entry in enumerate(labels):
        w,h=entry["width"],entry["height"]
        # Preserve the physical aspect ratio. A forced 512 px minimum width
        # with fixed 144 px height squeezed all multiline clipboard lettering.
        height=max(144,round(h*4096),round(128*h/w))
        width=max(1,round(height*w/h))
        image=Image.new("RGB",(width,height),(224,226,214))
        draw=ImageDraw.Draw(image)
        size=round(height*.55);font=ImageFont.truetype(font_path,size)
        while True:
            bounds=draw.multiline_textbbox((0,0),entry["text"],font=font,spacing=6)
            if bounds[2]-bounds[0]<=width-20 and bounds[3]-bounds[1]<=height-20:break
            size-=1;font=ImageFont.truetype(font_path,size)
        draw.multiline_text((width/2,height/2),entry["text"],font=font,
            fill=(19,24,29),anchor="mm",align="left",spacing=6)
        filename=folder/f"label_{i}.png";image.save(filename)
        p=entry["position"]
        g=UsdGeom.Mesh.Define(stage,f"/Labels/label_{i}")
        normal=np.asarray(entry.get("normal",[-1,0,0]),float)
        normal/=np.linalg.norm(normal)
        up=np.array([0.,0.,1.]) if abs(normal[2])<.95 else np.array([1.,0.,0.])
        up-=normal*np.dot(up,normal);up/=np.linalg.norm(up)
        right=np.cross(up,normal)
        corners=[np.asarray(p)+right*dx+up*dy for dx,dy in
            ((-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2))]
        g.CreatePointsAttr([Gf.Vec3f(*map(float,corner)) for corner in corners])
        g.CreateFaceVertexCountsAttr([4]);g.CreateFaceVertexIndicesAttr([0,1,2,3])
        g.CreateSubdivisionSchemeAttr("none")
        uv=UsdGeom.PrimvarsAPI(g).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"vertex")
        uv.Set([Gf.Vec2f(0,0),Gf.Vec2f(1,0),Gf.Vec2f(1,1),Gf.Vec2f(0,1)])
        m=UsdShade.Material.Define(stage,f"/LabelMaterials/label_{i}")
        shader=UsdShade.Shader.Define(stage,str(m.GetPath())+"/surface");shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.7)
        texture=UsdShade.Shader.Define(stage,str(m.GetPath())+"/texture");texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file",Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(filename.resolve())))
        reader=UsdShade.Shader.Define(stage,str(m.GetPath())+"/uv");reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),"result")
        shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),"rgb")
        m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)

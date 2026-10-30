"""Actual-state RTX rendering shared by fixtures and the final workcell."""
import base64
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from pencap.newton15.render import pose, mesh, camera

COLORS={
 "laminate":((.35,.40,.43),.46,.05), "aluminum":((.55,.58,.60),.28,.85),
 "concrete":((.25,.27,.29),.90,0.), "steel":((.15,.18,.22),.31,.85),
 "nickel":((.60,.64,.67),.24,.9), "cable_black":((.009,.012,.014),.52,0.),
 "robot_white":((.77,.79,.80),.29,0.), "robot_dark":((.026,.033,.043),.38,.1),
 "paint_blue":((.022,.082,.12),.36,.25), "polycarbonate":((.65,.69,.68),.30,0.),
 "copper":((.69,.32,.12),.20,.95), "brass":((.52,.33,.09),.22,.9),
 "robot_red":((.55,.012,.008),.3,0.),
 "wall":((.48,.51,.49),.85,0.),"cabinet":((.45,.50,.50),.40,.15),
 "duct":((.52,.56,.55),.63,0.),"belt":((.025,.035,.032),.8,0.),
 "safety_yellow":((.92,.52,.015),.43,.1),"red_lens":((.40,.015,.012),.14,0.),
 "cable_green":((.025,.21,.06),.53,0.),"cable_blue":((.025,.12,.34),.55,0.),
 "cable_brown":((.17,.065,.022),.56,0.),
 "safety_orange":((.91,.22,.018),.42,0.),
 "indicator_power":((.06,.65,.11),.20,0.),"indicator_sensor":((.90,.42,.01),.20,0.),
 "indicator_pass_":((.06,.65,.11),.20,0.),"indicator_fail":((.70,.012,.006),.20,0.)
}

def export(assembly, body_q, output):
    import newton
    from whiteboard.usd_scene import bevel_box
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(output/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage,"Z")
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,"/Lab").GetPrim())
    mats={}
    for name,(rgb,rough,metal) in COLORS.items():
        m=UsdShade.Material.Define(stage,"/Materials/"+name)
        s=UsdShade.Shader.Define(stage,"/Materials/"+name+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        s.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(rough)
        s.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(metal)
        if name.startswith("indicator_"):
            s.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(v*.15 for v in rgb)))
            s.CreateInput("emissiveColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0,0,0))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),"surface")
        mats[name]=m
    from whiteboard.visual_assets import copy_to
    from .materials import bind_floor
    textures=copy_to(output)
    # The same CC0 measured-scale surfaces as the reference environments.
    bind_floor(stage,mats["concrete"],textures["beige_wall_001"],3.)
    for i,q in enumerate(body_q):
        pose(UsdGeom.Xform.Define(stage,f"/Lab/body_{i}"),q[:3],q[3:])
    b=assembly.builder
    for i in range(b.shape_count):
        if not b.shape_flags[i]&int(newton.ShapeFlags.VISIBLE):
            continue
        body,label=b.shape_body[i],b.shape_label[i]
        path=f"/Lab/body_{body}/shape_{i}" if body>=0 else f"/Lab/static_{i}"
        kind,scale,source=b.shape_type[i],np.asarray(b.shape_scale[i]),b.shape_source[i]
        if source is not None and hasattr(source,"vertices"):
            # Do not average normals across the sharp folded-case and aperture
            # edges: it makes a planar cabinet face appear dented in the render.
            g=mesh(stage,path,np.asarray(source.vertices)*scale,source.indices,
                   smooth=label.startswith(("left/","right/")))
        elif kind==newton.GeoType.BOX:
            g=bevel_box(stage,path,2*scale,min(.001,float(scale.min())*.04))
        elif kind==newton.GeoType.SPHERE:
            g=UsdGeom.Sphere.Define(stage,path); g.CreateRadiusAttr(float(scale[0]))
        elif kind in (newton.GeoType.CYLINDER,newton.GeoType.CAPSULE):
            g=(UsdGeom.Cylinder if kind==newton.GeoType.CYLINDER else UsdGeom.Capsule).Define(stage,path)
            g.CreateRadiusAttr(float(scale[0])); g.CreateHeightAttr(float(2*scale[1])); g.CreateAxisAttr("Z")
        elif kind==newton.GeoType.PLANE:
            # Finite room floor only for visualization; the physics plane remains.
            g=bevel_box(stage,path,(7.,7.,.10),.002)
        else:
            raise ValueError(f"Unsupported visible geometry {label}: {kind}")
        local=np.asarray(b.shape_transform[i])
        if kind==newton.GeoType.PLANE:
            from scipy.spatial.transform import Rotation
            local=local.copy()
            local[:3]+=Rotation.from_quat(local[3:]).apply([0,0,-.05])
        pose(g,local[:3],local[3:])
        mat=assembly.materials.get(i,"steel")
        if label.startswith(("left/","right/")):
            rgb=np.asarray(b.shape_color[i])
            # Newton imports one DAE material group per shape. Its mesh loader
            # stores gamma-2.2 encoded colour; USD PreviewSurface expects linear.
            # Bind that source colour directly to this unchanged material group.
            mat="fr3_"+str(i)
            material=UsdShade.Material.Define(stage,"/Materials/"+mat)
            surface=UsdShade.Shader.Define(stage,"/Materials/"+mat+"/surface")
            surface.CreateIdAttr("UsdPreviewSurface")
            surface.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,rgb**2.2)))
            surface.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.35)
            material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),"surface")
            mats[mat]=material
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[mat])
    for i,pos in enumerate(((-.1,-.65,2.8),(.70,.80,2.8),(1.1,0,2.6))):
        light=UsdLux.RectLight.Define(stage,f"/Lights/panel_{i}")
        light.CreateWidthAttr(.65);light.CreateHeightAttr(1.2)
        # 3900 lm over a .78 m2 Lambertian panel: L=flux/(pi*A).
        palette={"neutral":(.96,.98,1.),"warm":(1.,.88,.72),"cool":(.82,.92,1.)}
        light.CreateIntensityAttr(1600.)
        light.CreateColorAttr(Gf.Vec3f(*palette[assembly.config.room_style]))
        pose(light,pos)
    light=UsdLux.DomeLight.Define(stage,"/Lights/ambient")
    light.CreateIntensityAttr(150.)
    if hasattr(assembly,"labels"):
        from .labels import export_labels
        export_labels(stage,assembly.labels,output)
    camera(stage,"room",(-1.55,-1.85,2.12),(.25,0,.90),31.)
    camera(stage,"bench",(.10,-.08,1.32),(.60,0,1.12),28.)
    camera(stage,"cable",(.02,-1.55,1.18),(.25,-1.22,.77),57.)
    camera(stage,"handling",(.04,-1.08,1.02),(.34,-.85,.835),45.)
    camera(stage,"connection",(.22,-.08,1.26),(.553,.10,1.16),40.)
    # External view: jaw/boot/socket stay visible beside the wrist.
    camera(stage,"connection_side",(.53,-.25,1.30),(.50,.10,1.16),40.)
    camera(stage,"inventory",(.90,-1.15,.61),(.25,-1.06,.43),32.)
    camera(stage,"schedule",(.25,-2.15,.607),(.25,-1.526,.607),40.)
    camera(stage,"overhead",(.42,0,2.40),(.42,0,.75),38.)
    camera(stage,"hardware_front",(.43,.10,.94),(.42,0,.900),72.)
    camera(stage,"hardware_side",(.52,.02,.920),(.42,0,.900),70.)
    camera(stage,"tool",(.70,-.36,1.15),(.41,0,1.01),40.)
    # A vertical look direction is parallel to the shared helper's Z-up vector.
    # Specify image-up along world Y so its camera matrix is nonsingular.
    overhead=UsdGeom.Xformable(stage.GetPrimAtPath("/Cameras/overhead"))
    overhead.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(.42,0,2.40),Gf.Vec3d(.42,0,.75),Gf.Vec3d(0,1,0)).GetInverse())
    stage.GetRootLayer().Save()
    from .cameras import export as export_mounted_cameras
    export_mounted_cameras(stage,assembly)
    stage.GetRootLayer().Save()
    return output/"scene.usdc"

def open_renderer(assembly,body_q,output,*,width=1280,height=720,samples=16):
    from titrate.newton15.renderer import Renderer
    return Renderer(export(assembly,body_q,output),width=width,height=height,
                    samples=samples,backend="rtx")

def main():
    import argparse, json
    from dataclasses import replace
    from .config import Config
    from .physics import Assembly
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--frame",type=int,default=-1)
    a=p.parse_args()
    result=json.loads((a.run/"result.json").read_text())
    c=Config(cable=replace(Config().cable,segments=result["config"]["cable"]["segments"]))
    assembly=Assembly(c,robots=result["robots"],attachment=result["attachment"])
    recorded=np.load(a.run/"poses.npz")
    q=recorded["body_q"][a.frame]
    tick=int(recorded["ticks"][a.frame])
    if len(q)!=assembly.builder.body_count:
        raise ValueError("Recorded body topology does not match the render assembly")
    renderer=open_renderer(assembly,q,a.run/"render")
    try:
        for name in ("room","bench","cable","overhead"):
            result=renderer.observe(tick,q,(name,))
            for name,blob in result["images"].items():
                (a.run/f"{name}_{tick}.png").write_bytes(base64.b64decode(blob))
            print(f"Rendered {name} at tick {tick}",flush=True)
        (a.run/"render_receipt.json").write_text(json.dumps(
            {k:v for k,v in result.items() if k!="images"},indent=2))
    finally:
        renderer.close()
if __name__=="__main__":
    main()

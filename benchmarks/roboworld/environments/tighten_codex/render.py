"""RTX views of actual Newton body poses. Rendering never advances dynamics."""

from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from pencap.newton15.render import pose, mesh, camera


def _printed_assets(folder):
    """Deterministic appearance-only work instructions and cell identification."""
    from PIL import Image, ImageDraw, ImageFont
    folder.mkdir(parents=True,exist_ok=True)
    font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    sign=Image.new("RGB",(1200,480),(13,36,57))
    d=ImageDraw.Draw(sign)
    d.rectangle((0,0,1200,55),fill=(230,181,45))
    d.text((64,90),"CELL 04",font=ImageFont.truetype(bold,110),fill=(244,247,247))
    d.text((64,225),"SENSOR SERVICE BENCH",font=ImageFont.truetype(bold,53),fill=(178,220,233))
    d.text((64,335),"MOUNT  /  AIM  /  LOCK  /  VERIFY",font=ImageFont.truetype(font,40),fill=(233,236,232))
    sign.save(folder/"inspection_sign.png")
    card=Image.new("RGB",(1200,700),(242,240,228))
    d=ImageDraw.Draw(card)
    d.rectangle((0,0,1200,120),fill=(22,76,111))
    d.text((48,23),"INSPECTION HEAD  /  SERVICE CARD",font=ImageFont.truetype(bold,45),fill="white")
    for n,(label,detail) in enumerate((
        ("01  BENCH", "W4F head on BEF-W4-A, white reference"),
        ("02  M3 SENSOR", "2.5 mm hex  |  0.35 N m driver"),
        ("03  M4 FRAME", "3 mm hex  |  0.60 N m driver"),
        ("04  VERIFY", "Driver clicks; return flag; detect"))):
        y=158+n*119
        d.rectangle((48,y+12,76,y+40),outline=(31,83,118),width=4)
        d.text((106,y),label,font=ImageFont.truetype(bold,38),fill=(20,51,70))
        d.text((106,y+49),detail,font=ImageFont.truetype(font,27),fill=(45,56,63))
    d.rectangle((45,642,1154,648),fill=(44,128,168))
    card.save(folder/"service_card.png")


def _print_material(stage,name,filename):
    from whiteboard.appearance import mdl_surface
    m=UsdShade.Material.Define(stage,"/Materials/"+name)
    s=mdl_surface(stage,m,"OmniPBR")
    s.CreateInput("diffuse_texture",Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/"+filename))
    s.CreateInput("project_uvw",Sdf.ValueTypeNames.Bool).Set(False)
    s.CreateInput("texture_scale",Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1,1))
    s.CreateInput("reflection_roughness_constant",Sdf.ValueTypeNames.Float).Set(.75)
    return m


PALETTE={
    "laminate":((.37,.42,.43),.55,.05),
    "concrete":((.24,.26,.27),.82,0.),
    "wall":((.72,.71,.67),.88,0.),
    "aluminum":((.63,.67,.69),.28,.85),
    "steel":((.30,.33,.36),.28,.9),
    "nickel":((.64,.67,.70),.22,.87),
    "sensor_blue":((.018,.16,.38),.32,0.),
    "red_lens":((.34,.025,.018),.10,0.),
    "cable_black":((.012,.016,.021),.54,0.),
    "paper":((.91,.90,.84),.90,0.),
    "robot_dark":((.024,.032,.040),.48,.1),
    "robot_white":((.78,.80,.80),.31,.0),
    "safety_yellow":((.91,.55,.015),.42,.08),
    "tool_blue":((.015,.12,.30),.36,.02),
    "tool_green":((.11,.52,.18),.31,.0),
    "tool_yellow":((.89,.48,.012),.35,.0),
    "floor_epoxy":((.14,.18,.20),.72,.0),
    "floor_joint":((.055,.070,.077),.84,.0),
    "cabinet_blue":((.052,.12,.20),.45,.18),
    "machine_gray":((.24,.29,.31),.48,.45),
    "belt_black":((.035,.046,.049),.77,.0),
    "status_green":((.04,.62,.27),.28,.0),
    "screen":((.025,.080,.105),.19,.05),
}


# Render-only indicator lamps: name -> (lit colour, emission). The renderer
# sets each observation's on/off state from simulator truth (RtxRenderer
# indicators). At emission 6 the 2.4 mm LED rendered near-white in bench_rgb.
INDICATORS={"w4f_output_led":((.95,.72,.02),1.5)}


def indicator_states(env):
    """Current on/off state of the exported indicator lamps (mixed only: the
    W4F's yellow output LED follows the evaluator's geometric optical ray)."""
    from .task import uses_eight
    if env.config.task!="mixed" and not uses_eight(env.config):
        return {}
    return {"w4f_output_led":bool(env.evaluation["optical"]["detected"])}


def _mixed_cues(stage,env,mats):
    """mixed work order, appearance only (no Newton shape, mass or contact):
    the W4F's yellow "object present" LED beside its green power LED (SICK
    WTB4FP display), and a yellow rating ring on the outside of the M4
    driver's grip collar, whose own yellow band lies hidden inside it."""
    from whiteboard.usd_scene import bevel_box
    rgb,glow=INDICATORS["w4f_output_led"]
    on=indicator_states(env)["w4f_output_led"]
    m=UsdShade.Material.Define(stage,"/Materials/w4f_output_led")
    m.GetPrim().SetCustomDataByKey("indicator_rgb",Gf.Vec3f(*rgb))
    m.GetPrim().SetCustomDataByKey("indicator_emission",float(glow))
    s=UsdShade.Shader.Define(stage,"/Materials/w4f_output_led/surface")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(v*(1. if on else .15) for v in rgb)))
    s.CreateInput("emissiveColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(v*(glow if on else 0.) for v in rgb)))
    s.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.3)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),"surface")
    # Same size and row as the modelled green LED (scene.py sensor_status_led).
    led=bevel_box(stage,f"/Lab/body_{env.items['sensor']}/w4f_output_led",(.0024,.0024,.0012),.000024)
    pose(led,(.0015,-.004,.0188))
    UsdShade.MaterialBindingAPI.Apply(led.GetPrim()).Bind(m)
    v,f=_smooth_cylinder(.0118,.006)
    ring=mesh(stage,f"/Lab/body_{env.items['M4_driver']['grip']}/M4_driver_rating_ring",v,f,smooth=True)
    pose(ring,(0.,0.,.072))
    UsdShade.MaterialBindingAPI.Apply(ring.GetPrim()).Bind(mats["tool_yellow"])


def _smooth_cylinder(radius, height, segments=64):
    """Side ring and two caps with their own vertices (crisp rims)."""
    t=np.linspace(0,2*np.pi,segments,endpoint=False)
    ring=np.stack([radius*np.cos(t),radius*np.sin(t)],1)
    z0,z1=-height/2,height/2
    side=[(x,y,z) for z in (z0,z1) for x,y in ring]
    caps=[(x,y,z0) for x,y in ring]+[(x,y,z1) for x,y in ring]+[(0,0,z0),(0,0,z1)]
    v=np.asarray(side+caps,float)
    n=segments;faces=[]
    for k in range(n):
        j=(k+1)%n
        faces+=[(k,j,n+j),(k,n+j,n+k)]
    b0=2*n;c0=b0+2*n
    for k in range(n):
        j=(k+1)%n
        faces+=[(c0,b0+j,b0+k),(c0+1,b0+n+k,b0+n+j)]
    return v,np.asarray(faces,np.int32).ravel()


def export(env, output):
    import newton
    from whiteboard.usd_scene import bevel_box, mesh as usd_mesh
    from whiteboard.visual_assets import copy_to
    from whiteboard.appearance import bind_pbr

    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(output/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage,"Z")
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,"/Lab").GetPrim())
    mats={}
    for name,(rgb,rough,metal) in PALETTE.items():
        m=UsdShade.Material.Define(stage,"/Materials/"+name)
        s=UsdShade.Shader.Define(stage,"/Materials/"+name+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        s.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(rough)
        s.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),"surface")
        mats[name]=m
    textures=copy_to(output)
    _printed_assets(output/"textures")
    bind_pbr(stage,mats["wall"],textures["beige_wall_001"],3.,coated=True,roughness=.82)
    bind_pbr(stage,mats["laminate"],textures["oak_veneer_01"],2.,coated=True,roughness=.54)
    sign_mat=_print_material(stage,"inspection_sign_print","inspection_sign.png")
    card_mat=_print_material(stage,"service_card_print","service_card.png")
    b=env.builder
    for i,q in enumerate(env.state.body_q.numpy()):
        pose(UsdGeom.Xform.Define(stage,f"/Lab/body_{i}"),q[:3],q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i]&int(newton.ShapeFlags.VISIBLE):
            continue
        body,label=b.shape_body[i],b.shape_label[i]
        path=f"/Lab/body_{body}/shape_{i}" if body>=0 else f"/Lab/static_{i}"
        kind,scale,source=b.shape_type[i],np.asarray(b.shape_scale[i]),b.shape_source[i]
        if source is not None and hasattr(source,"vertices"):
            g=mesh(stage,path,np.asarray(source.vertices)*scale,source.indices,
                   smooth=label.startswith(("left/","right/")))
        elif kind==newton.GeoType.BOX:
            g=bevel_box(stage,path,2*scale,min(.0008,float(scale.min())*.04))
        elif kind==newton.GeoType.SPHERE:
            g=UsdGeom.Sphere.Define(stage,path)
            g.CreateRadiusAttr(float(scale[0]))
        elif kind==newton.GeoType.CYLINDER:
            # A smooth 64-segment mesh: implicit USD cylinders render with
            # visible facets on the tool handles in every close-up.
            v,f=_smooth_cylinder(float(scale[0]),float(2*scale[1]))
            g=mesh(stage,path,v,f,smooth=True)
        elif kind==newton.GeoType.CAPSULE:
            g=UsdGeom.Capsule.Define(stage,path)
            g.CreateRadiusAttr(float(scale[0]))
            g.CreateHeightAttr(float(2*scale[1]))
            g.CreateAxisAttr("Z")
        elif kind==newton.GeoType.PLANE:
            g=bevel_box(stage,path,(7.,7.,.1),.002)
        else:
            raise ValueError(f"Unsupported visible shape {label}: {kind}")
        local=np.asarray(b.shape_transform[i]).copy()
        if kind==newton.GeoType.PLANE:
            from scipy.spatial.transform import Rotation
            local[:3]+=Rotation.from_quat(local[3:]).apply((0,0,-.05))
        pose(g,local[:3],local[3:])
        mat=env.materials.get(i,"steel")
        if label.startswith(("left/","right/")):
            # Pinned FR3 DAE shapes carry their original colours in Newton.
            rgb=np.asarray(b.shape_color[i])**2.2
            mat=f"fr3_{i}"
            m=UsdShade.Material.Define(stage,"/Materials/"+mat)
            s=UsdShade.Shader.Define(stage,"/Materials/"+mat+"/surface")
            s.CreateIdAttr("UsdPreviewSurface")
            s.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*map(float,rgb)))
            s.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.35)
            m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),"surface")
            mats[mat]=m
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[mat])
    from .task import uses_eight
    if env.config.task=="mixed" or uses_eight(env.config):
        _mixed_cues(stage,env,mats)
    # Non-interactive room context remains behind the robot's reachable
    # workcell. It adds no collision proxy or hidden task affordance; the
    # contact-bearing machine, tools, bench and robot above are Newton bodies.
    def scenic(name, center, size, material, bevel=.003):
        geom=bevel_box(stage,"/Lab/context/"+name,size,bevel)
        pose(geom,center)
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mats[material])
    scenic("epoxy_floor",(0,0,-.025),(5.5,5.5,.045),"floor_epoxy")
    scenic("left_room_wall",(-2.68,0,1.55),(.08,5.5,3.10),"wall")
    scenic("right_room_wall",(2.68,0,1.55),(.08,5.5,3.10),"wall")
    for k in range(-5,6):
        x=k*.50
        scenic(f"floor_seam_x_{k+5}",(x,0,.001),(.003,5.2,.001),"floor_joint",.0002)
        scenic(f"floor_seam_y_{k+5}",(0,x,.001),(5.2,.003,.001),"floor_joint",.0002)
    for i,x in enumerate((-.98,.98)):
        scenic(f"safety_lane_{i}",(x,-.75,.002),(.025,2.0,.001),"safety_yellow",.0002)
    # A stopped conveyor is the inspection station's production context;
    # its visual-only run sits behind the maintenance table and robot reach.
    scenic("conveyor_frame",(0,.71,.54),(1.85,.42,.22),"machine_gray")
    scenic("conveyor_belt",(0,.71,.655),(1.73,.32,.014),"belt_black")
    for k in range(12):
        x=-.78+k*.14
        scenic(f"conveyor_crossbar_{k}",(x,.71,.666),(.008,.29,.004),"steel",.001)
    for side,y in enumerate((.48,.94)):
        scenic(f"conveyor_guard_{side}",(0,y,.74),(1.84,.022,.14),"aluminum")
    # Service cupboards, control panel and cable raceway occupy the rear
    # wall, outside all normal tool paths. They are static appearance only.
    for i,x in enumerate((-1.25,1.25)):
        scenic(f"rear_cabinet_{i}",(x,1.46,.94),(.54,.33,1.34),"cabinet_blue")
        scenic(f"cabinet_door_{i}",(x,1.277,.98),(.48,.012,1.19),"machine_gray")
        scenic(f"cabinet_handle_{i}",(x+.16,1.264,.98),(.012,.012,.18),"nickel",.002)
        scenic(f"cabinet_plinth_{i}",(x,1.46,.17),(.54,.33,.16),"robot_dark")
    scenic("control_cabinet",(.74,1.13,1.02),(.34,.26,1.45),"cabinet_blue")
    scenic("control_screen",(.74,.992,1.32),(.25,.005,.22),"screen",.001)
    for i,length in enumerate((.16,.19,.11)):
        scenic(f"screen_line_{i}",(.70,.985,1.39-i*.055),(length,.0015,.007),
               "status_green" if i==0 else "aluminum",.0003)
    scenic("emergency_stop_housing",(.84,.988,1.07),(.052,.018,.052),"safety_yellow")
    scenic("emergency_stop_button",(.84,.975,1.07),(.035,.009,.035),"red_lens")
    scenic("cable_raceway",(0,1.61,2.12),(2.85,.08,.075),"machine_gray")
    scenic("station_sign_backing",(0,1.62,1.66),(.70,.025,.28),"machine_gray")
    sign_geom=usd_mesh(stage,"/Lab/context/station_sign_face",
        [(-.34,0,-.13),(.34,0,-.13),(.34,0,.13),(-.34,0,.13)],
        [[0,1,2,3]],[(0,0),(1,0),(1,1),(0,1)])
    pose(sign_geom,(0,1.604,1.66))
    UsdShade.MaterialBindingAPI.Apply(sign_geom.GetPrim()).Bind(sign_mat)
    for i,x in enumerate((-.70,.70)):
        scenic(f"ceiling_light_{i}",(x,.02,2.62),(.55,.19,.016),"paper")
    # Ink is a visual decal over the existing physical service card.
    card_geom=usd_mesh(stage,"/Lab/service_card_print",
        [(-.12,-.07,0),(.12,-.07,0),(.12,.07,0),(-.12,.07,0)],
        [[0,1,2,3]],[(0,0),(1,0),(1,1),(0,1)])
    pose(card_geom,(-.10,-.20,.78308))  # on the scene's service_card box
    UsdShade.MaterialBindingAPI.Apply(card_geom.GetPrim()).Bind(card_mat)
    for i,position in enumerate(((-.8,-.35,2.65),(.8,.1,2.60))):
        light=UsdLux.RectLight.Define(stage,f"/Lights/softbox_{i}")
        light.CreateWidthAttr(.75)
        light.CreateHeightAttr(1.20)
        light.CreateIntensityAttr(1800.)
        light.CreateColorAttr(Gf.Vec3f(1.,.96,.90))
        pose(light,position)
    dome=UsdLux.DomeLight.Define(stage,"/Lights/ambient")
    dome.CreateIntensityAttr(140.)
    camera(stage,"room",(1.50,-2.08,1.67),(.03,.24,1.04),36.)
    camera(stage,"bench",(.52,-.50,1.35),(.13,.08,.99),45.)
    camera(stage,"hardware",(.30,-.14,1.20),(.12,.09,1.082),48.)
    camera(stage,"flag_side",(.54,-.18,1.20),(.30,-.01,1.09),42.)
    camera(stage,"head",(.335,-.06,1.14),(.2655,.075,1.087),78.)
    camera(stage,"sensor_tool",(.42,-.11,1.17),(.2605,.07,1.078),54.)
    camera(stage,"hand_sensor",(-.06,-.27,1.22),(.165,.04,1.08),36.)
    # Unobstructed close views of the pivot M3 head for the bit insertion.
    camera(stage,"insert_side",(.315,.055,1.100),(.261,.0733,1.074),80.)
    camera(stage,"insert_below",(.228,.040,1.040),(.261,.0733,1.074),70.)
    camera(stage,"frame_detail",(.17,-.12,1.20),(.022,.10,1.068),54.)
    camera(stage,"frame_under",(.10,-.035,1.025),(.022,.10,1.080),58.)
    camera(stage,"frame_back_under",(.08,.21,.975),(.022,.10,1.078),60.)
    # Beside the access guard, below the rail: sees the upward M4, washer,
    # bit and the guard's return lip together.
    camera(stage,"access_under",(-.045,.175,.99),(.024,.10,1.074),48.)
    camera(stage,"frame_front_detail",(.09,-.12,1.14),(.0,.085,1.067),65.)
    # The green M3 driver's holder is at x 0.36 m, beside the right arm.
    camera(stage,"hand_detail",(.64,-.02,1.14),(.36,.21,.86),40.)
    camera(stage,"hand_side",(-.25,-.51,1.35),(-.05,-.26,1.23),35.)
    camera(stage,"drivers",(-.05,-.18,1.17),(-.24,.20,.80),52.)
    camera(stage,"cradles_close",(.36,-.08,1.02),(.36,.22,.81),24.)
    # The optical principal plane is 0.1 mm in front of the modeled lens
    # face (y = -0.246); a camera eye inside its own collidable glass is black.
    camera(stage,"bench_rgb",(.17,-.2469,1.16),(.2655,.03,1.085),38.,role="policy")
    camera(stage,"overview_rgb",(0,.330,2.196),(0,-.14,1.08),20.,role="policy")
    stage.GetRootLayer().Save()
    return output/"scene.usdc"


def open_renderer(env,output,*,width=1280,height=720,samples=12,interpreter=None,
                  reuse_scene=False):
    """Start a render worker. With reuse_scene, a USD stage already exported
    for this environment is reused: its geometry and materials are static and
    every observation supplies the current body poses."""
    from titrate.newton15.renderer import Renderer
    scene=Path(output)/"scene.usdc"
    if not (reuse_scene and scene.is_file()):
        scene=export(env,output)
    return Renderer(scene,width=width,height=height,
                    samples=samples,interpreter=interpreter,backend="rtx")

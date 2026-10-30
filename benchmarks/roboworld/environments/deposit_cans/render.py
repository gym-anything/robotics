"""Render exact Newton poses/surfaces; shared render-only worker never steps."""
from gym_anything_robotics.paths import worker_env
from gym_anything_robotics.render_lifecycle import close_worker, watch_worker

import os
from pathlib import Path
import secrets
import select
import socket
import subprocess
import threading
import time
import numpy as np
from whiteboard.ipc import send,receive
from wiping.render_payload import encode
from wiping.render import normals

COLORS=dict(steel=((.43,.46,.48),.28,.9),floor=((.28,.30,.29),.7,0.),
    counter=((.30,.32,.31),.30,.8),charcoal=((.035,.041,.043),.5,0.),
    rubber=((.025,.03,.028),.8,0.),blue_plastic=((.025,.085,.24),.3,0.),
    gray_plastic=((.22,.25,.26),.43,0.),robot=((.7,.72,.73),.35,0.),
    robot_dark=((.02,.028,.034),.35,0.),aluminum=((.6,.64,.65),.23,.92),
    wall=((.63,.65,.61),.85,0.),trim=((.08,.10,.095),.55,.2),
    ceiling=((.80,.81,.79),.8,0.),door=((.17,.22,.23),.42,.25),
    sign_white=((.87,.89,.86),.60,0.),diffuser=((.85,.88,.9),.25,0.),
    polycarbonate=((.50,.62,.61),.18,0.),paper_cup=((.86,.84,.79),.75,0.))

def export(sim,folder):
    import newton
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade,Vt
    from pencap.newton15.render import pose,mesh,camera
    from whiteboard.usd_scene import bevel_box
    from wiping.materials import scanned,planar_uv
    from .graphics import artwork,texture,crease_normals,printed_quad
    from wiping.materials import polycarbonate
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(Path(__file__).with_name('film.mdl'),folder/'film.mdl')
    stage=Usd.Stage.CreateNew(str(folder/'scene.usdc'))
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/Lab').GetPrim());mats={}
    def material(name,color,roughness,metal):
        m=UsdShade.Material.Define(stage,'/Materials/'+name)
        s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/surface');s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,color)))
        s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(roughness))
        s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(metal))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');mats[name]=m;return m
    for name,values in COLORS.items():material(name,*values)
    art=artwork(folder/'textures')
    for name,path in art.items():texture(stage,material(name,(1.,1.,1.),.35,0.),path)
    polycarbonate(stage,mats['polycarbonate'])
    # Archive the optical module alongside this exact scene as well.
    shutil.copy2(Path(__file__).resolve().parents[1]/'wiping/polycarbonate.mdl',folder/'polycarbonate.mdl')
    shader=UsdShade.Shader.Get(stage,'/Materials/polycarbonate/rtx_polycarbonate')
    shader.SetSourceAsset(Sdf.AssetPath(str((folder/'polycarbonate.mdl').resolve())),'mdl')
    scanned(stage,mats['floor'],'anti_skid_tiles')
    # Keep the exported scene replayable without reaching back into a changing
    # asset directory. Geometry is already embedded in this USD scene.
    from wiping.materials import ROOT as material_root
    import json
    asset_sources=json.loads((material_root/'sources.json').read_text())['anti_skid_tiles']
    for field in ('Diffuse','nor_gl','Rough'):
        name=asset_sources['files'][field]['path'];target=folder/'textures'/name
        shutil.copy2(material_root/name,target)
        shader=UsdShade.Shader.Get(stage,'/Materials/floor/scan_'+field)
        shader.GetInput('file').Set(Sdf.AssetPath(str(target.resolve())))
    licenses=folder/'licenses';licenses.mkdir(exist_ok=True)
    robot_source=getattr(sim.assembly,'sources',{}).get('fr3_urdf')
    if robot_source:
        asset_root=Path(robot_source['path']).parents[1]
        for name in ('LICENSE','README.md','package.xml'):
            if (asset_root/name).is_file():shutil.copy2(asset_root/name,licenses/('robot_'+name))
        from robots.catalog import ROOT
        for handle in sim.builder._gar_robots.values():
            shutil.copy2(ROOT/'models'/handle.model_id/'assets.lock.json',licenses/(handle.model_id+'_assets.lock.json'))
    (licenses/'asset_sources.json').write_text(json.dumps(dict(
        floor=asset_sources,robot=robot_source,
        authored='Can artwork, recycling print, WSG mounting, film shaders and scene reconstruction are repository-authored; no downloaded product artwork.'),indent=2)+'\n')
    film=material('film',(.8,.87,.84),.2,0.)
    shader=UsdShade.Shader.Define(stage,'/Materials/film/thin_film')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str((folder/'film.mdl').resolve())),'mdl')
    shader.SetSourceAssetSubIdentifier('film','mdl')
    film.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    b=sim.assembly.builder
    for i,q in enumerate(sim.state_0.body_q.numpy()):pose(UsdGeom.Xform.Define(stage,f'/Lab/body_{i}'),q[:3],q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i]&int(newton.ShapeFlags.VISIBLE):continue
        body,kind,scale=b.shape_body[i],b.shape_type[i],np.asarray(b.shape_scale[i])
        path=f'/Lab/body_{body}/shape_{i}' if body>=0 else f'/Lab/static_{i}'
        source=b.shape_source[i]
        if source is not None and hasattr(source,'vertices'):
            vertices=np.asarray(source.vertices)*scale;g=mesh(stage,path,vertices,source.indices)
            if not b.shape_label[i].startswith(('left/','right/')):crease_normals(g,vertices,source.indices)
        elif kind==newton.GeoType.BOX:g=bevel_box(stage,path,2*scale,min(.001,float(scale.min())*.1))
        elif kind in (newton.GeoType.CYLINDER,newton.GeoType.CAPSULE):
            g=(UsdGeom.Cylinder if kind==newton.GeoType.CYLINDER else UsdGeom.Capsule).Define(stage,path)
            g.CreateRadiusAttr(float(scale[0]));g.CreateHeightAttr(float(2*scale[1]));g.CreateAxisAttr('Z')
        elif kind==newton.GeoType.SPHERE:
            g=UsdGeom.Sphere.Define(stage,path);g.CreateRadiusAttr(float(scale[0]))
        else:raise ValueError('Unsupported visible shape '+b.shape_label[i])
        q=np.asarray(b.shape_transform[i]);pose(g,q[:3],q[3:])
        mat=mats[sim.assembly.materials[i]]
        if b.shape_label[i].startswith(('left/','right/')):mat=material('robot_'+str(i),np.asarray(b.shape_color[i])**2.2,.32,0.)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        if sim.assembly.materials[i]=='floor':planar_uv(g,(0,1),(1.8,1.8))
        if b.shape_label[i] in ('recycling_sign','landfill_sign'):
            center=np.asarray(b.shape_transform[i])[:3];x,y,z=center;hx,hy,hz=scale
            printed_quad(stage,path+'_print',[(x-hx-.00005,y+hy,z-hz),(x-hx-.00005,y-hy,z-hz),
                (x-hx-.00005,y-hy,z+hz),(x-hx-.00005,y+hy,z+hz)],mats[b.shape_label[i]])
    p=sim.state_0.particle_q.numpy()
    for surface in sim.assembly.surfaces:
        q=p[surface['first']:surface['first']+surface['count']];f=surface['triangles']
        g=mesh(stage,'/Lab/'+surface['name'],q,f.ravel());g.CreateDoubleSidedAttr(True)
        g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals(q,f)));g.SetNormalsInterpolation('vertex')
        g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([q.min(0),q.max(0)],np.float32)))
        mat=film
        if surface['kind']=='can':
            crease_normals(g,q,f)
            mat=mats['aluminum']
            uv=np.asarray(surface['label_uv'],np.float32).reshape(-1,2)
            UsdGeom.PrimvarsAPI(g).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(Vt.Vec2fArray.FromNumpy(uv))
            binding=UsdShade.MaterialBindingAPI.Apply(g.GetPrim())
            subset=binding.CreateMaterialBindSubset('printed_body',surface['label_faces'].tolist())
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mats['can_'+str(surface['can_type'])])
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        if surface['kind']=='bag'and sim.config.scene_revision>=5:
            from .bag_appearance import bind
            bind(stage,folder/'printed_bag',q,f,segments=sim.config.bag_segments,
                rows=sim.config.bag_rows,cuff=True)
    y0=-.184 if sim.config.scene_revision>=2 else -.205
    printed_quad(stage,f'/Lab/body_{sim.assembly.frame}/lid_print',[(-.24,y0,1.01606),(.24,y0,1.01606),(.24,-.105,1.01606),(-.24,-.105,1.01606)],mats['lid_label'])
    for i,position in enumerate(((-.6,-.8,2.969),(.7,.6,2.969),(-1.6,.6,2.969))):
        light=UsdLux.RectLight.Define(stage,f'/Lights/panel_{i}')
        light.CreateWidthAttr(.6);light.CreateHeightAttr(1.2);light.CreateIntensityAttr((1600.,1200.,600.)[i])
        light.CreateEnableColorTemperatureAttr(True);light.CreateColorTemperatureAttr(4500.);pose(light,position)
    dome=UsdLux.DomeLight.Define(stage,'/Lights/ambient');dome.CreateIntensityAttr(50.)
    camera(stage,'room',(-2.2,-1.95,2.10),(.02,0,.79),31.)
    camera(stage,'station',(1.45,-1.35,1.65),(.49,0,.60),36.)
    camera(stage,'slot',(.85,-.85,1.65),(.49,0,1.005),48.)
    camera(stage,'tub',(-1.10,-.70,1.6),(-.37,0,.95),40.)
    camera(stage,'bag_bottom',(1.05,-.90,.60),(.49,0,.15),46.)
    from control_cable.cameras import export as mounted_cameras
    mounted_cameras(stage,sim.assembly)
    stage.GetRootLayer().Save();return folder/'scene.usdc'

class Renderer:
    def __init__(self,sim,folder,*,width=1280,height=720,samples=32,scene=None):
        self.sim=sim;self.folder=Path(folder);self.closed=False;self.lock=threading.Lock();self.sequence=0;self.log_position=0
        self.folder.mkdir(parents=True,exist_ok=True)
        scene=Path(scene) if scene is not None else export(sim,self.folder)
        from pxr import Usd,UsdGeom
        from .graphics import CreaseNormals
        stage=Usd.Stage.Open(str(scene.resolve()))
        self.normal_functions={}
        for item in sim.assembly.surfaces:
            g=UsdGeom.Mesh.Get(stage,'/Lab/'+item['name'])
            if g.GetNormalsInterpolation()=='faceVarying':
                self.normal_functions[item['name']]=CreaseNormals(item['triangles'],item['count'])
        executable=os.environ.get('ISAAC_RENDER_PYTHON')
        if not executable or not Path(executable).is_file():raise RuntimeError('ISAAC_RENDER_PYTHON is not configured')
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(240)
        secret=secrets.token_urlsafe(32)
        env=dict(os.environ,WIPE_RENDER_PORT=str(listener.getsockname()[1]),WIPE_RENDER_SECRET=secret,
            WIPE_MINIMAL_RENDER='1',WIPE_TEXTURE_STREAMING='1',WIPE_TEXTURE_BUDGET='.005')
        self.log=(self.folder/'worker.log').open('wb')
        self.process = None
        try:
            self.process=subprocess.Popen([executable,'-m','wiping.render_worker','--scene',str(scene.resolve()),
                '--width',str(width),'--height',str(height),'--samples',str(samples)],cwd=Path(__file__).resolve().parents[1],
                env=worker_env(env),stdout=self.log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.connection,_=listener.accept();self.connection.settimeout(240)
            if not secrets.compare_digest(receive(self.connection).get('secret',''),secret):raise PermissionError('Renderer authentication failed')
            if receive(self.connection).get('phase')!='ready':raise RuntimeError('Renderer startup failed')
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:listener.close()
        watch_worker(self)
    def observe(self,cameras=('room',)):
        with self.lock:
            if self.closed:raise RuntimeError('Renderer is closed')
            s=self.sim;p=s.state_0.particle_q.numpy();meshes={}
            for a in s.assembly.surfaces:
                q=p[a['first']:a['first']+a['count']]
                optical=self.normal_functions[a['name']](q)if a['name']in self.normal_functions else normals(q,a['triangles'])
                meshes['/Lab/'+a['name']]=dict(points=encode(q,'float32'),normals=encode(optical,'float32'))
            self.sequence+=1
            send(self.connection,dict(op='observe',sequence=self.sequence,tick=s.tick,body_poses=s.state_0.body_q.numpy().tolist(),
                cameras=list(cameras),meshes=meshes,material_fields={},details={}))
            deadline=time.monotonic()+240
            while not select.select([self.connection],[],[],1.)[0]:
                self.check_graphics_errors()
                if self.process.poll() is not None:raise RuntimeError('Renderer exited before replying')
                if time.monotonic()>deadline:raise TimeoutError('Renderer timed out')
            result=receive(self.connection);self.check_graphics_errors()
            if not result.get('ok') or result.get('tick')!=s.tick or result.get('sequence')!=self.sequence or result.get('physics_steps')!=0:
                raise RuntimeError('Invalid render response: '+str(result))
            return result
    def check_graphics_errors(self):
        with (self.folder/'worker.log').open(errors='replace') as f:
            f.seek(self.log_position);text=f.read();self.log_position=f.tell()
        if any(x in text for x in ('ERROR_OUT_OF_DEVICE_MEMORY','Failed to execute RenderGraph','HydraEngine::render failed',
                                  'MDLC   comp error:','MdlModule is NOT in the DB','USD_MDL: in LoadModule','Not enough data size for HdInterpolationVertex')):
            raise RuntimeError('GPU rendering error; inspect '+str(self.folder/'worker.log'))
    def close(self):
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

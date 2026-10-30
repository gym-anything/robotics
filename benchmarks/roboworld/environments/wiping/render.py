"""Render-only USD built from exactly the current physical geometry.

Native Newton owns all state; the renderer is paused and returns zero physics
steps. Initial component scene: architectural dressing is added after physics.
"""
from gym_anything_robotics.paths import worker_env
from gym_anything_robotics.render_lifecycle import close_worker, watch_worker

import base64
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import select
import time
import numpy as np
from whiteboard.ipc import send,receive
from .render_payload import encode

COLORS={
    'steel':((.52,.55,.57),.27,.92),'aluminum':((.50,.52,.54),.32,.85),
    'charcoal':((.032,.043,.048),.40,.15),'floor':((.32,.34,.32),.80,0.),
    'robot_white':((.80,.82,.82),.32,0.),'robot_dark':((.025,.03,.035),.42,0.),
    'cloth_0':((.045,.29,.40),.88,0.),'cloth_1':((.60,.23,.025),.88,0.),
    'cloth_2':((.65,.69,.64),.88,0.),
    'pan_white':((.70,.72,.70),.40,0.),'pan_green':((.08,.24,.11),.43,0.),
    'pan_gray':((.13,.15,.16),.45,0.),'scraper_blue':((.025,.11,.28),.35,0.),
    'jug_frosted':((.45,.50,.49),.18,0.),'crumb':((.40,.20,.045),.9,0.),
    'bin_blue':((.018,.10,.19),.5,0.),'bin_gray':((.12,.14,.13),.5,0.),
    'tile_wall':((.46,.49,.46),.35,0.)}


def film_path(surface):
    from pxr import Tf
    parent=f'/Lab/body_{surface.body}' if surface.body>=0 else '/Lab'
    return parent+'/film_'+Tf.MakeValidIdentifier(surface.name)


def normals(points,triangles):
    n=np.zeros_like(points)
    a,b,c=points[triangles[:,0]],points[triangles[:,1]],points[triangles[:,2]]
    t=np.cross(b-a,c-a)
    for i in range(3):
        np.add.at(n,triangles[:,i],t)
    lengths=np.linalg.norm(n,axis=1)
    n/=np.maximum(lengths[:,None],1e-20)
    return n.astype(np.float32)


def export(sim,folder):
    # Published revision-four appearance; explicit edition overrides still win.
    # Each environment server owns its process. These are render settings only.
    if sim.config.scene_revision >= 4:
        for key,value in dict(WIPE_CERAMIC_WALL='1', WIPE_BRUSHED_SURFACES='1',
                WIPE_PAGED_FILM='1', WIPE_POLYCARBONATE='1', WIPE_FIXTURE_FINISH='1',
                WIPE_SPECIES_OPTICS='1', WIPE_LAYERED_COAT='1', WIPE_RESIDUE_DETAIL='1', WIPE_MINIMAL_RENDER='1',
                WIPE_TEXTURE_STREAMING='1', WIPE_TEXTURE_BUDGET='.005').items():
            os.environ.setdefault(key,value)
    import newton
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade,Vt
    from pencap.newton15.render import pose,mesh,camera
    from whiteboard.usd_scene import bevel_box
    from .optics import vertex_material,cloth_color,film_arrays,runoff_arrays
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(folder/'scene.usdc'))
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/Lab').GetPrim())
    mats={}
    def material(name,rgb,rough,metal):
        m=UsdShade.Material.Define(stage,'/Materials/'+name)
        s=UsdShade.Shader.Define(stage,'/Materials/'+name+'/surface')
        s.CreateIdAttr('UsdPreviewSurface')
        s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,rgb)))
        s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough)
        s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
        mats[name]=m
        return m
    for name,values in COLORS.items():
        material(name,*values)
    from .materials import scanned,planar_uv,textile,state_surface
    # Coating and bare substrate share the same BRDF. Otherwise equal low-mass
    # boundary colors can still produce a visible shader-conversion halo.
    for name in ('steel','aluminum','pan_white','pan_green','pan_gray','scraper_blue','bin_blue','bin_gray'):
        state_surface(stage,mats[name],COLORS[name])
    scan_textures=os.environ.get('WIPE_DIAGNOSTIC_NO_SCANS')!='1'
    ceramic_wall=os.environ.get('WIPE_CERAMIC_WALL')=='1'
    if scan_textures:
        scanned(stage,mats['floor'],'anti_skid_tiles')
        scanned(stage,mats['tile_wall'],'ceramic_tiles_107' if ceramic_wall else 'long_white_tiles')
    if sim.config.scene_revision>=1:
        plastic=UsdShade.Shader.Get(stage,'/Materials/jug_frosted/surface')
        plastic.GetInput('diffuseColor').Set(Gf.Vec3f(.78,.85,.86))
        plastic.GetInput('roughness').Set(.12)
        plastic.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(.22)
        plastic.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.586)
        if os.environ.get('WIPE_POLYCARBONATE')=='1':
            from .materials import polycarbonate
            polycarbonate(stage,mats['jug_frosted'])
    b=sim.assembly.builder
    for i,q in enumerate(sim.state_0.body_q.numpy()):
        pose(UsdGeom.Xform.Define(stage,f'/Lab/body_{i}'),q[:3],q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i]&int(newton.ShapeFlags.VISIBLE):continue
        body,kind,scale=b.shape_body[i],b.shape_type[i],np.asarray(b.shape_scale[i])
        path=f'/Lab/body_{body}/shape_{i}' if body>=0 else f'/Lab/static_{i}'
        source=b.shape_source[i]
        if b.shape_label[i]=='blender_jug' and sim.config.scene_revision>=1:
            from .kitchen import jug_geometry
            v,f=jug_geometry(hollow=True);g=mesh(stage,path,v,f)
            g.CreateDoubleSidedAttr(True)
        elif source is not None and hasattr(source,'vertices'):
            g=mesh(stage,path,np.asarray(source.vertices)*scale,source.indices)
        elif kind==newton.GeoType.BOX:
            g=bevel_box(stage,path,2*scale,min(.001,float(scale.min())*.1))
        elif kind==newton.GeoType.PLANE:
            g=bevel_box(stage,path,(7.,7.,.10),.002)
        elif kind==newton.GeoType.SPHERE:
            g=UsdGeom.Sphere.Define(stage,path);g.CreateRadiusAttr(float(scale[0]))
        elif kind in (newton.GeoType.CYLINDER,newton.GeoType.CAPSULE):
            g=(UsdGeom.Cylinder if kind==newton.GeoType.CYLINDER else UsdGeom.Capsule).Define(stage,path)
            g.CreateRadiusAttr(float(scale[0]));g.CreateHeightAttr(float(2*scale[1]));g.CreateAxisAttr('Z')
        else:raise ValueError(f'Unsupported visible shape {b.shape_label[i]}')
        local=np.asarray(b.shape_transform[i]).copy()
        if kind==newton.GeoType.PLANE:local[2]-=.05
        pose(g,local[:3],local[3:])
        mat=mats[sim.assembly.materials[i]]
        if b.shape_label[i].startswith(('left/','right/')):
            mat=material('fr3_'+str(i),np.asarray(b.shape_color[i])**2.2,.35,0.)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        if sim.assembly.materials[i]=='floor':planar_uv(g,(0,1),(1.8,1.8))
        elif b.shape_label[i]=='back_wall':planar_uv(g,(2,1),(1.,1.) if ceramic_wall else (1.27,1.27))
        elif b.shape_label[i]=='side_wall':planar_uv(g,(2,0),(1.,1.) if ceramic_wall else (1.27,1.27))
    positions=sim.state_0.particle_q.numpy()
    for cloth in sim.assembly.cloths:
        first,count=cloth['first'],cloth['count']
        points=positions[first:first+count]
        triangles=cloth['triangles'];material_uv=cloth['uv'].astype(np.float32)
        color=cloth_color(sim,cloth,COLORS[cloth['name']][0])
        if sim.config.scene_revision>=3:
            from .cloth_skin import geometry as skin_geometry
            points,triangles=skin_geometry(points,triangles,sim.config.cloth_radius)
            material_uv=np.tile(material_uv,(2,1));color=np.tile(color,(2,1))
        path='/Lab/'+cloth['name']
        g=mesh(stage,path,points,triangles.ravel())
        g.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals(points,triangles)))
        g.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray([points.min(0),points.max(0)],np.float32)))
        g.CreateDoubleSidedAttr(True)
        uv=UsdGeom.PrimvarsAPI(g).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        # Poly Haven scan covers367x363mm: map actual material coordinates,
        # rather than the former arbitrarily repeated16x generic cotton normal.
        uv.Set(Vt.Vec2fArray.FromNumpy(material_uv/np.array([.367054,.3632],np.float32)))
        vertex_material(stage,mats[cloth['name']],textile=True)
        if scan_textures:
            scanned(stage,mats[cloth['name']],'terry_cloth',diffuse=False,normal_strength=.65)
            textile(stage,mats[cloth['name']],sim.config.cloth_side)
        g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(
            color))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[cloth['name']])
    if sim.transport is not None:
        mat=material('film',(.2,.1,.04),.20,0.)
        vertex_material(stage,mat,film=True)
        state_surface(stage,mat)
        paged_film=os.environ.get('WIPE_PAGED_FILM')=='1'
        if paged_film:
            from .film_render import groups,author
            author(stage,groups(sim.transport))
        for i,s in ([] if paged_film else enumerate(sim.transport.surfaces)):
            data=film_arrays(sim.transport,i)
            parent=f'/Lab/body_{s.body}' if s.body>=0 else '/Lab'
            g=mesh(stage,film_path(s),data['points'],data['triangles'].ravel())
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(data['normals']))
            g.SetNormalsInterpolation('vertex')
            g.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray([data['points'].min(0),data['points'].max(0)],np.float32)))
            g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(data['color']))
            for field in ('roughness','metallic','coat_weight','food_coverage'):
                if field not in data:continue
                UsdGeom.PrimvarsAPI(g).CreatePrimvar(field,Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(data[field]))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        if sim.transport.flow is not None:
            drop=material('runoff',(.35,.42,.44),.14,0.);vertex_material(stage,drop)
            from .runoff_render import author
            author(stage,runoff_arrays(sim))
    for i,position in enumerate(((-.3,-.7,2.8),(.4,.8,2.8),(-1.,0,2.2))):
        light=UsdLux.RectLight.Define(stage,f'/Lights/panel_{i}')
        light.CreateWidthAttr(.6);light.CreateHeightAttr(1.2);light.CreateIntensityAttr((2200.,1300.,700.)[i])
        light.CreateEnableColorTemperatureAttr(True);light.CreateColorTemperatureAttr((5600.,5600.,5600.)[i])
        pose(light,position)
    light=UsdLux.DomeLight.Define(stage,'/Lights/ambient');light.CreateIntensityAttr(65.)
    if sim.config.scene_revision>=2:
        camera(stage,'room',(-2.05,-1.85,2.15),(0,.22,.80),27.)
    else:camera(stage,'room',(-1.9,-1.7,2.1),(0,0,.90),31.)
    camera(stage,'bench',(-1.25,-.90,1.7),(0,-.03,.91),36.)
    camera(stage,'contact',(-.52,-.61,1.17),(0,-.25,.94),42.)
    camera(stage,'grasp',(-.60,-.80,1.60),(0,-.325,1.26),43.)
    camera(stage,'grasp_side',(-.85,-.20,1.44),(0,-.325,1.25),45.)
    camera(stage,'overhead',(-.001,0,2.3),(0,0,.90),26.)
    camera(stage,'wipe',(-.75,-.65,1.65),(.02,0,.93),43.)
    # Offline diagnostic composition: view from the opposite side of the
    # working forearm and retain the raised cloth as well as the worktop.
    # Mounted policy cameras and physical state are unchanged.
    camera(stage,'wipe_reveal',(-1.20,.38,1.70),(-.10,-.015,1.10),24.)
    # Keep the complete raised hand and hanging cloth in the frame.
    # Diagnostic view only; policy-mounted camera calibration is unchanged.
    camera(stage,'wipe_action',(-1.20,.38,1.70),(-.10,-.015,1.23),24.)
    # Component view: show the actual floor-mounted used-cloth bin interior.
    # This is a diagnostic view, not a replacement for the mounted policy cameras.
    camera(stage,'disposal',(-1.18,-1.68,1.20),(-.55,-1.0,.36),38.)
    camera(stage,'textile',(-.84,.26,.995),(-.73,.5,.86),43.)
    camera(stage,'fixtures',(.10,.20,1.72),(1.25,1.16,1.18),36.)
    from control_cable.cameras import export as mounted_cameras
    mounted_cameras(stage,sim.assembly)
    from .appearance import finish
    finish(stage,sim,material,mats)
    stage.GetRootLayer().Save()
    return folder/'scene.usdc'


class Renderer:
    def __init__(self,sim,folder,*,width=1280,height=720,samples=16):
        self.sim=sim
        self.folder=Path(folder)
        scene=export(sim,self.folder)
        self.paged_film=os.environ.get('WIPE_PAGED_FILM')=='1'
        # Rendering editions are explicit: a new finish may re-render identical
        # recorded states, but must never imply that the physical run changed.
        import json,hashlib
        from .provenance import geometry_sha256
        files=('render.py','materials.py','appearance.py','state_surface.mdl','microfiber.mdl','polycarbonate.mdl',
               'render_worker.py','render_updates.py','optics.py','species_optics.py','film_render.py')
        (self.folder/'render_manifest.json').write_text(json.dumps(dict(
            width=width,height=height,samples=samples,geometry_sha256=geometry_sha256(sim.assembly),
            source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files},
            render_options={k:v for k,v in os.environ.items() if k in (
                'WIPE_CERAMIC_WALL','WIPE_BRUSHED_SURFACES','WIPE_PAGED_FILM','WIPE_POLYCARBONATE',
                'WIPE_MINIMAL_RENDER','WIPE_TEXTURE_STREAMING','WIPE_TEXTURE_BUDGET','WIPE_DIAGNOSTIC_NO_SCANS',
                'WIPE_FIXTURE_FINISH','WIPE_SPECIES_OPTICS','WIPE_LAYERED_COAT','WIPE_RESIDUE_DETAIL')},
            wall_source='https://ambientcg.com/view?id=Tiles107' if os.environ.get('WIPE_CERAMIC_WALL')=='1' else 'https://polyhaven.com/a/long_white_tiles'
        ),indent=2)+'\n')
        if getattr(sim,'render_only',False) and not getattr(sim,'render_only_cpu',False):
            # An offline state replay never performs collision queries. Keep
            # exact mesh vertices/poses, release ONLY its unused SDF GPU caches
            # before launching RTX. Live simulations must retain their SDFs.
            import gc,warp as wp
            for source in sim.assembly.builder.shape_source:
                if source is not None and hasattr(source,'clear_sdf'):source.clear_sdf()
            gc.collect();wp.set_mempool_release_threshold(sim.config.device,0)
            wp.synchronize_device(sim.config.device)
        executable=os.environ.get('ISAAC_RENDER_PYTHON') or os.environ.get('LAUNDRY_RENDER_PYTHON')
        if not executable or not Path(executable).is_file():
            raise RuntimeError('Set ISAAC_RENDER_PYTHON to the installed render interpreter')
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(240)
        secret=secrets.token_urlsafe(32)
        env=dict(os.environ,WIPE_RENDER_PORT=str(listener.getsockname()[1]),WIPE_RENDER_SECRET=secret)
        self.log=(self.folder/'worker.log').open('wb')
        self.process = None
        try:
            self.process=subprocess.Popen([executable,'-m','wiping.render_worker',
                '--scene',str(scene.resolve()),'--width',str(width),'--height',str(height),'--samples',str(samples)],
                cwd=Path(__file__).resolve().parents[1],env=worker_env(env),stdout=self.log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.connection,_=listener.accept();self.connection.settimeout(240)
            if not secrets.compare_digest(receive(self.connection).get('secret',''),secret):
                raise PermissionError('Renderer authentication failed')
            if receive(self.connection).get('phase')!='ready':raise RuntimeError('Renderer startup failed')
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:listener.close()
        watch_worker(self)
        self.sequence=0;self.closed=False;self.lock=threading.Lock()
        self.log_position=0

    def check_graphics_errors(self):
        # A nonempty/stale RGB array is not success after a failed GPU draw.
        with (self.folder/'worker.log').open('r',errors='replace') as stream:
            stream.seek(self.log_position);lines=stream.read();self.log_position=stream.tell()
        fatal=('ERROR_OUT_OF_DEVICE_MEMORY','Failed to execute RenderGraph','HydraEngine::render failed',
               'MDLC   comp error:','MdlModule is NOT in the DB','USD_MDL: in LoadModule',
               'Not enough data size for HdInterpolationVertex')
        if any(message in lines for message in fatal):
            raise RuntimeError(f'GPU rendering failed; rejecting RGB. Inspect {self.folder / "worker.log"}')

    def observe(self,cameras=('bench',)):
        with self.lock:
            if self.closed:raise RuntimeError('Renderer closed')
            from .optics import cloth_color,film_arrays,runoff_arrays
            sim=self.sim;p=sim.state_0.particle_q.numpy();meshes={};material_fields={}
            for cloth in sim.assembly.cloths:
                points=p[cloth['first']:cloth['first']+cloth['count']]
                triangles=cloth['triangles'];color=cloth_color(sim,cloth,COLORS[cloth['name']][0])
                if sim.config.scene_revision>=3:
                    from .cloth_skin import geometry as skin_geometry
                    points,triangles=skin_geometry(points,triangles,sim.config.cloth_radius)
                    color=np.tile(color,(2,1))
                meshes['/Lab/'+cloth['name']]=dict(points=encode(points,'float32'),
                    normals=encode(normals(points,triangles),'float32'))
                material_fields['/Lab/'+cloth['name']]=dict(color=encode(color,'float32'))
            film_groups=None
            if sim.transport is not None:
                if self.paged_film:
                    from .film_render import groups
                    film_groups={body:{key:encode(value,'float32') for key,value in fields.items()}
                                 for body,fields in groups(sim.transport).items()}
                for i,s in ([] if self.paged_film else enumerate(sim.transport.surfaces)):
                    parent=f'/Lab/body_{s.body}' if s.body>=0 else '/Lab'
                    path=film_path(s);data=film_arrays(sim.transport,i)
                    meshes[path]=dict(points=encode(data['points'],'float32'),normals=encode(data['normals'],'float32'),
                                      triangles=encode(data['triangles'],'int32'))
                    material_fields[path]={key:encode(data[key],'float32') for key in
                        ('color','roughness','metallic','coat_weight','food_coverage') if key in data}
            runoff={}
            if sim.transport is not None and sim.transport.flow is not None:
                data=runoff_arrays(sim);runoff={k:encode(v,'float32') for k,v in data.items()}
            self.sequence+=1
            send(self.connection,dict(op='observe',sequence=self.sequence,tick=sim.tick,
                body_poses=sim.state_0.body_q.numpy().tolist(),cameras=list(cameras),meshes=meshes,
                material_fields=material_fields,details={},pile={},runoff=runoff,
                **({'film_groups':film_groups} if film_groups is not None else {})))
            deadline=time.monotonic()+240
            while not select.select([self.connection],[],[],1.)[0]:
                self.check_graphics_errors()
                if self.process.poll() is not None:raise RuntimeError('Renderer process exited before replying')
                if time.monotonic()>deadline:raise TimeoutError('Renderer observation timed out')
            result=receive(self.connection)
            self.check_graphics_errors()
            if not result.get('ok') or result.get('sequence')!=self.sequence or result.get('tick')!=sim.tick or result.get('physics_steps')!=0:
                raise RuntimeError(f'Invalid render reply: {result}')
            return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

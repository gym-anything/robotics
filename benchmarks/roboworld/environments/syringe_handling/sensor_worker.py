"""Persistent CPU RGB sensor. Exact authored rigid geometry; explicit primitive liquid surfaces.

This software EGL rasterizer does no physics and receives no task success labels.
Isaac remains available for the authored material/texture renderer.
"""
import os
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBGL_ALWAYS_SOFTWARE='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
import argparse,base64,io,json,socket,time,traceback
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from PIL import Image
from whiteboard.ipc import send,receive
from gym_anything_robotics.render_lifecycle import watch_parent
from .sensor_geometry import draw


def main():
    watch_parent();ap=argparse.ArgumentParser();ap.add_argument('--scene');ap.add_argument('--width',type=int);ap.add_argument('--height',type=int);ap.add_argument('--samples',type=int);a=ap.parse_args()
    m=mujoco.MjModel.from_binary_path(a.scene);d=mujoco.MjData(m);meta=json.loads(Path(a.scene).with_suffix('.json').read_text());m.geom_matid[:]=-1
    for g,rgba in meta['colours'].items():m.geom_rgba[int(g)]=rgba
    m.vis.global_.offwidth=max(a.width,1920);m.vis.global_.offheight=max(a.height,1080);m.vis.map.znear=.0002
    m.vis.headlight.ambient[:]=.45;m.vis.headlight.diffuse[:]=.5;m.vis.headlight.specular[:]=.15;m.vis.headlight.active=1
    conn=socket.create_connection(('127.0.0.1',int(os.environ['SYRINGE_RENDER_PORT'])));send(conn,{'secret':os.environ.pop('SYRINGE_RENDER_SECRET')})
    renders={};send(conn,{'phase':'ready'})
    # Older saved models also identify the pad by its physical body name.
    pad=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pad')
    if pad>=0:meta['pad_body']=meta['body_map'].get(str(pad))
    while True:
        req=receive(conn)
        if req['op']=='close':break
        try:
            start=time.perf_counter();bq=np.array(req['body_poses']);rot=Rotation.from_quat(bq[:,3:]).as_matrix()
            for body,newton in meta['body_map'].items():
                body=int(body);d.xpos[body]=bq[newton,:3];d.xmat[body]=rot[newton].ravel()
            for g in range(m.ngeom):
                b=int(m.geom_bodyid[g]);R=d.xmat[b].reshape(3,3) if b else np.eye(3);p=d.xpos[b] if b else np.zeros(3)
                d.geom_xpos[g]=p+R@m.geom_pos[g];d.geom_xmat[g]=(R@Rotation.from_quat(np.roll(m.geom_quat[g],-1)).as_matrix()).ravel()
            images={}
            for name in req['cameras']:
                size=tuple(req['resolutions'][name]);r=renders.get(size)
                if r is None:r=mujoco.Renderer(m,height=size[1],width=size[0],max_geom=20000);renders[size]=r
                spec=meta['cameras'][name];eye,target,up=map(np.array,(spec['eye'],spec['target'],spec['up']));body=spec['body']
                if body>=0:eye=bq[body,:3]+rot[body]@eye;target=bq[body,:3]+rot[body]@target;up=rot[body]@up
                m.vis.global_.fovy=np.degrees(2*np.arctan(np.tan(np.radians(spec['fov'])/2)*size[1]/size[0]))
                r.update_scene(d)
                forward=target-eye;forward/=np.linalg.norm(forward);up=up-forward*(up@forward);up/=np.linalg.norm(up)
                for cam in r.scene.camera:cam.pos=eye;cam.forward=forward;cam.up=up
                fluid=req.get('liquid',{})
                draw(r,fluid,bq,rot,meta)
                im=r.render();buf=io.BytesIO();Image.fromarray(im).save(buf,format='PNG',compress_level=1);images[name]=base64.b64encode(buf.getvalue()).decode()
            send(conn,dict(ok=True,tick=req['tick'],physics_steps=0,images=images,timings=dict(total_s=time.perf_counter()-start)))
        except Exception as e:traceback.print_exc();send(conn,dict(ok=False,error=str(e)))
    for r in renders.values():r.close()
    conn.close()
if __name__=='__main__':main()

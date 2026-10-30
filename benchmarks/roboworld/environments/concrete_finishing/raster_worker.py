"""CPU RGB rasterizer of the exported USD scene and exact recorded paste meshes.

Mesa software EGL, no physics, no CUDA. Keeps the room, catalog visual meshes,
colours and diffuse textures. Replaces MDL/path-traced lighting with documented
ambient/directional/specular shading; not a second geometry or material state.
"""
import os
os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', LIBGL_ALWAYS_SOFTWARE='1',
                  __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
import argparse, base64, io, json, socket, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

VERT = '''#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
layout(location=3) in vec2 uv;
layout(location=4) in float gloss;
layout(location=5) in float textureAmplitude;
uniform mat4 model, view, projection;
out vec3 N, C, P; out vec2 UV; out float G, A;
void main(){vec4 p=model*vec4(position,1);P=p.xyz;N=mat3(transpose(inverse(model)))*normal;
 C=color;UV=uv;G=gloss;A=textureAmplitude;gl_Position=projection*view*p;}
'''
FRAG = '''#version 330 core
in vec3 N,C,P;in vec2 UV;in float G,A;
uniform sampler2D diffuse;uniform int textured;uniform vec3 eye;
out vec4 outputColor;
float grain(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
vec3 grainGradient(vec3 p){
 vec3 i=floor(p),f=fract(p),u=f*f*(3.-2.*f),d=6.*f*(1.-f);
 float a=grain(i),b=grain(i+vec3(1,0,0)),c=grain(i+vec3(0,1,0)),e=grain(i+vec3(1,1,0));
 float g=grain(i+vec3(0,0,1)),h=grain(i+vec3(1,0,1)),j=grain(i+vec3(0,1,1)),k=grain(i+vec3(1,1,1));
 return d*vec3(mix(mix(b-a,e-c,u.y),mix(h-g,k-j,u.y),u.z),
               mix(mix(c-a,e-b,u.x),mix(j-g,k-h,u.x),u.z),
               mix(mix(g-a,h-b,u.x),mix(j-c,k-e,u.x),u.y));
}
void main(){vec3 n=normalize(N);if(!gl_FrontFacing)n=-n;
 // Texture attribute is the existing material appearance mapping: one unit
 // corresponds to 1 mm amplitude (sealant uses its documented half response).
 // Fixed world-space grains represent unresolved grains, not extra paste volume.
 // Derivative filtering removes grains smaller than two pixels; residual slope
 // still broadens the highlight. This is an optical approximation, not metrology.
 if(A>0.){
 vec3 p1=P/.003,p2=P/.0013;
 float f1=1.-smoothstep(.3,.65,max(length(dFdx(p1)),length(dFdy(p1))));
 float f2=1.-smoothstep(.3,.65,max(length(dFdx(p2)),length(dFdy(p2))));
 // Unit uniform vertex noise has sigma sqrt((26/35)^3/12)=0.1848
 // after trilinear smoothstep interpolation. The two-scale 0.7/0.3 mix
 // has sigma ~0.1408: normalize so the mapped amplitude denotes RMS
 // microrelief, rather than only ~14% of the requested height variation.
 vec3 grad=A*.001/0.1408*(.7*grainGradient(p1)/.003*f1+.3*grainGradient(p2)/.0013*f2);
 n=normalize(n-(grad-n*dot(grad,n)));
 }
 float smoothGloss=G/(1.+8.*A*A);
 vec3 l=normalize(vec3(.35,-.45,1.0));vec3 v=normalize(eye-P);
 vec3 albedo=C;if(textured==1)albedo*=pow(texture(diffuse,UV).rgb,vec3(2.2));
 // Broad ceiling/work-light directions illuminate horizontal and vertical
 // stations; one overhead vector left the east joint in textureless ambient.
 vec3 l2=normalize(vec3(-.65,.25,.6)),l3=normalize(vec3(.15,.8,.7));
 float lighting=.30+.35*max(dot(n,l),0.)+.25*max(dot(n,l2),0.)+.10*max(dot(n,l3),0.);
 float spec=(.22*pow(max(dot(n,normalize(l+v)),0.),mix(12.,100.,smoothGloss))+
             .13*pow(max(dot(n,normalize(l2+v)),0.),mix(12.,100.,smoothGloss)))*smoothGloss;
 vec3 rgb=albedo*lighting+vec3(spec);
 outputColor=vec4(pow(clamp(rgb,0.,1.),vec3(1./2.2)),1.);}
'''


class Raster:
    def __init__(self, path, width=1920, height=1080):
        import mujoco
        from OpenGL import GL as gl
        from OpenGL.GL.shaders import compileProgram, compileShader
        from pxr import Usd, UsdGeom, UsdShade
        self.gl=gl;self.context=mujoco.GLContext(width,height);self.context.make_current()
        self.framebuffer=gl.glGenFramebuffers(1);gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,self.framebuffer)
        self.buffers=[]
        for fmt,attachment in ((gl.GL_RGBA8,gl.GL_COLOR_ATTACHMENT0),(gl.GL_DEPTH_COMPONENT24,gl.GL_DEPTH_ATTACHMENT)):
            rb=gl.glGenRenderbuffers(1);self.buffers.append(rb);gl.glBindRenderbuffer(gl.GL_RENDERBUFFER,rb)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER,fmt,width,height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER,attachment,gl.GL_RENDERBUFFER,rb)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)!=gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('software framebuffer incomplete')
        self.stage=Usd.Stage.Open(str(path));self.path=Path(path)
        self.program=compileProgram(compileShader(VERT,gl.GL_VERTEX_SHADER),compileShader(FRAG,gl.GL_FRAGMENT_SHADER))
        self.uniform={n:gl.glGetUniformLocation(self.program,n) for n in ('model','view','projection','diffuse','textured','eye')}
        self.meshes=[];self.dynamic={};self.textures={};self.cameras={}
        self.driver=gl.glGetString(gl.GL_RENDERER).decode()
        cache=UsdGeom.XformCache()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                self.cameras[str(prim.GetPath())]=prim
            if str(prim.GetPath()).startswith('/Paste'):
                continue
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule)):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':continue
            world=np.array(cache.GetLocalToWorldTransform(prim)).T
            body=None;local=world
            parts=str(prim.GetPath()).split('/')
            for i,x in enumerate(parts):
                if x.startswith('body_') and x[5:].isdigit():
                    body=int(x[5:]);bp=self.stage.GetPrimAtPath('/'.join(parts[:i+1]));bw=np.array(cache.GetLocalToWorldTransform(bp)).T
                    local=np.linalg.inv(bw)@world;break
            if prim.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(prim);v=np.array(mesh.GetPointsAttr().Get(),np.float32)
                if not len(v):continue
                counts=np.array(mesh.GetFaceVertexCountsAttr().Get(),int);indices=np.array(mesh.GetFaceVertexIndicesAttr().Get(),int)
                normals=np.array(mesh.GetNormalsAttr().Get() or [],np.float32)
                uvp=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st');uv=np.array(uvp.ComputeFlattened() if uvp else [],np.float32)
                tris=[];corners=[];face_ids=[];off=0
                for face,count in enumerate(counts):
                    for k in range(1,count-1):
                        corner=[off,off+k,off+k+1];tris.append(indices[corner]);corners.append(corner);face_ids.append(face)
                    off+=count
                tri=np.asarray(tris,int);corner=np.asarray(corners,int);face_ids=np.asarray(face_ids,int)
                if not len(tri):continue
                def expand(attr,interp):
                    if not len(attr):return None
                    if interp=='faceVarying':return attr[corner]
                    if interp=='uniform':return attr[face_ids,None,:].repeat(3,axis=1)
                    if interp=='constant':return np.broadcast_to(attr[0],tri.shape+attr.shape[1:])
                    return attr[tri]
                nr=expand(normals,str(mesh.GetNormalsInterpolation()));uvs=expand(uv,str(uvp.GetInterpolation()) if uvp else '')
                if nr is None:
                    n=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
                    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-15);nr=n[:,None,:].repeat(3,axis=1)
                if uvs is None:uvs=np.zeros(tri.shape+(2,),np.float32)
                assignments=np.full(len(counts),-1,int);subsets=UsdGeom.Subset.GetAllGeomSubsets(mesh)
                for k,sub in enumerate(subsets):assignments[np.array(sub.GetIndicesAttr().Get(),int)]=k
                for k in np.unique(assignments[face_ids]):
                    on=assignments[face_ids]==k;bound=subsets[k].GetPrim() if k>=0 else prim
                    col,gloss,texture,scale=self.material(bound)
                    data=self.pack(v[tri[on]],nr[on],col,uvs[on]*scale,gloss)
                    self.meshes.append((self.upload(data),local,body,texture,str(prim.GetPath())))
            else:
                import trimesh
                cylinder=UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
                r=float(cylinder.GetRadiusAttr().Get());h=float(cylinder.GetHeightAttr().Get())
                mesh=trimesh.creation.cylinder(r,h,sections=20) if prim.IsA(UsdGeom.Cylinder) else trimesh.creation.capsule(h,r,count=[12,20])
                v=np.asarray(mesh.vertices);nr=np.asarray(mesh.vertex_normals);tri=np.asarray(mesh.faces)
                axis=str(cylinder.GetAxisAttr().Get())
                turn=Rotation.from_euler('y',np.pi/2).as_matrix() if axis=='X' else (Rotation.from_euler('x',-np.pi/2).as_matrix() if axis=='Y' else np.eye(3))
                col,gloss,texture,scale=self.material(prim)
                data=self.pack((v@turn.T)[tri],(nr@turn.T)[tri],col,np.zeros(tri.shape+(2,)),gloss)
                self.meshes.append((self.upload(data),local,body,texture,str(prim.GetPath())))
        gl.glEnable(gl.GL_DEPTH_TEST);gl.glDisable(gl.GL_CULL_FACE)
        print(json.dumps(dict(renderer=self.driver,meshes=len(self.meshes),triangles=sum(m[0][2]//3 for m in self.meshes))),flush=True)

    def material(self, prim):
        from pxr import UsdShade
        from PIL import Image
        gl=self.gl;mat=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        col=np.array([.6,.6,.6]);gloss=.1;tex=0;scale=np.ones(2)
        if mat:
            shaders=[x for x in mat.GetPrim().GetChildren() if x.IsA(UsdShade.Shader)]
            if shaders:
                sh=UsdShade.Shader(shaders[0])
                def get(n,default):
                    i=sh.GetInput(n);return i.Get() if i and i.Get() is not None else default
                col=np.array(get('diffuse_color_constant',[1,1,1]),float)
                gloss=1-float(get('reflection_roughness_constant',.8))
                scale=np.array(get('texture_scale',[1,1]))
                asset=get('diffuse_texture',None)
                if asset:
                    path=asset.resolvedPath or str(self.path.parent/asset.path)
                    if path not in self.textures:
                        img=Image.open(path).convert('RGB');img.thumbnail((1024,1024));arr=np.asarray(img)[::-1].copy()
                        tex=gl.glGenTextures(1);gl.glBindTexture(gl.GL_TEXTURE_2D,tex);gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
                        gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGB,arr.shape[1],arr.shape[0],0,gl.GL_RGB,gl.GL_UNSIGNED_BYTE,arr)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER,gl.GL_LINEAR_MIPMAP_LINEAR);gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MAG_FILTER,gl.GL_LINEAR)
                        gl.glGenerateMipmap(gl.GL_TEXTURE_2D);self.textures[path]=tex
                    tex=self.textures[path]
        return col,gloss,tex,scale

    def pack(self, v,n,c,uv,g,a=0.):
        shape=v.shape[:-1]
        return np.concatenate([v,n,np.broadcast_to(c,v.shape),uv,np.broadcast_to(g,shape)[...,None],np.broadcast_to(a,shape)[...,None]],axis=-1).reshape(-1,13).astype(np.float32)

    def upload(self,data):
        import ctypes
        gl=self.gl;vao=gl.glGenVertexArrays(1);vbo=gl.glGenBuffers(1);gl.glBindVertexArray(vao);gl.glBindBuffer(gl.GL_ARRAY_BUFFER,vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,data.nbytes,data,gl.GL_STATIC_DRAW)
        off=0
        for loc,size in enumerate((3,3,3,2,1,1)):
            gl.glEnableVertexAttribArray(loc);gl.glVertexAttribPointer(loc,size,gl.GL_FLOAT,False,52,ctypes.c_void_p(off*4));off+=size
        return vao,vbo,len(data)

    def apply(self,req):
        from pxr import UsdGeom
        self.poses=np.asarray(req['body_poses'],float)
        if not np.isfinite(self.poses).all():raise ValueError('nonfinite body pose')
        self.transforms=np.tile(np.eye(4),(len(self.poses),1,1));self.transforms[:,:3,:3]=Rotation.from_quat(self.poses[:,3:]).as_matrix();self.transforms[:,:3,3]=self.poses[:,:3]
        requested=req.get('meshes',{})
        for name in set(self.dynamic)-set(requested):self.delete(self.dynamic.pop(name))
        for name,m in requested.items():
            if m.get('same'):
                if name not in self.dynamic:raise ValueError('unknown unchanged mesh')
                continue
            def array(k,dt,n):return np.frombuffer(base64.b64decode(m[k]),dt).reshape(-1,n)
            v=array('v',np.float32,3);t=array('t',np.int32,3);n=array('n',np.float32,3);c=array('c',np.float32,3);g=array('g',np.float32,1)[:,0]
            if not np.isfinite(v).all() or t.min()<0 or t.max()>=len(v):raise ValueError('invalid dynamic mesh')
            a=array('x',np.float32,1)[:,0]
            data=self.pack(v[t],n[t],c[t],np.zeros(t.shape+(2,)),g[t],a[t])
            if name in self.dynamic:self.delete(self.dynamic.pop(name))
            self.dynamic[name]=self.upload(data)

    def delete(self,mesh):
        self.gl.glDeleteVertexArrays(1,[mesh[0]]);self.gl.glDeleteBuffers(1,[mesh[1]])

    def camera(self,name):
        from pxr import UsdGeom
        alias=self.stage.GetPrimAtPath('/Cameras/'+name)
        path=alias.GetCustomDataByKey('mounted_camera_path') if alias else None
        prim=self.cameras.get(path or '/Cameras/'+name)
        if prim is None:raise ValueError('unknown camera '+name)
        matrix=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            x=parent.GetName()
            if x.startswith('body_') and x[5:].isdigit():
                initial=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(parent)).T
                matrix=self.transforms[int(x[5:])]@np.linalg.inv(initial)@matrix;break
            parent = parent.GetParent()
        cam=UsdGeom.Camera(prim)
        hfov=2*np.arctan(float(cam.GetHorizontalApertureAttr().Get())/(2*float(cam.GetFocalLengthAttr().Get())))
        return matrix,hfov

    def capture(self,name,size):
        from PIL import Image
        gl=self.gl;width,height=size;matrix,hfov=self.camera(name)
        near,far=.002,30.;sx=1/np.tan(hfov/2);sy=sx*width/height
        proj=np.array([[sx,0,0,0],[0,sy,0,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]],np.float32)
        gl.glViewport(0,0,width,height);gl.glClearColor(.16,.17,.19,1);gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT);gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.uniform['view'],1,True,np.linalg.inv(matrix).astype(np.float32));gl.glUniformMatrix4fv(self.uniform['projection'],1,True,proj)
        gl.glUniform3fv(self.uniform['eye'],1,matrix[:3,3].astype(np.float32));gl.glUniform1i(self.uniform['diffuse'],0)
        for mesh,local,body,texture,path in self.meshes:
            model=local if body is None else self.transforms[body]@local
            self.draw(mesh,model,texture)
        for mesh in self.dynamic.values():self.draw(mesh,np.eye(4),0)
        gl.glFinish();raw=gl.glReadPixels(0,0,width,height,gl.GL_RGB,gl.GL_UNSIGNED_BYTE)
        im=np.frombuffer(raw,np.uint8).reshape(height,width,3)[::-1].copy();buf=io.BytesIO();Image.fromarray(im).save(buf,format='PNG',compress_level=1)
        return buf.getvalue()

    def draw(self,mesh,matrix,texture):
        gl=self.gl;gl.glUniformMatrix4fv(self.uniform['model'],1,True,np.asarray(matrix,np.float32));gl.glUniform1i(self.uniform['textured'],int(bool(texture)))
        gl.glActiveTexture(gl.GL_TEXTURE0);gl.glBindTexture(gl.GL_TEXTURE_2D,texture);gl.glBindVertexArray(mesh[0]);gl.glDrawArrays(gl.GL_TRIANGLES,0,mesh[2])

    def close(self):self.context.free()


def main():
    from whiteboard.ipc import send,receive
    from gym_anything_robotics.render_lifecycle import watch_parent
    watch_parent();p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--width',type=int);p.add_argument('--height',type=int);p.add_argument('--samples',type=int);a=p.parse_args()
    conn=socket.create_connection(('127.0.0.1',int(os.environ['CF_RENDER_PORT'])));send(conn,dict(secret=os.environ.pop('CF_RENDER_SECRET')))
    r=Raster(a.scene,max(a.width,1920),max(a.height,1080));send(conn,dict(phase='ready'))
    try:
        while True:
            req=receive(conn)
            if req['op']=='close':break
            try:
                start=time.perf_counter();r.apply(req);apply_s=time.perf_counter()-start;images={}
                for name in req['cameras']:
                    size=req.get('resolutions',{}).get(name,req.get('resolution',[a.width,a.height]))
                    images[name]=base64.b64encode(r.capture(name,size)).decode()
                send(conn,dict(ok=True,tick=req['tick'],physics_steps=0,images=images,timings=dict(total_s=time.perf_counter()-start,apply_s=apply_s,renderer=r.driver,engine='software-raster')))
            except Exception as e:
                traceback.print_exc();send(conn,dict(ok=False,error=f'{type(e).__name__}: {e}'))
    finally:r.close();conn.close()
if __name__=='__main__':main()

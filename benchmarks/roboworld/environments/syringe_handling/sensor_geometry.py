"""Physical appearance cues for the software sensor; no evaluator or task outcome inputs.

Triangles are native MuJoCo decorative geometry (ordinary depth/occlusion). Printed
text is a cached small bitmap converted to ink rectangles on the object's surface.
The bubble is a volume-equivalent axial pocket at Chamber.gas_pos, not resolved CFD.
"""
import math
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import mujoco
from . import params as P
from .render import _cylinder_soup, _clip, clock_meshes

INK=(.025,.035,.06,1.)
COLOURS={'medication':(.05,.35,.95,.9),'saline':(.2,.7,.9,.65),'contrast':(.9,.55,.05,.9),'blood':(.7,.02,.02,1.),'air':(.8,.9,1.,.1)}

def add(r,kind,size,p,R,colour):
    if r.scene.ngeom>=r.scene.maxgeom: raise RuntimeError('Sensor geometry buffer full')
    g=r.scene.geoms[r.scene.ngeom]
    mujoco.mjv_initGeom(g,kind,np.asarray(size,float),np.asarray(p,float),np.asarray(R,float).ravel(),np.asarray(colour,np.float32))
    r.scene.ngeom+=1

def mesh(r,v,t,colour):
    # mjGEOM_TRIANGLE vertices are (0,0,0), (size.x,0,0), (0,size.y,0).
    # Affine columns map these directly to each triangle; normal is unit length.
    for tri in np.asarray(v)[np.asarray(t,int)]:
        a,b=tri[1]-tri[0],tri[2]-tri[0];n=np.cross(a,b);length=np.linalg.norm(n)
        if length>1e-16: add(r,mujoco.mjtGeom.mjGEOM_TRIANGLE,(1,1,1),tri[0],np.column_stack((a,b,n/length)),colour)

def segment(r,a,b,width,colour):
    axis=np.asarray(b)-a;L=np.linalg.norm(axis)
    if L<1e-10:return
    z=axis/L;x=np.cross(z,np.eye(3)[np.argmin(abs(z))]);x/=np.linalg.norm(x)
    add(r,mujoco.mjtGeom.mjGEOM_CYLINDER,(width,L/2,0),(np.asarray(a)+b)/2,np.column_stack((x,np.cross(z,x),z)),colour)

@lru_cache(maxsize=200)
def glyphs(text):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    box=font.getbbox(text);im=Image.new('L',(box[2]+1,box[3]-box[1]+1));ImageDraw.Draw(im).text((0,-box[1]),text,font=font,fill=255)
    a=np.array(im)>100;rect=[]
    for y,row in enumerate(a):
        edges=np.flatnonzero(np.diff(np.r_[False,row,False]))
        rect.extend((x0,y,x1,y+1) for x0,x1 in zip(edges[::2],edges[1::2]))
    return im.size,rect

def text(r,words,p,R,height,colour=INK):
    (w,h),rect=glyphs(words);scale=height/h
    for x0,y0,x1,y1 in rect:
        pos=np.asarray(p)+R@np.array([((x0+x1-w)/2)*scale,((h-y0-y1)/2)*scale,0.])
        add(r,mujoco.mjtGeom.mjGEOM_BOX,((x1-x0)*scale/2,scale/2,.000006),pos,R,colour)

def vial_meshes(vial,R):
    """Same piecewise cavity and gravity plane as Vial.port_state, no axial sign flip.

    Live frames carry the exact sampled physical plane. Older recordings lack it:
    solve the volume of the two clipped convex sections using one shared plane.
    """
    up=np.asarray(vial.get('up',R[2,:]),float)
    turn=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    sections=[]
    for radius,lo,hi in ((vial['r'],vial['z0'],vial['body_h']),(vial['neck_r'],vial['body_h'],vial['z0']+vial['h'])):
        v,t=_cylinder_soup(radius,lo,hi,n=32);sections.append((v@turn.T,t))
    if 'level' in vial:level=float(vial['level'])
    else:
        from .render import _volume
        bounds=np.concatenate([v@up for v,t in sections]);lo,hi=bounds.min(),bounds.max()
        for _ in range(24):
            mid=(lo+hi)/2
            if sum(_volume(*_clip(v,t,up,mid)) for v,t in sections)<vial['volume']:lo=mid
            else:hi=mid
        level=(lo+hi)/2
    return [_clip(v,t,up,level) for v,t in sections]

def draw(r,fluid,bq,rot,meta):
    for drop in fluid.get('drops',[]):add(r,mujoco.mjtGeom.mjGEOM_SPHERE,(drop['r'],0,0),drop['p'],np.eye(3),COLOURS[drop['species']])
    for pd in fluid.get('puddles',[]):
        b=pd['body'];p=np.asarray(pd['p']) if b<0 else bq[b,:3]+rot[b]@pd['p']
        # Deposits store a supporting-body-local point; the merged horizontal film
        # follows its body. No privileged outcome/score is painted on the image.
        normal=np.array([0.,0.,1.]);R=np.eye(3) if b<0 else rot[b]
        add(r,mujoco.mjtGeom.mjGEOM_ELLIPSOID,(pd['r'],pd['r'],.0004),p+.00025*normal,R,COLOURS[pd['species']])
    for s in fluid.get('syringes',[]):
        b=s['body'];R=rot[b];rr=s['bore']/2;x=max(s['x'],0.);liq=sum(v for _,v in s['parcels'])
        # Proportionally fit visible chamber volume; nozzle dead space is not a
        # fake extra length beyond the stopper. The air pocket tracks gas_pos.
        gas_length=x*max(s['gas'],0)/max(liq+s['gas'],1e-15)
        gap=float(np.clip(s['gas_pos'],0,1))*max(x-gas_length,0)
        pieces=((0.,gap),(gap+gas_length,x));total=x-gas_length;at=0.
        for sp,vol in s['parcels']:
            end=at+total*vol/max(liq,1e-15);offset=0.
            for lo,hi in pieces:
                aa=max(at,offset);bb=min(end,offset+hi-lo)
                if bb>aa+1e-9:
                    length=bb-aa;mid=lo+(aa+bb)/2-offset
                    add(r,mujoco.mjtGeom.mjGEOM_CYLINDER,(rr*.96,length/2,0),bq[b,:3]+R@np.array([mid,0,0]),R@Rotation.from_euler('y',math.pi/2).as_matrix(),COLOURS[sp])
                offset+=hi-lo
            at=end
        for tick in range(1,round(s['nominal_ml']/s['div_ml'])+1):
            ml=tick*s['div_ml'];at=ml*1e-6/(math.pi*rr*rr)
            for side in (-1,1):
                add(r,mujoco.mjtGeom.mjGEOM_BOX,(.00006,s['od']*(.23 if tick%5==0 else .13),.00003),bq[b,:3]+R@np.array([at,0,side*s['od']/2]),R,INK)
                if tick%5==0:
                    orient=R@np.diag([side,1,side])
                    text(r,f'{ml:g}',bq[b,:3]+R@np.array([at-.0012,s['od']*.30,side*(s['od']/2+.00008)]),orient,.0016)
        label=s.get('label',f"{s['nominal_ml']:g} mL")
        for side in (-1,1):text(r,label,bq[b,:3]+R@np.array([.012,0,side*(s['od']/2+.0001)]),R@np.diag([side,1,side]),.0014)
    for vial in fluid.get('vials',[]):
        b=vial['body']
        for v,t in vial_meshes(vial,rot[b]):
            if len(t):mesh(r,bq[b,:3]+v@rot[b].T,t,COLOURS[vial['species']])
        text(r,'TRAINING 10 mL',bq[b,:3]+rot[b]@np.array([0,-vial['r']-.0011,.015]),rot[b]@Rotation.from_euler('x',math.pi/2).as_matrix(),.002)
    for line in fluid.get('lines',[]):
        path=np.array(line['path']);lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);total=lengths.sum();vol=sum(v for _,v in line['parcels']);distance=0.
        for sp,v in line['parcels']:
            end=distance+total*v/max(vol,1e-15);offset=0.
            for p0,p1,L in zip(path[:-1],path[1:],lengths):
                lo=max(distance,offset);hi=min(end,offset+L)
                if hi>lo and sp!='air':segment(r,p0+(p1-p0)*(lo-offset)/L,p0+(p1-p0)*(hi-offset)/L,line['r'],COLOURS[sp])
                offset+=L
            distance=end
    pad=meta.get('pad_body')
    if pad is not None:
        z=P.PAD['board']+sum(P.LAYERS[k]['thick'] for k in ('muscle','fat','skin'))+.00015
        a=np.linspace(0,2*math.pi,65);rad=P.PAD['ring_d']/2-.00035
        pts=bq[pad,:3]+np.column_stack((rad*np.cos(a),rad*np.sin(a),np.full(len(a),z)))@rot[pad].T
        for aa,bb in zip(pts[:-1],pts[1:]):segment(r,aa,bb,.00035,(.02,.04,.20,1.))
        text(r,'INJECTION TRAINER',bq[pad,:3]+rot[pad]@np.array([0,-.027,z]),rot[pad],.003)
    # Physical wall clock face, marks, hour/minute/second hands at scene time.
    kx,kw,kh,kd,kz=P.WALL_FITTINGS['clock'];c=np.array([kx,P.ROOM['y1']-kd-.002,kz+kh/2]);face=Rotation.from_euler('x',math.pi/2).as_matrix()
    add(r,mujoco.mjtGeom.mjGEOM_CYLINDER,(.136,.0005,0),c,face,(.95,.95,.92,1.))
    for k in range(60):
        a=2*math.pi*k/60;v=np.array([math.sin(a),0,math.cos(a)])
        segment(r,c+v*(.117 if k%5==0 else .125)+[0,-.001,0],c+v*.132+[0,-.001,0],.001 if k%5==0 else .00045,INK)
    for n in (3,6,9,12):
        a=2*math.pi*n/12;text(r,str(n),c+np.array([math.sin(a)*.102,-.002,math.cos(a)*.102]),face,.016)
    for name,g in clock_meshes(fluid.get('t',0)).items():mesh(r,g['v'],g['t'],(.8,.02,.01,1.) if name=='clock_second' else INK)

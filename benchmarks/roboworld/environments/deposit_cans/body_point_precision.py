"""Instance-local same-law rigid contact points before rounding.

Keeps the exact saved float32 body poses, local points, normals and physical
force law. Transform, displacement and force evaluation share the float64
shell helper before rounding to native rigid buffers. The native approximate
search Hessian is retained; body state, contacts and coefficients are unchanged.
"""
import warp as wp
from deposit_cans.precise_body import point as original_point,law
from deposit_cans.relative_step import double
from deposit_cans.elastic_precision import single
from deposit_cans.contact_residual_precision import rigid_point


@wp.func
def transform64(pose:wp.transform):
    t=wp.transform_get_translation(pose);q=wp.transform_get_rotation(pose)
    return wp.transformd(double(t),wp.quatd(wp.float64(q[0]),wp.float64(q[1]),wp.float64(q[2]),wp.float64(q[3])))


@wp.func
def point(p:wp.vec3d,previous:wp.vec3d,radius:float,i:int,ke:float,kd:float,mu:float,eps:float,
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],dt:float):
    sh=shape[i];body=shape_body[sh];pose=wp.transform_identity();cm=wp.vec3(0.)
    if body>=0:pose=bq[body];cm=com[body]
    exact=transform64(pose);bx=wp.transform_point(exact,double(local[i]));n=normal[i];pad=float(0.)
    if margin.shape[0]>0:pad=margin[sh]
    depth=float(-(wp.dot(double(n),p-bx)-wp.float64(radius)-wp.float64(pad)))
    if depth>0.:
        if bprev:
            old=wp.transform_identity()
            if body>=0:old=bprev[body]
            displacement=bx-wp.transform_point(transform64(old),double(local[i]))
        else:
            velocity=wp.spatial_vector()
            if body>=0:velocity=bqd[body]
            displacement=wp.float64(dt)*(double(wp.spatial_top(velocity))+
                wp.cross(double(wp.spatial_bottom(velocity)),bx-wp.transform_point(exact,double(cm))))
        displacement+=wp.float64(dt)*wp.transform_vector(exact,double(vel[i]))
        f,h=law(depth,n,single((p-previous)-displacement),ke,kd,mu,eps,dt)
        value,exact_point=rigid_point(p,previous,wp.float64(radius),i,ke,kd,mu,
            shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,wp.float64(eps),wp.float64(dt))
        return single(value),h,single(exact_point)
    return wp.vec3(0.),wp.mat33(0.),single(bx)


@wp.kernel
def compare(dt:float,q:wp.array[wp.vec3d],previous:wp.array[wp.vec3d],radius:wp.array[float],
    bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],com:wp.array[wp.vec3],
    eps:float,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],
    vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],margin:wp.array[float],shape_body:wp.array[int],
    old_force:wp.array[wp.vec3],new_force:wp.array[wp.vec3],geometry:wp.array2d[wp.float64],
    current_point:wp.array[wp.vec3d],previous_point:wp.array[wp.vec3d]):
    i=wp.tid();ids=indices[i];a=ids[0]
    if a<0:return
    p=q[a];old=previous[a];r=radius[a]
    if ids[1]>=0:
        p=wp.float64(bary[i][0])*q[a];old=wp.float64(bary[i][0])*previous[a]
        for j in range(1,3):
            v=ids[j]
            if v>=0:
                p+=wp.float64(bary[i][j])*q[v];old+=wp.float64(bary[i][j])*previous[v];r=wp.max(r,radius[v])
    f,h,x=original_point(p,old,r,i,ke[i],kd[i],mu[i],eps,shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,dt)
    fe,he,xe=point(p,old,r,i,ke[i],kd[i],mu[i],eps,shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,dt)
    old_force[i]=f;new_force[i]=fe
    body=shape_body[shape[i]];pose=wp.transform_identity();prev=wp.transform_identity()
    if body>=0:pose=bq[body];prev=bprev[body]
    bx=wp.transform_point(pose,local[i]);px=wp.transform_point(prev,local[i])
    b64=wp.transform_point(transform64(pose),double(local[i]));p64=wp.transform_point(transform64(prev),double(local[i]))
    current_point[i]=b64;previous_point[i]=p64
    error=double(bx-px)-(b64-p64);n=double(normal[i])
    geometry[i,0]=wp.length(double(bx)-b64)
    geometry[i,1]=wp.abs(wp.dot(n,double(bx)-b64))
    geometry[i,2]=wp.length(error)
    geometry[i,3]=wp.length(error-n*wp.dot(n,error))
    geometry[i,4]=wp.length(b64-p64)


def install(g):
    """Instance-local dependency copies; runtime install or explicit diagnostic restore.

    The force law remains identical. Replace its geometry helper in BOTH force
    directions, the search matrix, and (via the returned adapter) telemetry.
    Never mutate installed Newton modules or another environment instance.
    """
    from types import FunctionType,MethodType,SimpleNamespace,CodeType
    from warp._src.context import Function,Kernel
    memo={id(original_point):point}
    changed=[]
    def clone(value):
        if id(value)in memo:return memo[id(value)]
        memo[id(value)]=value
        if isinstance(value,MethodType):
            f=clone(value.__func__)
            result=value if f is value.__func__ else MethodType(f,value.__self__)
        elif isinstance(value,SimpleNamespace):
            copied={k:clone(v)for k,v in vars(value).items()}
            result=SimpleNamespace(**copied)if any(copied[k]is not v for k,v in vars(value).items())else value
        elif isinstance(value,(Function,Kernel,FunctionType)):
            warp_function=isinstance(value,(Function,Kernel))
            f=value.func if warp_function else value
            if f is None:return value
            ns=dict(f.__globals__);modified=False
            def global_names(code):
                result=set(code.co_names)
                for child in code.co_consts:
                    if isinstance(child,CodeType):result.update(global_names(child))
                return result
            # Native scheduling launches its kernels inside a nested wave.
            # That code object's global names are absent from the outer names.
            for name in global_names(f.__code__):
                old=ns.get(name)
                if isinstance(old,(Function,Kernel,SimpleNamespace)):
                    new=clone(old)
                    if new is not old:ns[name]=new;modified=True
            closure=f.__closure__
            if closure:
                cells=[]
                def cell(x):return (lambda:x).__closure__[0]
                for old in closure:
                    v=old.cell_contents
                    new=clone(v)if isinstance(v,(MethodType,FunctionType,SimpleNamespace))else v
                    cells.append(cell(new));modified|=new is not v
                closure=tuple(cells)
            if not modified:return value
            copied=FunctionType(f.__code__,ns,f.__name__,f.__defaults__,closure)
            copied.__annotations__=dict(f.__annotations__);copied.__kwdefaults__=f.__kwdefaults__
            copied.__module__=__name__;copied.__qualname__=f.__qualname__+'_bodypoint64'
            if isinstance(value,Kernel):
                result=Kernel(copied,key=value.key+'_bodypoint64',module=wp.get_module(__name__),options=dict(value.options))
            elif isinstance(value,Function):result=wp.func(copied)
            else:result=copied
        else:result=value
        memo[id(value)]=result
        return result
    for obj,name in ((g.contact,'evaluate'),(g.s,'_solve_rigid_body_iteration'),(g.surface_search,'prepare')):
        old=getattr(obj,name);new=clone(old)
        if new is old:raise RuntimeError('No body-point dependency replaced in '+name)
        setattr(obj,name,new);changed.append(name)
    g.body_point_precision=dict(scope=__doc__,changed_roots=changed)
    g.private_body_point_precision=g.body_point_precision  # Historical probe compatibility.
    def adapt_reader(reader):
        old=reader.kernel;new=clone(old)
        if new is old:raise RuntimeError('No body-point dependency replaced in contact evidence')
        reader.kernel=new
    g.adapt_contact_reader=adapt_reader
    return adapt_reader

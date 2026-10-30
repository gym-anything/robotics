"""Float64 forms of the installed closest-point branches; unchanged topology."""
import warp as wp

@wp.func
def triangle(a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,p:wp.vec3d):
    ab=b-a;ac=c-a;ap=p-a;zero=wp.float64(0.);one=wp.float64(1.)
    d1=wp.dot(ab,ap);d2=wp.dot(ac,ap)
    if d1<=zero and d2<=zero:return a,wp.vec3d(one,zero,zero)
    bp=p-b;d3=wp.dot(ab,bp);d4=wp.dot(ac,bp)
    if d3>=zero and d4<=d3:return b,wp.vec3d(zero,one,zero)
    cp=p-c;d5=wp.dot(ab,cp);d6=wp.dot(ac,cp)
    if d6>=zero and d5<=d6:return c,wp.vec3d(zero,zero,one)
    vc=d1*d4-d3*d2
    if vc<=zero and d1>=zero and d3<=zero:
        v=d1/(d1-d3);return a+v*ab,wp.vec3d(one-v,v,zero)
    vb=d5*d2-d1*d6
    if vb<=zero and d2>=zero and d6<=zero:
        v=d2/(d2-d6);return a+v*ac,wp.vec3d(one-v,zero,v)
    va=d3*d6-d5*d4
    if va<=zero and d4-d3>=zero and d5-d6>=zero:
        v=(d4-d3)/((d4-d3)+(d5-d6));return b+v*(c-b),wp.vec3d(zero,one-v,v)
    denom=one/(va+vb+vc);v=vb*denom;w=vc*denom
    return a+v*ab+w*ac,wp.vec3d(one-v-w,v,w)

@wp.func
def edge(p1:wp.vec3d,q1:wp.vec3d,p2:wp.vec3d,q2:wp.vec3d,epsilon:wp.float64):
    d1=q1-p1;d2=q2-p2;r=p1-p2
    a=wp.dot(d1,d1);e=wp.dot(d2,d2);f=wp.dot(d2,r)
    zero=wp.float64(0.);one=wp.float64(1.);s=zero;t=zero
    if a<=epsilon and e<=epsilon:
        s=zero;t=zero
    elif a<=epsilon:t=wp.clamp(f/e,zero,one)
    elif e<=epsilon:s=wp.clamp(-wp.dot(d1,r)/a,zero,one)
    else:
        c=wp.dot(d1,r);b=wp.dot(d1,d2);cross=wp.cross(d1,d2)
        denom=wp.dot(cross,cross);si=zero
        # A Gram determinant subtraction can become tiny-positive under FMA
        # even for exactly parallel axis-aligned edges. The cross form retains
        # zero exactly. This is coordinate arithmetic, not contact softening.
        # The original Gram subtraction cannot resolve relative determinants
        # below its floating-point error. Use the parallel closest-point branch
        # there rather than differentiate an arbitrary point along an almost
        # parallel overlap. Contacts/loads remain present, with unchanged laws.
        if denom>wp.float64(1.4210854715202004e-14)*a*e:
            numerator=wp.dot(wp.cross(d2,r),cross)
            si=wp.clamp(numerator/denom,zero,one)
        ti=(b*si+f)/e
        if ti<zero:s=wp.clamp(-c/a,zero,one)
        elif ti>one:s=wp.clamp((b-c)/a,zero,one);t=one
        else:s=si;t=ti
    return wp.vec3d(s,t,wp.length((p1+s*d1)-(p2+t*d2)))


@wp.func
def edge_pair(i:int,j:int,q:wp.array[wp.vec3d],edges:wp.array2d[int],epsilon:wp.float64):
    # Parallel overlapping segments need not have a unique closest point.
    # Evaluate the unordered pair in ONE canonical order so reciprocal visits
    # use the same two points, even with reversed endpoints or varying slip.
    if i<j:
        return edge(q[edges[i,2]],q[edges[i,3]],q[edges[j,2]],q[edges[j,3]],epsilon)
    value=edge(q[edges[j,2]],q[edges[j,3]],q[edges[i,2]],q[edges[i,3]],epsilon)
    return wp.vec3d(value[1],value[0],value[2])

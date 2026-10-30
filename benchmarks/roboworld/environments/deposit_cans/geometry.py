"""Authored metric surfaces; the same triangles drive mechanics and rendering."""
import numpy as np

CAN_SPECS=((.066,.115,.000112),(.057,.146,.000118),(.066,.167,.000116))

def bridge(n,a,b,faces,reverse=False):
    for i in range(n):
        j=(i+1)%n
        pair=((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i))
        faces.extend(tuple(reversed(t)) if reverse else t for t in pair)

def can_mesh(kind=0,n=24,dent=0.,*,quality_end=False):
    diameter,h,t=CAN_SPECS[kind];r=diameter/2
    theta=np.arange(n)*2*np.pi/n
    profile=[(.65*r,-h/2+.006),(.91*r,-h/2),(.99*r,-h/2+.010)]
    profile.extend((r,z) for z in np.linspace(-h/2+.020,h/2-.020,max(3,round(h/.017))))
    profile.extend(((.86*r,h/2-.008),(.86*r,h/2),(.81*r,h/2-.002)))
    rings=[]
    for radius,z in profile:
        # Pre-existing shallow dent is part of rest geometry, not a runtime morph.
        depth=dent*np.exp(-(np.angle(np.exp(1j*(theta-.4)))/.35)**2-(z/.022)**2)
        rings.append(np.column_stack(((radius-depth)*np.cos(theta),(radius-depth)*np.sin(theta),np.full(n,z))))
    vertices=np.concatenate(rings);faces=[]
    for a in range(len(rings)-1):bridge(n,a,a+1,faces)
    if quality_end:
        # The end annulus is planar. Four interpolated rings compressed near
        # the off-centre opening produced slivers below the native hinge area
        # threshold. Constrained boundary verification makes this Delaunay
        # triangulation explicit: no triangle is allowed to bridge the hole.
        from scipy.spatial import Delaunay
        from matplotlib.path import Path
        nt=max(12,n//2);angle=np.arange(nt)*2*np.pi/nt
        hole=np.column_stack((.011+.010*np.cos(angle),.006*np.sin(angle),np.full(nt,h/2-.002)))
        outer_ids=np.arange((len(rings)-1)*n,len(rings)*n)
        hole_ids=np.arange(len(vertices),len(vertices)+nt)
        vertices=np.vstack((vertices,hole))
        spacing=2*np.pi*profile[-1][0]/n
        candidates=np.array([(x+(j%2)*spacing/2,y) for j,y in enumerate(np.arange(-r,r,spacing*np.sqrt(3)/2)) for x in np.arange(-r,r,spacing)])
        candidates=candidates[Path(vertices[outer_ids,:2]).contains_points(candidates)&~Path(hole[:,:2]).contains_points(candidates)]
        # Interior points avoid skinny fans in the broad side of the annulus.
        # Keep away from boundary segments, not merely boundary vertices.
        for loop in (vertices[outer_ids,:2],hole[:,:2]):
            for j,a in enumerate(loop):
                b=loop[(j+1)%len(loop)];edge=b-a
                fraction=np.clip((candidates-a)@edge/np.dot(edge,edge),0,1)
                distance=np.linalg.norm(candidates-(a+fraction[:,None]*edge),axis=1)
                candidates=candidates[distance>=.60*spacing]
        interior=np.column_stack((candidates,np.full(len(candidates),h/2-.002)))
        interior_ids=np.arange(len(vertices),len(vertices)+len(interior))
        vertices=np.vstack((vertices,interior));ids=np.r_[outer_ids,hole_ids,interior_ids]
        points=vertices[ids,:2];tess=Delaunay(points).simplices
        inside=Path(hole[:,:2]).contains_points(points[tess].mean(1))
        panel=ids[tess[~inside]]
        from collections import Counter
        count=Counter(tuple(sorted((int(t[j]),int(t[(j+1)%3])))) for t in panel for j in range(3))
        actual={edge for edge,nuse in count.items() if nuse==1}
        expected={tuple(sorted((int(loop[i]),int(loop[(i+1)%len(loop)])))) for loop in (outer_ids,hole_ids) for i in range(len(loop))}
        if actual!=expected:raise ValueError('Can end triangulation changed its opening boundary')
        faces.extend(panel.tolist())
    else:
        hole=np.column_stack((.011+.010*np.cos(theta),.006*np.sin(theta),np.full(n,h/2-.002)))
        outer=rings[-1].copy()
        for fraction in (.25,.50,.75,1.):rings.append(outer*(1-fraction)+hole*fraction)
        vertices=np.concatenate(rings)
        for a in range(len(rings)-5,len(rings)-1):bridge(n,a,a+1,faces)
    center=len(vertices);vertices=np.vstack((vertices,[0,0,-h/2+.008]))
    for i in range(n):faces.append((center,(i+1)%n,i))
    faces=np.asarray(faces,np.int32)
    thickness=np.full(len(faces),t)
    centroid=vertices[faces].mean(1)
    thickness[(centroid[:,2]<-h/2+.009)|(centroid[:,2]>h/2-.007)]=.00022
    return vertices,faces,thickness

def rectangle_ring(hx,hy,n,z):
    # Resolve all four corners exactly; near-corner sliver facets are unacceptable.
    nx=round(n*hx/(2*(hx+hy)));ny=n//2-nx;xy=[]
    for a in np.linspace(-hx,hx,nx,endpoint=False):xy.append((a,-hy,z))
    for a in np.linspace(-hy,hy,ny,endpoint=False):xy.append((hx,a,z))
    for a in np.linspace(hx,-hx,nx,endpoint=False):xy.append((a,hy,z))
    for a in np.linspace(hy,-hy,ny,endpoint=False):xy.append((-hx,a,z))
    return np.asarray(xy)

def bag_mesh(n=64,rows=36,*,cuff=False,star_bottom=False):
    # Revisions1..4 retain their original planar-bottom candidate. The opened
    # star-bottom revision regularizes the gathered seam as a continuous shell;
    # it does not simulate the manufacture/heat welding of eight film plies.
    # The cuff is retained at four patches, with no rigid bottom support.
    rings=[]
    for z in np.linspace(.022,.969 if cuff else .9985,rows):
        fraction=.92+.08*(z-.022)/.9765
        ring=rectangle_ring(.288*fraction,.197*fraction,n,z)
        # Millimetric initial slack, shared by physical and optical surface.
        if .025<z<.99:
            radial=ring[:,:2]/np.maximum(np.linalg.norm(ring[:,:2],axis=1)[:,None],1e-12)
            ring[:,:2]+=radial*(.0015*np.sin(np.arange(n)*np.pi/4+z*12))[:,None]
        rings.append(ring)
    if cuff:
        # Installed film routes INSIDE the retention loops, OVER their rounded
        # top, then OUTSIDE the cross rods. No bag/stand contact is disabled.
        # This addresses the measured2.791mm initial penetration; it is not yet
        # an exact flat-pattern/star-bottom reconstruction of the selected bag.
        profile=[(.288,.184,.980),(.288,.184,.993),(.288,.194,1.0105),
                 (.3123,.222,1.008),(.3123,.222,.985),
                 (.3123,.222,.960),(.3123,.222,.935),(.3123,.222,.910)]
        for i,(hx,hy,z) in enumerate(profile):
            ring=rectangle_ring(hx,hy,n,z)
            if i>=3:
                # Rectangular sheet corners must not cross the rounded skirt.
                center=np.array([hx-.006,hy-.006]);delta=np.abs(ring[:,:2])-center
                mask=np.all(delta>0,axis=1)
                ring[mask,:2]=np.sign(ring[mask,:2])*(center+.006*delta[mask]/np.linalg.norm(delta[mask],axis=1)[:,None])
                # Resolve fillet tangencies rather than cutting a long diagonal
                # from the corner's45-degree point to a distant side vertex.
                original=rectangle_ring(hx,hy,n,z)
                for corner in np.flatnonzero(mask):
                    for neighbor in ((corner-1)%n,(corner+1)%n):
                        difference=original[neighbor,:2]-original[corner,:2]
                        axis=int(np.argmax(abs(difference)))
                        ring[neighbor,:2]=original[corner,:2]
                        ring[neighbor,axis]+=.006*np.sign(difference[axis])
            rings.append(ring)
    vertices=np.concatenate(rings);faces=[]
    for a in range(len(rings)-1):
        if not cuff or a<rows-1:bridge(n,a,a+1,faces);continue
        # Choose the less skewed diagonal on the cuff transitions. This changes
        # triangulation only, not the designed sheet surface or collision flags.
        def quality(tris):
            p=vertices[np.asarray(tris)];score=1.
            for k in range(3):
                u=p[:,(k+1)%3]-p[:,k];v=p[:,(k+2)%3]-p[:,k]
                cosine=np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
                score=min(score,float(np.min(1.-cosine)))
            return score
        for i in range(n):
            j=(i+1)%n;A=a*n+i;B=a*n+j;C=(a+1)*n+j;D=(a+1)*n+i
            first=((A,B,C),(A,C,D));second=((A,B,D),(B,C,D))
            faces.extend(first if quality(first)>=quality(second) else second)
    # A quality planar triangulation closes the seam region. No high-valence
    # centre fan with near-zero angles or artificial support collider.
    from scipy.spatial import Delaunay
    hx,hy=.288*.92,.197*.92;nx=round(n*hx/(2*(hx+hy)));ny=n//2-nx
    interior=np.array([(x,y,.022) for x in np.linspace(-hx,hx,nx+1)[1:-1] for y in np.linspace(-hy,hy,ny+1)[1:-1]])
    indices=np.r_[np.arange(n),np.arange(len(vertices),len(vertices)+len(interior))]
    vertices=np.vstack((vertices,interior))
    bottom=Delaunay(vertices[indices,:2]).simplices
    for tri in bottom:faces.append(tuple(indices[tri[::-1]]))
    if star_bottom:
        # S28's opened star-seal photograph shows radial gathered pleats,
        # unlike the old perfectly flat rectangular panel. This is a resolved
        # geometric approximation, not an imported ClearStream flat pattern.
        # Keep the exact side boundary/cuff and inventory locations unchanged.
        # A6mm central gather and6mm radial folds are disclosed estimates.
        # Physical AND optical triangles use these coordinates.
        xy=vertices[indices,:2]
        rho=np.maximum(np.abs(xy[:,0])/hx,np.abs(xy[:,1])/hy)
        theta=np.arctan2(xy[:,1]/hy,xy[:,0]/hx)
        ramp=np.clip(rho,0.,1.)
        pleat=.006*np.sin(4*theta)**2*4*ramp*(1-ramp)
        vertices[indices,2]=.016+.006*ramp+pleat
        # The central weld is represented by connected material vertices,
        # not a point pinned to world. No mass/stiffness/support is added.
    mouth=np.arange((len(rings)-1)*n,len(rings)*n,dtype=np.int32)
    pin_ring=np.arange((rows+2)*n,(rows+3)*n,dtype=np.int32) if cuff else mouth
    corners=[]
    for x,y in ((-1,-1),(1,-1),(1,1),(-1,1)):
        location=[x*.24,y*.194] if cuff else [x*.288,y*.197]
        nearest=int(np.linalg.norm(vertices[pin_ring,:2]-location,axis=1).argmin())
        corners.extend(pin_ring[[(nearest+j)%n for j in (-1,0,1)]])
    return vertices,np.asarray(faces,np.int32),mouth,np.asarray(corners,np.int32)

def area(vertices,faces):
    p=vertices[faces]
    return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2

def validate_surface(v,f,*,open_edges=True):
    from collections import Counter
    if not np.isfinite(v).all() or area(v,f).min()<1e-9:raise ValueError('Degenerate surface')
    edges=Counter(tuple(sorted((int(t[j]),int(t[(j+1)%3])))) for t in f for j in range(3))
    if max(edges.values())>2 or (not open_edges and min(edges.values())!=2):raise ValueError('Nonmanifold surface')
    return dict(vertices=len(v),triangles=len(f),boundary_edges=sum(x==1 for x in edges.values()),area_m2=float(area(v,f).sum()))

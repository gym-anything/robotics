"""Conservative triangular corner cells on the formed pan's physical facets.

Same gravity, viscosity, yield and contact-driven slip as rectangular cells.
Upwind boundary flux crosses real shared edges; it is never silently discarded
or emitted into flight at an internal triangulation edge. One uniform material
cell per curved facet is an explicit spatial discretization.
"""
import numpy as np


def edge_network(surfaces,offsets):
    groups={}
    for i,s in enumerate(surfaces):groups.setdefault(s.shape,[]).append(i)
    result={}
    for i,s in enumerate(surfaces):
        if s.triangle_uv is None:continue
        ids=np.array(groups[s.shape]);others=[surfaces[j] for j in ids]
        origins=np.array([t.origin for t in others]);normals=np.array([t.normal for t in others])
        edges=[]
        for p,q in zip(s.triangle_uv,np.roll(s.triangle_uv,-1,axis=0)):
            d=q-p;length=float(np.linalg.norm(d));outward=np.array((d[1],-d[0]))/length
            uv=(p+q)/2;point=s.origin+uv[0]*s.u+uv[1]*s.v
            distance=abs(np.einsum('ij,ij->i',point-origins,normals))
            matches=[]
            for j in ids[(distance<2e-7)&(ids!=i)]:
                other=surfaces[j];relative=point-other.origin
                target=np.array((relative@other.u,relative@other.v))
                if other.contains(target[None],tolerance=2e-7)[0]:
                    cell=int(offsets[j]+np.clip(int(target[0]/other.du),0,other.nx-1)*other.ny+
                        np.clip(int(target[1]/other.dv),0,other.ny-1))
                    matches.append((j,cell))
            if len(matches)!=1:
                raise ValueError(f'Mesh film edge {s.name} has {len(matches)} adjacent mapped faces; expected one')
            edges.append((outward,length,matches[0][1]))
        result[i]=edges
    return result


def advance_cell(surface,mass,gravity_uv,drag_uv,edges,dt,density):
    """Return net remaining mass and edge outflows; every species balances."""
    height=float((mass/density).sum()/surface.cell_area)
    ratio=mass[0]/max(mass[1],1e-12)
    viscosity=np.array((.001,max(.005,.5/(1+10*ratio)),.08))
    tau=np.linalg.norm(gravity_uv)*height*density
    yielded=np.maximum(0.,1-np.array((0.,3/(1+ratio),0.))/np.maximum(tau,1e-20))
    speed=gravity_uv[None,:]*(density*height**2/(3*viscosity)*yielded)[:,None]+drag_uv[None,:]
    rates=np.array([np.maximum(0.,speed@normal)*length/surface.cell_area for normal,length,_ in edges])
    total=rates.sum(0)
    if not np.isfinite(rates).all() or float(total.max()*dt)>.45*1024:
        raise FloatingPointError(f'Unsupported deep/fast triangular film on {surface.name}')
    # Exact positive drain for the frozen per-step transport coefficients.
    sent=mass*(-np.expm1(-total*dt))
    flux=rates/np.maximum(total,1e-30)*sent
    return mass-flux.sum(0),[(edge[2],part) for edge,part in zip(edges,flux)]

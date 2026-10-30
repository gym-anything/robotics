"""Closed mating solids, in metres. Visual and collision surfaces share meshes.

ISO-style M12x1 flank geometry is represented geometrically, not by a
proximity-triggered screw joint. Product envelopes are separate from documented
interface-clearance estimates. These candidate meshes still require contact
qualification; construction alone does not establish successful mating.
"""
import math
import numpy as np
import trimesh

def annulus_surface(z, inner, outer, *, sectors=128):
    """Closed annular solid with angle/height-dependent inner and outer radii."""
    z = np.asarray(z, float)
    angles = np.arange(sectors)*2*math.pi/sectors
    verts = []
    for function in (inner, outer):
        for zz in z:
            for aa in angles:
                rr = float(function(zz, aa))
                if rr <= 0:
                    raise ValueError("Positive radii required for an annular surface")
                verts.append((rr*math.cos(aa), rr*math.sin(aa), zz))
    faces = []
    rings = len(z)
    def idx(surface, ring, sector):
        return surface*rings*sectors+ring*sectors+sector%sectors
    for surface in (0,1):
        for ring in range(rings-1):
            for sector in range(sectors):
                a,b,c,d = (idx(surface,ring,sector),idx(surface,ring,sector+1),
                           idx(surface,ring+1,sector+1),idx(surface,ring+1,sector))
                faces.extend(((a,b,c),(a,c,d)))
    for ring in (0,rings-1):
        for sector in range(sectors):
            a,b,c,d = (idx(0,ring,sector),idx(1,ring,sector),
                       idx(1,ring,sector+1),idx(0,ring,sector+1))
            faces.extend(((a,b,c),(a,c,d)))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError("Mating surface must be a closed positive-volume solid")
    return mesh

def metric_thread_radius(z, theta, *, major=.006, pitch=.001):
    """60 degree truncated metric-thread profile with right-hand helical lead."""
    phase = (z-pitch*theta/(2*math.pi)+pitch/2) % pitch-pitch/2
    root = major-0.61343*pitch
    crest_half = pitch/16
    return max(root, major-math.sqrt(3)*max(abs(phase)-crest_half,0))

def m12_external(*, length=.006, sectors=128, axial_samples_per_pitch=24):
    z = np.linspace(-length, 0, round(length/.001*axial_samples_per_pitch)+1)
    def outer(zz, aa):
        radius = metric_thread_radius(zz,aa)
        # Turned lead chamfer on the exposed end; not a hidden alignment force.
        chamfer = max(0., (zz+.0005)/.0005)*.00035
        return radius-chamfer
    return annulus_surface(z, lambda z,a:.00445, outer, sectors=sectors)

def m12_internal(*, length=.006, clearance=.000060, sectors=128,
                 axial_samples_per_pitch=24, negative_entry_chamfer=False):
    z = np.linspace(-length, 0, round(length/.001*axial_samples_per_pitch)+1)
    def inner(zz, aa):
        # A coupling nut receives the socket at its negative-Z mouth. The
        # positive-end candidate is kept only for source-matched controls.
        edge=(-length+.0005-zz) if negative_entry_chamfer else (zz+.0005)
        chamfer = max(0., edge/.0005)*.00035
        return metric_thread_radius(zz,aa)+clearance+chamfer
    return annulus_surface(z, inner, lambda z,a:.0074, sectors=sectors)

def open_rectangular_shell(*, width, height, depth, wall, rear_wall=True):
    """Socket shell as a union of real walls; its opening remains collision-free."""
    if min(width,height) <= 2*wall:
        raise ValueError("Socket wall consumes the opening")
    pieces = []
    for sign in (-1,1):
        piece = trimesh.creation.box((wall,height,depth))
        piece.apply_translation((sign*(width-wall)/2,0,-depth/2))
        pieces.append(piece)
        piece = trimesh.creation.box((width-2*wall,wall,depth))
        piece.apply_translation((0,sign*(height-wall)/2,-depth/2))
        pieces.append(piece)
    if rear_wall:
        piece = trimesh.creation.box((width,height,wall))
        piece.apply_translation((0,0,-depth-wall/2))
        pieces.append(piece)
    return pieces

def ramp_prism(width, length, rise, *, base=.0003):
    """A latch lead-in with a sloped front and a steep retaining back."""
    v=np.array([[-width/2,0,0],[width/2,0,0],
                [-width/2,length,0],[width/2,length,0],
                [-width/2,0,base],[width/2,0,base],
                [-width/2,length,base+rise],[width/2,length,base+rise]])
    return trimesh.convex.convex_hull(v)

def newton_mesh(tm, *, sdf=False, resolution=256, cache=None, device="cuda:0"):
    import newton
    if not tm.is_watertight or tm.volume <= 0:
        raise ValueError("Collision solids must be closed and outward")
    mesh=newton.Mesh(np.asarray(tm.vertices,np.float32),
                     np.asarray(tm.faces,np.int32).ravel())
    if sdf:
        mesh.build_sdf(device=device, max_resolution=resolution,
                       narrow_band_range=(-.0004,.0004), margin=.00005, cache_dir=cache)
    return mesh

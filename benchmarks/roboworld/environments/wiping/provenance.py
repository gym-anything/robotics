"""Bind physical recordings to the actual authored collision/rest geometry."""
import hashlib,json
import numpy as np

def save_source_manifest(folder):
    """Record source hashes at RUN START, including native solver kernels."""
    import importlib.metadata,subprocess
    from pathlib import Path
    import newton
    root=Path(__file__).resolve().parent
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*')) if p.suffix in ('.py','.mdl','.kit')}
    native=Path(newton.__file__).parent/'_src/solvers/vbd'
    kernels={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(native.glob('*.py'))}
    geometry=Path(newton.__file__).parent/'_src/geometry'
    collision={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(geometry.glob('*.py'))}
    result=dict(wiping_source_sha256=files,native_vbd_sha256=kernels,
                native_geometry_sha256=collision,
                versions={n:importlib.metadata.version(n) for n in ('newton','warp-lang','numpy')},
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                note='Hashes capture exact source at run start; HEAD alone may precede current task changes.')
    (Path(folder)/'source_manifest.json').write_text(json.dumps(result,indent=2)+'\n')

def save_inertial_ledger(assembly,folder):
    """Retain authored inertias for render-only CPU reconstruction.

    Parallel GPU versus CPU mesh volume reductions can differ by a few ulps.
    These are the original static metadata, not a changed physical trajectory.
    The COMPLETE geometry digest still has to match after loading this ledger.
    """
    from pathlib import Path
    b=assembly.builder
    np.savez_compressed(Path(folder)/'authored_inertias.npz',
                        body_mass=np.asarray(b.body_mass),body_inertia=np.asarray(b.body_inertia))

def geometry_sha256(assembly):
    b=assembly.builder;h=hashlib.sha256()
    def add(value):
        x=np.asarray(value)
        h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    h.update(json.dumps(b.shape_label).encode())
    for name in ('shape_body','shape_type','shape_scale','shape_transform','body_mass','body_inertia',
                 'particle_mass','tri_indices','tri_materials','edge_indices','edge_bending_properties'):
        h.update(name.encode());add(getattr(b,name))
    for source in b.shape_source:
        if source is not None and hasattr(source,'vertices'):
            add(source.vertices);add(source.indices)
    return h.hexdigest()

if __name__=='__main__':
    import argparse
    from pathlib import Path
    from .config import Config
    from .assembly import Assembly
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    meta=json.loads((a.run/'result.json').read_text())
    assembly=Assembly(Config(**meta['config']))
    actual=geometry_sha256(assembly)
    if actual!=meta['geometry_sha256']:raise RuntimeError('Reconstructed source no longer matches recording')
    save_inertial_ledger(assembly,a.run)
    print(json.dumps(dict(geometry_sha256=actual,physics_steps=0,inertias=str(a.run/'authored_inertias.npz'))))

"""Lossless numeric-column representation of rigid/shell contact evidence.

Only new teacher samples use this explicit schema. Existing row-form records
remain readable; HTTP reads still expand numeric arrays through PT1 unpack.
No contact selection, precision, force, sampling or retention rule changes.
"""
import numpy as np


def columns(surface,body,shape,points,forces,source):
    n=len(surface);p=np.asarray(points);f=np.asarray(forces)
    if len(body)!=n or len(shape)!=n or p.shape!=(n,3)or f.shape!=(n,3):
        raise ValueError('Contact column length/shape mismatch')
    if p.dtype.kind!='f'or f.dtype.kind!='f'or not np.isfinite(p).all()or not np.isfinite(f).all():
        raise FloatingPointError('Nonfinite or nonfloating contact evidence')
    if not isinstance(source,str):raise TypeError('Contact source must be a string')
    if any(not isinstance(s,str)for group in (surface,body,shape)for s in group):
        raise TypeError('Contact labels must be strings')
    return dict(schema='rigid-shell-contact-columns-v1',count=n,surface=list(surface),
        body=list(body),shape=list(shape),point_world_m=np.ascontiguousarray(p),
        force_on_surface_n=np.ascontiguousarray(f),source=source)


def rows(record):
    if isinstance(record,list):return record
    if record.get('schema')!='rigid-shell-contact-columns-v1':raise ValueError('Unknown contact record schema')
    n=record['count'];keys=('surface','body','shape','point_world_m','force_on_surface_n')
    if any(len(record[k])!=n for k in keys):raise ValueError('Malformed contact columns')
    points=record['point_world_m'];forces=record['force_on_surface_n']
    if isinstance(points,np.ndarray):points=points.tolist()
    if isinstance(forces,np.ndarray):forces=forces.tolist()
    return [dict(surface=record['surface'][i],body=record['body'][i],shape=record['shape'][i],
        point_world_m=points[i],force_on_surface_n=forces[i],source=record['source'])for i in range(n)]

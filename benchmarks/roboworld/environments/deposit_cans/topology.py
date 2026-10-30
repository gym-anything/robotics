"""Reciprocal edge exclusion for native two-sided edge contact accumulation.

Newton1.5's n-ring builder removes all hinge-adjacent rows as 'incident',
including opposite vertices. That can leave one edge excluded in only one
direction. Native edge force assembly updates only the first edge and relies
on the reverse query for its reaction. Use the symmetric closure of its
existing topological exclusions; never filter between disconnected surfaces.
"""
import numpy as np
import warp as wp

def enable_bag_local_contact(detector, bag_ids):
    """Remove n-ring exclusions on the film, not incident-primitive checks.

    Two neighbouring nonincident film primitives can fold through one another.
    Native n-ring filtering hid precisely that pair at transfer tick434. Can
    rim filtering and helper-anchor exclusions remain unchanged. The narrow
    phase still excludes primitives sharing vertices, as required geometrically.
    """
    selected=np.zeros(detector.model.particle_count,bool)
    selected[np.asarray(bag_ids,np.int32)]=True
    report={}
    for label,offset_label,bag_rows in (
        ('vertex_triangle_filtering_list','vertex_triangle_filtering_list_offsets',selected),
        ('edge_filtering_list','edge_filtering_list_offsets',
         selected[detector.model.edge_indices.numpy()[:,2:]].all(axis=1))):
        values=getattr(detector,label);offsets=getattr(detector,offset_label)
        if values is None:continue
        source=values.numpy();offset=offsets.numpy()
        rows=[np.empty(0,np.int32)if bag_rows[i]else source[offset[i]:offset[i+1]]
              for i in range(len(offset)-1)]
        newoffset=np.zeros(len(offset),np.int32)
        newoffset[1:]=np.cumsum([len(row)for row in rows],dtype=np.int32)
        data=np.concatenate(rows)
        setattr(detector,label,wp.array(data,dtype=int,device=detector.device))
        setattr(detector,offset_label,wp.array(newoffset,dtype=int,device=detector.device))
        report[label]=dict(before=len(source),after=len(data))
    return report


def reciprocal_edge_filter(detector):
    values=detector.edge_filtering_list
    if values is None:return dict(added_reverse_exclusions=0)
    source=values.numpy();offset=detector.edge_filtering_list_offsets.numpy()
    rows=[set(map(int,source[offset[i]:offset[i+1]])) for i in range(len(offset)-1)]
    additions=0
    for i in range(len(rows)):
        for j in tuple(rows[i]):
            if i not in rows[j]:rows[j].add(i);additions+=1
    newoffset=np.zeros(len(offset),dtype=np.int32)
    newoffset[1:]=np.cumsum([len(v) for v in rows],dtype=np.int32)
    data=np.concatenate([np.array(sorted(v),dtype=np.int32) for v in rows])
    detector.edge_filtering_list=wp.array(data,dtype=int,device=detector.device)
    detector.edge_filtering_list_offsets=wp.array(newoffset,dtype=int,device=detector.device)
    return dict(added_reverse_exclusions=additions,original_exclusions=len(source),
        corrected_exclusions=len(data),rule='symmetric closure of native topological exclusions')

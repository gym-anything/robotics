"""Read-only physical-surface intersection and linear substep-path queries.

IPC Toolkit supplies geometry predicates only; Newton still owns all dynamics.
No virtual containment cap is put into the collision mesh. No state is projected
or repaired here. A failed path remains a verifier failure, not an automatic reset.
"""
import numpy as np
import ipctk

class SurfaceIntegrity:
    def __init__(self,positions,faces,*,bag_ids=None):
        self.faces=np.asarray(faces,np.int32)
        self.ids=np.unique(self.faces)
        inverse=np.full(len(positions),-1,np.int32);inverse[self.ids]=np.arange(len(self.ids))
        f=inverse[self.faces]
        edges=np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)
        self.previous=np.asarray(positions[self.ids],np.float64).copy()
        self.mesh=ipctk.CollisionMesh(self.previous,edges,f)
        if bag_ids is not None:
            # The containment gate covers bag self-intersection and can/bag
            # crossing. Can/can and within-can mechanics remain active in Newton;
            # they are outside this geometric containment query's remit.
            labels=np.zeros(len(self.ids),np.int32)
            selected=np.isin(self.ids,np.asarray(bag_ids,np.int32))
            labels[selected]=np.arange(1,int(selected.sum())+1)
            self.mesh.can_collide=ipctk.make_vertex_patches_filter(labels)
        self.initial_intersection=bool(ipctk.has_intersections(self.mesh,self.previous))
        self.last_query=dict(predicate_queries=0,maximum_subdivision_depth=0,
            conservative_full_interval_rejection=False,certified_by_subintervals=False)
    def query(self,positions):
        q=np.asarray(positions[self.ids],np.float64)
        # Caller supplies every completed Newton substep. This tests its piecewise
        # linear geometric path, not an unobserved continuous real-world motion.
        free=bool(ipctk.is_step_collision_free(self.mesh,self.previous,q))
        receipt=dict(predicate_queries=1,maximum_subdivision_depth=0,
            conservative_full_interval_rejection=not free,certified_by_subintervals=False)
        if not free:
            # IPC1.6 reserves min(0.2*initial_distance,1e-4) even when the
            # requested minimum distance is zero. Thus false does not prove a
            # crossing. Certify smaller intervals of THE SAME linear path with
            # the same conservative predicate, tolerance, filters and geometry.
            # Their union covers the original interval; no physics is added.
            # src/ipc/ccd/tight_inclusion_ccd.cpp, ccd_strategy, v1.6.0.
            candidates=ipctk.Candidates()
            candidates.build(self.mesh,self.previous,q)
            def certify(left,right,depth):
                receipt['predicate_queries']+=1
                receipt['maximum_subdivision_depth']=max(receipt['maximum_subdivision_depth'],depth)
                if candidates.is_step_collision_free(self.mesh,left,right):return True
                if depth>=8:return False # Fail closed when certification is unresolved.
                middle=(left+right)*.5
                return certify(left,middle,depth+1) and certify(middle,right,depth+1)
            middle=(self.previous+q)*.5
            free=certify(self.previous,middle,1) and certify(middle,q,1)
            receipt['certified_by_subintervals']=bool(free)
        self.last_query=receipt
        self.previous=q.copy()
        return free

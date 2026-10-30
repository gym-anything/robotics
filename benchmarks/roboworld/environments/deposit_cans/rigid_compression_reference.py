"""Opt-in unilateral normal stabilization diagnostic, not a default change.

Native initialization stores C0_n=-separation, including negative air gaps.
Both the force and dual updates subtract alpha*C0_n after activation. This
can turn pre-impact travel across an air gap into a positive stabilized
penetration. Preserve the tangential reference and positive penetration;
only the normal reference for an initially separated candidate becomes zero.
Applied once to shared C0 storage, so force, dual and export see the SAME law.
No contact is discarded and no position, velocity or material is changed.
"""
import hashlib
from pathlib import Path
from types import MethodType
import warp as wp

@wp.kernel
def compression_only(count:wp.array[int],normal:wp.array[wp.vec3],reference:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=count[0]:return
    n=normal[i];value=reference[i];depth=wp.dot(n,value)
    if depth<0.:reference[i]=value-n*depth

def install(solver):
    from newton._src.solvers.vbd import rigid_vbd_kernels as native
    expected='0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review compression reference against changed Newton source')
    if getattr(solver,'compression_reference_candidate',False):raise ValueError('Already installed')
    original=solver._initialize_rigid_bodies
    def initialize(self,state_in,control,contacts,dt,refresh):
        original(state_in,control,contacts,dt,refresh)
        if contacts is not None and self.rigid_contact_hard:
            wp.launch(compression_only,contacts.rigid_contact_max,[contacts.rigid_contact_count,
                contacts.rigid_contact_normal,self.body_body_contact_C0],device=self.device)
    solver._initialize_rigid_bodies=MethodType(initialize,solver)
    solver.compression_reference_candidate=True

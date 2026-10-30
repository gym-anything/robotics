"""Environment solver hook for the shared physical WSG belt."""
import warp as wp
from robots.attachments.wsg50.mechanism import Mechanism as SharedMechanism,contribution,dual
from . import bounded_drive
from .bounded_drive import compile_tree
from .rigid_schedule import plan
wp.set_module_options({'enable_backward':False,'max_unroll':0},module=bounded_drive)

class Mechanism(SharedMechanism):
    def __init__(self,solver,pairs=None,rho=10000.):
        super().__init__(solver,pairs,rho,compile_tree=compile_tree,schedule_plan=plan)

"""Reuse the bounded native rigid-body recorder with accurate task metadata."""

from pencap.newton15.trace import Trace as RigidTrace


class Trace(RigidTrace):
    def __init__(self, env, path):
        super().__init__(env, path, source="lockkey-newton15",
            contact_description="Shape-pair transitions from all native collision contacts")

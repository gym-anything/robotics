"""Start of unlock: the bench as the native scene builds it (newton15/assembly.py), nothing moved: both keys on their
stands beside the padlock on the chest, the padlock locked with its shackle closed, the Panda at HOME.

setup(config, contract=None) receives the backend's Config and contract (unlock takes none: catalogue_v2 rejects one)
and returns the native environment, lockkey.newton15.env.LockKeyEnv, which builds this start itself on construction
and on every reset."""


def setup(config, contract=None):
    from lockkey.newton15.env import LockKeyEnv
    return LockKeyEnv(config)

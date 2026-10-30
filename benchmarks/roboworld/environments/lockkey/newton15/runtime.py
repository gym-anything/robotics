"""Pinned engine with a short Windows kernel-cache path; no GUI subprocesses."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "lockkey" / ".cache"


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    CACHE.mkdir(parents=True, exist_ok=True)
    short = (ROOT.drive + "/lkw") if os.name == "nt" else str(CACHE / "warp")
    wp.config.kernel_cache_dir = os.environ.get("LOCKKEY_WARP_CACHE", short)
    newton.use_coord_layout_targets = True
    return newton, wp

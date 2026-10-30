"""Pinned engine and local caches; never starts a GUI or changes other sessions."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "pencap" / ".cache"


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0, got {newton.__version__}")
    CACHE.mkdir(parents=True, exist_ok=True)
    # NVRTC still opens generated PTX through MAX_PATH-limited Windows APIs.
    # Newton's long specialized kernel names require a short cache prefix.
    kernel_cache = Path(os.environ.get("PENCAP_WARP_CACHE", (ROOT.drive + "/pcw")
                                     if os.name == "nt" else str(CACHE / "warp")))
    wp.config.kernel_cache_dir = str(kernel_cache)
    newton.use_coord_layout_targets = True
    return newton, wp

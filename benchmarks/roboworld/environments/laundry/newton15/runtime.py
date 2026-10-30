"""Isolated physics cache and version check; no GUI or unrelated services."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "laundry" / "output" / "cache"


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0" or wp.__version__ != "1.16.0":
        raise RuntimeError("Run laundry/bootstrap_newton15.ps1: Newton 1.5.0 / Warp 1.16.0 required")
    # Short Windows paths avoid NVRTC MAX_PATH failures. This is laundry-only.
    folder = Path(os.environ.get("LAUNDRY_WARP_CACHE", ROOT.drive+"/ln15w" if os.name == "nt" else str(CACHE/"warp")))
    folder.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(folder)
    newton.use_coord_layout_targets = True
    return newton, wp

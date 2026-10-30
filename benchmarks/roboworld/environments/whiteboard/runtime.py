"""Bounded workspace-local caches; never silently change the physics version."""

from importlib.metadata import version
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure():
    cache = ROOT / "whiteboard" / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    for key in (
        "TEMP",
        "TMP",
        "CUDA_CACHE_PATH",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "WARP_CACHE_PATH",
    ):
        path = cache / key.lower()
        path.mkdir(exist_ok=True)
        os.environ[key] = str(path)
    return cache


def require_newton():
    actual = version("newton")
    if actual != "1.5.0":
        raise RuntimeError(f"This environment requires Newton 1.5.0; found {actual}.")
    cache = configure()
    import warp as wp

    wp.config.kernel_cache_dir = str(cache / "warp")
    import newton

    newton.use_coord_layout_targets = True
    return newton, wp

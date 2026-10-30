"""gift_bags: what a process loaded, noted when the package is first imported.

Every evidence record carries a stamp of the code its run used (gift_bags.probes.run_stamp). A run may import
gift_bags.probes only when it writes its record, at the end, when the files on disk may differ from those it loaded.
The source hashes, the clock and the container's CPU counters are therefore read HERE, in the package's own import,
which no run can do later than its first use of the package. Nothing else happens here (no git, no physics import)."""
import hashlib as _hashlib
import time as _time
from pathlib import Path as _Path

_PKG = _Path(__file__).resolve().parent


def _source_hashes():
    """{file name: SHA-256} of every Python source directly in the package, as on disk now."""
    out = {}
    for f in sorted(_PKG.glob("*.py")):
        try:
            out[f.name] = _hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            pass
    return out


def _cpu_stat():
    """(wall clock s, scheduler periods, throttled periods, throttled s) of this container's CPU quota, or None."""
    try:
        d = dict(l.split() for l in _Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
        return (_time.time(), int(d["nr_periods"]), int(d["nr_throttled"]), int(d["throttled_usec"])/1e6)
    except Exception:
        return None


try:
    AT_IMPORT = dict(time=_time.time(), sources=_source_hashes(), cpu=_cpu_stat())
except Exception as _exc:                                    # custody code must never stop a run
    AT_IMPORT = dict(time=_time.time(), sources={}, cpu=None, error=repr(_exc))

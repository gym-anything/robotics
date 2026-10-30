"""Ownership-scoped render cleanup. Never signal a simulator or scan/kill GPUs."""
import logging
import subprocess
import weakref
from contextlib import contextmanager
from whiteboard.ipc import send

log = logging.getLogger(__name__)


def stop_worker(process, connection=None, logfile=None, *, graceful=True,
                grace_s=15., terminate_s=5., kill_s=5.):
    """Close, TERM, KILL if needed, and reap exactly this owned Popen child.

    Socket writes are bounded too. Cleanup still runs on cancellation; failures
    are raised, not converted to a closed flag while a worker remains alive.
    """
    escalations = []
    try:
        if process is None:
            return
        if process.poll() is None and graceful and connection is not None:
            try:
                connection.settimeout(min(2., grace_s))
                send(connection, {"op": "close"})
                process.wait(timeout=grace_s)
            except (OSError, subprocess.TimeoutExpired):
                escalations.append("graceful shutdown failed")
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=terminate_s)
            except subprocess.TimeoutExpired:
                escalations.append("TERM timed out; using KILL")
    finally:
        # Also reached when Ctrl-C interrupts graceful shutdown or TERM/wait.
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                process.wait(timeout=kill_s)
        finally:
            try:
                if connection is not None:
                    connection.close()
            finally:
                if logfile is not None:
                    logfile.close()
        if escalations:
            log.warning("Renderer PID %s: %s; child reaped", process.pid, "; ".join(escalations))


def _connection(owner):
    return getattr(owner, "connection", getattr(owner, "socket", None))


def watch_worker(owner):
    """GC/normal-interpreter-exit safety net; explicit batch close is primary."""
    owner._worker_finalizer = weakref.finalize(
        owner, stop_worker, owner.process, _connection(owner), owner.log)


def close_worker(owner, *, graceful=True):
    stop_worker(getattr(owner, "process", None), _connection(owner),
                getattr(owner, "log", None), graceful=graceful)
    finalizer = getattr(owner, "_worker_finalizer", None)
    if finalizer is not None:
        finalizer.detach()


@contextmanager
def renderer_operation(owner, attribute="renderer"):
    """Release one observation/batch, retaining the exact simulator object.

    The caller holds its existing environment/session lock. Entry may precede
    creation, so a subsequent payload or capture error also releases the worker.
    """
    try:
        yield
    finally:
        renderer = getattr(owner, attribute, None)
        if renderer is not None:
            try:
                renderer.close()
            finally:
                setattr(owner, attribute, None)


def _pidfd_open(pid):
    # uv's portable Python may predate the headers exposing os.pidfd_open,
    # even when this host kernel/libc support it. Use the installed libc API.
    import os
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_open
    except AttributeError:
        # RHEL 9/Babel: the kernel implements pidfd_open but glibc 2.34 and
        # the portable interpreter expose neither wrapper. Keep the same
        # identity-safe kernel primitive, not a PID poll or disabled cleanup.
        # Linux x86-64 UAPI asm/unistd_64.h defines __NR_pidfd_open = 434.
        import errno
        import platform
        import sys
        if (sys.platform != "linux" or platform.machine() != "x86_64"
                or ctypes.sizeof(ctypes.c_void_p) != 8):
            raise OSError(errno.ENOSYS, "pidfd_open wrapper unavailable on this architecture")
        function = libc.syscall
        function.argtypes = (ctypes.c_long, ctypes.c_int, ctypes.c_uint)
        function.restype = ctypes.c_long
        fd = function(434, pid, 0)
    else:
        function.argtypes = (ctypes.c_int, ctypes.c_uint)
        function.restype = ctypes.c_int
        fd = function(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return fd


def watch_parent():
    """Exit a Linux render-only worker if its owning process dies.

    pidfd follows process identity (no PID-reuse race), and waits for the whole
    process, not the short-lived HTTP thread that created the child. In
    particular, PR_SET_PDEATHSIG would incorrectly follow that creator thread.
    Non-Linux workers retain explicit close/IPC disconnect cleanup.
    """
    import os
    import select
    import sys
    import threading
    if sys.platform != "linux":
        return
    parent = os.getppid()
    if parent == 1:
        raise RuntimeError("Render owner exited before worker startup")
    fd = _pidfd_open(parent)
    if os.getppid() != parent:
        os.close(fd)
        raise RuntimeError("Render owner exited during worker startup")
    def wait_owner():
        try:
            select.select([fd], [], [])
        finally:
            os.close(fd)
        # Owner is dead: do not wait indefinitely inside Kit shutdown. Only
        # this disposable render child exits; no simulator or sibling signal.
        os._exit(1)
    try:
        threading.Thread(target=wait_owner, daemon=True, name="render-owner-lifetime").start()
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def interrupt_on_terminate():
    """CLI-only SIGTERM -> unwinding, so job finally blocks run on cancellation."""
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise InterruptedError("Job received SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)

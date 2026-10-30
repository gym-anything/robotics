"""Real child shutdown tests; no simulated physics or visual-quality claim."""
import gc
import os
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from gym_anything_robotics.render_lifecycle import (
    stop_worker, close_worker, watch_worker, renderer_operation, _pidfd_open)


class LifetimeTests(unittest.TestCase):
    def child(self, stubborn=False):
        code = ("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "print('ready',flush=True); time.sleep(120)" if stubborn else
                "import time; print('ready',flush=True); time.sleep(120)")
        p = subprocess.Popen([sys.executable, '-u', '-c', code], stdout=subprocess.PIPE)
        self.addCleanup(lambda: stop_worker(p, graceful=False, terminate_s=.1))
        self.assertEqual(p.stdout.readline(), b'ready\n')
        self.addCleanup(p.stdout.close)
        return p

    def test_graceful_protocol_closes_and_reaps(self):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(5)
        self.addCleanup(listener.close)
        code=("import socket; from whiteboard.ipc import receive; "
              f"s=socket.create_connection(('127.0.0.1',{listener.getsockname()[1]})); "
              "assert receive(s)=={'op':'close'}; s.close()")
        p=subprocess.Popen([sys.executable,'-c',code])
        self.addCleanup(lambda:stop_worker(p,graceful=False))
        connection,_=listener.accept()
        stop_worker(p,connection,grace_s=5)
        self.assertEqual(p.returncode,0)
        self.assertEqual(connection.fileno(),-1)

    @unittest.skipUnless(sys.platform=='linux','Linux pidfd ownership')
    def test_owner_process_death_but_not_creator_thread_exit(self):
        import ctypes,json,select
        # Reap our own orphaned grandchild, rather than leave a zombie to PID1.
        libc=ctypes.CDLL(None,use_errno=True);previous=ctypes.c_int()
        self.assertEqual(libc.prctl(37,ctypes.byref(previous),0,0,0),0)
        self.assertEqual(libc.prctl(36,1,0,0,0),0)
        self.addCleanup(lambda:libc.prctl(36,previous.value,0,0,0))
        child="from gym_anything_robotics.render_lifecycle import watch_parent; import time; watch_parent(); print('ready',flush=True); time.sleep(120)"
        owner="import subprocess,sys,threading,json,time\n"
        owner+="def run():\n p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"],stdout=subprocess.PIPE)\n print(json.dumps({'pid':p.pid,'ready':p.stdout.readline().decode()}),flush=True)\n"
        owner+="t=threading.Thread(target=run);t.start();t.join();time.sleep(120)"
        p=subprocess.Popen([sys.executable,'-u','-c',owner],stdout=subprocess.PIPE)
        self.addCleanup(p.stdout.close);self.addCleanup(lambda:stop_worker(p,graceful=False))
        self.assertTrue(select.select([p.stdout],[],[],5)[0])
        ready=json.loads(p.stdout.readline());self.assertEqual(ready['ready'],'ready\n')
        pid=ready['pid'];fd=_pidfd_open(pid);self.addCleanup(lambda:os.close(fd))
        try:
            self.assertFalse(select.select([fd],[],[],.1)[0],'Exited when creator thread ended')
            stop_worker(p,graceful=False)
            self.assertTrue(select.select([fd],[],[],5)[0],'Survived owner process exit')
        finally:
            if not select.select([fd],[],[],0)[0]:os.kill(pid,signal.SIGKILL)
            os.waitpid(pid,0)

    def test_term_reaps_child(self):
        p = self.child()
        stop_worker(p, graceful=False, terminate_s=.1)
        self.assertIsNotNone(p.returncode)
        if os.name == 'posix':
            with self.assertRaises(ChildProcessError):os.waitpid(p.pid, os.WNOHANG)

    @unittest.skipUnless(os.name == 'posix', 'POSIX signal behavior')
    def test_stubborn_child_is_killed_and_reaped(self):
        p = self.child(stubborn=True)
        with self.assertLogs('gym_anything_robotics.render_lifecycle',level='WARNING'):
            stop_worker(p, graceful=False, terminate_s=.05)
        self.assertEqual(p.returncode,-signal.SIGKILL)

    def test_cancel_during_close_still_reaps(self):
        p = self.child()
        connection = socket.socket()
        with patch('gym_anything_robotics.render_lifecycle.send',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):stop_worker(p,connection)
        self.assertIsNotNone(p.returncode)
        self.assertEqual(connection.fileno(),-1)

    def test_gc_safety_net_releases_forgotten_worker(self):
        class Owner:pass
        p = self.child();o=Owner();o.process=p;o.log=None
        watch_worker(o)
        del o;gc.collect()
        self.assertIsNotNone(p.returncode)

    def test_explicit_close_detaches_finalizer(self):
        class Owner:pass
        p=self.child();o=Owner();o.process=p;o.log=None
        watch_worker(o);close_worker(o);close_worker(o)
        self.assertFalse(o._worker_finalizer.alive)

    def test_operation_cancellation_clears_only_renderer(self):
        class Worker:
            closed=False
            def close(self):self.closed=True
        w=Worker();owner=SimpleNamespace(renderer=w,tick=77)
        with self.assertRaises(KeyboardInterrupt):
            with renderer_operation(owner):raise KeyboardInterrupt
        self.assertIsNone(owner.renderer);self.assertTrue(w.closed);self.assertEqual(owner.tick,77)


if __name__=='__main__':unittest.main()

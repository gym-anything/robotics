"""Observer-only pinned readbacks; no simulation arrays are written.

Pinned CPU destinations make Warp device copies asynchronous. Queue a known
batch on the same null stream used by array.numpy(), wait once, then copy the
CPU views so returned snapshots cannot change when the staging buffer is reused.
Reference: https://nvidia.github.io/warp/v1.15/deep_dive/concurrency.html
Exact numpy()/copy() behavior also checked in the installed Warp 1.16 source.
"""
import numpy as np
import warp as wp

class HostReads:
    def __init__(self):
        self.buffers={}

    def capture(self,arrays):
        arrays={id(a):a for a in arrays if a is not None}
        pending={};streams={};result={}
        for key,array in arrays.items():
            if not array.device.is_cuda or not array.is_contiguous:
                result[key]=array.numpy().copy();continue
            if key not in self.buffers:
                # Holding the source reference also prevents object-id reuse.
                if len(self.buffers)>=128:
                    result[key]=array.numpy().copy();continue
                host=wp.empty(array.shape,dtype=array.dtype,device="cpu",pinned=True)
                self.buffers[key]=(array,host)
            source,host=self.buffers[key]
            if source is not array:raise AssertionError("Readback source identity changed")
            stream=array.device.null_stream
            wp.copy(host,array,stream=stream)
            pending[key]=host;streams[str(array.device)]=stream
        for stream in streams.values():wp.synchronize_stream(stream)
        for key,host in pending.items():result[key]=np.asarray(host).copy()
        return result

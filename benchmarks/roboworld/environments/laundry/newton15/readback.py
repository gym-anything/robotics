"""Reusable pinned buffers: copy each requested array, synchronize once."""
import warp as wp


class Readback:
    def __init__(self):
        self.buffers = {}

    def read(self, arrays):
        devices = set()
        for name, source in arrays.items():
            old = self.buffers.get(name)
            if old is None or old.shape != source.shape or old.dtype != source.dtype:
                old = wp.empty(source.shape, dtype=source.dtype, device="cpu", pinned=True)
                self.buffers[name] = old
            wp.copy(old, source)
            if source.device.is_cuda:
                devices.add(source.device.alias)
        for device in devices:
            wp.synchronize_device(device)
        # Views live until the next call with these names. Consumers serialize
        # the sample immediately; no deferred reference to mutable history.
        return {name:self.buffers[name].numpy() for name in arrays}

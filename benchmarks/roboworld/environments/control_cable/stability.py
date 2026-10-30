"""Finite-window physical motion, not a single float32 substep derivative.

The position envelope includes EVERY physics step, not only its endpoints:
out-and-back movement cannot cancel. Raw solver speed remains independently
reported by Progress. No force, damping, pose or solver state is modified.
"""
import math
import numpy as np


class MotionWindow:
    def __init__(self,dt,positions,*,seconds=.05):
        steps=seconds/dt
        if not math.isclose(steps,round(steps),rel_tol=0,abs_tol=1e-8):
            raise ValueError("Motion window must be a whole number of physics ticks")
        if steps<1:raise ValueError("Motion window requires at least one step")
        self.steps=round(steps);self.dt=dt;self.seconds=self.steps*dt
        self.frames=np.empty((self.steps+1,*positions.shape),dtype=positions.dtype)
        self.frames[0]=positions;self.count=1;self.cursor=1
        self.rate=None

    def update(self,positions):
        if positions.shape!=self.frames.shape[1:] or not np.isfinite(positions).all():
            raise ValueError("Motion-window body inventory changed or became non-finite")
        self.frames[self.cursor]=positions
        self.cursor=(self.cursor+1)%len(self.frames)
        self.count=min(self.count+1,len(self.frames))
        if self.count<len(self.frames):return None
        extent=np.ptp(self.frames,axis=0)
        self.rate=float(np.linalg.norm(extent,axis=1).max(initial=0.)/self.seconds)
        return self.rate

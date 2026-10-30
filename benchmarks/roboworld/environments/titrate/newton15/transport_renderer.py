"""Persistent render-only scene using actual interior volume transport."""
import base64
import io
import threading
import time
import numpy as np
from pxr import Usd, Gf
from PIL import Image
from .transport_scene import LaboratoryScene


class TransportRenderer:
    def __init__(self, scene, width=1280, height=720, samples=32, **kwargs):
        if any(type(n) is not int or n<=0 for n in (width,height,samples)):
            raise ValueError('Positive integer image dimensions and samples required')
        self.width,self.height,self.samples = width,height,samples
        self.stage = Usd.Stage.Open(str(scene))
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.adapter = LaboratoryScene(self.stage)
        self.lock,self.closed,self.sequence = threading.Lock(),False,0
        self.denoiser = self.adapter.mi.OptixDenoiser([width,height])
        from .volume_optics import read_field, clear_field
        archived = read_field(self.stage)
        self.has_archived_field = archived is not None
        self.field = archived or clear_field()
        self.liquid = None

    def observe(self,tick,body_poses,cameras,liquid=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('Renderer is closed')
            start = time.perf_counter()
            for index,q in enumerate(body_poses):
                prim = self.stage.GetPrimAtPath(f'/Lab/body_{index}')
                prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*map(float,q[:3])))
                prim.GetAttribute('xformOp:orient').Set(Gf.Quatf(float(q[6]),Gf.Vec3f(*map(float,q[3:6]))))
            if liquid is not None:
                if liquid.pink_volume is None and liquid.pink_molar is not None:
                    raise ValueError('Interior concentration required; refusing boundary-only colour')
                from .volume_optics import clear_field
                self.field = liquid.pink_volume or clear_field()
                self.liquid = liquid
            elif self.liquid is None and not self.has_archived_field:
                legacy = self.stage.GetPrimAtPath('/Lab/liquid').GetAttribute('primvars:pink_molar')
                if legacy and legacy.HasValue() and np.any(np.asarray(legacy.Get())>0):
                    raise ValueError('Archived surface colour lacks a 3D field; provide original particle chemistry')
            images,timings = {},{}
            for name in cameras:
                began = time.perf_counter()
                scene = self.adapter.build(name,self.width,self.height,self.liquid,self.field)
                radiance = self.adapter.mi.render(scene,spp=self.samples,seed=0)
                linear = np.asarray(radiance)
                if not np.isfinite(linear).all():
                    raise RuntimeError('Non-finite volume-transport image')
                self.last_linear = linear.copy()
                denoised = np.asarray(self.denoiser(radiance))
                # Fixed exposure for this radiometric adapter; same for every
                # camera/frame/control. No histogram adaptation or colour boost.
                exposed = np.maximum(denoised,0.)/512.
                display = np.where(exposed<=.0031308,12.92*exposed,1.055*exposed**(1/2.4)-.055)
                pixels = np.uint8(np.round(np.clip(display,0,1)*255))
                stream = io.BytesIO()
                Image.fromarray(pixels).save(stream,format='PNG')
                images[name] = base64.b64encode(stream.getvalue()).decode()
                timings[name] = time.perf_counter()-began
            self.sequence += 1
            return dict(ok=True,tick=tick,sequence=self.sequence,physics_steps=0,images=images,
                optical_field=dict(representation='world-space RGB extinction; heterogeneous medium at dielectric boundary',
                    counts_xyz=self.field['counts_xyz'],spacing_m=self.field['spacing_m'],
                    maximum_molar=float(self.field['values'].max())),
                renderer_settings=dict(engine='Mitsuba 3.8.0',variant='cuda_ad_rgb',integrator='volpathmis',
                    samples=self.samples,seed=0,denoiser='OptiX',linear_exposure_divisor=512.),
                stage_times_s=timings,wall_s=time.perf_counter()-start)

    def close(self):
        with self.lock:
            self.closed = True
            self.adapter = self.stage = self.denoiser = self.field = self.liquid = None
            self.last_linear = None

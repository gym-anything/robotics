"""The scanner's imager: geometry, lens, light sources and sensor.

Rendering (Isaac RTX path tracing, render_worker.py) produces the linear RGB radiance reaching the
entrance pupil through the thin lens (depth of field included). This module turns that into the 8-bit
monochrome frame the decoder sees:

    L_mono = w_R L_R + w_G L_G + w_B L_B                 (sensor response to the three render bands)
    diffraction: Gaussian, sigma = 0.42 lambda / A        (angle; the renderer has no diffraction)
    N_e ~ Poisson(K t_e L_mono) + Normal(0, sigma_read)    (photoelectrons and read noise, t_e exposure)
    DN = clip(round(255 g N_e / FULL_WELL), 0, 255)        (analog gain g after the photon count, 8-bit ADC, glare saturates)

Auto-exposure sets t_e (<= the mode's motion-limited ceiling) and then gain from the previous frame, as a
real imager's firmware does. All constants are in params.py with their provenance.
"""

import math
import numpy as np
from . import params as P


class Imager:
    def __init__(self, mode="handheld", rng=None):
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.t_e = P.T_EXPOSURE_START_S
        self.gain = 1.0

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def ifov():
        return 2*math.tan(math.radians(P.HFOV_DEG/2))/P.WIDTH_PX

    @staticmethod
    def diffraction_sigma_px():
        return 0.42*P.WAVELENGTH_M/P.PUPIL_DIAMETER_M/Imager.ifov()

    @staticmethod
    def defocus_diameter_rad(d):
        """Thin-lens blur-circle diameter (angle) of a point at distance d (m) from the entrance pupil."""
        return P.PUPIL_DIAMETER_M*abs(1/d - 1/P.FOCUS_DISTANCE_M)

    # ---------------------------------------------------------------- sensor
    def t_max(self):
        return P.T_MAX_HANDHELD_S if self.mode == "handheld" else P.T_MAX_HANDSFREE_S

    def mono(self, rgb):
        w = np.asarray(P.SENSOR_WEIGHTS, np.float32)
        return np.tensordot(np.asarray(rgb, np.float32)[..., :3], w, axes=([-1], [0]))

    def expose(self, radiance_mono, *, noise=True):
        """8-bit frame from a linear mono radiance image with the current exposure/gain; then update the
        auto-exposure for the next frame. Returns (frame uint8, record)."""
        from scipy.ndimage import gaussian_filter
        L = np.maximum(np.asarray(radiance_mono, np.float64), 0.)
        s = self.diffraction_sigma_px()
        if s > 0.05:
            L = gaussian_filter(L, s, mode="nearest")
        # photoelectrons are counted before the analog gain amplifies them, so shot noise in DN grows with the gain
        signal = P.SENSOR_K*self.t_e*L                              # expected photoelectrons
        if noise:
            prnu = 1 + P.PRNU*self._prnu_map(L.shape)
            electrons = self.rng.poisson(np.maximum(signal*prnu, 0)).astype(np.float64)
            electrons += self.rng.normal(0., P.READ_NOISE_E, L.shape)
        else:
            electrons = signal
        dn = np.clip(np.round(255.*self.gain*electrons/P.FULL_WELL_E), 0, 255).astype(np.uint8)
        record = dict(exposure_s=self.t_e, gain=self.gain, saturated_fraction=float((dn == 255).mean()))
        self._auto_expose(dn)
        return dn, record

    def _prnu_map(self, shape):
        if getattr(self, "_prnu", None) is None or self._prnu.shape != shape:
            self._prnu = np.random.default_rng(12345).normal(0, 1, shape)      # fixed per sensor
        return self._prnu

    def _auto_expose(self, dn):
        h, w = dn.shape
        roi = dn[int(.35*h):int(.65*h), int(.35*w):int(.65*w)].astype(np.float64)
        mean = max(float(roi.mean()), 1.)
        # total "brightness" t_e * g needed to bring the ROI mean to the target; clip to the sensor's ranges
        want = self.t_e*self.gain*P.AE_TARGET_DN/mean
        if self.mode != "handheld":
            # hands-free highlight protection: no aim marks the target, and an item swept past the window off the
            # metering window (a lip balm) clipped white, bars and all, while the dark room drove the gain up. The
            # brightest 0.5 % of the frame's 8 x 8-pixel block means is held at AE_HIGHLIGHT_DN, and the
            # cap is predictive (the sensor is linear, so the next highlight is hi * want / (t_e g)): capping only once
            # the highlight was already over it let a dim centre's demand (x8) overshoot into clipping, then halve
            # back, a four-frame limit cycle. Block means, as a sensor's AE histogram is gathered, because the
            # per-pixel percentile also carries the render's Monte Carlo noise (~12 % per pixel) and exposed ~10 %
            # low. Handheld mode meters the aimed centre only: highlight protection there darkened the aimed code
            # under the LEDs' near-range hot band and moved the near limits out.
            B = P.AE_BLOCK_PX
            blocks = dn[:h//B*B, :w//B*B].reshape(h//B, B, w//B, B).mean(axis=(1, 3))
            hi = float(np.percentile(blocks, 99.5))
            if hi >= 250.:               # clipped: how far over is unknown, so halve (a 10 % step per frame lagged)
                want = min(want, self.t_e*self.gain*.5)
            else:
                want = min(want, self.t_e*self.gain*P.AE_HIGHLIGHT_DN/max(hi, 1.))
        if (roi >= 254).mean() > .5:                     # saturated: step down hard
            want = self.t_e*self.gain*.25
        t = min(max(want, P.T_MIN_S), self.t_max())
        g = min(max(want/t, 1.), P.GAIN_MAX)
        self.t_e, self.gain = t, g

    def state(self):
        return dict(mode=self.mode, exposure_s=self.t_e, gain=self.gain)

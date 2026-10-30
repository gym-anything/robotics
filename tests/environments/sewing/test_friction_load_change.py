"""A contact pressed harder than ever before keeps its tangential force. The elastic-plastic friction stores each
contact's stick displacement d and applies mu N d / delta. Carried as a bare displacement, a d stored under a light load
would come back multiplied by a later rise of N, and a ply that had slid on the feed dog's teeth under its own weight
would lock sideways force between the teeth and the presser foot once the foot came down. Rescaled on every rise
against the previous tick's load, a contact whose load jitters would lose force on every cycle and a draped stack would
creep off the table's edge. The stored displacement carries the peak load since its stick state was set, and only a
rise above that peak rescales it (on the cloth's contacts with shapes; ply on ply keeps the displacement). The rescale
and store kernels are checked here.
Run: python -m pytest tests/environments/sewing/test_friction_load_change.py"""

import numpy as np
import warp as wp

from sewing import params as P
from sewing.cloth import ep_force, k_fric_rescale, k_fric_store_n

DELTA = P.FRICTION_PRESLIP


@wp.kernel
def _force(d: wp.array(dtype=wp.vec3), muN: wp.array(dtype=float), delta: float, out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = ep_force(d[i], muN[i], delta)


def _ep(d, muN):
    out = wp.zeros(len(d), dtype=wp.vec3)
    wp.launch(_force, len(d), [wp.array(np.asarray(d, np.float32), dtype=wp.vec3),
                               wp.array(np.asarray(muN, np.float32), dtype=float), float(DELTA), out])
    return out.numpy()


def _rescale(d, n_peak, n_now, sid_prev=3, sid_now=3):
    """One tick start: returns (the stored displacement after the rescale, the peak carried to the store kernel)."""
    fe_cur = wp.array(np.array([[d, (0, 0, 0), (0, 0, 0), (0, 0, 0)]], np.float32), dtype=wp.vec3)
    np_cur = wp.zeros((1, 4), dtype=float)
    wp.launch(k_fric_rescale, 1, [wp.array([1], dtype=int),
                                  wp.array(np.array([[n_now, 0, 0, 0]], np.float32), dtype=float),
                                  wp.array(np.array([[sid_now, -1, -1, -1]], np.int32), dtype=int),
                                  wp.array(np.array([[n_peak, 0, 0, 0]], np.float32), dtype=float),
                                  wp.array(np.array([[sid_prev, -1, -1, -1]], np.int32), dtype=int),
                                  fe_cur, np_cur, float(DELTA)])
    return fe_cur.numpy()[0, 0], float(np_cur.numpy()[0, 0])


def _store(d_after_return, n_now, n_peak_cur):
    """One tick end: the peak stored for the next tick."""
    out = wp.zeros((1, 4), dtype=float)
    wp.launch(k_fric_store_n, 1, [wp.array([1], dtype=int),
                                  wp.array(np.array([[n_now, 0, 0, 0]], np.float32), dtype=float),
                                  wp.array(np.array([[n_peak_cur, 0, 0, 0]], np.float32), dtype=float),
                                  wp.array(np.array([[d_after_return, (0, 0, 0), (0, 0, 0), (0, 0, 0)]], np.float32),
                                           dtype=wp.vec3), float(DELTA), out])
    return float(out.numpy()[0, 0])


def test_a_contact_pressed_harder_than_its_peak_keeps_its_tangential_force():
    mu, d = 1.0, (0.5*DELTA, 0.2*DELTA, 0.0)
    for n_peak, n_now in ((1e-3, 20.0), (2.0, 20.0)):
        before = _ep([d], [mu*n_peak])[0]
        dn, peak = _rescale(d, n_peak, n_now)
        after = _ep([dn], [mu*n_now])[0]
        assert np.allclose(after, before, rtol=1e-5, atol=1e-9), (n_peak, n_now, before, after)
        assert peak == n_now


def test_below_its_peak_a_contact_keeps_its_displacement_and_its_force_follows_the_load():
    mu, d = 0.15, (0.5*DELTA, 0.1*DELTA, 0.0)
    for n_now in (1.0, 19.9, 20.0):
        dn, peak = _rescale(d, 20.0, n_now)
        assert np.allclose(dn, d) and peak == 20.0
        assert np.allclose(_ep([dn], [mu*n_now])[0], np.array(d)*mu*n_now/DELTA, rtol=1e-5)


def test_a_jittering_load_does_not_erode_the_stored_force():
    """A rule against the previous tick's load would lose force on every up-down cycle (a draped stack creeps);
    against the peak, a thousand cycles of +-0.5 % leave the force where the first rise set it."""
    mu, d, peak = 0.3, np.array([0.6*DELTA, 0.0, 0.0]), 0.0
    N0 = 0.05
    f_first = None
    for k in range(1000):
        n_now = N0*(1.005 if k % 2 else 0.995)
        d, peak = _rescale(tuple(d), peak, n_now)
        d = np.array(d)
        peak = _store(tuple(d), n_now, peak)
        f = _ep([d], [mu*N0*1.005])[0]          # the force at the jitter's top
        f_first = f if f_first is None and k == 1 else f_first
    assert np.allclose(f, f_first, rtol=1e-4), (f, f_first)


def test_a_slip_sets_the_stick_state_again_at_the_present_load():
    assert _store((DELTA, 0.0, 0.0), 3.0, 20.0) == 3.0                # slid this tick: the peak is this load
    assert _store((0.5*DELTA, 0.0, 0.0), 3.0, 20.0) == 20.0           # stuck: the peak is kept


def test_a_new_contact_or_another_shape_is_not_rescaled():
    d = (0.3*DELTA, 0.0, 0.0)
    assert np.allclose(_rescale(d, 0.0, 20.0)[0], d)                  # nothing stored at a load
    assert np.allclose(_rescale(d, 1e-3, 20.0, sid_prev=2, sid_now=3)[0], d)   # a different shape's record



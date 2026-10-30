"""Host-side geometry of the machine: the needle-versus-hand strike test (a segment against an oriented box).
Run: python -m pytest tests/environments/sewing/test_machine_geometry.py"""

import math

import numpy as np

from sewing.machine import _quat_matrix, _segment_hits_box


def test_segment_box_cases():
    h = np.array([0.01, 0.02, 0.005])
    # vertical needle through the box centre, from below to above: hit
    assert _segment_hits_box(np.array([0., 0., -0.02]), np.array([0., 0., 0.02]), h)
    # entirely above the box: miss
    assert not _segment_hits_box(np.array([0., 0., 0.006]), np.array([0., 0., 0.03]), h)
    # beside the box (x beyond the half extent): miss, even though z spans it
    assert not _segment_hits_box(np.array([0.0101, 0., -0.02]), np.array([0.0101, 0., 0.02]), h)
    # tip just inside the top face: hit
    assert _segment_hits_box(np.array([0.005, 0.01, 0.0049]), np.array([0.005, 0.01, 0.03]), h)
    # degenerate (point) segment inside and outside
    assert _segment_hits_box(np.zeros(3), np.zeros(3), h)
    assert not _segment_hits_box(np.array([0., 0.03, 0.]), np.array([0., 0.03, 0.]), h)


def test_rotated_box_through_quaternion():
    # a finger box 20 mm long along its x, rotated 90 deg about z: it now extends along world y
    q = (0., 0., math.sin(math.pi/4), math.cos(math.pi/4))
    R = _quat_matrix(q)
    assert np.allclose(R @ np.array([1., 0., 0.]), [0., 1., 0.], atol=1e-12)
    h = np.array([0.02, 0.004, 0.004])
    p = np.array([0., 0.015, 0.002])
    a, b = np.array([0., 0., 0.]), np.array([0., 0., 0.03])          # needle at the origin, vertical
    assert _segment_hits_box(R.T @ (a - p), R.T @ (b - p), h)          # the box reaches back over the needle
    p2 = np.array([0.006, 0.015, 0.002])                               # moved 6 mm right: clear (half width 4 mm)
    assert not _segment_hits_box(R.T @ (a - p2), R.T @ (b - p2), h)


def test_needle_strikes_a_body_riding_finger_at_its_world_position():
    """A compliant hand's pads ride on a body and are posed as offsets from it: the strike test must place them in
    the world (offset + body). Read as world, the offset would put a fingertip a hand's height too low and the needle
    would sew through it."""
    from types import SimpleNamespace
    from sewing.machine import Machine
    body_z = 0.012                                                   # the hand's body over the plate
    finger = dict(on=True, half=(0.004, 0.004, 0.004), body=0)
    cloth = SimpleNamespace(shapes=[finger], _sq_host=np.array([[0., 0., 0., 1.]]),
                            _sp_host=np.array([[0., 0., -0.006]]),    # 6 mm below its body: the fingertip at z = 6 mm
                            x=SimpleNamespace(numpy=lambda: np.array([[0., 0., body_z]])))
    for tip, struck in ((0.020, False), (0.008, True)):             # the point above the fingertip, then inside it
        m = SimpleNamespace(obstacles=[(0, "right/finger")], needle_state="ok", cloth=cloth, faults=[],
                            s=SimpleNamespace(needle_nm=80), speed_ramped=600.0)
        Machine._needle_obstacles(m, 0, tip, np.zeros(2))
        assert (m.needle_state == "broken") == struck, (tip, m.needle_state)

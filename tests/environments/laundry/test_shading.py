"""A shading correction must not silently move the finite-thickness shell."""
import numpy as np
from laundry.newton15.config import Config
from laundry.newton15.garment import load
from laundry.newton15.shading import deform
from laundry.garment_render_mesh import build_garment_panel_render_topology


def test_shading_preserves_geometry_and_interpolates_normals():
    c, p, top, x = load(Config())
    x = x.copy()
    x[:, 2] += .04*np.sin(14*x[:, 0])*np.cos(8*x[:, 1])
    panel = c.panels[0]
    visual = build_garment_panel_render_topology(p, top, panel.id, subdivision_factor=3)
    before, old = deform(visual, x, thickness_m=.003, shading="legacy")
    after, new = deform(visual, x, thickness_m=.003)
    assert np.array_equal(before, after)
    assert np.isfinite(new).all()
    np.testing.assert_allclose(np.linalg.norm(new, axis=1), 1., atol=1e-12)
    assert np.max(abs(new-old)) > .01
    # A physical vertex keeps the same shading normal as with the legacy shading.
    ids = visual.embedding.coarse_vertex_indices
    original = np.all(ids == ids[:, :1], axis=1)
    n = visual.embedding.vertex_count
    np.testing.assert_allclose(new[:n][original], old[:n][original], atol=1e-12)

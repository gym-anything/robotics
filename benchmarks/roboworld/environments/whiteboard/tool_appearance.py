"""Procedural, metric-scale printed markings; no changes to collision or ink.

These are authored label artwork, not photographs or measured optical spectra.
Use Pillow's bundled scalable font for portable, deterministic glyph rasterization.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def marker_label(folder):
    base = (242, 242, 234)
    label = Image.new("RGB", (1600, 350), base)
    draw = ImageDraw.Draw(label)
    blue = (23, 49, 104)
    draw.text((35, 28), "Lumocolor", font=ImageFont.load_default(size=95), fill=blue)
    draw.text((40, 143), "whiteboard", font=ImageFont.load_default(size=63), fill=blue)
    draw.text(
        (42, 242), "351 B    2 - 5 mm    DRY SAFE", font=ImageFont.load_default(size=34), fill=blue
    )
    draw.text((1060, 245), "MADE IN GERMANY", font=ImageFont.load_default(size=25), fill=blue)
    texture = Image.new("RGB", (1024, 2048), base)
    texture.paste(label.transpose(Image.Transpose.ROTATE_90), (337, 225))
    path = folder / "marker_print.png"
    texture.save(path)
    return path


def cylindrical_uv(vertices, faces, height):
    """Face-varying UVs; unwrap triangles across the circular seam locally."""
    result = []
    for face in faces:
        points = vertices[face]
        u = np.arctan2(points[:, 1], points[:, 0]) / (2 * np.pi) + 0.5
        if np.ptp(u) > 0.5:
            u[u < 0.5] += 1
        result.extend(np.c_[u, points[:, 2] / height + 0.5])
    return np.asarray(result)

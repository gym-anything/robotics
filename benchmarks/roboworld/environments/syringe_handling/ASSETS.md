# Asset provenance

What the scene is made of, where each thing came from, and under what licence. Nothing in the scene is a third-party
model of a syringe, needle, vial or trainer: those are authored here from the dimensions in `params.py`, each with its
basis (standards, makers' documents and published measurements).

## Robots

The FR3 and the Panda are Newton's `franka_emika_panda` asset bundle, retrieved by Newton from the
[franka_description](https://github.com/frankarobotics/franka_description) repository at `d9841cf` (the bundle's own
README), licence Apache-2.0 (the bundle's LICENSE). The repository's shared robot catalogue pins the bundle's bytes:
`robots/models/fr3/assets.lock.json` and `robots/models/panda/assets.lock.json` (SHA-256 of every file; `fr3.urdf`
`79f761a5322d64ab6e7c9827d5e57efc593ecffe55522b5bbcc3c700772bd217`). The Franka Hand's fingertip pads are boxes
authored in `scene.py` (the URDF's own boxes, with a rubber pad on each), not manufacturer CAD. The arms' controller
profiles are the catalogue's (`barcode_scanning` for the FR3, `drilling` for the Panda).

## Appearance assets from outside the repository

| asset | used for | source | author | licence | pin |
|---|---|---|---|---|---|
| `kiara_5_noon` HDRI (2k) | the daylight dome behind the window | Poly Haven, `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/kiara_5_noon_2k.hdr` | Greg Zaal | CC0 (`https://polyhaven.com/license`) | SHA-256 `f5709ff2bc42a29a655f3ca3964d76522b7a61846d668d41a3306e204b271a8c`, through `whiteboard/visual_assets.py` |

It is light only: no collider, mass or physical parameter comes from it.

## Authored here

- Textures drawn by code at build time (`dressing.py`, `render.py`): the beech worktop and the flecked vinyl floor
  (tileable noise), the syringes' printed scales (`render.scale_map`: a line every graduation of the syringe's
  `div_ml`, numbered at each millilitre or tenth), the flush syringe's and the vial's labels, the sharps container's
  label, the poster with the task's instruction, the wall clock's face and hands (drawn from the simulated time).
- Geometry: every body the robot touches (tray, basin, syringes, plungers, needles with hubs and sheaths, caps, vial,
  pad, training arm with its saline lock, sharps container with its flap) is built in `scene.py` and `devices.py` from
  `params.py`; the room and its furniture (casework, sink, trunking, bed, curtain, drip stand, cart) are built in
  `scene.py` and `dressing.py`, and what is dressing only (no collider) is listed in `dressing.py`.
- The training arm's forearm and hand are lofted meshes authored in `render.py` (`loft_mesh`): an approximation of a
  trainer, not a scan.

## Known differences from the real things

- The "kidney basin" is a rectangular steel basin of 250 by 120 by 45 mm built from five flat boxes; a real kidney
  dish is a crescent-shaped bowl.
- The vial has no flip-off cap; it stands crimped and bare on a ring printed on the counter.
- The training arm, the cart and the upper-arm model are authored from simple shapes; they are not scans.

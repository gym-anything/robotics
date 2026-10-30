# Asset provenance and reconstruction

## Standard robot

The actual `fr3.urdf` is distributed with Newton's `franka_emika_panda`
asset bundle. Its installed README identifies the source as
`frankarobotics/franka_description` at `d9841cf`, generated without an end
effector, with mesh paths adapted for the bundle. The installed per-asset
license is Apache 2.0, not an inferred license from the bundle's name.

The URDF's SHA-256 is
`79f761a5322d64ab6e7c9827d5e57efc593ecffe55522b5bbcc3c700772bd217`.
The renderer embeds the actual body meshes. The authored WSG
mount/adapter/fingers are explicitly described in `robot.py`, not presented as
manufacturer-supplied CAD. Arm limits are retained.

## Floor scan

The anti-skid floor uses the three bundled 2K diffuse, OpenGL-normal and
roughness maps from [Poly Haven / Amal Kumar](https://polyhaven.com/a/anti_skid_tiles),
recorded as CC0-1.0 in `wiping/assets/sources.json`. That manifest supplies the
original URLs, byte sizes and SHA-256 hashes. The mapping uses the recorded
approximately1.8m scan extent, not a texture stretched independently over each
object. Reusing these appearance assets does not reuse or modify wiping physics.

## Authored scene and optical approximations

`geometry.py`, `tub_geometry.py`, `station.py`, `room.py` and `props.py`
reconstruct the physical objects from the cited dimensions and explicit
engineering estimates. No manufacturer photograph is used as simulation
evidence. Fictional can branding and recycling signage are rasterized by
`graphics.py`; they do not encode target identities or success state.
`film.mdl`, `printed_film.mdl` and the shared polycarbonate shader are authored
optical models. Roughness, illumination and tint are estimates rather than
measured product appearance.

Asset documentation is separate from qualification: a source, license or hash
does not prove physical or visual correctness. Final recorded motion and stationary
states still require direct inspection.

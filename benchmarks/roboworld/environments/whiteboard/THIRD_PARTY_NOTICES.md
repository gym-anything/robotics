# Third-party assets

## Three.js operator viewer

The bootstrap downloads unmodified Three.js r180 ES modules, OrbitControls and
the upstream MIT LICENSE into `.cache/web_vendor`, checking their pinned hashes.
The operator server serves local files, not a runtime CDN dependency.
Upstream: https://github.com/mrdoob/three.js/tree/r180
The full MIT notice accompanies the provisioned modules at
`.cache/web_vendor/LICENSE`. This diagnostic viewer does not replace Newton
physics or the onboard RTX observations.

## Unitree robot

The environment imports the Unitree G1 robot description and meshes from the
revision-pinned upstream files provisioned by `python -m whiteboard.assets` into
`whiteboard/.cache/assets/unitree_ros`. `assets.lock.json` verifies all 51 files
(URDF, referenced meshes and license), independently of other environments.
Selected renders include that robot. Vendor files remain unmodified.

Source: https://github.com/unitreerobotics/unitree_ros

Verified commit: `cdf67e6eb4c98d73202f29a25e3d098e2aa3b247`.

The repository's LICENSE reads:

```text
BSD 3-Clause License

Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
# Additional appearance assets

Poly Haven CC0 assets, https://polyhaven.com/license:

- Oak Veneer 01, Jenelle van Heerden: https://polyhaven.com/a/oak_veneer_01
- Beige Wall 001, Dimitrios Savva and Rico Cilliers: https://polyhaven.com/a/beige_wall_001
- Kiara 5 Noon HDRI, Greg Zaal: https://polyhaven.com/a/kiara_5_noon

Download URLs, exact SHA-256 hashes and physical texture tiling are pinned in
`visual_assets.py`. Cached originals are not vendored or silently replaced.
The veneer texture represents the printed wood appearance of laminate, not a
change to desk physics. The HDRI supplies distant exterior appearance/lighting,
not interactable geometry. These appearance assets do not certify mechanics.

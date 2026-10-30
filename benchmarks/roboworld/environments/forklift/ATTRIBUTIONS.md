# Third-party assets and tools of the forklift environment (binaries are not committed; credits are)

Every scene file is fetched by `python -m forklift.assets.install` and checked against its SHA-256 in
`forklift/assets/assets.lock.json` (2,290 files): 55 NVIDIA Forklift C files, 2,039 NVIDIA Simple_Warehouse files, 144
NVIDIA DigitalTwin files, 1 NVIDIA Skies panorama, 47 Poly Haven files (36 texture maps, 11 model files) and 4 ambientCG
material archives. Which code uses each one is named below; the robot and the build-time tool are installed separately.
This list follows the lock file and the code that loads assets (`geometry.py`, `render.py`, `visuals/*.py`).
Licence texts were read where cited.

## Licences
- **NVIDIA content** (Forklift C, Simple_Warehouse, DigitalTwin, Skies). Author: NVIDIA Corporation. Licence: the NVIDIA
  Omniverse License Agreement, the text Isaac Sim 5.1 ships as `isaacsim/LICENSE.txt` (read, 2026-10-14; latest
  version at https://docs.omniverse.nvidia.com/dev-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).
  Section 1.4 defines Content as "a NVIDIA audio asset, 2D asset or 3D asset"; section 2.1 permits you to "Modify and
  create derivative works of ... the Content" and to "Distribute Content (unmodified or as modified by you) as
  incorporated into your products or services for the purpose of enhancing your work ... but not deploy or distribute
  the Content on a stand-alone basis". This repository does not redistribute the files; each user fetches them from
  NVIDIA's public bucket `https://omniverse-content-production.s3-us-west-2.amazonaws.com/`.
- **Poly Haven** assets: CC0. https://polyhaven.com/license (read, 2026-10-14): "All assets (HDRIs, textures and 3D
  models) on this site are ... licensed as CC0"; "You do not need to give credit or attribution when using them
  (although it is appreciated)". Credited here anyway.
- **ambientCG** materials: CC0 1.0 Universal. https://docs.ambientcg.com/license/ (read, 2026-10-14): "You can copy,
  modify, distribute and perform the assets, even for commercial purposes, all without asking permission"; credit is
  optional. Author: ambientCG.

## Truck: NVIDIA Isaac Sim 5.1 "Forklift C" (55 files)
- Source: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/IsaacSim/ForkliftC/`
  (`forklift_c.usd`, its materials and textures). NVIDIA content licence (above).
- Use: `geometry.py` reads its meshes to build the truck's bodies and collision hulls (re-rigged with Hyster H2.5A
  masses); the render-only Isaac worker references the USD and its materials.
- Changes made in this package (not in the asset):
  - **The overhead guard is raised 0.049 m** above the guard split (`params.GUARD_RAISE`), to the H2.5A's own guard
    height: 2,160 mm on Hyster's sheet (row 4.7, h6) against the asset's 2.111 m slat tops. At this height the seated H1-2's head clears the rear slat by 28.5 mm at rest and
    reaches it in a hard stop (`params.GUARD_RAISE` gives the measurements).
  - The seat stands 75 mm forward on its 150 mm rails (a setting inside the rails' range) and its cushion collider sits
    at the foam's loaded thickness, 30 mm under the operator (`params.SEAT_RAIL`, `SEAT_FOAM_SET`).
  - A horn button is drawn and simulated on the steering-wheel hub (the asset has a moulded cap there).
  - Its decal, glass and lamp meshes carry no collision; the cab interior's collision is a CoACD decomposition (below).

## Warehouse kit: NVIDIA Isaac Sim 5.1 "Simple_Warehouse" (2,039 files)
- Source: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/`.
  The lock holds the kit's whole folder (props, materials, textures) because the props' materials reference shared
  files; the code references only the files below. NVIDIA content licence (above).
- Props referenced (all render only):
  - `Props/SM_CardBoxA_01.usd`, `SM_CardBoxB_01.usd`, `SM_CardBoxC_01.usd`: cartons drawn on stored unit loads
    (`visuals/loads.py`).
  - `Props/SM_PushcartA_02.usd`, `SM_EmergencyBoardFull_01.usd`, `SM_FuseBox_01.usd`, `SM_FuseBox_04.usd`,
    `SM_SignA_02.usd`, `SM_SignB_02.usd`, `SM_SignCVer_01.usd`, `SM_FireExtinguisher_02.usd`: warehouse dressing off
    the routes (`visuals/interior.py`).
  - `Props/S_WetFloorSign.usd`: beside a wet floor patch (`visuals/interior.py`); the sign also has a collider in the
    layout (`scene.py`).
  - `Props/S_TrafficCone.usd`: the yard and the test pad (`visuals/dockyard.py`).
- Textures referenced: `Materials/Textures/T_WallA_01_D.png`, `_N.png`, `_ORM.png` (the lower band of the warehouse
  walls, `visuals/interior.py`).
- Not used: `SM_BeamA_9M`, `SM_LampCeilingA_04`,
  `SM_CardBoxD_04`, `SM_BarelPlastic_A_01`, `S_AisleSign`. The roof beams, ceiling lamps and aisle signs are drawn
  procedurally.

## Warehouse props and base materials: NVIDIA "DigitalTwin" warehouse pack (144 files)
- Source: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/` (listed by prefix in
  `forklift/assets/extra_sources_interior.json`, cached under `nvidia/Assets/DigitalTwin/`). NVIDIA content licence.
- Props (render only), under `Assets/Warehouse/`:
  - `Equipment/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd` (hand pallet truck),
    `Equipment/Hand_Trucks/Convertible_Steel_A/ConvertibleSteelHandTruck_A01_PR_V_NVD_01.usd` (hand truck),
    `Equipment/Fans/Drum_A/DrumFan_A01_01.usd` (drum fan),
    `Equipment/Signage/PlasticHardWiredExit_A/PlasticHardwiredExitSign_A01_Red_01.usd` (exit sign),
    `Storage/Drums/Steel_A/SteelDrum_A01_PR_NVD_01.usd`, `Storage/Drums/Plastic_A/PlasticDrum_A04_PR_V_NVD_01.usd`
    (drums), `Facilities/Trash_Cans/TrashCanWithWheels_A/TrashCanWithWheels_A01_Blue_01.usd` (waste bin),
    `Safety/Fire_Protection/FireAlarm_A/FireAlarm_A01_01.usd` (fire alarm): dressing, `visuals/interior.py`.
  - `Shipping/Cardboard_Boxes/` `Flat_A/FlatBox_A15_51x63x33cm`, `Multi-Depth_A/MultiDepthBox_A05_31x31x31cm`,
    `A10_31x46x31cm`, `A12_36x51x36cm`, `White_A/WhiteCorrugatedBox_A11_46x47x47cm`, `A12_51x52x52cm` (`*_PR_NVD_01.usd`)
    with their materials: cartons on stored and loose loads (`visuals/loads.py`).
  - `Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A05_102x102x103cm_PR_V_NVD_01.usd` and its textures:
    the stretch-film mesh and the pallet-wood and cardboard maps of wrapped unit loads (`visuals/loads.py`).
- Base materials, under `Materials/Base/` (MDL files and their textures): Metals (`Aluminium_Brushed_A`, `_B`,
  `Metal_Glossy_A`, `Metal_Painted_Blue_A`, `Green_B`, `Orange_A`, `Red_A`, `White_Glossy_A`, `Yellow_A`,
  `Metal_Rough_A`, `Steel_A`), Plastics (`Natural_Plastic_Translucent_A`, `Plastic_Black_A`, `Blue_A`, `Glossy_Black_A`,
  `Glossy_White_A`, `Red_A`, `Rough_Black_A`, `White_A`), Rubber (`Rubber_Black_Glossy_A`, `Rubber_New_A`), Wood
  (`Wood_Recycled_A`): referenced by the props above; their painted-steel, steel, rubber and plastic maps are also used
  on drawn parts, the racks, conveyor, guards and machines (`visuals/loads.py`, `visuals/interior.py`; the notes in
  `extra_sources_interior.json`).

## Yard sky: NVIDIA Skies collection, `kloofendal_48d_partly_cloudy.hdr` (1 file)
- Source: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Cloudy/kloofendal_48d_partly_cloudy.hdr`
  (`forklift/assets/extra_sources.json`). Used as the yard preset's dome light (`render.py`, render only). Obtained
  through NVIDIA's collection under its content licence. A Poly Haven HDRI of the same name exists (CC0); that this file
  is that panorama was not checked.

## Poly Haven textures (12 sets, 36 files) and models (3 models, 11 files), CC0
Source URLs `https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/<id>/<id>_{diff,rough,nor_gl}_2k.jpg` and
`https://dl.polyhaven.org/file/ph-assets/Models/{usd,jpg,png}/1k/<id>/...` (`forklift/assets/extra_sources_*.json`);
each asset's page is `https://polyhaven.com/a/<id>`. Authors as Poly Haven's asset index lists them
(`https://api.polyhaven.com/assets?t=textures`, read 2026-10-14) where read; the others are named on the asset's page.
All render only (`visuals/interior.py`, `visuals/dockyard.py`).

| Asset id | Author | Use |
|---|---|---|
| `concrete_floor_worn_001` | see asset page | sealed warehouse floor (3 m tile) |
| `painted_concrete_02` | see asset page | warehouse upper walls (normal and roughness maps); dock wall inside face (4 m tile) |
| `dirty_concrete` | see asset page | dock face, pit walls, building base (3 m tile) |
| `asphalt_02` | Rob Tuytel | yard asphalt, cracked (3 m tile) |
| `asphalt_pit_lane` | Dimitrios Savva | fresh asphalt for yard patches (2 m tile) |
| `asphalt_04` | Sergej Majboroda (photography), Jenelle van Heerden (processing) | concrete apron (4 m tile) |
| `aerial_asphalt_01` | Rob Tuytel | weathered yard asphalt seen from afar (30 m tile) |
| `aerial_grass_rock` | Rob Tuytel | distant ground around the yard (15 m tile) |
| `leafy_grass` | see asset page | grass verge (2 m tile) |
| `gravel_floor_02` | see asset page | gravel shoulder outside the yard fence (2 m tile) |
| `oak_wood_planks` | see asset page | trailer floor (1.2 m tile) |
| `plywood` | see asset page | trailer lining and door inner faces (0.5 m tile) |
| `island_tree_02` (model, 1k USD) | see asset page | trees around the yard and test pad |
| `searsia_lucida` (model, 1k USD) | see asset page | shrubs outside the yard and test-pad fences |
| `wild_rooibos_bush` (model, 1k USD) | see asset page | dry bushes on the surrounding ground |

## ambientCG materials (4 archives), CC0 1.0
Source `https://ambientcg.com/get?file=<id>_2K-JPG.zip`; page `https://ambientcg.com/view?id=<id>`; author ambientCG.
Render only (`visuals/dockyard.py`).

| Material | Use |
|---|---|
| `DiamondPlate004` | galvanised tread plate: trailer threshold plate, yard-ramp lip and approach plate |
| `DiamondPlate008D` | worn steel tread plate: dock leveler deck and lip |
| `PaintedMetal001` | scratched yellow paint: yard-ramp frame and curbs, bollards, track guards |
| `Rubber004` | rubber: dock bumpers, tyres, mud flaps |

## Robot: Unitree H1-2 with Inspire hands
- Source: unitreerobotics/unitree_ros at commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`,
  `robots/h1_2_description/h1_2.urdf` and its meshes, fetched from `https://raw.githubusercontent.com/unitreerobotics/unitree_ros/<commit>/`
  by `python -m robots.models.h1_2.install` (shared robot catalogue), with the repository's LICENSE.
- Licence: BSD 3-Clause, "Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")" (the
  installed LICENSE, read).
- Use: the seated operator (`forklift/operator.py`). Changes at load: collision pairs between the pelvis/torso and the
  thigh links are filtered (their convex hulls meet in the seated hip flexion while the meshes stay 9.5 mm apart), and
  between each palm and its thumb's proximal link (their hulls overlap in every thumb pose).

## Tool: CoACD 1.0.5 (approximate convex decomposition)
- Source: https://pypi.org/project/coacd/1.0.5/ (wheel pinned by URL and SHA-256 in `forklift/assets/install.py`);
  project page https://colin97.github.io/CoACD/. Licence: MIT.
- Use: build time only, to decompose the cab interior into collision hulls (cached in `forklift/.cache`); installed into
  a private folder, never into a shared interpreter.

## Specifications used as sources
Manufacturer sheets and standards read for parameters (Hyster H2.5A, Trelleborg tyres, Danfoss OSP steering units,
Hangcha and Crown operator manuals, OSHA 29 CFR 1910.178, GMA pallet dimensions, Wabash trailer data) are cited per
parameter in `forklift/params.py`; no text or images from them are redistributed.

# RoboWorld

RoboWorld is a benchmark of 25 simulations of economically valuable physical work, built with Gym-Anything for
Robotics. Each simulation lives in its own package under [`environments/`](environments), and the task splits are in
[`splits/`](splits).

## Splits

- [`splits/train.json`](splits/train.json): training tasks from the simulations.
- [`splits/test.json`](splits/test.json): the hardest task of each simulation, 25 tasks in total. These are the test
  tasks of the main results.

Each entry names an environment and a task selection, which `gar tasks resolve` turns into a runnable task:

```bash
gar tasks resolve welding --selector-json '{"task": "fillet-2f"}' --output welding.task.json
gar run --task-file welding.task.json --port 0 --policy-key
```

## Environments

| Environment | Job | Package | Robot setups (default first) | Test task | Report |
|---|---|---|---|---|---|
| `barcode-scanning` | Scanning barcodes with a handheld imager | `barcode_scanning` | dual-fr3, dual-panda, single-fr3, g1 | preset=stockroom, task=count-by-scanning | [PDF](environments/barcode_scanning/REPORT.pdf) |
| `classroom-arrangement` | Rearranging classroom furniture | `classroom_arrangement` | mobile-fr3-duo, mobile-fr3-single | task=group-work | [PDF](environments/classroom_arrangement/REPORT.pdf) |
| `concrete-finishing` | Finishing concrete, mortar and sealant | `concrete_finishing` | mobile-fr3-duo, mobile-fr3-single | task=pad-finish | [PDF](environments/concrete_finishing/REPORT.pdf) |
| `deposit-cans` | Putting cans into a film bag | `deposit_cans` | dual-fr3, dual-panda-wsg50 | variant=high_fill | [PDF](environments/deposit_cans/REPORT.pdf) |
| `drilling` | Drilling holes and setting anchors | `drilling` | dual-fr3, dual-panda, single-fr3, single-panda | preset=wall, task=ledger-anchor | [PDF](environments/drilling/REPORT.pdf) |
| `filleting` | Filleting fish and trimming meat | `filleting` | dual-fr3, dual-panda, single-fr3 | preset=trout, task=fillet-whole-fish | [PDF](environments/filleting/REPORT.pdf) |
| `football` | Kicking a football with a humanoid | `football` | g1 | default task | [PDF](environments/football/REPORT.pdf) |
| `forklift` | Driving a forklift as a seated humanoid | `forklift` | h1-2, drive-kit | task=load-bales | [PDF](environments/forklift/REPORT.pdf) |
| `gift-bags` | Opening gift bags, wrap and cartons | `gift_bags` | dual-fr3, dual-panda, single-fr3, single-panda | preset=counter, task=bag-purchase | [PDF](environments/gift_bags/REPORT.pdf) |
| `ingredient-mixing` | Mixing ingredients | `ingredient_mixing` | dual-fr3, single-fr3, dual-panda, single-panda | preset=hob, task=stir-sauce | [PDF](environments/ingredient_mixing/REPORT.pdf) |
| `label-application` | Applying adhesive labels | `label_application` | dual-fr3, dual-panda, single-fr3 | preset=specimen, task=tube-label | [PDF](environments/label_application/REPORT.pdf) |
| `laundry` | Folding laundry | `laundry` | dual-panda | garment=shirt | [PDF](environments/laundry/REPORT.pdf) |
| `lockkey` | Unlocking a padlock with a key | `lockkey` | panda | default task | [PDF](environments/lockkey/REPORT.pdf) |
| `painting` | Painting walls, crafts and spray work | `painting` | mobile-fr3-duo, mobile-fr3-single, g1 | preset=wall, task=behind-radiator | [PDF](environments/painting/REPORT.pdf) |
| `pencap` | Capping a pen | `pencap` | dual-panda | default task | [PDF](environments/pencap/REPORT.pdf) |
| `percussion` | Playing a xylophone and drum | `percussion` | dual-fr3, single-fr3, dual-panda, g1 | preset=therapy, task=melody | [PDF](environments/percussion/REPORT.pdf) |
| `sewing` | Sewing on an industrial machine | `sewing` | dual-fr3, dual-panda, single-fr3, g1 | task=curve | [PDF](environments/sewing/REPORT.pdf) |
| `soil-excavation` | Digging soil with hand tools | `soil_excavation` | mobile-fr3-duo, mobile-fr3-single | task=set-post | [PDF](environments/soil_excavation/REPORT.pdf) |
| `surface-rinsing` | Rinsing dishes and surfaces | `surface_rinsing` | mobile-fr3-duo, mobile-fr3-single | preset=pot-sink, task=scrub-rinse-pan | [PDF](environments/surface_rinsing/REPORT.pdf) |
| `syringe-handling` | Handling syringes and liquids | `syringe_handling` | dual-fr3, dual-panda, single-fr3, g1 | preset=injection, task=vial-to-pad | [PDF](environments/syringe_handling/REPORT.pdf) |
| `tighten-codex` | Tightening screws | `tighten_codex` | dual-fr3, single-fr3, dual-panda | task=mixed | [PDF](environments/tighten_codex/REPORT.pdf) |
| `titration` | Titrating a solution | `titrate` | panda | variation_id=0 | [PDF](environments/titrate/REPORT.pdf) |
| `welding` | Robotic MIG fillet welding | `welding` | ur10e-fr3, ur10e, dual-fr3, single-fr3, dual-panda, g1 | task=fillet-2f | [PDF](environments/welding/REPORT.pdf) |
| `whiteboard` | Writing and erasing on a whiteboard | `whiteboard` | g1 | variation_id=0 | [PDF](environments/whiteboard/REPORT.pdf) |
| `wine-pouring` | Pouring wine | `wine_pouring` | dual-fr3, single-fr3, dual-panda, single-panda | preset=service, task=split-bottle | [PDF](environments/wine_pouring/REPORT.pdf) |

Each environment folder also holds `REPORT.pdf`, the simulation's technical report: the job and its tasks, the scene, the physics models and their parameters with sources, validation experiments, demonstrations, runtime and limitations.

Every simulation can be started with any of its robot setups (`gar run ENV --robot SETUP`), and most support
variations of object poses, appearance and physics (`gar tasks inspect ENV` lists them).

## Support packages

`wiping`, `sweeping` and `control_cable` are simulations whose code other environments reuse: `wiping` provides the
render worker and the membrane and contact models of `deposit-cans`, `sweeping` is the base of `concrete-finishing`,
and `control_cable` provides camera and kinematics helpers. They are not part of the benchmark and have no tasks here.

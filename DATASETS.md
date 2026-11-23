# Datasets

Every image Ayas was trained or evaluated on is a **render of the Blender digital twin in this repository**. Procedural textures, no third-party photos. No public dataset was used for training.

There are no real-camera images yet; that is Phase 2.

The renders are not in git: they total about 90 GB of EXR renders and processed grids. They regenerate bit-for-bit from the saved plans with the same code and seeds.

## The twin data (rendered 15 Nov 2026 on a JarvisLabs L4, Blender 5.2.3 Cycles)

| Set | Seeds | Size | Use |
|---|---|---|---|
| Training scenes | `s000000`–`s000799` | 800 scenes, random lighting and device mode. Materials: cold-rolled 186, hot-rolled 166, aluminium 146, stainless brushed 128, stainless 2B 93, galvanised 81 | YOLOv8n training for both capture arms. Tiles: burst 5,176 train / 430 val; naive 7,531 train |
| Lighting test matrix | `s900100`–`s900113`, `s900124` | 15 plates, 3 of them defect-free. Hot-rolled 6, stainless brushed 3, cold-rolled 2, aluminium 2, stainless 2B 1, galvanised 1. Each plate × 14 rows (7 lighting conditions × open/hood) × 2 variants × 2 arms = 840 captures | The lighting experiment (`reports/lighting_matrix.md`). **Never trained, tuned or calibrated on** |
| JEPA paired plates | `s100000`–`s100047` | 48 plates × 14 rows, pixel-aligned across lighting | The JEPA experiment (`docs/JEPA.md`); not adopted |
| Pilot plates (excluded from test) | `s900000`–`s900068` | small | Tuned the capture quality gate, so they are not test plates |

The lighting conditions are C0 dark, C1 office (~350 lux), C2 warm bulb, C3 window daylight, C4 low side lamp + hand shadow, C5 lamp reflected in the plate, and C6 direct sun (~45,000 lux). Each is captured with the open device and with the hood. Definitions: `src/ayas/twin/lighting.py`, `configs/twin.yaml`.

Ground truth comes from the generator: defect class, mask, box, size in mm and depth. The defect classes are scratch, dent, pit, corrosion, crack and stain; their geometry is in `src/ayas/twin/sampler.py`.

## Regenerating

- **Plans** (exact scene parameters per seed) are produced by `ayas-twin plan` / `ayas-twin matrix` / `training/jepa/pairs.py`. Same code + same seed gives the same plan.
- **Render + post on a GPU box:** `tools/jl/render_all.sh` (see `tools/jl/README.md`). About 3.5 h of rendering on one L4 for all three sets (matrix 28 min, JEPA 1 h 22 min, training 1 h 43 min, measured 15 Nov), plus about 12 min of post on the CPUs.
- **A private backup of the exact renders** exists (tar bundles with an MD5 manifest). Ask the team if you need it instead of re-rendering.

## Real data (Phase 2)

- Real specimen captures (aluminium, mild steel, stainless coupons with seeded defects) are planned for Phase 2, split by plate.
- Public steel-defect datasets (NEU-DET, GC10-DET, Severstal) are surveyed in `docs/research/2026-11-14_steel_datasets.md`. They are **not** used here. Using one would need its licence checked first, and it would come in only through a download script, never committed.

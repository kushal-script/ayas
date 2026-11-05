# TriNetra digital twin: design contract

**Status:** build contract (5 Nov 2026).

Every module in `src/trinetra/twin/`, `twin/blender/`, `training/` and `hailo/` follows this file. If code and this file disagree, fix the code or update this file in the same change.

## 0. What the twin is for

**Goal.** Synthetic pretraining data for the surface-defect detector that runs on the Hailo-10H, plus structured defect records that regression-test the Llama advisor.

**The claim.** "Simulation-pretrained, real-calibrated":
- Every metric is measured on real photos of the physical coupons.
- The held-out test plate P5 is never rendered for training, and the twin is never tuned on P5.
- We never claim "trained in sim, works directly". The evidence is in `docs/research/2026-11-05_aero_sim2real_evidence.md`.

**Domain framing.** A handheld inspector's assistant for riveted aluminium aerostructure details at Tier-2/3 suppliers.
- Acceptance tables are illustrative and come only from public sources: FAA AC 43.13-1B and Piper SL 1095A.
- Dispositions follow Boeing Q219P logic: a supplier may only rework or scrap; use-as-is and repair go to the customer MRB.
- No airworthiness, certification, FPI or SRM claims. See `docs/research/2026-11-05_aero_qc_domain.md`.

```
configs/coupons.yaml ─┐
configs/twin.yaml ────┼─> sampler ──> plan.json ──> textures (numpy) ──> *.png maps
Fusion generator ─────┘   (random +                       │
 (LED geometry, rig.py)    replica scenes)                ▼
                                       Blender 5.2 LTS / Cycles (light groups L, R, UV, AMB)
                                                          │ render.exr (multilayer, linear)
                                                          ▼
                                    post: compose lights, camera noise/ISP, rectify to the
                                    plate grid (0.08 mm/px), visibility filter, YOLO tiles,
                                    sidecar JSON (mm truth), penetrant grids
                                                          │
             real captures + ground_truth.csv ─> realdata (same rectify/label/tile code)
                                                          ▼
               training/ (ablation: real-only, 25 % real, synth-only, synth->real, mixed)
               -> best.onnx -> hailo/ (DFC 5.1.0, Colab) -> .hef -> npu.py on the Pi
               sidecars -> twin/records.py -> aero rules + advisor regression
```

## 1. Frames and units (`src/trinetra/twin/frames.py`, tested)

- **Plate frame, mm.**
  - Origin at the plate's datum corner. u points right and v points down along the front face; w is height out of the face.
  - `spec.yaml` zones, `ground_truth.csv` and every `*_mm` value in a plan use this frame.
- **World frame (Blender), metres, right-handed.**
  - `X = u/1000`, `Y = -v/1000`, `Z = w/1000`.
  - The front face is `Z = 0` and the plate body is `-t <= Z <= 0`.
  - A front sheet (lap strip) of thickness `ts` occupies `0 <= Z <= ts`.
- **Camera (OpenCV).**
  - Axes: x right, y down, z forward; `X_cam = R @ X_world + t`, with t in metres.
  - Pixel centres sit on integer coordinates.
  - Use `frames.camera_pose`, `homography_plate_to_image`, `roi_K`, `blender_cam2world` and `blender_intrinsics`. Do not re-derive them.
- **Rectified grid.**
  - This is `config.Grid` with `mm_per_px = 0.08` and `border_mm = 5`.
  - `frames.homography_plate_to_grid(mpp, border)` maps plate mm to grid px.
  - A 100 × 100 plate gives a 1375 × 1375 grid, cut into 3 × 3 tiles of 640 at 0, 368 and 735 (`trinetra.tiles`).
- **Corner ordering.**
  - The Pi orders the 4 imaged plate corners TL, TR, BR, BL by image position (`frames.order_corners`). The grid is therefore image-aligned, not datum-aligned.
  - Post must do exactly the same. It maps the plate polygon through `H_img<-plate`, orders the corners by image position, and builds `H_grid<-img` from those 4 points. Labels go through `H_grid<-img ∘ H_img<-plate`.
  - The grid is `(W, H)` or `(H, W)`, whichever matches the imaged TL→TR edge.
- **Texture maps.**
  - Maps cover exactly the plate front face: `Nu = round(W / res)` columns and `Nv = round(H / res)` rows, with `res = render.texture_res_mm` (0.04).
  - Row r, column c has its centre at `u = (c + 0.5)·res`, `v = (r + 0.5)·res`. Row 0 is the top (v = 0).
  - Blender samples them with generated coordinates: `U = X·1000/W`, `V = 1 + Y·1000/H` (= 1 − v/H). Do not rely on UV maps; booleans destroy them.
  - **Normal map.** Object-space. From height h(u, v) in mm (positive out of the face), `n = normalize(-∂h/∂u, +∂h/∂v, 1)` in world (X, Y, Z). It is stored as 16-bit RGB `(n + 1)/2`, Non-Color, read by a Normal Map node in OBJECT space. The plate object has an identity transform.
  - **Front face only.** The shader applies the maps only where `Geometry.Normal.z > 0.9` (the front face, sheet tops excluded), and uses plain material elsewhere: hole walls, edges, back.

## 2. Classes

- **YOLO surface detector, ids:** 0 `rivet`, 1 `open_hole`, 2 `dent`, 3 `scratch`, 4 `crack`, 5 `corrosion`. These are `trinetra.tiles.CLASSES` and `configs/twin.yaml: classes`.
- **Kinds per class:**

  | Class | Kinds |
  |---|---|
  | `dent` | `punch`, `ball`, `smooth`, `nick` |
  | `scratch` | `scriber`, `gouge`, `abrasion` |
  | `crack` | `slit` (saw slit / EDM-notch analogue), `fatigue` (tight; mostly invisible without penetrant) |
  | `corrosion` | `pitting`, `galvanic` (ring round a fastener), `fretting` ("smoking rivet" streak) |

- **Penetrant scenes** (`mode: penetrant`): `linear_indication` and `rounded_indication`.
  - These are not YOLO classes. They go to `penetrant/` grids and sidecars for testing `penetrant.py`.
- **Distractors are never labelled:** `fingerprint`, `oil`, `water_spot`, `marker`, `dust`, `blend` (20:1 sanded dish, a rework mark), `burr` (at a hole), `glove`.
  - Burrs must come out as `unknown` → HOLD in the rules, which the anomaly map handles.
- **Geometry checks on fasteners are done by rules, not by YOLO:** edge distance, pitch, missing fastener and oval hole. YOLO only finds `rivet` and `open_hole`.
  - Ovality and centres are refined on the CPU (`aero.refine_hole`).

## 3. Scene plan JSON (`schema: trinetra.twin.scene/1`)

**Where it lives.** One file per scene, `<out>/plans/<scene_id>.json`. It is written by `sampler.py`, read by `textures.py`, `twin/blender/render_scene.py` and `post.py`. All randomness is decided here, so a plan plus its seed reproduces the scene. Keys not listed below must not be required by consumers.

```jsonc
{
  "schema": "trinetra.twin.scene/1",
  "scene_id": "s000042", "seed": 42, "split": "train",          // train | val | twin_test
  "mode": "surface",                                              // surface | penetrant
  "source": "random",                                             // random | replica:P2 | fixture
  "plate": {
    "size_mm": [100.0, 100.0], "thickness_mm": 3.0, "edge_bevel_mm": 0.3,
    "finish": "brushed",                                          // brushed | mill | clad
    "material": {"base_color": [0.91, 0.92, 0.92], "roughness": 0.25,
                 "anisotropy": 0.6, "aniso_rotation": 0.0,        // 0..1 (Blender units, x0.5 = 90 deg)
                 "grain_amp_um": 0.5, "grain_dir_deg": 0.0}
  },
  "sheets": [                                                     // front sheets (lap strip), optional
    {"id": "S1", "rect_mm": [0, 0, 100, 35], "thickness_mm": 1.0,
     "material": {"base_color": [0.9, 0.9, 0.9], "roughness": 0.3, "anisotropy": 0.5, "aniso_rotation": 0.25}}
  ],
  "holes": [                                                      // through holes (through sheet + plate)
    {"id": "H1", "u": 20.0, "v": 15.0, "d_mm": 4.1, "ovality": 1.0, "angle_deg": 0.0,
     "sheet": "S1", "filled": true}                               // filled: a rivet sits in it
  ],
  "rivets": [
    {"id": "R1", "hole": "H1", "head_d_mm": 8.0, "head_h_mm": 1.3, "mandrel_d_mm": 2.2,
     "tilt_deg": 0.0, "material": {"base_color": [0.86, 0.87, 0.88], "roughness": 0.35}}
  ],
  "defects": [
    {"id": "D1", "cls": "dent", "kind": "punch",
     "params": {"u": 62.0, "v": 70.0, "d_mm": 1.4, "depth_mm": 0.25, "rim_h_mm": 0.03},
     "footprint": {"polys_mm": [[[u, v], ...]],                    // outline(s) in plate mm
                   "measure": {"diameter_mm": 1.4, "depth_mm": 0.25}},
     "near": {"fastener_edge_mm": 9.2, "plate_edge_mm": 30.0},     // edge-to-edge distances
     "label": true}
  ],
  "distractors": [{"kind": "fingerprint", "params": {"u": 30, "v": 40, "r_mm": 7, "angle_deg": 20}}],
  "penetrant": null,                                              // or see section 3.2
  "camera": {
    "K": [[3058.06, 0, 2027.5], [0, 3058.06, 1519.5], [0, 0, 1]], "full_res": [4056, 3040],
    "R": [[...], [...], [...]], "t": [x, y, z],                   // world (m) -> OpenCV camera
    "roi": [x0, y0, w, h],                                        // rendered sensor window (full-res px)
    "fstop": 1.79, "focus_m": 0.2,
    "pose": {"target_mm": [50, 50], "distance_mm": 201.3, "tilt_deg": [3.1, -6.0], "roll_deg": 95.0}
  },
  "lights": [                                                     // world frame; group = light group
    {"name": "WL_L_0", "group": "L", "kind": "spot", "pos_m": [..], "dir": [..],
     "radius_m": 0.002, "spot_deg": 150, "blend": 0.6, "power_w": 0.35, "color": [1, 0.96, 0.9]}
  ],
  "ambient": {"world_color": [0.8, 0.8, 0.82], "world_strength": 0.25,
              "ceiling": [{"pos_m": [..], "size_m": [1.2, 0.05], "rot_deg": 30, "power_w": 40, "color": [..]}]},
  "extras": {"background": {"kind": "noise", "color": [0.2, 0.2, 0.22], "roughness": 0.8, "scale": 40},
             "gloves": [{"u": -5, "v": 60, "angle_deg": 180, "length_mm": 60, "radius_mm": 8}],
             "device_box": true},
  "render": {"samples": 96, "noise_threshold": 0.02, "max_bounces": 6, "glossy_bounces": 4,
             "texture_res_mm": 0.04, "clean_pass": false},
  "textures": {"normal": "normal.png", "albedo": "albedo.png", "roughness": "rough.png",
               "metallic": "metal.png", "uv_emit": null}
}
```

### 3.1 Defect params

All params are in plate mm. **`footprint.polys_mm` is computed by the sampler from these params.** It is the analytic outline that `textures.py` must stay inside: textures render the defect, they do not move it. Each kind's `measure` keys are listed in the table.

| Kind | Params | `measure` keys |
|---|---|---|
| `dent.punch` | u, v, d_mm (0.5–2.5), depth_mm (0.05–0.6), rim_h_mm (0.02–0.05) | diameter_mm, depth_mm |
| `dent.ball` | u, v, d_mm (3–12), depth_mm (0.1–0.8) | diameter_mm, depth_mm |
| `dent.smooth` | u, v, d_mm (5–40), depth_mm (0.1–1.5), aspect (1–2), angle_deg | diameter_mm (major), depth_mm |
| `dent.nick` | u, v (on an edge), edge (top/bottom/left/right), d_mm (1–3), depth_mm | diameter_mm |
| `scratch.scriber` | ctrl_mm (4 cubic-Bezier points), width_mm (0.05–0.5), depth_mm (0.005–0.08), burr (0–1) | length_mm, width_mm |
| `scratch.gouge` | same as scriber, with width 0.5–2 and depth 0.1–0.5 | length_mm, width_mm |
| `scratch.abrasion` | u, v, w_mm, h_mm (5–15), angle_deg, n_lines (20–200), depth_mm | area_mm2, length_mm (= max(w, h)), width_mm |
| `crack.slit` | u0, v0 (start: a hole edge or a plate edge), angle_deg, length_mm (1–10), width_mm (0.25–0.6) | length_mm, width_mm |
| `crack.fatigue` | pts_mm (jagged polyline), width_mm (0.01–0.08) | length_mm, width_mm |
| `corrosion.pitting` | u, v, r_mm (1.5–8), n_pits, pit_d_mm [min, max] (0.1–1), powder (0–1) | area_mm2, diameter_mm (equivalent) |
| `corrosion.galvanic` | hole (id), ring_w_mm (1–4), powder | area_mm2, diameter_mm (outer) |
| `corrosion.fretting` | rivet (id), angle_deg, length_mm (3–15), width_mm (1–4), darkness (0.3–0.9) | area_mm2, length_mm |

### 3.2 Penetrant scenes

```jsonc
"penetrant": {"developer": {"albedo": 0.88, "roughness": 0.9, "speckle": 0.05, "thickness_var": 0.1},
              "indications": [{"id": "I1", "cls": "linear_indication", "from_defect": "D3",
                               "params": {"pts_mm": [[..]], "bloom_mm": 0.6, "intensity": 0.8},
                               "footprint": {"polys_mm": [...], "measure": {"length_mm": .., "width_mm": ..}}},
                              {"id": "I2", "cls": "rounded_indication", "params": {"u": .., "v": .., "d_mm": 0.8}, ...}],
              "edge_bleed": 0.3}                                  // unlabelled bleed-out at plate edges
```

- Developer covers the plate, the sheets and the rivets: Blender gives them the developer material.
- Indication colour is the red of the Wesaf kit, `[0.55, 0.05, 0.06]` linear, until it is measured.
- **The dye does not fluoresce:** under UV it stays dark (see research).

### 3.3 Lights

- `rig.py` builds every LED from the Fusion generator: `layout`, `pod_frame`, `uv_frame`, `white_columns` and `uv_led_points`, imported with stub `adsk` modules.
  - **White:** each pod has 3 vertical modules of 3 LEDs, spaced `white_l / 3` along the strip, emitting along the pod-face normal.
  - **UV:** each pod has 9 LEDs, emitting along the UV sub-frame normal.
- **Model to camera frame.**
  - Positions are relative to the lens at model `(0, disp_cy, z_lens)`.
  - Model +Y is up, +Z is forward, and model X is seen from the front.
  - With `rig.model_x_is_image_left: true`, the camera frame is `x_cam = -X`, `y_cam = -(Y - disp_cy)` and `z_cam = Z - z_lens`.
  - **Groups:** the pod on the image left is group `L`; the other is `R`; all UV LEDs are group `UV`.
- **Camera to world.** `pos_world = C + R.T @ p_cam / 1000`, `dir_world = R.T @ d_cam`.
- **Ambient (group AMB):**
  - the world (uniform colour × strength);
  - 0–2 ceiling tube lights 1.5–2.5 m above the plate.
- **Device box.** A black box in the camera frame (`rig.device_box_mm`). It is invisible to camera rays but casts shadows and shows in glossy reflections: the plate mirrors the device underside, as on the real device.

## 4. Per-scene work directory and files

`<out>/work/<scene_id>/` holds the following. **It is deleted after `post` succeeds** (disk is about 10 GB on the Mac).

| File | Writer | Format |
|---|---|---|
| `plan.json` | sampler | as above |
| `normal.png` | textures | 16-bit RGB, object-space normals (section 1) |
| `albedo.png` | textures | 8-bit sRGB RGB |
| `rough.png` | textures | 8-bit grey, linear roughness 0–1 |
| `metal.png` | textures | 8-bit grey, metallic 0–1 (corrosion powder, developer, dye, ink → 0) |
| `uvemit.png` | textures (optional) | 8-bit grey; only materials that really fluoresce |
| `*_clean.png` | textures (if `render.clean_pass`) | the same maps without labelled defects (distractors and grain kept) |
| `render.exr` | Blender | multilayer EXR, half float, linear; passes `Combined_L`, `Combined_R`, `Combined_UV`, `Combined_AMB` (Cycles light groups), plus `Combined` |
| `render_clean.exr` | Blender (if clean pass) | same passes, rendered with the `*_clean.png` maps |
| `render_meta.json` | Blender | Blender version, device, render seconds, EXR channel names, ROI size |

### Outputs (kept)

| Path | Contents |
|---|---|
| `<out>/yolo/images/{train,val}/<scene>_v<k>_<tile>.jpg` | tiles (640², quality `post.jpeg_quality`) |
| `<out>/yolo/labels/{train,val}/<same>.txt` | YOLO lines (`trinetra.tiles.yolo_line`); empty file for background tiles |
| `<out>/yolo/data.yaml` | Ultralytics dataset: `path`, `train`, `val`, `names` |
| `<out>/sidecar/<scene>.json` | `schema: trinetra.twin.sidecar/1`: scene_id, grid size, `H_grid_plate` (3 × 3, the full label mapping), fasteners (id, cls, centre/diameter/ovality in mm and grid px), defects (id, cls, kind, measure, near, polys in grid px, bbox_grid, visible, contrast, tiles), variants (gains, exposure, noise) |
| `<out>/penetrant/<scene>_v<k>.png` + sidecar | full rectified grids for `penetrant.py` tests |
| `<out>/debug/<scene>_v<k>.jpg` | the first 50 scenes: grid with label overlays |

## 5. Post-processing rules (`post.py`)

1. **Read EXR.** Use the `OpenEXR` Python package (3.5 is installed).
   - Verified on Blender 5.2.3: each pass is its own part, e.g. `ViewLayer.Combined_L`. The part holds one channel entry under the same name, whose `.pixels` is an (H, W, 3) float16 RGB array (`Combined` is RGBA).
   - Iterate over `OpenEXR.File(p).parts` and their `.channels`. Match names with `(^|\.)Combined_(L|R|UV|AMB)(\.[RGB])?$`, and also accept split R/G/B channels.
   - A light group with no lights in it may be missing. Treat it as zeros.
2. **Compose (linear).** `I = gL·L + gR·R + gA·AMB`, with gains drawn per variant from `post.gains`.
   - `fusion: both_rgb` gives that RGB image.
   - `fusion: lr3` gives `[luma(gL·L + gA·AMB), luma(gR·R + gA·AMB), relief]`, where `relief = 0.5 + 0.5·(Lv − Rv)/(Lv + Rv + 1e-4)`.
3. **Camera model, in the ROI image in this order:**
   1. exposure so the plate-region median luma hits U(`exposure_p50`);
   2. white-balance jitter;
   3. Poisson–Gaussian noise in electrons;
   4. clip to [0, 1];
   5. sRGB OETF;
   6. unsharp mask;
   7. optional linear motion blur;
   8. quantise to 8 bits.
4. **Rectify** with `cv2.warpPerspective(INTER_LINEAR)`, using `H_grid<-roi = H_grid<-img · T(x0, y0)` and the corner ordering from section 1.
5. **Labels.**
   - Rasterise each labelled footprint (polys → grid px) into its own mask.
   - Fasteners: a `rivet` is the head-diameter circle; an `open_hole` is the hole ellipse.
   - **Visibility:**
     - With a clean pass: the mean |I − I_clean| inside the 1-px-dilated mask must be ≥ `diff_threshold`.
     - Without one: p90 of |I − medianBlur(I, k)| inside the mask, minus p90 in a ring of 3–8 px, must be ≥ `local_threshold`. Here k is odd and ≥ 3 × the width in px, capped at 31.
     - Classes in `always_visible` skip the test.
     - Invisible defects get no label: `visible: false` in the sidecar.
   - **Per tile:**
     - Take the bbox of the mask pixels inside the tile, padded to at least `min_box_px` and clipped to the tile.
     - Keep it if in-tile pixels ≥ `keep_frac` × total, or the in-tile bbox diagonal ≥ `keep_min_px`.
6. **Empty tiles** are kept with probability `empty_tile_keep`.

## 6. Real data (`realdata.py`)

- **Inputs.** Real captures: RGB images, with plate corners from `corners.json` or auto-detected by `calibration.find_plate_corners`. Plus `data/real/ground_truth.csv`.
- **Same path as the twin.** They go through the same rectify → rasterise → tile code, so the two are consistent.
- **CSV columns:** `defect_id, plate, face, cls, kind, u_mm, v_mm, length_mm, width_mm, diameter_mm, angle_deg, poly_mm, eval_class, notes`.
  - `poly_mm` is an optional JSON polygon. If it is absent, the footprint is a circle (diameter) or an oriented rectangle (length × width at `angle_deg` centred on u, v).
- **Datum corner.** The real corner ordering is by image position, so the datum corner on each physical plate is marked with a punched "A". The capture tool records which imaged corner it is, so plate mm can be recovered.

## 7. Deployment contract

- **Grid.** The Pi rectifies the frame from the 4 plate corners to the grid (0.08 mm/px, 5 mm border) and tiles it with `trinetra.tiles`.
- **Detection.** `npu.py` runs YOLOv8n per tile:
  - `HailoDetector` (HEF with on-chip NMS, uint8 RGB input) or `OnnxDetector` (CPU, float RGB / 255).
  - `merge_tiles` joins the results; grid px → mm via `config.Grid.px_to_mm`.
- **Rules.** `aero.py` adds fastener geometry and Piper-style dent classes; `rules.py` with `configs/spec_aero.yaml` sets severity and disposition.
- **Advice.** The advisor (Llama 3.2 1B, ExecuTorch) explains the result; it never changes it.

## 7.1 Verified Blender 5.2.3 LTS facts (smoke test, 5 Nov)

- **Binary:** `/Applications/Blender.app/Contents/MacOS/Blender` (5.2.3 LTS). Run it with `-b --factory-startup -P script.py -- args`.
- **Multilayer EXR.** Set `image_settings.media_type = 'MULTI_LAYER_IMAGE'` first, then `file_format = 'OPEN_EXR_MULTILAYER'`, `color_depth = '16'`, `exr_codec = 'ZIP'`.
  - `render.filepath = '/abs/render.exr'` with `bpy.ops.render.render(write_still=True)` writes exactly that file.
- **Light groups:**
  - `view_layer.lightgroups.add(name=g)`;
  - `obj.lightgroup = g` for lights and emissive meshes;
  - `world.lightgroup = 'AMB'`.
  - The passes come out as `ViewLayer.Combined_<g>`.
- **Denoising:** with denoising on, a `Noisy Image` pass also appears. Set `scene.cycles.use_denoising = False`, because light-group passes are never denoised.
- **Metal:**
  - Set `prefs.compute_device_type = 'METAL'`, call `prefs.get_devices()`, enable the "Apple M3 (GPU - 10 cores)" device, and set `scene.cycles.device = 'GPU'`.
  - The first Metal render compiles the kernels: about 200 s once, then cached. Budget for it in tests and timing.
- **Principled BSDF inputs (5.2):** Base Color, Metallic, Roughness, IOR, Alpha, Normal, Specular IOR Level, Specular Tint, Anisotropic, Anisotropic Rotation, Tangent, Coat Weight, Sheen Weight, Emission Color, Emission Strength.
- **Python:** Blender bundles its own Python with numpy. Do not import `trinetra` inside Blender. The driver precomputes the camera into `blender_cam.json`.
- **DOF.** For physically right depth of field, give Blender the physical sensor: `sensor_width_mm = max(w, h) · pixel_um / 1000`. Then `lens_mm` is about 4.74 and the aperture is `lens / fstop`.
  - Use `frames.blender_intrinsics(K_roi, w, h, sensor_width_mm=...)`.

## 8. Ownership (build phase)

| Area | Files |
|---|---|
| Coupons and templates | `configs/coupons.yaml`, `twin/coupons.py`, `twin/templates.py`, `docs/twin/coupon_fabrication.md` |
| Rig, sampler and footprints | `twin/rig.py`, `twin/sampler.py` |
| Textures | `twin/textures.py` |
| Blender | `twin/blender/render_scene.py`, `twin/render.py` |
| Post and real data | `twin/post.py`, `twin/realdata.py` |
| Detector, training and Hailo | `npu.py`, `training/*`, `hailo/*` |
| Aero QC | `configs/spec_aero.yaml`, `aero.py`, `ncr.py`, `rules.py`/`report.py`/`llm.py` changes, `twin/records.py` |
| Integration | `cli.py`, `twin/pipeline.py`, docs |

"""Real coupon captures -> the same rectified grids, labels and tiles as the twin (DESIGN 6,
DESIGN_STEEL E).

Two kinds of capture:
- arm "single": one RGB photo (as before);
- arm "burst": the 5-frame LED burst B1, L, A, R, B2 (linear, black-level-free frames in
  [0, 1] full scale: a .npz such as out/bursts/*.npz from the twin or the Pi, or one PNG
  per frame, 16-bit = /65535). It runs through lightproc.process_burst, exactly the Pi's
  path, with the capture's own meta (an npz's meta_json, or the corners.json entry's "meta",
  e.g. {"demosaiced": true} for Pi raw_to_rgb frames saved as PNGs); a RETAKE gate is
  recorded in the sidecar and writes no training tiles.

The 4 plate corners are given in PLATE order (datum "A" corner (0, 0), then (W, 0),
(W, H), (0, H)), so H_img<-plate comes from the 4 correspondences. Everything after
that is post.py's code: corner ordering by image position, rectify, footprint masks,
tile labels and the sidecar. Real images get no synthetic noise.

Footprints are planar (w = 0): there is no camera pose for a real photo, so features on
a lap strip are labelled where the plate plane maps them (a few px of parallax).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .. import tiles as T
from . import post as P

CSV_COLUMNS = ("defect_id", "plate", "face", "cls", "kind", "u_mm", "v_mm", "length_mm", "width_mm",
               "diameter_mm", "angle_deg", "poly_mm", "eval_class", "notes")
REQUIRED = ("defect_id", "plate", "cls")
NUMERIC = ("u_mm", "v_mm", "length_mm", "width_mm", "diameter_mm", "angle_deg")


def load_ground_truth(csv_path: str | Path) -> list[dict[str, Any]]:
    """Rows of ground_truth.csv with numbers parsed (empty -> None) and poly_mm as JSON."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in reader.fieldnames or []]
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        for raw in reader:
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not any(row.values()) or row["defect_id"].startswith("#"):
                continue
            for c in CSV_COLUMNS:
                row.setdefault(c, "")
            for k in NUMERIC:
                row[k] = float(row[k]) if row[k] else None
            row["poly_mm"] = json.loads(row["poly_mm"]) if row["poly_mm"] else None
            rows.append(row)
    return rows


def footprint_from_row(row: Mapping[str, Any]) -> list[np.ndarray]:
    """Polygons (plate mm) of a row: poly_mm, else a circle (diameter_mm), else an oriented
    length x width rectangle at angle_deg (from +u towards +v) centred on (u, v)."""
    poly = row.get("poly_mm")
    if poly:
        single = isinstance(poly[0][0], (int, float))
        return [np.asarray(p, float).reshape(-1, 2) for p in ([poly] if single else poly)]
    u, v = row.get("u_mm"), row.get("v_mm")
    if u is None or v is None:
        raise ValueError(f"{row.get('defect_id')}: needs poly_mm or u_mm, v_mm")
    if row.get("diameter_mm"):
        return [P.circle_poly(u, v, row["diameter_mm"])]
    if row.get("length_mm") and row.get("width_mm"):
        return [P.rect_poly(u, v, row["length_mm"], row["width_mm"], row.get("angle_deg") or 0.0)]
    raise ValueError(f"{row.get('defect_id')}: needs poly_mm, diameter_mm or length_mm + width_mm")


def rows_for(rows: Sequence[Mapping[str, Any]], coupon: str) -> list[Mapping[str, Any]]:
    """Rows of one coupon face: plate == coupon, or plate + face == coupon (e.g. P3 + A)."""
    return [r for r in rows if r["plate"] == coupon or f"{r['plate']}{r.get('face') or ''}" == coupon]


def row_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ground-truth rows -> post.rasterise_items footprints (planar). Holes and slots (post.FEATURES)
    are part features: rasterised so visibility rings skip them, never labelled. A row with
    label False (a twin defect its own ground truth found invisible, rows_from_labels) is kept
    as an unlabelled defect."""
    items = []
    for r in rows:
        polys = footprint_from_row(r)
        measure = {k: r[k] for k in ("length_mm", "width_mm", "diameter_mm") if r.get(k) is not None}
        extra = {"eval_class": r.get("eval_class") or None, "notes": r.get("notes") or None}
        it = {"id": r["defect_id"], "cls": r["cls"], "kind": r.get("kind") or r["cls"],
              "group": "feature" if r["cls"] in P.FEATURES else "defect",
              "label": r["cls"] in T.CLASSES and r["cls"] not in P.FEATURES and r.get("label", True) is not False,
              "polys_mm": polys, "w_mm": None, "measure": measure, "near": None, "extra": extra}
        if it["group"] == "defect":
            extra.update(u_mm=r.get("u_mm"), v_mm=r.get("v_mm"), angle_deg=r.get("angle_deg"))
        else:
            c = np.concatenate(polys).mean(axis=0) if r.get("u_mm") is None else np.array([r["u_mm"], r["v_mm"]])
            d = r.get("diameter_mm") or float(np.ptp(np.concatenate(polys)[:, 0]))
            it["feature"] = {"hole": None, "u_mm": float(c[0]), "v_mm": float(c[1]), "d_mm": float(d),
                              "hole_d_mm": None, "ovality": 1.0, "angle_deg": 0.0, "w_mm": None}
        items.append(it)
    return items


def read_rgb(image) -> np.ndarray:
    """A path (read with OpenCV, BGR -> RGB) or an RGB array -> uint8 RGB."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    img = img[..., :3]
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = P.to_u8(img)
    return np.ascontiguousarray(img)


def process_capture(image, corners_plate_order_px, plate_size_mm: Sequence[float],
                    gt_rows: Sequence[Mapping[str, Any]], out_dir: str | Path, split: str,
                    cfg: Mapping[str, Any], name: str, coupon: str | None = None,
                    filter_visibility: bool = False, empty_tile_keep: float = 1.0,
                    debug: bool = False, image_path: str | Path | None = None, arm: str = "single",
                    capture: Mapping[str, Any] | None = None, write_tiles: bool = True) -> dict[str, Any]:
    """One real photo (or a processed burst's image) -> grid PNG (out/grids/<split>/<name>.png),
    YOLO tiles and labels (unless write_tiles is false), and out/sidecar/<name>.json. gt_rows are
    this coupon face's rows. Ground truth is labelled whatever its contrast unless
    filter_visibility; contrast is always recorded. capture: the arm's gate and metrics."""
    t_start = time.perf_counter()
    out = Path(out_dir)
    img = read_rgb(image)
    h, w = img.shape[:2]
    W, H = float(plate_size_mm[0]), float(plate_size_mm[1])
    corners = np.asarray(corners_plate_order_px, np.float32).reshape(4, 2)
    H_img_plate = cv2.getPerspectiveTransform(P.plate_corners(W, H).astype(np.float32), corners).astype(float)
    grid_cfg = cfg.get("grid") or {}
    geom = P.grid_geometry(H_img_plate, grid_cfg, plate_size_mm=(W, H), roi=[0, 0, w, h])
    gw, gh = geom["size"]
    tiles = T.tile_grid(gw, gh, int(grid_cfg.get("tile", T.TILE)), int(grid_cfg.get("min_overlap", T.MIN_OVERLAP)))
    grid = P.rectify(img, geom)

    objects = P.rasterise_items(row_items(gt_rows), geom, geom["shape"])
    exclude = P.ring_exclude(objects, geom)
    objs = []
    for o in objects:
        vis, con = (None, None)
        if o["label"]:
            vis, con = P.visibility(grid, None, o, cfg, exclude)
            if not filter_visibility and o["n_px"]:
                vis = True
        objs.append({**o, "visible": vis, "contrast": con})

    boxes = P.label_boxes(objs, tiles, cfg.get("label") or {}) if write_tiles else []
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    keep = rng.random(len(tiles)) < empty_tile_keep
    quality = int((cfg.get("post") or {}).get("jpeg_quality", 95))
    written = (P.write_tiles(grid, tiles, P.lines_from_boxes(objs, tiles, boxes), out, split, name, quality, keep)
               if write_tiles else [])
    P.write_rgb(out / "grids" / split / f"{name}.png", grid)
    if debug:
        P.write_debug(out / "debug" / f"{name}.jpg", grid, objs, boxes, tiles)

    tiles_of: dict[str, list[str]] = {}
    for i, t, _ in boxes:
        tiles_of.setdefault(objs[i]["id"], []).append(t.name)
    cap = {"arm": arm, "gate": None, "gate_message": None, "metrics": None, **(capture or {})}
    sidecar = {
        "schema": P.SCHEMA, "scene_id": name, "split": split, "mode": "surface", "source": "real",
        "coupon": coupon, "image": str(image) if isinstance(image, (str, Path)) else _opt_str(image_path),
        "image_size": [w, h], "corners_plate_order_px": corners, "arm": arm, "gate": cap["gate"],
        "gate_message": cap["gate_message"], "capture": cap,
        **P.sidecar_base(geom, grid_cfg, tiles),
        "grid_png": f"grids/{split}/{name}.png",
        "features": [P.object_record(o, geom, tiles_of) for o in objs if o["group"] == "feature"],
        "defects": [P.object_record(o, geom, tiles_of) for o in objs if o["group"] == "defect"],
        "tiles_written": written, "n_labels": len(boxes),
        "post_s": round(time.perf_counter() - t_start, 2),
    }
    P.write_json(out / "sidecar" / f"{name}.json", sidecar)
    return P.json_safe(sidecar)


def _opt_str(x) -> str | None:
    return None if x is None else str(x)


def parse_split_map(text: str) -> dict[str, str]:
    """'P1A:train,P2A:train,P5A:test' -> {coupon: split}."""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        coupon, _, split = part.partition(":")
        if not split:
            raise ValueError(f"bad split-map entry {part!r} (want COUPON:split)")
        out[coupon.strip()] = split.strip()
    return out


def corners_for(entry: Mapping[str, Any], img: np.ndarray) -> np.ndarray:
    """Plate-order corners from a corners.json entry: corners_px, or auto-detected with
    calibration.find_plate_corners plus datum_image_corner (TL/TR/BR/BL or 0-3)."""
    if entry.get("corners_px") is not None:
        return np.asarray(entry["corners_px"], float).reshape(4, 2)
    datum = entry.get("datum_image_corner")
    if datum is None:
        raise ValueError("corners.json entry needs corners_px or datum_image_corner")
    from ..calibration import find_plate_corners

    k = P.CORNER_NAMES.index(datum) if isinstance(datum, str) else int(datum)
    image_order = find_plate_corners(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return np.roll(np.asarray(image_order, float), -k, axis=0)   # both orders run clockwise in the image


# ---- burst captures (DESIGN_STEEL E) ---------------------------------------------------

def _read_frame(path: str | Path) -> np.ndarray:
    """One burst frame from an image file -> float32 RGB in [0, 1] full scale (16-bit / 65535, 8-bit / 255)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read burst frame {path}")
    scale = 65535.0 if raw.dtype == np.uint16 else 255.0 if raw.dtype == np.uint8 else 1.0
    img = raw.astype(np.float32) / np.float32(scale)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    return np.ascontiguousarray(cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB))


def _npz_json(z, key: str):
    """A JSON entry of an npz: uint8 bytes (lightproc.save_burst's meta_json) or a 0-d string."""
    if key not in z.files:
        return None
    a = z[key]
    text = a.tobytes().decode() if a.dtype == np.uint8 else str(a[()])
    return json.loads(text) if text else None


def load_burst(src, base_dir: str | Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A burst capture -> (frames {B1, L, A, R, B2: float32 (H, W, 3) in [0, 1]}, extras).

    src: an .npz path in lightproc.save_burst's layout (frame_<name> arrays, meta_json; as
    post.save_burst and the Pi write them; plain <name> arrays are read too), or a mapping
    {frame name: image path} (relative paths from base_dir). extras holds what an npz
    carries: corners_plate_order_px (or rebuilt from corners_px + corner_plate_idx),
    plate_size_mm, labels (twin ground truth) and meta (the Burst meta)."""
    extras: dict[str, Any] = {"corners_plate_order_px": None, "plate_size_mm": None, "labels": None, "meta": {}}
    if isinstance(src, Mapping):
        base = Path(base_dir or ".")
        frames = {n: _read_frame(base / src[n]) for n in P.BURST_FRAMES if n in src}
    else:
        with np.load(str(src), allow_pickle=False) as z:
            frames = {n: z[k].astype(np.float32) for n in P.BURST_FRAMES for k in (f"frame_{n}", n) if k in z.files}
            if "corners_plate_order_px" in z.files:
                extras["corners_plate_order_px"] = z["corners_plate_order_px"].astype(float)
            elif "corners_px" in z.files and "corner_plate_idx" in z.files:
                extras["corners_plate_order_px"] = plate_order(z["corners_px"], z["corner_plate_idx"])
            if "plate_size_mm" in z.files:
                extras["plate_size_mm"] = [float(v) for v in z["plate_size_mm"]]
            extras["labels"] = _npz_json(z, "labels_json")
            extras["meta"] = _npz_json(z, "meta_json") or {}
    missing = [n for n in P.BURST_FRAMES if n not in frames]
    if missing:
        raise ValueError(f"burst {src}: missing frames {missing} (need {list(P.BURST_FRAMES)})")
    return frames, extras


def plate_order(corners_image_order, corner_plate_idx) -> np.ndarray:
    """Image-ordered corners (TL, TR, BR, BL) + which plate corner each is -> plate order."""
    c = np.asarray(corners_image_order, float).reshape(4, 2)
    out = np.zeros((4, 2))
    for i, j in enumerate(np.asarray(corner_plate_idx, int).reshape(4)):
        out[j] = c[i]
    return out


def display_image(frames: Mapping[str, np.ndarray], gamma: float = 2.2) -> np.ndarray:
    """uint8 RGB view of a burst (B1, p99.5-normalised, gamma) for corner detection and debugging."""
    b1 = np.clip(frames["B1"], 0.0, None)
    p = float(np.percentile(P.luma_lin(b1), 99.5)) or 1.0
    return P.to_u8(np.power(np.clip(b1 / p, 0.0, 1.0), 1.0 / gamma))


def rows_from_labels(labels: Sequence[Mapping[str, Any]], plate: str) -> list[dict[str, Any]]:
    """Twin ground truth (an npz's labels_json) -> ground_truth.csv-like rows (labelled defects only).

    A defect the twin's reference image found invisible (visible false: e.g. a tight fatigue
    crack) keeps its row with label False, so a replay never trains on a defect the image
    cannot show (process_capture labels every other row whatever its contrast)."""
    rows = []
    for d in labels:
        if d.get("label") is False or d.get("cls") not in T.CLASSES:
            continue
        c = d.get("centre_mm") or [None, None]
        row = {k: None for k in NUMERIC}
        seen = d.get("visible") is not False
        row.update(defect_id=str(d["id"]), plate=plate, face="", cls=d["cls"], kind=d.get("kind") or d["cls"],
                   u_mm=c[0], v_mm=c[1], poly_mm=d.get("polys_mm"), eval_class="",
                   notes="twin" if seen else "twin (invisible)", label=seen)
        rows.append(row)
    return rows


def process_burst_capture(frames: Mapping[str, np.ndarray], corners_plate_order_px, plate_size_mm: Sequence[float],
                          gt_rows: Sequence[Mapping[str, Any]], out_dir: str | Path, split: str,
                          cfg: Mapping[str, Any], name: str, coupon: str | None = None,
                          filter_visibility: bool = False, empty_tile_keep: float = 1.0, debug: bool = False,
                          image_path: str | Path | None = None, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One real burst -> lightproc.process_burst (roi = the plate polygon) -> process_capture on
    Processed.image_u8, as arm "burst". The gate verdict, message and metrics go in the
    sidecar; a RETAKE writes the grid and sidecar but no tiles."""
    LP = P.capture_modules()[0]
    cap_cfg = P.capture_config(cfg)
    corners = np.asarray(corners_plate_order_px, float).reshape(4, 2)
    shape = next(iter(frames.values())).shape
    roi = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(roi, [np.round(corners * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, P.SHIFT)
    burst = LP.Burst(frames={n: np.asarray(frames[n], np.float32) for n in P.BURST_FRAMES},
                     meta={"sequence": list(P.BURST_FRAMES), "source": "real", **(meta or {})})
    proc = LP.process_burst(burst, roi.astype(bool), cap_cfg)
    g = proc.gate
    capture = {"gate": g.verdict, "gate_reasons": list(g.reasons or []), "gate_message": g.message or "",
               "metrics": P.json_safe(proc.metrics)}
    return process_capture(np.asarray(proc.image_u8), corners, plate_size_mm, gt_rows, out_dir, split, cfg, name,
                           coupon=coupon, filter_visibility=filter_visibility, empty_tile_keep=empty_tile_keep,
                           debug=debug, image_path=image_path, arm="burst", capture=capture,
                           write_tiles=g.verdict != P.RETAKE)


def burst_meta(file_meta: Mapping[str, Any] | None, entry: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Burst meta for process_burst_capture: the capture's own meta (an npz's meta_json, e.g.
    the Pi's demosaiced=True, which sets lightproc's noise model for residual_sigma, and E_led),
    then the corners.json entry's "meta" and "material", then the file name."""
    meta = {**(file_meta or {}), **(entry.get("meta") or {}), "file": name}
    if entry.get("material"):
        meta["material"] = entry["material"]
    return meta


def _entry_kind(name: str, entry: Mapping[str, Any]) -> str:
    if entry.get("burst") is not None:
        return "burst_png"
    return "burst_npz" if name.lower().endswith(".npz") else "single"


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trinetra-twin real-tiles",
                                description="Real coupon photos or LED bursts + ground_truth.csv -> grids, YOLO tiles and sidecars.")
    p.add_argument("--captures", required=True, help="folder with the photos / bursts named in corners.json")
    p.add_argument("--corners", required=True,
                   help='{"<image>|<burst>.npz": {"coupon": "P3A", "corners_px": [[x, y] x4 in plate order]}, '
                        '"<name>": {"coupon": .., "burst": {"B1": "x_B1.png", "L": .., "A": .., "R": .., "B2": ..}, ..}}')
    p.add_argument("--gt", help="ground_truth.csv (DESIGN 6 columns); optional only for twin .npz bursts (their labels)")
    p.add_argument("--out", default="data/real")
    p.add_argument("--split-map", required=True, help="e.g. 'P1A:train,P2A:train,P3A:train,P4A:val,P5A:test'")
    p.add_argument("--config", default="configs/twin.yaml")
    p.add_argument("--plate-size", default="100x100", help="W x H mm when corners.json has no plate_size_mm")
    p.add_argument("--filter-visibility", action="store_true", help="drop low-contrast ground truth like the twin")
    p.add_argument("--debug", action="store_true", help="write label overlays to out/debug/")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    rows = load_ground_truth(a.gt) if a.gt else None
    corners = json.loads(Path(a.corners).read_text())
    split_map = parse_split_map(a.split_map)
    default_size = [float(s) for s in a.plate_size.lower().split("x")]
    done = 0
    for image_name, entry in corners.items():
        coupon = entry["coupon"]
        if coupon not in split_map:
            print(f"skip {image_name}: coupon {coupon} not in --split-map")
            continue
        kind, path = _entry_kind(image_name, entry), Path(a.captures) / image_name
        common = dict(coupon=coupon, filter_visibility=a.filter_visibility, debug=a.debug, image_path=path)
        if kind == "single":
            img = read_rgb(path)
            if rows is None:
                raise SystemExit(f"{image_name}: --gt is required for photos")
            sc = process_capture(img, corners_for(entry, img), entry.get("plate_size_mm") or default_size,
                                 rows_for(rows, coupon), a.out, split_map[coupon], cfg, Path(image_name).stem, **common)
        else:
            frames, extras = load_burst(entry["burst"] if kind == "burst_png" else path, a.captures)
            if entry.get("corners_px") is None and extras["corners_plate_order_px"] is not None:
                plate_corners = extras["corners_plate_order_px"]
            else:
                plate_corners = corners_for(entry, display_image(frames))
            if rows is not None:
                gt = rows_for(rows, coupon)
            elif extras["labels"] is not None:
                gt = rows_from_labels(extras["labels"], coupon)
            else:
                raise SystemExit(f"{image_name}: no --gt and the burst carries no labels")
            size = entry.get("plate_size_mm") or extras["plate_size_mm"] or default_size
            meta = burst_meta(extras["meta"], entry, image_name)
            sc = process_burst_capture(frames, plate_corners, size, gt, a.out, split_map[coupon], cfg,
                                       Path(image_name).stem, meta=meta, **common)
        gate = f", gate {sc['gate']}" if sc.get("gate") else ""
        print(f"{image_name}: {coupon} ({sc['arm']}) -> {split_map[coupon]}, {sc['n_labels']} labels in "
              f"{len(sc['tiles_written'])} tiles{gate}")
        done += 1
    print(f"{done} captures; wrote {P.write_data_yaml(a.out, cfg.get('classes', T.CLASSES))}")


if __name__ == "__main__":
    main()

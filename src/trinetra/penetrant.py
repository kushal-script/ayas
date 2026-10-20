"""Visible red-dye penetrant inspection (cracks and pores), on the same rig.

Process (off the rig, see docs/TriNetra_plan.md): clean -> red penetrant, dwell ->
wipe off excess -> white developer -> wait a fixed development time -> place the plate
in the jig. Here, the top camera under its own white light (minus the dark frame) sees
red indications on the white developer. Each indication is measured in mm and graded
the way industry grades penetrant indications:
- linear if length > 3 x width (crack-like), otherwise rounded (pore-like);
- relevant only if its major dimension is above the relevance threshold (1/16 in);
- ISO 23277 profiles: neighbours closer than the smaller one's major dimension count as one;
- ASME profile: 4 or more relevant rounded indications in a line, <= 1/16 in apart, reject;
- accept/reject limits come from a selectable profile in spec.yaml (ASME VIII App. 8
  style, or ISO 23277 acceptance levels 1-3). No code is written for plain plate, so
  the limits are placeholders to replace with your own specification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .align import warp_to_grid
from .calibration import Calibration
from .capture import channel, lit
from .config import CHANNELS, Config
from .rules import VERDICTS, load_spec, zone_for

DEFAULTS = {
    "redness_high": 0.45,  # seed: clearly red
    "redness_low": 0.22,  # extent: grows the seed into the faint bleed-out edge
    "min_area_mm2": 0.08,
    "smooth_px": 1.0,
}


@dataclass
class Indication:
    id: str
    mask: np.ndarray  # bool, bbox-sized
    bbox: tuple[int, int, int, int]
    x_mm: float
    y_mm: float
    length_mm: float
    width_mm: float
    area_mm2: float
    orientation_deg: float
    redness: float
    peak: float

    @property
    def kind(self) -> str:
        return "linear" if self.length_mm > 3.0 * self.width_mm else "rounded"

    @property
    def major_mm(self) -> float:
        return self.length_mm


def _measure(mask_full: np.ndarray, red: np.ndarray, grid, idx: int) -> Indication:
    ys, xs = np.nonzero(mask_full)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    local = mask_full[y0:y1, x0:x1]
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), angle = cv2.minAreaRect(pts)
    mpp = grid.mm_per_px
    area_px = float(local.sum())
    length_px = max(w, h, 1.0) + 1.0  # +1: pixel extent, not centre-to-centre
    short_px = min(w, h) + 1.0
    # Compact or straight shapes fill their box: the box gives the width. Curved lines
    # (cracks) do not: area / length is their true width.
    width_px = short_px if area_px / (length_px * short_px) > 0.6 else area_px / length_px
    width_px = max(min(width_px, length_px), 1.0)
    if w < h:
        angle += 90.0
    vals = red[y0:y1, x0:x1][local]
    cx, cy = grid.px_to_mm(float(xs.mean()), float(ys.mean()))
    return Indication(
        f"I{idx}",
        local,
        (int(x0), int(y0), int(x1), int(y1)),
        float(cx),
        float(cy),
        length_px * mpp,
        width_px * mpp,
        area_px * mpp * mpp,
        float(angle % 180.0),
        float(vals.mean()),
        float(vals.max()),
    )


def _gap_px(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest edge-to-edge distance between two full-size masks (px)."""
    dist = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dist[b].min())


def merge_groups(masks: list[np.ndarray], majors_px: list[float]) -> list[np.ndarray]:
    """ISO 23277: indications closer than the major dimension of the smaller one are one indication."""
    parent = list(range(len(masks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _gap_px(masks[i], masks[j]) < min(majors_px[i], majors_px[j]):
                parent[find(i)] = find(j)
    groups: dict[int, np.ndarray] = {}
    for i, m in enumerate(masks):
        r = find(i)
        groups[r] = m.copy() if r not in groups else (groups[r] | m)
    return list(groups.values())


def aligned_groups(inds: list[Indication], full_masks: dict[str, np.ndarray], count: int, gap_mm: float, mpp: float) -> list[list[str]]:
    """Rounded indications forming a line: >= count of them, each within gap_mm (edge to edge) of the next."""
    ids = [i.id for i in inds]
    if len(ids) < count:
        return []
    adj = {a: set() for a in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if _gap_px(full_masks[a], full_masks[b]) * mpp <= gap_mm:
                adj[a].add(b)
                adj[b].add(a)
    seen, out = set(), []
    centres = {i.id: np.array([i.x_mm, i.y_mm]) for i in inds}
    for a in ids:
        if a in seen:
            continue
        comp, todo = [], [a]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            todo += list(adj[n] - seen)
        if len(comp) >= count:
            pts = np.array([centres[c] for c in comp])
            ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T))
            if ev[0] <= 0.2 * max(ev[1], 1e-9):  # roughly collinear
                out.append(sorted(comp, key=lambda c: (centres[c][0], centres[c][1])))
    return out


def judge(inds: list[Indication], groups: list[list[str]], spec: dict[str, Any], profile_name: str) -> list[dict[str, Any]]:
    pen = spec["penetrant"]
    profile = pen["profiles"][profile_name]
    relevant_mm = float(profile.get("relevant_mm", pen.get("relevant_mm", 1.6)))
    in_group = {i for g in groups for i in g}
    out = []
    for ind in inds:
        kind = ind.kind
        entry: dict[str, Any] = {"relevant": ind.major_mm > relevant_mm}
        if not entry["relevant"]:
            entry.update(
                severity="negligible",
                disposition="PASS",
                rule=f"major dimension {ind.major_mm:.2f} mm <= {relevant_mm} mm: non-relevant, logged",
                action=pen["actions"]["accept"],
            )
        elif kind == "linear":
            limit = float(profile["linear_max_mm"])
            if ind.length_mm > limit:
                entry.update(
                    severity="critical",
                    disposition="REJECT",
                    rule=(
                        f"relevant linear indication {ind.length_mm:.2f} mm > {limit} mm ({profile_name})"
                        if limit > 0
                        else f"relevant linear indication {ind.length_mm:.2f} mm: any relevant linear indication rejects ({profile_name})"
                    ),
                    action=pen["actions"]["linear"],
                )
            else:
                entry.update(severity="minor", disposition="PASS", rule=f"linear {ind.length_mm:.2f} mm <= {limit} mm ({profile_name})", action=pen["actions"]["accept"])
        else:
            limit = float(profile["rounded_max_mm"])
            if ind.major_mm > limit:
                entry.update(
                    severity="major",
                    disposition="REJECT",
                    rule=f"relevant rounded indication {ind.major_mm:.2f} mm > {limit} mm ({profile_name})",
                    action=pen["actions"]["rounded"],
                )
            elif ind.id in in_group:
                a = profile.get("aligned", {})
                entry.update(
                    severity="major",
                    disposition="REJECT",
                    rule=f"{a.get('count', 4)} or more rounded indications in a line, <= {a.get('gap_mm', 1.6)} mm apart ({profile_name})",
                    action=pen["actions"]["aligned"],
                )
            else:
                entry.update(severity="minor", disposition="PASS", rule=f"rounded {ind.major_mm:.2f} mm <= {limit} mm ({profile_name})", action=pen["actions"]["accept"])
        out.append(entry)
    return out


def _conf(ind: Indication, high: float) -> float:
    return round(0.6 + 0.35 * float(np.clip(ind.peak / high - 1.0, 0.0, 1.0)), 2)


class PenetrantInspector:
    def __init__(self, cfg: Config, profile: str | None = None):
        self.cfg = cfg
        self.spec = load_spec(cfg.spec_path)
        if "penetrant" not in self.spec:
            raise ValueError(f"{cfg.spec_path} has no penetrant section")
        self.calibration = Calibration.load(cfg.calibration_path)
        self.p = {**DEFAULTS, **cfg.penetrant}
        self.profile = profile or self.spec["penetrant"].get("profile", "asme_app8")
        if self.profile not in self.spec["penetrant"]["profiles"]:
            raise ValueError(f"unknown penetrant profile {self.profile!r}")
        self.light = cfg.pair_light("top")
        self.roi = cfg.grid.plate_mask(cfg.grid.edge_margin_mm)

    def redness(self, frames: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """White-balanced colour image of the developed plate on the grid, and its redness map.
        Redness = (r - (g + b) / 2) / brightness after scaling each channel so the white
        developer is neutral: 0 on clean developer, about 1-2 on strong red dye."""
        img = lit(frames, "top", self.light)
        H = self.calibration.homographies["top"]
        chans, valid = {}, None
        for c in CHANNELS:
            warped, v = warp_to_grid(channel(img, c), H, self.cfg.grid)
            chans[c] = warped
            valid = v if valid is None else valid & v
        roi = self.roi & valid
        for c in CHANNELS:  # white balance on the developer itself
            chans[c] = chans[c] / max(float(np.median(chans[c][roi])), 1e-6)
        r, g, b = chans["r"], chans["g"], chans["b"]
        bright = np.maximum((r + g + b) / 3.0, 0.05)
        red = (r - 0.5 * (g + b)) / bright
        if self.p["smooth_px"] > 0:
            red = cv2.GaussianBlur(red, (0, 0), float(self.p["smooth_px"]))
        colour = np.dstack([b, g, r])
        return colour, np.where(roi, red, 0.0).astype(np.float32), roi

    def find(self, red: np.ndarray, merge: bool = False) -> list[tuple[np.ndarray, Indication]]:
        grid = self.cfg.grid
        low = (red > self.p["redness_low"]).astype(np.uint8)
        high = red > self.p["redness_high"]
        n, labels = cv2.connectedComponents(low, connectivity=8)
        masks = [labels == i for i in range(1, n) if high[labels == i].any()]
        min_px = self.p["min_area_mm2"] / grid.mm_per_px**2
        masks = [m for m in masks if m.sum() >= min_px]
        if merge and len(masks) > 1:
            majors = [_measure(m, red, grid, 0).major_mm / grid.mm_per_px for m in masks]
            masks = merge_groups(masks, majors)
        found = [(m, _measure(m, red, grid, 0)) for m in masks]
        found.sort(key=lambda t: (-t[1].length_mm, t[1].x_mm))
        out = []
        for i, (m, ind) in enumerate(found, 1):
            ind.id = f"I{i}"
            out.append((m, ind))
        return out

    def inspect(
        self,
        frames: Mapping[str, np.ndarray],
        part_id: str,
        process: dict[str, Any] | None = None,
        capture_dir: Path | None = None,
        overlay_path: Path | None = None,
        extra_timing_ms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        from .report import build_report

        t0 = time.perf_counter()
        colour, red, _roi = self.redness(frames)
        t1 = time.perf_counter()
        profile = self.spec["penetrant"]["profiles"][self.profile]
        found = self.find(red, merge=bool(profile.get("merge", False)))
        inds = [ind for _, ind in found]
        full = {ind.id: m for m, ind in found}
        relevant_mm = float(profile.get("relevant_mm", self.spec["penetrant"].get("relevant_mm", 1.6)))
        a = profile.get("aligned")
        rounded_rel = [i for i in inds if i.kind == "rounded" and i.major_mm > relevant_mm]
        groups = aligned_groups(rounded_rel, full, int(a["count"]), float(a["gap_mm"]), self.cfg.grid.mm_per_px) if a else []
        judged = judge(inds, groups, self.spec, self.profile)
        t2 = time.perf_counter()

        defects = []
        pen = self.spec["penetrant"]
        for ind, j in zip(inds, judged):
            kind = ind.kind
            defects.append(
                {
                    "id": ind.id,
                    "type": f"{kind}_indication",
                    "confidence": _conf(ind, self.p["redness_high"]),
                    "location_mm": {"x": round(ind.x_mm, 1), "y": round(ind.y_mm, 1)},
                    "zone": zone_for(self.spec, ind.x_mm, ind.y_mm),
                    "size": {
                        "length_mm": round(ind.length_mm, 2),
                        "width_mm": round(ind.width_mm, 2),
                        "area_mm2": round(ind.area_mm2, 2),
                        "diameter_mm": round(float(np.sqrt(4 * ind.area_mm2 / np.pi)), 2),
                    },
                    "orientation_deg": int(round(ind.orientation_deg)) % 180,
                    **j,
                    "evidence": {
                        "redness": round(ind.redness, 2),
                        "aspect_ratio": round(ind.length_mm / max(ind.width_mm, 1e-6), 1),
                        "basis": (
                            "red penetrant bled out along a line longer than 3x its width, typical of a crack"
                            if kind == "linear"
                            else "red penetrant bled out as a compact spot, typical of a pore or pit"
                        ),
                    },
                }
            )
        verdict = max((d["disposition"] for d in defects), key=VERDICTS.index, default="PASS")
        warnings = []
        dev = (process or {}).get("development_min")
        if dev is not None and not (10 <= float(dev) <= 60):
            warnings.append(f"development time {dev} min is outside 10-60 min; indication sizes may be off")
        if overlay_path is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), self.overlay(colour, inds, defects))
        timing = {**(extra_timing_ms or {}), "redness": (t1 - t0) * 1e3, "indications": (t2 - t1) * 1e3}
        return build_report(
            part_id,
            verdict,
            defects,
            self.cfg,
            self.spec,
            "red-dye indication rules v1",
            None,
            timing,
            capture_dir=str(capture_dir) if capture_dir else None,
            overlay_path=str(overlay_path) if overlay_path else None,
            mode="penetrant",
            extra={
                "penetrant": {
                    "profile": self.profile,
                    "profile_source": profile.get("source", ""),
                    "relevant_mm": relevant_mm,
                    "process": process or {},
                    "aligned_groups": groups,
                    "warnings": warnings,
                    "likely_causes": pen.get("likely_causes", {}),
                }
            },
        )

    def overlay(self, colour: np.ndarray, inds: list[Indication], defects: list[dict[str, Any]]) -> np.ndarray:
        from .pipeline import DISPOSITION_BGR

        img = np.clip(colour * 0.75 * 255, 0, 255).astype(np.uint8)
        for ind, d in zip(inds, defects):
            x0, y0, x1, y1 = ind.bbox
            color = DISPOSITION_BGR[d["disposition"]] if d["relevant"] else (160, 160, 160)
            pad = 5
            cv2.rectangle(img, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), color, 2)
            label = f"{d['id']} {ind.kind} {ind.major_mm:.1f}mm"
            cv2.putText(img, label, (x0 - pad, max(y0 - pad - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img

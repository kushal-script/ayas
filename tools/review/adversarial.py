"""Adversarial bursts built from real Blender light-group passes (independent review, 15 Nov).

Each case rebuilds a scene's burst the way burst.simulate_burst does (fixed exposure, LED ripple,
hand-shake walk, sensor noise in electrons), with one stress added, runs lightproc.process_burst
and scores the result against the noise-free, ambient-free LED image of the same render (what M
would be with a perfect sensor in the dark), both low-passed over 15 x 15 px (~1.1 mm). A case is
  OK       accepted and its error stays at the level of the same scene's unstressed captures,
  REFUSED  RETAKE (the message is printed: it must be one the operator can act on),
  SILENT   accepted with an error clearly above that level: silent corruption, a bug.
Error measures on the plate (eroded 20 px, masked pixels and a 8 px band around them left out):
p0.1 / p99.9 of the 8-bit DN error after the device's own encoding, and the share of the plate
with |error| > 6 DN. The unstressed level is the worst of 4 unstressed captures (k = 0..3).

Stresses (each a function of the frame f in B1 L A R B2):
  flicker s      a_f = 1 + N(0, s) of the ambient (5-10 %: a failing tube)
  step f0 g      the ambient changes by the factor g from frame f0 on (a light switched or a
                 cloud), the same over the whole plate
  shadow f0 g    a hand shadow: from frame f0 on the ambient on one half of the plate is x g
  refl v         the sharp part of the ambient (AMB minus its 5 mm low-pass: lamp reflections)
                 slides v px per frame against the plate (parallax of a mirrored lamp)
  amis dx dy     the A frame alone is displaced by (dx, dy) px (a frame captured late/shifted)
  droop d        both-LED frames B1, B2 carry (1 - d) x the LED light (supply sag with 2 pods on)
  amb x          the AMB pass x (a leakier hood or brighter room), combined with the others
  albedo x       all passes x (a darker plate: diffuse part only, approximately)
  .venv/bin/python tools/review/adversarial.py <exr keep dir> <out json> [scene ...]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from trinetra import lightproc as LP
from trinetra.twin import burst as B, post as P

ROOT = Path(__file__).resolve().parents[2]
SEQ = LP.SEQUENCE


def load(keep: Path, scene: str, cfg: dict):
    work = keep / scene
    plan = json.loads((work / "plan.json").read_text())
    names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
    g = P.read_light_groups(work / "render.exr", names)
    geom = P.grid_geometry(plan, cfg.get("grid") or {})
    pmask = P.plate_mask_image(geom, next(iter(g.values())).shape)
    return plan, {k: g[k] for k in ("L", "R", "AMB") if k in g}, pmask


def simulate(g, plan, rng, cap, pmask, st: dict) -> tuple[LP.Burst, float, dict]:
    """burst.simulate_burst with the stresses of st (see the module docstring)."""
    sensor = cap.get("sensor") or {}
    states = (cap.get("burst") or {}).get("led_state")
    ex = B.exposure_k(plan, cap)
    k = ex["k"]
    shape = g["L"].shape
    scale = B.res_scale(plan, shape)
    pcap = plan.get("capture") or {}
    step = float(pcap.get("shake_px_per_frame", 0.0) or 0.0) * scale
    jit = float(pcap.get("led_jitter", 0.0) or 0.0)
    ripple = float((cap.get("leds") or {}).get("frame_sigma", 0.0) or 0.0)
    gains = {"L": 1.0 + float(rng.uniform(-jit, jit)), "R": 1.0 + float(rng.uniform(-jit, jit))}
    walk = B.shake_walk(rng, len(SEQ), step)
    change = (plan.get("ambient") or {}).get("change") or {}
    sig = max(float(change.get("frame_sigma", 0.0) or 0.0), float(st.get("flicker", 0.0)))
    a_f = B.ambient_factors(rng, len(SEQ), sig)
    amb = g["AMB"] * np.float32(st.get("amb", 1.0) * st.get("albedo", 1.0))
    L0, R0 = (g[x] * np.float32(st.get("albedo", 1.0)) for x in ("L", "R"))
    ppm = B.px_per_mm(plan, shape)
    low = sharp = None
    if st.get("refl"):
        low = B._blur_small(amb, 5.0 * ppm)
        sharp = amb - low
    half = None
    if st.get("shadow"):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        ys, xs = np.nonzero(pmask)
        half = (xx > 0.5 * (xs.min() + xs.max()))[..., None].astype(np.float32)
    frames = {}
    for i, f in enumerate(SEQ):
        l_on, r_on = states[f]
        rip = 1.0 + rng.normal(0.0, ripple, 2) if ripple > 0 else np.ones(2)
        A = amb
        if sharp is not None:
            v = float(st["refl"]) * i
            A = low + LP.translate(sharp, v, 0.4 * v, interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT)
        if st.get("step") and i >= int(st["step"][0]):
            A = A * np.float32(st["step"][1])
        if half is not None and i >= int(st["shadow"][0]):
            A = A * (1.0 - half * np.float32(1.0 - st["shadow"][1]))
        S = A * np.float32(k * a_f[i])
        droop = 1.0 - float(st.get("droop", 0.0)) if (l_on and r_on) else 1.0
        if l_on:
            S = S + L0 * np.float32(k * gains["L"] * rip[0] * droop)
        if r_on:
            S = S + R0 * np.float32(k * gains["R"] * rip[1] * droop)
        dx, dy = float(walk[i, 0]), float(walk[i, 1])
        if f == "A" and st.get("amis"):
            dx, dy = dx + float(st["amis"][0]), dy + float(st["amis"][1])
        S = LP.translate(S, dx, dy, interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT)
        frames[f] = B.expose(S, rng, sensor)
    meta = {"illumination": B.illumination(plan), "px_scale": scale, "E_led": ex["E_led"],
            "material": ((plan.get("plate") or {}).get("material") or {}).get("kind"),
            "gains": gains, "a_f": [float(x) for x in a_f], "walk": {f: [float(v) for v in walk[i]] for i, f in enumerate(SEQ)}}
    return LP.Burst(frames, meta), k, gains


def score(p: LP.Processed, g, k, gains, cap, comb, pmask, albedo=1.0):
    col = np.asarray(cap["leds"]["colour"], np.float32)
    L, R = (np.clip(g[x], 0, None) * np.float32(k * gains[x] * albedo) for x in ("L", "R"))
    if comb == "mean":
        Mt = 0.5 * (L + R) / col
    else:
        Mt = np.where((LP.luma(L) <= LP.luma(R))[..., None], L, R) / col
    yM, yT = cv2.blur(LP.luma(p.M_lin), (15, 15)), cv2.blur(LP.luma(Mt), (15, 15))
    n = p.metrics["norm"]
    dn = 255.0 * (np.clip(yM / n, 0, 1) ** (1 / 2.2) - np.clip(yT / n, 0, 1) ** (1 / 2.2))
    inner = cv2.erode(pmask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    ok = inner & ~cv2.dilate(p.sat_mask.view(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    d = dn[ok]
    return {"p001": round(float(np.percentile(d, 0.1)), 2), "p999": round(float(np.percentile(d, 99.9)), 2),
            "frac6": round(float((np.abs(d) > 6).mean()), 5), "absmean": round(float(np.abs(d).mean()), 3)}


def run(keep: Path, scene: str, cases: list[tuple[str, dict]], cfg, cap, n_ref: int = 4):
    plan, g, pmask = load(keep, scene, cfg)
    comb = P.combine_of(plan, cap)
    out = []
    for name, st in [("none", {})] * n_ref + cases:
        k_var = len([o for o in out if o["case"] == name]) if name == "none" else 0
        rng = np.random.default_rng([int(plan["seed"]), 77, k_var, abs(hash(name)) % 10_000])
        burst, k, gains = simulate(g, plan, rng, cap, pmask, st)
        p = LP.process_burst(burst, pmask, cap)
        s = score(p, g, k, gains, cap, comb, pmask, st.get("albedo", 1.0))
        m = p.metrics
        out.append({"scene": scene, "case": name, **s, "gate": p.gate.verdict, "reasons": p.gate.reasons,
                    "message": p.gate.message, "r": round(float(m["r_ambient"]), 3),
                    "closure": round(float(m["closure_sigma"]), 2), "resid": round(float(m["residual_sigma"]), 2),
                    "ghost": m["ghost_bins"], "ghost_c": m["ghost_bins_coarse"], "masked": round(float(m["masked_frac"]), 4),
                    "masked_amb": round(float(m["masked_ambient_frac"]), 5), "led": round(float(m["led_level"]), 4),
                    "amb_dom": round(float(m["ambient_dominated_frac"]), 4), "motion": round(float(m["motion_px"]), 2),
                    "a_f": [round(x, 3) for x in burst.meta["a_f"]]})
    ref = [o for o in out if o["case"] == "none"]
    lo = min(o["p001"] for o in ref)
    hi = max(o["p999"] for o in ref)
    f6 = max(o["frac6"] for o in ref)
    for o in out:
        bad = o["p001"] < lo - 2.0 or o["p999"] > hi + 2.0 or o["frac6"] > max(2.0 * f6, f6 + 0.002)
        o["status"] = "REFUSED" if o["gate"] == "RETAKE" else ("SILENT" if bad else "OK")
        o["ref"] = {"p001": lo, "p999": hi, "frac6": f6}
    return out


def cases_for(illum: str) -> list[tuple[str, dict]]:
    c = [(f"flicker {s}", {"flicker": s}) for s in (0.05, 0.1)]
    c += [(f"step L->A {g}", {"step": (2, g)}) for g in (0.9, 0.8)]
    c += [(f"shadow from A x{g}", {"shadow": (2, g)}) for g in (0.8, 0.5)]
    c += [(f"refl {v} px/frame", {"refl": v}) for v in (0.5, 1.0, 2.0)]
    c += [(f"A off {d}", {"amis": d}) for d in ((2.0, 0.0), (3.0, 2.0), (4.0, 0.0))]
    c += [(f"droop {d}", {"droop": d}) for d in (0.005, 0.01, 0.02, 0.05)]
    c += [("albedo x0.3", {"albedo": 0.3})]
    return c


if __name__ == "__main__":
    keep, outp = Path(sys.argv[1]), Path(sys.argv[2])
    scenes = sys.argv[3:]
    amb_x = 1.0
    if scenes and scenes[0].startswith("amb="):
        amb_x, scenes = float(scenes[0][4:]), scenes[1:]
    cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
    cap = P.capture_config(cfg)
    res = json.loads(outp.read_text()) if outp.exists() else []
    t0 = time.time()
    for sc in scenes:
        plan = json.loads((keep / sc / "plan.json").read_text())
        cs = cases_for(B.illumination(plan))
        if amb_x != 1.0:
            cs = [(f"{n} | amb x{amb_x:g}", {**st, "amb": amb_x}) for n, st in [("base", {})] + cs]
        for o in run(keep, sc, cs, cfg, cap):
            o["amb_x"] = amb_x
            res.append(o)
            print(json.dumps({k: o[k] for k in ("scene", "case", "status", "gate", "reasons", "p001", "p999", "frac6",
                                                  "r", "closure", "ghost", "ghost_c")}), f"{time.time() - t0:.0f}s", flush=True)
        outp.write_text(json.dumps(res, indent=1))
    print("DONE", len(res))

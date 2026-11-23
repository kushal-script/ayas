"""Second-review adversarial bursts on kept full-resolution EXRs.

For each scene and case, build a burst from the render's light groups with a perturbation, and its
clean twin with the SAME shake, LED gains and noise draws; process both with lightproc; compare:
- the detector image (image_u8, each with its own normalisation, 15 px box low-pass = 1.1 mm)
  perturbed vs clean  -> d_img (what the perturbation does to the detector input);
- the device's linear M vs the noise-free, ambient-free LED image of the render (truth_err style,
  encoded with the device's norm) -> d_truth.
Class: REFUSED (gate RETAKE), OK, MARGINAL (accepted, 4-8 DN over >= 1 mm^2), SILENT (accepted, > 8 DN
over >= 1 mm^2: enough to create or hide a defect at post's 4 DN visibility threshold).
  .venv/bin/python tools/pilot/review_adversarial.py <exr keep dir> <out.jsonl> scene[,scene...] [--cases a,b] [--amb X] [--var k]
(--amb scales the render's ambient pass: x2-x5 on the 3 mm-lift renders emulate a leaking hood)
"""
import json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from ayas import lightproc as LP
from ayas.twin import burst as B, post as P

ROOT = Path(__file__).resolve().parents[2]
SEQ = ("B1", "L", "A", "R", "B2")


def build(groups, plan, cap, seed, mods):
    bcfg = cap.get("burst") or {}
    states = bcfg["led_state"]
    illum = B.illumination(plan)
    sensor = LP.sensor_for(cap.get("sensor"), illum)
    g = {k: np.asarray(v, np.float32) for k, v in groups.items()}
    shape = g["L"].shape
    k = B.exposure_k(plan, cap)["k"]
    capp = plan.get("capture") or {}
    change = (plan.get("ambient") or {}).get("change") or {}
    rs = np.random.default_rng([seed, 0])
    scale = B.res_scale(plan, shape)
    step = float(mods.get("shake", capp.get("shake_px_per_frame", 0.0)) or 0.0) * scale
    jit = float(capp.get("led_jitter", 0.0) or 0.0)
    gains = {"L": 1.0 + float(rs.uniform(-jit, jit)), "R": 1.0 + float(rs.uniform(-jit, jit))}
    walk = B.shake_walk(rs, 5, step)
    ripple = float((cap.get("leds") or {}).get("frame_sigma", 0.0) or 0.0)
    rip = [1.0 + rs.normal(0.0, ripple, 2) for _ in SEQ]
    a_f = B.ambient_factors(rs, 5, float(change.get("frame_sigma", 0.0) or 0.0))
    fl = float(mods.get("flicker", 0.0))
    fl_rng = np.random.default_rng([seed, 7, int(mods.get("fseed", 0))])
    if fl > 0:
        a_f = a_f * np.maximum(1.0 + fl_rng.normal(0.0, fl, 5), 0.0)
    ppm = B.px_per_mm(plan, shape)
    theta = float(rs.uniform(0.0, 2 * math.pi))
    sh = float(change.get("shadow_shift_mm", 0.0) or 0.0) * ppm
    sigma = float(bcfg.get("shadow_sigma_mm", 5.0)) * ppm
    roi = mods["_roi"]
    parts = B.illumination_parts(g["AMB"], sigma, roi) if sh > 0 else None
    frames = {}
    for i, f in enumerate(SEQ):
        l_on, r_on = states[f]
        amb = g["AMB"] if sh == 0 or i == 0 else B.shadow_shift(
            g["AMB"], [i * sh * math.cos(theta), i * sh * math.sin(theta)], sigma, parts=parts)
        if "amb_fn" in mods:
            amb = mods["amb_fn"](i, f, amb)
        am = float((mods.get("a_mult") or {}).get(f, 1.0))
        S = amb * np.float32(k * a_f[i] * am)
        droop = float(mods.get("both", 1.0)) if (l_on and r_on) else 1.0
        if l_on:
            S += g["L"] * np.float32(k * gains["L"] * rip[i][0] * droop)
        if r_on:
            S += g["R"] * np.float32(k * gains["R"] * rip[i][1] * droop)
        ex = (mods.get("extra_shift") or {}).get(f, (0.0, 0.0))
        S = LP.translate(S, float(walk[i, 0] + ex[0]), float(walk[i, 1] + ex[1]), interp=cv2.INTER_CUBIC,
                         border=cv2.BORDER_REFLECT)
        frames[f] = B.expose(S, np.random.default_rng([seed, 1, i]), sensor)
    meta = {"shifts": {f: [float(walk[i, 0]), float(walk[i, 1])] for i, f in enumerate(SEQ)},
            "px_scale": scale, "px_per_mm": ppm, "illumination": illum,
            "material": ((plan.get("plate") or {}).get("material") or {}).get("kind"),
            "E_led": B.exposure_k(plan, cap)["E_led"], "led_gains": gains, "k": k}
    return LP.Burst(frames, meta)


def bar(shape, cx, cy, length, width, angle_deg, peak):
    """A tube's mirror image: a Gaussian-profile bar (px units), (H, W, 3) radiance."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
    v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    prof = np.exp(-0.5 * (v / (width / 2.355)) ** 2) * (1.0 / (1.0 + np.exp((np.abs(u) - length / 2) / 3.0)))
    return (peak * prof)[..., None] * np.ones(3, np.float32)


def stats(d, ok, ppm):
    v = d[ok]
    a = np.abs(d) * ok
    mm2 = 1.0 / (ppm * ppm)                       # mm^2 per px
    return {"p001": round(float(np.percentile(v, 0.1)), 2), "p999": round(float(np.percentile(v, 99.9)), 2),
            "mm2_gt4": round(float((a > 4).sum() * mm2), 2), "mm2_gt8": round(float((a > 8).sum() * mm2), 2),
            "max": round(float(a.max()), 1)}


def main():
    a = sys.argv[1:]
    keep, outp, scenes = Path(a[0]), Path(a[1]), a[2].split(",")
    cases_sel = a[a.index("--cases") + 1].split(",") if "--cases" in a else None
    amb_x = float(a[a.index("--amb") + 1]) if "--amb" in a else 1.0
    var = int(a[a.index("--var") + 1]) if "--var" in a else 0
    cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
    cap = P.capture_config(cfg)
    names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
    post_cfg = cfg.get("post") or {}
    t0 = time.time()
    with open(outp, "a") as fo:
        for scene in scenes:
            work = keep / scene
            plan = json.loads((work / "plan.json").read_text())
            lights = P.read_light_groups(work / "render.exr", names)
            geom = P.grid_geometry(plan, cfg.get("grid") or {})
            pmask = P.plate_mask_image(geom, next(iter(lights.values())).shape)
            comb = P.combine_of(plan, cap)
            mult = P.amb_multiplier(np.random.default_rng([int(plan["seed"]), var]), plan.get("ambient"), var, post_cfg)
            base = P.scale_ambient(lights, mult * amb_x)
            shape = base["L"].shape
            inner = cv2.erode(pmask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
            ys, xs = np.nonzero(inner)
            cy, cx = float(ys.mean()), float(xs.mean())
            span = float(min(ys.max() - ys.min(), xs.max() - xs.min()))
            ppm = B.px_per_mm(plan, shape)
            led = float(np.median(LP.luma(base["L"] + base["R"])[inner]))
            amb_med = float(np.median(LP.luma(base["AMB"])[inner]))
            yA = LP.luma(base["AMB"])
            # a bright ambient region present in the render (a mirrored lamp) to move: AMB above 3x its median
            refl = None

            def tube(peak_x, mm_per_frame=0.0, flick=0.0, length_mm=40.0, width_mm=4.0):
                L = length_mm * ppm
                Wd = width_mm * ppm
                fr = np.random.default_rng(11)
                amp = 1.0 + flick * fr.normal(0, 1, 5)

                def fn(i, f, amb):
                    return amb + bar(shape, cx + 0.15 * span + i * mm_per_frame * ppm, cy - 0.1 * span, L, Wd, 25.0,
                                     peak_x * led) * np.float32(amp[i])
                return fn

            def amb_translate(px_per_frame):
                def fn(i, f, amb):
                    return LP.translate(amb, i * px_per_frame, 0.3 * i * px_per_frame, interp=cv2.INTER_CUBIC,
                                        border=cv2.BORDER_REFLECT)
                return fn

            def hand(frac, from_i=2):          # a hand shadow arriving from frame from_i (A) on, over half the plate
                yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
                edge = 1.0 / (1.0 + np.exp(-(xx - cx) / (2.0 * ppm)))
                m = (1.0 - (1.0 - frac) * edge)[..., None]

                def fn(i, f, amb):
                    return amb * m if i >= from_i else amb
                return fn

            cases = {
                "none": {},
                "flicker5_s0": {"flicker": 0.05, "fseed": 0}, "flicker5_s1": {"flicker": 0.05, "fseed": 1},
                "flicker10_s0": {"flicker": 0.10, "fseed": 0}, "flicker10_s1": {"flicker": 0.10, "fseed": 1},
                "flicker10_s2": {"flicker": 0.10, "fseed": 2},
                "A_only_+5%": {"a_mult": {"A": 1.05}}, "A_only_-10%": {"a_mult": {"A": 0.90}},
                "L_only_+10%": {"a_mult": {"L": 1.10}},
                "step_after_L_x0.8": {"a_mult": {"A": 0.8, "R": 0.8, "B2": 0.8}},
                "step_after_A_x1.2": {"a_mult": {"R": 1.2, "B2": 1.2}},
                "A_shift_2_0": {"extra_shift": {"A": (2.0, 0.0)}}, "A_shift_3_2": {"extra_shift": {"A": (3.0, 2.0)}},
                "A_shift_4_0": {"extra_shift": {"A": (4.0, 0.0)}},
                "droop_2%": {"both": 0.98}, "droop_5%": {"both": 0.95},
                "tube_r3_move1mm": {"amb_fn": tube(3.0, 0.06)}, "tube_r3_move0": {"amb_fn": tube(3.0, 0.0)},
                "tube_r6_move2px": {"amb_fn": tube(6.0, 2.0 / ppm)}, "tube_r3_flick10": {"amb_fn": tube(3.0, 0.0, 0.10)},
                "tube_r20_move1px_small": {"amb_fn": tube(20.0, 1.0 / ppm, 0.0, 15.0, 2.0)},
                "amb_moves_1px": {"amb_fn": amb_translate(1.0)}, "amb_moves_3px": {"amb_fn": amb_translate(3.0)},
                "hand_from_A_x0.5": {"amb_fn": hand(0.5, 2)}, "hand_from_R_x0.5": {"amb_fn": hand(0.5, 3)},
                "shake_3px": {"shake": 3.0}, "shake_6px": {"shake": 6.0},
            }
            sel = [c for c in cases if cases_sel is None or c in cases_sel or c == "none"]
            seed = 4242 + var
            Mt_cache = {}
            clean = None
            for name in sel:
                mods = {**cases[name], "_roi": pmask}
                cmods = {"_roi": pmask, **({"shake": cases[name]["shake"]} if "shake" in cases[name] else {})}
                bu = build(base, plan, cap, seed, mods)
                p = LP.process_burst(bu, pmask, cap)
                key = cmods.get("shake")
                if key not in Mt_cache:
                    pc = LP.process_burst(build(base, plan, cap, seed, cmods), pmask, cap) if name != "none" or key else p
                    Mt_cache[key] = pc
                pc = Mt_cache[key]
                kexp, gg = bu.meta["k"], bu.meta["led_gains"]
                Mt = P._led_min({"L": base["L"] * kexp * gg["L"], "R": base["R"] * kexp * gg["R"]}, cap, comb).astype(np.float32)
                n = p.metrics["norm"]
                enc = lambda y: 255.0 * np.clip(y / n, 0, 1) ** (1 / 2.2)  # noqa: E731
                lp = lambda y: cv2.blur(y.astype(np.float32), (15, 15))  # noqa: E731
                d_truth = enc(lp(LP.luma(p.M_lin))) - enc(lp(LP.luma(Mt)))
                d_img = lp(p.image_u8.astype(np.float32).mean(-1)) - lp(pc.image_u8.astype(np.float32).mean(-1))
                sat = p.sat_mask | pc.sat_mask
                ok = inner & ~cv2.dilate(sat.view(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
                si, st = stats(d_img, ok, ppm), stats(d_truth, ok, ppm)
                m = p.metrics
                if p.gate.verdict == "RETAKE":
                    cls = "REFUSED"
                elif si["mm2_gt8"] >= 1.0:
                    cls = "SILENT"
                elif si["mm2_gt4"] >= 1.0:
                    cls = "MARGINAL"
                else:
                    cls = "OK"
                r = {"scene": scene, "amb_x": amb_x, "var": var, "case": name, "class": cls, "gate": p.gate.verdict,
                     "reasons": p.gate.reasons, "message": p.gate.message,
                     "r": round(float(m["r_ambient"]), 4), "local_r_med": round(amb_med / max(led, 1e-12), 4),
                     "led_level": round(float(m["led_level"]), 4), "masked": round(float(m["masked_frac"]), 4),
                     "masked_amb": round(float(m["masked_ambient_frac"]), 5), "amb_dom": m["ambient_dominated_frac"],
                     "motion": round(float(m["motion_px"]), 2), "resid": round(float(m["residual_sigma"]), 2),
                     "closure": round(float(m["closure_sigma"]), 2), "ghost": m["ghost_bins"],
                     "ghost_c": m["ghost_bins_coarse"], "norm_ratio": round(n / pc.metrics["norm"], 4),
                     "clean_gate": pc.gate.verdict, "d_img": si, "d_truth": st}
                fo.write(json.dumps(r) + "\n")
                fo.flush()
                print(f"{scene} x{amb_x} {name:24s} {cls:8s} {p.gate.verdict:12s} {','.join(p.gate.reasons):28s} "
                      f"r={r['r']:.3f} g={m['ghost_bins']} gc={m['ghost_bins_coarse']} cl={r['closure']} "
                      f"img[{si['p001']},{si['p999']}] >8:{si['mm2_gt8']}mm2 truth[{st['p001']},{st['p999']}] "
                      f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

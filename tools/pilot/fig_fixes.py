"""reports/figures/pilot_gate_fixes.png: what the 15 Nov pilot fixes changed in the images.
Top: the s900028 (aluminium) pit cluster under the hood ring: naive, burst before the fix (LED-clipped
pixels inpainted; crops saved before the repost), burst after. Bottom: error of the burst's M against
the noise-free LED image (1.1 mm low-pass, DN after the device encoding) for the two open captures the
old gate accepted and the new gate refuses, next to their grid.
The top row needs the pre-fix crop (made from the pre-fix grids, which the fix replaced); it is not
reproducible from the current outputs.
  fig_fixes.py <old pit crop png> <matrix dir> <exr keep dir> <out png>
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, yaml
from trinetra import lightproc as LP
from trinetra.twin import post as P

ROOT = Path(__file__).resolve().parents[2]
old_crop, mx, keep, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
F = cv2.FONT_HERSHEY_SIMPLEX
W = 1800
# top row: old crops (naive v0, burst v0, burst v1, burst C3h v1: 4 tiles of 432 px) + the new burst v1 crop
old = cv2.imread(str(old_crop))
tw = old.shape[1] // 4
tiles = [old[:, i * tw:(i + 1) * tw] for i in range(3)]
x0, y0, x1, y1 = 1128 - 40, 1129 - 40, 1156 + 40, 1147 + 40
new = cv2.imread(str(mx / "grids/burst/test/s900028_C0h_v1.png"))[y0:y1, x0:x1]
new = cv2.resize(new, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
labels = ["naive (old method)", "burst before fix, v0", "burst before fix, v1", "burst after fix, v1"]
top = []
for t, lab in zip(tiles + [new], labels):
    t = t.copy()
    t[:28] = 24
    cv2.putText(t, lab, (6, 20), F, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    top.append(t)
top = np.hstack(top)
top = cv2.resize(top, (W, round(top.shape[0] * W / top.shape[1])), interpolation=cv2.INTER_AREA)
head1 = np.full((34, W, 3), 24, np.uint8)
cv2.putText(head1, "Hood, dark room, aluminium s900028 pit cluster D1: the ring's LED-clipped pixels were inpainted "
            "(contrast 101 -> 3-15 DN); now kept white (89-97 DN)", (8, 23), F, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

cfg = yaml.safe_load((ROOT / "configs/twin.yaml").read_text())
cap = P.capture_config(cfg)
names = tuple((cfg.get("render") or {}).get("light_groups") or P.GROUPS)
LPm, B = P.capture_modules()
bot = []
for scene, k, what in (("s900028_C4o", 0, "C4 open: moving hand shadow"), ("s900028_C1o", 0, "C1 open: mirrored ceiling tube")):
    plan = json.loads((keep / scene / "plan.json").read_text())
    g = P.read_light_groups(keep / scene / "render.exr", names)
    geom = P.grid_geometry(plan, cfg.get("grid") or {})
    pm = P.plate_mask_image(geom, next(iter(g.values())).shape)
    rng = np.random.default_rng([int(plan["seed"]), k])
    groups = P.scale_ambient(g, P.amb_multiplier(rng, plan.get("ambient"), k, cfg.get("post") or {}))
    b = B.simulate_burst(groups, plan, np.random.default_rng([int(plan["seed"]), k, 1 + P.ARMS.index("burst")]), cap, roi_mask=pm)
    p = LP.process_burst(b, pm, cap)
    kexp = B.exposure_k(plan, cap, b.meta.get("E_led"))["k"]
    Mt = P._led_min({"L": groups["L"] * kexp, "R": groups["R"] * kexp}, cap, P.combine_of(plan, cap)).astype(np.float32)
    n = p.metrics["norm"]
    yM, yT = cv2.blur(LP.luma(p.M_lin), (15, 15)), cv2.blur(LP.luma(Mt), (15, 15))
    dn = 255.0 * (np.clip(yM / n, 0, 1) ** (1 / 2.2) - np.clip(yT / n, 0, 1) ** (1 / 2.2))
    dn[~pm] = 0
    vis = np.clip(128 + 6 * dn, 0, 255).astype(np.uint8)          # grey = 0, 1 DN = 6 levels
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TWILIGHT_SHIFTED)
    grid = cv2.imread(str(mx / "grids/burst/test" / f"{scene}_v{k}.png"))
    gv = P.rectify(vis, geom)
    s = 440
    pair = [cv2.resize(grid, (s, s), interpolation=cv2.INTER_AREA), cv2.resize(gv, (s, s), interpolation=cv2.INTER_AREA)]
    for im, lab in zip(pair, (f"{what}: burst grid", "M error vs noise-free LEDs (+-21 DN full scale)")):
        im[:26] = 24
        cv2.putText(im, lab, (5, 18), F, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cap_txt = np.full((30, 2 * s, 3), 24, np.uint8)
    reasons = ",".join(p.gate.reasons) or "-"
    cv2.putText(cap_txt, f"old gate: accepted; new gate: {p.gate.verdict} ({reasons}); M error p1/p99 "
                f"{np.percentile(dn[pm], 1):+.0f}/{np.percentile(dn[pm], 99):+.0f} DN", (5, 20), F, 0.45,
                (120, 200, 255), 1, cv2.LINE_AA)
    bot.append(np.vstack([np.hstack(pair), cap_txt]))
bot = np.hstack(bot)
bot = cv2.resize(bot, (W, round(bot.shape[0] * W / bot.shape[1])), interpolation=cv2.INTER_AREA)
head2 = np.full((34, W, 3), 24, np.uint8)
cv2.putText(head2, "Open handheld, aluminium: two accepted captures that carried ambient error in M (the oracle's verdict was "
            "still right); the new gate refuses both", (8, 23), F, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(outp), np.vstack([head1, top, head2, bot]))
print("wrote", outp)

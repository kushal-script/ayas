"""Train the depth network on datagen.py shards; export ONNX for the Hailo compile.

usage: python training/depthnet/train.py --data artifacts/depthnet/data --out artifacts/depthnet/run1 --epochs 30
The shards of the highest --seed are the validation set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402


def ring_median(h: np.ndarray, n: int) -> float:
    k = max(2, n // 8)
    a = h[:n, :n]
    ring = np.ones((n, n), bool)
    ring[k:n - k, k:n - k] = False
    return float(np.median(a[ring]))


def load(paths):
    X, Y, V, D = [], [], [], []
    for p in paths:
        z = np.load(p)
        meta = json.loads(bytes(z["meta"]).decode())
        for obs, valid, h, m in zip(z["obs"], z["valid"], z["height"], meta):
            n = m["n"]
            X.append(M.prepare(obs, valid, m["illumination"], m["rel_u_mm"], m["rel_v_mm"], m["s_mm"]))
            hr = np.zeros((M.N, M.N), np.float32)
            hr[:n, :n] = (h[:n, :n] - ring_median(h, n)) / M.H_UNIT_MM
            Y.append(hr)
            V.append(valid.astype(np.float32))
            D.append({"kind": m["kind"], "depth_mm": -min(0.0, float(hr[:n, :n].min()) * M.H_UNIT_MM), "n": n})
    return np.stack(X), np.stack(Y)[:, None], np.stack(V)[:, None], D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--width", type=int, default=24); ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    shards = sorted(Path(a.data).glob("shard_*.npz"))
    seeds = sorted({p.name.split("_")[1] for p in shards})
    val = [p for p in shards if p.name.split("_")[1] == seeds[-1]]
    tr = [p for p in shards if p not in val]
    Xt, Yt, Vt, _ = load(tr)
    Xv, Yv, Vv, Dv = load(val)
    print(f"train {len(Xt)} val {len(Xv)} device {dev}", flush=True)
    net = M.build(a.width).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Xt) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    T = lambda x: torch.as_tensor(x, device=dev)
    Xv_t, Yv_t, Vv_t = T(Xv), T(Yv), T(Vv)

    def loss_fn(p, y, v):
        l1 = (torch.abs(p - y) * v).sum() / v.sum()
        dmin = ((p * v).amin(dim=(1, 2, 3)) - (y * v).amin(dim=(1, 2, 3))).abs().mean()   # the deepest point
        return l1 + 0.5 * dmin

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best, t0 = 1e9, time.time()
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        net.train()
        idx = rng.permutation(len(Xt))
        for k in range(0, len(idx) - a.bs + 1, a.bs):
            b = idx[k:k + a.bs]
            x, y, v = T(Xt[b]), T(Yt[b]), T(Vt[b])
            if rng.random() < 0.5:                       # mirror up-down (L/R geometry is left-right)
                x, y, v = x.flip(2), y.flip(2), v.flip(2)
            opt.zero_grad()
            loss = loss_fn(net(x), y, v)
            loss.backward()
            opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            P = torch.cat([net(Xv_t[k:k + 512]) for k in range(0, len(Xv_t), 512)])
            vl = float(loss_fn(P, Yv_t, Vv_t))
            pd = (-(P * Vv_t).amin(dim=(1, 2, 3)).clamp(max=0) * M.H_UNIT_MM).cpu().numpy()
        err = {k: float(np.mean([abs(pd[i] - d["depth_mm"]) for i, d in enumerate(Dv) if d["kind"] == k])) for k in ("dent", "pit", "scratch")}
        print(f"epoch {ep + 1:2d} val loss {vl:.4f}  depth MAE dent {err['dent']:.3f} pit {err['pit']:.3f} scratch {err['scratch']:.3f} mm  {time.time() - t0:.0f} s", flush=True)
        if vl < best:
            best = vl
            torch.save(net.state_dict(), out / "best.pt")
            json.dump({"epoch": ep + 1, "val_loss": vl, "depth_mae_mm": err, "width": a.width, "n_train": len(Xt)}, open(out / "best.json", "w"), indent=1)
    net.load_state_dict(torch.load(out / "best.pt", map_location=dev)); net = net.to("cpu").eval()
    torch.onnx.export(net, torch.zeros(1, 7, M.N, M.N), out / "depthnet.onnx", input_names=["crop"], output_names=["height"],
                      opset_version=11, dynamo=False)
    print("exported", out / "depthnet.onnx", flush=True)


if __name__ == "__main__":
    main()

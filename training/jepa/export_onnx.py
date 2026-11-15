"""Export the JEPA encoder and anomaly heads to ONNX and measure CPU latency (docs/JEPA.md, section 6).

Graphs (opset 17, batch 1, 1 x 3 x 224 x 224 ImageNet-normalised RGB in):
- <name>_encoder.onnx     EMA target encoder -> (1, 196, D) patch embeddings (the NPU candidate)
- <name>_knn.onnx         encoder + PatchCore head: z = (min_k ||e - bank_k|| - mu) / sigma -> (1, 14, 14)
                          (the coreset is a constant; this is what trinetra.anomaly.OnnxPatchScorer loads)
- <name>_knn_head.onnx    the head alone, (1, 196, D) -> (1, 14, 14) (CPU part if the encoder runs on Hailo)
- <name>_pred.onnx        predictor-error head: 1 target pass + 4 masked context passes + 4 predictor passes
- *_int8.onnx             onnxruntime dynamic INT8 (MatMul/Gemm weights) of encoder and knn
plus <name>_knn.json (scale, threshold, bank stats, for OnnxPatchScorer) and latency.json:
median / p90 ms on this CPU with 1 and 4 intra-op threads, and the ONNX-vs-PyTorch error.

  .venv-train/bin/python training/jepa/export_onnx.py --ckpt runs/jepa/smoke/encoder.pt \
      --train <pairs dir> --out runs/jepa/onnx --name jepa_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from jepa import grids as G  # noqa: E402
from jepa.eval_anomaly import Bank, Extractors, collect_bank  # noqa: E402
from jepa.masks import checker_groups  # noqa: E402
from jepa.model import gather_tokens, load_jepa  # noqa: E402

# ops the Hailo DFC maps to the NPU for the zoo's vit_tiny (conv patch embed, MatMul, Softmax,
# LayerNorm, GELU, Add/Mul, Reshape/Transpose); anything else is flagged for review
HAILO_OK = {"Conv", "MatMul", "Gemm", "Add", "Mul", "Sub", "Div", "Softmax", "LayerNormalization", "Erf", "Gelu",
            "Reshape", "Transpose", "Flatten", "Concat", "Slice", "Split", "Squeeze", "Unsqueeze", "Identity",
            "ReduceMean", "Pow", "Sqrt", "Constant", "Shape", "Cast", "Tanh"}


class EncoderOnly(nn.Module):
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class KnnHead(nn.Module):
    def __init__(self, bank: torch.Tensor, mu: float, sigma: float, grid: tuple[int, int]):
        super().__init__()
        self.register_buffer("bank_t", bank.t().contiguous())                # (D, K)
        self.register_buffer("bank_sq", (bank ** 2).sum(1)[None, None])      # (1, 1, K)
        self.mu, self.sigma, self.grid = float(mu), float(sigma), grid

    def forward(self, e: torch.Tensor) -> torch.Tensor:                      # (1, P, D)
        d2 = (e ** 2).sum(-1, keepdim=True) - 2 * (e @ self.bank_t) + self.bank_sq
        d = torch.sqrt(torch.clamp(d2.min(-1).values, min=0.0))
        return ((d - self.mu) / self.sigma).reshape(e.shape[0], *self.grid)


class EncKnn(nn.Module):
    def __init__(self, enc: nn.Module, head: KnnHead):
        super().__init__()
        self.enc, self.head = enc, head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


class PredHead(nn.Module):
    """Predictor error with fixed checkerboard masks (static gathers, ONNX-friendly)."""

    def __init__(self, jepa: nn.Module, mu: float, sigma: float, block: int = 2):
        super().__init__()
        self.j = jepa
        grid = jepa.encoder.grid
        n = grid[0] * grid[1]
        gs = checker_groups(grid, block)            # groups differ in size (14 is not a multiple of 4)
        self.n_groups = len(gs)
        for i, g in enumerate(gs):
            self.register_buffer(f"tgt{i}", torch.as_tensor(g, dtype=torch.long))
            self.register_buffer(f"keep{i}", torch.as_tensor(np.setdiff1d(np.arange(n), g), dtype=torch.long))
        order = np.argsort(np.concatenate(gs))
        self.register_buffer("order", torch.as_tensor(order, dtype=torch.long))
        self.mu, self.sigma, self.grid = float(mu), float(sigma), grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.j.target_encoder(x)
        h = torch.nn.functional.layer_norm(h, (h.shape[-1],))
        errs = []
        for i in range(self.n_groups):
            k, g = getattr(self, f"keep{i}")[None], getattr(self, f"tgt{i}")[None]
            p = self.j.predictor(self.j.encoder(x, k), k, g)
            errs.append(((p - gather_tokens(h, g)) ** 2).mean(-1))
        e = torch.cat(errs, 1)[:, self.order]
        return ((e - self.mu) / self.sigma).reshape(x.shape[0], *self.grid)


def export(model: nn.Module, path: Path, example: torch.Tensor, out_name: str, in_name: str = "image") -> None:
    model.eval()
    torch.onnx.export(model, (example,), str(path), input_names=[in_name], output_names=[out_name],
                      opset_version=17, do_constant_folding=True, dynamo=False)


def ops_of(path: Path) -> Counter:
    import onnx

    m = onnx.load(str(path))
    return Counter(n.op_type for n in m.graph.node)


def bench(path: Path, x: np.ndarray, threads: int, runs: int = 30, warm: int = 5) -> dict[str, float]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    for _ in range(warm):
        s.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        s.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(float(np.median(ts)), 2), "p90_ms": round(float(np.percentile(ts, 90)), 2),
            "min_ms": round(float(np.min(ts)), 2)}


def bench_only(out: Path, name: str, runs: int, report: str | None, pi_factor: Sequence[float]) -> dict[str, Any]:
    """Re-time the ONNX files of an earlier export (e.g. when the machine is quieter)."""
    res = json.loads((out / "latency.json").read_text())
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for k, r in res["models"].items():
        p = out / r["file"]
        inp = x if k != "knn_head" else run_onnx(out / f"{name}_encoder.onnx", x)
        r["cpu_1thread"] = bench(p, inp, 1, runs)
        r["cpu_4thread"] = bench(p, inp, 4, runs)
        print(k, r["cpu_1thread"], r["cpu_4thread"], flush=True)
    res["load_avg"] = list(os.getloadavg())
    (out / "latency.json").write_text(json.dumps(res, indent=1))
    if report:
        write_report(res, Path(report), pi_factor)
    return res


def run_onnx(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="TRAINING grid dirs for the memory bank")
    ap.add_argument("--out", default="runs/jepa/onnx")
    ap.add_argument("--name", default="jepa")
    ap.add_argument("--scale", type=float, default=2.0, help="window scale the app uses (2 = 448 px -> 224)")
    ap.add_argument("--coreset", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=3.0, help="z threshold written to the meta json")
    ap.add_argument("--max-train-grids", type=int, default=40)
    ap.add_argument("--runs", type=int, default=30, help="timed runs per model and thread count (0: no benchmark)")
    ap.add_argument("--report", help="also write a markdown latency report here (e.g. reports/jepa_onnx.md)")
    ap.add_argument("--pi-factor", nargs=2, type=float, default=[3.0, 5.0],
                    help="Pi 5 / Mac M3 per-thread slowdown range used for the projection")
    ap.add_argument("--bench-only", action="store_true", help="only re-time the files of an earlier export in --out")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.bench_only:
        return bench_only(out, a.name, a.runs, a.report, a.pi_factor)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    jepa, ck = load_jepa(a.ckpt)
    jepa = jepa.float().eval()

    train = [r for r in G.index_grids(a.train, ["burst"]) if r.accepted]
    G.refuse_test_plates(train)
    train = train[: a.max_train_grids]
    ex = Extractors(["jepa"], a.ckpt, dev)
    feats = collect_bank(ex, train, ["jepa_knn", "jepa_pred"], [a.scale], 2.0, 16)
    bank = Bank(feats[("jepa_knn", a.scale)], a.coreset, 0)
    pv = feats[("jepa_pred", a.scale)].squeeze(1)
    pmu, psig = float(pv.mean()), float(pv.std())
    grid = jepa.encoder.grid

    enc = EncoderOnly(jepa.target_encoder).eval()
    head = KnnHead(bank.bank, bank.mu, bank.sigma, grid).eval()
    full = EncKnn(jepa.target_encoder, head).eval()
    pred = PredHead(jepa, pmu, psig).eval()
    x = torch.randn(1, 3, 224, 224)
    paths = {"encoder": out / f"{a.name}_encoder.onnx", "knn": out / f"{a.name}_knn.onnx",
             "knn_head": out / f"{a.name}_knn_head.onnx", "pred": out / f"{a.name}_pred.onnx"}
    with torch.no_grad():
        export(enc, paths["encoder"], x, "patch_emb")
        export(full, paths["knn"], x, "patch_score")
        export(head, paths["knn_head"], enc(x), "patch_score", in_name="patch_emb")
        export(pred, paths["pred"], x, "patch_score")
        ref = {"encoder": enc(x).numpy(), "knn": full(x).numpy(), "pred": pred(x).numpy()}
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for k in ("encoder", "knn"):
        q = out / f"{a.name}_{k}_int8.onnx"
        quantize_dynamic(str(paths[k]), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        paths[f"{k}_int8"] = q
    xn = x.numpy()
    res: dict[str, Any] = {"host": platform.platform(), "cpu": platform.processor() or platform.machine(),
                           "ckpt_step": int(ck.get("step", 0)), "scale": a.scale, "coreset": int(bank.bank.shape[0]),
                           "bank": {"mu": bank.mu, "sigma": bank.sigma, "pool": bank.n_pool}, "models": {}}
    for k, p in paths.items():
        r: dict[str, Any] = {"file": p.name, "mb": round(p.stat().st_size / 1e6, 2)}
        o = run_onnx(p, xn) if k != "knn_head" else run_onnx(p, ref["encoder"])
        base = k.replace("_int8", "")
        if base in ref:
            r["max_abs_err_vs_torch"] = float(np.abs(o - ref[base]).max())
            r["corr_vs_torch"] = float(np.corrcoef(o.ravel(), ref[base].ravel())[0, 1])
        inp = xn if k != "knn_head" else ref["encoder"]
        if a.runs > 0:
            r["cpu_1thread"] = bench(p, inp, 1, a.runs)
            r["cpu_4thread"] = bench(p, inp, 4, a.runs)
        if not k.endswith("int8"):
            ops = ops_of(p)
            r["ops"] = dict(sorted(ops.items()))
            r["ops_not_in_hailo_vit_set"] = sorted(set(ops) - HAILO_OK)
        res["models"][k] = r
        print(k, json.dumps({kk: vv for kk, vv in r.items() if kk != "ops"}), flush=True)
    meta = {"img_size": 224, "scale": a.scale, "threshold": a.threshold, "grid": list(grid),
            "bank_mu": bank.mu, "bank_sigma": bank.sigma, "coreset": int(bank.bank.shape[0]),
            "train_plates": sorted({r.plate for r in train}), "ckpt_step": int(ck.get("step", 0)),
            "note": "z-scored NN distance to normal TRAINING-plate patches; see docs/JEPA.md"}
    (out / f"{a.name}_knn.json").write_text(json.dumps(meta, indent=1))
    (out / f"{a.name}_knn_int8.json").write_text(json.dumps(meta, indent=1))
    res["load_avg"] = list(os.getloadavg())
    (out / "latency.json").write_text(json.dumps(res, indent=1))
    if a.report:
        write_report(res, Path(a.report), a.pi_factor)
    return res


def write_report(res: dict[str, Any], path: Path, pi_factor: Sequence[float]) -> None:
    """reports/jepa_onnx.md: the measured Mac CPU latency and the Pi 5 projection (labelled as such)."""
    lo, hi = float(pi_factor[0]), float(pi_factor[1])
    L = ["# JEPA ONNX export and CPU latency\n",
         f"Host: {res['host']} ({res['cpu']}). Checkpoint step {res['ckpt_step']} (SMOKE model: the latency does not "
         "depend on the training, the scores do). Batch 1, 1 x 3 x 224 x 224, onnxruntime CPU EP, ORT_ENABLE_ALL; "
         "median (min) of the timed runs after 5 warm-ups. Other jobs (Blender renders, post) shared the Mac while "
         f"this ran (load average {', '.join(f'{v:.0f}' for v in res.get('load_avg', []))} on 8 cores), so the "
         "medians are upper-side numbers; the min is the closer estimate of an idle core.\n",
         f"Pi 5 columns are a PROJECTION, not a measurement: Mac min ms x {lo:g}-{hi:g} (Cortex-A76 2.4 GHz vs M3 P-core: "
         "about 3.4x less fp32 FMA throughput per core, about 6x less memory bandwidth; docs/JEPA.md section 6).\n",
         "| model | MB | Mac 1 thread ms median (min) | Mac 4 threads ms median (min) | Pi 5 1 thread ms (proj.) | Pi 5 4 threads ms (proj.) | "
         "max abs err vs torch | ops outside the ViT set |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res["models"].items():
        c1, c4 = r.get("cpu_1thread") or {}, r.get("cpu_4thread") or {}
        m1, m4 = c1.get("min_ms", c1.get("median_ms")), c4.get("min_ms", c4.get("median_ms"))
        f = lambda x, s: "–" if x is None else f"{x * s:.0f}"  # noqa: E731
        t1 = "–" if not c1 else f"{c1['median_ms']} ({c1.get('min_ms', '–')})"
        t4 = "–" if not c4 else f"{c4['median_ms']} ({c4.get('min_ms', '–')})"
        L.append(f"| {r['file']} | {r['mb']} | {t1} | {t4} | "
                 f"{f(m1, lo)}–{f(m1, hi)} | {f(m4, lo)}–{f(m4, hi)} | "
                 f"{r.get('max_abs_err_vs_torch', float('nan')):.2g} | {', '.join(r.get('ops_not_in_hailo_vit_set') or []) or '–'} |")
    L.append("\nOp counts per graph:\n")
    for k, r in res["models"].items():
        if r.get("ops"):
            L.append(f"- `{r['file']}`: " + ", ".join(f"{op} {n}" for op, n in r["ops"].items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

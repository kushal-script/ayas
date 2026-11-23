"""JEPA package (training/jepa): masks, grid pairing, canonical frame, metrics, model.

The numpy parts run in the dev venv; the torch parts skip cleanly when torch / timm are absent
(.venv has no torch; run them with .venv-train)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from jepa import grids as G  # noqa: E402
from jepa import masks as M  # noqa: E402
from jepa import metrics as MX  # noqa: E402
from jepa import pairs as P  # noqa: E402

MMPP, BORDER = 0.08, 5.0
SIZE = int(round((20 + 2 * BORDER) / MMPP))        # a small 20 x 20 mm plate -> 375 px grid


def _H(datum: str, size_px: int = SIZE) -> list[list[float]]:
    """plate mm -> grid px for a grid whose top-left is the plate's `datum` corner (as post writes)."""
    s, b, e = 1 / MMPP, BORDER / MMPP, size_px - BORDER / MMPP
    return {"TL": [[s, 0, b], [0, s, b], [0, 0, 1]],
            "BL": [[0, s, b], [-s, 0, e], [0, 0, 1]],
            "TR": [[0, -s, e], [s, 0, b], [0, 0, 1]],
            "BR": [[-s, 0, e], [0, -s, e], [0, 0, 1]]}[datum]


def _meta(datum: str = "TL", defects=None, **kw) -> dict:
    m = {"schema": "ayas.twin.grid/1", "plate_size_mm": [20.0, 20.0],
         "grid": {"mm_per_px": MMPP, "border_mm": BORDER, "size_px": [SIZE, SIZE], "plate_mm": [20.0, 20.0],
                  "datum_corner": datum},
         "H_grid_plate": _H(datum), "defects": defects or [], "features": [], "distractors": []}
    m.update(kw)
    return m


def _plate_to_grid(meta: dict, uv) -> np.ndarray:
    H = np.asarray(meta["H_grid_plate"], float)
    p = np.c_[np.asarray(uv, float), np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:]


def _plate_image() -> np.ndarray:
    """A canonical (TL) test image with an asymmetric pattern."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[70:120, 80:200] = (200, 50, 20)
    img[250:260, 100:300] = (10, 220, 30)
    return img


def _to_datum(img: np.ndarray, datum: str) -> np.ndarray:
    """What post would write for the same plate seen with another corner first."""
    M_ = G.canonical_affine(_meta(datum))            # grid -> canonical
    inv = cv2.invertAffineTransform(M_)
    return cv2.warpAffine(img, np.round(inv), (SIZE, SIZE), flags=cv2.INTER_NEAREST)


# ---- masks ------------------------------------------------------------------------------------

def test_sample_masks_targets_not_in_context():
    rng = np.random.default_rng(0)
    cfg = M.MaskCfg()
    for _ in range(50):
        tgts, ctx = M.sample_masks(rng, cfg)
        assert len(tgts) == 4
        union = np.concatenate(tgts)
        assert not np.intersect1d(union, ctx).size
        assert len(ctx) >= cfg.min_keep
        for t in tgts:
            assert 0.1 * 196 <= len(t) <= 0.25 * 196


def test_collate_equal_lengths_and_visible_from_context():
    rng = np.random.default_rng(1)
    out = M.collate_masks(rng, M.MaskCfg(n_visible=8), 6)
    B, Lt = out["tgt"].shape
    assert B == 6 and out["ctx"].shape[0] == 6 and out["vis"].shape == (6, Lt)
    assert out["vis"].sum(1).tolist() == [8] * 6
    for c, t, v in zip(out["ctx"], out["tgt"], out["vis"]):
        assert set(t[v]) <= set(c)                    # visible targets come from the context
        assert not set(t[~v]) & set(c)                # hidden targets never do
        assert (np.diff(c) > 0).all()


def test_checker_groups_partition():
    gs = M.checker_groups((14, 14), 2)
    allidx = np.sort(np.concatenate(gs))
    assert len(gs) == 4 and (allidx == np.arange(196)).all()


# ---- grids ------------------------------------------------------------------------------------

@pytest.mark.parametrize("datum", ["TL", "BL", "TR", "BR"])
def test_canonical_affine_maps_plate_mm_identically(datum):
    meta = _meta(datum)
    M_ = G.canonical_affine(meta)
    uv = np.array([[0, 0], [20, 0], [3.2, 17.5]])
    canon = G.warp_points(_plate_to_grid(meta, uv), M_)
    assert np.allclose(canon, uv / MMPP + BORDER / MMPP, atol=1e-6)


@pytest.mark.parametrize("datum", ["BL", "TR", "BR"])
def test_to_canonical_realigns_rotated_rows(datum):
    img = _plate_image()
    rotated = _to_datum(img, datum)
    assert not np.array_equal(rotated, img)
    back = G.to_canonical(rotated, _meta(datum))
    assert np.array_equal(back, img)


def test_defect_mask_follows_rotation():
    d = {"id": "D1", "cls": "pit", "label": True, "visible": True}
    poly_mm = np.array([[5, 5], [8, 5], [8, 7], [5, 7]], float)
    masks = []
    for datum in ("TL", "BL"):
        meta = _meta(datum)
        dd = dict(d, polys_grid=[_plate_to_grid(meta, poly_mm).tolist()])
        meta["defects"] = [dd]
        masks.append(G.defect_masks(meta)[0]["mask"])
    assert masks[0].sum() > 300
    assert (masks[0] ^ masks[1]).sum() <= 0.05 * masks[0].sum()      # the same canonical pixels
    ys, xs = np.nonzero(masks[0])
    assert abs(xs.mean() - (6.5 / MMPP + BORDER / MMPP)) < 2 and abs(ys.mean() - (6 / MMPP + BORDER / MMPP)) < 2


def test_plate_rect_and_margin():
    meta = _meta("TL")
    x0, y0, x1, y1 = G.plate_rect(meta)
    assert (x0, y0, x1, y1) == (63, 63, 313, 313)
    pm = G.plate_mask(meta)
    assert pm.sum() == 250 * 250
    d = {"id": "D1", "cls": "scratch", "label": True, "polys_grid": [[[150, 150], [151, 150], [151, 200], [150, 200]]]}
    meta["defects"] = [d]
    m0, m2 = G.any_defect_mask(meta, 0), G.any_defect_mask(meta, 2.0)
    assert m2.sum() > m0.sum() * 5


def _write_grid(root: Path, stem: str, cond: str, dev: str, seed: int, gate: str, k: int = 0, datum: str = "TL",
                split: str = "jepa_train") -> None:
    d = root / "grids" / "burst" / split
    d.mkdir(parents=True, exist_ok=True)
    img = _to_datum(_plate_image(), datum) if datum != "TL" else _plate_image()
    cv2.imwrite(str(d / f"{stem}.jpg"), img)
    meta = _meta(datum, stem=stem, arm="burst", split=split, k=k, condition=cond, device_mode=dev, gate=gate,
                 plate=f"s{seed:06d}", grid_image=f"{stem}.jpg", part={"material": "aluminium"},
                 matrix={"plate": f"s{seed:06d}", "plate_seed": seed, "condition": cond, "device_mode": dev,
                         "row": cond.split("_")[0] + dev[0]})
    (d / f"{stem}.json").write_text(json.dumps(meta))


def test_index_groups_and_test_plate_refusal(tmp_path):
    for cond in ("C0_dark", "C3_window", "C6_sun"):
        tag = cond.split("_")[0]
        _write_grid(tmp_path, f"s100000_{tag}h_v0", cond, "hood", 100000, "PASS")
        _write_grid(tmp_path, f"s100000_{tag}o_v0", cond, "open", 100000, "RETAKE" if tag == "C6" else "PASS",
                    datum="BL")
    recs = G.index_grids([tmp_path], ["burst"])
    assert len(recs) == 6
    gs = G.groups(recs)
    assert set(gs) == {("s100000", "hood", 0, "burst"), ("s100000", "open", 0, "burst")}
    assert sum(r.accepted for r in recs) == 5
    G.refuse_test_plates(recs)
    _write_grid(tmp_path, "s900001_C0h_v0", "C0_dark", "hood", 900001, "PASS", split="test")
    with pytest.raises(ValueError):
        G.refuse_test_plates(G.index_grids([tmp_path], ["burst"]))


def test_pairs_seed_rules():
    P.check_train_seed(100000)
    P.check_train_seed(899999)
    for bad in (99999, 900000, 900011):
        with pytest.raises(ValueError):
            P.check_train_seed(bad)
    cfg = P.jepa_config({"post": {"grid_splits": ["test"], "tile_splits": ["train"]}}, ["burst"], 2)
    assert cfg["post"]["grid_splits"] == [P.SPLIT] and cfg["post"]["tile_splits"] == []
    assert cfg["post"]["grid_format"] == "jpg" and cfg["post"]["eval_arms"] == ["burst"]


def test_pair_plans_identical_plate_across_rows(tmp_path):
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
    man = P.write_pair_plans(cfg, 1, tmp_path, 100000, ["C0_dark", "C6_sun"], ("open", "hood"))
    plate = man["plates"][0]
    assert [r["tag"] for r in plate["rows"]] == ["C0o", "C6o", "C0h", "C6h"]
    plans = [json.loads((tmp_path / r["plan"]).read_text()) for r in plate["rows"]]
    assert all(p["split"] == P.SPLIT and p["seed"] == 100000 for p in plans)
    for k in ("plate", "defects", "distractors"):
        assert len({json.dumps(p.get(k), sort_keys=True) for p in plans}) == 1
    with pytest.raises(ValueError):
        P.write_pair_plans(cfg, 1, tmp_path / "bad", 900000, ["C0_dark"])
    man10 = P.write_pair_plans(cfg, 1, tmp_path / "ten", 100000, None, ("open", "hood"),
                               ["C0_dark", "C4_side_lamp", "C6_sun"])
    tags = [r["tag"] for r in man10["plates"][0]["rows"]]
    assert len(tags) == 10 and tags[-3:] == ["C0h", "C4h", "C6h"] and "C3h" not in tags


# ---- metrics ----------------------------------------------------------------------------------

def test_auroc_ap_basic():
    assert MX.auroc([2, 3], [0, 1]) == 1.0
    assert MX.auroc([0, 1], [2, 3]) == 0.0
    assert MX.auroc([1, 1], [1, 1]) == 0.5
    assert np.isnan(MX.auroc([], [1]))
    assert MX.average_precision([3], [1, 2]) == 1.0
    assert abs(MX.average_precision([1], [2, 3]) - 1 / 3) < 1e-9
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 500), rng.normal(0, 1, 800)
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(500), np.zeros(800)]
    assert abs(MX.auroc(pos, neg) - roc_auc_score(y, np.r_[pos, neg])) < 1e-9


def test_normal_cells_and_recall_budget():
    heat = np.zeros((128, 128), np.float32)
    heat[10, 10] = 5.0
    plate = np.ones_like(heat, bool)
    excl = np.zeros_like(plate)
    excl[64:, 64:] = True
    cells = MX.normal_cells(heat, plate, excl, 64)
    assert len(cells) == 3 and cells.max() == 5.0
    rec, thr = MX.recall_at_budget(np.array([6.0, 1.0]), np.r_[np.zeros(99), 5.0], 100, 0.5)
    assert rec == 0.5 and 0 <= thr <= 5


def test_invariance_stats_identity_and_noise():
    rng = np.random.default_rng(0)
    e0 = rng.normal(size=(3, 50, 16)).astype(np.float32)
    s = MX.invariance_stats(e0.copy(), e0, np.zeros(16, np.float32), rng)
    assert abs(s["cos_same"] - 1) < 1e-5 and s["retrieval_at1"] == 1.0 and s["gap"] > 0.9
    s2 = MX.invariance_stats(rng.normal(size=e0.shape).astype(np.float32), e0, np.zeros(16, np.float32), rng)
    assert s2["retrieval_at1"] < 0.2 and abs(s2["gap"]) < 0.1


# ---- torch parts (skip without torch / timm) -------------------------------------------------

def _torch_cfg() -> dict:
    return {"seed": 0,
            "data": {"arms": ["burst"], "crop": 224, "scales": [1, 2], "pair": {"cross_prob": 1.0, "anchor": "C0"},
                     "aug": {"photometric": {"p": 1.0}}, "cache_grids": 8},
            "model": {"arch": "vit_tiny", "init": "scratch", "pred_dim": 32, "pred_depth": 1, "pred_heads": 2},
            "mask": {"n_visible": 4}, "train": {"vis_weight": 0.5}}


def test_model_forward_masked_equals_full_and_ema():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from jepa.model import JEPA

    torch.manual_seed(0)
    m = JEPA(_torch_cfg()).eval()
    x = torch.randn(2, 3, 224, 224)
    full = m.encoder(x)
    assert full.shape == (2, 196, 192)
    keep = torch.arange(196)[None].expand(2, -1)
    assert torch.allclose(m.encoder(x, keep), full, atol=1e-5)
    sub = torch.tensor([[0, 5, 17], [3, 4, 100]])
    assert m.encoder(x, sub).shape == (2, 3, 192)
    before = [p.clone() for p in m.target_encoder.parameters()]
    for p in m.encoder.parameters():
        p.data.add_(1.0)
    m.ema_update(0.5)
    after = list(m.target_encoder.parameters())
    assert all(torch.allclose(a, b + 0.5) for a, b in zip(after, before))
    assert not any(p.requires_grad for p in m.target_encoder.parameters())


def test_cross_pairs_and_one_train_step(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from jepa.data import MaskCollator, PairCrops
    from jepa.model import JEPA

    for cond in ("C0_dark", "C3_window"):
        tag = cond.split("_")[0]
        _write_grid(tmp_path, f"s100000_{tag}h_v0", cond, "hood", 100000, "PASS")
        _write_grid(tmp_path, f"s100000_{tag}o_v0", cond, "open", 100000, "PASS", datum="BL")
    recs = G.index_grids([tmp_path], ["burst"])
    cfg = _torch_cfg()
    ds = PairCrops(recs, cfg, length=8, seed=0)
    items = [ds[i] for i in range(4)]
    assert all(it["cross"] for it in items)
    assert {it["dt"] for it in items} <= {"C3h->C0h", "C3o->C0o"}
    assert torch.equal(ds[2]["ctx"], ds[2]["ctx"])                  # deterministic per index
    batch = MaskCollator(M.MaskCfg.from_dict(cfg["mask"]), 0)(items)
    m = JEPA(cfg)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    out = m(batch["ctx"], batch["tgt"], batch["ctx_idx"], batch["tgt_idx"], batch["vis"], batch["cross"])
    assert torch.isfinite(out["loss"]) and float(out["loss_visible"]) > 0
    out["loss"].backward()
    opt.step()
    with pytest.raises(ValueError):                                  # test plates are refused
        _write_grid(tmp_path, "s900002_C0h_v0", "C0_dark", "hood", 900002, "PASS", split="test")
        PairCrops(G.index_grids([tmp_path], ["burst"]), cfg, length=4)


def test_coreset_and_bank():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from jepa.eval_anomaly import Bank, greedy_coreset

    g = torch.Generator().manual_seed(0)
    f = torch.randn(2000, 32, generator=g)
    sel = greedy_coreset(f, 100, 0)
    assert len(set(sel.tolist())) == 100
    b = Bank(f, 200, 0, holdout=300)
    z_in = b.z(torch.randn(100, 32, generator=g))
    z_out = b.z(torch.randn(100, 32, generator=g) + 6)
    assert float(z_out.mean()) > float(z_in.mean()) + 3


# ---- pack + artifacts (upload bundle and the download-verify rule) ------------------------------

def test_pack_reencodes_png_and_anonymises(tmp_path):
    import tarfile

    from jepa import pack

    src = tmp_path / "pairs"
    _write_grid(src, "s100000_C0h_v0", "C0_dark", "hood", 100000, "PASS")
    d = src / "grids" / "burst" / "jepa_train"
    png = d / "s100000_C3h_v0.png"
    cv2.imwrite(str(png), _plate_image())
    meta = json.loads((d / "s100000_C0h_v0.json").read_text())
    meta.update(stem="s100000_C3h_v0", grid_image=png.name, condition="C3_window")
    (d / "s100000_C3h_v0.json").write_text(json.dumps(meta))
    (src / "pairs_manifest.json").write_text("{}")
    res = pack.pack([src], tmp_path / "b.tgz", splits=["jepa_train"])
    assert res["grids"] == 2
    with tarfile.open(tmp_path / "b.tgz") as tar:
        names = tar.getnames()
        assert "pairs/grids/burst/jepa_train/s100000_C3h_v0.jpg" in names and not any(n.endswith(".png") for n in names)
        assert {m.uname for m in tar.getmembers()} == {"root"}
        tar.extractall(tmp_path / "x")
    recs = G.index_grids([tmp_path / "x" / "pairs"], ["burst"])
    assert len(recs) == 2 and all(r.image.suffix == ".jpg" for r in recs)


def test_artifacts_verify_flags_missing_and_corrupt(tmp_path):
    from jepa import artifacts

    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n')
    (tmp_path / "summary.json").write_text('{"steps_done": 1}')
    (tmp_path / "bad.json").write_text("{not json")
    res = artifacts.verify(tmp_path, ["encoder.pt", "summary.json"])
    assert not res["ok"]
    assert any("missing required file encoder.pt" in p for p in res["problems"])
    assert any("bad.json" in p for p in res["problems"])
    md = (tmp_path / "MANIFEST.md").read_text()
    assert "summary.json" in md and "PROBLEMS" in md
    (tmp_path / "bad.json").unlink()
    (tmp_path / "encoder.pt").write_bytes(b"\x00")
    res = artifacts.verify(tmp_path, ["summary.json"])
    pt = [r for r in res["files"] if r["file"] == "encoder.pt"][0]
    try:
        import torch  # noqa: F401
        assert pt["check"].startswith("FAILED") and not res["ok"]
    except ImportError:
        assert pt["check"].startswith("not checked") and res["ok"]


def test_knn_head_onnx_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from jepa.export_onnx import KnnHead, export

    g = torch.Generator().manual_seed(0)
    bank = torch.randn(300, 24, generator=g)
    head = KnnHead(bank, 1.0, 0.5, (14, 14)).eval()
    e = torch.randn(1, 196, 24, generator=g)
    ref = head(e).detach().numpy()
    d = torch.cdist(e[0], bank).min(1).values
    assert np.allclose(ref.ravel(), ((d - 1.0) / 0.5).numpy(), atol=1e-4)
    p = tmp_path / "head.onnx"
    export(head, p, e, "patch_score", in_name="patch_emb")
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    out = s.run(None, {"patch_emb": e.numpy()})[0]
    assert out.shape == (1, 14, 14) and np.allclose(out, ref, atol=1e-3)

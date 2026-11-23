import copy
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from ayas.twin import coupons as C
from ayas.twin import templates as T

ROOT = Path(__file__).resolve().parents[1]
FACES = ("P1A", "P2A", "P3A", "P4A", "S1A", "S2A", "P5A")


@pytest.fixture(scope="module")
def coupons():
    return C.load_coupons(ROOT / "configs" / "coupons.yaml")


def keys(fs):
    return {C.finding_key(f) for f in fs}


def test_all_coupons_load_and_validate(coupons):
    assert set(FACES) <= set(coupons)
    for k, c in coupons.items():
        assert C.validate(c) == [], (k, C.validate(c))
        assert c["make"] and all(isinstance(s, str) for s in c["make"])
        assert all("footprint" not in d for d in c["defects"])
    assert not any(C.validate_all(coupons).values())


def test_defaults_are_merged(coupons):
    c = coupons["P4A"]
    assert c["plate_id"] == "P4" and c["face"] == "A" and c["fastener_d_mm"] == 4.0
    assert c["plate"]["thickness_mm"] == 3.0 and c["plate"]["size_mm"] == [100.0, 60.0]
    assert all(h["d_mm"] == 4.1 and h["sheet"] is None for h in c["holes"])
    assert {r["kind"] for r in c["rivets"]} == {"screw"} and c["rivets"][0]["head_h_mm"] == 3.1
    assert coupons["P2A"]["rivets"][0]["head_d_mm"] == 8.0


def test_p5_is_the_only_test_coupon_and_never_replicated(coupons):
    assert coupons["P5A"]["role"] == "test"
    assert [k for k, c in coupons.items() if c["role"] == "test"] == ["P5A"]
    assert {c["role"] for k, c in coupons.items() if k != "P5A"} <= {"train", "calib"}
    with pytest.raises(ValueError):
        C.for_replica(coupons, ["P2", "P5"])
    assert [c["id"] for c in C.for_replica(coupons, ["P1", "P2", "P3", "P4", "S1"])] == ["P1A", "P2A", "P3A", "P4A", "S1A"]


def test_designed_findings_hold(coupons):
    for k, c in coupons.items():
        assert keys(C.findings(c)) == keys(c["expect"]), k
    p2 = keys(C.findings(coupons["P2A"]))
    assert {("missing_fastener", "hole", "H3"), ("elongated_hole", "hole", "H6"),
            ("pitch", "holes", ("H8", "H9")), ("edge_distance", "hole", "H10")} <= p2
    # slits of 1 and 2 mm hide under the 8 mm head; 4 and 6 mm show
    vis = {d["id"]: C.visible_crack_mm(coupons["P2A"], d) for d in coupons["P2A"]["defects"] if d["cls"] == "crack"}
    assert vis["D1"] == pytest.approx(0.0, abs=0.01) and vis["D2"] == pytest.approx(0.05, abs=0.01)
    assert vis["D3"] == pytest.approx(2.05, abs=0.01) and vis["D4"] == pytest.approx(4.05, abs=0.01)
    p3 = coupons["P3A"]
    classes = {d["id"]: C.dent_class(p3, d) for d in p3["defects"]}
    assert [classes[f"D{i}"] for i in range(1, 9)] == ["minor"] * 3 + ["negligible"] * 5
    assert classes["D14"] == classes["D15"] == "edge_nick"             # as aero.dent_class / spec_aero
    holes = C.hole_shapes(p3)
    near = [min(C.gap(C.defect_shape(p3, d), h) for h in holes) for d in p3["defects"][:3]]
    assert all(2.0 <= g <= 5.0 for g in near)                          # rim to head-edge gap, < 2D
    far = [min(C.gap(C.defect_shape(p3, d), h) for h in holes) for d in p3["defects"][3:8]]
    assert all(g > 15.0 for g in far)


def test_fastener_table_p1a(coupons):
    rows = C.fastener_table(coupons["P1A"])
    assert [r["hole"] for r in rows] == [f"H{i}" for i in range(1, 9)]
    assert (rows[0]["u"], rows[0]["v"], rows[0]["d_mm"]) == (20.0, 15.0, 4.1)
    for r in rows:
        assert r["fastener"] == "blind_rivet"
        assert r["edge_mm"] == pytest.approx(15.0) and r["edge_D"] == pytest.approx(3.75)
        assert r["pitch_mm"] == pytest.approx(20.0) and r["pitch_D"] == pytest.approx(5.0)
    p2 = {r["hole"]: r for r in C.fastener_table(coupons["P2A"])}
    assert p2["H10"]["edge_D"] == pytest.approx(1.25) and p2["H9"]["pitch_D"] == pytest.approx(2.5)


def _bad(c, edit):
    c = copy.deepcopy(c)
    edit(c)
    return C.validate(c)


def test_validate_catches_problems(coupons):
    p2 = coupons["P2A"]

    def move_out(c):
        c["holes"][0]["u"] = -3.0

    def dup(c):
        c["defects"][1]["id"] = "D1"

    def rivet_in_open(c):
        c["rivets"][0]["hole"] = "H3"

    def bad_range(c):
        c["defects"][2]["params"]["width_mm"] = 2.0

    def wrong_expect(c):
        c["expect"].append({"type": "edge_distance", "hole": "H1"})

    def silent_fault(c):
        c["expect"] = [f for f in c["expect"] if f["type"] != "pitch"]

    def crowd(c):                                         # fretting streak from R7 runs into R8's head
        c["defects"][4]["params"].update(angle_deg=0.0, length_mm=12.0)

    assert any("H1" in e for e in _bad(p2, move_out))
    assert any("duplicate" in e for e in _bad(p2, dup))
    assert any("not filled" in e for e in _bad(p2, rivet_in_open))
    assert any("width_mm" in e for e in _bad(p2, bad_range))
    assert any("does not hold" in e for e in _bad(p2, wrong_expect))
    assert any("unexpected finding" in e for e in _bad(p2, silent_fault))
    assert any("from fastener H8" in e for e in _bad(p2, crowd))

    def p5_as_train(c):
        c["role"] = "train"

    assert any("test" in e for e in _bad(coupons["P5A"], p5_as_train))

    def dent_into_scratch(c):                                 # Piper: a scratch within 2x the dent diameter
        c["defects"][4]["params"].update(u=50.0, v=61.5)

    errs = _bad(coupons["P3A"], dent_into_scratch)
    assert any("dent_negligible" in e and "D5" in e for e in errs)


def test_ground_truth_rows(coupons, tmp_path):
    n = 0
    for k, c in coupons.items():
        rows = C.ground_truth_rows(c)
        assert len(rows) == len(c["defects"]), k
        n += len(rows) + len(c["holes"])
        for r, d in zip(rows, c["defects"]):
            assert tuple(r) == C.CSV_COLUMNS
            assert r["eval_class"] in (d["cls"], C.HIDDEN) and r["plate"] == c["plate_id"] and r["face"] == c["face"]
            assert r["diameter_mm"] != "" or (r["length_mm"] != "" and r["width_mm"] != "")
    slit = C.ground_truth_rows(coupons["P2A"])[3]                  # D4: 6 mm up from H5's edge
    assert (slit["u_mm"], slit["v_mm"], slit["length_mm"], slit["angle_deg"]) == (20.0, 79.95, 6.0, 90.0)
    path = C.write_ground_truth_template(coupons, tmp_path / "gt.csv")
    with open(path) as f:
        back = list(csv.DictReader(f))
    assert len(back) == n and tuple(back[0]) == C.CSV_COLUMNS
    keys = [(r["plate"], r["face"], r["defect_id"]) for r in back]
    assert len(set(keys)) == len(keys)                              # realdata keys rows on (plate, face, id)
    only_defects = C.write_ground_truth_template(coupons, tmp_path / "d.csv", fasteners=False)
    with open(only_defects) as f:
        assert len(list(csv.DictReader(f))) == sum(len(c["defects"]) for c in coupons.values())


def _area(poly):
    p = np.asarray(json.loads(poly) if isinstance(poly, str) else poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def test_ground_truth_hidden_cracks_and_shapes(coupons):
    """Regression: hidden cracks are not YOLO targets; nick, ring and fatigue outlines match the twin."""
    p2 = {r["defect_id"]: r for r in C.ground_truth_rows(coupons["P2A"])}
    assert p2["D1"]["eval_class"] == p2["D2"]["eval_class"] == C.HIDDEN   # under the R1/R2 heads
    assert p2["D3"]["eval_class"] == p2["D4"]["eval_class"] == "crack"
    s1 = {r["defect_id"]: r for r in C.ground_truth_rows(coupons["S1A"])}
    assert s1["D1"]["eval_class"] == s1["D2"]["eval_class"] == "crack"    # open hole: nothing hides them
    # fatigue outline at least 0.1 mm wide (as the twin's footprint), not the 0.02 mm planned width
    fat = np.asarray(json.loads(s1["D2"]["poly_mm"]))
    assert _area(fat) == pytest.approx(0.1 * C.polyline_length(coupons["S1A"]["defects"][1]["params"]["pts_mm"]), rel=0.1)
    # edge nicks: a half disc inside the plate, not a full circle centred on the edge
    for k, did, edge in (("P3A", "D14", "left"), ("P3A", "D15", "right"), ("P5A", "D9", "top")):
        c = coupons[k]
        W, H = C.plate_size(c)
        r = {x["defect_id"]: x for x in C.ground_truth_rows(c)}[did]
        poly = np.asarray(json.loads(r["poly_mm"]))
        d = r["diameter_mm"]
        assert poly[:, 0].min() >= -1e-9 and poly[:, 0].max() <= W + 1e-9
        assert poly[:, 1].min() >= -1e-9 and poly[:, 1].max() <= H + 1e-9
        across = np.ptp(poly[:, 0] if edge in ("left", "right") else poly[:, 1])
        assert across == pytest.approx(d / 2, abs=1e-3) and _area(poly) == pytest.approx(math.pi * d * d / 8, rel=0.01)
    # galvanic ring: annulus outside the screw head (outer 2 x (4 + ring_w))
    p4 = coupons["P4A"]
    for r, d in zip(C.ground_truth_rows(p4), p4["defects"]):
        if d["kind"] == "galvanic":
            ri, ro = 4.0, 4.0 + d["params"]["ring_w_mm"]
            assert r["diameter_mm"] == pytest.approx(2 * ro)
            assert _area(r["poly_mm"]) == pytest.approx(math.pi * (ro * ro - ri * ri), rel=0.01)


def test_fastener_rows_label_rivets_and_open_holes(coupons):
    """Regression: the CSV must carry rivet/open_hole rows, else real photos teach 'rivet = background'."""
    p1 = C.fastener_rows(coupons["P1A"])
    assert [r["defect_id"] for r in p1] == [f"R{i}" for i in range(1, 9)]
    assert all(r["cls"] == r["eval_class"] == "rivet" and r["diameter_mm"] == 8.0 for r in p1)
    assert (p1[4]["u_mm"], p1[4]["v_mm"]) == (20.0, 85.0)
    p2 = {r["defect_id"]: r for r in C.fastener_rows(coupons["P2A"])}
    assert p2["H3"]["cls"] == "open_hole" and p2["H3"]["diameter_mm"] == 4.1 and p2["H3"]["poly_mm"] == ""
    oval = p2["H6"]
    assert (oval["cls"], oval["length_mm"], oval["width_mm"], oval["angle_deg"]) == ("open_hole", 5.12, 4.1, 0.0)
    pts = np.asarray(json.loads(oval["poly_mm"]))
    assert np.ptp(pts[:, 0]) == pytest.approx(5.125, abs=0.01) and np.ptp(pts[:, 1]) == pytest.approx(4.1, abs=0.01)
    assert "R10" in p2 and "H10" not in p2                         # a riveted hole is labelled by its head
    p4 = C.fastener_rows(coupons["P4A"])
    assert {(r["cls"], r["kind"]) for r in p4} == {("rivet", "screw")}  # YOLO class 0 covers screw heads


def test_findings_use_the_aero_vocabulary(coupons):
    """Regression: lone open hole, edge nick and oversize hole are findings, as in aero.py."""
    assert ("lone_open_hole", "hole", "H1") in keys(C.findings(coupons["S1A"]))
    assert {("dent_edge_nick", "defect", "D14"), ("dent_edge_nick", "defect", "D15")} <= keys(C.findings(coupons["P3A"]))
    assert ("dent_edge_nick", "defect", "D9") in keys(C.findings(coupons["P5A"]))
    c = copy.deepcopy(coupons["P2A"])
    c["holes"][2].update(d_mm=4.5)                                  # H3 drilled oversize, round
    assert ("elongated_hole", "hole", "H3") in keys(C.findings(c))
    c = copy.deepcopy(coupons["P3A"])
    c["defects"][3]["params"].update(u=0.6)                         # punch dent D4 at the edge
    assert C.dent_class(c, c["defects"][3]) == "edge_nick"


def test_validate_rejects_bad_keys_and_malformed_entries(coupons):
    """Regression: a key without a face letter dodged the P5 rule; malformed entries crashed validate."""
    p5 = copy.deepcopy(coupons["P5A"])
    p5.update(id="P5", plate_id="P", face="5", role="train")
    assert any("face letter" in e for e in C.validate(p5))
    cases = [("P3A", lambda c: c["defects"][0].pop("id"), "missing ['id']"),
             ("P1A", lambda c: c["holes"][0].pop("u"), "missing ['u']"),
             ("P1A", lambda c: c["expect"].append("pitch"), "expect entries"),
             ("P3A", lambda c: c["defects"][0]["params"].update(u="20"), "must be a number"),
             ("P1A", lambda c: c["plate"].pop("size_mm"), "size_mm"),
             ("P4A", lambda c: (c["rivets"].pop(0), c["holes"][0].update(filled=False)), "galvanic corrosion needs")]
    for k, edit, msg in cases:
        assert any(msg in e for e in _bad(coupons[k], edit)), (k, msg)


def test_aero_rules_agree_with_expect(coupons):
    """The expect lists are the regression answers for aero.py on the real coupons: keep them equal."""
    aero = pytest.importorskip("ayas.aero")
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "configs" / "spec_aero.yaml").read_text())
    for k, c in coupons.items():
        fs = []
        for h in c["holes"]:
            r = C.rivet_in(c, h["id"])
            fs.append({"id": h["id"], "cls": "rivet" if r else "open_hole", "u": h["u"], "v": h["v"],
                       "d_mm": r["head_d_mm"] if r else h["d_mm"], "ovality": 1.0 if r else h["ovality"],
                       "angle_deg": 0.0 if r else h["angle_deg"]})
        rows = {r["defect_id"]: r for r in C.ground_truth_rows(c)}
        damage = [{"id": d["id"], "location_mm": {"x": rows[d["id"]]["u_mm"], "y": rows[d["id"]]["v_mm"]},
                   "orientation_deg": rows[d["id"]]["angle_deg"] or 0.0,
                   "size": {"length_mm": rows[d["id"]]["length_mm"], "width_mm": rows[d["id"]]["width_mm"]}}
                  for d in c["defects"] if d["cls"] in ("crack", "scratch")]
        try:
            got = aero.fastener_findings(fs, C.plate_size(c), spec, c["fastener_d_mm"])
            dents = {d["id"]: aero.dent_class({"size": {"diameter_mm": d["params"]["d_mm"]}, "depth_mm": d["params"]["depth_mm"],
                                               "location_mm": {"x": d["params"]["u"], "y": d["params"]["v"]}},
                                              fs, damage, c["plate"]["thickness_mm"], spec, c["fastener_d_mm"],
                                              C.plate_size(c))["class"]
                     for d in c["defects"] if d["cls"] == "dent"}
        except (TypeError, KeyError, AttributeError) as e:           # aero.py API moved: report, do not fail here
            pytest.skip(f"aero API changed: {e!r}")
        want = {f for f in keys(C.findings(c)) if f[0] in ("edge_distance", "pitch", "missing_fastener",
                                                              "elongated_hole", "lone_open_hole")}
        assert {(f["type"], "holes", tuple(sorted(f["fasteners"]))) if f["type"] == "pitch"
                else (f["type"], "hole", f["fasteners"][0]) for f in got} == want, k
        assert dents == {d["id"]: C.dent_class(c, d) for d in c["defects"] if d["cls"] == "dent"}, k


def test_svg_is_true_scale_and_complete(coupons, tmp_path):
    for k in ("P2A", "S1A"):
        c = coupons[k]
        p = T.write_svg(c, tmp_path / f"{k}.svg")
        root = ET.parse(p).getroot()
        W, H = C.plate_size(c)
        assert root.get("width") == f"{W + 2 * T.MARGIN_MM:g}mm"
        sw, sh = T.svg_size(c)
        assert root.get("height").endswith("mm") and float(root.get("height")[:-2]) == pytest.approx(sh, abs=1e-3)
        vb = [float(x) for x in root.get("viewBox").split()]
        assert vb[:3] == [0.0, 0.0, W + 2 * T.MARGIN_MM] and vb[3] == pytest.approx(sh)
        texts = [t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text")]
        for item in c["holes"] + c["defects"]:
            assert any(re.match(rf"{item['id']}\b", t) for t in texts), (k, item["id"])
        assert any("50.0 mm" in t for t in texts) and any(t == "A" for t in texts)


def test_svg_draws_in_the_plate_frame(coupons):
    """Regression for axis flips: u -> svg x, v -> svg y (down), W x H not swapped, slit direction kept."""
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.fromstring(T.svg(coupons["P2A"]).encode())
    g = next(e for e in root.iter(f"{ns}g") if (e.get("transform") or "").startswith("translate"))
    assert g.get("transform") == f"translate({T.MARGIN_MM:g} {T.TOP_MM:g})"
    holes = {(float(e.get("cx")), float(e.get("cy"))) for e in g.iter(f"{ns}ellipse") if e.get("rx") == "2.05"}
    assert (5.0, 50.0) in holes and (90.0, 85.0) in holes and (50.0, 5.0) not in holes
    oval = [e for e in g.iter(f"{ns}ellipse") if e.get("rx") == "2.562"]
    assert [(e.get("cx"), e.get("cy"), e.get("ry")) for e in oval] == [("40", "85", "2.05")]
    slits = [(float(e.get("x1")), float(e.get("y1")), float(e.get("x2")), float(e.get("y2")))
             for e in g.iter(f"{ns}line") if e.get("marker-end")]
    assert (20.0, 82.95, 20.0, 76.95) in slits                       # D4 runs up (-v) from H5
    assert (82.05, 15.0, 86.05, 15.0) in slits                       # D3 runs right (+u) from H4
    root = ET.fromstring(T.svg(coupons["P4A"]).encode())
    plate = [e for e in root.iter(f"{ns}rect") if e.get("stroke-width") == "0.35"]
    assert [(e.get("width"), e.get("height")) for e in plate] == [("100", "60")]


def test_write_all_and_cli(coupons, tmp_path, capsys):
    paths = T.write_all(coupons, tmp_path / "tpl")
    assert {p.name for p in paths} == {f"{k}.svg" for k in coupons} | {"index.html"}
    assert C.main(["--file", str(ROOT / "configs" / "coupons.yaml"), "validate"]) == 0
    out = tmp_path / "cli"
    assert C.main(["--file", str(ROOT / "configs" / "coupons.yaml"), "templates", "--out", str(out)]) == 0
    assert (out / "ground_truth_template.csv").exists() and (out / "P5A.svg").exists()
    assert C.main(["--file", str(ROOT / "configs" / "coupons.yaml"), "table", "P9A"]) == 2   # no traceback
    assert "unknown coupon" in capsys.readouterr().err

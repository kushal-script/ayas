import sys
import types

import cv2
import numpy as np
import onnxruntime
import pytest

from ayas import npu
from ayas.tiles import CLASSES, Det

NC = len(CLASSES)
DENT, SCRATCH, CORROSION = (CLASSES.index(c) for c in ("dent", "scratch", "corrosion"))


def yolo_output(boxes, n=40, nc=NC):
    """(1, 4 + nc, n) raw YOLOv8 output: given (cx, cy, w, h, cls, score) columns, rest near zero."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    out[0, :4] = 10.0
    out[0, 4:] = 0.01
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


class FakeSession:
    def __init__(self, out, shape=(1, 3, 640, 640), names=CLASSES):
        self.out, self.shape, self.feeds, self.names = out, list(shape), [], names

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=self.shape)]

    def get_modelmeta(self):
        meta = {"names": str(dict(enumerate(self.names)))} if self.names else {}
        return types.SimpleNamespace(custom_metadata_map=meta)

    def run(self, names, feeds):
        self.feeds.append(feeds["images"])
        return [self.out]


@pytest.fixture()
def fake_onnx(monkeypatch):
    def make(out, shape=(1, 3, 640, 640), names=CLASSES, path="fake.onnx"):
        sess = FakeSession(out, shape, names)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda path, providers=None: sess)
        return npu.OnnxDetector(path, conf=0.25, iou=0.5), sess
    return make


def tile(rgb=(0, 0, 0)):
    t = np.zeros((640, 640, 3), np.uint8)
    t[:] = rgb
    return t


def test_onnx_decode_nms_and_clip(fake_onnx):
    out = yolo_output([
        (100, 100, 40, 40, 2, 0.90),     # class 2
        (104, 102, 40, 40, 2, 0.80),     # duplicate class 2: suppressed
        (100, 100, 40, 40, 3, 0.70),     # same place, other class: kept (class-wise NMS)
        (500, 500, 30, 30, 4, 0.20),     # below conf
        (630, 10, 40, 20, 0, 0.60),      # class 0 over the right edge: clipped
    ])
    det, sess = fake_onnx(out)
    assert det.names == list(CLASSES) and det.size == 640 and det.batch == 1
    (dets,) = det.detect([tile()])
    assert [(d.cls, round(d.conf, 2)) for d in dets] == [(2, 0.9), (3, 0.7), (0, 0.6)]
    assert (dets[0].x0, dets[0].y0, dets[0].x1, dets[0].y1) == pytest.approx((80, 80, 120, 120))
    assert dets[2].x1 == pytest.approx(640.0) and dets[2].x0 == pytest.approx(610.0)
    assert all(d.tile == "" and not d.cut for d in dets)


def test_onnx_best_class_per_box_and_layout(fake_onnx):
    out = yolo_output([(320, 320, 50, 50, 1, 0.3)])
    out[0, 4 + 5, 0] = 0.6                          # class 5 beats class 1 in the same column
    det, _ = fake_onnx(out.transpose(0, 2, 1))      # (B, N, 4 + nc) layout is accepted too
    (dets,) = det.detect([tile()])
    assert len(dets) == 1 and dets[0].cls == 5 and dets[0].conf == pytest.approx(0.6)


def test_onnx_preprocess_rgb_nchw(fake_onnx):
    det, sess = fake_onnx(yolo_output([]))
    res = det.detect([tile((255, 0, 128)), tile((0, 51, 0))])
    assert res == [[], []]
    assert len(sess.feeds) == 2                     # fixed batch 1: one run per tile
    x = sess.feeds[0]
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert x[0, :, 5, 7] == pytest.approx([1.0, 0.0, 128 / 255])
    assert sess.feeds[1][0, 1, 0, 0] == pytest.approx(0.2)


def test_onnx_names_from_model_json_when_no_metadata(fake_onnx, tmp_path):
    model = tmp_path / "v8n_burst.onnx"
    (tmp_path / "v8n_burst.json").write_text('{"classes": ["stain", "scratch"]}')
    det, _ = fake_onnx(yolo_output([(320, 320, 50, 50, 1, 0.9)], nc=2), names=None, path=str(model))
    assert det.names == ["stain", "scratch"] and npu.class_names(det) == ["stain", "scratch"]
    det2, _ = fake_onnx(yolo_output([], nc=NC).transpose(0, 2, 1), names=None, path=str(tmp_path / "x.onnx"))
    assert det2.names is None and npu.class_names(det2) == list(CLASSES)
    assert det2.raw([tile()]).shape == (1, 4 + NC, 40)     # unnamed: (B, N, 4 + nc) still recognised


def test_onnx_rejects_resized_or_bgr_float_tiles(fake_onnx):
    det, _ = fake_onnx(yolo_output([]))
    with pytest.raises(ValueError):
        det.detect([np.zeros((320, 320, 3), np.uint8)])
    with pytest.raises(ValueError):
        det.detect([np.zeros((640, 640, 3), np.float32)])


def test_batched_nms_keeps_classes_apart_with_negative_coords():
    boxes = np.array([[-10, 0, 50, 40], [41, 0, 101, 40], [-10, 0, 50, 40]], float)
    keep = npu.batched_nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 1, 0]), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_parse_nms_by_class_list_form():
    raw = [np.zeros((0, 5), np.float32) for _ in range(NC)]
    raw[2] = np.array([[0.1, 0.2, 0.3, 0.4, 0.9], [0.5, 0.5, 0.6, 0.6, 0.1]], np.float32)   # y0 x0 y1 x1 s
    raw[0] = np.array([[0.0, 0.9, 0.05, 1.2, 0.5]], np.float32)                             # x1 > 1: clipped
    raw[4] = np.zeros((0,), np.float32)
    dets = npu.parse_nms_by_class(raw, NC, conf=0.25, size=640)
    assert [(d.cls, d.conf) for d in dets] == [(2, pytest.approx(0.9)), (0, pytest.approx(0.5))]
    assert (dets[0].x0, dets[0].y0, dets[0].x1, dets[0].y1) == pytest.approx((128, 64, 256, 192))
    assert dets[1].x1 == pytest.approx(640.0)


def test_parse_nms_by_class_flat_form():
    flat = [0, 0, 2, 0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.5, 0.75, 0.75, 0.3, 0, 1, 0.0, 0.0, 0.5, 0.5, 0.8, 0]
    flat += [0.0] * 30                                                  # trailing padding
    dets = npu.parse_nms_by_class(np.array(flat, np.float32), NC, conf=0.25, size=640)
    assert [(d.cls, round(d.conf, 2)) for d in dets] == [(2, 0.9), (4, 0.8), (2, 0.3)]
    assert (dets[2].x0, dets[2].y0, dets[2].x1, dets[2].y1) == pytest.approx((320, 320, 480, 480))
    with pytest.raises(ValueError):
        npu.parse_nms_by_class([np.zeros((0, 5))] * 3, NC)
    with pytest.raises(ValueError):
        npu.parse_nms_by_class(np.array([5.0, 0.1, 0.2]), NC)


def fake_hailo(per_frame):
    """A stand-in hailo_platform module recording calls; output = per_frame(input buffer)."""
    log = {"batch": None, "runs": [], "closed": False}

    class Stream:
        def __init__(self, shape):
            self.shape = shape

        def set_format_type(self, t):
            self.fmt = t

    class BindStream:
        def set_buffer(self, buf):
            self.buf = buf

        def get_buffer(self):
            return self.out

    class Bindings:
        def __init__(self):
            self.i, self.o = BindStream(), BindStream()

        def input(self):
            return self.i

        def output(self):
            return self.o

    class Configured:
        def create_bindings(self):
            return Bindings()

        def run(self, bindings, timeout_ms):
            log["runs"].append((len(bindings), timeout_ms))
            for b in bindings:
                assert b.i.buf.shape == (640, 640, 3) and b.i.buf.dtype == np.uint8
                b.o.out = per_frame(b.i.buf)

    class Ctx:
        def __enter__(self):
            return Configured()

        def __exit__(self, *a):
            log["closed"] = True

    class InferModel:
        def set_batch_size(self, b):
            log["batch"] = b

        def input(self):
            return self._i

        def output(self):
            return self._o

        def configure(self):
            return Ctx()

    class VDevice:
        def __init__(self, params):
            pass

        @staticmethod
        def create_params():
            return types.SimpleNamespace(scheduling_algorithm=None)

        def create_infer_model(self, path):
            m = InferModel()
            m._i, m._o = Stream((640, 640, 3)), Stream((NC, 501))
            return m

        def release(self):
            pass

    mod = types.ModuleType("hailo_platform")
    mod.VDevice = VDevice
    mod.FormatType = types.SimpleNamespace(UINT8="u8", FLOAT32="f32")
    mod.HailoSchedulingAlgorithm = types.SimpleNamespace(ROUND_ROBIN="rr")
    return mod, log


def test_hailo_detector_batches_pads_and_parses(monkeypatch):
    def per_frame(buf):
        raw = [np.zeros((0, 5), np.float32) for _ in range(NC)]
        v = float(buf[0, 0, 0]) / 255           # encode the tile's value in the box
        if v > 0:
            raw[3] = np.array([[0.1, v / 2, 0.2, v, 0.8]], np.float32)
        return raw

    mod, log = fake_hailo(per_frame)
    monkeypatch.setitem(sys.modules, "hailo_platform", mod)
    with npu.HailoDetector("model.hef", conf=0.25, batch_size=9) as det:
        res = det.detect([tile((255, 0, 0)), tile((128, 0, 0)), tile()])
    assert det.n_classes == NC and det.names == list(CLASSES)          # no model JSON: tiles.CLASSES
    assert log["batch"] == 9 and log["runs"] == [(9, 10000)] and log["closed"]
    assert len(res) == 3 and res[2] == []
    assert res[0][0].cls == 3 and res[0][0].x1 == pytest.approx(640) and res[0][0].y0 == pytest.approx(64)
    assert res[1][0].x1 == pytest.approx(128 / 255 * 640)


def test_hailo_detector_classes_from_model_json_or_explicit(monkeypatch, tmp_path):
    mod, _ = fake_hailo(lambda buf: [np.zeros((0, 5), np.float32)] * 2)
    monkeypatch.setitem(sys.modules, "hailo_platform", mod)
    hef = tmp_path / "ayas_v8n.hef"
    (tmp_path / "ayas_v8n.json").write_text('{"classes": ["pit", "stain"]}')
    det = npu.HailoDetector(hef)
    assert det.n_classes == 2 and det.names == ["pit", "stain"]
    stock = npu.HailoDetector(tmp_path / "yolov8n_stock.hef", n_classes=80)      # COCO check on the Pi
    assert stock.n_classes == 80 and stock.names is None
    named = npu.HailoDetector(tmp_path / "other.hef", names=list(CLASSES))
    assert named.names == list(CLASSES) and named.n_classes == NC


def test_hailo_detector_needs_hailo_platform(monkeypatch):
    monkeypatch.setitem(sys.modules, "hailo_platform", None)
    with pytest.raises(ImportError, match="hailo-h10-all"):
        npu.HailoDetector("model.hef")


def test_module_import_is_light():
    assert "hailo_platform" not in sys.modules or sys.modules["hailo_platform"] is None
    assert "ultralytics" not in sys.modules and "torch" not in sys.modules


class ColourDetector:
    """Finds solid-colour rectangles in each tile: red = dent, green = scratch, blue = corrosion."""

    COLOURS = {DENT: (255, 0, 0), SCRATCH: (0, 255, 0), CORROSION: (0, 0, 255)}

    def __init__(self, names=None):
        self.calls = 0
        if names is not None:                       # a model whose class order differs from tiles.CLASSES
            self.names = list(names)
            self.COLOURS = {self.names.index(CLASSES[c]): rgb for c, rgb in self.COLOURS.items()}

    def detect(self, tiles):
        self.calls += 1
        out = []
        for t in tiles:
            dets = []
            for c, rgb in self.COLOURS.items():
                m = np.all(t == rgb, axis=2).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(m)
                for x, y, w, h, _ in stats[1:]:
                    dets.append(Det(c, 0.9, float(x), float(y), float(x + w), float(y + h)))
            out.append(dets)
        return out


def test_detect_grid_merges_overlap_duplicates_and_long_scratch():
    grid = np.full((1375, 1375, 3), 128, np.uint8)
    grid[100:150, 400:450] = (255, 0, 0)        # dent inside the overlap of tiles r0c0 and r0c1
    grid[800:806, 100:1300] = (0, 255, 0)       # 96 mm scratch across both vertical seams
    grid[1200:1260, 1200:1260] = (0, 0, 255)    # corrosion in the last tile only
    det = ColourDetector()
    dets = npu.detect_grid(grid, det)
    assert det.calls == 1                        # all 9 tiles in one call (one Hailo batch)
    by = {c: [d for d in dets if d.cls == c] for c in (DENT, SCRATCH, CORROSION)}
    assert len(by[DENT]) == 1 and (by[DENT][0].x0, by[DENT][0].y0, by[DENT][0].x1, by[DENT][0].y1) == (400, 100, 450, 150)
    assert len(by[SCRATCH]) == 1
    s = by[SCRATCH][0]
    assert (s.x0, s.y0, s.x1, s.y1) == (100, 800, 1300, 806)
    assert len(by[CORROSION]) == 1 and by[CORROSION][0].tile == "r2c2"


def test_remap_classes_by_name():
    names = ["stain", "dent", "rivet"]                      # a model with its own order (+ an unknown class)
    dets = [Det(0, 0.9, 0, 0, 1, 1), Det(1, 0.8, 0, 0, 1, 1), Det(2, 0.7, 0, 0, 1, 1)]
    out = npu.remap_classes(dets, names)
    want = [CLASSES.index(n) for n in names[:2] if n in CLASSES] + ([CLASSES.index("rivet")] if "rivet" in CLASSES else [])
    assert [d.cls for d in out] == want
    assert npu.remap_classes(dets, list(CLASSES)) is dets


def test_detect_grid_remaps_model_class_order_before_the_seam_merge():
    grid = np.full((1375, 1375, 3), 128, np.uint8)
    grid[800:806, 100:1300] = (0, 255, 0)       # one long scratch across both seams
    grid[100:150, 400:450] = (255, 0, 0)
    order = list(reversed(CLASSES))             # model ids differ from tiles.CLASSES ids
    dets = npu.detect_grid(grid, ColourDetector(order))
    assert sorted(d.cls for d in dets) == sorted([SCRATCH, DENT])     # scratch merged once, ids in tiles.CLASSES
    s = next(d for d in dets if d.cls == SCRATCH)
    assert (s.x0, s.x1) == (100, 1300)


def test_detect_grid_scratch_inside_overlap_comes_back_whole():
    """Was an xfail (a truncated seam piece could suppress the whole box in tiles.nms); fixed by
    tiles.join_seam_pieces, so this now guards the fix."""
    grid = np.full((1375, 1375, 3), 128, np.uint8)
    grid[800:806, 500:800] = (0, 255, 0)        # 24 mm scratch: whole in tile c1, cut in c0 and c2
    dets = npu.detect_grid(grid, ColourDetector())
    assert [(d.x0, d.x1) for d in dets] == [(500, 800)]


def test_detect_grid_small_grid_is_padded_and_clipped():
    grid = np.full((500, 600, 3), 128, np.uint8)
    grid[480:500, 580:600] = (255, 0, 0)
    dets = npu.detect_grid(grid, ColourDetector())
    assert len(dets) == 1 and (dets[0].x1, dets[0].y1) == (600, 500)

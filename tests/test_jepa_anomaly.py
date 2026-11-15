"""trinetra.anomaly: the inference-side heat map helpers (numpy + cv2; no torch)."""

import numpy as np
import pytest

from trinetra import anomaly as A


def test_window_starts_cover_and_end_flush():
    assert A.window_starts(100, 224, 224) == [0]
    s = A.window_starts(1250, 448, 448)
    assert s == [0, 448, 802]
    s1 = A.window_starts(1250, 224, 224)
    assert s1[-1] + 224 == 1250 and len(s1) == 6


def test_windows_inside_rect():
    ws = A.windows((63, 63, 1313, 1313), 448)
    assert len(ws) == 9
    assert all(63 <= x and x + 448 <= 1313 and 63 <= y and y + 448 <= 1313 for x, y in ws)


def test_crop_pads_outside():
    img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    c = A.crop(img, -1, -1, 3)
    assert c[0, 0] == 0 and c[1, 1] == img[0, 0] and c[2, 2] == img[1, 1]


def test_preprocess_shape_and_norm():
    x = A.preprocess(np.full((448, 448, 3), 255, np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0, :, 0, 0], (1 - A.MEAN) / A.STD, atol=1e-5)


def test_assemble_averages_overlaps_and_masks():
    s1 = np.zeros((14, 14), np.float32)
    s2 = np.full((14, 14), 2.0, np.float32)
    mask = np.ones((300, 300), bool)
    mask[:, 290:] = False
    heat = A.assemble((300, 300), [(0, 0, s1), (76, 0, s2)], 224, mask)
    assert heat[10, 10] == 0 and heat[10, 250] == 2.0 and heat[10, 150] == 1.0
    assert heat[10, 295] == 0 and heat[250, 10] == 0          # masked / not covered


def test_summarize_and_unexplained():
    heat = np.zeros((400, 400), np.float32)
    heat[100:150, 100:130] = 5.0            # 50 x 30 px = 4 x 2.4 mm
    heat[300:302, 300:302] = 9.0            # a 0.03 mm2 speck: below min_area
    s = A.summarize(heat, 0.08, 3.0, origin_px=(62.5, 62.5), min_area_mm2=0.5, model="jepa_knn")
    assert s["schema"] == A.SCHEMA and s["n_regions"] == 1 and s["score_max"] == 9.0
    r = s["regions"][0]
    assert r["peak"] == 5.0 and r["box_px"] == [100, 100, 130, 150]
    assert abs(r["x_mm"] - (100 - 62.5) * 0.08) < 0.06 and abs(r["area_mm2"] - 9.6) < 0.1
    assert A.unexplained(s, [[90, 90, 140, 160]]) == []
    assert len(A.unexplained(s, [[0, 0, 20, 20]])) == 1


def test_onnx_scorer_optional(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    # a stand-in head: mean over channels, 16 x 16 average pool -> (1, 14, 14)
    inp = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])
    out = helper.make_tensor_value_info("patch_score", TensorProto.FLOAT, [1, 14, 14])
    nodes = [helper.make_node("ReduceMean", ["image"], ["m"], axes=[1], keepdims=1),
             helper.make_node("AveragePool", ["m"], ["p"], kernel_shape=[16, 16], strides=[16, 16]),
             helper.make_node("Squeeze", ["p", "ax"], ["patch_score"])]
    ax = helper.make_tensor("ax", TensorProto.INT64, [1], [1])
    g = helper.make_graph(nodes, "head", [inp], [out], initializer=[ax])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    p = tmp_path / "head.onnx"
    onnx.save(model, str(p))
    (tmp_path / "head.json").write_text('{"img_size": 224, "scale": 2, "threshold": 3.0}')
    sc = A.OnnxPatchScorer(p, threads=1)
    grid = np.zeros((600, 600, 3), np.uint8)
    heat = sc.score_grid(grid, (50, 50, 550, 550))
    assert heat.shape == (600, 600) and np.isfinite(heat).all()
    assert ort is not None

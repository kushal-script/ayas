from pathlib import Path

import numpy as np
import pytest
import yaml

from ayas import tiles as T
from ayas.twin import frames as F

ROOT = Path(__file__).resolve().parents[1]
SCRATCH, DENT = T.CLASSES.index("scratch"), T.CLASSES.index("dent")


def test_straight_down_view_is_upright_and_at_nominal_scale():
    K = F.nominal_K()
    R, t = F.camera_pose((50, 50), 200.0)
    px = F.project(K, R, t, F.plate_to_world([(50, 50), (60, 50), (50, 60)]))
    assert np.allclose(px[0], K[:2, 2], atol=1e-6)            # target on the principal point
    assert px[1, 0] > px[0, 0] and abs(px[1, 1] - px[0, 1]) < 1e-6   # +u is image right
    assert px[2, 1] > px[0, 1] and abs(px[2, 0] - px[0, 0]) < 1e-6   # +v is image down
    mm_per_px = 10.0 / (px[1, 0] - px[0, 0])
    assert mm_per_px == pytest.approx(200 * 1.55e-3 / 4.74, rel=1e-6)   # about 0.065 mm/px


@pytest.mark.parametrize("tilt", [(0, 0, 0), (10, -7, 33), (-15, 12, 181)])
def test_homography_matches_projection(tilt):
    K = F.nominal_K()
    R, t = F.camera_pose((40, 55), 210.0, *tilt)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    uv = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [37.5, 81.2]])
    H = F.homography_plate_to_image(K, R, t)
    assert np.allclose(F.apply_h(H, uv), F.project(K, R, t, F.plate_to_world(uv)), atol=1e-6)
    # the optical axis still hits the target and the distance is kept
    C = F.camera_centre(R, t)
    assert np.linalg.norm(C - F.plate_to_world([(40, 55)])[0]) * 1000 == pytest.approx(210.0)
    assert np.allclose(F.project(K, R, t, F.plate_to_world([(40, 55)]))[0], K[:2, 2], atol=1e-6)


def test_world_plate_round_trip():
    uvw = np.array([[12.0, 34.0, -1.5]])
    assert np.allclose(F.world_to_plate(F.plate_to_world(uvw[:, :2], uvw[:, 2])), uvw)


def test_blender_intrinsics_centred_and_roi():
    K = F.nominal_K()
    b = F.blender_intrinsics(K, 4056, 3040)
    assert b["lens_mm"] == pytest.approx(4.74 * 36 / (4056 * 1.55e-3), rel=1e-6)
    assert b["shift_x"] == pytest.approx(0.0) and b["shift_y"] == pytest.approx(0.0)
    Kr = F.roi_K(K, 1000, 500)
    br = F.blender_intrinsics(Kr, 1600, 1600)
    assert br["shift_x"] < 0 or Kr[0, 2] < 799.5   # principal point left of the window centre


def test_blender_cam2world_axes():
    R, t = F.camera_pose((50, 50), 200.0)
    M = F.blender_cam2world(R, t)
    # Blender camera looks along its local -z: straight down is world -Z
    assert np.allclose(M[:3, :3] @ np.array([0, 0, -1.0]), [0, 0, -1])
    assert np.allclose(M[:3, 3], [0.05, -0.05, 0.2])


def test_order_corners():
    pts = np.array([[900, 120], [100, 110], [880, 700], [120, 690]], float)
    o = F.order_corners(pts)
    assert np.allclose(o, [[100, 110], [900, 120], [880, 700], [120, 690]])


def test_tile_starts_cover_grid():
    assert T.tile_starts(1375) == [0, 368, 735]
    assert T.tile_starts(640) == [0] and T.tile_starts(500) == [0]
    for n in (641, 900, 1375, 2000, 3100):
        s = T.tile_starts(n)
        assert s[0] == 0 and s[-1] + T.TILE == n
        assert all(b - a <= T.TILE - T.MIN_OVERLAP for a, b in zip(s, s[1:]))


def test_cut_pads_small_grids():
    img = np.ones((500, 700, 3), np.uint8)
    tiles = T.tile_grid(700, 500)
    crops = T.cut(img, tiles)
    assert all(c.shape == (640, 640, 3) for c in crops) and len(crops) == 2


def test_steel_classes_match_the_design_and_config():
    assert T.CLASSES == ("scratch", "dent", "pit", "corrosion", "crack", "stain")   # DESIGN_STEEL A, ids 0-5
    assert T.ELONGATED == ("scratch", "crack") and set(T.ELONGATED) <= set(T.CLASSES)
    cfg = yaml.safe_load((ROOT / "configs" / "twin.yaml").read_text())
    assert tuple(cfg["classes"]) == T.CLASSES
    assert T.Det(T.CLASSES.index("stain"), 0.5, 0, 0, 1, 1).name == "stain" and T.Det(9, 0.5, 0, 0, 1, 1).name == "9"


def test_merge_removes_overlap_duplicates_and_joins_seam_scratch():
    tiles = T.tile_grid(1375, 1375)
    t00, t01 = tiles[0], tiles[1]
    dent = T.Det(DENT, 0.9, 400, 100, 420, 120)                   # dent in the overlap: seen by both tiles
    dent_b = T.Det(DENT, 0.8, 400 - t01.x0, 100, 420 - t01.x0, 120)
    s_left = T.Det(SCRATCH, 0.7, 500, 300, 639.5, 310)            # scratch cut by tile 0's right border
    s_right = T.Det(SCRATCH, 0.6, 639 - t01.x0, 302, 900 - t01.x0, 312)
    per = [[] for _ in tiles]
    per[0] = [dent, s_left]
    per[1] = [dent_b, s_right]
    out = T.merge_tiles(per, tiles, 1375, 1375)
    dents = [d for d in out if d.cls == DENT]
    scratches = [d for d in out if d.cls == SCRATCH]
    assert len(dents) == 1 and dents[0].conf == 0.9
    assert len(scratches) == 1 and scratches[0].x0 == 500 and scratches[0].x1 == 900


def test_merge_joins_a_seam_crack_but_not_seam_pits():
    """crack is elongated (joined across a seam); pits cut by the seam are not merged."""
    tiles = T.tile_grid(1375, 1375)
    t01 = tiles[1]
    crack, pit = T.CLASSES.index("crack"), T.CLASSES.index("pit")
    per = [[] for _ in tiles]
    per[0] = [T.Det(crack, 0.7, 600, 400, 639.5, 404), T.Det(pit, 0.6, 630, 500, 639.8, 509)]
    per[1] = [T.Det(crack, 0.6, 639.2 - t01.x0, 401, 700 - t01.x0, 405), T.Det(pit, 0.5, 642 - t01.x0, 500, 652 - t01.x0, 509)]
    out = T.merge_tiles(per, tiles, 1375, 1375)
    cracks = [d for d in out if d.cls == crack]
    assert len(cracks) == 1 and (cracks[0].x0, cracks[0].x1) == (600, 700)
    assert len([d for d in out if d.cls == pit]) == 2


def _seen_by_tiles(tiles, objects):
    """What a perfect detector reports per tile: each object's grid box clipped to every tile
    that sees part of it (tile-local), confidence from the tile index so either piece may win."""
    per = [[] for _ in tiles]
    for i, t in enumerate(tiles):
        for cls, box, conf in objects:
            local = T.clip_box(box, t)
            if local is not None:
                per[i].append(T.Det(cls, conf[i % len(conf)] if isinstance(conf, tuple) else conf, *local))
    return per


@pytest.mark.parametrize("conf", [(0.9, 0.6, 0.7), (0.6, 0.9, 0.7)])
@pytest.mark.parametrize("cls,box", [
    ("scratch", (300, 500, 700, 512)),          # 32 mm over the r0c0 | r0c1 seam: was cut to 27 mm by NMS
    ("crack", (450, 500, 1050, 506)),           # 48 mm over two seams: came back in two pieces
    ("corrosion", (450, 450, 1050, 900)),       # 48 x 36 mm rust patch over 4 tiles: came back in two
    ("stain", (200, 300, 900, 520)),            # 56 mm oil stain
])
def test_merge_rebuilds_defects_that_cross_seams(cls, box, conf):
    """Regression: overlapping tiles both see a seam-crossing defect, partly. NMS used to keep
    one truncated piece (IoU/IoM against the other tile's piece) before the elongated join, so
    the merged box was short (wrong length -> wrong severity) or split in two."""
    tiles = T.tile_grid(1375, 1375)
    c = T.CLASSES.index(cls)
    out = T.merge_tiles(_seen_by_tiles(tiles, [(c, box, conf)]), tiles, 1375, 1375)
    assert [(d.cls, d.x0, d.y0, d.x1, d.y1) for d in out] == [(c, *map(float, box))]
    assert out[0].conf == max(conf) and not out[0].cut


def test_merge_keeps_neighbours_apart_across_a_seam():
    """Two pits either side of the r0c0 | r0c1 seam (the first one cut by it), a dent in the
    overlap seen whole by both tiles, and two parallel scratches crossing the seam 2 mm apart."""
    tiles = T.tile_grid(1375, 1375)
    pit = T.CLASSES.index("pit")
    objs = [(pit, (628, 500, 645, 517), 0.8), (pit, (648, 500, 662, 514), 0.7), (DENT, (400, 100, 420, 120), 0.9),
            (SCRATCH, (500, 800, 900, 806), 0.6), (SCRATCH, (500, 831, 900, 837), 0.5)]
    out = T.merge_tiles(_seen_by_tiles(tiles, objs), tiles, 1375, 1375)
    got = sorted((d.cls, d.x0, d.y0, d.x1, d.y1) for d in out)
    assert got == sorted((c, *map(float, b)) for c, b, _ in objs)


def test_clip_and_yolo_line():
    t = T.Tile(0, 1, 368, 0)
    assert T.clip_box((300, 10, 400, 30), t) == (0, 10, 32, 30)
    assert T.clip_box((10, 10, 20, 20), t) is None
    assert T.yolo_line(5, (0, 10, 32, 30)) == "5 0.025000 0.031250 0.050000 0.031250"

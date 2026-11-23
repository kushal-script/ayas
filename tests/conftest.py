import zlib
from pathlib import Path

import pytest
import yaml

from ayas.calibration import Calibration, calibrate
from ayas.capture import bright_view, capture_set
from ayas.config import VIEWS, load_config
from ayas.golden import GoldenBuilder
from ayas.sim import SimRig

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(root: Path, edit=None):
    """Simulator config writing into a temp dir, with calibration and golden built."""
    raw = yaml.safe_load((ROOT / "configs" / "rig.sim.yaml").read_text())
    raw["paths"] = {
        "data_dir": "data",
        "spec": str(ROOT / "configs" / "spec.yaml"),
        "classifier": "models/none.joblib",
    }
    if edit:
        edit(raw)
    (root / "configs").mkdir()
    path = root / "configs" / "rig.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)

    rig = SimRig(cfg, seed=11)
    rig.place_part(None)
    frames = capture_set(rig.lights, rig.cameras, cfg.states, settle_s=0)
    calibrate({v: bright_view(frames, v, cfg.lights) for v in VIEWS}, cfg).save(cfg.calibration_path)

    builder = GoldenBuilder(cfg, Calibration.load(cfg.calibration_path))
    for _ in range(15):
        rig.place_part(None)
        builder.add(capture_set(rig.lights, rig.cameras, cfg.states, settle_s=0))
    builder.finalize().save(cfg.golden_path)
    return cfg


@pytest.fixture(scope="session")
def sim_cfg(tmp_path_factory):
    return make_cfg(tmp_path_factory.mktemp("rig"))


@pytest.fixture(scope="session")
def arch_only_cfg(tmp_path_factory):
    """The 3-light build: one light beside each camera on the arch, nothing else."""

    def edit(raw):
        leds = raw["lights"]["leds"]
        for name in ("P210", "P330"):
            leds.pop(name)
        leds["L90"]["shape"] = True

    return make_cfg(tmp_path_factory.mktemp("arch"), edit)


@pytest.fixture(scope="session")
def inspector(sim_cfg):
    from ayas.pipeline import Inspector

    return Inspector(sim_cfg)


@pytest.fixture()
def rig(sim_cfg, request):
    return SimRig(sim_cfg, seed=zlib.crc32(request.node.name.encode()))


def capture(cfg, rig):
    return capture_set(rig.lights, rig.cameras, cfg.states, settle_s=0)

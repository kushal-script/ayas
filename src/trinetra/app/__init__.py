"""TriNetra on-device app (docs/twin/DESIGN_STEEL.md H, docs/APP.md).

camera (or twin bursts) -> lightproc.process_burst -> rectify -> tiles -> detector -> mm
measures -> rules.judge_steel -> record 3.0 (one JSON hand-off) -> alert -> SQLite + JSONL log
-> Llama via ExecuTorch explains it (after the alert: the verdict never waits for the model).

Modules: sources (TwinSource, CameraSource, FileSource), pipeline (InspectionPipeline,
FakeDetector), advisor (AdvisorService: mock, llama_main runner, in-process TextLLMRunner),
ui (stdlib http.server station page), bench (per-stage timing, FPS, memory, Llama TTFT and
tokens/s), cli (trinetra-app). Offline by design: nothing here imports a network client; the
only socket is the local UI server.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(path: str | Path) -> Path:
    """A config path: absolute as is, else relative to the cwd if it exists there, else to the repo root."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p

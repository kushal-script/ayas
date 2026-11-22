# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T17:03:29+05:30

30 inspections after 2 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `None`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 546.2 | 724.2 | 573.8 | 870.4 |
| process | 1117.5 | 1354.0 | 1133.5 | 1358.0 |
| rectify | 6.9 | 7.3 | 6.5 | 7.4 |
| tile | 4.2 | 4.3 | 4.2 | 4.3 |
| detect | 1328.7 | 1338.9 | 1330.7 | 1361.6 |
| merge | 0.3 | 0.4 | 0.2 | 0.7 |
| measure | 7.6 | 12.6 | 6.5 | 15.4 |
| rules | 0.2 | 0.3 | 0.1 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 6.1 | 10.8 | 6.9 | 11.7 |
| total | 2856.4 | 3330.5 | 2568.3 | 3338.4 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.39 |
| FPS (loop incl. source + log) | 0.39 |
| Camera-to-result p50 / p95 ms | 2856.4 / 3330.5 |
| Camera-to-advice p50 / p95 ms | - / - |
| Peak RSS (app) MB | 1076.7 |
| Peak RSS (child: llama runner) MB | 0.0 |
| CPU temp start / max / end C | 58.4 / 66.7 / 66.7 |
| Verdicts | REJECT 2, HOLD_FOR_QA 10, RETAKE 11, PASS 7 |
| Capture gates | PASS 19, RETAKE 11 |

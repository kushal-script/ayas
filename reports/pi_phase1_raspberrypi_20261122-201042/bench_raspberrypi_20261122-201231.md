# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T20:12:31+05:30

30 inspections after 2 warm-up; source `twin:data/twin_demo/bursts`, detector `hailo:models/v8n_burst.hef`, advisor `None`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 546.5 | 714.2 | 571.5 | 858.6 |
| process | 1110.0 | 1381.2 | 1132.0 | 1384.8 |
| rectify | 6.9 | 7.5 | 6.5 | 9.7 |
| tile | 4.2 | 4.2 | 4.2 | 4.3 |
| detect | 47.1 | 47.7 | 47.1 | 47.8 |
| merge | 0.3 | 0.6 | 0.3 | 0.7 |
| measure | 7.5 | 13.1 | 6.1 | 13.1 |
| rules | 0.2 | 0.2 | 0.1 | 0.2 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 4.8 | 7.4 | 5.5 | 7.5 |
| total | 1665.3 | 2106.1 | 1750.0 | 2244.8 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.57 |
| FPS (loop incl. source + log) | 0.56 |
| Camera-to-result p50 / p95 ms | 1665.3 / 2106.1 |
| Camera-to-advice p50 / p95 ms | - / - |
| Peak RSS (app) MB | 984.1 |
| Peak RSS (child: llama runner) MB | 0.0 |
| CPU temp start / max / end C | 55.6 / 59.5 / 59.0 |
| Verdicts | HOLD_FOR_QA 12, RETAKE 11, PASS 7 |
| Capture gates | PASS 19, RETAKE 11 |

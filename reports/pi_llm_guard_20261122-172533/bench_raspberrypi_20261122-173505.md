# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T17:35:05+05:30

8 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 577.5 | 725.4 | 596.7 | 741.1 |
| process | 1095.5 | 1389.9 | 1165.1 | 1396.9 |
| rectify | 5.4 | 7.1 | 6.1 | 7.1 |
| tile | 4.2 | 4.3 | 4.2 | 4.3 |
| detect | 1348.2 | 1385.9 | 1352.2 | 1400.8 |
| merge | 0.3 | 0.7 | 0.4 | 0.8 |
| measure | 8.3 | 14.4 | 10.4 | 14.8 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 7.9 | 12.1 | 7.8 | 12.8 |
| total | 2927.4 | 3410.5 | 2971.9 | 3417.1 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.34 |
| FPS (loop incl. source + log) | 0.02 |
| Camera-to-result p50 / p95 ms | 2927.4 / 3410.5 |
| Camera-to-advice p50 / p95 ms | 63953.0 / 86049.5 |
| Peak RSS (app) MB | 1076.8 |
| Peak RSS (child: llama runner) MB | 1376.0 |
| CPU temp start / max / end C | 49.6 / 65.0 / 65.0 |
| Verdicts | HOLD_FOR_QA 6, REJECT 1, RETAKE 1 |
| Capture gates | PASS 7, RETAKE 1 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 6, mock-template 2.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 9394.0 | 11041.2 | 8 |
| First text ms (wall, incl. load) | 11999.5 | 13637.7 | 8 |
| Decode tokens/s | 8.1 | 9.9 | 8 |
| Prefill tokens/s | 94.2 | 110.1 | 8 |
| Generation ms (model clock) | 58019.5 | 80522.3 | 8 |
| Advisor call ms (wall) | 60654.6 | 83153.2 | 8 |
| Prompt tokens | 884.5 | 1008.3 | 8 |
| Generated tokens | 396.5 | 544.6 | 8 |
| Model load ms | 2508.0 | 2510.0 | 8 |

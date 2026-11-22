# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T17:45:32+05:30

12 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 578.4 | 726.5 | 597.3 | 741.5 |
| process | 1099.3 | 1411.7 | 1167.5 | 1418.4 |
| rectify | 5.4 | 7.1 | 6.1 | 7.1 |
| tile | 4.3 | 4.3 | 4.2 | 4.3 |
| detect | 1354.0 | 1359.5 | 1345.7 | 1361.4 |
| merge | 0.3 | 0.6 | 0.4 | 0.7 |
| measure | 8.9 | 14.4 | 10.2 | 16.0 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 7.7 | 11.4 | 8.0 | 11.8 |
| total | 2911.2 | 3368.1 | 2798.3 | 3409.2 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.36 |
| FPS (loop incl. source + log) | 0.04 |
| Camera-to-result p50 / p95 ms | 2911.2 / 3368.1 |
| Camera-to-advice p50 / p95 ms | 23689.2 / 29377.7 |
| Peak RSS (app) MB | 1077.5 |
| Peak RSS (child: llama runner) MB | 1376.0 |
| CPU temp start / max / end C | 52.4 / 66.1 / 65.0 |
| Verdicts | HOLD_FOR_QA 8, REJECT 1, RETAKE 3 |
| Capture gates | PASS 9, RETAKE 3 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 12.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 8927.5 | 10090.5 | 12 |
| First text ms (wall, incl. load) | 11531.5 | 12690.0 | 12 |
| Decode tokens/s | 8.9 | 11.2 | 12 |
| Prefill tokens/s | 96.5 | 112.4 | 12 |
| Generation ms (model clock) | 18142.0 | 23607.9 | 12 |
| Advisor call ms (wall) | 20776.0 | 26240.9 | 12 |
| Prompt tokens | 847.0 | 947.1 | 12 |
| Generated tokens | 82.5 | 114.1 | 12 |
| Model load ms | 2510.0 | 2516.0 | 12 |

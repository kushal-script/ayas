# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T18:13:50+05:30

16 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 560.7 | 773.8 | 603.3 | 873.1 |
| process | 1158.3 | 1399.7 | 1183.0 | 1401.8 |
| rectify | 5.4 | 7.4 | 6.2 | 7.4 |
| tile | 4.3 | 4.3 | 4.3 | 4.4 |
| detect | 1352.8 | 1356.9 | 1349.6 | 1357.2 |
| merge | 0.3 | 0.4 | 0.3 | 0.4 |
| measure | 7.7 | 12.8 | 7.7 | 13.3 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 5.9 | 7.8 | 6.1 | 8.9 |
| total | 2904.4 | 3395.8 | 2733.5 | 3395.9 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.37 |
| FPS (loop incl. source + log) | 0.04 |
| Camera-to-result p50 / p95 ms | 2904.4 / 3395.8 |
| Camera-to-advice p50 / p95 ms | 24161.2 / 54492.2 |
| Peak RSS (app) MB | 1077.3 |
| Peak RSS (child: llama runner) MB | 1376.4 |
| CPU temp start / max / end C | 52.4 / 67.8 / 67.8 |
| Verdicts | HOLD_FOR_QA 8, REJECT 1, RETAKE 5, PASS 2 |
| Capture gates | PASS 11, RETAKE 5 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: mock-template 6, llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 3, rules-engine 7.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 11272.0 | 12731.4 | 9 |
| First text ms (wall, incl. load) | 13653.1 | 14507.2 | 9 |
| Decode tokens/s | 8.1 | 9.2 | 9 |
| Prefill tokens/s | 90.0 | 103.0 | 9 |
| Generation ms (model clock) | 20611.0 | 24927.0 | 9 |
| Advisor call ms (wall) | 21267.7 | 51218.6 | 16 |
| Prompt tokens | 1032.0 | 1125.0 | 9 |
| Generated tokens | 97.0 | 109.2 | 9 |
| Model load ms | 2507.0 | 2513.2 | 9 |

# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T18:32:58+05:30

16 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 560.9 | 770.9 | 602.2 | 872.5 |
| process | 1124.0 | 1404.3 | 1169.3 | 1406.2 |
| rectify | 6.9 | 7.4 | 6.3 | 7.4 |
| tile | 4.1 | 4.3 | 4.2 | 4.3 |
| detect | 1359.1 | 1362.2 | 1349.0 | 1362.7 |
| merge | 0.3 | 0.4 | 0.3 | 0.4 |
| measure | 8.2 | 14.6 | 8.4 | 15.1 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.5 | 0.3 | 0.7 |
| image | 7.4 | 12.5 | 8.1 | 14.6 |
| total | 2899.7 | 3354.3 | 2720.7 | 3356.5 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.37 |
| FPS (loop incl. source + log) | 0.08 |
| Camera-to-result p50 / p95 ms | 2899.7 / 3354.3 |
| Camera-to-advice p50 / p95 ms | 14105.6 / 29063.9 |
| Peak RSS (app) MB | 1076.7 |
| Peak RSS (child: llama runner) MB | 1376.3 |
| CPU temp start / max / end C | 49.6 / 67.2 / 67.2 |
| Verdicts | HOLD_FOR_QA 8, REJECT 1, RETAKE 5, PASS 2 |
| Capture gates | PASS 11, RETAKE 5 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: mock-template 3, llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 6, rules-engine 7.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 5315.0 | 6097.0 | 9 |
| First text ms (wall, incl. load) | 7919.9 | 8693.1 | 9 |
| Decode tokens/s | 10.5 | 11.3 | 9 |
| Prefill tokens/s | 111.8 | 112.3 | 9 |
| Generation ms (model clock) | 10094.0 | 11029.0 | 9 |
| Advisor call ms (wall) | 11227.1 | 25864.9 | 16 |
| Prompt tokens | 596.0 | 675.6 | 9 |
| Generated tokens | 50.0 | 55.0 | 9 |
| Model load ms | 2504.0 | 2508.8 | 9 |

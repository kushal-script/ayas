# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T18:02:37+05:30

16 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 559.8 | 771.8 | 603.3 | 873.4 |
| process | 1156.8 | 1400.8 | 1179.0 | 1409.0 |
| rectify | 5.4 | 7.3 | 6.2 | 7.4 |
| tile | 4.3 | 4.3 | 4.3 | 4.3 |
| detect | 1354.2 | 1357.1 | 1347.0 | 1358.1 |
| merge | 0.3 | 0.6 | 0.3 | 0.7 |
| measure | 8.1 | 13.3 | 8.0 | 13.4 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 7.4 | 10.9 | 7.5 | 11.2 |
| total | 2903.3 | 3360.6 | 2729.2 | 3398.3 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.37 |
| FPS (loop incl. source + log) | 0.07 |
| Camera-to-result p50 / p95 ms | 2903.3 / 3360.6 |
| Camera-to-advice p50 / p95 ms | 21230.8 / 26189.9 |
| Peak RSS (app) MB | 1076.9 |
| Peak RSS (child: llama runner) MB | 1376.0 |
| CPU temp start / max / end C | 53.5 / 67.8 / 67.8 |
| Verdicts | HOLD_FOR_QA 8, REJECT 1, RETAKE 5, PASS 2 |
| Capture gates | PASS 11, RETAKE 5 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: mock-template 5, llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 4, rules-engine 7.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 9197.0 | 10864.4 | 9 |
| First text ms (wall, incl. load) | 11799.1 | 13462.1 | 9 |
| Decode tokens/s | 8.9 | 9.5 | 9 |
| Prefill tokens/s | 94.3 | 103.6 | 9 |
| Generation ms (model clock) | 18427.0 | 20805.8 | 9 |
| Advisor call ms (wall) | 18362.0 | 23164.0 | 16 |
| Prompt tokens | 862.0 | 993.0 | 9 |
| Generated tokens | 76.0 | 98.8 | 9 |
| Model load ms | 2509.0 | 2514.6 | 9 |

# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T17:54:45+05:30

16 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 560.5 | 771.1 | 603.3 | 874.4 |
| process | 1148.8 | 1395.6 | 1175.9 | 1398.0 |
| rectify | 6.9 | 7.4 | 6.3 | 7.4 |
| tile | 4.3 | 4.3 | 4.2 | 4.3 |
| detect | 1341.3 | 1343.2 | 1335.8 | 1343.3 |
| merge | 0.3 | 0.6 | 0.3 | 0.7 |
| measure | 7.7 | 13.3 | 7.9 | 13.3 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 7.3 | 9.1 | 6.8 | 9.8 |
| total | 2888.3 | 3336.9 | 2717.8 | 3380.9 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.37 |
| FPS (loop incl. source + log) | 0.06 |
| Camera-to-result p50 / p95 ms | 2888.3 / 3336.9 |
| Camera-to-advice p50 / p95 ms | 23234.0 / 26827.9 |
| Peak RSS (app) MB | 1076.8 |
| Peak RSS (child: llama runner) MB | 1376.4 |
| CPU temp start / max / end C | 51.8 / 66.1 / 64.5 |
| Verdicts | HOLD_FOR_QA 8, REJECT 1, RETAKE 5, PASS 2 |
| Capture gates | PASS 11, RETAKE 5 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 9, mock-template 2, rules-engine 5.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 9094.0 | 10629.0 | 11 |
| First text ms (wall, incl. load) | 11702.8 | 13221.5 | 11 |
| Decode tokens/s | 8.8 | 11.2 | 11 |
| Prefill tokens/s | 95.0 | 112.6 | 11 |
| Generation ms (model clock) | 18593.0 | 21688.0 | 11 |
| Advisor call ms (wall) | 20210.6 | 23816.2 | 16 |
| Prompt tokens | 862.0 | 980.0 | 11 |
| Generated tokens | 84.0 | 104.5 | 11 |
| Model load ms | 2510.0 | 2516.5 | 11 |

# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T17:12:30+05:30

8 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 580.4 | 725.0 | 597.6 | 740.2 |
| process | 1118.2 | 1385.2 | 1170.6 | 1400.4 |
| rectify | 5.4 | 7.1 | 6.1 | 7.1 |
| tile | 4.3 | 4.4 | 4.3 | 4.4 |
| detect | 1347.5 | 1350.5 | 1348.1 | 1351.1 |
| merge | 0.3 | 0.4 | 0.3 | 0.4 |
| measure | 8.2 | 13.0 | 9.8 | 13.3 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 5.3 | 8.8 | 6.2 | 8.9 |
| total | 2906.4 | 3394.7 | 2972.6 | 3399.3 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.34 |
| FPS (loop incl. source + log) | 0.02 |
| Camera-to-result p50 / p95 ms | 2906.4 / 3394.7 |
| Camera-to-advice p50 / p95 ms | 56537.0 / 66628.5 |
| Peak RSS (app) MB | 1075.9 |
| Peak RSS (child: llama runner) MB | 1376.0 |
| CPU temp start / max / end C | 62.8 / 66.1 / 64.5 |
| Verdicts | HOLD_FOR_QA 6, REJECT 1, RETAKE 1 |
| Capture gates | PASS 7, RETAKE 1 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 7, mock-template 1.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 15797.5 | 17641.2 | 8 |
| First text ms (wall, incl. load) | 18405.2 | 20253.5 | 8 |
| Decode tokens/s | 6.7 | 8.4 | 8 |
| Prefill tokens/s | 84.9 | 101.0 | 8 |
| Generation ms (model clock) | 50917.5 | 61024.2 | 8 |
| Advisor call ms (wall) | 53548.2 | 63667.0 | 8 |
| Prompt tokens | 1341.5 | 1456.9 | 8 |
| Generated tokens | 233.0 | 413.2 | 8 |
| Model load ms | 2512.5 | 2516.7 | 8 |

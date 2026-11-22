# TriNetra benchmark: raspberrypi (Raspberry Pi 5 Model B Rev 1.1), 2026-11-22T18:40:24+05:30

16 inspections after 1 warm-up; source `twin:data/twin_demo/bursts`, detector `onnx:models/v8n_burst.onnx`, advisor `llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte)`. Python 3.13.5, 4 CPUs.

## Vision pipeline (ms per inspection)

| Stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| capture | 561.4 | 773.6 | 605.3 | 877.0 |
| process | 1135.0 | 1389.1 | 1160.6 | 1397.3 |
| rectify | 5.4 | 8.6 | 6.3 | 9.7 |
| tile | 4.1 | 4.2 | 4.2 | 4.3 |
| detect | 1337.1 | 1365.8 | 1341.9 | 1366.8 |
| merge | 0.3 | 0.4 | 0.3 | 0.4 |
| measure | 8.8 | 12.7 | 8.2 | 13.3 |
| rules | 0.2 | 0.3 | 0.2 | 0.3 |
| report | 0.3 | 0.4 | 0.3 | 0.4 |
| image | 7.5 | 11.1 | 7.3 | 11.1 |
| total | 2866.2 | 3344.2 | 2709.2 | 3349.7 |

## System

| Metric | Value |
|---|---|
| FPS (vision, 1000 / mean total) | 0.37 |
| FPS (loop incl. source + log) | 0.11 |
| Camera-to-result p50 / p95 ms | 2866.2 / 3344.2 |
| Camera-to-advice p50 / p95 ms | 13543.9 / 13940.0 |
| Peak RSS (app) MB | 1077.0 |
| Peak RSS (child: llama runner) MB | 1376.0 |
| CPU temp start / max / end C | 51.2 / 66.7 / 66.1 |
| Verdicts | HOLD_FOR_QA 8, REJECT 1, RETAKE 5, PASS 2 |
| Capture gates | PASS 11, RETAKE 5 |

## Llama advisor (llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte))

Answered by: llama via executorch (/home/rp5/trinetra/models/llama3_2_1b_instruct.pte) 9, rules-engine 7.

| Metric | p50 | p95 | n |
|---|---|---|---|
| TTFT ms (model clock) | 5429.0 | 5504.8 | 9 |
| First text ms (wall, incl. load) | 8028.5 | 8099.7 | 9 |
| Decode tokens/s | 11.0 | 11.4 | 9 |
| Prefill tokens/s | 112.5 | 113.3 | 9 |
| Generation ms (model clock) | 8159.0 | 8583.8 | 9 |
| Advisor call ms (wall) | 10276.1 | 11081.1 | 16 |
| Prompt tokens | 611.0 | 611.0 | 9 |
| Generated tokens | 29.0 | 41.0 | 9 |
| Model load ms | 2507.0 | 2510.0 | 9 |

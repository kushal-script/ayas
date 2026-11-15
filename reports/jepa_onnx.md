# JEPA ONNX export and CPU latency

Host: macOS-27.0.1-arm64-arm-64bit (arm). Checkpoint step 2000 (SMOKE model: the latency does not depend on the training, the scores do). Batch 1, 1 x 3 x 224 x 224, onnxruntime CPU EP, ORT_ENABLE_ALL; median (min) of the timed runs after 5 warm-ups. Other jobs (Blender renders, post) shared the Mac while this ran (load average 13, 25, 50 on 8 cores), so the medians are upper-side numbers; the min is the closer estimate of an idle core.

Pi 5 columns are a PROJECTION, not a measurement: Mac min ms x 3-5 (Cortex-A76 2.4 GHz vs M3 P-core: about 3.4x less fp32 FMA throughput per core, about 6x less memory bandwidth; docs/JEPA.md section 6).

| model | MB | Mac 1 thread ms median (min) | Mac 4 threads ms median (min) | Pi 5 1 thread ms (proj.) | Pi 5 4 threads ms (proj.) | max abs err vs torch | ops outside the ViT set |
|---|---|---|---|---|---|---|---|
| jepa_smoke_encoder.onnx | 22.19 | 106.95 (90.55) | 62.81 (32.18) | 272–453 | 97–161 | 1.5e-05 | – |
| jepa_smoke_knn.onnx | 25.35 | 117.79 (103.1) | 63.71 (39.78) | 309–516 | 119–199 | 1.5e-05 | Clip, ReduceMin, ReduceSum |
| jepa_smoke_knn_head.onnx | 3.16 | 9.23 (8.24) | 2.93 (2.52) | 25–41 | 8–13 | nan | Clip, ReduceMin, ReduceSum |
| jepa_smoke_pred.onnx | 46.92 | 392.86 (228.43) | 122.58 (116.11) | 685–1142 | 348–581 | 2.3e-06 | ConstantOfShape, Equal, Expand, Gather, GatherElements, Where |
| jepa_smoke_encoder_int8.onnx | 6.36 | 34.12 (31.5) | 16.69 (14.64) | 94–158 | 44–73 | 0.63 | – |
| jepa_smoke_knn_int8.onnx | 7.16 | 31.03 (30.53) | 16.07 (14.5) | 92–153 | 44–72 | 0.71 | – |

Op counts per graph:

- `jepa_smoke_encoder.onnx`: Add 85, Cast 24, Concat 1, Constant 148, Conv 1, Div 24, Erf 12, LayerNormalization 25, MatMul 72, Mul 48, Reshape 25, Shape 13, Slice 13, Softmax 12, Split 12, Sqrt 36, Squeeze 36, Transpose 37
- `jepa_smoke_knn.onnx`: Add 86, Cast 24, Clip 1, Concat 1, Constant 155, Conv 1, Div 25, Erf 12, LayerNormalization 25, MatMul 73, Mul 49, Pow 1, ReduceMin 1, ReduceSum 1, Reshape 26, Shape 13, Slice 13, Softmax 12, Split 12, Sqrt 37, Squeeze 36, Sub 2, Transpose 37
- `jepa_smoke_knn_head.onnx`: Add 1, Clip 1, Constant 7, Div 1, MatMul 1, Mul 1, Pow 1, ReduceMin 1, ReduceSum 1, Reshape 1, Sqrt 1, Sub 2
- `jepa_smoke_pred.onnx`: Add 550, Cast 152, Concat 7, Constant 1018, ConstantOfShape 24, Conv 2, Div 153, Equal 24, Erf 76, Expand 24, Gather 1, GatherElements 16, Identity 198, LayerNormalization 162, MatMul 464, Mul 328, Pow 4, ReduceMean 4, Reshape 155, Shape 78, Slice 82, Softmax 76, Split 76, Sqrt 228, Squeeze 228, Sub 5, Transpose 230, Where 24

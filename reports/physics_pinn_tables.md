### Coupons: depth error by ring partition and method (dark room, hood, Blender per-LED renders + sensor noise)

| LED groups | method | class | n | MAE mm | median abs mm | bias mm | median rel | within 0.05 mm | within 25 % | median err / T1 | CI coverage | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L/R | pinn | dent | 66 | 0.053 | 0.035 | -0.020 | -0.02 | 62 % | 68 % | 0.34 | 89 % | 73 % |
| L/R | pinn | pit | 30 | 0.154 | 0.137 | -0.154 | -0.54 | 20 % | 17 % | 0.77 | 90 % | 13 % |
| L/R | classical | dent | 66 | 0.153 | 0.131 | -0.082 | -0.40 | 17 % | 23 % | 0.49 | 85 % | 100 % |
| L/R | classical | pit | 30 | 0.226 | 0.213 | -0.225 | -0.75 | 10 % | 3 % | 1.35 | 90 % | 100 % |
| 3 sectors | pinn | dent | 66 | 0.034 | 0.021 | -0.001 | 0.03 | 79 % | 86 % | 0.08 | 89 % | 80 % |
| 3 sectors | pinn | pit | 30 | 0.114 | 0.096 | -0.113 | -0.35 | 27 % | 37 % | 0.63 | 90 % | 30 % |
| 3 sectors | classical | dent | 66 | 0.089 | 0.050 | -0.057 | -0.06 | 48 % | 52 % | 0.64 | 89 % | 100 % |
| 3 sectors | classical | pit | 30 | 0.201 | 0.190 | -0.201 | -0.68 | 13 % | 7 % | 1.13 | 90 % | 100 % |
| 6 sectors | pinn | dent | 66 | 0.021 | 0.013 | 0.009 | 0.03 | 92 % | 94 % | 0.06 | 89 % | 80 % |
| 6 sectors | pinn | pit | 30 | 0.107 | 0.091 | -0.106 | -0.28 | 30 % | 33 % | 0.55 | 90 % | 40 % |
| 6 sectors | classical | dent | 66 | 0.071 | 0.045 | -0.037 | -0.03 | 55 % | 64 % | 0.42 | 89 % | 100 % |
| 6 sectors | classical | pit | 30 | 0.164 | 0.146 | -0.164 | -0.56 | 13 % | 10 % | 0.71 | 90 % | 100 % |

### Coupons: PINN dent depth error by material

| material | LED groups | n | MAE mm | bias mm | within 0.05 mm |
|---|---|---|---|---|---|
| aluminium | L/R | 11 | 0.077 | -0.050 | 36 % |
| aluminium | 3 sectors | 11 | 0.056 | 0.003 | 55 % |
| aluminium | 6 sectors | 11 | 0.032 | 0.009 | 82 % |
| galvanised | L/R | 11 | 0.077 | -0.027 | 45 % |
| galvanised | 3 sectors | 11 | 0.047 | -0.014 | 64 % |
| galvanised | 6 sectors | 11 | 0.026 | -0.001 | 82 % |
| ms_cold_rolled | L/R | 11 | 0.027 | 0.007 | 82 % |
| ms_cold_rolled | 3 sectors | 11 | 0.023 | 0.011 | 100 % |
| ms_cold_rolled | 6 sectors | 11 | 0.016 | 0.016 | 100 % |
| ms_hot_rolled | L/R | 11 | 0.012 | -0.006 | 100 % |
| ms_hot_rolled | 3 sectors | 11 | 0.016 | -0.003 | 100 % |
| ms_hot_rolled | 6 sectors | 11 | 0.011 | -0.003 | 100 % |
| ss_2b | L/R | 11 | 0.056 | -0.055 | 55 % |
| ss_2b | 3 sectors | 11 | 0.020 | -0.008 | 82 % |
| ss_2b | 6 sectors | 11 | 0.012 | 0.009 | 100 % |
| ss_brushed | L/R | 11 | 0.066 | 0.013 | 55 % |
| ss_brushed | 3 sectors | 11 | 0.044 | 0.005 | 73 % |
| ss_brushed | 6 sectors | 11 | 0.027 | 0.022 | 91 % |

### Coupons: disposition with the estimated depth vs with the true depth

| LED groups | method | n | depth used | used and agrees | agrees (all) | more lenient: depth used | more lenient: fallback | stricter | today agrees | today more lenient | today stricter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L/R | pinn | 96 | 11 | 11 | 38 | 0 | 19 | 39 | 34 | 23 | 39 |
| L/R | classical | 96 | 4 | 4 | 36 | 0 | 23 | 37 | 34 | 23 | 39 |
| 3 sectors | pinn | 96 | 30 | 30 | 49 | 0 | 15 | 32 | 34 | 23 | 39 |
| 3 sectors | classical | 96 | 6 | 6 | 36 | 0 | 22 | 38 | 34 | 23 | 39 |
| 6 sectors | pinn | 96 | 48 | 48 | 59 | 0 | 10 | 27 | 34 | 23 | 39 |
| 6 sectors | classical | 96 | 5 | 5 | 35 | 0 | 22 | 39 | 34 | 23 | 39 |

### Pilot plates: PINN depth across the 7 lighting conditions (L/R, real burst path)

| plate | defect | material, t | true mm | hood: median [min–max] mm (n, retakes) | open: median [min–max] mm (n, retakes) | classical hood median | disposition true / today / estimated (hood) |
|---|---|---|---|---|---|---|---|
| s900011 | D2 scratch.scriber | ms_cold_rolled, 2.5 | 0.071 | 0.137 [0.128–0.148] (7, 0) | 0.004 [0.004–0.004] (1, 6) | 0.123 | REWORK / HOLD_FOR_QA / HOLD_FOR_QA 7 |
| s900011 | D4 scratch.scriber | ms_cold_rolled, 2.5 | 0.095 | 0.117 [0.108–0.130] (7, 0) | 0.002 [0.002–0.002] (1, 6) | 0.054 | REWORK / HOLD_FOR_QA / HOLD_FOR_QA 7 |
| s900027 | D1 dent.punch | ms_hot_rolled, 5.2 | 0.353 | 0.353 [0.349–0.372] (7, 0) | 0.232 [0.218–0.236] (3, 4) | 0.141 | PASS / PASS / PASS 7 |
| s900027 | D2 scratch.gouge | ms_hot_rolled, 5.2 | 0.260 | 0.295 [0.292–0.297] (7, 0) | 0.147 [0.145–0.147] (3, 4) | 0.270 | HOLD_FOR_QA / HOLD_FOR_QA / HOLD_FOR_QA 7 |
| s900028 | D1 pit.cluster | aluminium, 5.7 | 0.292 | 0.115 [0.099–0.125] (7, 0) | 0.045 [0.040–0.050] (2, 5) | 0.039 | PASS / PASS / PASS 7 |
| s900028 | D2 scratch.scriber | aluminium, 5.7 | 0.013 | 0.019 [0.018–0.025] (7, 0) | 0.022 [0.021–0.023] (2, 5) | 0.080 | PASS / PASS / PASS 7 |
| s900028 | D3 dent.ball | aluminium, 5.7 | 0.627 | 0.441 [0.434–0.449] (7, 0) | 0.410 [0.409–0.412] (2, 5) | 0.375 | REJECT / REJECT / REJECT 7 |
| s900028 | D5 dent.ball | aluminium, 5.7 | 0.163 | 0.126 [0.119–0.128] (7, 0) | 0.191 [0.186–0.195] (2, 5) | 0.187 | PASS / REJECT / PASS 7 |
| s900068 | D1 pit.cluster | ss_brushed, 4.9 | 0.165 | 0.074 [0.048–0.094] (7, 0) | 0.018 [0.018–0.018] (1, 6) | 0.042 | PASS / HOLD_FOR_QA / HOLD_FOR_QA 7 |
| s900068 | D2 pit.single | ss_brushed, 4.9 | 0.388 | 0.046 [0.039–0.061] (7, 0) | 0.058 [0.058–0.058] (1, 6) | 0.035 | REWORK / HOLD_FOR_QA / HOLD_FOR_QA 7 |

### Pilot plates: depth error per class (every accepted capture, all 7 conditions)

| mode | method | class | n | MAE mm | median abs mm | bias mm | within 0.05 mm | CI coverage | converged |
|---|---|---|---|---|---|---|---|---|---|
| hood | pinn | dent | 21 | 0.077 | 0.037 | -0.073 | 67 % | 100 % | 76 % |
| hood | pinn | pit | 21 | 0.204 | 0.177 | -0.204 | 0 % | 62 % | 0 % |
| hood | pinn | scratch | 28 | 0.033 | 0.032 | 0.033 | 75 % | 75 % | 18 % |
| hood | classical | dent | 21 | 0.164 | 0.212 | -0.146 | 33 % | 100 % | 100 % |
| hood | classical | pit | 21 | 0.242 | 0.253 | -0.242 | 0 % | 38 % | 100 % |
| hood | classical | scratch | 28 | 0.041 | 0.043 | 0.021 | 61 % | – | 100 % |
| open | pinn | dent | 7 | 0.123 | 0.121 | -0.107 | 29 % | 29 % | 100 % |
| open | pinn | pit | 4 | 0.242 | 0.247 | -0.242 | 0 % | 0 % | 0 % |
| open | pinn | scratch | 7 | 0.074 | 0.093 | -0.069 | 29 % | 29 % | 71 % |
| open | classical | dent | 7 | 0.209 | 0.219 | -0.091 | 0 % | 100 % | 100 % |
| open | classical | pit | 4 | 0.262 | 0.259 | -0.262 | 0 % | 0 % | 100 % |
| open | classical | scratch | 7 | 0.102 | 0.078 | -0.102 | 43 % | – | 100 % |

### Pilot plates: dispositions over every accepted capture (dents and pits, PINN, L/R)

| mode | n | depth used | used and agrees | agrees (all) | more lenient: depth used | more lenient: fallback | stricter | today agrees | today more lenient | today stricter |
|---|---|---|---|---|---|---|---|---|---|---|
| hood | 42 | 7 | 7 | 28 | 0 | 0 | 14 | 21 | 0 | 21 |
| open | 11 | 0 | 0 | 7 | 0 | 0 | 4 | 7 | 0 | 4 |

### Interval calibration (coupons, per class and partition; SYS_ABS = 0.02 mm)

| key | SYS_REL |
|---|---|
| pinn_dent_2groups | 0.461 |
| pinn_pit_2groups | 2.152 |
| pinn_dent_3groups | 0.232 |
| pinn_pit_3groups | 1.115 |
| pinn_dent_6groups | 0.103 |
| pinn_pit_6groups | 1.061 |
| classical_dent_2groups | 4.874 |
| classical_pit_2groups | 5.648 |
| classical_dent_3groups | 1.218 |
| classical_pit_3groups | 5.724 |
| classical_dent_6groups | 0.891 |
| classical_pit_6groups | 2.193 |

### Runtime (macOS-27.0.1-arm64-arm-64bit, torch 2.15.0; load average at the end [37.00244140625, 55.3046875, 111.12109375])

| threads | defect | defect mm | crop grid px | working px (block) | PINN s (seeds) | classical s |
|---|---|---|---|---|---|---|
| 1 | D12 pit.single | 0.8 | 30 | 30×31 (1×) | 33.57 (2) | 5.67 |
| 1 | D1 dent.ball | 3.0 | 71 | 36×36 (2×) | 16.49 (2) | 5.5 |
| 1 | D13 dent.ball | 5.0 | 119 | 39×40 (3×) | 15.06 (2) | 7.55 |
| 1 | D6 dent.ball | 6.0 | 142 | 47×47 (3×) | 16.48 (2) | 9.78 |
| 1 | D10 dent.ball | 10.0 | 238 | 47×47 (5×) | 22.39 (2) | 11.33 |
| 4 | D12 pit.single | 0.8 | 30 | 30×31 (1×) | 50.07 (2) | 9.53 |
| 4 | D1 dent.ball | 3.0 | 71 | 36×36 (2×) | 26.59 (2) | 13.22 |
| 4 | D13 dent.ball | 5.0 | 119 | 39×40 (3×) | 12.51 (2) | 5.24 |
| 4 | D6 dent.ball | 6.0 | 142 | 47×47 (3×) | 12.74 (2) | 6.59 |
| 4 | D10 dent.ball | 10.0 | 238 | 47×47 (5×) | 12.64 (2) | 5.86 |

# Step D (GC v2) on A30: results

Six cumulative runs, rd 100k steps each, same seed, data and lambda ladder. Real streams at 8 levels on the 30 CLIC images, scored with Qhat v0 and LPIPS (alex). BD-rate: negative means fewer bits for the same quality. Gate to keep a change: <= -2% on both metrics vs the previous run. 'Low' means levels 0-3.

| Run | Change | vs prev Qhat | vs prev LPIPS | Gate | vs D0 Qhat | vs D0 LPIPS |
|---|---|---:|---:|---|---:|---:|
| D0 | v1 baseline | | | | | |
| D1 | + E1/E2 | -2.07% | -1.58% | fail | -2.07% | -1.58% |
| D2 | + CCA loss 0.3 | -0.06% | +4.53% | fail | -2.16% | +2.93% |
| D3 | + 10 slices x 32 | +0.00% | -4.84% | fail | -2.12% | -1.94% |
| D4 | + split Sq path (integer sigma on q, float mu) | +12.68% | +18.23% | fail | +10.11% | +15.35% |
| D5 | + g_s v2 | -9.86% | -6.34% | pass | -0.54% | +8.20% |

Mean over the 30 images per level (bpp / PSNR / LPIPS / Qhat v0 / decode s on A30):

| Level | D0 | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|---|
| q0 | 0.034 / 25.32 / 0.178 / -2.66 / 0.31 | 0.034 / 25.33 / 0.177 / -2.65 / 0.33 | 0.038 / 25.70 / 0.172 / -2.42 / 0.34 | 0.040 / 25.76 / 0.164 / -2.39 / 0.43 | 0.038 / 25.71 / 0.177 / -2.59 / 0.40 | 0.038 / 25.70 / 0.180 / -2.54 / 0.42 |
| q1 | 0.054 / 26.52 / 0.139 / -1.82 / 0.30 | 0.053 / 26.53 / 0.140 / -1.86 / 0.33 | 0.059 / 26.87 / 0.135 / -1.63 / 0.33 | 0.060 / 26.93 / 0.130 / -1.61 / 0.42 | 0.061 / 26.88 / 0.139 / -1.74 / 0.40 | 0.060 / 26.92 / 0.140 / -1.69 / 0.41 |
| q2 | 0.082 / 27.69 / 0.109 / -1.14 / 0.29 | 0.080 / 27.69 / 0.109 / -1.07 / 0.33 | 0.087 / 27.94 / 0.106 / -0.96 / 0.32 | 0.089 / 28.01 / 0.101 / -0.95 / 0.42 | 0.096 / 28.08 / 0.108 / -1.03 / 0.40 | 0.091 / 28.10 / 0.108 / -0.97 / 0.41 |
| q3 | 0.118 / 28.71 / 0.086 / -0.55 / 0.29 | 0.115 / 28.67 / 0.086 / -0.55 / 0.33 | 0.124 / 28.88 / 0.085 / -0.44 / 0.33 | 0.127 / 28.97 / 0.081 / -0.41 / 0.43 | 0.144 / 29.14 / 0.085 / -0.41 / 0.40 | 0.130 / 29.13 / 0.084 / -0.39 / 0.41 |
| q4 | 0.162 / 29.50 / 0.071 / -0.09 / 0.30 | 0.158 / 29.44 / 0.071 / -0.09 / 0.33 | 0.168 / 29.58 / 0.071 / -0.04 / 0.33 | 0.171 / 29.71 / 0.067 / +0.02 / 0.43 | 0.206 / 30.00 / 0.068 / +0.07 / 0.40 | 0.175 / 29.90 / 0.069 / +0.05 / 0.42 |
| q5 | 0.213 / 30.04 / 0.061 / +0.23 / 0.30 | 0.208 / 29.97 / 0.062 / +0.22 / 0.33 | 0.220 / 30.09 / 0.061 / +0.28 / 0.33 | 0.223 / 30.22 / 0.058 / +0.33 / 0.43 | 0.284 / 30.67 / 0.057 / +0.44 / 0.40 | 0.225 / 30.44 / 0.059 / +0.36 / 0.42 |
| q6 | 0.270 / 30.42 / 0.055 / +0.49 / 0.30 | 0.265 / 30.31 / 0.056 / +0.45 / 0.33 | 0.278 / 30.42 / 0.056 / +0.49 / 0.33 | 0.281 / 30.56 / 0.053 / +0.56 / 0.43 | 0.381 / 31.17 / 0.050 / +0.72 / 0.40 | 0.279 / 30.78 / 0.053 / +0.57 / 0.42 |
| q7 | 0.337 / 30.65 / 0.051 / +0.65 / 0.30 | 0.338 / 30.55 / 0.052 / +0.62 / 0.34 | 0.347 / 30.64 / 0.052 / +0.66 / 0.33 | 0.350 / 30.76 / 0.049 / +0.71 / 0.43 | 0.505 / 31.54 / 0.045 / +0.95 / 0.40 | 0.338 / 30.98 / 0.050 / +0.73 / 0.42 |

Reading (straight from the gate; the keep/drop call belongs to the analyst):

- Best cumulative config is D3 (-2.12% Qhat, -1.94% LPIPS vs D0); no change through D3 passes the gate on both metrics.
- D4 (split Sq) is clearly worse (+12.7% / +18.2% vs D3; also worse on levels 0-3) and stretches the rate ladder (q7 0.51 bpp vs 0.35).
- D5 (g_s v2) is the only step that passes the gate vs its predecessor (-9.9% / -6.3%), but it sits on D4, so vs D0 it is -0.5% / +8.2%. A D3 + g_s v2 run without Sq would isolate g_s v2.
- Decode time on A30 rises from 0.30 s/image (D0) to 0.43 (D3); g_s v2 adds nothing (0.42).

`STEPD_DONE`

Code: `experiments/gc_stepd/`. `code_snapshot/` holds only the files of the shipped snapshot that differ from origin/main 7cb09d6; all other snapshot files are identical to main. `configs/` are as shipped (lab09 data paths); `configs_a30/` are what `make_cfgs.py` generated and the runs used here (only data paths differ).

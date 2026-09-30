# B6: VTM thật, VTM-SCC, và DC-AE + residual

Base của `dcae+res` giả định tốn 0.03 bpp (danh nghĩa), residual được mã hóa bằng VTM 4:4:4 với phần bit còn lại. Đây là **cận dưới** cho một lớp residual học được.

## screen

| rate | cfg | bpp thật | PSNR | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ |
|---|---|---|---|---|---|---|---|
| ∞ | dcae (trần) | – | 26.55 | 0.0485 | 0.0438 | 16.68 | 0.578 |
| 0.075 | vtm420 | 0.073 | 32.93 | 0.1257 | 0.1163 | 27.88 | 0.256 |
| 0.075 | vtmscc | 0.071 | 34.00 | 0.1161 | 0.1060 | 28.62 | 0.184 |
| 0.075 | dcae+res | 0.069 | 30.98 | 0.0454 | 0.0435 | 24.34 | 0.316 |
| 0.15 | vtm420 | 0.142 | 36.30 | 0.0794 | 0.0802 | 32.15 | 0.124 |
| 0.15 | vtmscc | 0.147 | 38.38 | 0.0656 | 0.0659 | 34.26 | 0.090 |
| 0.15 | dcae+res | 0.142 | 34.25 | 0.0343 | 0.0360 | 28.60 | 0.161 |
| 0.3 | vtm420 | 0.284 | 39.84 | 0.0455 | 0.0492 | 36.45 | 0.060 |
| 0.3 | vtmscc | 0.280 | 43.17 | 0.0417 | 0.0430 | 40.30 | 0.060 |
| 0.3 | dcae+res | 0.285 | 38.03 | 0.0231 | 0.0272 | 33.72 | 0.094 |

## natural

| rate | cfg | bpp thật | PSNR | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ |
|---|---|---|---|---|---|---|---|
| ∞ | dcae (trần) | – | 23.03 | 0.1101 | 0.0744 | 23.70 | 0.277 |
| 0.075 | vtm420 | 0.070 | 24.82 | 0.4009 | 0.2382 | 27.47 | 0.340 |
| 0.075 | vtmscc | 0.070 | 24.97 | 0.3910 | 0.2344 | 27.13 | 0.381 |
| 0.075 | dcae+res | 0.065 | 24.61 | 0.1048 | 0.0700 | 26.30 | 0.173 |
| 0.15 | vtm420 | 0.147 | 27.44 | 0.2636 | 0.1714 | 30.32 | 0.284 |
| 0.15 | vtmscc | 0.139 | 27.49 | 0.2669 | 0.1732 | 30.18 | 0.178 |
| 0.15 | dcae+res | 0.129 | 26.07 | 0.0923 | 0.0682 | 28.08 | 0.165 |
| 0.3 | vtm420 | 0.273 | 29.97 | 0.1696 | 0.1179 | 32.94 | 0.099 |
| 0.3 | vtmscc | 0.283 | 30.57 | 0.1610 | 0.1124 | 33.17 | 0.157 |
| 0.3 | dcae+res | 0.282 | 28.29 | 0.0684 | 0.0610 | 30.54 | 0.114 |

## Từng ảnh (PSNR / OCR CER)

| ảnh | nhãn | cfg | PSNR | CER |
|---|---|---|---|---|
| 2684452d | natural | dcae | 25.59 | 0.304 |
| 2684452d | natural | dcae+res@0.075 | 28.33 | 0.046 |
| 2684452d | natural | dcae+res@0.15 | 29.71 | 0.031 |
| 2684452d | natural | dcae+res@0.3 | 32.36 | 0.028 |
| 2684452d | natural | vtm420@0.075 | 29.17 | 0.080 |
| 2684452d | natural | vtm420@0.15 | 31.77 | 0.067 |
| 2684452d | natural | vtm420@0.3 | 34.32 | 0.098 |
| 2684452d | natural | vtmscc@0.075 | 29.35 | 0.061 |
| 2684452d | natural | vtmscc@0.15 | 31.94 | 0.055 |
| 2684452d | natural | vtmscc@0.3 | 34.82 | 0.064 |
| 608cb09e | natural | dcae | 23.21 | 0.250 |
| 608cb09e | natural | dcae+res@0.075 | 24.52 | 0.300 |
| 608cb09e | natural | dcae+res@0.15 | 26.09 | 0.300 |
| 608cb09e | natural | dcae+res@0.3 | 27.87 | 0.200 |
| 608cb09e | natural | vtm420@0.075 | 24.82 | 0.600 |
| 608cb09e | natural | vtm420@0.15 | 27.32 | 0.500 |
| 608cb09e | natural | vtm420@0.3 | 29.56 | 0.100 |
| 608cb09e | natural | vtmscc@0.075 | 24.79 | 0.700 |
| 608cb09e | natural | vtmscc@0.15 | 27.54 | 0.300 |
| 608cb09e | natural | vtmscc@0.3 | 30.26 | 0.250 |
| da52c4f6 | natural | dcae | 20.28 | – |
| da52c4f6 | natural | dcae+res@0.075 | 20.97 | – |
| da52c4f6 | natural | dcae+res@0.15 | 22.41 | – |
| da52c4f6 | natural | dcae+res@0.3 | 24.64 | – |
| da52c4f6 | natural | vtm420@0.075 | 20.46 | – |
| da52c4f6 | natural | vtm420@0.15 | 23.23 | – |
| da52c4f6 | natural | vtm420@0.3 | 26.03 | – |
| da52c4f6 | natural | vtmscc@0.075 | 20.76 | – |
| da52c4f6 | natural | vtmscc@0.15 | 22.99 | – |
| da52c4f6 | natural | vtmscc@0.3 | 26.65 | – |
| 2a760bf1 | screen | dcae | 34.84 | 0.292 |
| 2a760bf1 | screen | dcae+res@0.075 | 44.36 | 0.000 |
| 2a760bf1 | screen | dcae+res@0.15 | 48.38 | 0.000 |
| 2a760bf1 | screen | dcae+res@0.3 | 51.39 | 0.000 |
| 2a760bf1 | screen | vtm420@0.075 | 46.53 | 0.000 |
| 2a760bf1 | screen | vtm420@0.15 | 49.95 | 0.000 |
| 2a760bf1 | screen | vtm420@0.3 | 51.37 | 0.000 |
| 2a760bf1 | screen | vtmscc@0.075 | 48.30 | 0.000 |
| 2a760bf1 | screen | vtmscc@0.15 | 52.76 | 0.000 |
| 2a760bf1 | screen | vtmscc@0.3 | 55.41 | 0.000 |
| 86127fbd | screen | dcae | 21.82 | 0.886 |
| 86127fbd | screen | dcae+res@0.075 | 22.85 | 0.598 |
| 86127fbd | screen | dcae+res@0.15 | 24.23 | 0.435 |
| 86127fbd | screen | dcae+res@0.3 | 25.90 | 0.332 |
| 86127fbd | screen | vtm420@0.075 | 23.52 | 0.560 |
| 86127fbd | screen | vtm420@0.15 | 25.68 | 0.348 |
| 86127fbd | screen | vtm420@0.3 | 28.45 | 0.207 |
| 86127fbd | screen | vtmscc@0.075 | 23.57 | 0.451 |
| 86127fbd | screen | vtmscc@0.15 | 26.29 | 0.277 |
| 86127fbd | screen | vtmscc@0.3 | 28.55 | 0.223 |
| bb7344a2 | screen | dcae | 27.87 | 0.373 |
| bb7344a2 | screen | dcae+res@0.075 | 32.17 | 0.009 |
| bb7344a2 | screen | dcae+res@0.15 | 35.47 | 0.000 |
| bb7344a2 | screen | dcae+res@0.3 | 39.87 | 0.000 |
| bb7344a2 | screen | vtm420@0.075 | 34.64 | 0.000 |
| bb7344a2 | screen | vtm420@0.15 | 38.49 | 0.000 |
| bb7344a2 | screen | vtm420@0.3 | 43.25 | 0.000 |
| bb7344a2 | screen | vtmscc@0.075 | 35.12 | 0.000 |
| bb7344a2 | screen | vtmscc@0.15 | 40.06 | 0.000 |
| bb7344a2 | screen | vtmscc@0.3 | 45.04 | 0.000 |
| ebfd571f | screen | dcae | 21.66 | 0.759 |
| ebfd571f | screen | dcae+res@0.075 | 24.52 | 0.655 |
| ebfd571f | screen | dcae+res@0.15 | 28.93 | 0.208 |
| ebfd571f | screen | dcae+res@0.3 | 34.97 | 0.046 |
| ebfd571f | screen | vtm420@0.075 | 27.01 | 0.466 |
| ebfd571f | screen | vtm420@0.15 | 31.08 | 0.148 |
| ebfd571f | screen | vtm420@0.3 | 36.30 | 0.033 |
| ebfd571f | screen | vtmscc@0.075 | 29.00 | 0.285 |
| ebfd571f | screen | vtmscc@0.15 | 34.41 | 0.083 |
| ebfd571f | screen | vtmscc@0.3 | 43.70 | 0.016 |
# Bake-off sơ bộ, 2026-10-06 (Kaggle session A: T4, session B: CPU), commit 137158d

Nguồn: `2026-10-06_sessiona/` (msillm, mbt, s1res; Q̂ v0 nay ở `2026-10-06_qhat_v0/`), `2026-10-06_sessionb/` (vtm420: chỉ có byte, không có metrics).
Session C (vtmscc, CPU) hết 11 h sau 55/210 điểm (qp26–30); output không được giữ lại.

Q̂ v0: train trên 2021t/2021v/2022t, test trên 2024t đạt **0.812** (trần thực tế ~0.72–0.78 = độ đồng ý giữa hai người chấm
trên câu lặp; 2024t có thể dễ hơn). Trọng số lớn nhất: MS-SSIM, DISTS; PSNR mang dấu âm khi đã có các thước đo khác.

28/30 ảnh (thiếu `f67f11de`, `ff32adfa`: metrics chưa xong khi hết giờ), ngân sách co theo số pixel, phân bổ knapsack theo Q̂:

| rate | cand | bpp | PSNR | MS-SSIM | LPIPS | DISTS | Q̂ | t_dec T4 (s) |
|---|---|---|---|---|---|---|---|---|
| 0.075 | msillm | 0.0748 | 24.92 | 0.9118 | 0.0970 | 0.0966 | -0.541 | 1.95 |
| 0.075 | s1res | 0.0749 | 19.59 | 0.7655 | 0.2465 | 0.1987 | -2.805 | 8.96 |
| 0.075 | mbt | không đạt được (điểm rẻ nhất 0.131 bpp) | | | | | | |
| 0.075 | mix | 0.0748 | 23.16 | 0.8933 | 0.1014 | 0.0942 | -0.503 | 2.43 |
| 0.15 | msillm | 0.1498 | 26.88 | 0.9470 | 0.0608 | 0.0633 | 0.473 | 1.99 |
| 0.15 | mbt | 0.1499 | 27.21 | 0.9463 | 0.2727 | 0.1748 | -2.275 | 0.75 |
| 0.15 | s1res | 0.1498 | 20.90 | 0.8460 | 0.1249 | 0.0956 | -0.502 | 8.89 |
| 0.3 | msillm | 0.2996 | 29.14 | 0.9722 | 0.0350 | 0.0376 | 1.344 | 2.01 |
| 0.3 | mbt | 0.2991 | 29.23 | 0.9698 | 0.1863 | 0.1227 | -0.653 | 0.76 |
| 0.3 | s1res | 0.2772 | 24.18 | 0.9048 | 0.0983 | 0.0826 | -0.023 | 8.93 |

`mix` chọn msillm cho 26/28 ảnh ở 0.075 (2 ảnh sang s1res) và 28/28 ở 0.15 và 0.3.

Kết luận:
- **MS-ILLM là base mạnh nhất ở cả ba rate**: cách s1res 2.3 (0.075), 1.0 (0.15) và 1.4 (0.3) đơn vị Q̂, PSNR hơn 3–5 dB, decode nhanh gấp 4–5 lần.
- C1 (DC-AE + S1 + residual VTM) thua xa: S1 trên ảnh CLIC tốn bpp gấp ~2 lần so với Kodak (L5 = 0.17 bpp), PSNR base ≤ 21.6 dB.
- mbt2018-mean chỉ là mốc MSE: PSNR ngang msillm nhưng LPIPS/DISTS kém gấp 3–4 lần.

Bytes vtm420 theo QP (corpus bpp): qp26 0.795, qp30 0.517, qp34 0.330, qp38 0.201, qp42 0.116, qp46 0.061, qp50 0.029.
Vì vậy qp26/30 bị bỏ khỏi lưới (`VTM_QPS = 34 37 40 43 46 50`).

Lỗi pipeline: `pack` bỏ `recon/`, nên một session timeout giữa bước points và metrics làm mất mọi điểm chưa có metrics
(toàn bộ vtm420, 39 điểm của session A). Đã sửa: metrics được tính ngay khi mỗi điểm vừa xong; khi resume, điểm nào
không có cả recon lẫn metrics sẽ được làm lại.

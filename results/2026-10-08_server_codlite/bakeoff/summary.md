# So sánh base trên 30 ảnh validation CLIC 2027

Ngân sách cả bộ ảnh (byte): {'0.075': 808550, '0.15': 1617100, '0.3': 3234201}. Mỗi ảnh chọn một điểm vận hành sao cho tổng byte ≤ ngân sách và tổng mục tiêu lớn nhất; mục tiêu: **qhat_v0**.

Q̂ là tiện ích tuyến tính của Q̂ v0 (cao hơn = người chấm thích hơn; chênh 1.0 ≈ odds 2.7 lần). `t_dec`: giây/ảnh trên GPU của lần chạy này (không gồm VTM). `mix`: mỗi ảnh được chọn mode tốt nhất.

## 0.075 bpp — all

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 30 | 0.0749 | 22.02 | 0.8579 | 0.1091 | 0.0934 | -0.382 | 0.12 |
| mix | 30 | 0.0749 | 22.02 | 0.8579 | 0.1091 | 0.0934 | -0.382 | 0.12 |

## 0.075 bpp — natural

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 25 | 0.0756 | 23.04 | 0.8713 | 0.1093 | 0.0912 | -0.375 | 0.11 |
| mix | 25 | 0.0756 | 23.04 | 0.8713 | 0.1093 | 0.0912 | -0.375 | 0.11 |

## 0.075 bpp — screen

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 4 | 0.0884 | 21.93 | 0.9380 | 0.0653 | 0.0743 | 0.540 | 0.10 |
| mix | 4 | 0.0884 | 21.93 | 0.9380 | 0.0653 | 0.0743 | 0.540 | 0.10 |

## 0.15 bpp — all

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 30 | 0.1497 | 23.87 | 0.9206 | 0.0798 | 0.0697 | 0.285 | 0.13 |
| mix | 30 | 0.1497 | 23.87 | 0.9206 | 0.0798 | 0.0697 | 0.285 | 0.13 |

## 0.15 bpp — natural

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 25 | 0.1432 | 24.56 | 0.9188 | 0.0822 | 0.0691 | 0.256 | 0.13 |
| mix | 25 | 0.1432 | 24.56 | 0.9188 | 0.0822 | 0.0691 | 0.256 | 0.13 |

## 0.15 bpp — screen

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 4 | 0.0884 | 21.93 | 0.9380 | 0.0653 | 0.0743 | 0.540 | 0.10 |
| mix | 4 | 0.0884 | 21.93 | 0.9380 | 0.0653 | 0.0743 | 0.540 | 0.10 |

## 0.3 bpp — all

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 30 | 0.2996 | 25.77 | 0.9518 | 0.0583 | 0.0536 | 0.906 | 0.16 |
| mix | 30 | 0.2996 | 25.77 | 0.9518 | 0.0583 | 0.0536 | 0.906 | 0.16 |

## 0.3 bpp — natural

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 25 | 0.2935 | 26.78 | 0.9503 | 0.0600 | 0.0536 | 0.863 | 0.16 |
| mix | 25 | 0.2935 | 26.78 | 0.9503 | 0.0600 | 0.0536 | 0.863 | 0.16 |

## 0.3 bpp — screen

| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |
|---|---|---|---|---|---|---|---|---|
| codlite | 4 | 0.2814 | 23.84 | 0.9748 | 0.0428 | 0.0500 | 1.409 | 0.14 |
| mix | 4 | 0.2814 | 23.84 | 0.9748 | 0.0428 | 0.0500 | 1.409 | 0.14 |

## Ảnh nào được `mix` giao cho base nào

| rate | vtm420 | vtmscc | s1res | msillm | mbt | codlite | turbo | coolchic |
|---|---|---|---|---|---|---|---|---|
| 0.075 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| 0.15 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| 0.3 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 |

Kích thước decoder (fp32): codlite 224 MB

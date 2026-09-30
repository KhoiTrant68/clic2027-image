# Nhánh B: kết quả

## PSNR toàn tập theo cách CLIC (MSE gộp, trọng số theo số pixel)

Mốc so sánh trên leaderboard @0.075: HM 25.83, VTM 26.52, Vcoder (hạng 1) 23.99

| cfg | PSNR |
|---|---|
| dcae | 22.86 |
| hevc@0.075 | 24.58 |
| hevc@0.15 | 25.98 |
| hevc@0.3 | 28.23 |
| sdvae | 24.73 |

## Nhóm: all

| cfg | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ (n ảnh) |
|---|---|---|---|---|---|---|
| dcae | 25.54 | 0.8978 | 0.0821 | 0.0598 | 20.42 | 0.672 (13) |
| hevc@0.075 | 27.51 | 0.8929 | 0.3401 | 0.1895 | 23.30 | 0.641 (13) |
| hevc@0.15 | 29.64 | 0.9247 | 0.2587 | 0.1464 | 26.53 | 0.420 (13) |
| hevc@0.3 | 32.33 | 0.9546 | 0.1720 | 0.1017 | 30.33 | 0.269 (13) |
| sdvae | 26.79 | 0.9369 | 0.0727 | 0.0552 | 23.64 | 0.461 (13) |

## Nhóm: natural

| cfg | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ (n ảnh) |
|---|---|---|---|---|---|---|
| dcae | 24.63 | 0.8782 | 0.0947 | 0.0657 | 20.59 | 0.867 (8) |
| hevc@0.075 | 26.15 | 0.8742 | 0.3925 | 0.2058 | 21.82 | 0.913 (8) |
| hevc@0.15 | 27.76 | 0.9094 | 0.3068 | 0.1638 | 24.27 | 0.637 (8) |
| hevc@0.3 | 29.99 | 0.9447 | 0.2074 | 0.1169 | 27.35 | 0.424 (8) |
| sdvae | 25.66 | 0.9248 | 0.0843 | 0.0607 | 23.31 | 0.656 (8) |

## Nhóm: screen

| cfg | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ (n ảnh) |
|---|---|---|---|---|---|---|
| dcae | 28.51 | 0.9624 | 0.0405 | 0.0401 | 20.16 | 0.361 (5) |
| hevc@0.075 | 32.00 | 0.9541 | 0.1679 | 0.1358 | 25.67 | 0.206 (5) |
| hevc@0.15 | 35.80 | 0.9749 | 0.1007 | 0.0893 | 30.14 | 0.074 (5) |
| hevc@0.3 | 40.01 | 0.9870 | 0.0557 | 0.0515 | 35.09 | 0.022 (5) |
| sdvae | 30.50 | 0.9765 | 0.0346 | 0.0369 | 24.17 | 0.148 (5) |

## Kiểm tra điều kiện B3 (HEVC dùng thay VTM, cộng thêm 0.7 dB)

- [screen/game] CER trần của DC-AE ≤ CER HEVC@0.075: **False**
- [screen/game] PSNR trần của DC-AE ≥ VTM@0.3 − 1 dB: **False**
- [screen/game] CER trần của f8 ≤ CER HEVC@0.075: **True**
- [natural] PSNR trần của DC-AE < VTM@0.3 − 2 dB (cần residual ở rate cao): **True**

→ **Thêm nhánh residual / latent f8** cho ảnh có chữ và ảnh màn hình.
→ **Ảnh tự nhiên ở 0.15–0.3 bpp cần đường residual** (cũng ảnh hưởng tới CVPR).

*Chú ý: `label` trong inventory.csv cần được xác nhận bằng mắt. Nhãn hiện tại có thể chỉ là nhãn gợi ý tự động.*
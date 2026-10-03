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
| dcae | 25.77 | 0.9035 | 0.0844 | 0.0594 | 22.08 | 0.714 (9) |
| hevc@0.075 | 27.44 | 0.8920 | 0.3688 | 0.2001 | 22.95 | 0.740 (9) |
| hevc@0.15 | 29.42 | 0.9254 | 0.2833 | 0.1560 | 25.61 | 0.523 (9) |
| hevc@0.3 | 31.95 | 0.9549 | 0.1895 | 0.1098 | 28.80 | 0.349 (9) |
| sdvae | 26.81 | 0.9397 | 0.0735 | 0.0543 | 25.09 | 0.519 (9) |

## Nhóm: screen

| cfg | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ (n ảnh) |
|---|---|---|---|---|---|---|
| dcae | 26.55 | 0.9496 | 0.0485 | 0.0438 | 16.68 | 0.578 (4) |
| hevc@0.075 | 30.04 | 0.9352 | 0.1747 | 0.1451 | 24.10 | 0.418 (4) |
| hevc@0.15 | 33.62 | 0.9647 | 0.0985 | 0.0976 | 28.61 | 0.189 (4) |
| hevc@0.3 | 37.66 | 0.9818 | 0.0591 | 0.0622 | 33.77 | 0.088 (4) |
| sdvae | 29.14 | 0.9736 | 0.0387 | 0.0405 | 20.39 | 0.331 (4) |

## Nhóm: texture

| cfg | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ (n ảnh) |
|---|---|---|---|---|---|---|
| dcae | 15.75 | 0.5497 | 0.1571 | 0.1319 | nan | nan (0) |
| hevc@0.075 | 19.08 | 0.7454 | 0.2830 | 0.1024 | nan | nan (0) |
| hevc@0.15 | 19.08 | 0.7454 | 0.2830 | 0.1024 | nan | nan (0) |
| hevc@0.3 | 20.37 | 0.8390 | 0.1850 | 0.0566 | nan | nan (0) |
| sdvae | 16.94 | 0.7192 | 0.1886 | 0.1363 | nan | nan (0) |

## Kiểm tra điều kiện B3 (HEVC dùng thay VTM, cộng thêm 1.9 dB)

- [screen/game] CER trần của DC-AE ≤ CER HEVC@0.075: **False**
- [screen/game] PSNR trần của DC-AE ≥ VTM@0.3 − 1 dB: **False**
- [screen/game] CER trần của f8 ≤ CER HEVC@0.075: **True**
- [natural] PSNR trần của DC-AE < VTM@0.3 − 2 dB (cần residual ở rate cao): **True**

→ **Thêm nhánh residual / latent f8** cho ảnh có chữ và ảnh màn hình.
→ **Ảnh tự nhiên ở 0.15–0.3 bpp cần đường residual** (cũng ảnh hưởng tới CVPR).

*Chú ý: `label` trong inventory.csv cần được xác nhận bằng mắt. Nhãn hiện tại có thể chỉ là nhãn gợi ý tự động.*
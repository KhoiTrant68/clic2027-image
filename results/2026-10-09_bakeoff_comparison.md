# Bake-off trên server L40, 2026-10-09: Q̂ v0 so với Q̂ v1

Cả 5 ứng viên chạy lại trong cùng một môi trường (server L40, commit 46a1134), 30 ảnh validation, có đặc trưng
Wasserstein: `2026-10-09_server_full_gpu` (MS-ILLM, mbt, CoD-Lite, 57 phút) và `2026-10-09_server_full_vtm`
(VTM 4:2:0, VTM-SCC trên cả 30 ảnh, 24 CPU, 115 phút). Gộp và chia bit hai lần: `2026-10-09_bakeoff_v0` (Q̂ v0) và
`2026-10-09_bakeoff_v1` (Q̂ v1 = wd3 + MS-SSIM). Thang đo của hai bản khác nhau: chỉ so thứ hạng trong một cột.

| rate | ứng viên | Q̂ v0 | Q̂ v1 | PSNR | giải mã L40 (s/ảnh) |
|---|---|---|---|---|---|
| 0.075 | VTM 4:2:0 | -2.52 | 1.83 | 26.0 | CPU |
| 0.075 | VTM-SCC | -2.40 | 1.90 | 26.2 | CPU |
| 0.075 | MS-ILLM | -0.45 | **2.15** | 25.2 | 0.8 |
| 0.075 | CoD-Lite | **-0.38** | 1.85 | 22.0 | 0.15 |
| 0.075 | mix | -0.17 | 2.18 | | |
| 0.15 | VTM-SCC | -1.10 | 2.38 | 28.1 | |
| 0.15 | MS-ILLM | **+0.55** | **2.62** | 27.2 | |
| 0.15 | CoD-Lite | +0.29 | 2.23 | 23.9 | |
| 0.15 | mbt | -2.01 | 2.02 | 27.4 | 0.3 |
| 0.3 | VTM-SCC | 0.00 | 2.83 | 31.7 | |
| 0.3 | MS-ILLM | **+1.38** | **3.17** | 29.4 | |
| 0.3 | CoD-Lite | +0.91 | 2.68 | 25.8 | |

`mix` ở 0.075: theo v0 thì 14 ảnh CoD-Lite, 14 MS-ILLM, 2 VTM-SCC; theo v1 thì 25 MS-ILLM, 3 CoD-Lite, 2 VTM-SCC.

Kết luận:
- Hai bản đồng ý: MS-ILLM tốt nhất ở 0.15 và 0.3 bpp, và hơn VTM ở mọi rate.
- Hai bản lệch nhau ở 0.075: v1 xếp VTM ngang CoD-Lite và chỉ kém MS-ILLM 0.3. Ở CLIC 2025 (0.075 bpp), VTM được
  1405 Elo, đội thắng 1929 với PSNR thấp hơn 2.5 dB. v0 khớp với khoảng cách đó, v1 thì không: wd3 (log2 sigma 3)
  tương quan 0.96 với PSNR trên dữ liệu rating nên v1 gần như không phạt ảnh mờ. **Mặc định quay về Q̂ v0**
  (pipeline `--qhat-model` để chọn bản khác); v1 chỉ dùng tham khảo.
- MS-ILLM hay CoD-Lite ở 0.075: chưa phân định được bằng Q̂; cần tự chấm bằng mắt.

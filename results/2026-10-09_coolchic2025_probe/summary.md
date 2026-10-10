# Cool-chic tag clic2025 (bài Orange nộp CLIC 2025), thử trên server L40, 09/10/2026

Code: `experiments/bakeoff/bakeoff.py` ứng viên `coolchic` (tag clic2025 @ a229849, decoder C/AVX2; `setup.py` của
tag thiếu `common_randomness.cpp` cho bản AVX2, bake-off tự thêm vào).

## Decoder C

| file | AVX2 (1 lõi) | CPU thường | ghi chú |
|---|---|---|---|
| mẫu a365_wd.cool (Orange) | 1.24 s | 2.02 s | ảnh 2048x1360 |
| mẫu a365_mse.cool (Orange) | 0.52 s | 1.64 s | |
| 2684452d, decoder `lop` (bên dưới) | 1.1 s | | 694 MAC/pixel |
| 937476dd, decoder `lop` (lần chạy thử bake-off) | 1.39 s | | 2048x1536 |

Thư viện decoder chỉ cần libstdc++/libc, không dùng torch: chạy được trên server CLIC (torch 2.6). Ảnh AVX2 và CPU
lệch tối đa 1 mức xám.

## Encode, `cfg/enc/intra/perceptive.cfg` (tune wasserstein = 0.8 MSE + 0.2 WD, 10k vòng + 2.8k warm-up)

Ảnh 2684452d (2048x1360), λ 0.004, decoder `lop`, GPU 0 L40:
- thời gian 1 h 39 min (warm-up ~4 min, train ~0.45 s/vòng, quantize 155 s); giới hạn 20 GB thì OOM khi bắt đầu
  pha Wasserstein, 28 GB thì chạy được (bản 5.0: đỉnh 21 GB).
- rate 0.557 bpp (λ 0.004 ở bản 5.0 cho 0.111 bpp: thang λ khác vì trọng số MSE/WD khác) → muốn 0.075–0.3 bpp
  phải dùng λ lớn hơn nhiều, khoảng 0.02–0.2.

Q̂ trên cùng ảnh (cc_score.py):

| điểm | bpp | PSNR | Q̂ v0 | Q̂ v1 |
|---|---|---|---|---|
| MS-ILLM q4 | 0.267 | 31.77 | +1.46 | +2.85 |
| MS-ILLM q5 | 0.385 | 33.08 | +1.77 | +3.17 |
| CoD-Lite 0_5000 | 0.518 | 31.30 | +1.98 | +3.18 |
| **Cool-chic clic2025** | **0.557** | 32.67 | **+1.33** | +3.53 |

Lần chạy thử bake-off (2000 vòng, λ 0.004): texture 937476dd ra 1.25 bpp, ngoài khoảng của CLIC; Q̂ v0 kém VTM-SCC
0.54 ở cùng bpp.

## Kết luận

Ba phép đo độc lập cùng một hướng (Cool-chic 5.0, file Orange tự encode a365, bản clic2025 trên 2684452d): ở cùng
rate Cool-chic kém MS-ILLM khoảng 0.4–0.6 Q̂ v0 trên ảnh tự nhiên; ở 2684452d, Cool-chic 0.56 bpp còn thua MS-ILLM
0.27 bpp. Q̂ v1 xếp Cool-chic cao hơn, nhưng v1 đã trượt phép kiểm tra với Elo thật (thưởng PSNR). Encode mất
~1.5 h/ảnh/λ trên L40. Còn đáng thử: ảnh màn hình và texture (nơi codec học từ ảnh tự nhiên yếu), ở λ 0.02–0.2.

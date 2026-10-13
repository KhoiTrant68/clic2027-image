# Chưng cất decoder DC-AE của DiT-IC (q3), 11/10/2026

Mục tiêu: giải mã nhanh hơn mà giữ chất lượng DiT-IC, vì đó là phần đã thắng 69% khi tự chấm mù. Code: `experiments/ditic_distill/`.

## Cách làm

- **Thầy:** decoder DC-AE đã gộp LoRA của checkpoint DiT-IC q3 (bf16).
- **Học trò:** cùng cấu trúc, nhưng ba tầng ResBlock độ phân giải cao còn một nửa số kênh (512/256/128 → 256/128/64).
  `conv_in` và ba tầng EfficientViT (1/32 … 1/8) được chép từ thầy và đóng băng. Thầy và trò dùng chung phần đầu này,
  nên chỉ 6.2 M tham số được train.
- **Dữ liệu:** latent mà DiT-IC thật sinh ra cho 800 ảnh DIV2K train (encoder → rANS → DiT một bước bf16).
  Mỗi bước lấy cắt ngẫu nhiên 384×384, batch 4. Không dùng ảnh gốc: học trò chỉ học bắt chước đầu ra của thầy.
- **Đợt 1:** L1 + LPIPS(VGG), 20 000 bước.
- **Đợt 2:** khởi tạo từ đợt 1, thêm PatchGAN (spectral norm, hinge; đầu ra của thầy là "thật"), 10 000 bước.
- **Chấm:** 30 ảnh validation CLIC, cùng bitstream q3 (0.076 bpp), so với ảnh gốc; giải mã cả khung hình + gom kích thước.

## Chất lượng (30 ảnh, so với ảnh gốc)

| decoder | PSNR | LPIPS | DISTS | WD3 | Q̂ v0 | trò − thầy |
|---|---|---|---|---|---|---|
| thầy | 24.70 | 0.083 | 0.069 | 0.214 | **+0.157** | |
| đợt 1: L1 + LPIPS, 20k | 25.25 | 0.111 | 0.084 | 0.263 | −0.269 | −0.43 (30/30 ảnh kém hơn) |
| đợt 2: GAN 0.1, 10k | 24.85 | 0.101 | 0.080 | 0.254 | −0.086 | −0.24 (27/30) |
| **đợt 2: GAN 0.1, chạy tiếp tới 30k** (khởi động lại LR ở 10k) | 24.77 | 0.095 | 0.077 | 0.243 | **−0.012** | **−0.17** (19/30) |
| đợt 2: GAN 0.3, 10k | 24.60 | 0.109 | 0.086 | 0.283 | −0.240 | −0.40 |
| *MS-ILLM ở 0.075 (bake-off, để so)* | | | | | *−0.45* | |

- Đợt 1 làm mịn ảnh: PSNR cao hơn thầy, nhưng mọi thước đo cảm nhận đều kém.
- GAN 0.1 thu hẹp khoảng cách từ 0.43 xuống 0.24 Q̂ sau 10k bước:
  −0.639 / −0.322 / −0.215 / −0.112 / −0.086 tại 2k / 4k / 6k / 8k / 10k.
- Chạy tiếp tới 30k bước: −0.208 / −0.112 / −0.058 / −0.023 / −0.012 tại 14k / 18k / 22k / 26k / 30k.
  Khoảng cách còn **0.17 Q̂ (≈ −29 Elo)**, đang chậm lại, có vẻ gần bão hoà.
- GAN 0.3 kém hơn GAN 0.1.
- Học trò 30k hơn MS-ILLM ≈ 0.44 Q̂. Đã chép sang `~/clic/distill/students/q3.pt` cho ứng viên bake-off `ditics`.

## Tốc độ DC-AE (30 ảnh, giải mã cả khung hình, bf16, GPU L40 lúc rảnh)

| decoder | mỗi ảnh | 30 ảnh | so với thầy |
|---|---|---|---|
| thầy | 0.43 s | 12.9 s | |
| học trò 64/128/256 | 0.24 s | 7.3 s | 1.77× |
| học trò rộng 96/192/384 (chưa train) | 0.35 s | 10.4 s | 1.22× (không đáng train) |

Lưu ý: các số trong `2026-10-11_ditic_speed.md` đo lúc server bận và chậm hơn khoảng 2.2 lần (thầy 0.96 s so với 0.43 s).
Lúc rảnh, cả decoder (entropy + DiT + DC-AE) khoảng 17–18 s cho 30 ảnh với thầy và ~11 s với học trò, trên L40.
**L4 vẫn chưa đo.**

## Cả ba mức q (13/10)

Cùng công thức cho q4 và q2: 20k bước L1 + LPIPS, rồi 30k bước GAN 0.1. Snapshot 30k là tốt nhất ở cả ba mức.
Mỗi mức tính trên cả 30 ảnh ở đúng mức q đó, Q̂ v0:

| mức | thầy | học trò | trò − thầy | số ảnh kém hơn > 0.1 |
|---|---|---|---|---|
| q2 | −0.450 | −0.584 | −0.13 | 14/30 |
| q3 | +0.157 | −0.012 | −0.17 | 19/30 |
| q4 | +0.272 | +0.115 | −0.16 | 17/30 |

## Bake-off `ditics` (13/10, `2026-10-13_bakeoff_v0`)

Thầy (`ditic`) và trò (`ditics`) chạy lại cùng lúc với cấu hình mặc định (cả khung hình, gom 16), cùng bitstream. Gộp với các
ứng viên cũ rồi chia bit theo Q̂ v0. Ở **0.075 bpp**:

| ứng viên | Q̂ v0 | LPIPS | DISTS | giải mã cả decoder (s/ảnh, L40) |
|---|---|---|---|---|
| MS-ILLM | −0.451 | 0.094 | 0.094 | 0.81 |
| DiT-IC thầy | **+0.164** | 0.083 | 0.069 | 0.55 (16.5 s / 30 ảnh) |
| DiT-IC học trò | **−0.000** | 0.094 | 0.076 | **0.36** (10.8 s / 30 ảnh) |
| mix | +0.184 | | | |

- **Học trò so với thầy:** kém 0.16 Q̂ (≈ −29 Elo), giải mã nhanh gấp 1.5 lần. Vẫn hơn MS-ILLM 0.45 Q̂ (≈ +78 Elo).
- **Ở 0.15 bpp:** `mix` giao 16 ảnh cho DiT-IC (thầy), 11 cho MS-ILLM; mix +0.758 so với MS-ILLM +0.545.
- **Chọn thầy hay trò:** phụ thuộc thời gian giải mã thật trên L4 và luật "25% chậm nhất không được giải". Cần đo trên L4.

## Tiếp theo

- **Đo trên L4:** thời gian giải mã của thầy và trò, đồng thời kiểm tra nén và giải giữa hai máy L4
  (xem `2026-10-12_ditic_determinism/summary.md`).
- Học trò đã sẵn ở `~/clic/distill/students/{q2,q3,q4}.pt` trên server (snapshot 30k). Ứng viên bake-off: `ditics`.
- Kết quả từng snapshot: `2026-10-11_ditic_distill/*/`.

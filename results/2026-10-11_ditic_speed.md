# Tăng tốc decoder DiT-IC, 11/10/2026

Mục tiêu: decoder DiT-IC (entropy decode → DiT một bước → DC-AE decoder) đủ nhanh cho track GPU. Đội top 2025 mất 21–29 s
cho 30 ảnh trên L4, và 25% bài chậm nhất mỗi track không được giải.

Đo trên server L40, giới hạn 10 GB/GPU. Server dùng chung (người khác chiếm ~7 GB mỗi GPU), nên thời gian tuyệt đối dao động
theo giờ. Mọi so sánh dưới đây **chạy xen kẽ trong cùng một tiến trình** và lấy min của 3 lần.
- Script: `experiments/ditic_speed/dcae_bench*.py` (vòng 1–5). Chép vào `/tmp` trên server rồi chạy với cwd = checkout DiT-IC, `PYTHONPATH=.`; các vòng sau `exec` phần đầu của vòng trước.
- Log rút gọn: `2026-10-11_server_ditic_fast/bench_logs.txt`.

## Thời gian đi đâu (vòng 3, profiler, bf16, tile 1280)

- Convolution chỉ chiếm ~40% thời gian GPU.
- Phần còn lại là các phép theo từng phần tử (ép kiểu, cộng, nhân, bình phương). Nguyên nhân là `RMSNorm` của diffusers trong
  mỗi ResBlock: nó đổi trục kênh ra cuối rồi tạo bản sao fp32 của cả tensor 2K, và phép nhân cho ra fp32 trước khi ép về bf16.

## Các bước tối ưu (không train lại)

| bước | DC-AE, ảnh 2048×1360 | chất lượng |
|---|---|---|
| fp32, tile 1024 / chồng 128 (bake-off 10/10) | 3.58 s | mốc |
| autocast bf16 (thử 10/10) | chậm hơn | như cũ |
| **weights bf16** (không dùng autocast) | 1.34 s | 47.6 dB so với fp32 |
| channels_last, cudnn.benchmark | không đổi | |
| tile 1280 / chồng 128 | 1.18–1.37 s (tuỳ lần đo) | so với ảnh gốc: như tile 1024 (LPIPS 0.0647 so với 0.0640) |
| **RMSNorm NCHW**: tổng bình phương tích lũy fp32 trong một phép rút gọn, scale và residual làm tại chỗ (`addcmul`) | 0.92 s (−22%) | 53 dB so với bản trước |
| **cắt latent theo ảnh** (+2 hàng/cột) thay vì đệm lên bội số 256 | −7% | không đổi |
| fp16 thay bf16 | không nhanh hơn | |
| không chia tile (cả khung hình) | ≈ −15% trên 3 ảnh | ảnh 2048×2048 vượt 10 GB (L4 có 24 GB) |

## Cả decoder (vòng 5, 8 ảnh, xen kẽ)

| cấu hình | entropy | DiT | DC-AE | tổng | LPIPS / DISTS so với ảnh gốc |
|---|---|---|---|---|---|
| bake-off 10/10: fp32, tile 1024 | 0.09 | 0.28 | 2.57 | **2.94 s** | 0.0719 / 0.0602 |
| bf16 + RMSNorm mới + tile 1280 + cắt latent | 0.09 | 0.10 | 0.91 | **1.10 s** | 0.0724 / 0.0608 |

**Nhanh gấp 2.7 lần.**

Đường giải mã mới đã nằm trong bake-off (`ditic`; lúc đó mặc định `--ditic-dtype bf16 --ditic-tile 1280 --ditic-overlap 128`,
mặc định mới ở vòng 8 bên dưới).
Chạy lại cả 30 ảnh × 3 điểm (`2026-10-11_server_ditic_fast`) so với lần fp32 ngày 10/10:
- DISTS 0.0733 → 0.0737, PSNR −0.05 dB, Q̂ v0 trung bình −0.010 (≈ 2 Elo);
- bytes không đổi.

## Hai tối ưu còn lại (vòng 6)

- **Tách conv sau upsample** (nearest ×2 rồi conv 3×3 → bốn conv 2×2 ở độ phân giải thấp, ghép bằng `pixel_shuffle`).
  Đúng tuyệt đối: so với khối gốc ở fp64 chỉ lệch 1e-7 tương đối, kể cả ở viền. Nhưng **chậm hơn**: DC-AE 1.16 s so với 1.02 s,
  bộ nhớ 4.7 GB so với 3.9 GB. Conv 3×3 của cuDNN đã tối ưu rất kỹ; bốn conv 2×2 cộng phần ghép tốn hơn số FLOPs tiết kiệm
  được. **Bỏ.**
- **Giải mã cả khung hình:** giờ lọt dưới 10 GB (đỉnh 6.0–6.4 GB, kể cả ảnh 2048×2048, nhờ RMSNorm mới).
  Khi đo nóng thì nhanh hơn 7–8%; LPIPS 0.0724 → 0.0731, có lẽ vì decoder được tinh chỉnh trên ảnh cắt nhỏ.

## Đo đúng kiểu CLIC (vòng 8)

Các vòng trên lấy min của 3 lần chạy, tức là đo khi đã "nóng". CLIC giải mã mỗi ảnh **một lần**, mà mỗi ảnh một kích thước:
mỗi hình dạng tensor mới khiến cuDNN phải dựng lại kế hoạch cho ~60 lớp conv. Vòng 8 đo đúng cách đó: một tiến trình mới,
nạp model, giải mã lần lượt 30 ảnh (bitstream q3, 808 804 byte), mỗi ảnh một lần.

| cấu hình | số hình dạng latent | ảnh đầu | ảnh 2–30 (trung bình) | tổng 30 ảnh | LPIPS / DISTS (8 ảnh) |
|---|---|---|---|---|---|
| tile 1280 (mặc định trước đó) | 11 | 9.9 s | 2.81 s | 91 s | 0.0763 / 0.0684 |
| tile 1280 + gom kích thước | 4 | 9.8 s | 2.13 s | 72 s | 0.0764 / 0.0685 |
| cả khung hình | 11 | 4.3 s | 2.16 s | 67 s | 0.0770 / 0.0687 |
| **cả khung hình + gom kích thước** | 4 | 4.2 s | **1.87 s** | **58.5 s** | 0.0770 / 0.0688 |

- **Gom kích thước:** latent đã cắt được đệm (replicate) lên bội số của 16 latent (512 px), nên 30 ảnh chỉ còn 4 hình dạng.
- **Chi phí khởi động:** giải mã lại đúng ảnh vừa xong mất 1.53 s, so với 2.16 s lần đầu, tức ~0.6 s cho mỗi kích thước mới.
  Lúc đo, server đang bận hơn vòng 5 (bản nóng 1.53 s so với 0.95 s).
- Nạp model mất ~20 s, nhưng đó là dựng mạng fp32 từ checkpoint 4.3 GB. Bản nộp sẽ lưu sẵn bf16.

**Mặc định mới của `ditic`:** cả khung hình và gom 16 (`--ditic-tile 0 --ditic-bucket 16`). Nhanh gấp 1.56 lần tile 1280
khi giải mã mỗi ảnh một lần, đổi lại LPIPS tăng ~1%. Encoder (fp32, không tính giờ) vẫn chia tile 1024.

## Biên dịch decoder (vòng 7)

Trên một tile 1280×1280: `torch.compile` chế độ mặc định nhanh gấp **1.45 lần** (0.454 → 0.312 s, 47.8 dB so với bản thường),
biên dịch mất 34 s. Chế độ max-autotune chỉ nhanh gấp 1.09 lần mà biên dịch mất 29 phút.

Luật CLIC tính cả thời gian biên dịch, nên muốn dùng thì phải biên dịch sẵn bằng **AOTInductor** thành `.so` rồi chỉ việc nạp.
Điều kiện: biên dịch bằng đúng torch của server CLIC (2.6) trên GPU sm_89 (L40 và L4 đều là sm_89), với hình dạng cố định,
và việc gom kích thước ở trên giúp giữ số hình dạng ít. Chưa làm.

## Còn thiếu bao nhiêu

- **L40, kiểu CLIC:** 58.5 s cho 30 ảnh lúc server bận. Lúc rảnh có lẽ ~35–40 s (bản nóng chậm hơn vòng 5 khoảng 1.6 lần).
- **L4:** chưa đo, và sẽ chậm hơn L40: băng thông bộ nhớ ~1/3, tensor bf16 ~2/3. Đội top 2025 mất 21–29 s.
  **Cần đo thật trên L4** (thuê ~1 giờ) bằng `experiments/ditic_speed/dcae_bench8.py`.
- **Các tối ưu không cần train coi như đã hết.** Hai hướng còn lại:
  1. AOTInductor: ~1.45 lần cho DC-AE.
  2. **Đổi kiến trúc ở đúng chỗ chậm:** chưng cất (distill) một decoder DC-AE nhẹ hơn, ít kênh hơn ở hai tầng độ phân giải
     cao nhất (~60% FLOPs). Decoder mới học từ chính DC-AE của DiT-IC trên latent của DiT-IC (loss L1 + LPIPS + GAN).
     Phần DiT và latent, tức phần đã thắng 69% khi chấm mù, giữ nguyên.

  Hai hướng gộp lại có thể nhanh thêm ~3 lần, đủ để vào mức 21–29 s trên L4. Đây là ước lượng, chưa đo.

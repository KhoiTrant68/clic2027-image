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

Đường giải mã mới đã nằm trong bake-off (`ditic`, mặc định `--ditic-dtype bf16 --ditic-tile 1280 --ditic-overlap 128`).
Chạy lại cả 30 ảnh × 3 điểm (`2026-10-11_server_ditic_fast`) so với lần fp32 ngày 10/10:
- DISTS 0.0733 → 0.0737, PSNR −0.05 dB, Q̂ v0 trung bình −0.010 (≈ 2 Elo);
- bytes không đổi.

## Còn thiếu bao nhiêu

- **L40:** 1.10 s/ảnh lúc server bận, tức ≈ 33 s cho 30 ảnh. Lúc server rảnh có thể nhanh hơn: bản fp32 đo đêm 10/10 là
  1.40 s, còn trong phép đo xen kẽ hôm nay là 2.94 s.
- **L4:** chưa đo. Băng thông bộ nhớ của L4 bằng ~1/3 L40, tensor bf16 bằng ~2/3. Ước lượng 30 ảnh mất ~35–70 s, vẫn trên mức
  21–29 s của đội top. **Cần đo thật trên L4** (thuê khoảng 1 giờ).
- **Còn làm được mà không cần train:**
  - giải mã cả khung hình trên L4 (−15%);
  - tách conv sau bước upsample nearest ×2 thành 4 conv 2×2 trên độ phân giải thấp (giảm 56% FLOPs của 3 conv đó, ước −10%).
- **Nếu vẫn thiếu thì đổi kiến trúc ở đúng chỗ chậm:** chưng cất (distill) một decoder DC-AE nhẹ hơn, ít kênh hơn ở hai tầng
  độ phân giải cao nhất (chiếm ~60% FLOPs). Decoder mới học từ chính DC-AE của DiT-IC trên latent của DiT-IC (loss
  L1 + LPIPS + GAN). Phần DiT và latent, tức phần đã thắng 69% khi chấm mù, giữ nguyên.

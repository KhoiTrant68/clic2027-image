# Kit L4 chạy thử trên L40 của server (13/10/2026)

Kit: `experiments/l4_kit/`. Đây là cùng lệnh sẽ chạy trên L4 thuê, ở đây chạy trên GPU 1 của server (NVIDIA L40, driver 580).
Môi trường dựng theo devkit CLIC: Python 3.12.3, torch 2.6.0+cu124, cuDNN 9.1.0 (90100).
DiT-IC q3, 30 ảnh validation, nén 0.076 bpp.

## Tất định

- Tự kiểm L40 → L40 (nén rồi giải lại trong một tiến trình khác, cùng máy): **BIT-EXACT**.
  0 chỉ số scale lệch, 0 ký hiệu lệch trên 30 ảnh. PSNR trung bình 24.89 dB.
- Chạy lại với cache HuggingFace trống, chế độ offline: chỉ cần 3 file config của SANA và `q3_merge_ema.pt`.
- Lệnh `compare` (offline): hai lần nén trên cùng máy cho bitstream giống hệt.

## Tốc độ giải mã (CLIC-style: tiến trình mới, mỗi ảnh giải một lần)

| decoder | nạp mạng | ảnh đầu | ảnh 2..30 (trung bình) | giải 30 ảnh | peak VRAM |
|---|---|---|---|---|---|
| thầy (lần 1 / lần 2) | 18.8 / 17.3 s | 0.94 / 0.82 s | 0.570 / 0.568 s | 17.5 / 17.3 s | 5.5 GB |
| trò (lần 1 / lần 2) | 18.6 / 20.6 s | 0.62 / 0.74 s | 0.371 / 0.368 s | 11.4 / 11.4 s | 4.4 GB |

- Khớp bake-off ngày 13/10 (thầy 0.55 s/ảnh, trò 0.36 s/ảnh). Trò nhanh hơn 1.5 lần.
- `nạp mạng` là dựng mạng fp32 từ checkpoint 4.3 GB rồi đổi sang bf16. Bản nộp sẽ gói sẵn trọng số bf16.
- L4 chậm hơn L40, đặc biệt băng thông bộ nhớ: 300 GB/s so với 864 GB/s. Số trên L4 là mục tiêu của lần thuê.

Số liệu: `decode_self.json`, `speed_*.json`, log từng bước.

## Chạy lại từ chính file kit

Giải nén `~/clic/clic27_l4_kit.tgz` vào thư mục trống rồi chạy `run_all.sh`, đúng như trên L4 (môi trường đã dựng sẵn): 5 phút.
- Tự kiểm: BIT-EXACT. Giải `streams_l40.pt` đi kèm kit (nén ở lần chạy đầu): BIT-EXACT.
- Tốc độ lặp lại: thầy 0.572 / 0.566 s mỗi ảnh, trò 0.371 / 0.369 s.

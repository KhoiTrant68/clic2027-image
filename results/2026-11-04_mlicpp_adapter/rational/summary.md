# Pilot MLIC++ q2 — kết quả test crop256

160 ảnh train / 32 validation / 32 test độc lập theo ảnh cho adapter; chưa audit overlap pretraining.

| Nhánh | bpp thực | PSNR gộp | LPIPS-Alex ↓ | DISTS ↓ | Edge mean ↓ | Edge CVaR ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.19934 | 30.60 | 0.17350 | 0.20233 | 0.30518 | 0.49922 |
| mean | 0.19934 | 29.43 | 0.09806 | 0.14320 | 0.35236 | 0.52803 |
| cvar | 0.19934 | 29.49 | 0.09837 | 0.14214 | 0.35089 | 0.52501 |
| risk | 0.19934 | 29.53 | 0.10133 | 0.14397 | 0.34863 | 0.52654 |
| risk_hide_change | 0.19934 | 28.60 | 0.12538 | 0.20272 | 0.40031 | 0.54115 |

Ba nhánh × 1.000 bước, cùng khởi tạo/kiến trúc/data order; không GAN. Byte cố định, checkpoint chọn ở bước cuối.
96/96 ảnh giải mã test bằng tiến trình mới khớp pixel với đường cache-evaluation trên cùng L40.
CVaR theo vùng, scale từ train; transform=rational. Không so giá trị edge giữa hai transform.

## Kiểm tra bão hòa surrogate

- base: 0.0% vùng ở ngưỡng 1.
- mean: 0.0% vùng ở ngưỡng 1.
- cvar: 0.0% vùng ở ngưỡng 1.
- risk: 0.0% vùng ở ngưỡng 1.
- risk_hide_change: 0.0% vùng ở ngưỡng 1.

## Giới hạn

32 ảnh test, một seed, crop 256, legacy q2 với rate đo được; không phải kết quả full30 tại 0,15 bpp.
LPIPS-Alex có trong loss; DISTS là metric ngoài loss. Chưa có Q̂ đầy đủ, OCR transcript, human study, L4 hoặc PerQ/PICO reproduction.
Bootstrap 2.000 lần theo ảnh là kiểm tra thăm dò, không hiệu chỉnh nhiều phép so sánh và không thay thế chạy nhiều seed.
suffix_s không gồm entropy/load/I/O, không dùng để suy ra thời gian submission.
Checkpoint, stream, ảnh và cache lớn được giữ ngoài Git trên server.

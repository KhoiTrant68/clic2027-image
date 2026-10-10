# Rà soát model thay MS-ILLM: 2026-10-10

Câu hỏi: có codec công khai nào **tốt hơn MS-ILLM** ở 0.075 / 0.15 / 0.3 bpp, có **weights**, và **giải mã đủ nhanh** cho track GPU (L4,
đội top 2025 mất 21–29 s cho 30 ảnh) không?

Cách tìm:
- Quét 150 bài arXiv mới nhất về nén ảnh generative / perceptual (tới 05/10/2026).
- Đọc README và số liệu của các repo GitHub / HF liên quan.

Số liệu dưới đây là **số do tác giả công bố**, chưa tự đo trên 30 ảnh của mình.

## Kết luận ngắn

| rate | nên làm gì | lý do |
|---|---|---|
| **0.075** | **Thử DiT-IC**, giữ CoD-Lite | DiT-IC (CVPR 2026) có weights; LPIPS/DISTS tốt hơn MS-ILLM nhiều ở ≤ 0.1 bpp. Rủi ro: tốc độ và DC-AE |
| **0.15 / 0.3** | **Giữ MS-ILLM**, thử thêm CRDR | Chưa có model công khai nào hơn MS-ILLM rõ ràng ở các mức này. Khoảng cách ở đây nên được bù bằng fine-tune và chia bit |

## Ứng viên

| model | năm | bpp có weights | so với MS-ILLM (theo tác giả) | giải mã | weights / license | đánh giá |
|---|---|---|---|---|---|---|
| **DiT-IC** (NJU, CVPR 2026, 2603.13162) | 2026 | 0.009 / 0.044 / 0.081 / 0.100 (Kodak) | BD-rate so với cùng một anchor: LPIPS −84% (MS-ILLM −41%), DISTS −88% (MS-ILLM +3%). Người chấm (61 người) chọn DiT-IC 57% so với StableCodec 28% | A100 fp16 0.15 s cho 1024². Ảnh 2K full-frame cần 16 GB; tile 1024 thì < 7 GB | HF `JunqiShi/DiT-IC`, Apache-2.0; base SANA-600M ("chỉ cho nghiên cứu") | **Ứng viên số 1 cho 0.075** |
| CoD-Lite (MSRA) | 2026 | 0.016 / 0.031 / 0.125 / 0.5 | Ngang MS-ILLM ở 0.075 trên Q̂ v0 (−0.38 so với −0.45), đã đo | 0.15 s/ảnh trên L40 | MIT | Đã có trong bake-off |
| **CRDR** (WACV 2024, 2405.16817) | 2024 | **Một model cho 0.08 → 0.72 bpp**, có núm chỉnh độ thật β | Chưa có so sánh trực tiếp. CLIC2020 ở 0.080 bpp: LPIPS 0.084, DISTS 0.075, FID 5.7 (DiT-IC: 0.067 / 0.038 / 3.06) | ELIC + GAN, nhanh | GDrive, repo không ghi license | Đáng thử ở 0.15 / 0.3 |
| EGIC (ECCV 2024, 2309.03244) | 2024 | dải thấp, chưa rõ | Tác giả nói hơn HiFiC, MS-ILLM, DIRAC | — | Apache-2.0, nhưng **TensorFlow 2** | Phải port sang torch. Để sau |
| ProGIC (2603.02897) | 2026 | 0.039 → 0.195 (RVQ, cùng một model) | Kodak: LPIPS BD-rate −59%, DISTS −58% so với MS-ILLM; nhanh hơn MS-ILLM 10× | 11 ms (Kodak, A100) | **Không tìm thấy code** | Theo dõi |
| Control-GIC (ICLR 2025) | 2025 | 0.1 → 0.6 | **Kém hơn** MS-ILLM (LPIPS BD-rate +33%, theo bảng của ProGIC) | VQGAN | có weights | Bỏ |
| OSDiff (2602.01570) | 2026 | chưa rõ (các checkpoint `lambda_*`) | So với diffusion codec, không so MS-ILLM | Một bước, U-Net | Apache-2.0 | Chưa rõ |
| PULSE (MSRA, 2609.18602) | 2026 | 4 model có biến thể perceptual | Thiết kế cho CPU đơn luồng, không nhắm chất lượng cao nhất | 5.2 kMAC/px | MIT | Chỉ đáng xem nếu nhắm track CPU |
| GLC, RDVQ, StableCodec, OSCAR, Gen2-IC | 2024–26 | chủ yếu < 0.05 | — | — | — | Ngoài dải bitrate của CLIC |

## DiT-IC: những điều phải kiểm tra trước khi dùng

1. **Tốc độ trên L4.** Bảng 3 của paper (A100, FP32, 2048²) cho các bước khoảng 0.85 + 0.012 + 0.12 s. Trên L4, 30 ảnh có thể mất 40–90 s.
   Con số này lớn hơn mức 21–29 s của đội top 2025, nên dễ rơi vào nhóm 25% chậm nhất. Cần đo, rồi thử fp16, INT8, tile.
2. **DC-AE f32 (VAE của SANA).** Mình đã đo DC-AE gốc trên 30 ảnh: trần 22.86 dB, làm hỏng chữ nhỏ.
   DiT-IC có fine-tune decoder bằng LoRA nên có thể đỡ hơn, nhưng phải đo lại. Ảnh màn hình vẫn có thể đi qua VTM-SCC theo mode từng ảnh.
3. **Chỉ tới 0.1 bpp.** Chỉ dùng được cho mức 0.075. Ảnh khó trong bộ có thể cần > 0.1 bpp, nên bộ chia bit phải trộn với MS-ILLM.
4. **Môi trường.** Repo test trên torch 2.8 + CompressAI 1.2.8; server là torch 2.6. Entropy model (hyperprior + context 4 bước ở f64) chạy float,
   nên cần kiểm tra tính tất định giữa các GPU, hoặc chuyển sang số nguyên (repo đã có hạ tầng này).
5. **License.** Card HF ghi Apache-2.0, nhưng SANA ghi "chỉ cho nghiên cứu", giống tình trạng CC-BY-NC của MS-ILLM.

## Việc tiếp theo đề xuất

1. Thêm `ditic` (q2, q3, q4) và `crdr` (q 0–3, β 3.84) vào bake-off. Chạy trên server L40 với 30 ảnh, chấm bằng Q̂ v0.
   Đo thời gian giải mã và xem ảnh cắt bằng mắt, nhất là ảnh có chữ và mặt người.
2. Nếu DiT-IC hơn MS-ILLM rõ ở 0.075: đo trên L4, rồi quyết định giữa DiT-IC và CoD-Lite cho mức này.
3. Với 0.15 / 0.3: giữ MS-ILLM làm base. Bù khoảng cách bằng fine-tune (GAN + loss gần Q̂) và chia bit.

## Nguồn

- DiT-IC: https://arxiv.org/abs/2603.13162, https://github.com/Eric-qi/DiT-IC, https://huggingface.co/JunqiShi/DiT-IC
- CRDR: https://arxiv.org/abs/2405.16817, https://github.com/iwa-shi/CRDR (số CLIC2020 trong `rd_results/CLIC.csv`)
- ProGIC: https://arxiv.org/abs/2603.02897
- EGIC: https://arxiv.org/abs/2309.03244, https://github.com/Nikolai10/EGIC
- Control-GIC: https://arxiv.org/abs/2406.00758, https://github.com/lianqi1008/Control-GIC
- OSDiff: https://arxiv.org/abs/2602.01570, https://github.com/cheesejiang/OSDiff
- GenCodec (CoD, CoD-Lite, PULSE): https://github.com/microsoft/GenCodec
- SANA-600M: https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers

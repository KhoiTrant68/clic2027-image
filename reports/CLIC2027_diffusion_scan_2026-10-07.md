# Diffusion / flow codec cho CLIC 2027: rà soát 2026-10-07

Câu hỏi: codec diffusion một bước nào có **weights công khai**, **decode nhanh** và **chạy được ở 0.075–0.3 bpp**?
Số liệu dưới đây lấy từ paper và README của từng codec, chưa tự đo; cột "trong bake-off" cho biết cái gì sẽ được đo trên 30 ảnh validation.

| codec | loại | bpp có checkpoint | decode | weights / license | hợp với CLIC? | trong bake-off |
|---|---|---|---|---|---|---|
| **CoD-Lite** (Jia et al., MSRA, arXiv 2604.12525) | diffusion 1 bước trong không gian **pixel**, conv, 52M tham số decoder | 0.0039, 0.0078, 0.0156, 0.0312, **0.125, 0.5** | 24 ms cho 1080p trên A100 (42 FPS) | HF `zhaoyangjia/CoD_Lite`, **MIT** cả code lẫn weights | **cao**: không bị trần VAE, nhanh, nhỏ (~400 MB fp32), license sạch | **C5 `codlite`** |
| CoD (Jia et al., CVPR 2026, 2511.18706) | foundation diffusion model cho nén, có bản 1 bước | 0.0039, 0.0312, 0.125 | chưa rõ; lớn hơn nhiều (repo 46.7 GB) | HF `zhaoyangjia/CoD`, MIT | trung bình: nặng, không vượt 0.125 | chưa |
| OSCAR (Guo et al., NeurIPS 2025, 2505.16091) | 1 bước trên SD 2.1, nhiều bpp trong một model | danh sách trong `rate_config.py` (chưa đọc được) | không công bố số | GitHub `jp-guo/OSCAR` + HF | thấp–trung bình: trần SD-VAE 24.7 dB, UNet ~865M | chưa |
| StableCodec (Zhang et al., ICCV 2025, 2506.21977) | 1 bước trên SD-Turbo, nhánh kép cho fidelity | < 0.05 bpp | tương đương VAE/GAN codec (theo paper) | có code | thấp: thiết kế cho rate cực thấp | chưa |
| OSDiff (Jia, Wei et al., arXiv 2602.01570) | 1 bước, discriminator trên feature | chưa rõ | nhanh hơn 46× so với diffusion nhiều bước | Google Drive, Apache-2.0 | chưa rõ | chưa |
| OneDC (2025) | 1 bước, SD 1.5 + DMD2 | đường cong lambda 12.2–0.6 + 0.0034 | nhanh hơn 20× so với multi-step | OneDrive, **CC BY-NC-SA** | thấp: license, SD 1.5 VAE | chưa |
| SODEC (2508.04979) | 1 bước, SD 2.1 + nhánh fidelity | rate cực thấp | 26× nhanh hơn PerCo | **chưa phát hành weights** | — | không |

## Nhận xét

1. **CoD-Lite là ứng viên diffusion duy nhất vừa với cả ba ràng buộc.** Nó decode trực tiếp trong không gian pixel, nên không bị trần VAE (SD-VAE f8: 24.7 dB, DC-AE: 22.9 dB trên 30 ảnh validation). Có checkpoint ở 0.125 và 0.5 bpp, và chạy nhanh hơn MS-ILLM.
2. **Hai điểm yếu của CoD-Lite và hướng xử lý:**
   - *Bitstream độ dài cố định.* Chỉ số VQ được ghi thẳng 4 hoặc 8 bit/token, không có entropy coding. Thêm một entropy model (hyperprior hoặc context trên chỉ số) vào chỉ mục có thể giảm khá nhiều byte. Repo đã có sẵn kinh nghiệm này (S1 với bitstream nguyên, bit-exact).
   - *Train ở 512×512.* Paper tự nhận chất lượng giảm ở độ phân giải rất cao; ảnh CLIC khoảng 2K. Bake-off sẽ cho thấy mức giảm thực tế.
3. **Khoảng trống giữa các rate.** Không có checkpoint ở 0.075 hay 0.3. Bộ phân bổ chọn mỗi ảnh một điểm (0.031 / 0.125 / 0.5) dưới ngân sách cả bộ ảnh. Nếu CoD-Lite mạnh, việc tiếp theo là fine-tune ở 0.0625 (ds 8, 4 bit) và 0.25 bpp (ds 4, 4 bit). Theo README, mỗi rate tốn ~244 giờ A100.
4. **Các codec trên SD (OSCAR, StableCodec, OneDC, SODEC)** bị trần VAE, UNet nặng và thường không có rate cao. Chỉ đáng thử ở 0.075 nếu CoD-Lite thất bại.
5. **Q̂ v0 gần như chưa thấy ảnh diffusion.** Nó được fit trên rating CLIC 2021–2024 (đa số GAN/MSE), nên phải xem ảnh cắt (`crops/*.jpg`) trước khi kết luận. Nếu đi tiếp hướng này, nên tự thu một ít rating cặp đôi (diffusion vs MS-ILLM) để hiệu chỉnh lại Q̂.

## C4: thăm dò refiner (`turbo`)

Ảnh tái tạo MS-ILLM q1–q3 được đưa qua SD-Turbo img2img một bước, strength 0.15 / 0.3, prompt rỗng, cùng số byte với MS-ILLM.
Đây chỉ là phép thử rẻ, không cần train:
- **Nếu Q̂/ảnh cắt tốt hơn MS-ILLM ở 0.075:** đáng train một refiner flow-matching một bước trong không gian pixel, có điều kiện trên ảnh MS-ILLM (hoặc CoD-Lite).
- **Nếu không tốt hơn:** chưa đủ để bác bỏ hướng refiner, vì SD-Turbo bị trần VAE và chưa từng thấy artefact của nén. Nhưng khi đó refiner sẽ xếp sau CoD-Lite.

License của SD-Turbo là non-commercial; nó chỉ được dùng để đo, không dùng trong bài nộp.

## Nguồn

- CoD-Lite: https://arxiv.org/abs/2604.12525, https://github.com/microsoft/GenCodec (commit 9b39a94), https://huggingface.co/zhaoyangjia/CoD_Lite
- CoD: https://arxiv.org/abs/2511.18706, https://huggingface.co/zhaoyangjia/CoD
- OSCAR: https://arxiv.org/abs/2505.16091, https://github.com/jp-guo/OSCAR
- StableCodec: https://arxiv.org/abs/2506.21977
- OSDiff: https://arxiv.org/abs/2602.01570, https://github.com/cheesejiang/OSDiff
- OneDC: https://github.com/onedc-codec/onedc
- SODEC: https://arxiv.org/abs/2508.04979, https://github.com/zhengchen1999/SODEC
- CLIC 2025 (không có bài diffusion; decoder tốt nhất 0.8 s/ảnh trên L4): https://www.mabyduck.com/blog/challenge-on-learned-image-compression-2025/

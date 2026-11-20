# Codec perceptual mới: OSCAR và OSDiff trên CLIC30 — 20/11/2026

**Kết luận:** loại cả hai. Ở mọi ngân sách, OSCAR và OSDiff đều thua mix hiện tại và thua các codec đã có trong pool (DiT-IC, MS-ILLM, CoD-Lite). Khi cho bộ phân bổ chọn thêm, không ảnh nào được chuyển sang hai codec này (ΔQ̂ = 0 ở cả ba ngân sách). Ở độ phân giải 2K, cả hai còn cần khoảng 26 GiB VRAM fp32 (vượt 24 GiB của L4) và decode 3,7–5 s/ảnh trên H100.

## 1. Chọn model

Agent quét tài liệu ngày 20/11 (arXiv, GitHub, HuggingFace) tìm codec perceptual/generative có trọng số công khai ở 0,05–0,35 bpp:

- **Chạy thử:** OSCAR (NeurIPS 2025, `jp-guo/OSCAR`, Apache-2.0, một model nhiều rate tới 0,125 bpp) và OSDiff (`cheesejiang/OSDiff`, Apache-2.0, hai checkpoint λ).
- **Không chạy:**
  - AFP-GIC: đã thử ngày 15/10, không vượt baseline.
  - EGIC: chỉ có TensorFlow 2.
  - ProGIC và Tree-VQ: chưa có code.
  - OneDC: license NC, tối đa khoảng 0,11 bpp.
  - NeFIC: base 5B, cần ≥ 24 GB.
  - StableCodec: chỉ dưới 0,035 bpp.
- **Base SD 2.1:** bản chính thức đã bị gỡ khỏi HuggingFace. Dùng bản sao `sd2-community`, đã đối chiếu hash:
  - checkpoint 768 bắt đầu bằng `ad2a33c361…`, checkpoint 512 bắt đầu bằng `88ecb78256…`, trùng hash phổ biến của bản Stability;
  - VAE fp16 trùng từng byte với repo chính thức `stabilityai/sd-turbo`.
  
  Chỉ tải dạng safetensors.

## 2. Giao thức

Giống ELIC/MLIC++: cùng 30 ảnh, encode và decode ở hai tiến trình riêng, bitrate tính từ byte của stream thật, metric và Q̂ v0 dùng module chung. Mã nguồn: `experiments/perceptual/` (`oscar_run.py`, `osdiff_run.py`, `run.sh`, `score.py`). Chạy trên lab09, H100 dùng chung, fp32.

**OSCAR: những điểm đã phải bổ sung so với repo gốc**
- **Bitstream thật:** repo gốc không ghi stream (bpp chỉ là số danh nghĩa của cấu hình VQ) và resize ảnh về bội số 128. Ở đây stream gồm:
  - header 12 byte;
  - 4 chuẩn kênh latent dạng float32. Decoder cần các giá trị này để rescale, và chúng tính từ latent gốc nên phải truyền đi; đã tính vào byte;
  - chỉ số VQ ở độ dài cố định ⌈log2 K⌉ bit.
- **Padding:** ảnh được pad cạnh (replicate) lên bội số 64, sau đó cắt về kích thước gốc.
- **Encoder tất định:** encoder VAE dùng posterior mean thay cho lấy mẫu ngẫu nhiên.
- **Attention của VAE:** tính theo khối query. Kết quả giống hệt về toán; nếu không làm vậy, ảnh 2K cần ma trận 7,6 GiB.
- **Thư viện:** `einx` ghim về 0,3.0, đúng môi trường của tác giả.

**OSDiff: những điểm đã phải bổ sung**
- **Stream:** dùng hàm ghi stream của tác giả, thêm header 8 byte chứa kích thước gốc.
- **Nhiễu x_T:** sampler một bước bắt đầu từ nhiễu x_T. Decoder sinh nhiễu này từ seed cố định 231, không tốn bit.
- **Nạp trọng số:** `strict=False` như tác giả, nhưng dừng chạy nếu thiếu bất kỳ trọng số nào ngoài discriminator (chỉ dùng khi train) và `logvar`.

**Tính nhất quán decode**
- **OSDiff:** 60/60 ảnh giải mã ở tiến trình mới khớp từng bit.
- **OSCAR:** 150/150 stream khớp. Pixel lệch tối đa 1 mức ở khoảng 0,008% điểm ảnh, do phép tính float trên GPU phía decoder không tất định. Vì chỉ số VQ có độ dài cố định nên việc giải mã stream không bị ảnh hưởng.

## 3. Kết quả

### Từng điểm vận hành (cả 30 ảnh)

| Codec | Điểm | bpp | Q̂ ↑ | LPIPS ↓ | DISTS ↓ | PSNR gộp | Decode/ảnh |
|---|---|---:|---:|---:|---:|---:|---:|
| OSCAR | 0,0430 | 0,0441 | −2,012 | 0,2183 | 0,1533 | 19,62 | 3,7 s |
| OSCAR | 0,0625 | 0,0641 | −1,609 | 0,1974 | 0,1437 | 19,82 | 3,7 s |
| OSCAR | 0,0781 | 0,0801 | −1,132 | 0,1673 | 0,1251 | 20,80 | 3,7 s |
| OSCAR | 0,0937 | 0,0961 | −0,676 | 0,1377 | 0,1057 | 21,64 | 3,7 s |
| OSCAR | 0,1250 | 0,1281 | −0,249 | 0,1182 | 0,0908 | 22,21 | 3,7 s |
| OSDiff | λ2 | 0,0886 | −0,817 | 0,1329 | 0,1252 | 21,72 | 4,8 s |
| OSDiff | λ1 | 0,1300 | −0,297 | 0,1057 | 0,1037 | 22,33 | 5,0 s |

Bpp thực của OSCAR cao hơn bpp danh nghĩa 2–3%, phần chênh đến từ padding lên bội số 64 và header.

### Cùng ngân sách (phân bổ theo từng ảnh, Q̂ v0)

| Ngân sách | Mix hiện tại | OSCAR một mình | OSDiff một mình | DiT-IC | MS-ILLM | Thêm OSCAR / OSDiff vào mix |
|---:|---:|---:|---:|---:|---:|---:|
| 0,075 | **0,184** | −0,908 | không đạt | 0,164 | −0,451 | 0 / 0 |
| 0,15 | **0,789** | −0,247 | −0,273 | 0,279 | 0,545 | 0 / 0 |
| 0,3 | **1,498** | −0,247 | −0,273 | 0,279 | 1,383 | 0 / 0 |

- **0,075 bpp:** OSCAR kém DiT-IC khoảng 1,07 Q̂, và kém cả MS-ILLM.
- **0,15 bpp:** điểm cao nhất của hai model chỉ ngang mức −0,25…−0,30, thấp hơn CoD-Lite (0,285) và MS-ILLM (0,545).
- **0,3 bpp:** cả hai dừng ở khoảng 0,13 bpp nên không dùng hết ngân sách.

### Tài nguyên

- **VRAM:** reserved đỉnh 25,8–26,5 GiB cho ảnh 2048×2048 ở fp32. Con số này vượt 24 GiB của L4, nên nếu dùng sẽ phải chuyển sang fp16 hoặc tile.
- **Thời gian decode cho 30 ảnh** (API, H100 đang chạy chung với tiến trình khác):
  - OSCAR: 110–112 s;
  - OSDiff: 143–151 s;
  - so sánh: các đội top 2025 mất 21–29 s trên L4.

## 4. Giới hạn

- Chỉ có 30 ảnh và chưa có chấm mù. Q̂ v0 là mô hình sàng lọc.
- Chạy đúng checkpoint của tác giả, không fine-tune. Các thay đổi ở mục 2 chỉ nhằm có bitstream thật và chạy được ảnh 2K; chúng có thể làm chất lượng thấp hơn đôi chút so với cách tác giả resize ảnh.
- Trang theo dõi `/results` đã có hai codec này (nhóm "Sàng lọc codec", trạng thái Loại).

## 5. Bằng chứng

- Kết quả nhẹ (metric từng ảnh, JSON encode/decode, parity, log): `results/2026-11-20_perceptual/{oscar,osdiff}/`.
- Stream và ảnh tái tạo: lab09 `/mnt/data/khoitm8/clic2027/perc/{oscar,osdiff}/full30/`.
- Hash của trọng số được ghi trong `q*-encode.json`.

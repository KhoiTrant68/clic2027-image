# Kế hoạch CLIC 2027 v2: tập trung để thắng

Cập nhật 03/10/2026. Thay thế lịch ở mục C của `plans/Plan CLIC 2027.md`. Các phần kỹ thuật A1–A6 và B của bản cũ vẫn dùng được.

## 0. Quyết định ngày 03/10

- **CLIC 2027 là mục tiêu duy nhất của repo.** Bài CVPR 2027 bị bỏ hoặc hoãn. Tháng 10–11 dành cho CLIC.
- **Compute tháng 10:** Kaggle (T4/P100), cộng một máy L4 thuê theo giờ.
- **Tiêu chí cho mọi quyết định:** có tăng Elo ở 0.075 / 0.15 / 0.3 bpp mà vẫn qua luật tốc độ và dung lượng không. Không chọn vì có lợi cho lý thuyết hay cho một bài báo.

## 1. Điều kiện thắng

| Điều kiện | Hệ quả |
|---|---|
| Elo cao nhất do người thật chấm theo cặp, có ảnh gốc để so | Cần một thước đo thay người chấm (Q̂) để tự đánh giá |
| Phải có bài nộp ở cả 3 mức bitrate | Mỗi mức có thể dùng một model riêng nếu việc đó tốt hơn |
| Không nằm trong 25% bài giải mã chậm nhất | Mục tiêu: 30 ảnh trong ≤ 20 giây trên L4 |
| Decoder ≤ 4 GB, bị khóa trước 01/03 | Khóa nội bộ vào 22/02. Mọi mẹo sau đó chỉ nằm ở encoder |
| Server: torch 2.6, không có diffusers, 2 CPU | Decoder viết bằng torch thuần; entropy coding bằng số nguyên |

## 2. Những gì đã đo được (29/9 – 03/10)

- **Trần tái tạo trên 30 ảnh validation:** DC-AE f32 đạt 22.86 dB; SD-VAE f8 đạt 24.73 dB; Vcoder (hạng 1 năm 2025) đạt 23.99 dB ở 0.075 bpp. DC-AE làm hỏng chữ nhỏ (CER 0.58 trên ảnh màn hình).
- **Ảnh màn hình:** VVC-SCC hơn VTM 4:2:0 từ 1.1 đến 3.3 dB và cho chữ rõ nhất. Mode này gần như sẵn sàng.
- **Ảnh tự nhiên:** cách "DC-AE + bù sai số bằng VTM" ngang VTM về PSNR ở 0.075 bpp, với LPIPS tốt hơn 4 lần. Nhưng con số này giả định phần DC-AE đạt trần, mà S1 thật ở 0.034 bpp còn kém trần 4 dB.
- **Tính tất định:** float lệch giữa GPU và CPU; mạng số nguyên khớp từng bit. Hạ tầng entropy số nguyên đã có.
- **Tốc độ:** DC-AE decoder mất 4.6 s (fp16) cho một ảnh 2K trên T4, chậm hơn mục tiêu 9–13 lần.

**Kết luận:** chưa có bằng chứng DC-AE là base tốt nhất cho CLIC. Phải so sánh trực tiếp.

## 3. Năm câu hỏi phải trả lời trong tháng 10

| # | Câu hỏi | Cách trả lời | Hạn |
|---|---|---|---|
| Q1 | Base nào cho Q̂ cao nhất ở từng mức, trong giới hạn thời gian? | So sánh trực tiếp các base trên 30 ảnh validation (mục 4, tuần 2–3) | 26/10 |
| Q2 | Q̂ dự đoán lựa chọn của người chấm tốt đến đâu? | Fit trên dữ liệu chấm theo cặp của CLIC, đo độ chính xác trên phần giữ lại | 12/10 |
| Q3 | Các đội đứng đầu 2024–2025 đã làm gì? | Đọc whitepaper trên OpenReview (`compression.cc/CLIC/2025`) | 12/10 |
| Q4 | Mỗi ứng viên giải mã nhanh đến đâu trên L4? | Đo trong image `clic-gpu` trên L4 thuê | 26/10 |
| Q5 | Server tính byte và chạy decoder ra sao? | Nộp thử baseline VTM ngay khi server validation mở | Khi server mở |

## 4. Lịch

### Tuần 1 (03–12/10): thước đo và thông tin

- [ ] **Q̂ v0:** tải dữ liệu chấm CLIC 2021 / 2022 / 2024 từ https://archive.compression.cc/datasets/. Tính trên Kaggle các thước đo LPIPS, DISTS, PSNR, MS-SSIM, và nếu được thì thêm vài thước đo không cần ảnh gốc. Fit Bradley–Terry (hoặc logistic) trên hiệu các thước đo. Báo độ chính xác dự đoán trên phần giữ lại, so với từng thước đo đơn lẻ.
- [ ] **Đọc whitepaper** của Vcoder, Evolve, IronMan, Thanos (2025) và các đội đứng đầu 2024. Ghi lại: kiến trúc, cách dùng GAN, cách chia bit, thời gian giải mã.
- [ ] **L4, khoảng 2 giờ:** dựng `clic-gpu` (`clic/l4/setup_l4.sh`); chạy probe tất định A0; đo và profile `DCAE._tiled_decode` (fp16, bf16, tile lớn, không chia tile).

### Tuần 2–3 (13–26/10): so sánh trực tiếp các base

Chạy trên đúng 30 ảnh validation, ở đúng 3 mức bitrate, dùng ngân sách cả bộ ảnh:

| Ứng viên | Nội dung | Ghi chú |
|---|---|---|
| C0 | VTM và VTM-SCC | Mốc chuẩn (đã có) |
| C1 | DC-AE + S1 (đã có), cộng residual bằng VTM | Đại diện cho hướng hiện tại |
| C2 | Codec học được + GAN, có checkpoint công khai ở dải 0.075–0.3 bpp (ví dụ MS-ILLM hoặc HiFiC; cần kiểm tra mức bpp của checkpoint) | Kiểu codec nhanh, đã chứng minh được ở dải bitrate này |
| C3 | Codec PSNR mạnh (ELIC/TCM) | Mốc độ trung thực; ứng viên cho lớp residual |
| C4 | VAE f8 một bước (StableCodec/CADC) | Chỉ có checkpoint ≤ 0.04 bpp. Chỉ dùng để so trần và tốc độ, trừ khi kết quả rất hứa hẹn |

Đầu ra cho mỗi ứng viên và mỗi mức: Q̂, PSNR, MS-SSIM, thời gian giải mã trên L4, dung lượng decoder, và lưới ảnh crop để xem bằng mắt.

**Mốc quyết định 27/10:** chọn base cho từng mức bitrate. Mục 7 liệt kê những việc chỉ làm lại nếu chọn hướng DC-AE.

### Tháng 11: train base và dựng khung nộp bài

- [ ] Train hoặc fine-tune base đã chọn ở 3 mức (GAN + LPIPS + phạt theo Q̂), trên Kaggle và L4.
- [ ] Khung nộp bài: định dạng container, `budget.py`, `check_submission.py`. Nộp thử lần 1 (VTM) và lần 2 (một model học được) ngay khi server mở.
- [ ] Ghép mode VVC-SCC và cờ chọn mode cho từng ảnh.

### Tháng 12: chia bit và tất định

- [ ] Chia bit cho cả bộ ảnh theo độ dốc ΔQ̂/ΔR. Phần này thắng được dù chọn base nào.
- [ ] Entropy coding số nguyên cho base đã chọn. Nộp thử lần 3: PSNR trên server phải khớp ở máy em tới 3 chữ số thập phân.

### Tháng 1: tốc độ, dung lượng, nộp đủ 3 mức

- [ ] Tối ưu trên L4 để 30 ảnh ≤ 20 giây.
- [ ] **Nộp validation đủ cả 3 mức trước 15/01.** Sau đó chỉ cải tiến.
- [ ] Tự chấm khoảng 300 cặp ảnh để kiểm tra Q̂.

### Tháng 2 – 3

- [ ] Mẹo phía encoder: thử nhiều seed, tối ưu latent, chọn mode.
- [ ] Khai báo 3 phiên bản. **Khóa decoder 22/02.**
- [ ] 02–09/03: chỉ chạy encoder trên bộ ảnh test rồi nộp.

## 5. Tuần này: ai làm gì

| Việc | Claude | Bạn |
|---|---|---|
| Script Q̂: tải dữ liệu, tính thước đo (Kaggle), fit Bradley–Terry | Viết | Chạy trên Kaggle |
| Whitepaper 2024–2025 | Đọc và tóm tắt | Mở OpenReview và qua bước xác minh bot (mình không được làm bước này) |
| L4: `clic-gpu`, probe A0, profile DC-AE | Viết script | Thuê máy, chạy |
| Script so sánh base: tải checkpoint C2/C3, encode/decode ở 3 mức theo ngân sách | Viết | Chạy trên Kaggle |

## 6. Rủi ro mới

| Rủi ro | Cách xử lý |
|---|---|
| Không tìm được checkpoint công khai ở 0.075–0.3 bpp cho C2 | Tự fine-tune một codec ELIC/TCM có sẵn với GAN + LPIPS trên Kaggle |
| Q̂ dự đoán kém (độ chính xác < 65%) | Thêm thước đo học được; tăng phần tự chấm |
| Server validation mở muộn | Dùng `check_submission.py` trên `clic-gpu` làm phép thử thay thế |

## 7. Tạm dừng (chỉ làm lại nếu base được chọn là DC-AE)

- Lý thuyết cho CVPR (Định lý 4, toy Gaussian vòng 2).
- Chạy lại go/no-go 1.
- Bridge S2/S3 trên SANA.
- Codec S1 vẫn giữ làm ứng viên C1.

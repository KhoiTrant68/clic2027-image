# CLIC 2025: kết quả, cách chấm, và khoảng cách của mình (2026-10-10)

Nguồn: bảng xếp hạng test 2025 (archive.compression.cc), slide tổng kết của ban tổ chức tại PCS 2025, blog Mabyduck.
**Whitepaper của các đội (OpenReview) chưa đọc được:** trang và API đều bị chặn bởi bước xác minh chống bot, phải có người qua bước đó.
Vì vậy báo cáo này **chưa có phần phương pháp** của Vcoder / Evolve / Thanos.

Bộ test 2025 (30 ảnh) chính là bộ validation 2027, tức đúng 30 ảnh của bake-off. Số dưới đây so được trực tiếp về ảnh,
nhưng không cùng người chấm.

## 1. Bảng xếp hạng test 2025 (Elo, PSNR, thời gian giải mã cho 30 ảnh)

| đội | track | 0.075 | 0.15 | 0.3 | PSNR @0.075 | decoder | giải mã (s) |
|---|---|---|---|---|---|---|---|
| **Vcoder** (GPU hạng 1) | GPU | **1929** | **2127** | 2247 | 24.0 | 1.07 GB | 24 |
| **Evolve** (GPU hạng 2) | GPU | 1903 | 2119 | 2264 | 24.6 | 268 MB | 29 |
| **Thanos** (GPU hạng 3) | GPU | 1888 | 2106 | **2269** | 25.3 | 950 MB | 22 |
| PKUSZ-AliMerlin | GPU | 1792 | 2116 | 2262 | 24.1 | 294 MB | 21 |
| TestC | GPU? | 1774 | 2053 | 2267 | 25.5 | 122 MB | 25 |
| IronMan (CPU hạng 1) | CPU | 1890 | 2106 | 2291 | 25.6 | 551 MB | 640 |
| RunRun (CPU hạng 2) | CPU | 1776 | 2043 | 2262 | 25.5 | 122 MB | 500 |
| CoolChic (CPU hạng 3) | CPU | 1683 | 1985 | 2229 | 25.7 | 8 MB | 50 |
| VTM | mốc | 1405 | 1708 | 1983 | 26.5 | 1.4 MB | 49 |
| HM | mốc | 1311 | 1644 | 1923 | 25.8 | 0.4 MB | 40 |

- Elo là **một thang chung cho cả 3 mức** (61 điều kiện = 20 phương pháp × 3 rate + ảnh gốc), nên so được giữa các cột.
- Ở 0.15 và 0.3, nhóm đầu GPU sát nhau: 0.15 cách nhau 21 Elo (4 đội), 0.3 cách nhau 22 Elo (5 đội). Ở 0.075 khoảng cách lớn hơn: Vcoder hơn PKUSZ 137 Elo.
- Đội thắng ở 0.075 có PSNR **thấp nhất** (24.0 dB, VTM 26.5 dB). Người chấm thưởng cho độ thật, không thưởng cho độ trung thực theo pixel.

## 2. Cách chấm (từ slide của ban tổ chức)

- **Ảnh cắt 512×512**, so theo cặp, có ảnh gốc để đối chiếu. 289 người chấm (đã sàng lọc mù màu, độ nhạy tương phản), 29 520 cặp.
- **Bayesian Elo, chọn cặp chủ động: ưu tiên các cặp có Elo gần nhau.** Các đội top bị so trực tiếp với nhau nhiều nhất.
- **Xếp hạng chung tính theo thứ hạng trong từng rate**, không cộng Elo. Slide ghi `Score = 1R1 + 1R2 + 1R3` (mất định dạng khi trích text).
  Tính lại từ Elo trên bảng xếp hạng:
  - Nếu là tổng nghịch đảo hạng (1/R₁ + 1/R₂ + 1/R₃): Vcoder 2.2, Thanos 1.58, Evolve 1.33. Ngược với kết quả chính thức (Evolve hạng 2).
  - Nếu là tổng hạng (nhỏ hơn là tốt hơn): Vcoder 7, Evolve 7, Thanos 8. Khớp thứ tự chính thức (có hòa ở đầu).
  - **Chưa chắc chắn**: Elo chính thức có thể khác Elo trên bảng xếp hạng. Cần xem trang 16 của file PDF slide.
- Ban tổ chức bootstrap lại theo phiên chấm và công bố **xác suất** về hạng 1/2/3: chênh 10–20 Elo nằm trong nhiễu.
- Quy định 2025 mới: 25% bài giải mã chậm nhất mỗi track không được giải. Đội GPU top mất 21–29 s cho 30 ảnh.

## 3. Kết luận của ban tổ chức và Mabyduck

- **Chất lượng không tăng** từ 2024 lên 2025: phương pháp tốt nhất 2024 (HIT_AIIA, 4.9 s/ảnh) ngang phương pháp tốt nhất 2025 (0.8 s/ảnh).
  Phần tiến bộ là **tốc độ**, không phải chất lượng. Ban tổ chức đặt câu hỏi "nén ảnh bằng mạng nơ-ron đã chạm trần chưa?".
- **Không có bài diffusion nào** năm 2025, vì luật tốc độ. Mabyduck dự đoán sẽ có khi chúng đủ rẻ.
- Cool-chic: decoder 8.4 MB, nhỏ nhất trong các phương pháp học được, đã thu hẹp khoảng cách với top.

## 4. Mình đang ở đâu (ước lượng thô)

Quy đổi: Q̂ là logit Bradley–Terry, nên 1.0 Q̂ ≈ 400/ln 10 ≈ 174 Elo. Lấy VTM 2025 làm mốc ở từng rate (bake-off 09/10, Q̂ v0):

| rate | mình (Q̂ v0, so với VTM) | Elo ước lượng | GPU hạng 1 năm 2025 | còn thiếu |
|---|---|---|---|---|
| 0.075 | mix −0.17 so với VTM 4:2:0 −2.52 → +2.35 | ≈ 1810 | 1929 | **≈ 120** |
| 0.15 | MS-ILLM +0.55 so với VTM-SCC −1.10 → +1.65 | ≈ 1995 | 2127 | **≈ 130** |
| 0.3 | MS-ILLM +1.38 so với VTM-SCC 0.00 → +1.38 | ≈ 2220 | 2269 | **≈ 50** |

Ở 0.15 và 0.3, mình lấy VTM-SCC làm mốc thay cho VTM 4:2:0. VTM-SCC tốt hơn một chút, nên khoảng thiếu thực tế có thể lớn hơn bảng khoảng 20 Elo.
Cả bảng giả định Q̂ v0 được hiệu chỉnh đúng (độ chính xác chỉ 0.81), nên chỉ coi là cỡ độ lớn.

## 5. Hệ quả cho kế hoạch

1. **Luật chấm dựa trên thứ hạng ở từng rate.** Chiến thắng rõ ràng ở một rate rất đáng giá, nhưng công thức chính xác cần kiểm tra lại (mục 2).
   Khi chọn variant để khai báo, phải tính theo thứ hạng, không theo Elo trung bình.
2. **0.3 bpp là rate gần nhất** (thiếu ≈ 50 Elo, nhóm đầu rất sát). MS-ILLM chỉ là weights có sẵn, chưa tinh chỉnh.
3. **0.075 và 0.15 thiếu ≈ 120–130 Elo.** Đây là nơi Vcoder thắng: PSNR thấp, thật hơn. Muốn lấy hạng ở đây, cần một decoder sinh chi tiết mạnh hơn MS-ILLM.
   CoD-Lite (one-step, 0.15 s/ảnh) đã ngang MS-ILLM ở 0.075 mà chưa train gì thêm, và chưa đội nào từng nộp diffusion đủ nhanh.
4. **Chấm trên ảnh cắt 512×512:** lỗi cục bộ (mặt người, chữ) trong một ảnh cắt có thể làm thua cả cặp. Chia bit nên tránh để vùng nào quá tệ.
   Q̂ đã được fit trên ảnh cắt của dữ liệu chấm CLIC, cùng kiểu.
5. **Tốc độ:** mục tiêu ≤ 20 s cho 30 ảnh là đúng (top GPU 21–29 s). MS-ILLM 0.8 s/ảnh trên L40 ≈ 24 s; cần đo trên L4.

## 6. Còn thiếu

- **Whitepaper OpenReview** của Vcoder, Evolve, Thanos, IronMan (kiến trúc, loss, GAN hay không, chia bit, mẹo encoder). Cần có người qua bước xác minh.
- Hình trong slide (Elo theo rate, xác suất bootstrap, công thức xếp hạng ở trang 16): file PDF 8 MB, chưa tải.

## Nguồn

- Bảng xếp hạng test 2025: https://archive.compression.cc/2025/leaderboard/image_0_075/test/ (cùng `image_0_15`, `image_0_3`)
- Giải thưởng 2025: https://archive.compression.cc/2025/awards/
- Slide tổng kết CLIC 2025 (PCS, Aachen): https://drive.google.com/file/d/1paUh79j5UFTfhoI8_74VYhRfa7FSnPCL/view
- Mabyduck: https://www.mabyduck.com/blog/challenge-on-learned-image-compression-2025/
- Bảng xếp hạng test 2024 (để so): https://archive.compression.cc/2024/leaderboard/image_0_075/test/
- Whitepaper 2025 (bị chặn): https://openreview.net/group?id=compression.cc/CLIC/2025/Challenge

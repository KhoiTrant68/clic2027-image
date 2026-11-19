# MLIC++: kiểm định phép đo nhận dạng trên font mới

**Đã hoàn tất CNN reader, kiểm nguồn giữ riêng, mã hóa q2, giải mã lại và đối chứng Tesseract.** Reader mới vượt ngưỡng nguồn đã đặt trước. Trên tập tổng hợp này, chênh lệch nhận dạng giữa MSE và adapter Mean rất nhỏ theo CNN, dù Tesseract giảm rõ hơn. Kết quả chưa ủng hộ việc dùng mức giảm OCR làm bằng chứng chính về tổn hại nội dung.

Đây là bước tiếp theo của [screen CVPR](MLICpp_CVPR_screen_2026-11-05.md), thực hiện theo [protocol khóa trước](../plans/MLICpp_recognition_gate_2026-11-05.md). Ưu tiên vẫn là công trình đủ vững cho CVPR, chấp nhận kỳ sau. Không có model nén mới hoặc claim novelty từ phép thử này.

## 1. Thiết kế và tiêu chí

| Thành phần | Thiết lập thực tế |
| --- | --- |
| Nhiệm vụ | Phân loại ký tự giữa trong bốn lớp `6/8/E/F`, không phải đọc đúng cả từ |
| Train reader | 2.400 ảnh tổng hợp mới, sáu font |
| Validation | 256 ảnh, font Ubuntu-C giữ riêng theo file; họ Ubuntu vẫn có trong train |
| Test | 256 ảnh, bốn kiểu chữ SourceCodePro của **một họ font** chưa có trong screen trước |
| Thay đổi bố cục | Train/validation đặt chữ ở phần trên; test đặt ở phần dưới ảnh 256 × 256 |
| Đầu vào reader | Crop ký tự từ bbox generator, grayscale 48 × 48, chuẩn hóa cố định; đây là thông tin vị trí đặc quyền |
| Reader | CNN ba block convolution, MLP; Adam 0,001, batch 64, 30 epoch, seed 7301 |
| Chọn checkpoint | Accuracy validation cao nhất, lấy lần sớm nhất nếu hòa; checkpoint được chọn ở **epoch 28** |
| Gate đặt trước | Accuracy nguồn validation và test đều ≥95%; không đạt thì dừng trước codec |
| Codec | MLIC++ MSE legacy q2 và adapter Mean đã có; không fine-tune trên dữ liệu này |
| Runtime | Một L40, FP32 deterministic, TF32 tắt, cuDNN v7/NVRTC như screen trước |

Test bao phủ đầy đủ font × cỡ chữ 14/18/24/32 × tương phản 64/192 × nền phẳng/texture × bốn nhãn. Vị trí và nền được sinh với seed cố định. Sau khi crop theo bbox, phép thử không đánh giá khả năng tự phát hiện vị trí chữ hay độ bất biến theo bố cục của một OCR hoàn chỉnh.

Checkpoint reader khóa trước khi mở test; sau đó không sửa reader theo kết quả test. Accuracy nguồn: **245/256 = 95,70% trên validation; 255/256 = 99,61% trên test**. Cả hai đạt gate. Việc qua gate chỉ xác nhận năng lực phân biệt bốn ký tự trong miền tổng hợp đang thử, không chứng nhận reader cho ảnh tự nhiên.

## 2. Kết quả chính: CNN reader đã khóa

| Ảnh đầu vào | Số đúng / tổng | Accuracy |
| --- | ---: | ---: |
| Nguồn | 255/256 | 99,61% |
| MSE q2 | 251/256 | 98,05% |
| q2 + adapter Mean | 250/256 | 97,66% |

Hai decoder dùng **cùng từng bitstream**, bpp thực trung bình **0,02693367**. Đây là ảnh tổng hợp nhiều nền trống, không tương đương bài toán ảnh tự nhiên ở 0,15 hoặc 0,30 bpp.

Từ MSE sang Mean: **một ảnh đúng→sai, không ảnh sai→đúng**. Chênh lệch accuracy là −0,39 điểm phần trăm; CI 95% bootstrap ghép cặp 10.000 lần theo ảnh: **[−1,17; 0,00] điểm phần trăm**. Trong 255 ảnh nguồn được CNN đọc đúng, kết quả vẫn là 251 so với 250; không do một lỗi nguồn tạo ra chênh lệch.

Không diễn giải CI này thành hai decoder tương đương hoặc không có tổn hại. Mẫu có rất ít lỗi và gần trần accuracy; nghiên cứu chưa thiết kế một phép kiểm tương đương với biên sai khác cho phép. Kết quả chỉ cho thấy **mức giảm lớn chưa xuất hiện theo endpoint chính đã khóa trên tập này**.

## 3. Đối chứng độc lập: Tesseract

Tham số OCR được cố định trước khi đọc accuracy codec: Tesseract 5.5.3, eng, psm 10, oem 1, whitelist `68EF`, cùng bbox generator, phóng 4× bicubic. Đây là endpoint phụ, không dùng để chọn lại checkpoint CNN hoặc đổi kết luận chính sau khi thấy số liệu.

| Ảnh đầu vào | Đúng trên toàn bộ 256 ảnh | Accuracy |
| --- | ---: | ---: |
| Nguồn | 202 | 78,91% |
| MSE q2 | 196 | 76,56% |
| q2 + adapter Mean | 178 | 69,53% |

Từ MSE sang Mean, OCR có 37 ảnh đúng→sai và 19 ảnh sai→đúng; giảm ròng 18 ảnh, tương đương −7,03 điểm phần trăm. CI theo ảnh: [−12,89; −1,17] điểm phần trăm. Nếu chỉ giữ 202 nguồn được OCR đọc đúng, kết quả là MSE 164/202 và Mean 146/202; chênh lệch −8,91 điểm phần trăm, CI [−14,85; −2,97].

**Tesseract không đạt ngưỡng nguồn 95%** đặt cho một bộ đo đủ năng lực trong pilot này. Sự bất đồng giữa CNN và OCR cho thấy không thể đồng nhất OCR accuracy với khả năng khôi phục nội dung hoặc khả năng đọc của người xem. CNN có thể bỏ qua biến dạng mà OCR nhạy với; OCR cũng có thể phát hiện biến đổi mà CNN bỏ sót. Chưa có human evaluation để xác định công cụ nào gần cảm nhận người đọc hơn.

Một trường hợp cụ thể: ảnh `0116`, ký tự `6`, SourceCodePro-Light cỡ 32, tương phản 64 trên nền texture. CNN đổi dự đoán từ `6` ở MSE sang `E` ở Mean, còn Tesseract vẫn đọc `6` ở cả hai. Ngược lại, có các ảnh Tesseract đổi đúng→sai nhưng CNN giữ đúng. Vì vậy, không chọn riêng các ảnh hoặc bộ đo thuận lợi cho câu chuyện mong muốn.

Hai đợt screen khác dữ liệu, nhiệm vụ và cấu hình readout. Không dùng chênh lệch giữa các đợt để quy toàn bộ lỗi trước đây cho Tesseract. Đợt mới chỉ bổ sung bằng chứng rằng kết luận về nhận dạng cần kiểm bằng nhiều công cụ có năng lực trên nguồn.

## 4. Kiểm chứng và tái lập

- **512/512 đầu ra** MSE/Mean khớp pixel hash khi giải mã stream bằng tiến trình mới, không đọc ảnh nguồn.
- Dự đoán CNN trên nguồn trong stage codec khớp toàn bộ vector dự đoán của stage gate; không đổi reader hoặc tiền xử lý giữa hai stage.
- Kiểm hash mã chạy, font, checkpoint reader và ảnh nguồn. Reader lưu cả trên server và cục bộ; tệp `.pt` được Git ignore.
- Pipeline chính hoàn tất trong **391,76 giây**, khoảng 6 phút 32 giây; đây là thời gian toàn chuỗi, không phải latency codec trên L4 hay GPU active time. OCR độc lập chạy bổ sung trên CPU.
- Bootstrap theo ảnh là thăm dò; các ảnh dùng chung một họ font. CI không mô tả độ bất định khi chuyển sang họ font hay dữ liệu tự nhiên khác.

Bằng chứng: [analysis.json](../results/2026-11-05_mlicpp_recognition_gate/analysis.json), [config](../results/2026-11-05_mlicpp_recognition_gate/config.json), [source gate](../results/2026-11-05_mlicpp_recognition_gate/source_gate.json), [fresh parity](../results/2026-11-05_mlicpp_recognition_gate/fresh.json). Bản sao mã nguồn nằm trong `source_archive` cùng thư mục. Ảnh và stream lớn ở server: `/home/jovyan/clic2027-artifacts/mlicpp/cvpr20261105/source-qualified-glyph-cnn-q2`.

Lệnh phân tích lại số liệu nhỏ sau khi thu bằng chứng:

```bash
python experiments/mlicpp/analyze_recognition_gate.py \
  --root results/2026-11-05_mlicpp_recognition_gate
```

## 5. Quyết định cho hướng CVPR

**Phép đo nhận dạng bốn ký tự đã qua bước kiểm nguồn; giả thuyết tổn hại nội dung lớn do adapter vẫn chưa được xác nhận bằng công cụ có năng lực trên nguồn.** Không mở rộng huấn luyện một cơ chế bảo vệ chữ dựa riêng vào mức giảm Tesseract hiện tại.

Bộ reader có thể dùng làm một control đã khóa cho thí nghiệm sau. Trước khi chọn can thiệp model, cần nhiệm vụ nhiều lớp/từ dài hoặc scene text có ground truth độc lập để tránh trần accuracy; một tập test mới giữ riêng hoàn toàn khi thiết kế can thiệp; và nhiều bộ đo đạt kiểm nguồn. Không dùng lại 256 ảnh này làm test xác nhận sau khi đã xem và phân nhóm kết quả.

**Chưa chạy latent readout mới**, nên chưa kết luận thông tin mất ở encoder hay decoder. Source gate của CNN ảnh không tự xác nhận năng lực của một classifier trên latent. Nếu theo hướng đó, phải có dữ liệu latent train thực và kiểm năng lực readout riêng; không dùng 1-NN yếu của screen trước làm bằng chứng mất thông tin.

Đợt này hoàn tất bước đo lường, chưa tạo đóng góp model. Các giới hạn còn nguyên: một seed, một họ font test, bốn lớp, bbox biết trước, dữ liệu tổng hợp, một rate và một backbone legacy; thiếu human evaluation, thử full-resolution, L4 latency và đối chứng phương pháp perceptual hiện đại. Không gộp số liệu này vào bảng 30 ảnh CLIC như cùng protocol.

# Phép đo nhận dạng có kiểm định trên nguồn

**Đã hoàn tất:** [báo cáo kết quả](../reports/MLICpp_recognition_gate_2026-11-05.md). CNN qua gate nguồn (validation 245/256, test 255/256); q2 MSE đúng 251/256, Mean 250/256. Tesseract chỉ đúng 202/256 nguồn và cho mức giảm codec lớn hơn. Không nâng mức giảm OCR thành bằng chứng tổn hại nội dung; chưa có latent readout mới. Protocol bên dưới giữ nguyên các quyết định trước khi chạy.

## Câu hỏi và tiêu chí khóa trước khi chạy

1-NN của screen trước chỉ đúng 42,19% trên ảnh nguồn khi đổi font, nên chưa đủ năng lực phân giải thông tin mất ở đâu. Đợt này thay bằng CNN nhỏ nhận dạng bốn ký tự `6/8/E/F`. Đây là **công cụ đo**, không phải codec mới hay một đóng góp paper.

- Train: 2.400 ảnh tổng hợp mới, sáu font; validation: 256 ảnh, font Ubuntu-C giữ riêng theo file. Font validation vẫn thuộc họ Ubuntu đã có trong train.
- Test: 256 ảnh, bốn font SourceCodePro thuộc một họ font chưa dùng trong screen trước, đầy đủ tích font × size × contrast × background × glyph. Không gọi bốn kiểu chữ này là bốn họ font độc lập.
- Train/validation đặt chữ ở phần trên ảnh; test ở phần dưới. Biết bbox glyph từ generator là đầu vào đặc quyền của phép đo; sau crop/resize, khả năng nhận dạng không kiểm tra năng lực tự phát hiện vị trí.
- CNN 3 convolution blocks + MLP; 30 epoch, Adam 0,001, batch 64, seed 7301. Chọn checkpoint accuracy validation cao nhất, lấy checkpoint sớm nhất nếu hòa; không xem test trong lúc train.
- Chỉ mở test nguồn một lần sau khi khóa checkpoint. **Validation và test nguồn cùng phải đạt ≥95%**. Nếu không đạt, dừng trước mã hóa; không sửa model trên chính test này rồi gọi nó là giữ riêng.
- Nếu qua gate: q2 MSE và q2 với adapter Mean có sẵn, cùng từng stream; không fine-tune codec, không chọn checkpoint bằng kết quả nén. Chạy tiến trình mới để kiểm 512 pixel hashes.

## Ma trận và dự đoán

| Nhánh | Yếu tố thay đổi | Giữ cố định | Dự đoán trước khi chạy |
| --- | --- | --- | --- |
| Nguồn | Chưa qua codec | Reader đã khóa, bbox, nhãn, layout test | Accuracy ≥95%; thấp hơn thì phép đo chưa hợp lệ |
| q2 MSE | Thêm codec MSE | Cùng source, reader, bbox | Có thể giảm accuracy, đặc biệt chữ nhỏ/tương phản thấp |
| q2 + Mean | Chỉ thay synthesis bằng adapter Mean | Cùng actual stream, reader, bbox | Có thể giảm nhận dạng so với MSE, như tín hiệu OCR screen trước; chưa giả định mức giảm |

Chỉ số chính: chênh lệch accuracy Mean − MSE theo từng ảnh, đếm đúng→sai/sai→đúng; bootstrap ghép cặp 10.000 lần theo ảnh, CI 95%. Chỉ số phụ: accuracy theo nhóm đã định nghĩa font/size/contrast/background và actual bpp. CI theo ảnh không cho suy luận tổng quát ra mọi họ font. Không dùng accuracy của bốn ký tự để thay cho whole-word OCR/human readability.

Đối chứng công cụ được khóa sau khi reader qua source gate, **trước khi đọc accuracy codec**: Tesseract 5.5.3, eng, psm 10, oem 1, whitelist `68EF`; cùng bbox generator và crop phóng 4× bicubic. Chạy trên nguồn/MSE/Mean, không sửa tham số theo kết quả. Đây là chỉ số phụ, không thay CNN làm endpoint chính sau khi nhìn số liệu.

## Tài nguyên, chạy và dừng

Tham chiếu vận hành: các screen q2 cùng runtime đã qua kiểm fresh parity. Thay đổi có chủ đích là dữ liệu mới và CNN thay 1-NN; không đổi checkpoint codec hay adapter. Không lấy mặc định framework làm cấu hình nghiên cứu.

Ước tính 10–30 phút trên một L40; giới hạn cứng 2 giờ wall time cho chuỗi stage trên một GPU (≤2 GPU-giờ). Dưới khoảng 1 GB đầu ra, không API trả phí. Giới hạn VRAM như các screen trước. Smoke kiểm generate/train trước; output chính dùng thư mục mới. Monitoring qua `status.json`, `training.json`, `codec.progress.json`, log từng stage và VRAM guard; không cần dịch vụ tracker mới.

Lệnh chạy từ repo server, sau khi kiểm paths và smoke:

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/mlicpp/run_recognition_gate.sh \
  --stage run --config experiments/mlicpp/recognition_gate.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/cvpr20261105/source-qualified-glyph-cnn-q2
```

Đây là pilot xác minh phép đo. Không huấn luyện model nén lớn. Nếu reader đủ tốt nhưng chưa có bằng chứng latent giữ thông tin, chỉ được kết luận về nhận dạng từ ảnh tái tạo; không tự quy lỗi cho entropy bottleneck hay decoder. Muốn so trực tiếp latent cần readout riêng có năng lực, dữ liệu train codec thực và đối chứng phù hợp; không dùng latent 1-NN yếu trước đó làm bằng chứng.

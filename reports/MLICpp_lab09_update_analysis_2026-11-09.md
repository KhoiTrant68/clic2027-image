# Phân tích cập nhật portable từ lab09

Nguồn chỉ đọc: `D:/AwesomeCV/clic2027_mlicpp_portable`, campaign `results/lab09-migrated-fullrd-3rates-2seeds`. Không sửa nội dung thư mục người dùng cung cấp.

## Kết luận

Đây là kết quả chạy lại đầy đủ trên máy mới, không phải cập nhật thuật toán: không thiếu file trong manifest gói gốc và không phát hiện thay đổi nội dung sau khi chuẩn hóa LF/CRLF. Source archive của campaign khớp code portable. Giá trị của cập nhật là xác nhận pipeline chạy được và bổ sung một kết quả âm rõ ràng: recipe fine-tune hiện tại chưa vượt checkpoint MLIC++ gốc; learned steps chưa tạo lợi ích nhất quán.

## Kiểm bằng chứng

- 12 lượt train complete, mỗi lượt 2.000 bước; config train nhất quán. Tổng thời gian train ghi nhận 3,789 giờ, mỗi lượt khoảng 18,5–19,7 phút. Không suy ra tên GPU hay lợi thế phần cứng chỉ từ tốc độ này.
- 15 cấu hình đánh giá trên cùng 32 tên ảnh. Tính lại bpp, pooled PSNR, các metric trung bình và timing từ metrics.json khớp comparison.json tới sai số dưới 1e-9.
- Kiểm trực tiếp 480 file bitstream: số byte và SHA-256 đều khớp metadata.
- Kiểm trực tiếp 480 PNG: số pixel và hash pixel RGB đều khớp metadata.
- Các fresh.json báo tổng 480 lượt qua kiểm giải mã. Kiểm hash artifact không đồng nghĩa đã tự chạy lại decoder ở máy local.
- Không có checkpoint `.pt` trong bản cập nhật; chỉ có `SHA256SUMS.checkpoints` và hash trong decoder.json. Không thể tự tái chạy inference hoặc xác minh tensor trọng số từ thư mục này.
- SQLite chấm mù có 0 sessions, 0 trials, 0 votes. Study được chuẩn bị không có nghĩa đã xác nhận human preference.
- Không có ảnh nguồn test đầy đủ độc lập để tính lại toàn bộ metric: kiểm trên đây xác nhận số liệu tổng hợp và tính nhất quán artifact, không phải đánh giá lại LPIPS/DISTS từ đầu.

## Learned steps so với baseline, cùng bitrate nội suy

Nội suy tuyến tính theo log(bpp), chỉ trong vùng giao khoảng 0,129–0,281 bpp. Bảng là trung bình mô tả của năm điểm grid trong matched_rate.json, không phải BD-rate, không phải trung bình trên toàn phân bố rate và không phải kiểm định thống kê.

| Seed | Δ pooled PSNR (joint − baseline) | Δ LPIPS | Δ DISTS |
|---|---:|---:|---:|
| 9041 | −0,001917 dB | +0,000428 | −0,000297 |
| 9042 | +0,004798 dB | +0,000270 | +0,000307 |

LPIPS tăng là xấu hơn ở cả hai seed xét trung bình grid. DISTS đổi dấu giữa seeds; PSNR khác biệt rất nhỏ và cũng đổi dấu. Không có cơ sở coi đây là cải thiện ổn định.

## Fine-tune so với checkpoint gốc

Ở cả hai seeds, cả hai nhánh fine-tune đều kém checkpoint gốc về pooled PSNR, LPIPS và DISTS tại cả năm điểm grid đã nội suy. PSNR thấp hơn khoảng 0,079–0,109 dB. Không ngoại suy kết luận ra mọi bitrate hoặc mọi recipe fine-tune.

Ví dụ ở q2, seed 9041 (chưa cùng bitrate):

| Model | bpp | pooled PSNR | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| Gốc | 0,194132 | 30,6893 | 0,170164 | 0,137498 |
| Baseline fine-tune | 0,197523 | 30,6648 | 0,169941 | 0,138133 |
| Joint learned steps | 0,197436 | 30,6610 | 0,170387 | 0,137578 |

Một LPIPS thấp hơn chút ở baseline q2 không đủ kết luận thắng vì dùng nhiều byte hơn; nội suy cùng rate cho thấy hướng ngược lại.

## Giới hạn và hệ quả

- Đây là loss RD/MSE và learned steps, không phải synthesis perceptual đã thử ở server cũ. Kết quả âm không phủ định hướng perceptual.
- Ba điểm rate không phủ mục tiêu 0,075 bpp và không tự đáp ứng chính xác 0,15/0,3 bpp. Không phải gói đã chứng nhận phù hợp CLIC.
- Blind study chỉ so baseline với joint seed 9041 quanh q2, lệch bitrate trung bình khoảng 0,0438%. Không phải so với checkpoint gốc, bộ codec mix, hay xác nhận rate từng ảnh bằng nhau.
- Hai seeds và 32 ảnh hạn chế độ tin cậy. Chênh lệch giữa hai máy không tự là lỗi: deterministic trên một runtime không đảm bảo tái lập bit-exact xuyên phần cứng/runtime. Chưa có đủ thông tin để quy nguyên nhân cho thiết bị, dữ liệu hay thuật toán.
- Dữ liệu test này đã được campaign trước đánh giá. Nếu dùng kết quả để chọn hướng mới, cần tập kiểm định cuối khác để tránh chọn theo test lặp lại.

## Đề xuất

Giữ MLIC++ gốc làm baseline RD; không mở rộng riêng learned steps với recipe này. Giữ bộ codec mix làm đối chứng mục tiêu thi. Hướng ưu tiên là decoder perceptual cùng stream và phủ ba ngân sách byte, đánh giá trên bộ ảnh thích hợp rồi chấm mù thật. Nếu tiếp tục fine-tune RD, trước hết kiểm validation curve và các giả thuyết lệch objective estimated-rate/actual-rate, dữ liệu ít, lịch học; chưa kết luận nguyên nhân khi chưa có phép thử.

Trước khi chuyển model hoặc train tiếp từ lab09, lấy đủ 12 `last.pt` (hoặc ít nhất checkpoint cần dùng), xác minh SHA-256 theo manifest đi kèm và lưu runtime/package versions. Chưa có lý do chạy lại toàn bộ campaign này chỉ để xác nhận những artifact đã kiểm.

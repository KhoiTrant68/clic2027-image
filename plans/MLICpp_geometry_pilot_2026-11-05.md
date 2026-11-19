# Pilot lượng tử hóa theo hình học decoder

Trạng thái ban đầu: thiết kế trước khi chạy; chưa có kết quả. Người dùng cho phép triển khai và train thử.

## Câu hỏi và phạm vi

Với backbone MLIC++ q2 MSE đóng băng, phân bổ bước lượng tử hóa theo độ nhạy của decoder có cải thiện rate–distortion không? Một mạng nhỏ chỉ đọc hyperprior đã giải mã có dự đoán được phân bổ này tốt hơn thang cố định không?

Đây là pilot **đường chéo theo kênh**, chưa phải lattice xoay đầy đủ hoặc đóng góp mới đã được xác nhận. Không dùng decoder perceptual đã fine-tune để tránh thay hai yếu tố. Teacher dùng hình học RGB (Phi = identity), chưa dùng hình học LPIPS.

## Thiết kế

| Nhánh | Thang tương đối | Train | Thông tin decoder |
|---|---|---|---|
| uniform | 1 | không | cấu hình |
| global | độ nhạy trung bình tập train | thống kê teacher | model asset |
| predicted | mạng MLP từ trung bình/độ lệch chuẩn hyperprior | 1.000 bước Adam, lr 0.001, batch 16 | hyperprior đã giải mã |
| oracle | độ nhạy từng ảnh | không | gửi 320 float32 = 1.280 byte/ảnh |

Tất cả dùng cùng quantizer checkerboard, scale table và LRP gốc. Quét hệ số chung {0.8, 1.0, 1.25, 1.6}, không chọn lại theo test. Header bổ sung được tính ở mọi nhánh. Oracle chỉ là teacher heuristic có nhiễu, không phải nghiệm tối ưu hay cận trên được chứng minh.

Teacher: 4 vector Rademacher, VJP của synthesis tại latent được giải mã bằng codec gốc; lấy trung bình bình phương gradient theo vị trí và mẫu để ước lượng đường chéo J^T J theo kênh. Bước tương đối có log được chiếu vào [-log 2, log 2] và tổng log bằng 0. Đây là ràng buộc thể tích cell, **không bảo đảm bitrate bằng nhau**. LRP và context thay đổi sau lượng tử hóa; xấp xỉ hình học không mô hình hóa đầy đủ các hiệu ứng đó.

Train: 128 crop từ cache train hiện có (64 ảnh nguồn), validation: 16 crop riêng. Test: một crop 256² từ mỗi ảnh DIV2K mới, shuffle seed 1984, indices [354:370]. Kiểm tra SHA và không trùng nguồn với các tập pilot trước. Crop seed 9041 theo tên ảnh. Tiền huấn luyện MLIC có thể đã nhìn thấy nguồn DIV2K: chưa kiểm toán.

Seed train 9041, FP32, TF32 off, 2 CPU threads, không DataLoader worker. MLP 1280→128→320, đầu ra khởi tạo tại log metric trung bình. Train bằng loss dự đoán log metric; **chưa train end-to-end RD**. Không chọn checkpoint bằng test; dùng bước 1000.

## Kiểm tra và phân tích đã chốt

1. Smoke: identity scale phải tạo payload và pixel giống codec gốc; ảnh train/smoke không lấy từ test mới. Sai thì không chạy chính.
2. Mỗi bitstream phải giải mã lại từ file ở tiến trình mới, khớp SHA pixel; decoder predicted chỉ đọc hyperprior và checkpoint, không đọc source/teacher/cache.
3. Báo cáo bpp thực, RGB MSE/PSNR, LPIPS và DISTS. Primary: thay đổi MSE tại bitrate chung, nội suy tuyến tính từng cặp trong khoảng giao của đường RD trung bình; không ngoại suy. Bootstrap theo ảnh 2.000 lần cho chênh lệch ở scale 1, kèm chênh lệch bitrate, không gọi đây là so sánh cùng rate.
4. Oracle báo cả payload diagnostic và tổng bitrate có side information, không dùng payload-only để tuyên bố thắng codec.
5. Không có cải thiện thì báo kết quả âm; không tự mở rộng train dài hoặc đổi hyperparameter theo test.

## Tài nguyên và theo dõi

Một L40 GPU 0; guard mặc định cap allocator 18 GiB và tổng GPU dưới nửa VRAM. Dự kiến 15–40 phút tùy teacher/eval; hard timeout 90 phút gồm các stage. Log JSONL, progress/status JSON, checkpoint, source snapshot và SHA; không cần dịch vụ tracking ngoài. Dung lượng dự kiến dưới 2 GiB (không tính checkpoint backbone dùng chung). Không đụng tác vụ khác.

## Kết quả thực hiện

Đã chạy xong ngày 05/11/2026, 332,8 giây cho pilot chính; 256/256 fresh decode và 160/160 crop audit đạt. MLP 205.248 tham số đã train 1.000 bước. Nội suy trong khoảng bitrate giao nhau cho MSE predicted cao hơn uniform 1,29%; khoảng giao hẹp và chưa có CI cho thống kê nội suy. Không mở rộng recipe này thành train dài. [Báo cáo kết quả và giới hạn](../reports/MLICpp_geometry_pilot_2026-11-05.md).

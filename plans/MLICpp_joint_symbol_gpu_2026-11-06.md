# Chạy tìm kiếm phối hợp symbol trên A30

User yêu cầu sử dụng GPU 2 đang trống và tăng mức sử dụng VRAM. Tái sử dụng pilot lookahead; thay CPU bằng 6 worker CUDA, mỗi worker xử lý một nguồn development riêng, không cấp phát bộ nhớ giả để giữ chỗ.

Tham chiếu cấu hình: `lookahead_pilot.json`. Giữ seed 9041, checkpoint legacy q2, lambda 0,0035, crop 256 × 256 và dữ liệu development. Thay 4 nguồn bằng 6 nguồn đầu validation, 3 bước bằng 8 bước `[0,2,4,6,8,10,14,18]`, tìm kiếm 4 vòng phối hợp. Mỗi nhóm có 5 candidate: ±1 trên hai vị trí gần biên và flip block 8 vị trí. Chấp nhận phương án chỉ khi J file thật giảm; 161 phép đánh giá mỗi ảnh, tổng 966 phép đánh giá. Không đọc geometry test, không train neural weights.

Không tiếp tục áp trần tổng nửa GPU của pilot cũ vì user đã yêu cầu sử dụng nhiều VRAM trên GPU đang trống. Mỗi worker có allocator cap 10% của A30; tổng allocator cap 60%, phần còn lại dành cho CUDA context, thư viện và headroom. Monitor mỗi 2 giây, dừng chỉ worker của run nếu trống dưới 1.536 MiB; yêu cầu VRAM ban đầu dưới 512 MiB. Không thay đổi job GPU 0/1. Các cap không phải reservation và không ngăn job khác khởi chạy.

Nguồn script được snapshot, progress JSON/JSONL theo worker, GPU memory JSONL và status toàn run. Output mới, timeout 1 giờ. Identity bắt buộc khớp codec native trước search; cuối run decoder mới kiểm baseline và joint stream từng ảnh. So sánh nội bộ cùng GPU; chưa có đối chứng latent optimization cùng runtime, nên giảm J không xác lập novelty. ETA cập nhật từ throughput thực sau launch; dự kiến vài đến vài chục phút, không dùng làm cam kết.

Tên run: `div2k-mlicpp-q2-joint-symbol-6workers-4sweeps`. Bộ dữ liệu, checkpoint và interpreter đã tồn tại trên server qua pilot trước; launcher kiểm tra manifest và checkpoint khi load. Không optimizer, AMP hay dataloader workers.

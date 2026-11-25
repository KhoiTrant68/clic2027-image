# Pilot tác động của symbol lên phần mã hóa phía sau

Khóa thiết kế trước khi chạy ngày 06/11/2026. Tham chiếu: `operational_quantizer.json` và [đề xuất nghiên cứu](../reports/MLICpp_novelty_search_2026-11-06.md).

- Cùng backbone MLIC++ legacy q2, SHA được kiểm tra bởi `probe.load`, lambda MSE 0,0035 và crop 256 × 256.
- Thay tìm gain bằng can thiệp integer tại bước 0, 8, 16 trong 20 bước checkerboard. Hai vị trí có residual gần biên nhất, mỗi vị trí thử ±1; thêm một action đổi 8 vị trí gần biên theo hướng vượt biên. 15 action + baseline/ảnh; không ghép các action thắng sau khi xem kết quả.
- Chọn 4 nguồn validation đầu của manifest geometry, đã được dùng làm development; không mở tập test geometry. Không train critic/backbone trong pilot này.
- Khóa hyperstream và prefix. Suffix tiếp tục theo lượng tử gốc nhưng tính lại context/LRP. Số bit chính là toàn bộ file gồm header gốc. Không thêm side information/action header: symbol được truyền trực tiếp.
- So lựa chọn theo J thực với baseline. Phân tích riêng việc bỏ future-rate bằng cùng final distortion và cùng candidate: comparator này chỉ là chẩn đoán, không phải baseline search cùng runtime. CDF có escape sẽ không được dùng cho phép phân rã rate.
- Gate tham khảo: giảm J ít nhất 1% so với comparator bỏ future-rate và CI bootstrap ảnh 95% dưới 0. Nếu đạt vẫn phải chạy local search/SGA cùng runtime trước khi train dài. Với 4 ảnh development, CI chỉ có tính thăm dò.
- Invariant bắt buộc: identity khớp payload/pixel codec gốc; can thiệp đúng integer; prefix/hyperstream không đổi; fresh-process decode tất cả stream khớp hash pixel.
- Tài nguyên: cả 3 GPU đang 100%, VRAM vượt headroom guard. Chạy CPU với 2 luồng trên server 96 CPU, RAM available khoảng 189 GB. CUDA bị ẩn, chỉ bỏ lời gọi synchronize phục vụ timer trong subprocess CPU. Không chỉnh job khác hay nới guard GPU.
- Monitoring: status JSON, progress JSONL từng candidate, log tmux, metrics và source snapshot. Giới hạn 3.600 giây. ETA sẽ ước lượng lại từ thời gian candidate đầu; chưa có benchmark CPU đáng tin cậy.
- Tên run: `div2k-mlicpp-q2-cpu-symbol-suffix-4dev`. Không ghi đè output cũ, không tự chạy lại cùng thư mục khi lỗi.

Kiểm tra cấu hình so với run trước: số nguồn giảm 8→4 (chẩn đoán CPU); bỏ train/gain/sweep/factor; thêm danh sách bước, vị trí và block; device CPU, 2 threads; seed/lambda/weights/reference giữ nguyên. Không optimizer, LR, AMP, dataloader workers hoặc gradient accumulation vì không training.

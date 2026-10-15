# MS-ILLM: kiểm stream thật trước khi ghép decoder toàn mix

Ngày 15/10/2026. Chạy trên L40, Torch 2.11.0+cu128, hai CPU thread. Chọn đúng 5 ảnh MS-ILLM trong allocation mix 0.3 bpp hiện tại (bốn q4, một q5), giữ nguyên checkpoint. Đây là kiểm deployment, chưa train thêm.

## Kết quả

| Phép kiểm | Kết quả |
|---|---|
| Cấu hình GPU mặc định, encoder và decoder ở hai tiến trình | 5/5 ảnh có pixel lệch; kiểm trực tiếp ghi nhận sai lệch tối đa 1/255 |
| Smoke deterministic, một ảnh | Khớp pixel hoàn toàn |
| Deterministic, đầy đủ 5 ảnh | **5/5 khớp pixel hoàn toàn** |
| Metadata thực trong stream | 28 byte/ảnh, thay cho 4 byte ước lượng trước đây |
| Tăng dung lượng của 5 điểm | **120 byte**, payload entropy giữ nguyên độ dài |
| Tổng byte mix sau riêng hiệu chỉnh MS-ILLM | **3,231,412 / 3,234,201 byte** budget; còn 2,789 byte |
| Wall time tiến trình decoder deterministic | **19.565 giây / 5 ảnh**, gồm import, nạp hai model, decode và ghi PNG |

Stream truyền magic/version, quality, kích thước ảnh, kích thước đã pad, chiều dài hai entropy payload và payload. Decoder đọc quality/kích thước từ stream, không nhận ảnh gốc hoặc metadata ảnh từ encoder. Nạp runtime và checkpoint local với strict state-dict; Python TCP bị chặn. Đây chưa phải gói di chuyển độc lập vì vẫn dùng runtime/checkpoint ở đường dẫn của server.

Cấu hình deterministic bật `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, tắt `cudnn.benchmark` và đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi khởi tạo CUDA. Kết quả ủng hộ việc cố định đường tính toán GPU để tránh sai lệch lượng tử hóa đầu ra; chưa cô lập từng kernel để kết luận nguyên nhân chi tiết. Không suy rộng kết quả này thành bảo đảm trên máy hoặc kiến trúc GPU khác.

## Ảnh hưởng tới kết luận chất lượng

Ảnh deterministic **không giống hoàn toàn baseline cũ**: bốn ảnh sai lệch tối đa 1 mức màu, một ảnh tối đa 3 mức; MAE nằm khoảng 0.00354–0.01036 trên thang 0–255. Vì thế **chưa gán Q̂ 1.4978 của mix cũ cho cấu hình decoder mới**. Cần chấm lại năm ảnh, rồi tính lại điểm mix. Không cộng 19.565 giây này với các benchmark PULSE khác để khẳng định thời gian toàn mix.

Các lượt GPU đều dùng guard nửa thiết bị và allocator tối đa 18 GiB, thấp hơn hạn mức 30 GB/GPU. Log VRAM của từng tiến trình được lưu cùng kết quả. Sau khi hoàn tất, hai GPU trở về 0% utilization, 1,247 MiB/device do dịch vụ nền.

## Bằng chứng và việc tiếp theo

- [Đối chiếu 5 ảnh và budget](../results/2026-10-15_msillm_stream_audit/msillm-selected-mix03-deterministic-full/verification.json).
- [Profile encode](../results/2026-10-15_msillm_stream_audit/msillm-selected-mix03-deterministic-full/encode.json) và [decode](../results/2026-10-15_msillm_stream_audit/msillm-selected-mix03-deterministic-full/decode.json), gồm SHA256 checkpoint và stream.
- [Runner](../experiments/new_codecs/verify_msillm.py), [framing](../experiments/new_codecs/msillm_stream.py) và [unit test](../tests/test_msillm_stream.py).

Ưu tiên tiếp theo: chấm lại output deterministic; lưu và kiểm stream của một ảnh DiT-IC cùng hai ảnh VTM-SCC; sau đó mới ghép dispatcher và đo toàn mix từ tiến trình sạch. Vòng này đã hoàn tất, không còn job GPU chạy nền.

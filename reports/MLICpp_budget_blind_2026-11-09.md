# Phân bổ byte và chuẩn bị chấm mù

## Công việc đã triển khai

- `budget_blind.py` chọn một factor trong 0,5 / 1 / 2 / 4 cho từng ảnh, tối ưu giảm LPIPS bằng greedy dưới trần byte toàn tập. Đây không phải nghiệm tối ưu toàn cục.
- Tạo ZIP riêng cho decoder gốc/perceptual. Dung lượng kiểm cuối gồm stream, manifest chứa factor cho từng ảnh và overhead ZIP. Hai ZIP có cùng lựa chọn stream và cùng kích thước, khác mode decoder 0/1.
- Nếu mức thấp nhất vẫn vượt ngân sách, ghi `feasible=false`; không giả lập bitrate bằng nội suy hoặc đổi kích thước ảnh.
- Tạo bộ chấm mù cho mỗi ngân sách khả thi, ảnh tham chiếu và A/B cùng kích thước. Công cụ ẩn danh codec, ngẫu nhiên A/B, lưu phiếu; chưa có phiếu người thật.

ZIP này là container thí nghiệm; chưa có kiểm giải mã end-to-end từ ZIP mới hoặc chứng nhận format submission CLIC. Các stream nguồn đã qua fresh decode ở từng factor. Giữ `manifest.json` vì factor là thông tin cần cho decoder; không bỏ overhead khi báo bpp.

## Hai bộ dữ liệu, không trộn kết quả

1. **Development 32 ảnh:** dùng lại recon full-resolution đã hoàn thành để phân bổ 0,075 / 0,15 / 0,3 bpp và chuẩn bị ba study. Bộ này đo hiệu quả decoder trên cùng stream; không phải test độc lập.
2. **CLIC validation 30 ảnh:** pipeline mới kiểm đúng 30 ảnh, 86.245.376 pixel và lưu SHA-256. Chạy 8 cấu hình đầy đủ trên GPU 1, fresh decode, rồi tự phân bổ và tạo study. Bộ ảnh này cũng đã được dự án dùng trước, không phải test challenge tương lai.

## Đối chứng mix lịch sử

Thêm bước `compare_budget_history.py`, chờ pipeline CLIC xong rồi kiểm danh sách tên 30 ảnh và so sánh LPIPS/DISTS với mix trong báo cáo 15/10. Không so mean PSNR mới với pooled PSNR cũ. Bảng giữ cả bitrate thực của hai bên.

Các giới hạn: ảnh tái tạo mix chưa tìm được tại đường dẫn artifact dự kiến trên server; chưa kiểm lại hash pixel nguồn của lịch sử, chưa fresh-decode mix, objective phân bổ khác nhau và overhead có thể khác. Vì vậy đây chỉ là đối chiếu metric lịch sử, chưa phải so sánh trực tiếp bằng người chấm hoặc bằng chứng thắng mix mạnh nhất hiện tại.

## Tài nguyên và theo dõi

GPU 1 có job khác sau khi thí nghiệm trước kết thúc. Lượt mới chỉ dùng allocator tối đa 8 GiB, yêu cầu 10 GiB trống trước khi chạy và trước mỗi cấu hình; không kill hoặc sửa job khác. Nếu OOM hoặc headroom giảm, dừng và giữ log. Tốc độ chịu ảnh hưởng của job đồng thời, không dùng để xếp hạng decoder.

Sessions:

- `mlic-dev32-budgets-20261109`: phân bổ và tạo ảnh chấm mù trên CPU.
- `mlic-clic30-budgets-20261109`: GPU infer/eval rồi phân bổ/chấm mù.
- `mlic-mix-compare-20261109`: chờ kết quả CLIC để đối chiếu metric lịch sử.

Đường dẫn server:

```text
/home/jovyan/clic2027-artifacts/mlicpp/perceptual-full-20261109/dev32-budgets-blind
/home/jovyan/clic2027-artifacts/mlicpp/perceptual-full-20261109/clic30-budget-study
```

Khi study hoàn thành, chạy ví dụ trên server (không gửi private manifest cho người chấm):

```bash
cd /home/jovyan/clic2027-image
/home/jovyan/clic2027-envs/mlicpp/bin/python tools/blind_score.py serve \
  --study /home/jovyan/clic2027-artifacts/mlicpp/perceptual-full-20261109/dev32-budgets-blind/bpp-0.15/study \
  --port 8766
```

Từ máy cá nhân có SSH alias đã cấu hình:

```bash
ssh -N -L 8766:127.0.0.1:8766 nexai-sv20
```

Mở `http://127.0.0.1:8766`. Chỉ chạy serve sau khi có `private_manifest.json` và bước tạo study đã hoàn thành. Đổi study sang `bpp-0.075` hoặc `bpp-0.3` để chấm mức khác. Chưa công khai service hoặc gửi lời mời chấm trong phiên này.

## Kiểm tra code

Đã kiểm trên dữ liệu tổng hợp: mọi ZIP khả thi không vượt ngân sách, hai decoder có kích thước package bằng nhau, manifest giữ đúng decoder mode, tạo được ba study. Kiểm cú pháp server đã qua. Chưa có kết quả hoàn chỉnh của pipeline CLIC khi viết báo cáo này.

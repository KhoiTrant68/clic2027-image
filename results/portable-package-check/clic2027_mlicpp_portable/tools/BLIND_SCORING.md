# Chấm mù chất lượng ảnh theo cặp

Tool chạy trên trình duyệt, dùng Python + Pillow có sẵn trong project. Không cần GPU, Node hoặc dịch vụ ngoài. Người chấm xem ảnh gốc cùng A/B, chọn **A tốt hơn / Khó phân biệt / B tốt hơn / Bỏ qua**. Đây là preference có ảnh tham chiếu, không phải thang điểm MOS.

## Chạy thử ngay

Từ thư mục repo:

```powershell
python tools/blind_score.py demo --out results/blind-scoring-demo
python tools/blind_score.py serve --study results/blind-scoring-demo/study
```

Mở **http://127.0.0.1:8765**. Demo có ba ảnh tổng hợp để thử thao tác, không phải số liệu codec thật. Lệnh tạo study yêu cầu thư mục output chưa tồn tại, tránh ghi đè dữ liệu chấm.

## Chuẩn bị ảnh thật

Mỗi thư mục codec chứa recon có **tên file và kích thước trùng ảnh gốc**. Ví dụ:

```text
study-input/
  config.json
  reference/001.png
  reference/002.png
  mlicpp/001.png
  mlicpp/002.png
  proposed/001.png
  proposed/002.png
  vtmscc/001.png
  vtmscc/002.png
```

`config.json` (đường dẫn tương đối tính từ file config):

```json
{
  "reference": "reference",
  "codecs": {
    "MLIC++ q2": "mlicpp",
    "Proposed q2": "proposed",
    "VTM-SCC": "vtmscc"
  },
  "question": "So với ảnh gốc, bản nào có chất lượng thị giác tốt hơn?"
}
```

```powershell
python tools/blind_score.py prepare --config study-input/config.json --out results/blind-scoring-study
python tools/blind_score.py serve --study results/blind-scoring-study --port 8765
```

Tool tạo mọi cặp codec trên từng ảnh, xáo thứ tự riêng cho mỗi người và random vị trí A/B. Có thể dùng `--trials 40` để lấy tối đa 40 câu ngẫu nhiên cho **phiên mới**. Đây là lấy mẫu ngẫu nhiên, không bảo đảm cân bằng chính xác số lượt cho mọi cặp/ảnh; mặc định chấm tất cả cặp. Các phiên đã tạo giữ nguyên lịch.

Khi so codec ở cùng bitrate, người tổ chức phải chuẩn bị recon theo ngân sách byte phù hợp trước khi import. Tool không tự kiểm bitrate, thời gian decode hoặc tự ghép điểm RD tương đương.

Ảnh được chuyển thành PNG RGB 8-bit mới, bỏ metadata và đổi tên opaque. Không dùng tool hiện tại cho nghiên cứu HDR, alpha hoặc color-management chuyên biệt. Bản nguồn không bị sửa. Chưa hỗ trợ tên file khác nhau giữa các codec; cần chuẩn bị mapping/tên file trước khi import.

## Cách chấm

- Chọn zoom vừa khung, 100% hoặc 200%. Cả ba ảnh dùng cùng tỉ lệ; cuộn một khung sẽ cuộn cả ba. 100% là pixel CSS, không bảo đảm một pixel màn hình vật lý trên màn hình HiDPI.
- Phím `1`: A; `2`: khó phân biệt; `3`: B; `4`: bỏ qua. Chỉ cho chọn sau khi cả ba ảnh tải xong.
- Phiếu được ghi vào SQLite ngay khi gửi, chống ghi trùng khi retry. Đóng/mở trang bằng cùng trình duyệt sẽ tiếp tục câu chưa chấm.
- Mỗi người dùng một hồ sơ trình duyệt riêng. Không thu tên/email. Không có cơ chế xác minh một người chỉ dùng một phiên.
- Nếu mất mạng sau khi gửi, tải lại trang để tiếp tục. Không xóa localStorage trong một phiên đang chấm.

## Xuất kết quả — dành cho người tổ chức

```powershell
python tools/blind_score.py export --study results/blind-scoring-study --out results/blind-scoring-export
```

- `votes.csv`: từng phiếu, mã người chấm, ảnh, khóa A/B, lựa chọn, thời gian xem tự báo từ trình duyệt và timestamp Unix.
- `summary.json`: win/tie/skip theo cặp; preference share tính win + 0,5 × tie, chia số phiếu có đánh giá. Skip không nằm trong mẫu số.

Không xem bảng tổng hợp khi đang chấm để tránh gợi ý. Các thống kê này chỉ mô tả dữ liệu, chưa là kiểm định ý nghĩa; phiếu cùng người hoặc cùng ảnh có tương quan. Khi viết paper, cần protocol cố định, số người/ảnh đủ lớn và phân tích phù hợp theo người/ảnh. Tool chưa có câu kiểm tra độ chú ý, lặp đo độ nhất quán, Elo/Bradley–Terry hoặc CI.

## Chạy trên server

Tool mặc định chỉ nghe localhost. Có thể chạy trên server rồi tunnel:

```powershell
ssh -L 8765:127.0.0.1:8765 nexai-sv20
```

Trong terminal server, chạy `python tools/blind_score.py serve --study <study-folder>`. Sau đó mở localhost trên máy người chấm. Có `--host 0.0.0.0` cho mạng nội bộ tin cậy; bản này không có TLS/login/quản lý truy cập công khai.

Giữ riêng `private_manifest.json` và `ratings.sqlite3`. HTTP chỉ phục vụ giao diện, API đã bỏ tên codec và asset opaque; không chia sẻ thư mục study hoặc mount nó bằng static file server khác. Blindness ngăn lộ nhãn phương pháp, không ngăn người chấm suy đoán từ đặc trưng ảnh.

Test: `python -m unittest discover -s tests -p test_blind_score.py -v`.

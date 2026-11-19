# Chấm mù CLIC qua Cloudflare Free

Gói này dùng lại giao diện chấm mù của dự án. Worker tự cấp phiên cho người truy cập và lưu phiếu trong D1, không cần nhập mã. Chỉ thư mục `deployment/public` được tải lên dưới dạng static assets. Ánh xạ A/B nằm trong D1. Các mã mời cũ vẫn được giữ để tương thích; lịch vòng 2 là nguồn để tạo các phiên mới.

## Chuẩn bị

Chạy từ `tools/blind-cloud` sau khi cài Node.js và `pnpm install`. `<study>` là thư mục đã tạo bởi `tools/blind_score.py prepare`, có `private_manifest.json` và `assets/`.

```powershell
python prepare.py --study <study> --out deployment --people 20 --trials 15
Copy-Item wrangler.example.jsonc wrangler.jsonc
```

Với bộ 30 ảnh, 20 mã × 15 cặp chia 10 lượt dự kiến cho mỗi ảnh nếu mọi người hoàn thành. A/B cân bằng trong lịch đã chia và thứ tự câu được xáo riêng cho mỗi mã. Mỗi người nhận một mã duy nhất. Giữ cùng mã khi chuyển thiết bị hoặc trình duyệt.

## Đưa lên Cloudflare

Người có quyền trên tài khoản tự đăng nhập và cấp quyền cho Wrangler. Dùng Workers Free và D1 Free.

```powershell
node node_modules/wrangler/bin/wrangler.js login --scopes account:read user:read workers:write workers_scripts:write d1:write
node node_modules/wrangler/bin/wrangler.js d1 create clic-blind-study --location apac
```

Ghi ID D1 trả về vào `wrangler.jsonc`, thay ID toàn số 0. Nếu đã có database cùng tên, xác nhận đó là database của đúng vòng chấm trước khi nhập dữ liệu. Dùng database riêng cho bản thử nếu người thử sẽ bỏ phiếu. Khởi tạo schema và lịch một lần rồi triển khai.

```powershell
node node_modules/wrangler/bin/wrangler.js d1 execute clic-blind-study --remote --file schema.sql
node node_modules/wrangler/bin/wrangler.js d1 execute clic-blind-study --remote --file deployment/seed.sql
node node_modules/wrangler/bin/wrangler.js deploy
```

Wrangler trả link `https://…workers.dev`. Trước khi gửi, mở link từ trình duyệt khác, kiểm mã hợp lệ, ảnh hiển thị và tải lại trang vẫn tiếp tục. Dùng tài khoản Cloudflare Free và tránh bật tính năng trả phí.

Khi chỉ sửa giao diện của vòng chấm hiện tại, chạy `python prepare.py --page-only --out deployment` rồi chạy `wrangler deploy`. Bước này chỉ tạo lại HTML, giữ mã mời, lịch chấm và phiếu đã lưu.

## Gửi cho người chấm

> Mời bạn chấm chất lượng ảnh theo cặp tại `<link>`. Không cần đăng ký hay nhập mã. Bấm “Bắt đầu chấm 15 câu”. Hãy xem ảnh gốc và chọn bản A hoặc B mà bạn thấy tốt hơn. Chọn “Hai bản tương đương” nếu chất lượng tương đương và “Bỏ qua câu này” khi không thể đánh giá. Dùng màn hình máy tính nếu có thể. Bạn có thể đóng trang và quay lại bằng cùng trình duyệt.

Chỉ cần gửi đường dẫn. `invitations.csv` là dữ liệu quản trị cũ, không cần gửi cho người chấm.

## Bảng kết quả và vòng chấm mở rộng

- `/results` là bảng kết quả công khai, không cần mã quản lý. Bảng điểm và ảnh tái tạo được công khai theo xác nhận của chủ dự án. `catalog.py` gom kết quả theo bộ ảnh, giao thức, model và mức bitrate; chọn một hàng có chi tiết để xem điểm từng ảnh. Các kết quả không tìm được đủ ảnh vẫn có thể xem ở bảng số liệu hoặc danh mục nguồn.
- Vòng `clic30-v1` dùng danh sách `deployment/invitations.csv`: MLIC++ gốc và perceptual ở 0.15 bpp. Vòng `clic30-v2` dùng `deployment/round-v2/invitations.csv`: bốn decoder MLIC++ (gốc, perceptual, single, mixed) ở ba ngân sách 0.075, 0.15 và 0.30 bpp. Hai bản trong mỗi cặp giải mã từ cùng bitstream. Mỗi người có 15 câu trên 15 ảnh khác nhau, 5 câu mỗi ngân sách. Người chấm đã dùng vòng 1 có thể bấm **Đổi mã mời** để vào vòng 2.
- `round_assets.py` xác minh ảnh và bitstream rồi đóng gói ảnh với mã ngẫu danh. `multi_round.py` sinh SQL bổ sung cùng mã mời riêng. Chỉ ảnh, giao diện và catalog đi vào `deployment/public`; mã mời, khóa quản lý, manifest và SQL để bên ngoài. Triển khai ảnh trước rồi mới nhập `deployment/round-v2/seed.sql` vào D1 hiện tại. Chỉ nhập seed một lần. Bản cập nhật không thay mã mời hay phiếu của vòng trước.

Trang `/` hiển thị danh sách đợt chấm, số câu mỗi lượt, bitrate và các cặp model. Có thể gửi trang tổng hoặc gửi thẳng liên kết `/?round=<id>` của đợt cần chấm. Trong phiên chỉ hiện A/B. Nút **Xem ảnh gốc** lật từng khung để đối chiếu, và lăn chuột trên ảnh để zoom đồng bộ. Mỗi đợt lưu tiến độ riêng trong cùng trình duyệt. Đổi thiết bị hoặc xóa cookie sẽ tạo một phiên khác, nên số phiên không đồng nghĩa với số người duy nhất.

### Thêm đợt khi có bản decoder mới

Đặt ID phiên bản rõ ràng trong `per_image.json`, ví dụ `mixed-v3`, và dùng một ID đợt mới, ví dụ `clic30-v3`. Chọn bản đối chứng và mức bitrate cần đánh giá. Ví dụ sau tạo 5 câu mỗi lượt ở 0,075 bpp cho Mixed v3 so với MLIC++ gốc, trên cùng bitstream.

```powershell
python round_assets.py --budgets <frozen-budgets> --extended <eval-v3> --out deployment/clic30-v3-assets --model mixed-v3 --rate 0.075
python multi_round.py --assets deployment/clic30-v3-assets --public deployment/public --out deployment/round-v3 --round clic30-v3 --model original --model mixed-v3 --rate 0.075
node node_modules/wrangler/bin/wrangler.js deploy
node node_modules/wrangler/bin/wrangler.js d1 execute clic-blind-study --remote --file deployment/round-v3/seed.sql
```

Triển khai ảnh trước khi nhập SQL. Đợt tự xuất hiện trên trang khi nhập seed thành công. Gửi `/?round=clic30-v3` cho người chấm. Nếu cần sửa ảnh hoặc đổi model sau khi đã có phiếu, dùng ID đợt mới để giữ đúng phiên bản gắn với phiếu đã thu. Gói này kiểm tra hai decoder dùng cùng bitstream, nên thí nghiệm thay encoder cần giao thức ghép theo ngân sách bitrate riêng.

## Xuất phiếu

```powershell
$query = Get-Content -Raw export_v2.sql
node node_modules/wrangler/bin/wrangler.js d1 execute clic-blind-study --remote --command $query --json | Set-Content -Encoding utf8 export.json
python export.py export.json export --versioned
```

Nếu chưa nhập seed vòng 2, dùng `export.sql` và bỏ `--versioned`. `export/votes.csv` ghi từng phiếu; `export/summary.json` tách số người, thắng, hòa và bỏ qua theo vòng, ngân sách và cặp codec. Xuất dữ liệu trước khi kết thúc hoặc gỡ site. Để khóa nhận thêm phiếu ở cả hai vòng trong khi vẫn giữ link, đặt biến Worker `STUDY_CLOSED` bằng `1` và triển khai lại.

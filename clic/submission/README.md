# Khung decoder toàn mix và bộ kiểm tra local

Phần này hoàn thiện **định dạng gói, điều phối decoder và kiểm tra đầu ra** khi server GPU chưa hoạt động. Công cụ chạy được trên CPU với Python ≥3.10 và Pillow. Các worker giải mã thực tế vẫn cần runtime, trọng số và phần cứng tương ứng.

## 1. Phần đã làm và phần còn thiếu

| Hạng mục | Trạng thái |
| --- | --- |
| Đọc gói thư mục hoặc ZIP, kiểm manifest và hash stream | Đã có |
| Kiểm danh sách ảnh độc lập, số ảnh và kích thước | Đã có |
| Tính dung lượng stream, metadata và phần phụ trội của ZIP | Đã có |
| Điều phối bốn codec, chạy mỗi nhóm codec trong một tiến trình mới | Đã có; kiểm trên CPU bằng worker giả lập |
| Từ chối đầu ra thiếu/thừa ảnh, PNG lỗi, sai kích thước hoặc sai hash pixel | Đã có |
| Đọc header MS-ILLM, đối chiếu quality và kích thước với manifest | Đã có; dùng chung mã framing với runner nghiên cứu |
| Kiểm header PULSE | Đã có cho 4 byte framing ngoài; chưa phân tích envelope bên trong |
| Cấu hình gọi decoder PULSE-L và MS-ILLM hiện có | Có ví dụ; chưa chạy xác nhận với trọng số thật |
| Worker stream DiT-IC và VTM-SCC theo giao diện này | Chưa có; thiếu worker sẽ báo lỗi trước khi giải mã |
| Chạy toàn mix với stream thật, đo GPU/L4, chấm lại Q̂ | Chưa thực hiện |

Đây là **khung tích hợp local**, chưa phải decoder đã được xác nhận trên hệ thống chấm. Không có chế độ tự thay decoder thật bằng worker giả lập.

## 2. Gói truyền đi

Gói đầu vào có đúng cấu trúc dưới đây; không chứa runtime hoặc trọng số:

```text
bs.zip
├── manifest.json
└── streams/
    ├── ten_anh_1.bin
    └── ten_anh_2.bin
```

`manifest.json` chứa toàn bộ thông tin điều phối cần truyền. Ví dụ về cấu trúc, cần thay hash minh họa bằng SHA-256 thực của tệp `.bin`:

```json
{
  "version": 1,
  "images": [
    {
      "name": "ten_anh_1",
      "codec": "msillm",
      "quality": 4,
      "width": 2048,
      "height": 1360,
      "sha256": "<64 ký tự hex của SHA-256 stream>"
    }
  ]
}
```

- `name` là tên ảnh **không có phần mở rộng**. Công cụ đọc `streams/{name}.bin` và yêu cầu đầu ra `{name}.png`.
- Tên chỉ dùng chữ Latin không dấu, chữ số, `_`, `-`; tối đa 128 ký tự, bắt đầu bằng chữ hoặc số. Không cho phép đường dẫn, tên trùng không phân biệt hoa/thường hoặc tên thiết bị dành riêng trên Windows.
- `codec` nhận `ditic`, `msillm`, `pulse`, `vtmscc`. `quality` lần lượt nằm trong 0–4, 1–6, 0–4, 0–63; với VTM-SCC, đây là QP. Đây là miền nhãn của định dạng nội bộ, không có nghĩa đã đóng gói được mọi checkpoint.
- `width` và `height` là kích thước gốc, không phải kích thước sau padding. Mỗi chiều tối đa 65.536; diện tích tối đa 100 triệu pixel.
- `sha256` tính trên **toàn bộ stream gốc, bao gồm header riêng của codec**. Không gỡ header `MI` hay `PL` trước khi đóng gói.
- Thư mục đầu vào không được chứa tệp dư hoặc symlink. ZIP được đọc trực tiếp, không giải nén các đường dẫn do gói cung cấp. Giới hạn nội bộ: tổng dữ liệu 256 MiB, manifest 4 MiB; đây không phải giới hạn của cuộc thi.

Các stream từ bake-off chưa chắc đã lưu đủ metadata để đóng gói theo định dạng này. Khi server hoạt động lại, cần xuất các stream được chọn bằng encoder tương ứng; không thể suy ra payload từ CSV metric hoặc coi tệp `.pt` của kit L4 là stream nộp trực tiếp.

## 3. Danh sách ảnh kỳ vọng

Bộ kiểm tra nhận một tệp `expected.json` **nằm ngoài gói truyền đi**, được lập từ danh sách ảnh phải giải mã. Không dùng chính manifest trong `bs.zip` làm đối chứng; nếu làm vậy, việc thiếu ảnh trong cả manifest lẫn ZIP có thể bị bỏ sót.

```json
{
  "images": [
    {"name": "ten_anh_1", "width": 2048, "height": 1360}
  ]
}
```

Có thể thêm `pixel_sha256` vào từng hàng để kiểm tra độ khớp pixel. Hash được tính trên `Image.open(path).convert("RGB").tobytes()`, theo thứ tự RGB của ảnh tham chiếu đã giải mã, không phải hash tệp PNG và không phải ảnh gốc trước nén. Khi không có trường này, công cụ chỉ kiểm định dạng/kích thước và ghi lại hash đầu ra, không tuyên bố khớp pixel.

## 4. Cách chạy

Chạy từ thư mục gốc repo. Trên Windows, thay `python` bằng đường dẫn Python thực tế nếu cần.

**Kiểm gói, chưa gọi decoder:**

```bash
python clic/tools/check_submission.py --input bs.zip --expected expected.json --budget-bpp 0.3 --report preflight.json
```

**Giải mã bằng các worker đã cấu hình:**

```bash
python clic/submission/decode.py --input bs.zip --runtime clic/submission/runtime.json --output images --report decode-report.json
```

**Giải mã và kiểm tra với danh sách ảnh độc lập trong cùng lượt chạy:**

```bash
python clic/tools/check_submission.py --input bs.zip --expected expected.json --budget-bpp 0.3 --runtime clic/submission/runtime.json --output images --report checked-decode.json
```

**Kiểm thư mục ảnh đã giải mã:**

```bash
python clic/tools/check_submission.py --input bs.zip --expected expected.json --output images
```

Thư mục đầu ra khi giải mã và tệp báo cáo phải chưa tồn tại. Báo cáo phải nằm ngoài thư mục đầu vào và thư mục ảnh đầu ra. Gói nguồn không bị sửa. Lỗi kiểm tra hoặc worker trả về mã thoát khác 0 làm lệnh thất bại; không xuất bộ ảnh dở dang nếu một worker hoặc kiểm tra ảnh thất bại. `--timeout` đặt thời gian tối đa cho từng worker, mặc định 300 giây.

Trên Linux, có thể gọi `bash clic/submission/decode` từ gốc repo sau khi chuẩn bị `runtime.json`; script mặc định đọc `bs.zip` và ghi `images/`. Muốn đóng gói độc lập cần đưa vào gói decoder cả `src/clic27`, entrypoint, worker, runtime và trọng số cần thiết; không chỉ sao chép riêng script `decode`.

### Cách tính byte

Báo cáo tách `stream_bytes`, `metadata_bytes`, `uncompressed_bytes`, `archive_bytes`. Kiểm ngân sách dùng giá trị bảo thủ:

```text
accounted_bytes = max(tổng byte các tệp gồm manifest, kích thước ZIP)
budget_bytes = floor(budget_bpp × tổng pixel / 8)
```

Với đầu vào là thư mục, không có `archive_bytes`. Công cụ không tự khẳng định đây là cách hệ thống chấm tính rate; cần đối chiếu lại khi tích hợp devkit. Việc thêm manifest và metadata có thể khiến allocation cũ vượt ngân sách, nên phải chạy bộ phân bổ lại với chi phí thực trước khi chốt gói.

## 5. Giao diện worker

`runtime.json` thuộc **gói decoder tin cậy**, không lấy từ `bs.zip`. Sao chép [runtime.example.json](runtime.example.json), sửa đường dẫn runtime/trọng số và bổ sung worker còn thiếu. Ví dụ chỉ có MS-ILLM và PULSE; chưa đủ để giải mã mix bốn codec.

Mỗi worker là một danh sách đối số lệnh, chạy không qua shell. Đường dẫn tương đối được tính từ thư mục chứa `runtime.json`. Các biến được hỗ trợ:

| Biến | Giá trị |
| --- | --- |
| `{python}` | Python đang chạy bộ điều phối |
| `{work}` | Thư mục làm việc riêng của codec |
| `{input}` | Thư mục `streams/` chứa stream gốc của codec đó |
| `{output}` | Đường dẫn `decoded/` chưa tồn tại; worker tạo thư mục và ghi PNG RGB |
| `{metadata}` | `selection.json`, gồm `version` và danh sách `images` lấy từ manifest truyền đi |
| `{report}` | Đường dẫn báo cáo riêng của worker, nằm ngoài thư mục ảnh |

Mỗi codec được gọi một lần với tất cả ảnh đã chọn cho codec đó; worker chịu trách nhiệm nạp từng checkpoint khi có nhiều quality. Đầu ra phải có đúng các PNG đã yêu cầu, không ghi log hoặc JSON vào thư mục ảnh. Bộ điều phối chỉ công bố thư mục ảnh cuối sau khi tất cả nhóm đã qua kiểm tra.

Ví dụ MS-ILLM gọi lại `verify_msillm.py --stage decode --deterministic`, hiện chỉ có ánh xạ trọng số cho q4/q5. Ví dụ PULSE gọi `decode.py` của gói do `package_pulse.py` tạo ra. Hai runner đều cần môi trường GPU đã chuẩn bị và `CUDA_VISIBLE_DEVICES` trỏ tới một GPU. Cần chạy xác nhận trên stream thật trước khi sử dụng kết quả.

Các biến môi trường offline và thiết lập hai luồng CPU được truyền cho worker; đây không phải giới hạn CPU bằng container. Chúng **không chặn mạng ở cấp hệ điều hành**. Khi kiểm triển khai phải dùng container có `--network none`; bộ chạy Docker hiện có ở [run_like_server.sh](../l4/run_like_server.sh). Thời gian trong báo cáo local gồm thời gian worker chạy ở tiến trình mới; chưa thay thế số đo toàn quy trình ở môi trường đích, và chưa đo đỉnh RAM/VRAM.

## 6. Kiểm thử CPU và việc tiếp theo

```bash
python -m unittest discover -s tests -p test_submission.py -v
python -m unittest discover -s tests -p "test_*stream.py" -v
```

Worker tại `tests/fixtures/submission_worker.py` chỉ dùng trong kiểm thử: payload là PNG mẫu, có bọc framing MS-ILLM/PULSE khi phù hợp. Các ca này kiểm luồng điều phối và xử lý lỗi, không kiểm entropy decoder, trọng số hay chất lượng ảnh của codec.

Khi có lại server:

1. Xuất stream thật, lập manifest và danh sách ảnh kỳ vọng độc lập; chạy preflight để kiểm lại ngân sách.
2. Chuẩn bị runtime/trọng số PULSE-L và MS-ILLM, chạy riêng từng nhánh rồi tính lại metric trên đầu ra deterministic.
3. Bổ sung worker đọc stream DiT-IC và VTM-SCC; chưa đăng ký worker thì lệnh phải tiếp tục báo lỗi.
4. Chạy toàn mix trên L40, kiểm hash pixel với ảnh tham chiếu, rồi đo và kiểm chéo trên L4.

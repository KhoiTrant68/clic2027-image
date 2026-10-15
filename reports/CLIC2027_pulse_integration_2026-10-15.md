# PULSE-L: tích hợp bake-off và kiểm decoder offline

Ngày: 15/10/2026. Tiếp nối vòng sàng lọc model mới. Vòng này chạy lại bằng CLI bake-off chính, bổ sung q3 và kiểm gói PULSE-L độc lập; chưa train, chưa chấm mù và chưa đo trên L4.

## 1. Những việc đã hoàn thành

- Thêm candidate `pulse` vào `experiments/bakeoff/bakeoff.py`, opt-in qua CLI, dùng bundle/runtime local.
- Chạy đủ **150 điểm: 30 ảnh × q0/1/2/3/4**, lưu stream thật và chấm qua metric của repo. Tách interpreter Torch 2.6 của codec và Torch 2.11 của bộ metric bằng runner một lệnh.
- Thêm **4 byte header thực** cho codec/version/flags. Quality, kích thước và định danh bundle nằm trong envelope PULSE. Byte dùng bởi allocator chính là kích thước file, không cộng header giả lần thứ hai.
- Đối chiếu **120/120 payload** q0/1/2/4 với run trước: giống hệt sau khi bỏ header mới. q3 bổ sung 30 điểm. Baseline được sao chép sang thư mục mới, không chỉnh sửa bản gốc.
- Đóng gói decoder, chạy từ thư mục sạch và tiến trình mới, không có ảnh gốc trong đầu vào decoder; kiểm **30/30 ảnh khớp pixel** ở cả hai phiên bản writer PNG.

## 2. Kết quả chất lượng

![So sánh Q̂ của mix trước và sau PULSE-L](figures/pulse_2026-10-15/quality.png)

*Điểm tuyệt đối và mức tăng so baseline; Q̂ chưa được xác nhận bằng chấm mù.*

Q̂ v0, cùng 30 ảnh và allocator của repo; cao hơn là tốt hơn. “Mix mới” thêm PULSE-L q0–q4 vào các điểm baseline DiT-IC, student, MS-ILLM, CoD-Lite và VTM-SCC.

| Phương án | 0.075 bpp | 0.15 bpp | 0.3 bpp |
|---|---:|---:|---:|
| Mix baseline trước PULSE | 0.1836 | 0.7582 | 1.4215 |
| Vòng sàng lọc PULSE-L q0/1/2/4 | 0.1841 | 0.7891 | 1.4912 |
| **Mix tích hợp q0–q4, header thật cho PULSE** | **0.1841** | **0.7891** | **1.4978** |
| PULSE-L riêng, q0–q4 | Không vừa budget | Không vừa budget | 1.4234 |

Tại 0.3 bpp, mix mới chọn **22 ảnh PULSE-L, 5 MS-ILLM, 2 VTM-SCC và 1 DiT-IC**. Tổng stream là 3,231,292 byte, tương ứng 0.299730 bpp. PULSE-L riêng dùng 3,231,720 byte, tương ứng 0.299770 bpp.

Việc thêm q3 làm đầy đường RD và nâng điểm tại 0.3; không có thay đổi trọng số model. ΔQ̂ của mix mới so baseline là khoảng +0.0309 ở 0.15 và +0.0763 ở 0.3. Đây vẫn là metric sàng lọc, chưa có bằng chứng chấm mù cho mix mới.

Nguồn: [summary chính](D:/AwesomeCV/ratflow-codec/results/2026-10-15_pulse_integration/integrated_bakeoff/summary.json), [allocation](D:/AwesomeCV/ratflow-codec/results/2026-10-15_pulse_integration/integrated_bakeoff/allocation.csv), [kiểm tính toàn vẹn](D:/AwesomeCV/ratflow-codec/results/2026-10-15_pulse_integration/integration_checks.json).

![Số ảnh được chọn cho từng codec tại 0.3 bpp](figures/pulse_2026-10-15/allocation.png)

*Tỷ trọng theo số ảnh, không phải tỷ trọng byte bitstream.*

## 3. Tổng thời gian offline: phát hiện nút thắt ghi PNG

![Thời gian offline và VRAM](figures/pulse_2026-10-15/runtime_memory.png)

*Thời gian dành cho PULSE-L riêng trên L40. “Phần còn lại” là hiệu giữa tổng wall time và các mục profile, không phải phép đo riêng. VRAM dùng GB thập phân; bảng bên dưới giữ đơn vị MiB gốc.*

Phần cứng L40; Torch 2.6.0+cu124, hai CPU thread. Chọn allocation PULSE-L riêng ở 0.3 trên đủ 30 ảnh. Mỗi bản chạy từ thư mục mới, giải nén ZIP rồi gọi Python `-I`; không dùng repo nguồn làm đường import. Python TCP bị chặn trong decoder, bật offline flags và dùng HF cache mới. Đây không phải cô lập mạng ở cấp hệ điều hành.

| Đại lượng | V1: PNG mặc định | V2: PNG lossless, compression level 0 |
|---|---:|---:|
| Giải nén gói | 0.610 s | 0.592 s |
| Wall time tiến trình | 40.763 s | 11.642 s |
| **Tổng wall time** | **41.374 s** | **12.235 s** |
| Nạp model | 0.624 s | 0.667 s |
| Tổng API decode | 2.319 s | 2.452 s |
| Chuyển/ghi PNG | 34.885 s | 5.754 s |
| Peak RSS tiến trình | 1,037,264 KiB | 1,041,648 KiB |
| Đỉnh tổng VRAM thiết bị quan sát | 4,130 MiB | 4,130 MiB |
| Ảnh khớp pixel với bake-off | 30/30 | 30/30 |

V1 cho thấy ghi PNG chiếm phần lớn thời gian. V2 chỉ tắt nén lại PNG đầu ra, giữ nguyên phép đổi tensor sang uint8; ảnh đầu ra lớn hơn nhưng pixel không đổi. Bitrate codec được tính trên stream đầu vào, không phải PNG tái tạo. Thời gian wall gồm giải nén ZIP, khởi động Python, kiểm checksum, import, nạp model, đọc stream, decode và ghi PNG. Không gồm bước chuẩn bị bộ stream hoặc so sánh pixel sau chạy. Hai bản đều dùng cùng selection và cùng stream.

Đây là một lần chạy mỗi cấu hình, có thể chịu ảnh hưởng cache hệ điều hành và tải máy. 12.235 giây là số **PULSE-L riêng trên L40**, không phải toàn mix, L4 hoặc kết quả server CLIC. Không cộng thời gian API của các codec đo ở môi trường khác nhau để ước lượng chính xác thời gian mix.

Nguồn: [V1](D:/AwesomeCV/ratflow-codec/results/2026-10-15_pulse_integration/offline_decode_v1/benchmark.json), [V2](D:/AwesomeCV/ratflow-codec/results/2026-10-15_pulse_integration/offline_decode_v2/benchmark.json), [profile V2](D:/AwesomeCV/ratflow-codec/results/2026-10-15_pulse_integration/offline_decode_v2/decode.json).

## 4. Gói decoder và tài nguyên

Gói V2: **74,727,260 byte ZIP**, **122,568,001 byte giải nén**. Có code PULSE, rANS extension, ba file bundle, giấy phép nguồn, decoder wrapper và manifest checksum. Chưa đóng gói Python/Torch/NumPy/Pillow/CUDA; extension hiện được build cho Linux x86_64 và Python 3.12 của server. Vì vậy kích thước này chưa phải xác nhận submission CLIC hợp lệ.

SHA256 ZIP: `30ec65ab1d437ed289a66f5532ddae96af29d3253cac7785a8f5d2368f1934a1`.

Toàn lượt bake-off/metric ghi nhận đỉnh tổng VRAM **5,920 MiB**. Các bước GPU đều có guard nửa L40, allocator tối đa 18 GiB; còn thấp hơn hạn mức người dùng cho phép **30 GB/GPU**. Guard phần mềm lấy mẫu tổng VRAM, không phải phân vùng cứng.

Gói lớn, stream và PNG ở server: `/mnt/home/guess03/clic/research_20261015/pulse_decoder_v2.zip`, `integrated_bakeoff`, `offline_decode_v2`. Chỉ artifact nhẹ được tải về repo. Giữ nguyên các run cũ và checkout server vốn có thay đổi chưa commit.

## 5. Kiểm tra và phần chưa hoàn thành

Đã qua hai unit test framing (round-trip/byte accounting; từ chối stream ngắn, sai version/flags), kiểm cú pháp script, 150 điểm thực và hai lượt độc lập 30 ảnh. File `integration_checks.json` xác nhận đủ lưới q×ảnh, số byte stream, 120 payload cũ khớp và parity của output. `source_hashes.json` ghi snapshot source đã dùng cho phép đo.

Chưa hoàn thành gói decoder **toàn mix**: DiT-IC/MS-ILLM/VTM-SCC vẫn là điểm baseline đã lưu; vòng này chưa ghép và đo chúng trong một executable chung. Header PULSE đã hiện thực, còn định dạng/mode của các nhánh khác phải được kiểm ở gói mix cuối. Chưa đo L4/cross-machine parity, chưa có chấm mù nhiều người và chưa train xuống rate thấp.

Thứ tự tiếp theo: (1) dùng writer PNG nhanh cho các decoder sau khi kiểm pixel; (2) xây dispatcher và gói mix, đo chi phí load thực; (3) chạy L4 khi có máy, chấm mù mix cũ/mới; (4) mới quyết định ngân sách fine-tune. Hướng mới về lựa chọn codec có tính chi phí load nên dùng số đo toàn gói này thay cho thời gian API đơn thuần.

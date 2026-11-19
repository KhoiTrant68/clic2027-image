# MLIC++: thử nghiệm tác động tiếp diễn của quyết định lượng tử hóa

**Kết luận:** Pilot chạy thành công, nhưng chưa có bằng chứng đủ mạnh để train critic. Trong tập phương án hữu hạn đã thử, chọn theo RD thực giảm mục tiêu trung bình **0,1701%** so với codec gốc. Riêng việc thêm chi phí CDF của phần mã hóa phía sau làm đổi lựa chọn trên 2/4 ảnh, nhưng J trung bình hơi xấu hơn đối chứng bỏ chi phí đó, **+0,00138%**. Không đạt gate thăm dò 1% đã đặt.

## Thiết kế và phần đã thực hiện

[Protocol khóa trước khi chạy](../plans/MLICpp_lookahead_pilot_2026-11-06.md), từ [giả thuyết nghiên cứu](MLICpp_novelty_search_2026-11-06.md).

- MLIC++ MSE legacy q2, toàn bộ 116.716.928 tham số backbone đóng băng; không train critic hoặc neural backbone.
- 4 crop 256 × 256 từ 4 nguồn development đã dùng trước: `0206`, `0214`, `0239`, `0586`. Không mở test geometry để chọn phương án.
- Mỗi ảnh có baseline và 15 can thiệp độc lập: tại các bước checkerboard 0, 8, 16, chọn hai vị trí gần biên làm tròn nhất và thử ±1; thêm một phương án đổi 8 vị trí gần biên theo hướng vượt biên. Không cộng dồn các action thắng.
- Can thiệp trực tiếp integer symbol ở encoder; hyperstream và prefix giữ nguyên. Toàn bộ context, LRP và suffix được tính lại theo lượng tử gốc.
- Dùng định dạng bitstream và decoder gốc, không thêm action header hay side information. Mục tiêu là \(J=\mathrm{bpp}_{file}+0.0035\,\mathrm{MSE}_{RGB,255}\), đo trên pixel uint8 sau clamp/round.
- GPU đều bận và thiếu headroom; chạy **2 luồng CPU trên server**, không cấp phát GPU và không chỉnh các job khác. Toàn bộ search + fresh decode hoàn tất trong **175,27 giây**; mỗi candidate encode/decode khoảng **1,50 giây**.

Nhánh CPU chỉ bỏ các lời gọi CUDA synchronize phục vụ timer trong subprocess. Loader được bổ sung tùy chọn device, mặc định GPU giữ nguyên. Kết quả chỉ so nội bộ cùng CPU; chưa kiểm tra tính xác định CPU–GPU.

## Kết quả đo bằng file thật

| Cách chọn phương án | Bpp thực trung bình | MSE RGB ↓ | J ↓ |
|---|---:|---:|---:|
| Codec gốc | 0,129761 | 54,74394 | 0,321364527 |
| J thực nhỏ nhất trong tất cả candidate | 0,130005 | 54,51798 | 0,320817817 |
| J thực nhỏ nhất trong candidate đủ điều kiện phân tích CDF | 0,129761 | 54,65284 | 0,321045672 |
| Bỏ future-rate, vẫn dùng distortion cuối | 0,129883 | 54,62388 | 0,321066378 |
| Có future-rate CDF, dùng cùng distortion cuối | 0,129639 | 54,69490 | 0,321070810 |

Đây là **cùng lambda, không cùng bitrate**. Giảm J 0,1701% không phải giảm bitrate 0,1701%; thực tế nhánh chọn tốt nhất trong toàn bộ candidate có bpp hơi cao hơn baseline. Chưa đo đường RD hoặc BD-rate.

Đối chứng “bỏ future-rate” vẫn cần biết distortion ảnh cuối. Nó chỉ cô lập một phần cơ chế, **không phải local search thực dụng có cùng thời gian chạy**, và không thay thế đối chứng Code Editing/SGA.

### Kiểm tra lại tập candidate và escape symbol

Có 5/64 phương án không đủ điều kiện phân rã CDF ở đoạn bị tác động vì có escape symbol. Chúng vẫn có bitstream và RD thực hợp lệ, nhưng không được giả định chi phí escape bằng một CDF entry thông thường.

Vì thế, con số **−0,0774%** trong `analysis.json` ban đầu so best-all với đối chứng CDF là so hai tập candidate không hoàn toàn giống nhau; không dùng làm bằng chứng hiệu quả cơ chế. [Audit độc lập](../results/2026-11-06_mlicpp_lookahead/local_audit.json) giữ nguyên artifact gốc và tính lại:

| So sánh | Thay đổi J tương đối | Chênh lệch J tuyệt đối, CI bootstrap 95% |
|---|---:|---:|
| Best-all so với baseline | −0,17012% | [−0,00096331; −0,00013011] |
| Best-eligible so với bỏ future-rate, cùng candidate hợp lệ | −0,00645% | [−0,00004260; 0] |
| Có future-rate CDF so với bỏ future-rate, cùng candidate hợp lệ | +0,00138% | [−0,00002782; +0,00004111] |

Bootstrap 2.000 lần theo ảnh, seed 9041. Chỉ có 4 ảnh development; CI không chứng minh khả năng tổng quát hóa. Best-all chứa baseline trong không gian lựa chọn nên luôn không tệ hơn baseline theo J đã đo: CI âm của phép so sánh đó không tự nó chứng minh đóng góp mới hoặc lợi ích của critic.

## Cơ chế có quan sát được không?

Có. Một can thiệp trong tập thử làm thay đổi tới **1.102 symbol ở suffix**, dù chỉ sửa một nhóm nhỏ tại bước đang xét. Tính thêm future-rate CDF làm đổi lựa chọn trên **2/4 ảnh**. Tuy nhiên, “context thay đổi nhiều” không đồng nghĩa với nén tốt hơn; lợi ích ròng đang rất nhỏ và không nhất quán.

Kết quả cũng cho thấy tổng CDF cost và số byte của file có thể xếp hạng khác nhau ở mức chênh lệch nhỏ. Không được thay byte thực bằng likelihood rồi tuyên bố đã tối ưu rate vận hành.

## Kiểm tra tính đúng và bằng chứng

- Identity khớp cả payload và pixel với encode/decode gốc trên 4 ảnh.
- Mỗi can thiệp sửa đúng integer đã chỉ định; prefix symbol/index và hyperstream không đổi.
- **64/64 bitstream** giải mã trong tiến trình mới bằng decoder gốc khớp SHA pixel; không cài hook encoder và không đọc ảnh nguồn trong tiến trình decode.
- Tải artifact về workspace và kiểm tra **79 checksum file**, gồm 64 stream; tính lại byte, bpp, J, phép chọn candidate và bootstrap độc lập.
- Chưa train model mới; chưa đo LPIPS, DISTS, Q̂, human preference hoặc giới hạn runtime CLIC.

Code: [pilot](../experiments/mlicpp/lookahead_pilot.py), [config](../experiments/mlicpp/lookahead_pilot.json), [audit](../experiments/mlicpp/audit_lookahead.py). Bằng chứng: [thư mục kết quả](../results/2026-11-06_mlicpp_lookahead).

Server run: `/home/jovyan/clic2027-artifacts/mlicpp/lookahead-20261106/div2k-mlicpp-q2-cpu-symbol-suffix-4dev`.

SHA256 archive: `4e49dbf72353a6be978258b8a8f0e01c466168beb84a50ce394b633ba37abda1`.

Checkpoint SHA256: `5577dcccc7d154c07b9cc14f222606717cac12ac263f569cba11b012994e9fde`; upstream revision: `4aa9d2a512eb9db382172058e6c278fb0d376e17`.

## Quyết định sau pilot

**Không train critic dài từ kết quả hiện tại.** Teacher chỉ thử perturbation đơn/nhóm nhỏ chưa tạo lợi ích đủ lớn; distill một teacher yếu khó tạo đóng góp đáng kể.

Pilot chưa bác bỏ toàn bộ hướng quyết định tuần tự: candidate còn rất hẹp, chưa tối ưu phối hợp nhiều bước, chưa học representation và mới xét bốn crop. Nếu tiếp tục, phép thử có ý nghĩa là tìm kiếm phối hợp một số block với ngân sách encode cố định, so trực tiếp với latent optimization cùng ngân sách trước khi thu dữ liệu train critic. Đó là thí nghiệm khác, chưa được thực hiện trong báo cáo này. Không đưa các số crop development vào bảng benchmark 30 ảnh.

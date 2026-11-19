# Thử lớp hiệu chỉnh chi tiết trên CLIC30 ở 0,075 bpp

Phép thử dùng lại bitstream và ảnh giải mã đã lưu của MLIC++ Perceptual ở bốn hệ số lượng tử. Bộ ảnh Mixed chưa có đủ bốn lựa chọn cho từng ảnh, nên Perceptual được dùng nhất quán cho lớp nền và đối chứng. Mixed vẫn là mốc chất lượng bên ngoài phép đối chứng.

Encoder lấy sai số độ sáng giữa ảnh gốc và lớp nền, chia ảnh thành ô 16×16 và chọn tối đa một hệ số DCT từ tám mẫu cố định mỗi ô. Điểm chọn ưu tiên sai số ở vùng có gradient lớn. Mỗi token tốn 7 byte, gồm chỉ số ảnh, vị trí ô, mã mẫu và biên độ có dấu. Gói ZIP tính cả các token, tiêu đề và chi phí đóng gói. Phương pháp dùng codebook cố định để thăm dò giá trị của việc chuyển byte sang lớp chi tiết.

| Gói | Byte |
| --- | ---: |
| Perceptual dùng toàn bộ ngân sách | 802.034 |
| Perceptual giảm ngân sách lớp nền | 766.278 |
| Lớp nền giảm ngân sách và token chi tiết | 802.029 |

Trình thử đọc lại packet từ ZIP rồi áp dụng nó vào ảnh lớp nền đã giải mã trước. Kiểm tra tổng hợp đã xác nhận khôi phục đúng pixel với token có dấu và từ chối hai token cùng ô. Đây là thí nghiệm lớp bổ sung với dữ liệu validation. Tích hợp bộ giải mã lớp nền và lớp bổ sung vào cùng entry point của devkit là bước riêng.

Đối chứng trực tiếp là gói Perceptual dùng toàn bộ ngân sách. So sánh với lớp nền giảm ngân sách cho biết token bù lại được bao nhiêu chất lượng sau khi rút byte khỏi lớp nền. LPIPS-Alex, DISTS và PSNR được tính trên 30 ảnh đầy đủ. Kết quả chưa có chấm mù hoặc đối chiếu RDVQ. [Mã phép thử](../experiments/mlicpp/sparse_detail_pilot.py)

| Gói | bpp thực | LPIPS-Alex ↓ | DISTS ↓ | PSNR TB ↑ |
| --- | ---: | ---: | ---: | ---: |
| Perceptual dùng toàn bộ ngân sách | 0,074396 | **0,21398** | **0,17638** | **27,7636** |
| Perceptual giảm ngân sách lớp nền | 0,071079 | 0,22380 | 0,18268 | 27,5651 |
| Lớp nền giảm ngân sách và token chi tiết | 0,074395 | 0,22347 | 0,18240 | 27,5744 |

5.093 token cùng overhead dùng 35.751 byte. Token bù khoảng 3,4% phần LPIPS-Alex bị mất khi giảm byte của lớp nền. Gói bổ sung vẫn kém gói dùng toàn bộ ngân sách cho Perceptual trên cả ba chỉ số. Kết luận chỉ áp dụng cho codebook DCT cố định, một token mỗi ô, cách chấm điểm có trọng số gradient và mức phân bổ byte đã thử.

Dữ liệu từng ảnh, ba gói ZIP và ảnh kết quả được lưu trên server tại /home/jovyan/clic2027-artifacts/mlicpp/sparse-detail-clic30-0075-20261111.

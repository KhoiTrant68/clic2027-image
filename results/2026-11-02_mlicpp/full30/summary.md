# Thử nghiệm MLIC++ trên 30 ảnh

Checkpoint MSE chính thức, kiến trúc legacy theo hướng dẫn tác giả; không huấn luyện lại.

| Phương án | Trần bpp | bpp thực | PSNR gộp | LPIPS-Alex ↓ | DISTS ↓ | Q̂ ↑ | Ảnh dùng MLIC++ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlicpp_q1 | - | 0.11732 | 29.07 | 0.2355 | 0.1644 | -1.7157 | 30 |
| mlicpp_q2 | - | 0.17939 | 30.70 | 0.1904 | 0.1374 | -0.9372 | 30 |
| mlicpp_q3 | - | 0.26105 | 32.33 | 0.1532 | 0.1124 | -0.2198 | 30 |
| mlicpp_q4 | - | 0.39348 | 34.13 | 0.1184 | 0.0853 | 0.5535 | 30 |
| frozen_pulse_mix | 0.075 | 0.07492 | 22.38 | 0.0847 | 0.0711 | 0.1841 | 0 |
| reoptimized_reference_pool | 0.075 | 0.07492 | 22.38 | 0.0847 | 0.0711 | 0.1841 | 0 |
| pulse_mix_plus_mlicpp | 0.075 | 0.07492 | 22.40 | 0.0844 | 0.0709 | 0.1853 | 1 |
| frozen_pulse_mix | 0.15 | 0.14991 | 24.59 | 0.0653 | 0.0536 | 0.7891 | 0 |
| mlicpp_allocated | 0.15 | 0.14982 | 29.55 | 0.1993 | 0.1359 | -0.9476 | 30 |
| reoptimized_reference_pool | 0.15 | 0.14991 | 24.59 | 0.0653 | 0.0536 | 0.7891 | 0 |
| pulse_mix_plus_mlicpp | 0.15 | 0.14991 | 24.59 | 0.0653 | 0.0536 | 0.7891 | 0 |
| frozen_pulse_mix | 0.3 | 0.29973 | 29.34 | 0.0364 | 0.0338 | 1.4978 | 0 |
| mlicpp_allocated | 0.3 | 0.29913 | 31.96 | 0.1327 | 0.0946 | 0.2566 | 30 |
| reoptimized_reference_pool | 0.3 | 0.29973 | 29.34 | 0.0364 | 0.0338 | 1.4978 | 0 |
| pulse_mix_plus_mlicpp | 0.3 | 0.29973 | 29.34 | 0.0364 | 0.0338 | 1.4978 | 0 |

## Độ phủ bitrate

- 0.075 bpp: các checkpoint MLIC++ đã thử chưa đủ thấp để nén toàn tập trong ngân sách.
- 0.15 bpp: có allocation MLIC++ đủ 30 ảnh.
- 0.3 bpp: có allocation MLIC++ đủ 30 ảnh.

## Phạm vi kết luận

- Stream MLIC++ tính cả header thực 31 byte/ảnh, không dùng entropy ước lượng.
- Đã kiểm hash pixel giữa giải mã trong tiến trình encode và một tiến trình decoder mới trên cùng GPU; phần cứng cụ thể được ghi trong qN-decode.json. Chưa kiểm L4 hoặc khác máy.
- Baseline là mix PULSE-L lịch sử. Metadata của các nhánh cũ chưa được kiểm lại toàn bộ; đây chưa phải so sánh hai gói nộp cuối.
- Mix mở rộng được tối ưu lại trên toàn bộ point pool baseline và MLIC++; không chỉ thay thế các điểm đã chọn trước đó. Hàng reoptimized_reference_pool là đối chứng cùng bộ phân bổ, chưa thêm MLIC++.
- Dùng cùng module metric, tile và Q̂ v0 của repo. Môi trường PyTorch/CompressAI khác lần benchmark cũ; chênh lệch nhỏ cần được kiểm chứng lại.
- Không có chấm mù trong thử nghiệm này; Q̂ là mô hình sàng lọc.
- Thời gian từng ảnh trong q*-decode.json là decode API, process_times.json gồm nạp mô hình và ghi PNG. Không suy ra tốc độ toàn mix bằng cách cộng các API riêng.

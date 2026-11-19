# Đánh giá CPU để tiếp tục khi GPU bận

## Kết quả đã hoàn thành

Đã đánh giá q1 gốc và bốn checkpoint fine-tune trên cùng 8 crop trung tâm 256 × 256 lấy từ development. Mỗi cấu hình encode thật rồi giải mã trong tiến trình mới: **40/40 pixel hash khớp**. Không sử dụng tập test độc lập của campaign. Đây là kiểm tra development nhỏ, chưa phải đánh giá chất lượng CLIC.

| Cấu hình | bpp thực | PSNR trung bình ↑ | LPIPS-Alex ↓ | DISTS ↓ |
|---|---:|---:|---:|---:|
| MLIC++ gốc q1 | 0,153687 | 28,9411 | 0,229679 | 0,224645 |
| Baseline seed 9041 | 0,155090 | 28,9436 | 0,228525 | 0,221924 |
| Learned steps seed 9041 | 0,155151 | 28,9567 | 0,226809 | 0,224583 |
| Baseline seed 9042 | 0,154053 | 28,9543 | 0,228962 | 0,224027 |
| Learned steps seed 9042 | 0,154236 | 28,9399 | 0,227633 | 0,222211 |

Nguồn: [JSON kết quả](../results/2026-11-07_mlicpp_end_to_end/cpu_dev8_comparison.json).

Learned steps giảm LPIPS nhẹ ở cả hai seeds, nhưng DISTS xấu hơn ở seed 9041 và tốt hơn ở seed 9042; PSNR cũng không nhất quán. Bitrate không hoàn toàn bằng nhau. Không có đủ bằng chứng để chọn nhánh này làm phương pháp chủ lực, tuyên bố ý nghĩa thống kê hay suy ra người xem thích hơn. Chưa tính BD-rate từ một rate point.

## Các phép thử tiếp theo đã đặt lịch

- `cpu-rate8-q1`: factor lượng tử 2/4/8 trên cùng development, đo byte thực và fresh decode. Đang chạy khi ghi báo cáo này.
- `cpu-perceptual8-q2`: q2 gốc so với synthesis perceptual đã train, factor 1/2. Chờ rate probe hoàn thành, kiểm hash stream giữa hai decoder trong mỗi factor.
- Campaign full train đã được resume, giữ năm checkpoint và chờ GPU để chạy bảy lượt còn lại. Điều kiện bộ nhớ vẫn 18 GiB; không dừng bất kỳ job khác.

Các phép thử CPU chạy tuần tự, mỗi tiến trình Torch dùng hai luồng, CUDA bị ẩn. Không cần GPU trống để thu bằng chứng mới. Thời gian crop CPU không đại diện cho runtime full-resolution trên L4 của challenge.

## Quyết định cho mục tiêu thi

Giữ bộ codec kết hợp hiện tại làm đối chứng. Ưu tiên hoàn thiện rate thấp và decoder perceptual, rồi đánh giá cùng byte và chấm mù. Các cải tiến nhỏ MSE/quantization vẫn được kiểm công bằng, nhưng chưa coi là đóng góp mới hoặc bằng chứng khả năng thắng CLIC. Xem [kế hoạch cạnh tranh](../plans/CLIC2027_competitive_path_2026-11-07.md).

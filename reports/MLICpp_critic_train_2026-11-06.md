# MLIC++: kết quả huấn luyện critic dự đoán chi phí RD

**Đã train thật 2.000 bước AdamW trên GPU 2**, với backpropagation và cập nhật 1.064.193 tham số. Toàn run, gồm replay tạo dataset và training, mất 96,71 giây. Critic chưa tạo lựa chọn tốt hơn baseline không sửa symbol trên validation; chưa có cải thiện codec mới.

## Model và dữ liệu

[Protocol](../plans/MLICpp_critic_train_2026-11-06.md), [code](../experiments/mlicpp/critic_train.py), [config](../experiments/mlicpp/critic_train.json).

Train **critic ở encoder**, backbone MLIC++ q2 đóng băng. MLP 3.898→256→256→1 dự đoán delta J của action; input là thống kê symbol/context/residual và vị trí, dấu can thiệp. Chuẩn hóa chỉ bằng train. Batch 64, lr 3e-4, AdamW, FP32, seed 9041; gradient được kiểm tra hữu hạn và clip norm 1. Tổng thay đổi L1 của trọng số sau train là 2.310,8184, xác nhận optimizer thực sự cập nhật model.

Dữ liệu lấy từ rollout phối hợp 6 ảnh development vừa hoàn tất: 966 lần đánh giá, giảm J trung bình 0,3379% so baseline cùng GPU, 12/12 stream cuối giải mã đúng. Đây là search với ảnh nguồn, chưa là hiệu quả của critic.

Replay lại các state được chọn và kiểm hash stream trước khi tạo feature. Mỗi group gồm 5 action và no-change:

| Tập | Ảnh nguồn | Mẫu | Nhóm lựa chọn |
|---|---:|---:|---:|
| Train | 4 | 768 | 128 |
| Validation | 2 | 384 | 64 |

Hai tập không trùng ảnh nguồn, nhưng đều là development đã dùng trước. 768 mẫu tương quan trong bốn ảnh không tương đương 768 ảnh độc lập. Không mở geometry test và không gọi validation này là test độc lập.

## Kết quả

Checkpoint tốt nhất theo validation regret ở bước **1.800**; giữ cả checkpoint cuối bước 2.000 và optimizer state. Các số dưới được kiểm tra lại bằng cách tải model trên CPU trong tiến trình riêng.

| Phương án | Validation MSE dự đoán delta J ↓ | Regret so với action teacher tốt nhất ↓ | Delta J trung bình của action chọn ↓ |
|---|---:|---:|---:|
| Dự đoán 0 / chọn no-change | 3,48184 × 10⁻⁶ | 4,75347 × 10⁻⁵ | 0 |
| Dự đoán mean train | 2,31247 × 10⁻⁶ | Không đánh giá chính sách | Không đánh giá chính sách |
| Critic best, bước 1.800 | 2,24501 × 10⁻⁶ | 4,75347 × 10⁻⁵ | 0 |
| Critic last, bước 2.000 | 2,27960 × 10⁻⁶ | 1,54946 × 10⁻⁴ | +1,07412 × 10⁻⁴ |

Teacher hữu hạn có delta J trung bình −4,75347 × 10⁻⁵ trên 64 group validation. Critic best chỉ đạt delta J bằng 0: **MSE dự đoán tốt hơn không chuyển thành quyết định mã hóa tốt hơn**. Critic cuối còn chọn action làm J tăng trung bình. Không dùng độ chính xác top-1 riêng lẻ để kết luận thắng, vì no-change là đối chứng mạnh và nhiều group có lợi ích rất nhỏ.

Đây là đánh giá offline trên state của teacher, chưa chạy policy critic theo chuỗi để tạo bitstream mới. Không cộng delta J giữa các state này để tuyên bố gain toàn ảnh. Checkpoint best được chọn trên validation nên kết quả không phải ước lượng test không thiên lệch.

## VRAM và thời gian

Trong lúc chạy, một job khác dùng khoảng 14,7 GiB trên GPU 2. Critic dùng allocator cap 16% của 24 GiB, kiểm headroom tối thiểu 1,5 GiB; không dừng hoặc sửa job khác. Số VRAM ghi trong `train.jsonl` là **tổng GPU**, không phải VRAM riêng của critic. MLP và dataset nhỏ nên không cần lấp VRAM để hoàn tất training.

## Bằng chứng và quyết định

[Artifact local](../results/2026-11-06_mlicpp_critic) gồm dataset, config, log loss, source snapshot, checkpoint best/last, optimizer cuối và kết quả reload verification. Script [verify_critic.py](../experiments/mlicpp/verify_critic.py) kiểm split nguồn, số mẫu, tải trọng số strict và tính lại validation trên CPU.

- Best checkpoint SHA256: `a92c784b1094deff1cd3968161d505046f92e1c95d5730d3611f94683dad1984`.
- Last checkpoint SHA256: `546ee3e38ac40ffc5b7a32791dcae567687e9a52613d3c535c6d91a6437b6e90`.
- Archive SHA256: `6a82e47585fa6b1d2559666bc1ea2d509e8dda9b560080090d72c7a0e00d3d80`.
- Server run: `/home/jovyan/clic2027-artifacts/mlicpp/critic-20261106/div2k-mlicpp-cost-critic-4train-2val-2000steps`.

Chưa đưa critic vào codec chính, chưa train end-to-end backbone, chưa đo human preference. Không kéo dài số bước trên bốn ảnh để tạo cảm giác tiến bộ. Nếu tiếp tục hướng này, cần thêm nguồn train độc lập và mục tiêu học ưu tiên chất lượng quyết định/ranking, rồi kiểm policy rollout với baseline không sửa và search cùng runtime. Hiện chưa có bằng chứng hướng này đủ mạnh cho paper.

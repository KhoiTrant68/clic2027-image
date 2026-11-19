# Ablation mục tiêu critic

Giả thuyết, chưa phải kết luận: regression MSE không phù hợp với quyết định có lợi ích nhỏ và nhiều action gây hại. So MSE với expected regret dưới softmax(-predicted cost), cộng 0,1 MSE để giữ thang dự đoán.

Preflight theo launch:

1. Reference: critic_train.json, cùng dataset, MLP hidden 256, 2.000 bước, AdamW lr 3e-4, betas 0,9/0,999, decay 1e-4, eps mặc định PyTorch 1e-8, clip 1, FP32. Đổi sampling sang 16 nhóm × 6 action = 96 mẫu để có đầy đủ lựa chọn trong mỗi nhóm. Cả hai nhánh dùng cùng sampling, khởi tạo và budget; seeds 9041–9043. Không chọn checkpoint tốt nhất trên validation, so checkpoint cuối cùng cùng bước.
2. Run: div2k-critic-mse-vs-decision-3seeds-2000steps-cpu, output mới, không xóa run trước.
3. Server xác nhận dataset.npz tồn tại và Torch import được. Dataset đã được kiểm source split ở run trước; script assert lại. Không thêm ảnh test hay chọn feature mới từ validation.
4. TensorBoard chưa cài; giữ tracker file JSON/JSONL đã sử dụng trong các pilot trước, có status/progress/checkpoint mỗi run và timeout 1 giờ. Đây là pilot ngắn, không chạy nhiều giờ không giám sát.
5. GPU 2 dùng 22.313/24.576 MiB, GPU 0/1 gần đầy. Chạy CPU 2 luồng, không dùng GPU, không thay đổi job khác. ETA cần đo throughput sau launch, dự kiến vài phút đến khoảng 20 phút; giới hạn 1 giờ.

Primary: chosen delta J và regret trên hai nguồn development validation; luôn báo train và harmful-action fraction. Baseline no-change delta J = 0. Kết quả positive phải nhất quán qua seed, rồi mới mở rộng nguồn và đánh giá policy rollout thực. Với chỉ hai ảnh validation đã xem nhiều lần, không gọi kết quả là test độc lập hoặc chứng minh novelty.

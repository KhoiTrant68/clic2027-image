# Critic: đổi mục tiêu học chưa giải quyết được tổng quát hóa

Đã hoàn tất 6 lượt train (2 loss × 3 seed), mỗi lượt 2.000 bước, tổng 83,52 giây CPU. [Protocol/preflight](../plans/MLICpp_critic_objective_2026-11-07.md), [code](../experiments/mlicpp/critic_objective_compare.py), [số liệu](../results/2026-11-07_mlicpp_critic_objective/results.json).

Giữ cùng dataset, kiến trúc, khởi tạo và nhóm minibatch giữa hai loss trong từng seed. MSE so với expected regret dưới softmax của chi phí dự đoán + 0,1 MSE. Mỗi batch 16 nhóm đầy đủ, tương ứng 96 mẫu; vì khác sampling với run trước, dùng nhánh MSE được train lại làm đối chứng chính. Chỉ so checkpoint cuối bước 2.000, không chọn checkpoint đẹp nhất.

| Seed | MSE: delta J validation | Decision loss: delta J validation | Tỉ lệ chọn action gây hại của decision loss |
|---|---:|---:|---:|
| 9041 | 0 | 0 | 0% |
| 9042 | 0 | 0 | 0% |
| 9043 | 0 | +0,000286797 | 18,75% |

Delta J âm mới tốt; baseline no-change là 0. Đây là trung bình quyết định trên state teacher, không phải RD toàn ảnh. Decision loss cải thiện lựa chọn trên train ở một số seed nhưng không cải thiện trên hai nguồn validation. Không kết luận MSE là nguyên nhân duy nhất; dữ liệu ít, feature gộp làm mất cấu trúc và teacher có lợi ích nhỏ vẫn là các giả thuyết cần phân biệt.

Quyết định: không dùng loss mới trong codec và không kéo dài training trên dataset này. Bước hợp lý tiếp theo là thu thêm nguồn train với target có lợi ích đủ rõ, đồng thời kiểm khả năng giữ thông tin của feature: cùng candidate, so feature thống kê hiện tại với patch latent/context tại vị trí can thiệp. Tách ảnh calibration khỏi ảnh đánh giá mới, và kiểm policy rollout thực trước khi báo cải thiện codec. Chưa thực hiện các bước tiếp theo này.

Preflight: reference/config đã đối chiếu; output tên mô tả và mới; dataset/môi trường tồn tại; monitoring JSON/JSONL/checkpoint hoạt động; ETA cập nhật từ throughput thực. TensorBoard chưa cài, dùng tracker file của các pilot trước. GPU gần đầy nên chạy CPU 2 luồng; không sửa job khác. Không có restart hay xóa bằng chứng cũ.

Checkpoint sáu lượt nằm trên server tại `/home/jovyan/clic2027-artifacts/mlicpp/critic-20261107/div2k-critic-mse-vs-decision-3seeds-2000steps-cpu`. Hai nguồn validation đã dùng nhiều lần, không phải test độc lập. Chưa có bằng chứng novelty, generalization rộng hoặc lợi thế CLIC.

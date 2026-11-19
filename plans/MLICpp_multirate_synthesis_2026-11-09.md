# Pilot thích nghi synthesis với nhiều mức lượng tử

## Lý do

Campaign full-backbone/learned steps chưa có cải thiện nhất quán. Synthesis perceptual giảm metric so với MLIC++ gốc, nhưng đối chiếu lịch sử trên CLIC30 vẫn kém bộ mix. Chưa thể chốt MLIC++ làm phương án thi. Pilot này kiểm khả năng cải thiện decoder khi đầu vào có lượng tử mạnh hơn; không gọi đây là đóng góp mới.

## Đối chứng cố định

| Thuộc tính | single | mixed |
|---|---|---|
| Khởi tạo | Cùng synthesis reconstruction ngày 05/11 | Cùng checkpoint |
| Encoder/entropy | MLIC++ q2 legacy đóng băng | Giống single |
| Bước lượng tử train | factor 1 | Chọn đều factor 1/2/4 |
| Train | 160 nguồn, một crop 256 cố định/nguồn | Cùng nguồn và crop |
| Tối ưu | 1.000 bước, batch 2, Adam 1e-5, betas 0,9/0,99, clip 1 | Giống single |
| Objective | 10 MSE + LPIPS-Alex | Giống single |
| Seed và thứ tự ảnh | 9041 | 9041; RNG factor riêng 77 |

So với pilot synthesis cũ, bỏ edge loss ở **cả hai nhánh mới** để đối chứng đơn giản. Vì vậy, không quy mọi thay đổi từ checkpoint cũ cho multi-rate: so mixed với single mới là phép cô lập tác động chính.

Cache latent lấy từ giải mã entropy thật cho 160 crop × 3 factors, không dùng latent liên tục thay thế. Học khoảng 12 triệu tham số synthesis; encoder/entropy không học. Không thêm block inference.

## Đánh giá và giới hạn

So initial/single/mixed tại factor 1/2/4 trên 32 crop development trung tâm. Tổng 9 cấu hình; fresh decode cho mỗi cấu hình và kiểm stream hash giống nhau giữa ba decoder tại mỗi factor. Ảnh development đã được dùng trước; không phải test độc lập và không đo full-image challenge.

Đầu ra checkpoint cuối cố định, không chọn theo metric test. Nếu mixed không cải thiện ổn định so với single ở factor 2/4 hoặc mất nhiều ở factor 1, không tự mở rộng train. Khi có tín hiệu tốt mới khóa recipe và kiểm full-resolution/cùng ngân sách với đối chứng mạnh hơn.

## Kiểm tra trước chạy

GPU vật lý 1, giới hạn allocator 8 GiB, cần 12 GiB trống khi bắt đầu; FP32, deterministic, hai luồng CPU, cuDNN v8 tắt. Đã xác nhận weights q2, synthesis checkpoint và nguồn ảnh trên server. Không dùng trọng số lab09 vì bản cập nhật không kèm `.pt`.

Smoke: 2 nguồn train, 1 development, hai bước mỗi nhánh rồi 9 lượt decode mới. Chỉ mở pilot chính khi status smoke complete. Thư mục mới, không ghi đè checkpoint cũ; không dừng job khác.

Monitoring: status.json, train.jsonl, progress.json, checkpoint mỗi 250 bước, runtime.json và comparison.json. Budget train hai giờ/nhánh; không cưỡng bức kill subprocess. Ước tính pilot khoảng 20–60 phút sau smoke, hiệu chỉnh theo tốc độ cache/train thực tế; không hứa ETA trước khi đo.

```text
/home/jovyan/clic2027-artifacts/mlicpp/multirate-20261109/smoke
/home/jovyan/clic2027-artifacts/mlicpp/multirate-20261109/div2k160-q2-decoder-single-vs-mixed-1000steps
```

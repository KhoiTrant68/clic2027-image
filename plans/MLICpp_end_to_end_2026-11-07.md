# MLIC++: fine-tune toàn bộ mô hình và ablation bước lượng tử

## Câu hỏi và phạm vi

Kiểm tra việc học đồng thời phép biến đổi, mô hình entropy và 320 bước lượng tử theo kênh có cải thiện đường cong rate–distortion so với fine-tune MLIC++ thông thường hay không. Đây là **ablation kỹ thuật**, chưa phải đóng góp khoa học mới. Bước lượng tử toàn cục có thể trùng chức năng với phép biến đổi đã học; kết quả âm cũng có giá trị để loại hướng.

## Thiết kế cố định trước khi xem test

| Thành phần | Cấu hình |
|---|---|
| Baseline | Toàn bộ backbone MLIC++ được fine-tune, lượng tử mặc định |
| Nhánh thử nghiệm | Cùng backbone, thêm 320 log-step học được, bước lượng tử trong [0,5; 2] |
| Khởi tạo | Checkpoint gốc q1, q2, q3; cùng checkpoint cho hai nhánh |
| Seeds | 9041 và 9042 |
| Tổng số lượt train | 12: 2 nhánh × 3 mức rate × 2 seeds |
| Mỗi lượt | 2.000 bước, batch 2, crop ngẫu nhiên 256 × 256, FP32 |
| Optimizer | Adam, learning rate 1e-5; auxiliary entropy Adam 1e-3; clip gradient 1 |
| Loss | Estimated bpp + lambda × MSE trên thang 255; lambda 0,0018 / 0,0035 / 0,0067 |
| Dữ liệu | 160 nguồn train; 32 nguồn development được giữ riêng, chưa dùng để chọn checkpoint trong lịch này |
| Test mới | 32 nguồn ở vị trí 370–401 trong thứ tự DIV2K xáo trộn seed 1984 |
| Checkpoint đánh giá | Bước cuối 2.000, không chọn theo test |

So với pilot critic trước, thay đổi có chủ đích là học toàn bộ 116.716.352 tham số backbone trên 160 nguồn thay vì huấn luyện bộ dự đoán chi phí trên 4 nguồn. Hai nhánh mới giữ nguyên lịch chọn crop, optimizer, loss và số bước; khác biệt chính là bước lượng tử học được. Không dùng kết quả test để sửa lịch chạy còn lại.

## Đánh giá

- Đánh giá checkpoint gốc và cả hai nhánh trên ảnh test đầy đủ: tổng cộng 15 cấu hình × 32 ảnh. Padding replicate tới bội số 64, áp dụng giống nhau.
- Bitrate lấy từ số byte bitstream có header chia số pixel ảnh gốc. Bước lượng tử thuộc checkpoint decoder dùng chung, không phải side information riêng từng ảnh. Checkpoint không nằm trong số byte ảnh; cần tính riêng nếu quy định challenge yêu cầu.
- Giải mã lại bằng tiến trình mới, kiểm SHA-256 bitstream và pixel; lưu hash checkpoint để xác định đúng decoder.
- Báo cáo pooled PSNR, PSNR trung bình theo ảnh, LPIPS-Alex và DISTS. Hai metric perceptual dùng tile 512, trọng số diện tích, bỏ tile có cạnh dưới 64 theo evaluator hiện tại; không xem là metric toàn ảnh chuẩn của challenge.
- Đo encode/decode có CUDA synchronize; số đo bao gồm lần chạy đầu, chưa tách warm-up. Đây là thời gian thực nghiệm, chưa là chứng nhận giới hạn challenge.
- Nội suy tuyến tính theo log(bpp) trong khoảng chung của ba đường cong; tuyệt đối không ngoại suy. Ba điểm rate và hai seeds chỉ đủ thăm dò, chưa đủ kết luận thống kê hay BD-rate chắc chắn.
- Chuẩn bị bộ chấm mù seed 9041 khi bitrate trung bình hai cấu hình lệch không quá 5%. Chọn theo bitrate, không chọn theo chất lượng. Đây là preview gần cùng rate, không bảo đảm từng ảnh cùng rate. Cần người thật chấm; chưa có human evaluation.

## Kiểm tra trước chạy và giới hạn tài nguyên

1. **Config:** đã đối chiếu các thay đổi với pilot trước và cố định cấu hình hai nhánh như trên.
2. **Tên chạy:** `div2k160-fullrd-vs-learnedsteps-3rates-2seeds`; thư mục mới, không ghi đè kết quả trước.
3. **Đường dẫn:** đã xác nhận checkpoint q1–q3, dữ liệu train và môi trường; audit hash train/development đã qua. Dataset test mới được tải và kiểm CRC/SHA trước train.
4. **Theo dõi:** `status.json`, `progress.json`, `train.jsonl`, `pipeline.log`, checkpoint mỗi 250 bước; giữ mã nguồn và config. Không cần dịch vụ tracker ngoài.
5. **Thời gian:** tối đa 4 giờ mỗi lượt, tối đa 24 giờ toàn chiến dịch kể từ khi khởi chạy. Chờ GPU tối đa 6 giờ mỗi giai đoạn trong ngân sách tổng. Chưa có tốc độ GPU đo được để đưa ETA hoàn thành đáng tin; mốc tối đa tính theo giờ Việt Nam/UTC+7 là thời điểm khởi chạy +24 giờ. Hết ngân sách sẽ dừng, không coi là hoàn thành.

GPU A30 phải còn ít nhất 18.432 MiB trước mỗi giai đoạn GPU. Giới hạn allocator PyTorch ở 70% tổng VRAM. Đây không phải reservation của scheduler: tiến trình khác vẫn có thể chiếm thêm bộ nhớ. Nếu OOM hoặc lỗi decode, dừng và giữ bằng chứng, không tự giảm kích thước test hay can thiệp job khác.

## Bằng chứng hiện có và điều kiện tiếp tục

CPU smoke đã qua: step=1 giữ nguyên forward, bitstream và pixel giải mã; gradient log-step hữu hạn, L1 = 0,048388; một bước optimizer đổi log-step L1 = 0,003199. Đây là kiểm tra cơ chế, **không phải kết quả train lớn**.

Chỉ mở rộng training khi cả hai nhánh hoàn thành công bằng, fresh decode qua và có tín hiệu cải thiện nhất quán trong khoảng bitrate chung. Nếu nhánh học bước lượng tử không tốt hơn, giữ baseline và loại ablation này khỏi tuyên bố đóng góp. Muốn có paper mạnh vẫn cần cơ chế mới được đối chiếu prior art, dữ liệu lớn hơn, nhiều seeds và đánh giá ngoài DIV2K; chưa thể hứa hẹn thứ hạng CLIC hay acceptance.

Giới hạn dữ liệu: test mới chỉ chưa dùng trong các pilot của dự án; chưa xác nhận không trùng dữ liệu pretraining checkpoint gốc. Các test cũ đã dùng được cách ly khỏi campaign này.

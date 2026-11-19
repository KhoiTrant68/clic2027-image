# Kế hoạch tăng sức cạnh tranh CLIC 2027

## Quyết định

Không đồng nhất giảm training loss với thắng CLIC. Giữ MLIC++ làm baseline có bitstream kiểm được; ưu tiên ba vấn đề: phủ đủ bitrate, chất lượng người xem và decoder đủ nhanh. Không dừng tiến trình nào khác trên server. Chưa có cơ sở hứa hẹn thứ hạng.

Giữ bộ codec kết hợp hiện tại làm đối chứng chính của bài thi. Báo cáo full30 ngày 02/11 cho thấy MLIC++ MSE không vượt nó về các metric cảm nhận; không thay đối chứng chỉ vì đã đầu tư train MLIC++. Qhat cũng chưa được xác nhận bằng người chấm.

## Quy định đã kiểm ngày 07/11/2026

[Trang tasks chính thức](https://clic2027.compression.cc/tasks/) quy định ba mức 0,075 / 0,15 / 0,3 bpp, chọn người thắng bằng đánh giá cặp ảnh của người chấm, và loại 25% submission giải mã chậm nhất khỏi giải thưởng. Decoder tối đa 4 GB; môi trường image track có 2 CPU, 12 GB RAM, GPU track thêm L4 24 GB; giới hạn 5 giờ. Phải có decoder hợp lệ trong validation, giữ nguyên decoder đã đăng ký khi nộp test và có whitepaper.

[FAQ](https://clic2027.compression.cc/faq/) xác nhận cần ảnh cùng kích thước gốc và model không tính vào bitrate năm này. Dù vậy, mọi thông tin riêng cho ảnh vẫn phải ở dữ liệu nén. Các trang lịch có thể cập nhật; không suy diễn ngày nộp từ thông tin không thống nhất trên trang chủ.

## Việc đã thực thi trong phiên này

1. Khôi phục campaign từ 5 lượt hoàn thành, bỏ qua đúng các thư mục có trạng thái complete và checkpoint. Giữ nguyên config train; không ghi đè checkpoint. Session `mlic-e2e-resume-20261107` chờ GPU tối đa 24 giờ mỗi giai đoạn, ngân sách 48 giờ trước khi bắt đầu giai đoạn mới. Không cưỡng bức dừng subprocess; mỗi lượt train vẫn tự giới hạn 4 giờ theo config cũ.
2. Chạy `cpu_development.py`: q1 gốc và baseline/joint của hai seeds, 8 crop trung tâm 256 từ development, đo byte thực, PSNR/LPIPS/DISTS, fresh decode. Hai luồng CPU, không GPU. Đây là chẩn đoán, không thay thế 32 ảnh test đầy đủ.
3. Đặt `cpu_rate_probe.py` chạy tuần tự sau bước 2: cùng checkpoint q1 gốc, hệ số bước lượng tử 2/4/8. Mỗi factor là một decoder cấu hình cố định và phải qua fresh decode. Đây là thử khả năng phủ rate, không phải thuật toán mới hay bảo đảm đáp ứng bitrate từng ảnh.
4. Chuẩn bị `cpu_perceptual_probe.py` chạy sau rate probe: q2 gốc và decoder synthesis perceptual đã train trước, tại factor 1 và 2. Kiểm hash bitstream giống hệt giữa hai decoder trong mỗi factor. Đây là thí nghiệm generalization qua bước lượng tử trên development; không train thêm, không đọc test mới.

## Ma trận quyết định tiếp theo

| Thí nghiệm | Biến thay đổi | Giữ cố định | Điều kiện để mở rộng |
|---|---|---|---|
| q1 original / baseline / joint | Fine-tune và bước lượng tử | Development crop, metric, runtime, checkpoint cuối | Joint phải có lợi sau khi xét bitrate; không dùng training loss để chọn |
| q1 factor 1/2/4/8 | Bước lượng tử vô hướng | Model, nguồn ảnh, codec, metric | Có điểm rate thấp giải mã đúng; kiểm trade-off trước khi fine-tune perceptual ở rate thấp |
| Decoder perceptual | Objective/decoder synthesis | Encoder, entropy và bitstream | So sánh cùng byte; blind preference và metric độc lập; kiểm chữ, biên và texture |
| Decoder nhanh | Kiến trúc synthesis hoặc context được rút gọn | Dữ liệu và ngân sách byte | Không mất lợi thế preference; đo trên cấu hình gần L4, không suy từ crop CPU |

Hai dòng đầu đang chạy/đặt lịch. Decoder perceptual dùng checkpoint reconstruction của pilot cũ để chạy lại so sánh cùng byte; chưa train thêm hoặc rút gọn decoder trong phiên này. Pilot synthesis trước đã có kết quả development thuận lợi, không coi đó là bằng chứng human preference.

## Cách sử dụng tài nguyên

Đánh giá crop CPU tận dụng RAM trống, hai luồng; dự kiến dưới một giờ cho toàn bộ hai phép thử, cần hiệu chỉnh bằng thời gian thực. Không thuê GPU hay gọi API trả phí. Còn 7 lượt full train, tốc độ quan sát 28–37 phút/lượt: khoảng 3,3–4,3 giờ GPU nếu tài nguyên liên tục sẵn có, chưa gồm full-image evaluation và chờ.

Không hạ batch/crop của chỉ một nhánh để chen vào vài GB VRAM: sẽ phá đối chứng đã khóa. Nếu cần một protocol tiết kiệm bộ nhớ mới, chạy lại cả cặp với gradient accumulation và xác minh gradient; không trộn số liệu vào campaign này.

## Hướng model và prior art

[PICO](https://arxiv.org/html/2605.05148v1) kết hợp tối ưu cảm nhận, runtime, điều khiển rate và xử lý lỗi chữ/tiling; không thể nhận các thành phần chung đó là mới. [PULSE](https://arxiv.org/abs/2609.18602) là đối chứng mới về codec ít tính toán và entropy coding bit-exact; số liệu tác giả chưa phải kết quả tái lập tại đây.

Hướng có thể đầu tư cho thi là synthesis cảm nhận chạy một lượt, huấn luyện ở đúng vùng rate thấp, với đối chứng cùng bitstream. Hướng paper về thông tin hiệu chỉnh prior chỉ tiếp tục khi vượt decoder-only mạnh và thêm bit cho base ở cùng tổng byte. Learned scalar/channel gains hiện là ablation kỹ thuật, không đóng vai trò đóng góp mới nổi bật.

## Phân tích và điều kiện loại hướng

- Không đánh giá ngang giữa rate khác nhau như thể cùng rate. Đường cong cuối nội suy trong vùng giao, không ngoại suy.
- Trên development, gain nhỏ/không ổn định giữa seeds không đủ để mở chiến dịch lớn hơn. Crop chưa phản ánh chi phí hay texture toàn ảnh.
- Blind scoring phải có người thật, ngẫu nhiên A/B, ghi tie/skip; không thay người chấm bằng Qhat. Cần thống kê ghép cặp theo ảnh và người chấm khi đã có đủ dữ liệu.
- Test độc lập chỉ dùng sau khi khóa recipe. Không chọn recipe trên 32 ảnh test mới hoặc trên ảnh test CLIC tương lai.
- Đánh giá package và decoder khác thiết bị phải thực hiện riêng: fresh CPU decode cùng môi trường chưa chứng minh entropy coding ổn định giữa CPU và GPU.

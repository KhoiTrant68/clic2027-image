# AEIC: kiểm tra khả năng fine-tune

Kiểm tra mã nguồn tác giả ở commit `7222c2c85e8db7ea0b829a139313ab7f4fdea88c`, ngày 20/10/2026. Đây là kế hoạch có điều kiện; **chưa khởi động huấn luyện**.

## Những gì đã xác nhận

- Có 800 ảnh tại `~/clic/data/DIV2K_train_HR` (khoảng 3,3 GiB). Tập 30 ảnh đánh giá hiện tại không được dùng để huấn luyện.
- `src/finetune.py` dùng `AEIC_ME_Trainer`, AdamW, huấn luyện codec + LoRA + convolution đầu/cuối UNet. VAE và phần trọng số UNet còn lại được đóng băng.
- `config/finetune_AEIC_ME.yaml`: patch 512, batch/GPU 1, batch hiệu dụng 32, learning rate ban đầu 5e-5, lambda rate 2, lambda L2/DISTS/CLIP/adversarial lần lượt 2/1/0,1/0,1; 30.000 bước.
- `H5Dataset` đọc từng ảnh dạng H×W×C từ HDF5. Dữ liệu DIV2K hiện có ở dạng thư mục ảnh, chưa được xác nhận/chuyển thành HDF5 cho phép thử này.
- Trainer ME có hàm nạp `state_dict_codec` và các khóa LoRA/conv của checkpoint. Khả năng khởi tạo toàn bộ trainer từ ME-ft2 chưa được smoke-test; không coi đọc mã là bằng chứng đã nạp thành công.
- Script tạo DINOv3 discriminator, CLIP loss, OCEA-DISTS và LPIPS. Kiểm tra import trong môi trường `py310` cho thấy chưa có `h5py` và `vision_aided_loss`; các weights dành riêng cho training cũng chưa được xác nhận đầy đủ. Suy luận ít VRAM không chứng minh training sẽ vừa bộ nhớ.

## Các điểm cần xử lý trước khi chạy

1. File shell tác giả đặt 2 process, nhưng YAML ghi batch hiệu dụng 32 với batch/GPU 1 và accumulation 8. Script có assertion `total = processes × per_gpu × accumulation`, nên không thể dùng nguyên cặp shell/YAML. Pilot một GPU cần accumulation 32 để giữ batch 32.
2. Hàm forward của trainer ME gọi trực tiếp UNet/VAE. Flag checkpointing hiện chỉ bật cơ chế của UNet; chưa có bằng chứng VAE backward được checkpoint đầy đủ. Đo backward trên patch 512 trước khi khẳng định vừa giới hạn.
3. Tên experiment hiện ép lambda thành số nguyên (`int(float(lambda_rate))`), nên lambda 0,5 và 0,25 sẽ cùng hậu tố 0. Khi thử lambda thấp phải đặt `exp_name` riêng và thư mục mới.
4. Vòng lặp gốc không break giữa epoch khi đạt `max_train_steps`. Một pilot có giới hạn bước cần sửa điều kiện dừng và lưu checkpoint cuối, tránh chạy vượt kế hoạch.
5. Đánh giá gốc đòi đúng 24 ảnh Kodak. Cần chuẩn bị split riêng hoặc thay evaluator một cách rõ ràng; không thay bằng 30 ảnh rồi bỏ assertion và coi là tái lập nguyên bản.

## Pilot tối thiểu nếu qua các ngưỡng đánh giá

Giữ nguyên kiến trúc, bắt đầu từ ME-ft2. Ưu tiên phép thử lambda=1 trước (so với lambda=2 của ft2), không đồng thời thay loss và kiến trúc. Đây là giả thuyết tăng bitrate, chưa bảo đảm đạt 0,075 bpp.

1. Cố định seed và manifest train/held-out riêng từ DIV2K; kiểm tra trùng ảnh với tập đánh giá. Chỉ dùng held-out riêng để chỉnh lambda.
2. Smoke 2 optimizer steps trên 1 GPU, patch 512, batch 1, accumulation 32, checkpointing. Dùng guard bộ nhớ và log NVML; giữ yêu cầu ít VRAM. Nếu không vừa mức hiện tại, dừng và báo số đo trước khi đổi recipe.
3. Nếu smoke đạt: pilot 200 optimizer steps, checkpoint 100 và 200, giới hạn thời gian; đo loss/rate, finite gradients, thời gian và peak VRAM.
4. Đánh giá bitrate từ bitstream thật và giải mã độc lập trên held-out; estimated entropy chỉ là tín hiệu huấn luyện. Chọn lambda từ held-out trước khi đánh giá lại tập 30 ảnh.
5. Chỉ tăng thời lượng sau khi chất lượng cảm nhận, chi phí decoder và lợi ích khi phân bổ cùng ngân sách đều có bằng chứng tốt.

Nếu ngưỡng metric full30 không đạt, quyết định mặc định là **không fine-tune AEIC ở vòng này**. Việc không chạy một thí nghiệm không qua tiêu chí đã định là kết quả của quy trình lựa chọn, không phải một run hoàn thành.

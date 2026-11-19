# Huấn luyện critic từ rollout MLIC++

User yêu cầu train thực sự sau pilot tìm kiếm. Chạy supervised training critic ở encoder; không tuyên bố đang fine-tune backbone MLIC++. Đây là pilot học từ teacher yếu, không đảo kết luận novelty/gate của các run trước.

Nguồn dữ liệu: run joint-symbol 6 nguồn, 4 sweeps, 8 nhóm/sweep, 5 action/nhóm. Replay chỉ các phương án đã được teacher chấp nhận và kiểm SHA stream từng state. Tạo 6 mẫu/group gồm 5 action và no-change. Tổng dự kiến 1.152 mẫu: 768 train từ worker 0–3, 384 validation từ worker 4–5. Khóa split theo nguồn trước train; toàn bộ 6 ảnh từng dùng làm development, không gọi hai nguồn validation là test độc lập.

Input critic: thống kê theo kênh của symbol, CDF index và rounding residual ở 20 bước, cộng biểu diễn vị trí/delta/local residual của action. Không dùng chi phí cuối candidate làm input. Target: chênh lệch J thực của candidate so với state hiện tại. Feature state có source residual chỉ ở encoder, không truyền sang decoder.

MLP 3.898→256→256→1, GELU; 2.000 bước AdamW, lr 3e-4, betas (0,9;0,999), eps 1e-8, weight decay 1e-4, batch 64, seed 9041, clip gradient norm 1, FP32, một GPU, không accumulation/dataloader worker. So với geometry distillation MLP trước: đổi target log-geometry thành delta J, đổi kích thước input từ 1.280 lên 3.898 và hidden lên 256; AdamW có regularization vì ít ảnh. Chuẩn hóa input/target chỉ bằng train, clipping feature chuẩn hóa ±20.

Đo train/validation MSE, baseline dự đoán 0 và mean-train, top-1 trong nhóm và regret theo J thật trên state teacher. Chọn checkpoint bằng validation regret, giữ checkpoint cuối + optimizer và log mỗi 100 bước. Ranking offline không tương đương rollout policy trên state mới; chưa là kết quả codec cải thiện.

Tài nguyên: GPU 2 có job khác vừa dùng khoảng 15.054 MiB; không dừng job đó. Chờ joint-search hoàn tất rồi cấp allocator cap 16% GPU (~3,84 GiB), kiểm còn ≥1.536 MiB định kỳ. Không cố tăng batch chỉ để lấp VRAM; dataset rất nhỏ. Timeout logic 1 giờ, không restart/ghi đè output cũ.

Monitoring: `status.json`, `prepare.progress.json`, `train.progress.json`, `train.jsonl`, checkpoint best/last, training summary. ETA dự kiến vài phút sau khi replay nguồn xong; sẽ cập nhật bằng runtime thật. Checkpoint/manifest/reference đã có trên server, script yêu cầu reference status complete trước prepare.

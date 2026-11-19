# Decoder MLIC++ với hiệu chỉnh latent theo vùng

Giả thuyết là decoder hưởng lợi khi điều chỉnh latent theo mức biến thiên tại từng vị trí. Module đo độ lệch tuyệt đối trung bình giữa latent giải mã và trung bình lân cận 3×3. Tín hiệu này điều khiển một cổng sigmoid trên nhánh residual 16 kênh. Nhánh residual có 10.768 tham số với latent 320 kênh. Lớp cuối khởi tạo bằng 0, vì vậy decoder cho đúng đầu ra ban đầu trước khi cập nhật trọng số. Module lấy đầu vào từ latent đã giải mã nên cùng một bitstream có thể chạy qua cả ba decoder.

Phép thử tập trung vào việc dùng thống kê latent theo từng vị trí trong decoder và xác minh bằng bitstream cố định. Nén ảnh đã có các công trình khai thác chi tiết cục bộ trong transform, như [STF](https://arxiv.org/abs/2203.08450) và [CMamba](https://arxiv.org/abs/2502.04988).

| Nhánh | Kiến trúc | Câu hỏi |
| --- | --- | --- |
| Plain | Decoder MLIC++ gốc được fine-tune lại | Mức cải thiện từ cùng lịch huấn luyện là bao nhiêu? |
| Global | Nhánh residual mới, dùng một thống kê trung bình toàn ảnh | Thêm tham số và một cổng dùng chung có giúp không? |
| Local | Cùng nhánh residual và số tham số như Global, dùng thống kê riêng từng vị trí | Điều chỉnh theo vị trí có giúp hơn cổng toàn ảnh không? |

Ba nhánh dùng cùng checkpoint khởi tạo, dữ liệu DIV2K, chuỗi crop, factor 1/2/4, loss `10 MSE + LPIPS-Alex`, Adam và 1.000 bước. Phép so Local với Global cô lập ảnh hưởng của vị trí. Phép so Local với Plain đo tác động tổng thể của việc thêm nhánh. Encoder và mô hình entropy đóng băng.

Smoke CPU dùng hai crop train, hai bước mỗi nhánh và một ảnh development. Lượt GPU có ba nhánh train và bốn decoder đánh giá (gồm checkpoint khởi tạo) tại ba factor trên 32 crop. Lượt Mixed trước đây mất khoảng 48 giây cho 1.000 bước train trên L40, chưa tính cache và đánh giá. Dự trù 10–30 phút cho cả lượt GPU, tùy tốc độ đọc dữ liệu và tải máy. Ba checkpoint khoảng 140 MiB, chưa tính bitstream và ảnh đánh giá. Chi phí API bằng 0.

Chạy trên server từ thư mục dự án với môi trường MLIC++ hiện có.

```bash
base=/home/jovyan/clic2027-artifacts/mlicpp
cfg="$base/multirate-20261109/div2k160-q2-decoder-single-vs-mixed-1000steps/config.json"
py=/home/jovyan/clic2027-envs/mlicpp/bin/python
nvrtc=/home/jovyan/clic2027-envs/mlicpp/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export LD_LIBRARY_PATH="$nvrtc${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$py" experiments/mlicpp/multirate_synthesis.py --config "$cfg" \
  --out "$base/spatial-synthesis-20261111/cpu-smoke" --architecture --smoke --device cpu

CUDA_VISIBLE_DEVICES=1 TORCH_CUDNN_V8_API_DISABLED=1 "$py" \
  experiments/mlicpp/multirate_synthesis.py --config "$cfg" \
  --out "$base/spatial-synthesis-20261111/gpu-smoke" --architecture --smoke

CUDA_VISIBLE_DEVICES=1 TORCH_CUDNN_V8_API_DISABLED=1 "$py" \
  experiments/mlicpp/multirate_synthesis.py --config "$cfg" \
  --out "$base/spatial-synthesis-20261111/div2k160-q2-local-vs-global" --architecture

CUDA_VISIBLE_DEVICES=1 TORCH_CUDNN_V8_API_DISABLED=1 "$py" \
  experiments/mlicpp/extend_clic_decoders.py --gpu 1 --arms plain global local \
  --source "$base/perceptual-full-20261109/clic30-budget-study" \
  --checkpoints "$base/spatial-synthesis-20261111/div2k160-q2-local-vs-global" \
  --out "$base/spatial-synthesis-20261111/clic30-frozen"
```

Đọc `status.json` để theo dõi, `comparison.json` để so sánh và `metrics.json` để lấy số liệu từng ảnh. Kiểm hash bitstream giữa các nhánh, giải mã lại trong tiến trình riêng và kiểm hash pixel. Chỉ số chính là LPIPS-Alex tại từng factor. DISTS, PSNR và thời gian giải mã là chỉ số phụ. Tính chênh lệch Local–Global và Local–Plain theo cặp ảnh với bootstrap 10.000 lần, seed 2037. Đọc kết quả crop để quyết định có tiếp tục với CLIC30 ở ba ngân sách hay không. Khi đánh giá CLIC30, dùng đúng các byte đã lưu và so sánh thêm với Mixed và Perceptual theo từng ngân sách. Kết quả smoke xác minh tích hợp; phép thử 32 crop dùng để thăm dò hiệu quả.

Smoke CPU và smoke CUDA trên A30 đều hoàn tất với hai bước mỗi nhánh và 12 lượt giải mã khớp hash pixel mỗi lượt. Hai checkpoint Global và Local có đủ 10.768 tham số mới và trọng số lớp chiếu residual đã thay đổi khỏi 0. Lượt CPU mở rộng với 8 ảnh train, 3 crop validation và 12 bước mỗi nhánh có 36 lượt giải mã khớp. Chênh lệch tuyệt đối LPIPS-Alex giữa Local và Plain dưới 0,00003 ở từng factor trong lượt CPU mở rộng. Lượt train 1.000 bước và đánh giá CLIC30 của kiến trúc này chưa chạy.

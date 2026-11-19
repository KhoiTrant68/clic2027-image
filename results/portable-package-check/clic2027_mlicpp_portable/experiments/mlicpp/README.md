# Thử nghiệm MLIC++ trên CLIC validation

Đánh giá checkpoint MSE có sẵn của tác giả trên cùng 30 ảnh đang dùng trong dự án. Không huấn luyện lại và không coi đây là tái lập toàn bộ kết quả paper.

## Nguồn và môi trường

- [Mã nguồn chính thức](https://github.com/JiangWeibeta/MLIC), pin commit `4aa9d2a512eb9db382172058e6c278fb0d376e17`.
- Bốn checkpoint MSE lambda `0.0018 / 0.0035 / 0.0067 / 0.0130` từ mục **Old Weights (2023-09)** của README chính thức. Các URL tải được lưu trong `download_weights.sh`; SHA-256 thực được ghi vào metadata từng lượt chạy.
- Theo hướng dẫn tác giả, các checkpoint này phải dùng `SynthesisTransformOld` và `LatentResidualPredictionOld`. Runner chọn đúng hai lớp trước khi dựng mô hình, không sửa tensor hoặc dùng nạp trọng số không đầy đủ.
- Python 3.10, PyTorch `2.0.1+cu118`, torchvision `0.15.2+cu118`, CompressAI **`1.2.0b3`**, NumPy `1.26.4`, CUDA NVRTC `11.8.89`. Dependency đầy đủ được ghi trong `requirements.freeze.txt` của venv. Đường dẫn thư viện NVRTC phải có trong `LD_LIBRARY_PATH` khi chạy.
- Checkpoint chính thức là archive huấn luyện có đối tượng `collections.Counter`. Runner dùng `torch.load(..., weights_only=False)` cho các tệp tác giả được chỉ định, không dùng cơ chế này để đọc stream hoặc checkpoint không rõ nguồn.

CompressAI 1.2.4 khác API mà mã MLIC++ đang dùng. Phải dùng đúng 1.2.0b3; không bỏ qua lỗi khởi tạo hay chuyển sang `strict=False`. Trên server mới, build CompressAI cần compiler C++; compiler được cài ở prefix riêng `clic2027-envs/mlicpp-build`, không thay đổi môi trường dự án khác.

## Dữ liệu

`prepare_data.py` đọc ZIP validation chính thức, chỉ lấy ảnh PNG hợp lệ và đối chiếu tên với allocation baseline đã lưu. Chỉ chấp nhận đúng **30 ảnh / 86.245.376 pixel**, đồng thời lưu hash ảnh nguồn. Không đổi kích thước ảnh, không thay tập dữ liệu.

```bash
python experiments/mlicpp/prepare_data.py \
  --zip /home/jovyan/clic2027-artifacts/data/clic2025_image_test.zip \
  --allocation results/2026-10-15_pulse_integration/integrated_bakeoff/allocation.csv \
  --out /home/jovyan/clic2027-artifacts/data/valid30
```

## Chạy thử trước khi chạy toàn tập

Các đường dẫn dưới đây dành cho server `nexai-sv20-dmh`. Thư mục output phải mới; kết quả các lần lỗi được giữ để đối chiếu.

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mlicpp/run_server.sh \
  --upstream /home/jovyan/clic2027-image/refs/repos/MLIC \
  --weights /home/jovyan/clic2027-artifacts/mlicpp/weights \
  --data /home/jovyan/clic2027-artifacts/data/valid30 \
  --out /home/jovyan/clic2027-artifacts/mlicpp/smoke-new \
  --legacy --qualities 1 --limit 1 --skip-metrics
```

Smoke đạt khi checkpoint nạp đủ trọng số, nén thành stream thật, giải mã đúng kích thước và hash pixel khớp với ảnh giải mã ở một tiến trình mới. Không đưa vào chạy toàn tập nếu chưa đạt các điều kiện này.

## Chạy toàn tập và tính metric

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mlicpp/run_server.sh \
  --upstream /home/jovyan/clic2027-image/refs/repos/MLIC \
  --weights /home/jovyan/clic2027-artifacts/mlicpp/weights \
  --data /home/jovyan/clic2027-artifacts/data/valid30 \
  --out /home/jovyan/clic2027-artifacts/mlicpp/full30-new \
  --legacy --qualities 1 2 3 4

/home/jovyan/clic2027-envs/mlicpp/bin/python experiments/mlicpp/compare.py \
  --out /home/jovyan/clic2027-artifacts/mlicpp/full30-new
```

- Mỗi checkpoint được encode và decode ở hai tiến trình riêng. Decoder chỉ đọc tệp stream và trọng số, không nhận ảnh gốc.
- Header thực **31 byte/ảnh** truyền magic, version, quality, kiến trúc legacy, kích thước gốc, latent shape và chiều dài hai entropy payload. Tất cả byte đều được tính vào rate.
- Padding theo tác giả: thêm zero ở cạnh phải/dưới đến bội số 64. Ảnh ra được cắt về kích thước gốc, clamp về [0,1], nhân 255 và làm tròn về uint8 RGB. Không dùng tile cho mô hình nén.
- Inference fp32, tắt TF32, bật deterministic, seed 0, hai luồng CPU. Dùng guard VRAM hiện có của repo; cấu hình bảo thủ thấp hơn giới hạn 30 GB/GPU đã cho phép.
- Wrapper cố định `TORCH_CUDNN_V8_API_DISABLED=1` và đường dẫn NVRTC 11.8 cho cả encode lẫn decode. Không dùng stream tạo từ runtime khác để đánh giá gói này: các đối chứng ban đầu đã ghi nhận lệch CDF khi dùng cấu hình cuDNN v8 mặc định.
- Các metric dùng module chung của repo: PSNR RGB, MS-SSIM toàn ảnh, LPIPS/DISTS theo tile 512×512, Q̂ tính từng ảnh rồi lấy trung bình. Thiếu một feature của Q̂ thì báo lỗi, không đổi sang mục tiêu khác.
- Metric weights có thể được tải về cache khi tính lần đầu. Việc giải mã thử này chưa được cô lập mạng bằng container.

Nếu chỉ bước metric lỗi do tải trọng số, có thể tiếp tục bằng `--stage score`, giữ nguyên `--out`, `--qualities`, `--data`, `--upstream`, `--weights`, `--legacy`. Runner kiểm lại độ phủ metric và hash giữa các tiến trình rồi tự cập nhật trạng thái hoàn tất; không chạy lại encode vào thư mục cũ.

## Đọc kết quả

| Tệp | Nội dung |
| --- | --- |
| `manifest.json` | Danh sách ảnh, cấu hình, hash Q̂ và mã runner |
| `status.json`, `failure.json` | Trạng thái hoặc lỗi; giữ nguyên log của lần thất bại |
| `qN-encode.json`, `qN-decode.json` | Hash checkpoint/stream/pixel, thời gian và bộ nhớ từng ảnh |
| `process_times.json` | Thời gian từ tiến trình mới, gồm nạp mô hình và ghi PNG |
| `metrics.csv` | Sáu feature Q̂, MSE, Q̂ từng ảnh, byte thực |
| `comparison.csv`, `summary.md` | MLIC++ từng checkpoint, allocation ở ba ngân sách, và mix lịch sử khi thêm MLIC++ |
| `expanded_allocation.csv` | Ảnh nào được chuyển sang MLIC++ khi bổ sung các điểm mới |

`compare.py` tái sử dụng bộ phân bổ byte của repo. Nếu các checkpoint đã thử vẫn vượt một ngân sách thì ghi không khả thi, không nội suy hoặc bỏ ảnh. Mix được tối ưu lại trên toàn bộ point pool baseline cùng các điểm MLIC++; đối chứng `reoptimized_reference_pool` dùng cùng bộ phân bổ nhưng chưa thêm MLIC++. Khi phép làm tròn chi phí của DP làm giảm điểm mục tiêu, giữ phương án tham chiếu khả thi thay vì báo một mức giảm giả do thuật toán.

Baseline mix là kết quả lịch sử đã lưu; byte metadata và đầu ra deterministic của các nhánh cũ chưa được kiểm lại toàn bộ. Q̂ chỉ là mô hình sàng lọc, chưa có chấm mù cho MLIC++ trong thử nghiệm này. Thời gian trên A30 hoặc L40 không thay thế phép đo L4. Ảnh, checkpoint và stream lớn nằm trong `clic2027-artifacts/`, chỉ sao chép CSV/JSON/log/báo cáo nhẹ vào repo.

## Kiểm thử framing trên CPU

Pilot huấn luyện adapter decoder được mô tả riêng trong [PILOT.md](PILOT.md): ba nhánh cùng kiến trúc và stream, split DIV2K theo ảnh, metric test và kiểm decoder ở tiến trình mới. Pilot này không thay thế phép thử inference 30 ảnh ở trên.

```bash
python -m unittest discover -s tests -p test_mlicpp_stream.py -v
```

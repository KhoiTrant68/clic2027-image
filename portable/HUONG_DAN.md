# Chuyển thí nghiệm CLIC / MLIC++ sang máy khác

Gói code ngày 08/11/2026. Đây là **gói nghiên cứu train / infer / eval**, không phải ZIP submission CLIC và không phải một model đã được chứng minh thắng challenge. Bắt đầu từ tài liệu này; các README/config cũ trong `experiments/mlicpp` là lịch sử và có thể chứa đường dẫn server cũ.

## 1. Gói này có gì?

- Code thí nghiệm MLIC++, full-backbone fine-tune, learned steps, đánh giá bitstream thật và fresh decode.
- Các phép thử CPU development, bitrate thấp và synthesis perceptual.
- Công cụ chấm mù, metric, test stream và báo cáo/protocol đã viết.
- `portable/setup.sh`: tạo môi trường riêng, cài dependency, clone đúng revision MLIC.
- `portable/configure.py`: sinh config theo đường dẫn máy mới, chọn GPU vật lý.
- `portable/run.py`: launcher thay thế các shell wrapper gắn đường dẫn server cũ.
- `MANIFEST.sha256.json` và `portable/verify_bundle.py`: kiểm tra file trong gói.

**Không có** ảnh dataset, checkpoint, thư viện Python, repo MLIC bên thứ ba, tài khoản/token SSH, log/code của các job Qwen/LLM khác. Code không đủ để tái tạo trọng số fine-tune cũ: cần chuyển checkpoint theo mục 8 hoặc train lại. Các thành phần bên thứ ba được tải riêng và giữ điều kiện sử dụng của tác giả.

## 2. Máy phù hợp

Khuyến nghị Ubuntu Linux x86_64, Python 3.10, GPU NVIDIA 24 GB trở lên và driver hỗ trợ CUDA 11.8. Windows nên dùng WSL2 với NVIDIA GPU passthrough hoặc chuyển sang Linux; script Bash không chạy trực tiếp trong PowerShell.

Full train yêu cầu ít nhất **18.432 MiB VRAM trống** trước mỗi lượt. Giới hạn allocator hiện là 70% tổng VRAM. RAM máy nên từ 32 GB; decoder submission CLIC có giới hạn riêng và chưa được chứng nhận bằng yêu cầu máy train này. Dành khoảng 40 GB đĩa cho môi trường, dữ liệu, weights và checkpoints; có thể cần nhiều hơn khi giữ toàn bộ ảnh recon.

CPU smoke/đánh giá crop chạy được khi không có GPU, nhưng không thay thế kiểm định full-resolution hoặc runtime trên L4. Không cần root cho venv; máy cần sẵn `git`, `curl`, compiler C++ (`g++`), Python headers và `python3.10-venv`. Nếu thiếu, nhờ quản trị viên cài đúng gói hệ thống; đừng cài vào môi trường Python dùng chung.

## 3. Giải nén và kiểm file

```bash
unzip clic2027_mlicpp_portable_2026-11-08.zip
cd clic2027_mlicpp_portable
python3 portable/verify_bundle.py
bash portable/setup.sh
source .venv/bin/activate
nvidia-smi
```

Setup cần Internet. Môi trường theo bản đã dùng trên server: torch 2.0.1+cu118, torchvision 0.15.2+cu118, numpy 1.26.4, CompressAI **1.2.0b3**, timm 0.6.13, LPIPS 0.1.4, PIQ 0.8.0. Không tự nâng CompressAI lên 1.2.4 hoặc chuyển load state sang `strict=False` để bỏ qua lỗi. Một số dependency gián tiếp không khóa đầy đủ; bản cài mới được ghi vào `portable/requirements.installed.txt`.

MLIC phải ở revision `4aa9d2a512eb9db382172058e6c278fb0d376e17`. Setup checkout revision này trong `refs/repos/MLIC`; chạy trên thư mục bundle mới để không ảnh hưởng checkout đang làm việc khác.

## 4. Cấu hình máy mới, tải weights và dữ liệu

Các lệnh dưới giả sử đang đứng tại thư mục gốc của bundle và venv đã active:

```bash
export ARTIFACTS="$HOME/clic2027-artifacts"
mkdir -p "$ARTIFACTS"
python portable/configure.py --artifacts "$ARTIFACTS" --gpu 0
export CFG="$PWD/portable/local.json"
bash experiments/mlicpp/download_weights.sh "$ARTIFACTS/mlicpp/weights"
python experiments/mlicpp/fetch_pilot_data.py --out "$ARTIFACTS/data/div2k/pilot224"
python experiments/mlicpp/fetch_pilot_data.py \
  --out "$ARTIFACTS/data/div2k/end-to-end-test32" \
  --holdout-only --holdout-start 370 --holdout-count 32
```

`--gpu 0` là chỉ số GPU vật lý hiển thị bởi `nvidia-smi` trong môi trường máy mới. Launcher sẽ ánh xạ GPU đó thành `cuda:0` trong Python. Trên container/MIG có namespace GPU riêng cần đối chiếu `nvidia-smi` trong container, không dùng chỉ số máy host theo phỏng đoán.

Config được tạo với đường dẫn tuyệt đối và không ghi đè file đã tồn tại. Muốn cấu hình khác, dùng `--out portable/another.json` rồi đổi `CFG`. Không dùng thẳng `end_to_end.json` hoặc `run_end_to_end_campaign.sh` cũ trên máy mới.

Weights tải từ liên kết Old Weights 2023-09 trong repo tác giả, khoảng 1,4 GB/file, script tải q1–q4. Trường hợp Google Drive trả trang HTML hoặc quota: tải từ nguồn tác giả bằng cách khác và giữ đúng tên file; không coi HTTP 200 là checkpoint hợp lệ. Chỉ load checkpoint đáng tin vì format này cần `torch.load(weights_only=False)`.

Fetcher tạo split cố định seed 1984: 160 ảnh train, 32 development, 32 test cũ cách ly. Test mới lấy vị trí 370–401, tổng 32 ảnh. Fetcher kiểm CRC với ZIP chính thức và SHA-256 từng ảnh. Nếu dịch vụ tải ảnh không khả dụng, chuyển nguyên dataset cùng manifest từ máy cũ; không tự thay ảnh mà giữ nguyên tên thí nghiệm. Trùng lặp với pretraining của weights gốc chưa xác nhận.

Nếu đã chuyển dữ liệu/weights ở mục 8, bỏ các lệnh tải tương ứng. Fetcher không ghi đè manifest có sẵn.

## 5. Kiểm tra trước khi chạy dài

```bash
python -m unittest discover -s tests -p test_mlicpp_stream.py
python portable/run.py --config "$CFG" end_to_end -- \
  --stage audit --out "$ARTIFACTS/mlicpp/migration-audit"
python portable/run.py --config "$CFG" --cpu end_to_end -- \
  --stage smoke --out "$ARTIFACTS/mlicpp/migration-cpu-smoke"
```

Audit kiểm hash train/development. Smoke kiểm forward/stream/pixel tại bước lượng tử 1, gradient và một bước cập nhật. Smoke không tạo checkpoint train hoàn chỉnh. Đầu ra dùng thư mục mới; nếu cần chạy lại hãy chọn tên khác để giữ bằng chứng cũ.

Chạy GPU smoke trước full train:

```bash
python portable/run.py --config "$CFG" end_to_end -- \
  --stage smoke --out "$ARTIFACTS/mlicpp/migration-gpu-smoke"
```

Nếu CPU qua nhưng GPU lỗi, xử lý lỗi môi trường/GPU trước khi train. Không bỏ kiểm tra decode.

## 6. Train toàn bộ và đánh giá tự động

```bash
export RUN="$ARTIFACTS/mlicpp/migrated-fullrd-3rates-2seeds"
python portable/run.py --config "$CFG" end_to_end_campaign -- \
  --out "$RUN" --gpu-wait-hours 24 --campaign-hours 48
```

Có thể chạy trong `tmux new -s clic-train`, rồi thực hiện lệnh trên trong tmux; detach bằng Ctrl+B rồi D. Không tự chạy nhiều bản cùng `RUN`.

Campaign gồm 12 lượt: baseline/joint × q1/q2/q3 × seed 9041/9042, mỗi lượt 2.000 bước, batch 2, crop 256. Kế tiếp là 15 cấu hình full-image evaluation và fresh decode, tổng hợp metric, nội suy trong vùng rate giao nhau và tạo bộ chấm mù khi bitrate đủ gần. Trên A30 cũ mỗi lượt train khoảng 28–37 phút; toàn bộ khoảng 5,6–7,4 giờ GPU, chưa gồm evaluation/chờ. Không coi đây là ETA chắc chắn cho máy mới.

Theo dõi:

```bash
cat "$RUN/status.json"
tail -n 30 "$RUN/pipeline.log"
find "$RUN" -maxdepth 2 -name progress.json -print
```

`waiting_for_gpu` không phải đang train. `complete` của từng thư mục con không phải toàn campaign complete. `failed/stopped` cần xem log. Ngân sách 48 giờ được kiểm trước giai đoạn mới, không cưỡng bức kill giai đoạn đang chạy; train có giới hạn riêng 4 giờ/lượt. Không kill tiến trình khác.

Resume trên **cùng config và cùng đường dẫn**:

```bash
python portable/run.py --config "$CFG" end_to_end_campaign -- \
  --out "$RUN" --resume --gpu-wait-hours 24 --campaign-hours 48
```

Resume hiện chỉ bỏ qua lượt train `complete` có `last.pt`; từ chối ghi đè lượt train dở dang. Chưa hỗ trợ tự resume optimizer giữa lượt hoặc eval dở dang. Không xóa kết quả để ép resume; dùng checkpoint theo mục 8 và thư mục lượt mới nếu cần.

## 7. Infer/eval một checkpoint hoặc decoder perceptual

```bash
export EVAL="$ARTIFACTS/mlicpp/eval-baseline-q1"
python portable/run.py --config "$CFG" end_to_end_eval -- \
  --data "$ARTIFACTS/data/div2k/end-to-end-test32" --out "$EVAL" \
  --quality 1 --checkpoint "$RUN/baseline-q1-seed9041/last.pt"
python portable/run.py --config "$CFG" end_to_end_eval -- \
  --data "$ARTIFACTS/data/div2k/end-to-end-test32" --out "$EVAL" \
  --quality 1 --checkpoint "$RUN/baseline-q1-seed9041/last.pt" --fresh
```

Bỏ `--checkpoint` để đánh giá MLIC++ gốc. Thêm `--cpu` trước tên script để dùng CPU. Đánh giá ảnh đầy đủ trên CPU có thể chậm và dùng nhiều RAM.

Decoder perceptual cũ thuộc **q2**: dùng `--quality 2 --synthesis /duong/dan/reconstruction/last.pt` và bỏ `--checkpoint`; đây là checkpoint `synthesis`, khác với checkpoint full-backbone `state_dict`. Muốn thử bước lượng tử vô hướng thêm `--quantizer-factor 2`; giữ y nguyên mọi option cho lượt `--fresh`. Factor này là cấu hình decoder ngoài stream, phải đóng gói đúng decoder tương ứng khi triển khai.

Đầu ra: `streams/`, `recon/`, `metrics.json`, `decoder.json`, `fresh.json`. Byte thực gồm header ảnh; không tính weights model vào bpp. LPIPS/DISTS đang dùng tile 512 theo evaluator dự án, không đồng nhất với đánh giá của người chấm. Không dùng test để chọn recipe.

## 8. Chuyển checkpoint đã train, không cần train lại

ZIP này **không chứa checkpoint**. Dùng tài khoản và kết nối SSH của bạn; không có token nào đi kèm. Từ máy có quyền truy cập server cũ, ví dụ alias `nexai-sv20` đã được bạn cấu hình:

```bash
mkdir -p "$ARTIFACTS/imported"
scp -r nexai-sv20:/home/jovyan/clic2027-artifacts/mlicpp/end-to-end-20261107/div2k160-fullrd-vs-learnedsteps-3rates-2seeds \
  "$ARTIFACTS/imported/"
scp -r nexai-sv20:/home/jovyan/clic2027-artifacts/mlicpp/full-synthesis-20261105/div2k-q2-full-synthesis-conditional-gan/reconstruction \
  "$ARTIFACTS/imported/"
```

Chỉ sao chép run có `status.json: complete` nếu muốn checkpoint ổn định; không sao chép `last.pt.tmp` hoặc checkpoint đang được ghi. Kiểm SHA-256 file nguồn/đích bằng `sha256sum`; có thể chuyển từng `last.pt` cùng config/status/source_archive để giảm dữ liệu. Mỗi checkpoint backbone chứa optimizer khoảng 1,4 GB. Synthesis nhỏ hơn. Các file imported giữ lại metadata đường dẫn cũ để truy vết.

**Không chạy `--resume` trực tiếp trên campaign imported sau khi sửa config đường dẫn:** runner kiểm config trùng tuyệt đối. Có thể đánh giá trực tiếp các `last.pt` imported bằng mục 7, hoặc chạy các lượt còn thiếu bằng `end_to_end --stage train --quality Q --seed S --arm baseline|joint --out THU_MUC_MOI`, giữ cùng hyperparameter; sau đó tổng hợp rõ nguồn run. Không tự sửa lịch sử config để giả như cùng môi trường.

Muốn dùng đúng nguồn ảnh cũ, chuyển thêm `data/div2k/pilot224` và `data/div2k/end-to-end-test32` từ cây `/home/jovyan/clic2027-artifacts/` vào cùng vị trí tương đối trong `ARTIFACTS` rồi chạy audit.

## 9. Chấm mù

Xem `tools/BLIND_SCORING.md`. Bộ chấm thật cần ảnh cùng tên/kích thước và bitrate đủ gần; không lấy phiếu AI thay người. Trước tiên thử demo:

```bash
python tools/blind_score.py --help
python tools/blind_score.py demo --out "$ARTIFACTS/blind-demo"
```

Đọc subcommand `serve --help` và hướng dẫn đi kèm để mở giao diện. Không công khai private manifest hoặc đáp án codec cho người chấm. Nhận xét về preference chỉ có giá trị sau khi thu phiếu thật.

## 10. Lỗi thường gặp và giới hạn kiểm chứng

- `Need ... MiB free`: chọn GPU khác bằng config mới hoặc chờ; không can thiệp job khác.
- `libnvrtc.so`/cuDNN: dùng launcher portable để đặt `LD_LIBRARY_PATH` và `TORCH_CUDNN_V8_API_DISABLED=1`; kiểm symlink NVRTC trong venv.
- Không build được CompressAI: kiểm compiler C++/Python headers và phiên bản 1.2.0b3; không nâng version tùy tiện.
- Google Drive quota hoặc nguồn DIV2K lỗi: chuyển artifact đã kiểm hash từ máy cũ; không tiếp tục với file HTML hoặc dataset thiếu ảnh.
- Fresh decode không khớp: giữ log, kiểm đúng weights, q, synthesis, factor, runtime. Fresh decode cùng máy chưa bảo đảm bit-exact giữa GPU/CPU hoặc hai loại GPU.
- Không có `--synthesis` checkpoint: nhánh perceptual phải chuyển từ máy cũ hoặc tự train; gói code không chứa trọng số đó.
- Gói đã được kiểm integrity, cấu hình và test stream trên máy đóng gói. Chưa cài lại toàn bộ CUDA/PyTorch hoặc chạy full training trên máy đích; bắt buộc thực hiện smoke ở mục 5.

Các kết quả CPU crop trong báo cáo là development, không phải đủ điều kiện hoặc thứ hạng CLIC. Bộ codec kết hợp cũ vẫn là đối chứng; gói này tập trung workflow MLIC++ và không mang toàn bộ weights của bộ kết hợp.

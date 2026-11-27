# Chạy bước D (GC v2) trên máy A30

**Mục đích.** Bước D là phép thử quyết định cho entropy model và g_s của GC v2: sáu run cộng dồn, mỗi run train rd 100k bước, cùng seed, dữ liệu và thang λ. Tiến độ ghi ở mục 5 và 8 của `plans/GC_v2_architecture_2026-11-23.md`.

| Run | Thay đổi so với run trước |
|---|---|
| D0 | v1 (mốc) |
| D1 | + E1/E2 |
| D2 | + loss CCA 0,3 |
| D3 | + 10 slice × 32 |
| D4 | + tách đường Sq (σ số nguyên trên q, μ float) |
| D5 | + g_s v2 |

Sau mỗi run, script tự:
1. encode stream thật ở 8 level trên 30 ảnh CLIC,
2. chấm Q̂ v0 và LPIPS,
3. tính BD-rate so với D0 và so với run liền trước.

**Cổng giữ một thay đổi:** BD-rate ≤ −2% trên cả Q̂ v0 lẫn LPIPS.

D0 và D1 đã train trên lab09 tới khoảng 21k bước; gói kèm checkpoint để chạy tiếp từ đó.

## 1. Yêu cầu máy

- **GPU:**
  - Một run dùng 5–10 GB VRAM, nên A30 24 GB chạy được 1–2 run cùng lúc. Nếu là A100 40 GB thì chạy được 2–3.
  - Kiểm tra tên GPU bằng `nvidia-smi`; A30 chỉ có bản 24 GB.
  - Không giảm `batch` hay `crop` trong cấu hình, vì kết quả sẽ không còn so được với nhau.
- **CPU, RAM, đĩa:**
  - Khoảng 8 nhân cho mỗi run song song, vì mỗi run có 6 worker DataLoader.
  - RAM ≥ 32 GB.
  - Đĩa khoảng 20 GB: gói 2 GB, dữ liệu 6,8 GB, checkpoint và ảnh giải mã khoảng 8 GB.
- **Python 3.12:**
  - `torch 2.6.0+cu124`, `torchvision 0.21.0`
  - `lpips 0.1.4`, `piq 0.8.0`
  - `numpy 1.26.4`, `pillow`, `scipy`
  - Bản khác có lẽ vẫn chạy, nhưng hãy cố giữ đúng phiên bản, nhất là `piq`, vì Q̂ v0 dùng DISTS và CLIP-IQA của nó.
- **Không cần mạng khi chạy.** Mọi trọng số đã nằm trong gói: AlexNet, VGG16, DISTS và CLIP RN50.

## 2. Chép sang máy A30

Có hai thứ cần chép.

**a) Gói bước D**, khoảng 2 GB: `/mnt/data/khoitm8/clic2027/handoff/stepd_a30.tar`. Gói gồm:
- code bản chụp của bước D,
- cấu hình D0–D5,
- script đánh giá và chấm Q̂,
- trọng số,
- checkpoint D0 và D1.

**b) Dữ liệu**, khoảng 6,8 GB, nằm trong `/mnt/data/khoitm8/clic2027/data/`: `patches512_div2k/` (3.200 file), `patches512_clic2020/` (12.872 file) và `valid30/` (30 ảnh cùng `manifest.json`). Nếu đã chép cho lần hiệu chỉnh λ trước đó thì dùng lại được.

Chạy từ máy A30 (thay `lab09` bằng địa chỉ SSH của máy này):

```bash
mkdir -p ~/stepd && cd ~/stepd
```

```bash
rsync -a lab09:/mnt/data/khoitm8/clic2027/handoff/stepd_a30.tar .
```

```bash
rsync -a lab09:/mnt/data/khoitm8/clic2027/data/{patches512_div2k,patches512_clic2020,valid30} ./data/
```

```bash
tar xf stepd_a30.tar
```

Kiểm tra dữ liệu chép sang có khớp từng file với lab09 không. Lệnh dưới phải im lặng và in `rc=0`:

```bash
cd ~/stepd/data && sha256sum -c --quiet ~/stepd/stepd_a30/data_checksums.sha256; echo rc=$?
```

### Cách khác: tải lại dữ liệu từ nguồn công khai

Không cần chép 6,8 GB từ lab09: mọi dữ liệu đều tải lại được, còn patch được cắt bằng code có seed cố định. Script `make_data.sh` trong gói làm toàn bộ các bước:

| Thư mục | Nguồn | Cách tạo |
|---|---|---|
| `div2k_train/` (800 PNG) | `https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip` (3,3 GB) | giải nén phẳng |
| `clic2020_train/train/` (1.633 PNG) | `https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip` (2 GB) và `.../mobile_train_2020.zip` (4 GB) | giải nén thư mục `train/` của cả hai, bỏ `__MACOSX` |
| `valid30/` (30 PNG) | `https://dhldkwazkze5h.cloudfront.net/data/clic2025_image_test.zip` (116 MB) = bộ test CLIC 2025 | giải nén phẳng |
| `patches512_div2k/` (3.200) | từ `div2k_train/` | `prep_patches`, 4 patch/ảnh, 512 px |
| `patches512_clic2020/` (12.872) | từ `clic2020_train/train/` | `prep_patches`, 8 patch/ảnh, 512 px |

Cần khoảng 9,5 GB tải về, 25 GB đĩa trống, `curl` và `unzip`:

```bash
cd ~/stepd && ./stepd_a30/make_data.sh ~/stepd/data ~/stepd/env/bin/python
```

Script kiểm sha256 của các file zip CLIC, rồi so **từng PNG** với `data_checksums.sha256`, danh sách checksum tạo trên lab09. Kết quả:
- `DATA_OK`: dữ liệu giống hệt lab09.
- `DATA_DIFFERS`: thường do bản Pillow khác lab09 (lab09 dùng Pillow 13.0.0), làm patch bị resample khác đi. Khi đó vẫn chạy được, vì cả sáu run dùng chung một bộ dữ liệu. Nhưng nên xóa `resume/` trước khi chạy để D0/D1 train lại từ đầu trên cùng dữ liệu với D2–D5.

Đã kiểm trên lab09: cắt lại bộ DIV2K cho ra 3.200 patch khớp từng byte. Phần tải về chưa được thử lại.

Sau khi xong có thể xóa `~/stepd/data/dl` (các file zip) để lấy lại khoảng 9,5 GB.

## 3. Môi trường Python

Nếu đã tạo môi trường cho lần hiệu chỉnh λ thì chỉ cần thêm `piq` và `scipy`:

```bash
~/calib/env/bin/pip install piq==0.8.0 scipy
```

Nếu chưa có, tạo môi trường mới:

```bash
python3.12 -m venv ~/stepd/env
```

```bash
~/stepd/env/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

```bash
~/stepd/env/bin/pip install lpips==0.1.4 piq==0.8.0 numpy==1.26.4 pillow scipy tqdm
```

## 4. Chạy

Tham số là `DATA_ROOT`, `GPU_ID`, `PYTHON` và `PARALLEL`. `PARALLEL` là số run chạy cùng lúc: 1 cho A30 24 GB, 2 cho A100 40 GB. Chạy trong tmux:

```bash
tmux new -s stepd
```

```bash
cd ~/stepd/stepd_a30 && ./run_stepd.sh ~/stepd/data 0 ~/stepd/env/bin/python 1
```

Script lần lượt:
1. Ghi cấu hình với đường dẫn dữ liệu của máy đó vào `cfg/`.
2. Chép checkpoint D0/D1 vào `work/`, chỉ ở lần chạy đầu.
3. Train từng run; log ở `logs/Dn.train.log`.
4. Đánh giá sau mỗi run: ảnh giải mã ở `eval/Dn/`, log ở `logs/Dn.encode.log` và `logs/Dn.score.log`.
5. Cập nhật `bd_vs_D0.md` và `bd_step.md` sau mỗi đợt.

**Theo dõi:**

```bash
~/stepd/stepd_a30/status.sh
```

Lệnh này in bước, bpp và it/s gần nhất của từng run, cùng các bảng BD-rate đã có.

**Thời gian ước tính trên A30** (khoảng 3–4 it/s mỗi run):

| Việc | Thời gian |
|---|---|
| Phần còn lại của D0 và D1 (khoảng 79k bước mỗi run) | khoảng 6–7 giờ mỗi run |
| D2 → D5 (100k bước mỗi run) | khoảng 7–9 giờ mỗi run; D5 nặng hơn vì g_s v2 |
| Đánh giá mỗi run (encode 240 ảnh và chấm Q̂) | khoảng 30–45 phút |
| **Tổng với PARALLEL=1** | **khoảng 2 ngày** |

Với `PARALLEL=2` có thể nhanh hơn khoảng 1,3 lần nếu GPU chưa bão hòa. Xem it/s trong `status.sh` để quyết định. Trên A100 40 GB tổng thời gian còn khoảng một nửa.

**Nếu bị ngắt** (mất điện, hết phiên, Ctrl-C): chạy lại đúng lệnh ở trên. Run đang train sẽ tiếp tục từ `work/Dn/rd/last.pt`; run và phần đánh giá đã xong thì được bỏ qua.

## 5. Gửi lại cho mình

- **Sau mỗi run:** `bd_vs_D0.md`, `bd_step.md` và kết quả của `status.sh`. Gửi được giữa chừng; không cần đợi cả sáu run.
- **Khi xong** (log có dòng `STEPD_DONE`): thêm `bd_vs_D0.json`, `eval/*/metrics.csv` và `work/*/rd/eval.jsonl`.
- **Nếu có lỗi:** 50 dòng cuối của log tương ứng trong `logs/`.

Không cần gửi model (`work/*/rd/*.pt`). Hãy giữ chúng lại cho tới khi mình xác nhận.

## 6. Lưu ý

- Code là bản chụp `gcv2/stepd/src_snapshot` trên lab09. Đừng thay bằng code repo khác: các run phải dùng cùng một bản code để so được với nhau.
- Thang λ cố định `[16, 8, 4, 2, 1, 0.5, 0.25, 0.12]` đã được hiệu chỉnh ngày 24/11 (mục 8.1 của plan). Đừng sửa.
- Gói đã được chạy thử trên lab09 GPU 6 ngày 24/11, với cấu hình thu nhỏ (300 bước, 2 ảnh) và `PARALLEL=2`.
- `VRAM_GUARD=off` trong `run_stepd.sh` tắt chốt VRAM của lab09, vì máy A30 chỉ chạy việc này.

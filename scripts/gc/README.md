# GC: codec cảm nhận cho CLIC 2027, chạy end-to-end

**Hướng dẫn đầy đủ** (cấu hình, theo dõi train, đo trên L4, chấm mù, nộp CLIC, xử lý từng tình huống theo mã của `diagnose`): [docs/GC_HUONG_DAN.md](../../docs/GC_HUONG_DAN.md).

Package `src/clic27/gc` cài đặt hệ thống trong [kế hoạch 19/11](../../plans/CLIC2027_win_plan_2026-11-19.md):

| Thành phần | File | Ghi chú |
|---|---|---|
| Backbone ELIC (g_a, g_s), gain unit cho 8 mức rate, β realism, bản đồ chất lượng theo ô 128 px | `model.py`, `layers.py` | g_s nhận (level, β) qua FiLM |
| Entropy model checkerboard × 5 lát kênh không đều, hyper-synthesis | `model.py` | Toàn bộ mạng quyết định CDF là mạng số nguyên (`clic27.entropy.intnet`): CPU và CUDA cho kết quả giống hệt |
| rANS bằng C, cùng định dạng với bản numpy | `_rans.c`, `rans_c.py` | Một lane mỗi blob; `.so` được build sẵn trong gói decoder |
| Bitstream (header, qmap, z, 10 blob y) | `codec.py` | Encoder và decoder dùng chung một hàm `_entropy_pass` |
| Discriminator PatchGAN đa tỉ lệ có điều kiện | `disc.py` | Loss non-saturating, trọng số nhân β |
| Loss: điều khiển rate kiểu HiFiC, MSE, LPIPS, Wasserstein distortion, reward | `losses.py` | |
| Q̂ v2: khoảng cách cảm nhận khả vi fit trên cặp chấm CLIC (Bradley–Terry, pooling vùng xấu nhất học được) | `reward.py` | `fetch` tải crop bằng HTTP range, `fit` báo accuracy giữ riêng 2024 |
| Phân bổ: quỹ đạo bản đồ chất lượng theo crop xấu nhất + knapsack trên ngân sách byte toàn tập | `alloc.py` | Kiểm lại bằng byte thật của `bs.zip` |
| Train 4 giai đoạn rd → gan → rate → align, resume, DDP | `train.py` | |
| Driver end-to-end | `pipeline.py`, `scripts/gc/run_e2e.sh` | |
| Chẩn đoán run theo mã tình huống | `diagnose.py` | Stage cuối của pipeline; ghi `diagnose.md` |
| Đo tốc độ và VRAM của decoder | `bench.py`, `scripts/gc/l4_verify.sh`, `clic/l4/Dockerfile.gc` | Chạy trên L4 trong container giới hạn như server |
| Bộ chấm mù giữa các run | `blind.py` + `tools/blind_score.py` | Crop 512 nơi các run khác nhau nhiều nhất |
| Gói decoder CLIC (`decode`, `bs.zip` → `images/*.png`) | `package.py`, `decode_main.py` | Chỉ cần python3, torch, numpy, pillow |

## Chạy

```bash
export GC_ROOT=/duong/dan/o/dia/lon            # mọi env, cache, data, run nằm dưới thư mục này
bash scripts/gc/setup_env.sh                  # cài đúng phiên bản trong requirements-gc.lock (torch 2.6.0 cu124)
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full prefetch   # trên máy có Internet: tải data, crop reward, trọng số
bash scripts/gc/run_e2e.sh configs/gc_smoke.json runs/smoke   # kiểm code: 10–15 phút
```

Train đầy đủ ở máy khác: sửa `configs/gc_full.json` (`data.train_dirs`, `screen_dirs`, `pipeline.nproc`, số bước), rồi

```bash
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full                 # toàn bộ
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full train           # chỉ train (tự resume)
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full encode package verify report
```

Các stage mặc định: `env data reward train encode package verify report diagnose`; `prefetch` gọi riêng. Mỗi stage bỏ qua phần đã xong. Run tương đối nằm dưới `$GC_ROOT/runs/`.

**Bị thu hồi máy hoặc hết giờ.** SIGTERM / SIGINT / SIGUSR1 gửi cho `run_e2e.sh`, hoặc hết `pipeline.max_minutes_per_stage`:
- trainer lưu `last.pt`, ghi `train/<stage>/partial.json` rồi thoát mã **75**; chạy lại đúng lệnh để train tiếp;
- `model.pt` chỉ được ghi khi stage chạy đủ bước, nên stage sau không bao giờ nhận model train dở;
- ngoài ra `last.pt` được ghi lại mỗi `train.ckpt_minutes` (15 phút).

Có checkpoint GC sẵn thì khai báo `pipeline.init`, ví dụ `{"rd": ".../model.pt"}`; trọng số được nạp strict.

## Đầu ra trong `runs/<tên>/`

- `train/<stage>/{model.pt,last.pt,log.jsonl,eval.jsonl}`
- `reward/{reward.pt,report.json}`
- `encode/bpp-<b>/{bs.zip,allocation.json}`: bitstream nộp và lựa chọn từng ảnh
- `package/{decoder/,decoder.zip,package.json}`: decoder nộp CLIC kèm SHA-256
- `encode/bpp-<b>/verify/`: giải mã lại bằng gói decoder trong tiến trình mới, `metrics.csv`
- `report.md`

## Đã kiểm trên lab09 (H100, GPU 6, 19/11/2026)

- 15 test trong `tests/test_gc.py` (10) và `tests/test_entropy.py` (5) đều qua (20/11). Các test này kiểm:
  - rANS C cho cùng byte với bản numpy.
  - Encode bằng CUDA rồi decode bằng CPU cho ra cùng latent.
  - Conv số nguyên qua đường FP32 tách mảnh cho kết quả bằng đường FP64.
  - Knapsack cho kết quả khớp với vét cạn.
  - `reward.pt` nạp lại cho điểm giống hệt; dữ liệu thiếu thì báo lỗi; patch cũ không bị dùng lại.
  - Dừng vì tín hiệu hoặc hết giờ: có `last.pt`, mã thoát 75, không có `model.pt`; chạy lại thì xong; stage sau không bắt đầu từ stage dang dở; `--init` sai thì báo lỗi.
- DDP 2 rank trên 1 GPU (gloo, 20/11): SIGTERM giữa chừng thì checkpoint ở bước 74; chạy lại resume từ 74 và train tới hết. Qua `run_e2e.sh`, SIGTERM chỉ gửi cho script bash cũng cho mã thoát 75, chạy lại thì đi tiếp `rd → gan → rate`.
- Smoke end-to-end (`configs/gc_smoke.json`) chạy đủ 8 stage. Decoder đóng gói giải mã được 4/4 ảnh trong tiến trình mới. Stream encode bằng đường FP64 giải mã bằng đường FP32 ra đúng cùng ảnh.
- Train ngắn 1.500 bước (model nhỏ): bpp trên crop giảm từ 0,48 xuống khoảng 0,10, PSNR tăng từ 10 lên khoảng 22 dB. Encode trong ngân sách: 0,134 bpp với ngân sách 0,15.
- Model cỡ đầy đủ (31,6 M tham số, g_s 22,9 M), trọng số ngẫu nhiên, một ảnh 2048×1360: entropy decode khoảng 0,4–0,6 s, synthesis khoảng 0,05 s trên H100. rANS C chỉ khoảng 0,08 s; phần lớn còn lại là chuyển dữ liệu CPU↔GPU.

## Giới hạn hiện tại

- **Tốc độ trên L4 chưa đo.** L4 có FP64 rất yếu, nên decode mặc định dùng đường FP32 tách mảnh (`codec.EXACT["mode"] = "f32"`). Đường này cho kết quả bằng FP64, nhưng phải đo trên L4 trong Docker devkit trước khi chốt kích thước mạng context.
- Kiểm tra khớp bit đã có giữa CPU và CUDA trên cùng máy. Kiểm tra giữa máy encode và L4 cần chạy `verify` trên L4.
- Cách server CLIC tính byte của `bs.zip` cần xác nhận bằng một lần nộp thử.
- Loader đọc ảnh lớn khá chậm (smoke đạt khoảng 10 it/s). Với dữ liệu lớn, bật `pipeline.prep_patches` để cắt sẵn patch 512.
- Chưa có EMA của generator, discriminator non-binary kiểu MS-ILLM, hay nhánh diffusion ở 0,075 bpp. Đây là các mở rộng sau khi có model nền.

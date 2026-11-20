# Hướng dẫn GC: train, infer, nộp CLIC 2027 và xử lý tình huống

Tài liệu cho người vận hành codec GC (`src/clic27/gc`) trên máy train riêng. Kế hoạch tổng thể nằm ở [plans/CLIC2027_win_plan_2026-11-19.md](../plans/CLIC2027_win_plan_2026-11-19.md). Mục 9 liệt kê các tình huống có thể gặp, theo mã mà `python -m clic27.gc.diagnose` in ra.

Mục lục

1. Hệ thống làm gì
2. Chuẩn bị máy, môi trường, dữ liệu
3. Cấu hình
4. Chạy pipeline
5. Theo dõi khi train
6. Sau train: encode, đóng gói, kiểm tra
7. So sánh và chọn bản nộp
8. Nộp CLIC
9. Tình huống và cách xử lý
10. Phụ lục

---

## 1. Hệ thống làm gì

```
ảnh ──g_a──► y ──× gain(level, ô chất lượng)──► y_s ──h_a──► z ──► [Gaussian theo kênh] ─► rANS
                                               │                     │
                                               │      h_s (mạng số nguyên) ◄─ z
                                               ▼                     ▼
                         5 lát kênh × 2 lượt checkerboard: mạng số nguyên → (scale, mean) → rANS
                                               │
                                     y_hat_s / gain ──g_s(level, β, ô chất lượng)──► ảnh tái tạo
```

- **Ba ngân sách, một model.** Model có 8 level rate (gain unit). Ở mỗi ngân sách 0,075 / 0,15 / 0,3 bpp, bộ phân bổ chọn level cho từng ảnh. Thể lệ cho phép mỗi rate một decoder; ở đây ba decoder dùng chung gói nhưng vẫn có thể train riêng (mục 7).
- **β (realism).** Encoder chọn β cho từng ảnh và ghi vào stream (4 bit). β cao cho texture thật hơn, β thấp an toàn hơn cho chữ và cạnh.
- **Ô chất lượng 128×128 px.** Mỗi ô nhận giá trị k ∈ [−4, 4], nhân bước lượng tử với 2^(−k/4). Bộ phân bổ dùng ô chất lượng để nâng các crop 512 xấu nhất, đúng đơn vị mà người chấm CLIC xem.
- **Khớp bit.** Mọi thứ quyết định bảng CDF đều tính bằng số nguyên. Máy encode và GPU L4 của server suy ra cùng bảng, nên decoder không bao giờ giải mã sai stream. Synthesis g_s là float/fp16, chỉ ảnh hưởng pixel, không ảnh hưởng việc giải mã.
- **Reward Q̂ v2.** Đây là khoảng cách cảm nhận khả vi, fit trên cặp chấm người của CLIC 2021/2022/2024. Nó được dùng cho ba việc: chấm crop khi phân bổ, làm loss ở stage `align`, và so sánh run.

Các file chính:

| File | Vai trò |
|---|---|
| `config.py` | Toàn bộ khóa cấu hình và giá trị mặc định |
| `model.py`, `layers.py` | Kiến trúc |
| `codec.py`, `rans_c.py`, `_rans.c` | Bitstream |
| `train.py` | Các stage `rd`, `gan`, `rate`, `align` |
| `alloc.py` | Ô chất lượng và knapsack |
| `reward.py` | Q̂ v2 |
| `pipeline.py` | Driver end-to-end |
| `package.py`, `decode_main.py` | Gói decoder CLIC |
| `diagnose.py` | Chẩn đoán run |
| `bench.py` | Đo tốc độ |
| `blind.py` | Tạo bộ chấm mù |

## 2. Chuẩn bị máy, môi trường, dữ liệu

### 2.1 Máy

- **Train:** 1–8 GPU ≥ 24 GB. Batch 16 crop 256 với model mặc định (31,6 M tham số) cùng discriminator chạy được trên một GPU 40–80 GB. GPU 24 GB thì giảm `batch` xuống 8.
- **Đĩa:** dữ liệu cộng patch cắt sẵn, khoảng 200–500 GB với OpenImages cỡ vài trăm nghìn ảnh. Checkpoint chỉ chiếm vài trăm MB mỗi stage.
- **Đo tốc độ cho quyết định:** một máy có GPU NVIDIA L4 và Docker.

### 2.2 Môi trường

```bash
tar xzf clic27_gc_YYYY-MM-DD.tar.gz && cd clic27_gc
export GC_ROOT=/o/dia/lon/clic27        # env, cache pip/torch/HF, data, runs đều nằm dưới đây
bash scripts/gc/setup_env.sh            # cài đúng phiên bản trong scripts/gc/requirements-gc.lock (torch 2.6.0 cu124)
$GC_ROOT/envs/gc/bin/python -m pytest tests/test_gc.py tests/test_entropy.py -q
```

- `setup_env.sh` cài **đúng phiên bản** của env đã chạy test và smoke trên lab09 (`requirements-gc.lock`, Python 3.12).
- Script dừng nếu Python ngoài khoảng 3.10–3.13, nếu thiếu `gcc`, hoặc nếu GPU là Blackwell (sm_100 trở lên, torch 2.6 cu124 không hỗ trợ). Driver NVIDIA cần ≥ 525.
- Trên node không có GPU (ví dụ node login dùng để chạy `prefetch`), script chỉ cảnh báo thiếu CUDA.
- Mọi test phải qua. Nếu `test_c_rans_matches_numpy` in "skipped", máy chưa có `gcc`; xem P2.

### 2.3 Dữ liệu train

| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| OpenImages v7 (tập con 200k–500k ảnh lớn) | Ảnh tự nhiên đa dạng, nguồn chính của các đội top 2024 | Ảnh JPEG: pipeline tự giảm kích thước ngẫu nhiên để xóa artefact |
| LSDIR, DIV2K, Flickr2K | Ảnh chất lượng cao | DIV2K có thể tải bằng `pipeline.download` |
| Ảnh màn hình, game, slide, UI | Tập validation/test CLIC có loại nội dung này | Đặt vào `data.screen_dirs`, lấy mẫu theo `screen_prob` (mặc định 10%) |

- **Không train trên 30 ảnh validation CLIC.** Đó là tập test 2025 và là thước đo duy nhất mình có.
- Kiểm giấy phép dữ liệu trước khi dùng. Whitepaper phải nêu nguồn dữ liệu.

Mỗi mục trong `train_dirs` hoặc `screen_dirs` có thể là một thư mục (quét đệ quy) hoặc một file `.txt` chứa danh sách đường dẫn.

- **Thiếu dữ liệu là lỗi, không bị bỏ qua.** Một mục trong `train_dirs` hoặc `screen_dirs` không tồn tại hoặc không có ảnh sẽ làm train và stage `data` dừng ngay với thông báo nêu rõ thư mục. Khi bắt đầu, log in số ảnh của từng nguồn (`[data] train_dirs: ...`): kiểm dòng này trước khi để train chạy dài.
- **Patch cắt sẵn.** Với tập lớn, bật `pipeline.prep_patches`. Stage `data` cắt sẵn 4 patch 512 mỗi ảnh vào `@data_root/patches512`, và các lần chạy sau tự train trên patch. Đọc PNG 2K trực tiếp chậm: smoke chỉ đạt khoảng 10 it/s vì loader.
- **Patch gắn với nguồn đã cắt.** File `patches512/.done` ghi danh sách nguồn và số ảnh mỗi nguồn. Nếu sau đó bạn thêm hoặc đổi thư mục dữ liệu, pipeline dừng và yêu cầu xóa `patches512/` để cắt lại, thay vì âm thầm train trên bộ patch cũ.
- Thư mục patch còn file của một lần cắt dở (không có `.done`) cũng phải xóa trước khi chạy lại.

### 2.4 Máy train không có Internet: stage `prefetch`

Các bước sau cần Internet:
- tải zip CLIC validation và DIV2K;
- tải crop các câu hỏi chấm CLIC cho reward (HTTP range request tới `downloads.compression.cc`, không tải cả file zip 46–135 GiB);
- tải trọng số VGG16, AlexNet (LPIPS) và DISTS vào `$TORCH_HOME`.

Stage `prefetch` gom hết các việc này. Chạy nó một lần trên máy có mạng, cùng `GC_ROOT` và cùng config:

```bash
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full prefetch
```

Nếu node tính toán dùng ổ khác, chép `$GC_ROOT/runs/full/data/` (hoặc `pipeline.data_root`), `$GC_ROOT/runs/full/reward/` và `$GC_ROOT/cache/` sang. Sau đó mọi stage khác chạy được không cần mạng. `prefetch` không nằm trong danh sách stage mặc định.

Trọng số Q̂ v2 (`reward/reward.pt`) giờ chứa luôn phần VGG, nên điểm khi train và encode khớp đúng điểm lúc fit. File `reward.pt` cũ (thiếu VGG) vẫn nạp được, nhưng dùng VGG ImageNet tải từ `$TORCH_HOME` và in cảnh báo; nên fit lại.

## 3. Cấu hình

`configs/gc_full.json` chỉ ghi những gì khác mặc định trong `config.py`; các dict con được gộp. `@data_root` được thay bằng `pipeline.data_root` (mặc định `<run>/data`). Mỗi lần chạy pipeline sẽ ghi cấu hình đã gộp vào `<run>/config.json`. Các stage train đọc file này, nên sửa config rồi chạy lại là áp dụng từ lần resume tiếp theo.

| Khóa | Ý nghĩa | Khi nào đổi | Cần train lại? |
|---|---|---|---|
| `model.gs_width`, `gs_blocks` | Độ rộng/sâu synthesis, quyết định chất lượng texture và thời gian decode | L4 chậm (giảm) hoặc thừa thời gian (tăng) | Có, từ đầu |
| `model.ctx_hidden`, `slices`, `hyper_hidden` | Kích thước entropy model (số nguyên) | Entropy decode trên L4 chậm | Có |
| `model.n_levels`, `rate.targets` | Số level và bpp mục tiêu theo crop | Không phủ 0,075–0,3 bpp (E1) | Có hoặc fine-tune |
| `model.qmap_block`, `qmap_range` | Kích thước ô chất lượng và biên độ k | Ô chất lượng quá thô hoặc tốn byte | Có |
| `rate.lambda_hi`, `lambda_lo` | Ép rate về mục tiêu (kiểu HiFiC) | T1, T2 | Không, áp dụng khi resume |
| `loss.k_mse`, `k_lpips`, `k_gan`, `beta_max` | Cân bằng fidelity và realism | T7, Q1–Q4 | Không |
| `loss.k_wd` | Wasserstein distortion (VGG) | Muốn texture giống thống kê hơn | Không |
| `loss.k_reward` | Trọng số Q̂ v2 ở stage `align` | R2 (hack reward) | Không |
| `stages.<s>.steps`, `lr`, `lr_d`, `batch`, `levels` | Lịch train từng stage | Theo compute | Không |
| `train.amp`, `ema`, `grad_clip` | bf16, EMA 0,999, clip gradient | T6, NaN | Không |
| `alloc.levels` | Level được xét ở mỗi ngân sách (khóa là chuỗi "0.075"…) | A4 | Không |
| `alloc.betas` | Các β thử cho mỗi ảnh | Q2 (chữ bị bịa) | Không |
| `alloc.objective`, `kappa` | mean / max / mean + κ·max của khoảng cách crop | Thí nghiệm paper | Không |
| `alloc.qmap_iters` | Số bước của quỹ đạo ô chất lượng (mỗi bước là một candidate) | Encode chậm (giảm) | Không |
| `alloc.safety` | Tỉ lệ ngân sách dùng cho knapsack | A2, A3 | Không |
| `alloc.refine_steps` | Tối ưu latent ở encoder (0 = tắt) | Muốn thêm chất lượng, chấp nhận encode chậm | Không |
| `runtime.entropy_device`, `synth_dtype` | Thiết bị cho mạng số nguyên; dtype synthesis | D5 | Không, đóng gói lại là đủ |
| `train.ckpt_minutes` | Ngoài `ckpt_every`, ghi `last.pt` mỗi N phút (mặc định 15) | Máy hay bị thu hồi (giảm) | Không |
| `train.init_strict` | Trọng số `--init` / stage trước phải khớp model hoàn toàn (mặc định `true`) | Cố ý đổi kiến trúc giữa chừng | Không |
| `pipeline.init` | `{"rd": "/path/model.pt"}`: stage đó bắt đầu từ trọng số này thay vì từ đầu | Có checkpoint GC sẵn (run trước, bản nhỏ hơn) | Không |
| `pipeline.max_minutes_per_stage` | Ngân sách phút cho mỗi lần gọi stage; hết giờ thì checkpoint và thoát 75 | Máy thuê theo khung giờ | Không |
| `pipeline.nproc` | Số GPU (DDP qua torchrun) | Theo máy | Không |
| `pipeline.max_images` | Giới hạn số ảnh validation khi encode | Chạy thử nhanh | Không |

## 4. Chạy pipeline

Thư mục run tương đối được đặt dưới `$GC_ROOT` (`runs/full` → `$GC_ROOT/runs/full`).

```bash
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full prefetch   # một lần, trên máy có mạng (mục 2.4)
bash scripts/gc/run_e2e.sh configs/gc_smoke.json runs/smoke    # 10–15 phút: kiểm toàn bộ code trên máy mới
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full      # toàn bộ
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full train                      # chỉ train
bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full encode package verify report diagnose
```

- **Thứ tự stage:** `env → data → reward → train → encode → package → verify → report → diagnose` (cộng `prefetch` khi gọi riêng).
- **Stage `env` dừng nếu không build được thư viện rANS C** (thiếu `gcc`). Bản numpy chậm khoảng 100 lần và server CLIC không có compiler, nên gói decoder bắt buộc phải có `_rans.so`. Chỉ đặt `GC_ALLOW_NUMPY_RANS=1` cho lần chạy debug rất nhỏ.
- **Tiếp tục khi bị ngắt.** Chạy lại đúng lệnh cũ.
  - Train tiếp từ `train/<stage>/last.pt`, file này được ghi nguyên tử nên không hỏng khi máy tắt giữa chừng.
  - Encode dùng lại `encode/candidates.pt`.
- **Bị thu hồi máy / hủy job (SIGTERM, SIGINT, SIGUSR1).** Gửi tín hiệu cho `run_e2e.sh`, pipeline hay torchrun đều được.
  - Trainer lưu `last.pt` ngay ở bước kế tiếp (dưới 1 giây), ghi `train/<stage>/partial.json` và thoát mã **75**. `run_e2e.sh` in hướng dẫn chạy lại.
  - Các stage sau không chạy.
  - Bị kill cứng (SIGKILL, mất điện) thì mất tối đa `train.ckpt_minutes` phút.
  - torchrun cho worker khoảng 30 giây sau SIGTERM, đủ để lưu checkpoint.
- **`model.pt` chỉ được ghi khi stage chạy đủ số bước.** Stage dừng sớm (hết `max_minutes_per_stage` hoặc nhận tín hiệu) không bao giờ chuyển model train dở sang stage sau. Gọi stage sau khi stage trước còn dang dở (có `last.pt` mà không có `model.pt`) sẽ báo lỗi, chứ không âm thầm train từ trọng số ngẫu nhiên.
- **Train lại một stage từ đầu:** xóa `runs/<run>/train/<stage>/`. Stage sau tự lấy `model.pt` của stage trước.
- **Bắt đầu từ checkpoint có sẵn:** đặt `pipeline.init`, ví dụ `{"rd": "/data/old_run/train/rd/model.pt"}`.
  - Nạp strict: sai kiến trúc là dừng ngay.
  - Khi stage đã có `last.pt` riêng thì resume được ưu tiên, `init` bị bỏ qua.
- **Encode lại sau khi đổi model:** xóa `runs/<run>/encode/`, vì candidate cũ gắn với model cũ.
- **Nhiều GPU:** đặt `pipeline.nproc`; pipeline gọi `torch.distributed.run`.
  - Mọi quyết định dừng (NaN, tín hiệu, hết giờ) được thống nhất giữa các rank, nên không rank nào bị treo.
  - Nếu số GPU nhìn thấy ít hơn số rank (thử DDP trên 1 GPU), backend tự chuyển sang `gloo`. Có thể ép bằng `GC_DDP_BACKEND`.
  - Đã kiểm 2 rank trên 1 GPU với gloo; NCCL nhiều GPU chưa chạy, nên chạy `gc_smoke.json` với `nproc` bằng số GPU trước khi chạy full.
- **Giới hạn thời gian mỗi stage:** dùng `pipeline.max_minutes_per_stage`. Khi hết giờ, stage lưu checkpoint và thoát mã 75; chạy lại để train tiếp. `model.pt` chỉ có khi đủ bước.
- **Chạy tay một stage train:**
  ```bash
  python -m clic27.gc.train --config $RUN/config.json --work $RUN/train --stage gan [--init X.pt] [--max-minutes M]
  ```

Thời gian train phụ thuộc phần cứng và tải máy. Hãy đo `it_s` trong 1.000 bước đầu rồi mới chốt số bước. Đội HIT ở CLIC 2024 báo cáo khoảng 2 triệu iteration.

## 5. Theo dõi khi train

`train/<stage>/log.jsonl` ghi một dòng mỗi `log_every` bước:

| Trường | Ý nghĩa | Mong đợi |
|---|---|---|
| `bpp` | Rate ước lượng trên crop | Dao động quanh mục tiêu của các level |
| `over` | Tỉ lệ mẫu vượt bpp mục tiêu của level | Khoảng 0,3–0,6 khi đã hội tụ (T1, T2) |
| `psnr`, `mse` | Fidelity trên crop | Tăng ở `rd`. Có thể giảm 0,5–1,5 dB khi sang `gan`; đó là chuyện bình thường |
| `lpips`, `wd`, `reward` | Các khoảng cách cảm nhận | Giảm dần |
| `g` | Loss generator (non-saturating, đã nhân β) | Ổn định, không tăng mãi |
| `d`, `d_real`, `d_fake` | Loss của D; P(real) trên ảnh thật và trên ảnh tái tạo | `d_real` khoảng 0,6–0,9, `d_fake` khoảng 0,1–0,4 (T4, T5) |
| `gnorm` | Norm gradient trước khi clip | Không bị clip liên tục (T6) |
| `it_s` | Tốc độ train | Ổn định (T8) |

`train/<stage>/eval.jsonl` ghi đánh giá trên `eval_images` ảnh validation full-size với rate ước lượng: bpp, PSNR và LPIPS theo từng level. Xem mục E1 và T3.

Chạy chẩn đoán bất cứ lúc nào, kể cả khi đang train:

```bash
python -m clic27.gc.diagnose --work $RUN      # ghi $RUN/diagnose.md
```

## 6. Sau train: encode, đóng gói, kiểm tra

1. **`encode`.** Với mỗi ảnh, mỗi level và mỗi β, đi một quỹ đạo ô chất lượng theo crop xấu nhất (`qmap_iters + 1` candidate). Knapsack chọn một candidate mỗi ảnh để tổng mục tiêu nhỏ nhất dưới ngân sách. Sau đó entropy-code thật và đo lại kích thước `bs.zip`; nếu vượt thì hạ `safety` và chạy lại. Kết quả nằm ở `encode/bpp-<b>/{bs.zip, allocation.json}`.
2. **`package`.** Tạo `package/decoder/` và `decoder.zip`, gồm `decode`, `decode_main.py`, `lib/`, `_rans.so` build sẵn và `model.pt` đã bỏ phần encoder. SHA-256 nằm trong `package.json`.
3. **`verify`.** Giải mã từng `bs.zip` bằng chính gói decoder, trong tiến trình mới và không có PYTHONPATH của repo. Tính PSNR, MS-SSIM, LPIPS, DISTS và ghi `expected.json` (hash pixel) cho bước kiểm trên L4.
4. **Đo trên L4**, bắt buộc trước khi chốt kiến trúc:
   ```bash
   bash scripts/gc/l4_verify.sh $RUN/package/decoder.zip $RUN/encode/bpp-0.075/bs.zip <thư_mục_ảnh_gốc_có_expected.json>
   python -m clic27.gc.bench --decoder $RUN/package/decoder --bs $RUN/encode/bpp-0.075/bs.zip
   ```
   Chép `encode/bpp-0.075/expected.json` vào thư mục ảnh gốc để script so cả hash pixel. Hash có thể khác nhẹ giữa GPU khác loại vì synthesis là fp16 (xem D4). Kích thước và số ảnh thì phải khớp tuyệt đối.
5. **`diagnose`.** Đọc `diagnose.md` và xử lý mọi dòng LỖI hoặc CHÚ Ý theo mục 9.

## 7. So sánh và chọn bản nộp

Người chấm quyết định kết quả, metric chỉ để sàng lọc. Quy trình:

1. Tạo bộ chấm mù giữa hai run (hoặc với bộ codec kết hợp cũ), ở cùng ngân sách:
   ```bash
   python -m clic27.gc.blind --ref <ảnh_gốc> --out study_in_015 --budget 0.15 \
     --run A=runs/x/encode/bpp-0.15/verify/images --run B=runs/y/encode/bpp-0.15/verify/images
   python tools/blind_score.py prepare --config study_in_015/config.json --out study_015
   python tools/blind_score.py serve --study study_015
   ```
   Mặc định mỗi ảnh có hai crop 512: crop mà hai run khác nhau nhiều nhất theo LPIPS, cộng một crop ngẫu nhiên.
2. **Cỡ mẫu.** Để thấy chênh lệch 60/40 với độ tin cậy 95% cần khoảng (1,96 / (2 × 0,1))² ≈ 96 phiếu quyết định (không tính "khó phân biệt"). Chênh lệch 55/45 cần khoảng 385 phiếu. Nên có ít nhất 3 người chấm để không phụ thuộc vào gu của một người.
3. **Quy tắc chọn.** Bản mới chỉ thay bản cũ khi thắng trong chấm mù và vẫn đạt thời gian trên L4. LPIPS, DISTS hay Q̂ tốt hơn mà người chấm không xác nhận thì không đủ.
4. **Decoder riêng theo rate.** Chạy `rate` với `stages.rate.levels` chỉ gồm các level của một ngân sách, ở ba thư mục run khác nhau (cùng khởi tạo từ `gan/model.pt`), rồi đóng gói ba decoder. Chỉ làm vậy nếu chấm mù cho thấy lợi ích.

## 8. Nộp CLIC

Lịch: validation đóng 01/03/2027; test 02–09/03; whitepaper hạn 16/03 (AoE).

- [ ] Ba ngân sách đều có `bs.zip` với `within_budget: true`. Kiểm `allocation.json`.
- [ ] Mỗi decoder định dùng ở vòng test **đã được nộp ở vòng validation**. Hash phải trùng, nên lưu `package.json`.
- [ ] `decoder.zip` dưới 4 GB và có `_rans.so` (P1, P2).
- [ ] `l4_verify.sh` chạy được trên L4 với giới hạn 2 CPU, 12 GB RAM, không mạng. Thời gian cho 30 ảnh nằm trong mục tiêu (gợi ý ≤ 20 s; các đội top 2025 mất 21–29 s).
- [ ] Nộp một lần sớm (ví dụ giữa tháng 12) để xác nhận server chạy được gói, đọc được `bs.zip` đúng cách và tính byte giống mình (S1).
- [ ] Leaderboard validation chỉ hiện PSNR và MS-SSIM. PSNR thấp hơn VTM là chuyện bình thường với codec GAN (Vcoder 2025: 24,0 dB so với VTM 26,5 dB), không phải lỗi.
- [ ] Khi encode bộ test: dùng **đúng** `model.pt` đã đóng gói (so SHA-256), chạy `encode → package → verify`, kiểm `diagnose`.
- [ ] Whitepaper ghi nguồn dữ liệu, kiến trúc, loss, cách phân bổ bit và việc dùng dữ liệu chấm CLIC cho reward.

## 9. Tình huống và cách xử lý

Mỗi mục gồm dấu hiệu, nguyên nhân có thể, cách kiểm tra và cách xử lý (làm theo thứ tự). Mã T, E, R, A, D, P trùng với `diagnose`.

### 9.1 Khi train

**T0. Theo dõi tiến độ (thông tin).** `diagnose` in số bước, tốc độ và việc đã có `model.pt` chưa. Một stage có `last.pt` mà chưa có `model.pt` là đang chạy hoặc bị ngắt: chạy lại cùng lệnh để tiếp tục. `partial.json` cho biết lần dừng gần nhất (bước, lý do: `signal` hay `max_minutes`).

**T0b. Lệnh kết thúc với mã 75.** Đây không phải lỗi: train đã checkpoint vì nhận tín hiệu hoặc hết `max_minutes_per_stage`. Chạy lại đúng lệnh. Nếu muốn dừng hẳn một stage ở số bước hiện tại, giảm `stages.<s>.steps` xuống bằng bước đó rồi chạy lại (stage sẽ ghi `model.pt` ngay).

**T1. Rate không xuống mục tiêu (`over` > 0,6 kéo dài).**
- *Nguyên nhân:* `lambda_hi` nhỏ so với các trọng số distortion; mới train ít bước; `log_gain` của level quá lớn.
- *Kiểm tra:* xem `over` và `bpp` theo thời gian, và `eval.jsonl` từng level.
- *Xử lý:*
  1. Train thêm. Ở `rd`, rate thường cần vài chục nghìn bước mới bám mục tiêu.
  2. Tăng `rate.lambda_hi` (×2).
  3. Nếu chỉ các level thấp không xuống được, hạ `rate.targets` của các level đó một chút hoặc thêm level thấp hơn (phải train lại).

**T2. Rate thấp hơn mục tiêu nhiều (`over` < 0,15).** Model đang bỏ phí chất lượng: phạt rate quá mạnh hoặc mục tiêu quá cao. Giảm `lambda_lo`. Nếu sau phân bổ các ảnh vẫn dồn về level cao nhất (A4) thì đây không phải vấn đề.

**T3. bpp hoặc PSNR không tăng theo level.**
- *Nguyên nhân:* các level chưa tách. Gain unit mới khởi tạo, hoặc `rate` fine-tune chỉ trên một vài level làm các level khác trôi.
- *Kiểm tra:* `eval.jsonl` và `model.log_gain.exp().mean(1)` (phải tăng dần).
- *Xử lý:*
  1. Train thêm `rd` với đủ 8 level.
  2. Ở `rate`, giữ một dải level liền nhau (mặc định trong full config là 1–6).
  3. Trong `alloc.levels`, chỉ dùng những level đã đơn điệu.

**T4. Discriminator thắng tuyệt đối (`d_real` > 0,95 và `d_fake` < 0,05).** Generator không còn tín hiệu hữu ích, ảnh dễ có artefact lặp.
1. Giảm `lr_d` (×0,5).
2. Tăng dần `k_gan` hoặc `beta_max`.
3. Kiểm tra stage `gan` có khởi tạo từ `rd/model.pt` đã hội tụ không. Bắt đầu GAN từ model chưa hội tụ thường dẫn tới tình huống này.

**T5. D không phân biệt được (`d_real` ≈ `d_fake`).** Có hai khả năng:
- G đã rất tốt; khi đó LPIPS và chấm mù cũng tốt.
- D không học: kiểm tra `lr_d` khác 0 và `d` có giảm ở giai đoạn đầu không.

Nếu ảnh vẫn nhòe mà D không phân biệt được, tăng `lr_d` hoặc độ rộng D (`MultiScaleD(width=...)` trong `disc.py`).

**T6. Gradient bị clip liên tục, hoặc loss NaN/Inf.** Train tự bỏ qua bước NaN và dừng sau `max_bad_steps` bước liên tiếp.
1. Giảm `lr` (×0,5).
2. Kiểm tra dữ liệu: ảnh hỏng hoặc toàn đen làm loss bằng 0. Loader đã bỏ qua file không đọc được nhưng không bỏ qua ảnh "lạ".
3. Đổi `train.amp` sang `none` để loại trừ lỗi bf16.
4. Resume từ `last.pt` cũ hơn: lưu bản sao `last.pt` định kỳ, vì pipeline chỉ giữ bản mới nhất.

**T7. PSNR giảm mạnh ở stage GAN (> 2 dB).** GAN đang hy sinh fidelity, dễ làm sai màu và sai cấu trúc. Tăng `k_mse` hoặc `k_lpips`, giảm `beta_max` hoặc `k_gan`. Đánh giá bằng chấm mù, không chỉ bằng PSNR.

**T8. Train chậm (`it_s` thấp).**
- *Kiểm tra:* `nvidia-smi` có cho thấy GPU rảnh không; nếu có, nút thắt là loader.
- *Xử lý:*
  1. Bật `pipeline.prep_patches`.
  2. Tăng `data.workers`.
  3. Đặt dữ liệu trên SSD.
  4. Tắt `k_wd` (thêm một VGG forward).
  5. Dùng nhiều GPU.

**E1. Các level không phủ 0,075–0,3 bpp trên ảnh full.** Phân bổ sẽ không đạt ngân sách (A1) hoặc phí byte. Chỉnh `rate.targets` hoặc thêm level ở đầu thiếu, rồi fine-tune `rd` hoặc `rate` với đủ level.

**E2. bpp ảnh full khác mục tiêu crop.** Bình thường: ảnh 2K có nhiều vùng phẳng hơn crop 256. Phân bổ làm việc trên ảnh full, nên chỉ cần các level phủ đủ dải (E1).

**R0. Không có reward model.** Bộ phân bổ dùng LPIPS-Alex; stage `align` bị bỏ qua.

**R1. Q̂ v2 không hơn khởi tạo trên tập giữ riêng 2024.**
- *Nguyên nhân:* quá ít câu hỏi (`n_per_set`), hoặc dữ liệu 2024 khác phân phối các năm trước.
- *Xử lý:* tăng `reward.n_per_set` (4000 → 8000) và `steps`. Nếu vẫn không hơn, đặt `pipeline.use_reward: false` và dùng LPIPS cho phân bổ.
- *Lưu ý:* độ đồng thuận giữa từng người chấm CLIC chỉ khoảng 72–74%; accuracy trên từng phiếu khó vượt mức này nhiều.

**R2. Hack reward ở stage `align`.** Reward giảm nhưng LPIPS hoặc DISTS xấu đi, hoặc chấm mù thua bản trước `align`. Ví dụ đã gặp trong literature: model xóa chữ hoặc tạo texture lặp.
1. Giảm `loss.k_reward`.
2. Rút ngắn `align`.
3. Giữ `rate/model.pt` làm phương án; chỉ dùng `align` nếu thắng chấm mù.

### 9.2 Khi encode và phân bổ

**A1. Không có phân bổ khả thi (`feasible: false`).** Model không xuống đủ thấp, thường ở 0,075 bpp. Pipeline vẫn ghi stream ở mức rate thấp nhất và báo lỗi; không được nộp `bs.zip` này.
1. Thêm level thấp vào `alloc.levels["0.075"]`.
2. Nếu mọi level đều đã được xét, train thêm hoặc hạ `rate.targets` của level thấp (E1).
3. Tạm thời: đặt `alloc.qmap_iters` > 0, vì quỹ đạo có cả bước hạ chất lượng ở crop tốt.

**A2. `bs.zip` vượt ngân sách sau khi code thật.** Pipeline tự hạ `safety` tối đa 8 lần. Nếu vẫn vượt, ước lượng rate lệch nhiều (A5). Giảm `alloc.safety` (0,995 → 0,98) rồi encode lại.

**A3. Dùng dưới 97% ngân sách.** Phí byte. Tăng `alloc.safety` về 0,998, thêm candidate (`qmap_iters`, nhiều `betas`), hoặc mở rộng `alloc.levels` lên trên.

**A4. Nhiều ảnh dồn vào level biên.** Dải level đang hẹp. Mở rộng `alloc.levels` của ngân sách đó về phía bị dồn.

**A5. Byte thật lệch ước lượng > 5%.**
- *Nguyên nhân:* bảng scale rời rạc (64 mức), escape, hoặc ước lượng tính ở bf16.
- *Kiểm tra:* `allocation.json`, so `bytes` với `est_bytes`.
- *Xử lý:* chấp nhận và hạ `safety` tương ứng. Nếu lệch có hệ thống ở level thấp, đó là chi phí header và hyper; tăng hằng số `+ 24` byte trong `st_encode` của `pipeline.py`.

**A6. Encode quá chậm.** Mỗi ảnh tốn (số level) × (số β) × (`qmap_iters` + 1) lượt forward cộng phần chấm crop. Giảm `qmap_iters` và số β, chấm crop bằng `stride` 512 thay vì 256. Encode không bị giới hạn thời gian ở CLIC, nhưng cần xong trong tuần test.

### 9.3 Khi decode và chạy trên L4

**D1. Decoder ra thiếu ảnh hoặc sai kích thước.** Xem log của `verify`. Các nguyên nhân thường gặp:
- Tên file trong `bs.zip` không phải `<tên>.gc`.
- Server đã giải nén `bs.zip` vào thư mục làm việc. Decoder đã xử lý trường hợp này: không thấy `bs.zip` thì đọc `*.gc` ở thư mục hiện tại.
- Ảnh có cạnh > 65535 (header là u16).

**D2. Thời gian decode (thông tin).** Số đo trên máy train không đại diện cho L4. Luôn dùng `l4_verify.sh` và `bench`.

**D3. Lỗi "rANS integrity check failed" hoặc "trailing bytes".** Stream và decoder không cùng model: encode bằng một `model.pt`, đóng gói bằng `model.pt` khác (ví dụ train tiếp sau khi encode, hoặc nhầm `model_raw.pt` với `model.pt` EMA).
1. Đối chiếu SHA-256 trong `package.json` với gói đang dùng.
2. Chạy lại `encode package verify` liền nhau, cùng một run.
3. Không sửa `model.pt` sau khi encode bộ test.

**D4. Ảnh decode trên L4 khác hash so với máy train.**
- Nếu chỉ khác ít, mắt thường không thấy: do synthesis fp16 khác nhau giữa các GPU. Không ảnh hưởng tính đúng của stream. Có thể đặt `runtime.synth_dtype: "fp32"` nếu cần ảnh trùng tuyệt đối (chậm hơn).
- Nếu ảnh vỡ hoàn toàn: phần entropy bị lệch, không nên xảy ra vì mạng số nguyên khớp bit trên mọi thiết bị.
  1. Chạy `pytest tests/test_gc.py` trên chính máy L4.
  2. Kiểm tra không ai đổi `codec.EXACT` hoặc bật TF32/cuDNN trong `_ent_ctx`.
  3. Kiểm tra `model.pt` (D3).

**D5. Decode trên L4 quá chậm (> 0,7 s/ảnh, tức > 20 s cho 30 ảnh).** Đọc `bench.json` để biết phần nào chậm.
- *Entropy chậm:*
  1. Giữ `exact=f32` (mặc định); FP64 trên L4 rất chậm.
  2. Thử `runtime.entropy_device: "cpu"` (thường chậm hơn, vì server chỉ có 2 CPU).
  3. Giảm `model.ctx_hidden` hoặc gộp lát kênh (`slices` ít phần tử hơn); phải train lại.
  4. Tối ưu code: gộp các lần chuyển CPU↔GPU trong `_entropy_pass`. Phần này chiếm phần lớn thời gian trong đo đạc trên H100.
- *Synthesis chậm:*
  1. Giữ `synth_dtype: fp16`.
  2. Giảm `model.gs_width` hoặc `gs_blocks`, phải train lại. Mỗi lần giảm, kiểm tra chất lượng bằng chấm mù.
- *Khởi động chậm (load model, import torch):* tính một lần cho cả bộ. Thời gian 30 ảnh ≈ `load_s` + 30 × `per_image_s`.

**D6. Hết bộ nhớ GPU trên L4.** Ảnh rất lớn cộng `gs_width` lớn. Kiểm tra `peak_vram_gib` trong `bench.json`. Giảm width, hoặc thêm synthesis theo tile có chồng lấn (chưa cài, cần viết thêm trong `codec.decompress`).

**D7. Hết RAM 12 GB (CPU).** Ảnh được giải mã lần lượt, PNG ghi song song 2 luồng. Nếu vẫn hết RAM, giảm số luồng ghi trong `decode_main.py` xuống 1.

### 9.4 Khi đóng gói và nộp

**P1. `decoder.zip` gần 4 GB.** Không xảy ra với kiến trúc mặc định (vài trăm MB). Nếu thêm nhiều model (ba decoder riêng, nhánh diffusion), lưu g_s ở fp16 trong `package.strip_model`.

**P2. Gói không có `_rans.so`, hoặc `.so` không load được trên server.** Không có `.so` thì decoder rơi về rANS numpy, rất chậm.
- *Kiểm tra:* `objdump -T lib/clic27/gc/_rans.so | grep GLIBC`. File chỉ cần `malloc`, `free` và `memcpy` của glibc.
- *Xử lý:* chạy `package` trên máy có `gcc` và glibc không mới hơn image của server (Ubuntu 22.04 an toàn), rồi chạy lại `l4_verify.sh`; image Docker trong script không có compiler nên sẽ lộ lỗi này.

**S1. Server tính byte khác mình.** Mình tính trên toàn bộ `bs.zip` (stored, không nén). Nộp thử sớm và so bpp trên leaderboard với `allocation.json`. Nếu server tính cao hơn, tăng `zip_overhead` trong `pipeline.py` hoặc giảm `safety`.

**S2. Quên đăng ký decoder ở validation.** Không cứu được ở vòng test. Mọi biến thể decoder có thể dùng phải được nộp trước 01/03.

**S3. Nộp nhầm stream của model khác.** Xem D3. Giữ một thư mục `submission/<ngày>/` chứa `bs.zip`, `decoder.zip`, `package.json` và `allocation.json` của đúng lần nộp.

### 9.5 Chất lượng ảnh nhìn bằng mắt

**Q1. Ảnh nhòe ở 0,075 bpp.** Realism chưa đủ.
1. Tăng `alloc.betas` lên 1,0.
2. Tăng `k_gan` hoặc `beta_max`, train thêm `gan`.
3. Kiểm tra T5.

**Q2. Chữ bị méo hoặc bị "bịa".** Đây là lỗi nặng nhất với người chấm.
1. Thêm `0.0` hoặc `0.5` vào `alloc.betas` để bộ phân bổ có thể chọn β thấp cho ảnh có chữ.
2. Tăng `screen_prob` và bổ sung dữ liệu có chữ.
3. Đo OCR bằng `clic27.eval.metrics` (`detect_text`, `cer`) trên các ảnh có chữ trong validation.

**Q3. Lệch màu hoặc mảng màu loang.** Hiện tượng hay gặp của GAN. Tăng `k_mse`; kiểm tra T7. Nếu chỉ xảy ra ở level thấp, có thể thêm một loss màu ở độ phân giải thấp (chưa cài).

**Q4. Artefact dạng lưới hoặc ô cờ.** Thường do sub-pixel upsampling trong g_s khi train GAN chưa ổn định.
1. Giảm `lr`.
2. Dùng EMA (mặc định bật) và so `model.pt` với `model_raw.pt`.
3. Kiểm tra T4.

**Q5. Viền ảnh xấu.** Ảnh được pad replicate tới bội số 64 rồi cắt lại. Nếu viền phải hoặc viền dưới xấu, kiểm tra với ảnh có kích thước không chia hết 64.

**Q6. Mặt người hoặc chi tiết nhỏ bị biến dạng ở một vùng.** Đúng là trường hợp mà phân bổ theo crop xấu nhất nhắm tới.
1. Tăng `alloc.kappa` hoặc dùng `objective: "max"`.
2. Tăng `qmap_iters`.
3. Dùng Q̂ v2 thay LPIPS cho việc chấm crop.

**Q7. Metric tốt nhưng chấm mù thua.** Không tin metric. Xem các crop thua (`blind.py --crops disagree`) để biết loại lỗi, rồi quay lại Q1–Q6. Nếu có đủ phiếu, có thể thêm chính các phiếu này vào dữ liệu reward.

### 9.6 Rủi ro chung

- **Overfit vào 30 ảnh validation.** Mọi tham số phân bổ (κ, β, safety…) đều được chọn trên 30 ảnh này. Bộ test sẽ tương tự nhưng khác. Ưu tiên tham số đơn giản và ổn định; đừng chọn một cấu hình chỉ thắng nhờ 1–2 ảnh.
- **Máy chạy dùng chung.** Thời gian và tốc độ đo khi GPU có tải khác không đại diện. Số liệu L4 phải đo trên máy L4 riêng.
- **Giấy phép trọng số và dữ liệu.** Không dùng trọng số CC-BY-NC (như MS-ILLM) làm khởi tạo cho bản nộp nếu chưa rõ điều khoản.

## 10. Phụ lục

### 10.1 Định dạng stream

```
"GC" | version u8 | H u16 | W u16 | level u8 | beta u8 | qmap flag u8
[qmap blob] | z blob | (anchor, non-anchor) × 5 lát kênh
```

Mỗi blob gồm varint độ dài, theo sau là rANS một lane cộng phần escape. Chi tiết trong `codec.py`.

### 10.2 Vì sao khớp bit

- Mạng hyper và mạng context dùng trọng số int8 với bias và activation fixed-point (`clic27.entropy.intnet`).
- Khi decode, phép conv chạy trên số nguyên:
  - Đường `f64`: số nguyên chứa trong float64.
  - Đường `f32` (mặc định): tách đầu vào thành các mảnh 6 bit để mọi tổng riêng phần là số nguyên < 2^24, tắt TF32, cuDNN và oneDNN.
- Kết quả là số nguyên chính xác, không phụ thuộc thứ tự cộng hay thiết bị. Test `test_exact_f32_equals_f64` và `test_roundtrip_cpu_and_cuda` kiểm điều này.
- Bảng CDF được tính một lần từ float khi tạo model, rồi lưu dưới dạng số nguyên trong checkpoint.

### 10.3 Lệnh tiện ích

Chạy từ thư mục gốc của bundle. `run_e2e.sh` tự đặt các biến dưới đây; khi gọi lệnh lẻ thì cần đặt tay. Run nằm dưới `$GC_ROOT/runs/`, không phải trong thư mục bundle:

```bash
export PYTHONPATH=$PWD/src TORCH_HOME=$GC_ROOT/cache/torch TMPDIR=$GC_ROOT/tmp
PATH=$GC_ROOT/envs/gc/bin:$PATH
RUN=$GC_ROOT/runs/full
```

```bash
python -m clic27.gc.diagnose --work $RUN
python -m clic27.gc.bench --decoder $RUN/package/decoder --bs $RUN/encode/bpp-0.15/bs.zip
python -m clic27.gc.reward fit --out $RUN/reward --steps 8000
python -m clic27.gc.blind --ref <gốc> --out <in> --run A=<dir> --run B=<dir>
python -m clic27.gc.rans_c build
```

### 10.4 Thư mục một run

```
runs/<run>/
  config.json          cấu hình đã gộp (stage train đọc file này)
  run.log              log toàn bộ
  data/                clic_valid/, downloads/, patches512/
  reward/              crops/, index.jsonl, reward.pt, report.json
  train/<stage>/       last.pt, model.pt (EMA), model_raw.pt, disc.pt, log.jsonl, eval.jsonl
  encode/              candidates.pt, summary.json, bpp-<b>/{bs.zip, allocation.json, expected.json, verify/}
  package/             decoder/, decoder.zip, package.json
  verify.json, report.md, diagnose.md
```

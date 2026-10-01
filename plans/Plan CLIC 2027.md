# Kế hoạch CLIC 2027: Image Compression, GPU track

Mục tiêu: giành hạng 1 ở track Image GPU của CLIC 2027 (https://clic2027.compression.cc/), dùng lại codec Rate-as-Time Flow Bridge của bản nộp CVPR.

Kế hoạch có hai nhánh:
- **A. Hạ tầng decoder và entropy coding tất định**: đảm bảo nộp được, giải mã đúng bit trên server, đủ nhanh và đủ nhỏ.
- **B. Đo mức hỏng trên ảnh game/màn hình**: biết codec hiện tại thua ở đâu (loại nội dung nào, mức bpp nào) để quyết định có cần thêm nhánh xử lý riêng hay không.

---

## 0. Các sự thật đã kiểm tra (29/9/2026)

| Mục | Giá trị | Nguồn |
|---|---|---|
| Mức bpp | 0.075 / 0.15 / 0.3. Phải nộp đủ cả 3 mới được xét giải | trang /tasks |
| Cách chấm | Người chấm so sánh từng cặp ảnh với ảnh gốc, xếp hạng bằng Elo. Leaderboard validation chỉ hiện PSNR/MS-SSIM | /tasks, leaderboard |
| Phần cứng | L4 24GB, 2 CPU, RAM 12GB, giới hạn 5 giờ cho mỗi lần nộp, không có mạng | FAQ, devkit |
| Luật tốc độ | 25% bài nộp giải mã chậm nhất mỗi track bị loại khỏi giải | FAQ |
| Decoder | Tối đa 4GB, file zip có script `decode` ở thư mục gốc. Server tự giải nén `bs.zip` và ghi ảnh vào `images/<tên gốc>.png` | devkit README |
| Môi trường | `nvidia/cuda:12.9.0-runtime-ubuntu24.04`, Python 3.12, **torch 2.6.0**, triton 3.2.0, numpy 1.26.4, **range-coder 1.1**. Không có diffusers hay compressai | `refs/repos/clic-devkit/docker/` |
| **Luật đóng băng** | Decoder dùng ở vòng test phải **có hash giống hệt** một decoder đã nộp ở vòng validation. Weights và code decoder phải chốt trước 01/03 | devkit README |
| Biến thể | Tối đa 3 biến thể chạy song song, phải khai báo từ giai đoạn validation | FAQ |
| Dữ liệu | Tập validation 2027 chính là tập test 2025: 30 ảnh (Unsplash, game, màn hình), 116 MB | /tasks |
| Mốc cần vượt (2025 @0.075) | Vcoder: 1929 Elo, PSNR 23.99, decoder 1.07GB, 24 s cho 30 ảnh. VTM: 1405 Elo, 26.5 dB | archive 2025 |
| Hạn chót (AoE) | Validation 01/03/2027. Tập test phát hành 02/03. Nộp test 09/03. Whitepaper 16/03. Paper PCS (không bắt buộc) 26/01 | /schedule/2 |

**Ba hệ quả thiết kế:**
1. **Encoder không bị giới hạn thời gian và có thể tiếp tục cải tiến sau 01/03.** Dồn "trí khôn" sang encoder: tìm seed, tối ưu latent, chọn mode và phân bổ rate. Decoder thì nhẹ và cố định.
2. Định dạng bitstream phải có sẵn chỗ cho các lựa chọn phía encoder: version, mode, seed, rate index, cờ bật/tắt. Nhờ vậy encoder có thể thêm mẹo mới sau khi decoder đã đóng băng.
3. Chỉ những phần *parse bitstream* mới bắt buộc phải tất định bit-exact, gồm entropy model và context model. Còn DiT và DC-AE thì chỉ cần kết quả gần đúng.

---

## A. Hạ tầng decoder và entropy coding tất định

### A0. Kiểm tra sớm độ tất định (tuần này, ~0.5 ngày, không đụng tới CVPR)

**Mục đích:** biết ngay sai lệch float giữa các thiết bị lớn đến đâu, để thiết kế hyperprior S1 của bản CVPR theo hướng thân thiện với số nguyên ngay từ đầu.

- [ ] Viết `experiments/determinism/probe.py`:
  - Dùng một hyperprior kiểu Ballé-2018 (khởi tạo ngẫu nhiên là đủ) trên latent DC-AE thật.
  - Tính `scales = h_s(ẑ)` rồi lượng tử hóa thành chỉ số trong bảng scale 64 mức (như CompressAI).
  - Lưu `ẑ` và `scale_idx` ra file.
- [ ] Chạy trên CPU (local), Kaggle T4, Kaggle P100, và Colab L4 nếu có. Dùng torch 2.6.0 cho giống server.
- [ ] Viết `compare.py` để đếm số phần tử có `scale_idx` khác nhau giữa từng cặp thiết bị.
- **Kết quả mong đợi:** nếu có bất kỳ phần tử nào khác nhau (gần như chắc chắn sẽ có trên 1 ảnh 2K), thì phương án entropy model bằng số nguyên (A2) là bắt buộc.
- **Kết quả lần 1 (29/9, Kaggle T4 + CPU của host; lưu ở `results/determinism/T4_v1/`):**

  | Phép so sánh | Kết quả |
  |---|---|
  | Chạy 2 lần trên cùng T4, cùng phiên bản torch | Mọi chế độ khớp 100% |
  | T4 và CPU, cùng phiên bản torch: `float32` | Lệch 76–80 trên 7.9 triệu chỉ số, tức khoảng 20 chỉ số mỗi ảnh 2K. Như vậy **mọi ảnh đều hỏng bitstream** |
  | T4 và CPU, cùng phiên bản torch: `bf16` | Lệch ~1.2 nghìn chỉ số |
  | T4 và CPU, cùng phiên bản torch: **`intsim`** | **Khớp 100%**, cả với torch 2.10 lẫn torch 2.6 |
  | So sánh giữa torch 2.10 và 2.6 | Không dùng được: do lỗi của probe (NEP 50 của numpy 2 làm weights lệch nhau), đã sửa. Cần chạy lại |

  → **Kết luận: A2 (mạng số nguyên) là bắt buộc. Cách tính "exact integer trong float64" đã được kiểm chứng giữa GPU và CPU.**
- **Kết quả lần 2 (29/9, đã sửa probe, vẫn là T4 + CPU; lưu ở `results/determinism/T4_v2/`):**
  - **`intsim` khớp 100% trên cả 10 cặp so sánh**, gồm cả GPU so với CPU và torch 2.10 so với 2.6.
  - Float khớp khi **cùng thiết bị, khác phiên bản torch** (T4 2.10 = T4 2.6, CPU = CPU). Nhưng **lệch khi khác thiết bị**: float32 lệch 79 chỉ số, bf16 lệch 1,114, trên 7.9 triệu.
  - Hệ quả: nếu encode trên L4 thuê, dùng float trong image `clic-gpu`, thì *có thể* khớp với server. Đây chỉ là phương án tạm, cần kiểm chứng trên hai máy L4 khác nhau.
  - Còn thiếu: P100 và L4.

### A1. Khung decoder và lần nộp đầu tiên (17/11 → 01/12)

- [ ] Tạo cấu trúc `clic/` trong repo chung. Từ 30/9 đã gộp repo: code codec nằm ở `src/ratflow/`, và được vendor vào zip decoder lúc đóng gói.
  ```
  clic/
    submission/          # nội dung zip decoder
      decode             # bash: python3 decode.py
      decode.py          # giải nén bs.zip, parse container, gọi codec, ghi PNG
      (ratflow/)         # copy src/ratflow lúc pack (DiT tối giản + DC-AE decoder + entropy), chỉ phụ thuộc torch
      weights/           # safetensors fp16/bf16 (không commit)
    encoder/             # chạy trên máy mình: encode, chọn rate, tìm seed
    tools/               # budget.py, pack.py, check_submission.py, local_eval.py
    l4/                  # đã có: setup_l4.sh, run_like_server.sh
  experiments/determinism/, experiments/clic_b/   # A0 và nhánh B (đã có)
  ```
- [ ] **Định dạng container** (một file duy nhất trong `bs.zip` để bớt overhead của zip):
  - Header toàn cục: magic, version, số ảnh, bảng tên ảnh đã nén bằng zlib.
  - Mỗi ảnh gồm các trường sau:

    | Trường | Kích thước |
    |---|---|
    | H, W | varint |
    | mode | 4 bit |
    | rate_idx | 6 bit |
    | seed_idx | 0–8 bit |
    | flags | – |
    | độ dài stream | – |
    | stream z | – |
    | stream y | – |
    | stream phụ | tùy chọn |
  - Nén `bs.zip` ở chế độ `ZIP_STORED`, vì dữ liệu đã được entropy code nên deflate không giúp gì.
- [ ] Viết `tools/budget.py`:
  - Tính ngân sách chính xác bằng `floor(bpp × Σ H·W / 8)` cho từng mức rate.
  - Đối chiếu với data size của baseline trên leaderboard (VTM @0.075 = 803,121 B) để xác định chính xác cách server tính: tính trên zip hay trên tổng các file.
- [ ] Viết `tools/check_submission.py` để mô phỏng server:
  - Chạy docker GPU của devkit, hoặc venv có torch 2.6.0 nếu máy không có Docker và GPU.
  - Giải nén, chạy `decode`, kiểm tra tên file, kích thước ảnh và thời gian.
  - Tính PSNR/MS-SSIM theo đúng cách server làm: gộp MSE toàn tập, có trọng số theo số pixel.
- [ ] **Lần nộp số 1, chỉ để kiểm tra đường ống:** gói lại baseline VTM của devkit bằng container của mình. Kiểm tra PSNR trên server khớp với số tính ở local.
- [ ] **Lần nộp số 2, kiểm tra tất định trên server thật:**
  - Dùng một model học sẵn nhỏ (ví dụ mbt2018 của CompressAI, vendor code) với entropy coding bằng float.
  - Nếu PSNR server khác PSNR local, hoặc decode bị lỗi, thì đã chứng minh được là cần A2.
  - Leaderboard chính là phép đo chuẩn để so sánh.

### A2. Entropy coding tất định (01/12 → 20/12)

**Nguyên tắc:** mọi phép tính quyết định CDF phải được thực hiện bằng số nguyên, chạy trên CPU.

- [ ] **Model trên z:** dùng factorized prior với bảng CDF đã lượng tử sẵn, lưu trong weights. Cách này tất định sẵn.
- [ ] **h_s và context model cho y chạy bằng số nguyên** (theo *Integer Networks for Data Compression*, Ballé et al., ICLR 2019):
  - Weights int8 hoặc int16, bias int32, tích lũy bằng int64.
  - Activation dùng ReLU và dịch bit, hoặc dùng LUT.
  - Đầu ra là trực tiếp `scale_idx` (và `mean` đã lượng tử nếu có).
  - Train theo kiểu QAT: fine-tune h_s sau khi codec S1 đã hội tụ, với fake-quant.
- [ ] **Context:** dùng checkerboard 2 pass, cộng channel-wise (ELIC rút gọn, 4–5 slice). Mức 0.3 bpp cần context mạnh. Tất cả nhánh context đều chạy bằng số nguyên.
- [ ] **Coder:** dùng `range_coder` 1.1 (đã có trong môi trường) với CDF 16 bit.
  - Viết test round-trip 10,000 ảnh crop.
  - Thêm fallback: nếu gói này có giới hạn, vendor một rANS C nhỏ qua `ctypes` và biên dịch sẵn cho amd64.
- [ ] **Nhiễu và seed của bridge:** tạo nhiễu bằng `torch.Generator('cpu').manual_seed(seed)` rồi mới `.to(cuda)`. Cách này tất định trên mọi GPU.
- [ ] **Ma trận kiểm thử:**
  - Encode trên thiết bị {CPU local, T4, P100, L4}, decode trên thiết bị còn lại.
  - Yêu cầu `ŷ` và `ẑ` giống nhau 100% trên 1,000 ảnh × 3 mức rate.
  - Gắn test này vào `tools/check_submission.py`.
- **Chấp nhận khi:** lần nộp số 3 lên server cho PSNR khớp local tới 3 chữ số thập phân.

### A3. Điều khiển rate theo ngân sách cả tập (15/12 → 10/01)

- [ ] **Mỗi ảnh có thể chọn rate liên tục:**
  - Tận dụng rate-as-time: điều kiện hóa theo `t_rate`.
  - Hoặc dùng gain vector hoặc bước lượng tử có thể co giãn (λ biến thiên).
  - Encoder tạo ra đường R(q) và Q̂(q) cho từng ảnh trên một lưới 16–32 điểm.
- [ ] **Phân bổ bit:**
  - Chạy Lagrangian hoặc tham lam theo độ dốc ΔQ̂/ΔR giữa các ảnh, với ràng buộc Σ bytes ≤ ngân sách − 256 B dự phòng.
  - Sau cùng, "đổ đầy" phần bit còn thừa vào ảnh có độ dốc lớn nhất.
- [ ] **Thước đo thay cho Elo (Q̂)**, ban đầu gồm:
  - `-DISTS`, `-LPIPS`, và một phạt độ trung thực (PSNR dưới ngưỡng).
  - Phạt thêm OCR-CER trên vùng có chữ (lấy từ nhánh B).
  - Hiệu chỉnh trọng số bằng dữ liệu chấm theo cặp nếu tìm được: kiểm tra xem CLIC/Mabyduck có công bố dữ liệu rating 2024/2025 không. Nếu không có, tự chấm một bộ nhỏ (A6).
  - **Đã tìm thấy (30/9)** ở https://archive.compression.cc/datasets/, gồm các file:
    - `clic2021_perceptual_{valid,test}_{crops,ratings}.zip`
    - `clic2022_perceptual_test_{crops,ratings}.zip`
    - `clic2024_perceptual_image_test_{crops,ratings}.zip` (6 MB ratings)

    Đây là dữ liệu chấm theo cặp của người thật trên các crop từ bài nộp CLIC. Dùng để fit trọng số của Q̂: logistic hoặc Bradley–Terry trên hiệu các thước đo (LPIPS, DISTS, PSNR, CER, …), rồi kiểm tra độ chính xác dự đoán lựa chọn của người chấm.
- **Chấp nhận khi:** trên tập validation, phân bổ tối ưu hơn phân bổ đều về Q̂ trung bình, và luôn nằm dưới ngân sách ở cả 3 mức.

### A4. Tốc độ và kích thước (05/01 → 31/01)

- **Chỉ tiêu:** giải mã 30 ảnh trong **≤ 20 s trên L4**, tính cả thời gian nạp model. Các đội GPU năm 2025 cần 21–29 s, và phải lọt khỏi nhóm 25% chậm nhất.
- ⚠️ **Số đo đầu tiên (30/9, parity trên Kaggle T4):**
  - Decoder DC-AE fp32 theo tile mất **6.9 s cho một ảnh 2048×1360**. Chỉ tiêu là khoảng 0.5 s mỗi ảnh cho toàn bộ pipeline, tức chậm hơn **gấp ~14 lần**.
  - **Decoder DC-AE là nút cổ chai, không phải DiT.**
  - Hướng xử lý: dùng fp16/bf16 (Tensor Core của L4 nhanh hơn fp32 nhiều lần); tile lớn hơn để bớt phần chồng lấn; thử dùng thẳng `torch.compile` trong submission. Nếu vẫn chậm, thì distill một decoder nhỏ hơn, hoặc cho lớp residual gánh phần chi tiết.
  - Đo lại trên L4 thật ngay khi thuê được máy.
- **Lần đo thứ hai (01/10, parity bản 3, T4):**
  - fp32: 6.74 s. fp16: **4.59 s**, tức chỉ nhanh hơn 1.5 lần.
  - fp16 có sai lệch tương đối 1.0% so với fp32 (lớn nhất 0.17 trên thang [-1, 1]), không có NaN/Inf.
  - Nhanh lên ít như vậy nghĩa là thời gian không nằm ở các phép conv. Nghi phạm là vòng lặp Python khi trộn các tile (64 phép gán từng hàng/cột cho mỗi đường nối) và phần attention phải ép lên fp32.
  - **Việc đầu tiên của A4 là profile `DCAE._tiled_decode`.** Hướng xử lý: trộn tile bằng phép tính vector hóa với mask, tile lớn hơn (hoặc không chia tile khi VRAM 24 GB đủ chỗ), rồi thử bf16 trên L4.
  - Chất lượng fp16 cần kiểm tra lại bằng PSNR/LPIPS trên ảnh thật, không dùng ảnh nhiễu ngẫu nhiên.

- [ ] **Ước lượng ngân sách thời gian** trên Colab L4 cho 30 ảnh ~2K:

  | Khâu | Thời gian |
  |---|---|
  | Nạp weights | ~3–6 s |
  | Entropy decode trên CPU (int) | ~0.1 s/ảnh |
  | DiT một bước | ~0.1–0.2 s/ảnh |
  | DC-AE decoder theo tile | ~0.2–0.4 s/ảnh |
- [ ] Nạp weights bằng `safetensors` và bf16. Không dùng `torch.compile` vì thời gian compile bị tính vào thời gian giải mã. Chỉ dùng CUDA graph nếu nó thực sự giúp.
- [ ] **DC-AE decode theo tile có chồng lấn** (overlap latent 2–4, trộn cosine), vì ảnh lớn nhất có thể tới ~4K. Kiểm tra không lộ đường nối tile.
- [ ] **Kích thước:**

  | Thành phần | Dung lượng |
  |---|---|
  | SANA-0.6B bf16 | ~1.2 GB |
  | DC-AE decoder | ~0.3 GB |
  | Codec | < 0.1 GB |
  | **Tổng** | **~1.6 GB** (an toàn) |

  SANA-1.6B bf16 thì khoảng 3.2 GB, cộng phần còn lại là sát giới hạn 4 GB. Chỉ dùng bản này nếu thắng rõ rệt, và khi đó cần lượng tử weight-only int8.
- [ ] Vendor code DiT/DC-AE ở dạng torch thuần, bỏ text encoder. Kiểm tra số liệu khớp với diffusers (sai lệch tối đa < 1e-3).
- **Chấp nhận khi:** lần nộp validation ở 0.075 bpp có thời gian giải mã ≤ 20 s và decoder ≤ 2 GB.

### A5. Mẹo phía encoder (tháng 2, tiếp tục được sau 01/03)

- [ ] **Tìm seed:** dùng K = 16–64 seed (tốn 4–6 bit mỗi ảnh). Encoder chạy decoder cho từng seed rồi chọn seed có Q̂ tốt nhất.
- [ ] **Tối ưu latent:** tinh chỉnh `y` bằng SGA hoặc gradient xuyên qua decoder, theo mục tiêu R + λ·(1 − Q̂). Encode chậm cũng không sao.
- [ ] **Chọn mode cho từng ảnh:** chọn giữa đường sinh ảnh (generative), đường có residual, và đường nội dung màn hình (dựa trên kết quả nhánh B).

### A6. Biến thể, tự chấm, và đóng băng (01/02 → 22/02)

- [ ] **Tối đa 3 biến thể:**
  - V1 nghiêng về độ trung thực.
  - V2 nghiêng về độ chân thực.
  - V3 lai: chọn mode theo nội dung.
  - Cả 3 phải được khai báo và nộp trong giai đoạn validation.
- [ ] **Tự chấm theo cặp:**
  - Dùng một trang HTML đơn giản, 2–3 người chấm, khoảng 300 cặp.
  - So các biến thể với nhau, và so với VTM và các codec perceptual mã nguồn mở (StableCodec/AEIC/MS-ILLM) ở cùng bpp.
- [ ] **Đóng băng decoder vào 22/02** (chừa 1 tuần dự phòng). Nộp bản cuối cho cả 3 mức × 3 biến thể. Lưu hash của từng bản.
- **02/03 → 09/03:** chỉ chạy encoder trên tập test (phân bổ rate, tìm seed, tối ưu latent). Kiểm tra bằng `check_submission.py` rồi nộp.
- **09/03 → 16/03:** viết whitepaper trên OpenReview, tái sử dụng nội dung từ bản CVPR.

---

## B. Đo mức hỏng trên ảnh game/màn hình (làm ngay, ~2 ngày)

Nhánh này trùng với rủi ro *"Trần chất lượng của DC-AE f32"* trong kế hoạch CVPR, nên làm một lần là dùng được cho cả hai.

**Phép tính sơ bộ về trần DC-AE:**
- Latent f32c32 có 32/1024 = 0.031 phần tử trên mỗi pixel.
- Ở 0.3 bpp, mỗi phần tử latent được ~9.6 bit, tức gần như không mất gì do lượng tử.
- Nghĩa là **ở 0.15–0.3 bpp, chất lượng bị giới hạn bởi trần tái tạo của DC-AE, không phải bởi rate.**
- Nếu trần này thấp hơn VTM ở 0.3 bpp, thì chỉ dùng DC-AE sẽ thua ở mức rate cao, nơi người chấm coi trọng độ trung thực.

### B1. Kiểm kê tập validation (0.5 ngày)

- [ ] Tải `clic2025_image_test.zip` (116 MB) từ trang CLIC vào `data/clic_valid/` (thư mục `data/` ở gốc repo, không commit).
- [ ] Viết bước `inventory` trong `experiments/clic_b/b_analysis.py` để tạo bảng CSV cho từng ảnh:
  - tên, H×W, số pixel
  - số màu khác nhau, tỉ lệ pixel thuộc cạnh sắc
  - có chữ hay không (kiểm tra bằng OCR)
  - nhãn **natural / game / screen**: tự động gợi ý, bạn xác nhận lại bằng mắt
- [ ] Ghi ngân sách byte cho 3 mức (sau này dùng chung với `clic/tools/budget.py`), và tỉ lệ ngân sách rơi vào nhóm game/screen nếu chia đều.

### B2. Đo trần và mức hỏng (1 ngày, GPU Kaggle)

Chạy các bước `recon` và `metrics` của `experiments/clic_b/b_analysis.py` trên từng ảnh với các cấu hình sau:

| Cấu hình | Ý nghĩa |
|---|---|
| **DC-AE f32c32 tự mã hóa rồi giải mã** (không lượng tử) | Trần tuyệt đối của đường ống SANA |
| SD-VAE f8 (của StableCodec) tự mã hóa rồi giải mã | So sánh với trần VAE 8× |
| DC-AE + lượng tử proxy (từ `experiments/quant_noise`) ở 0.075/0.15/0.3 bpp | Ước lượng thô của codec mình khi chưa train |
| VTM (devkit baseline, bitstream ở cùng mức bpp) | Mốc 1405 Elo |
| StableCodec / AEIC (có checkpoint trong `refs/repos`) ở mức gần 0.075 nhất | Đối thủ kiểu generative |

**Chỉ số đo:**
- Toàn ảnh: PSNR, MS-SSIM, LPIPS, DISTS.
- Vùng chữ (bounding box từ OCR trên ảnh gốc): **CER của OCR** trên ảnh tái tạo so với ảnh gốc, PSNR và SSIM của cạnh trong vùng chữ.
- Game: LPIPS trên các crop nhiều chi tiết (HUD, UI).

**Kết quả xuất ra:** `results/clic_b/b_results/metrics.csv`, và một lưới crop 256×256 (gốc | DC-AE | f8 | VTM | StableCodec) cho mỗi ảnh screen/game. Hình vẽ theo phong cách figures4papers.

### B3. Điểm quyết định (sau B2)

| Điều kiện (tính trên nhóm screen/game) | Hành động |
|---|---|
| CER trần của DC-AE ≤ CER của VTM@0.075 và PSNR trần ≥ VTM@0.3 − 1 dB | Không cần nhánh riêng. Chỉ cần phân bổ nhiều bit hơn cho ảnh có chữ (A3) |
| CER trần của DC-AE cao hơn rõ, nhưng trần f8 thì ổn | Thêm **nhánh residual ở pixel hoặc latent f8** cho vùng chữ/ảnh screen (giống `res` của StableCodec). Nhánh này có lợi cho cả CVPR ở mức rate cao |
| Cả DC-AE và f8 đều hỏng chữ | Thêm **mode "classic SCC"** cho ảnh screen: vendor decoder VVC có sẵn trong devkit (`VVCDecoder_23.8`), encoder dùng VTM với công cụ SCC (IBC, palette), rồi dùng mạng hậu xử lý nhẹ để làm đẹp phần không phải chữ |
| PSNR trần DC-AE < VTM@0.3 − 2 dB trên **ảnh tự nhiên** | Ở 0.3 bpp (có thể cả 0.15) cần đường residual cho mọi ảnh. Đây là tín hiệu quan trọng cho cả CVPR |

### B4. Cách chia việc

| Việc | Ai làm |
|---|---|
| Viết mọi script (`inventory.py`, `ceiling.py`, `text_metrics.py`, `make_crops.py`) và các notebook chạy trên Kaggle | Claude |
| Tải dữ liệu, chạy trên Kaggle, xác nhận nhãn nội dung, xem lưới crop | Bạn |
| Công cụ: `easyocr` hoặc `tesseract` cho OCR, `piq` hoặc `pyiqa` cho LPIPS/DISTS | – |

### B5. Kết quả lần 1 (29/9, Kaggle T4; lưu ở `results/clic_b/b_results/`)

**Dữ liệu:**
- 30 ảnh, tổng 86,245,376 pixel.
- Ngân sách: 808,550 / 1,617,100 / 3,234,201 byte. Năm 2025 Vcoder nộp 808,827 byte, nên có thể server cho phép dư một chút, hoặc tính khác. Cần xác nhận khi nộp.
- Nhãn đã sửa bằng mắt:
  - **screen (4 ảnh):** `ebfd571f` (Wikipedia), `bb7344a2`, `86127fbd` (UI), `2a760bf1`
  - **texture (1 ảnh):** `937476dd` (nhiễu 256 màu, mọi phương pháp đều hỏng)
  - **natural (25 ảnh):** phần còn lại
  - Chưa thấy ảnh game rõ ràng. Nhãn tự động sai 5 ảnh.

**PSNR toàn tập theo cách CLIC:**

| Cấu hình | PSNR (dB) |
|---|---|
| **Trần DC-AE f32** | **22.86** |
| Trần SD-VAE f8 | 24.73 |
| x265@0.075 | 24.58 |
| x265@0.15 | 25.98 |
| x265@0.3 | 28.23 |

- Mốc so sánh @0.075: HM 25.83, VTM 26.52, **Vcoder (hạng 1) 23.99**.
- Proxy x265 kém HM khoảng 1.25 dB và kém VTM khoảng 1.9 dB. Ở 0.075, x265 còn vượt ngân sách trên 10 ảnh dù đã dùng QP 51, nên đây là mốc yếu.

**Kết luận:**
1. **Trần DC-AE f32 (22.86 dB) thấp hơn PSNR của đội thắng @0.075 (23.99 dB), dù chưa hề lượng tử.** Ước tính VTM@0.3 ≈ 30.1 dB, tức DC-AE thua khoảng 7 dB. Chỉ dùng DC-AE thì không thể cạnh tranh ở 0.15/0.3, và đã sát giới hạn ở 0.075.
2. **Chữ bị biến thành "chữ ảo"** (ký tự bịa ra) ngay ở mức trần:
   - Nhóm screen: DC-AE có text PSNR 16.7 dB và CER 0.58, trong khi x265@0.075 có 24.1 dB và CER 0.42, x265@0.15 có CER 0.19.
   - SD-VAE f8 cũng làm hỏng chữ nhỏ (text PSNR 20.4 dB). **Chuyển sang f8 thôi là không đủ.**
3. **Về cảm nhận, các AE thắng lớn ở mức rate thấp:** LPIPS 0.08 (DC-AE) so với 0.37 (x265@0.075) trên ảnh tự nhiên. Nhưng khuôn mặt bị trôi danh tính (xem crop `da52c4f6`). Người chấm có ảnh gốc để so, nên đây là rủi ro.
4. Ảnh screen nén bằng codec cổ điển rất rẻ (`50d13125`, `2a760bf1` đạt 37–42 dB ở 0.075). Có thể **chuyển bớt bit từ ảnh screen sang ảnh tự nhiên**, vì ngân sách tính trên cả tập.

**Quyết định (B3):**
- **Bắt buộc có lớp tăng cường ở miền pixel:** base là codec flow-bridge trên DC-AE (cho độ chân thực), cộng thêm **một residual codec học được, có điều kiện theo ảnh base** (kiểu ELIC nhẹ, dùng LPIPS và GAN nhẹ), dùng phần bit còn lại.
  - Encoder chia bit giữa base và residual cho từng ảnh (mở rộng A3).
  - Ở 0.075 phần lớn bit dành cho base; ở 0.3 phần lớn dành cho residual.
- **Mode cổ điển (VVC) cho ảnh screen:** devkit đã có sẵn decoder VVC, nên chi phí kỹ thuật thấp.
  - Cần đo VTM có bật công cụ SCC trên 4 ảnh screen: lần B6.
- **Ảnh hưởng tới CVPR:** rủi ro "trần DC-AE f32" trong kế hoạch CVPR **đã thành hiện thực** (thấp hơn f8 1.9 dB trên tập này, chữ bị hỏng). Cách giảm thiểu "nhánh residual giống `res` của StableCodec" nên được kích hoạt.

### B6. Việc tiếp theo cho nhánh B

- [x] Build VTM có SCC trên Kaggle (CPU). Encode 4 ảnh screen và 3 ảnh natural đại diện ở 0.075/0.15/0.3, tìm QP theo ngân sách. Cách này thay proxy x265 bằng mốc thật và đánh giá được mode SCC.
- [x] Đo trần khi thêm residual: DC-AE + residual lý tưởng (ví dụ x265 hoặc VTM trên phần dư `x − x̂_DCAE` với ngân sách còn lại). Mục đích là ước lượng xem lớp tăng cường lấy lại được bao nhiêu dB và bao nhiêu CER.

### B7. Kết quả B6 (30/9, Kaggle; lưu ở `results/clic_b/b6_results/`)

**Thiết lập:** VTM 23.8. `dcae+res` = trần DC-AE, cộng phần dư `x − x̂` được mã hóa bằng VTM 4:4:4, giả định base tốn 0.03 bpp.

**Hai thiên lệch ngược chiều nhau:**
- Base được giả định bằng *trần* DC-AE, nên **lạc quan**: một base thật ở 0.03 bpp sẽ kém hơn.
- Phần dư được mã hóa bằng VTM, vốn không hợp với tín hiệu kiểu nhiễu, nên **bi quan**: một residual học được và có điều kiện sẽ tốt hơn.

**Nhóm screen (4 ảnh):**

| Mức bpp | Cấu hình | PSNR | LPIPS | Text PSNR | CER |
|---|---|---|---|---|---|
| 0.075 | vtm420 | 32.9 | 0.126 | 27.9 | 0.26 |
| 0.075 | **vtmscc** | **34.0** | 0.116 | **28.6** | **0.18** |
| 0.075 | dcae+res | 31.0 | **0.045** | 24.3 | 0.32 |
| 0.15 | vtm420 | 36.3 | – | – | 0.12 |
| 0.15 | **vtmscc** | **38.4** | – | – | **0.09** |
| 0.15 | dcae+res | 34.3 | – | – | 0.16 |
| 0.3 | vtm420 | 39.8 | – | – | 0.06 |
| 0.3 | **vtmscc** | **43.2** | – | – | 0.06 |
| 0.3 | dcae+res | 38.0 | – | – | 0.09 |

**Nhóm natural (3 ảnh):**

| Mức bpp | Cấu hình | PSNR | LPIPS |
|---|---|---|---|
| 0.075 | vtm420 | 24.82 | 0.401 |
| 0.075 | dcae+res | 24.61 | **0.105** |
| 0.15 | vtm420 | 27.44 | 0.264 |
| 0.15 | dcae+res | 26.07 | **0.092** |
| 0.3 | vtm420 | 29.97 | 0.170 |
| 0.3 | dcae+res | 28.29 | **0.068** |

**Kết luận:**
1. **Ảnh thuần màn hình → dùng mode VVC-SCC** (4:4:4, IBC, palette, BDPCM, transform skip).
   - SCC hơn 4:2:0 lần lượt +1.1 / +2.1 / +3.3 dB ở 3 mức bpp, và có CER thấp nhất.
   - Đường lai DC-AE vẫn làm vỡ chữ ở 0.075 (crop `ebfd571f`); phải tới 0.15 bpp chữ mới đọc được.
   - Decoder VVC 23.8 có sẵn trong devkit giải mã được luồng này, vì bitstream được tạo bằng cùng phiên bản VTM 23.8.
2. **Ảnh tự nhiên → dùng base generative cộng residual.**
   - Dù residual được mã hóa theo cách bi quan, ở 0.075 PSNR vẫn **ngang VTM** (−0.2 dB) mà LPIPS tốt hơn 4 lần.
   - Ở 0.15 và 0.3, PSNR thấp hơn VTM 1.4–1.7 dB nhưng LPIPS tốt hơn 2.5–3 lần.
   - Một residual học được và có điều kiện phải thu hẹp được khoảng cách PSNR này. Đó là mục tiêu thiết kế.
3. **Ảnh hỗn hợp** (UI đè lên cảnh render 3D, như `86127fbd`): SCC gần như không giúp gì, còn đường lai tốt hơn về cảm nhận. Nghĩa là **encoder phải chọn mode cho từng ảnh** (A5): thử cả hai đường rồi chọn bằng Q̂, tốn 1–2 bit cờ mode.
4. **Chia bit giữa các ảnh:** ảnh screen đạt chất lượng rất cao với ít bit (ví dụ `2a760bf1` ở 0.075 cần QP 26, đạt ~42 dB). Có thể lấy bớt bit từ nhóm này cho ảnh tự nhiên (A3).
5. **Điều chưa biết lớn nhất:** chất lượng thật của base ở khoảng 0.03 bpp. Câu trả lời sẽ đến từ S1/S2 của bản CVPR. Khi có checkpoint, chạy lại B6 với base thật thay cho trần DC-AE.

**Giới hạn:** chỉ có 7 ảnh, và CER từ OCR khá nhiễu với chữ nhỏ.

---

## C. Lịch tổng (đan xen với CVPR)

| Thời gian | Việc | Khối lượng |
|---|---|---|
| 30/9 – 06/10 | **B1, B2, B3** và **A0** (kiểm tra tất định) | ~2.5 ngày, chủ yếu chạy nền |
| 07/10 – 16/11 | **Chỉ làm CVPR.** Hai ràng buộc thiết kế mang từ CLIC sang: hyperprior S1 thân thiện số nguyên, và rate được điều kiện hóa liên tục. Nếu B3 kết luận cần residual thì đưa vào S1 | – |
| 17/11 – 01/12 | A1: khung decoder, lần nộp số 1 và số 2 | ~1 tuần |
| 01/12 – 20/12 | A2: entropy coding số nguyên, lần nộp số 3 (khớp PSNR) | ~2.5 tuần |
| 15/12 – 10/01 | A3: điều khiển rate, codec của mình ở 0.075 bpp | ~3 tuần |
| 05/01 – 31/01 | A4: tốc độ và kích thước, mở rộng lên 0.15/0.3, nhánh riêng theo B3 | ~4 tuần |
| 26/01 | Paper PCS (không bắt buộc) | – |
| 01/02 – 22/02 | A5, A6: mẹo phía encoder, 3 biến thể, tự chấm, **đóng băng decoder 22/02** | ~3 tuần |
| 01/03 | Hạn validation | – |
| 02/03 – 09/03 | Chỉ chạy encoder trên tập test, rồi nộp | 1 tuần |
| 16/03 | Whitepaper | – |

## D. Rủi ro chính

| Rủi ro | Cách giảm thiểu |
|---|---|
| Server decode sai vì float không tất định | A0 sớm, A2 dùng số nguyên, lần nộp số 2–3 kiểm tra trên server thật |
| Bị lọt vào nhóm 25% chậm nhất | Chỉ tiêu ≤ 20 s, không dùng torch.compile, đo trên L4 từ tháng 1 |
| Hết TPU/GPU vào đợt nước rút (TRC chỉ 30 ngày) | Việc train cho CLIC chủ yếu là fine-tune (QAT cho h_s, nhánh residual), đủ chạy trên Kaggle. Cân nhắc lùi thời điểm kích hoạt TRC nếu CVPR chưa cần tới |
| 0.3 bpp thua VTM về độ trung thực | Điểm quyết định B3, đường residual |
| Thước đo Q̂ lệch so với Elo của người chấm | Tự chấm theo cặp (A6), 3 biến thể để rải rủi ro |
| Môi trường chạy lệch so với server | `check_submission.py` chạy trên docker của devkit hoặc venv torch 2.6.0, và coi PSNR trên leaderboard là phép đo chuẩn |

## E. Máy L4 thuê (chốt 29/9: bạn tự thuê)

**Nên thuê máy ảo đầy đủ có Docker**, ví dụ GCP `g2-standard-8` hoặc AWS `g6.2xlarge`. Khi đó build được đúng `devkit/docker/Dockerfile.gpu` (image `clic-gpu`), và chạy giả lập được đúng giới hạn của server: `--cpus=2 --memory=12g --network none`.
- RunPod hoặc Vast cũng dùng được, nhưng chỉ ở dạng venv (`setup_l4.sh --venv`), tức là gần giống chứ không trùng hoàn toàn với server.
- Thuê máy spot hoặc interruptible là đủ, vì không có job nào chạy quá vài giờ.

| Việc | Khi nào | Số giờ L4 ước tính |
|---|---|---|
| A0: `probe.py` trong `clic-gpu` (thêm một lần trên máy thuê thứ hai để kiểm tra khác driver) | Tuần này | ~1 giờ |
| A1: dựng khung, `run_like_server.sh` cho lần nộp số 1–2 | Cuối tháng 11 | ~3 giờ |
| A2: ma trận tất định (L4 ↔ CPU ↔ T4/P100) | Tháng 12 | ~3 giờ |
| A4: đo tốc độ, tối ưu tile, bf16 | Tháng 1 | ~10–15 giờ |
| A5: tìm seed và tối ưu latent phía encoder trên tập test (30 ảnh × 3 mức × K seed) | 02–09/03 | ~10–20 giờ |

**Lợi ích phụ:** nếu A0 cho thấy float **khớp** giữa hai máy L4 khác nhau, thì có thể encode trên L4 trong `clic-gpu` làm phương án tạm cho các lần nộp sớm, trong lúc chờ A2 (entropy bằng số nguyên). A2 vẫn là giải pháp cuối cùng.

Script:
- `clic/l4/setup_l4.sh`: cài Docker, NVIDIA toolkit, build `clic-gpu`.
- `clic/l4/run_like_server.sh`: chạy decoder giống server và đo thời gian.
- `experiments/determinism/probe.py` và `compare.py`: kiểm tra A0.

Lưu ý: torch trên máy Windows bị Application Control chặn, nên mọi thứ chạy trên L4 hoặc Kaggle.

## F. Câu hỏi còn mở

1. Đăng ký team và tài khoản CLIC ngay, để có quyền nộp bài và xem giới hạn số lần nộp mỗi ngày.

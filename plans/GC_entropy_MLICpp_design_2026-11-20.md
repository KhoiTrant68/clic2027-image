# Thiết kế: entropy model kiểu MLIC++ cho GC, giữ khớp bit (20/11/2026)

**Mục tiêu:** đưa vào entropy model của GC những thành phần đã làm MLIC++ tiết kiệm 10,9% bit (decoder MSE) và 16,9% bit (decoder GAN) so với ELIC trên CLIC30, theo ablation ngày 20/11 (`reports/CLIC2027_GC_backbone_ablation_2026-11-20.md`). Phải giữ hai ràng buộc của GC:

1. **Khớp bit:** mọi thứ quyết định CDF đều tính bằng số nguyên, để máy encode và L4 suy ra cùng bảng. MLIC++ Mixed từng giải mã sai ảnh trên CPU so với GPU.
2. **Thời gian giải mã:** ≤ 20 s cho 30 ảnh trên L4, gồm cả synthesis. MLIC++ Mixed mất 100 s/30 ảnh trên L40, nên không thể bê nguyên MLIC++ sang.

**Tóm tắt đề xuất:**
- Giữ khung GC hiện tại: hyperprior, slice kênh không đều, checkerboard hai pass, mạng số nguyên `QConv2d`.
- Thêm bốn thành phần của MLIC++, mỗi cái bật/tắt được:
  - E1: context cục bộ 5×5 trên anchor;
  - E2: mạng tham số sâu hơn;
  - E3: context toàn cục giữa các slice;
  - E4: context toàn cục trong slice.
- E3 và E4 dùng **attention tuyến tính số nguyên** (mục 3.3), chính xác tuyệt đối và tất định.
- Thêm **LRP** (dự đoán phần dư lượng tử). LRP chỉ đi vào g_s và không quyết định CDF, nên chạy float bình thường.
- Trước khi sửa GC, chạy **pha 0** (~1 ngày code + ~5 giờ GPU): đo xem lợi thế đến từ entropy model hay từ transform, và từng thành phần đóng góp bao nhiêu.

## 1. Hiện trạng hai bên

| | GC hiện tại (`src/clic27/gc/model.py`) | MLIC++ (`refs/MLIC/MLIC++`, Apache-2.0) |
|---|---|---|
| Latent | M = 320, 1/16 | M = 320, 1/16 |
| Slice kênh | 5 slice không đều [16, 16, 32, 64, 192] | 10 slice đều × 32 |
| Spatial trong slice | checkerboard; context anchor qua 2 lớp depthwise 3×3 | checkerboard; **attention cửa sổ 5×5** trên anchor (softmax, LayerNorm) |
| Channel context | concat các slice trước vào cùng `ctx_net` | `ChannelContext`: 3 conv 3×3 (192→128→4C) |
| Toàn cục giữa slice | không có | `LinearGlobalInterContext`: attention tuyến tính softmax trên toàn ảnh, từ các slice trước |
| Toàn cục trong slice | không có | `LinearGlobalIntraContext`: bản đồ attention học từ slice trước (query = non-anchor, key = anchor), áp lên anchor của slice hiện tại |
| Mạng tham số | `ctx_net`: 1×1 → dw3×3 → dw3×3 → 1×1 → 1×1, hidden 160 | `EntropyParameters`: 4 lớp 1×1 (320 → 256 → 128 → 2C), GELU |
| LRP | không có | có, cho cả anchor và non-anchor |
| Số học | số nguyên mang trong float64, khớp bit CPU/GPU | float32, không khớp bit |
| Rate | 8 level gain + quality map k + β | mỗi λ một model |

**Ghi chú khi quy lợi thế:** transform của MLIC++ (`AnalysisTransform`: residual block kiểu Cheng 2020) đơn giản hơn của ELIC (có attention block). Vì vậy nhiều khả năng phần lớn lợi thế nằm ở entropy model. Đây mới là suy đoán, pha 0 sẽ đo trực tiếp.

## 2. Pha 0: đo trên latent cố định của MLIC++ (trước khi sửa GC)

Ý tưởng: giữ g_a của MLIC++ cố định. Khi đó y không đổi, nên mọi khác biệt về bpp chỉ do entropy model, không lẫn transform hay GAN.

- **Dữ liệu:**
  - dùng lại pool latent của `experiments/gc_backbone/train_gs.py` (32.768 crop DIV2K, seed 12345);
  - tính y bằng g_a của MLIC++ ở q1 (≈ 0,12 bpp) và q2 (≈ 0,18 bpp);
  - đánh giá bằng y của CLIC30 ở cùng q.
- **Train:** chỉ train h_a, h_s và entropy model, loss chỉ là rate (−log₂ p của y dưới nhiễu đều; với z cũng vậy). Batch 16, latent 16×16, khoảng 100k bước. Tốc độ dự kiến nhanh hơn nhiều so với train GC đầy đủ, vì không có g_a, g_s hay discriminator.
- **Đo:**
  - bpp ước lượng và bpp mã hóa thật (rANS của GC) trên CLIC30;
  - PSNR khi đưa ŷ qua g_s MSE của MLIC++. ŷ = round(y − μ) + μ phụ thuộc μ, nên distortion có thể lệch nhẹ;
  - thời gian entropy decode trên H100 (chỉ dùng để so tương đối giữa các biến thể).
- **Các biến thể, train trên cùng dữ liệu và cùng số bước:**

| Mã | Entropy model | Mục đích |
|---|---|---|
| R | MLIC++ pretrained, float, không train lại | Mức trần tham chiếu |
| V0 | GC hiện tại, số nguyên | Điểm xuất phát |
| V0f | V0 nhưng float | Chi phí của việc số nguyên hóa |
| V1 | V0 + E1 (5×5 cục bộ) + E2 (mạng sâu hơn) | Thành phần rẻ |
| V2 | V1 + E3 (toàn cục giữa slice) | |
| V3 | V2 + E4 (toàn cục trong slice) | Đủ bộ |
| V3s | V3 với 10 slice × 32 | Slice đều có đáng thêm 10 pass tuần tự không |

- **Cổng G0:**
  - Nếu khoảng cách R − V0 < 3% bpp, entropy không phải nguồn lợi thế. Dừng thiết kế này và chuyển sang thử g_a kiểu MLIC++. g_a chỉ chạy ở encoder nên không tốn thời gian giải mã.
  - Ngược lại, chọn biến thể rẻ nhất lấy lại được ≥ 70% khoảng cách R − V0, với thời gian entropy decode ≤ 2× V0.
- **Chi phí:**
  - code: script `experiments/gc_entropy/rate_only.py` dùng lại `GCodec._slices` và pool latent, khoảng 1 ngày;
  - GPU 6: 7 biến thể × ~40 phút ≈ 5 giờ, giữ số worker thấp (pool nằm sẵn trên GPU, không cần worker nạp dữ liệu).

### 2.1 Sửa đổi G0 (20/11)

**Nguồn gốc:**
- G0 ở mục 2 là tiêu chí đăng ký trước.
- Sửa đổi này do assistant đề xuất **sau khi đã thấy số giữa chừng** của V0: bước 40k, cao hơn R 39–46%.
- Người dùng duyệt trong chat ("Duyệt sửa đổi") khi các biến thể V0f–V3s chưa chạy xong; chưa có kết quả 50k nào của các biến thể mới.
- Kết quả theo G0 nguyên bản ở 50k bước vẫn được báo cáo song song.

**Lý do:** khi chạy pha 0, các biến thể được train 50k bước (batch 32, lr 1e-4; tăng batch từ 16 lên 32 để giữ số mẫu, vì tốc độ bị giới hạn bởi số lần gọi kernel). Kết quả giữa chừng: V0 cao hơn R 39–46%, và đường cong vẫn đang giảm. Khoảng cách R − V0 vì vậy lẫn cả phần do train chưa đủ, chứ không chỉ do kiến trúc.

**Sửa đổi:**
- Kết quả 50k bước chỉ dùng để **chọn** biến thể: biến thể có bpp CLIC30 thấp nhất trong V1, V2, V3, V3s, chọn riêng cho từng q.
- V0 và biến thể được chọn được **train tiếp tới 300k bước** từ checkpoint 50k: Adam mới, lr 3e-4, giảm ×0,1 ở 10% cuối.
- **G0 áp dụng nguyên văn** (mốc 3% và 70%, forward ≤ 2× V0) lên kết quả 300k bước của hai mô hình này.
- Biến thể được chọn cũng là biến thể duy nhất được xét ở nhánh 70%; nếu nó không đạt mốc forward thì báo cáo như vậy, không đổi biến thể.
- Số 50k bước vẫn được báo cáo đầy đủ, kèm ghi chú về hạn chế ngân sách train.

## 3. Thiết kế các thành phần

Mọi module dưới đây có hai đường, giống `QConv2d`:
- `forward`: float, lượng tử hóa với STE, dùng khi train;
- `forward_int`: số nguyên mang trong float64, dùng khi encode/decode.

Mọi `QConv2d` nhận `n_cond = L`, tức bias riêng cho từng level rate như hiện nay. Quality map k tiếp tục được nối vào làm một kênh đầu vào.

### 3.1 E1: context cục bộ 5×5 trên anchor

Thay 2 lớp depthwise 3×3 trong `ctx_n` bằng `QConv2d(c → 2c, 5×5)` đầy đủ trên anchor đã che (`anchors * ma`). Sau đó qua 1×1 rồi vào mạng tham số.

- Không dùng attention cửa sổ như MLIC++: softmax và LayerNorm khó số nguyên hóa, mà lợi thế so với conv 5×5 chưa rõ.
- Tầm nhìn 5×5 trên lưới checkerboard tương đương cửa sổ 5×5 của MLIC++.
- **Biên chính xác:** fan-in = c·25. Với c = 192 thì fan-in = 4800, accumulator 2¹⁶·127·4800 < 2⁴⁹. Kiểm thêm điều kiện `s ≥ 1` của `_conv_exact_f32`: khi `w_max·fan_in` quá lớn thì tách conv theo nhóm kênh. Thêm assert vào `check_ranges`.

### 3.2 E2: mạng tham số sâu hơn

Tách `ctx_net` thành hai phần: phần trích đặc trưng (conv không gian) và phần **tham số** gồm 4 lớp 1×1 số nguyên. Kích thước theo MLIC++, thu nhỏ theo cổng tốc độ:
- slice lớn: (vào → 256 → 192 → 128 → 2c);
- slice nhỏ: hidden 160 như hiện nay.

### 3.3 Attention tuyến tính số nguyên (dùng cho E3 và E4)

MLIC++ chuẩn hóa key bằng softmax theo không gian và query bằng softmax theo kênh. Ở đây thay bằng attention tuyến tính với hàm đặc trưng ReLU (Katharopoulos 2020): chuẩn hóa theo từng query, cùng bản chất là trung bình có trọng số nhưng chỉ dùng phép cộng, nhân và một phép chia số nguyên.

```
q, k = ReLU-QConv (1×1 → dw3×3), lượng tử về b_q = b_k = 8 bit không âm
v    = QConv có dấu, kẹp về [−2^(b_v−1), 2^(b_v−1) − 1], b_v = 10      # cần act="clamp" mới cho QConv2d
Mỗi head h, kênh i, j:
  KV[i, j] = Σ_n k[n, i] · v[n, j]          # tổng trên toàn ảnh (hoặc chỉ anchor)
  Ks[i]    = Σ_n k[n, i]
  num[n, j] = Σ_i q[n, i] · KV[i, j]
  den[n]    = Σ_i q[n, i] · Ks[i] + 1
  out[n, j] = floor(num[n, j] / den[n])      # cùng số bit phân số với v
```

**Vì sao chính xác và tất định:**
- **Mọi tích và tổng đều là số nguyên.** Với N ≤ 2¹⁴ vị trí latent (ảnh CLIC tới 2048×2048 cho lưới 128×128) và d ≤ 64 kênh mỗi head:
  - |KV| ≤ N·2⁸·2⁹ = 2³¹;
  - |num| ≤ d·2⁸·2³¹ ≤ 2⁴⁵ < 2⁵³.

  Do đó matmul float64 cho kết quả đúng bất kể thứ tự cộng hay kernel (CPU, cuBLAS).
- **Phép chia cho kết quả floor đúng.** Phép chia float64 theo IEEE làm tròn đúng. Với |num| < 2⁵³ và den ≥ 1 nguyên, nếu thương thật không nguyên thì nó cách số nguyên gần nhất ít nhất 1/den, lớn hơn sai số làm tròn |num/den|·2⁻⁵³. Vì vậy `floor(num/den)` tính bằng float64 bằng đúng floor của thương thật.
  - Rủi ro duy nhất là một backend dùng chia nhanh, không làm tròn đúng.
  - Đã kiểm nhanh trên lab09 (torch của env `gc`, CPU và H100): khoảng 20 triệu cặp, gồm |num| tới 2⁵³ và cặp hiểm `num = k·den − 1` (thương nằm sát dưới một số nguyên). Không lệch cặp nào so với chia số nguyên. Vẫn đưa test này vào bộ test và chạy lại trên L4.
  - Phương án dự phòng: chuẩn hóa bằng lũy thừa 2, `den → 2^ceil(log2 den)`; khi đó chia chỉ là dịch bit.
- **Đường f32-split:** KV có thể vượt 2²⁴, nên E3/E4 chạy float64. Kích thước nhỏ (N·d²·2 ≈ 1,3·10⁸ FLOP mỗi head cho ảnh 2K). FP64 trên L4 chậm (~1/64 FP32), nhưng ước tính vẫn chỉ vài ms mỗi slice. Đây mới là ước tính, phải đo trên L4.

**E3, toàn cục giữa slice** (cho anchor và non-anchor của slice s ≥ 1):
- q, k, v tính từ ŷ_s của tất cả slice trước (đã giải mã hết, nên không có vấn đề thứ tự);
- 2 head, d = 32;
- đầu ra qua `QConv 5×5` rồi nối vào mạng tham số.

**E4, toàn cục trong slice** (chỉ cho non-anchor của slice s ≥ 1):
- query lấy từ non-anchor của slice s − 1, key từ anchor của slice s − 1, value từ anchor của slice s;
- tổng chỉ lấy trên vị trí anchor. Lúc giải mã non-anchor, mọi anchor của slice s đã có, nên decoder tính được.

Context của mọi module được tính trên `y_hat_s` **trước LRP**, đã kẹp `Y_CLAMP` như hiện nay. Nhờ vậy phần quyết định CDF không phụ thuộc LRP chạy float.

### 3.4 LRP (float, chỉ đi vào g_s)

```
ŷ_final = ŷ + 0.5 · tanh(LRP_s([hyper, ŷ_<s, ŷ_s]))
```

- LRP_s gồm 3 conv nhỏ, float/fp16, chạy trên GPU cùng g_s.
- LRP không ảnh hưởng CDF, nên lệch fp giữa các máy chỉ làm pixel khác rất ít, không phá stream. Điều này giống g_s hiện nay.
- Chi phí: 5 mạng nhỏ, đặt ở đầu `synthesize`.
- Lợi ích với decoder GAN chưa rõ; đo ở pha 2.

### 3.5 Slice

Giữ 5–6 slice không đều. Mỗi slice thêm 2 pass tuần tự (rANS + mạng số nguyên), nên 10 slice đều chỉ dùng nếu V3s thắng rõ ở pha 0. Ứng viên trung gian: [16, 16, 32, 32, 64, 160].

## 4. Thay đổi trong mã nguồn (pha 1)

- **`src/clic27/entropy/intnet.py`:**
  - `QConv2d(act="clamp")` cho đầu ra có dấu;
  - module `IntLinearAttention` (forward STE + `forward_int` float64);
  - `check_exact_range` mở rộng cho attention.
- **`src/clic27/gc/model.py`:**
  - `cfg["entropy"] = {"local5": bool, "deep_params": bool, "global_inter": bool, "global_intra": bool, "lrp": bool}`. Mặc định tắt hết, khi đó model giữ nguyên hành vi hiện tại và checkpoint cũ vẫn nạp được;
  - `_slices` gọi các module theo cờ.
- **`src/clic27/gc/codec.py`:**
  - `_entropy_pass` dùng cùng các module, nên encoder và decoder vẫn chạy chung một đoạn mã;
  - stream `VERSION = 2` khi có bất kỳ cờ nào bật. Decoder đọc version và từ chối nếu không khớp với model.
- **`src/clic27/gc/package.py`, `decode_main.py`:** đóng gói cờ `entropy` trong `model.pt`.
- **`tests/test_gc.py`, thêm test:**
  - (a) `IntLinearAttention.forward_int` CPU = CUDA = float64 tham chiếu, kể cả với giá trị gần biên;
  - (b) phép chia floor khớp chia số nguyên Python trên 10⁶ cặp ngẫu nhiên;
  - (c) roundtrip compress/decompress với mọi cờ bật, mã hóa trên CUDA rồi giải mã trên CPU;
  - (d) bpp ước lượng gần bpp thật;
  - (e) `check_ranges` từ chối cấu hình vượt 2⁵³.

## 5. Pha 2: so sánh trong GC đầy đủ, rồi train

1. **Stage rd ngắn, hai cấu hình:** cấu hình cũ và cấu hình chọn từ pha 0, cùng seed và dữ liệu, mỗi cái 100k bước trên GPU 6 (~2 giờ ở 14,7 it/s; bản mới có thể chậm hơn). Đo BD-rate trên CLIC30 bằng stream thật, theo PSNR và LPIPS.
   - **Cổng G1:** BD-rate ≤ −5%.
2. **Đo LRP:** thêm/bớt LRP trên cấu hình chọn, stage gan 50k bước, so Q̂ và LPIPS.
3. **Cổng G2 (tốc độ và khớp bit):**
   - giải mã 30 ảnh CLIC trên L4 bằng `scripts/gc/l4_verify.sh` và `python -m clic27.gc.bench`;
   - phần entropy ≤ 0,25 s/ảnh, tổng ≤ 20 s/30 ảnh;
   - 30/30 hash pixel khớp với stream encode trên lab09.
   - Trước khi có L4 chỉ ghi số đo H100 và đánh dấu là tạm.
4. **Train đầy đủ** theo `configs/gc_full.json`, thêm khối `entropy`. Hiện ước tính khoảng 23 giờ trên GPU 6 cho bản cũ; bản mới phải đo lại tốc độ ở bước 1.

## 6. Rủi ro

- **Số nguyên hóa làm mất lợi thế.** STE trên floor và kẹp 8–10 bit có thể làm attention kém hơn bản float. Pha 0 đo trực tiếp qua V0f và V3.
- **Lợi thế nằm ở transform.** Khi đó G0 chuyển sang thay g_a, vốn chỉ tốn thời gian encode.
- **FP64 trên L4.** Nếu E3/E4 chậm hơn ước tính, chuyển phép nhân sang f32-split theo khối (như `_conv_exact_f32`), hoặc giảm d và số head.
- **Ảnh rất lớn.** N > 2¹⁴ làm biên 2⁵³ sát hơn. Cần assert theo kích thước ảnh, và hạ b_k nếu vượt.
- **Train bất ổn khi bật GAN.** Entropy model và decoder độc lập về CDF, nhưng tổng loss thay đổi. Phải theo dõi `over` và `gnorm` ở pha 2.

## 7. Thứ tự và thời gian (ước tính)

| Bước | Việc | Công | GPU 6 |
|---|---|---|---|
| Pha 0 | `rate_only.py`, 7 biến thể trên latent MLIC++ | ~1 ngày | ~5 giờ |
| G0 | chọn thành phần, hoặc dừng và chuyển sang g_a | | |
| Pha 1 | module số nguyên, codec v2, test | ~2 ngày | test nhỏ |
| Pha 2 | rd 100k × 2, LRP 50k × 2, đo tốc độ | ~1 ngày | ~6 giờ |
| G1, G2 | BD-rate, tốc độ L4, khớp bit | | cần L4 |
| Train | đầy đủ ~1M bước | | ~1 ngày+ |

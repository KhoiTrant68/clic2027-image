# Plan: Rate-as-Time Flow Bridge cho nén ảnh bitrate cực thấp (CVPR 2027)

Sep 27, 2026 · @Khoi Tran Minh

Mục tiêu: nộp CVPR 2027 trước Nov 16, 2026 với một codec ảnh dưới 0.05 bpp dùng flow bridge một bước, kèm lý thuyết distortion–perception.

## Mục tiêu & đóng góp

Câu chuyện của paper: **"Quantization noise is not diffusion noise."** Các codec diffusion/flow một bước hiện tại giả định sai số lượng tử hóa là Gaussian. Paper này mô hình hóa sai số đó chính xác, rồi chứng minh rằng một flow bridge có OT coupling cho ra đúng đường cong distortion–perception tối ưu, với một bước decode.

**Ba đóng góp dự kiến:**

1. **Lý thuyết.** (a) Một bước Euler của flow matching luôn bằng ước lượng MMSE, giải thích vì sao decoder một bước bị mờ. (b) Bridge với OT coupling xuất phát từ output MMSE đi đúng geodesic D–P tối ưu, nên thời gian t là núm tradeoff có cơ sở. (c) Chặn khoảng cách khi dùng coupling độc lập.
2. **Phương pháp.** Rate-as-Time Mean-Flow Bridge: subtractive dither cho mô hình nhiễu chính xác, ánh xạ bước lượng tử Δ sang thời điểm bắt đầu t, và vận tốc trung bình kiểu MeanFlow. Một model, một bước decode, mọi bitrate.
3. **Thực nghiệm.** Vượt OSCAR / StableCodec / OneDC về LPIPS, DISTS và FID ở dưới 0.05 bpp trên Kodak, CLIC2020, DIV2K-val. Thêm thí nghiệm toy Gaussian, nơi đường cong D–P lý thuyết tính được dạng đóng.

**Tiêu chí thành công tối thiểu để nộp:** đóng góp 1(a)+(b) có chứng minh đầy đủ, và kết quả nhiều bước vượt OSCAR ở cùng bpp. Phần một bước và 1(c) là mục tiêu mở rộng.

## Bối cảnh & đối thủ

Mọi đối thủ trực tiếp đều hoặc giả định nhiễu Gaussian, hoặc né bài toán bằng module riêng cho từng bitrate; chưa paper nào nối decoder với lý thuyết D–P.

| Paper | Venue | Cách xử lý latent lượng tử hóa | Điểm khác biệt của mình |
| --- | --- | --- | --- |
| [FlowCodec](https://arxiv.org/html/2606.21030v1) | arXiv 06/2026 | FLUX/Qwen-image, refine 1 bước, LoRA theo bitrate | Một model cho mọi bitrate; backbone nhỏ hơn nhiều |
| [OSCAR](https://arxiv.org/html/2505.16091) | arXiv 2025 | SD2.1; quy bitrate về timestep giả qua cosine similarity, ép nhiễu giống Gaussian | Mô hình nhiễu chính xác (dither), ánh xạ Δ → t có cơ sở |
| [StableCodec](https://arxiv.org/abs/2506.21977) | arXiv 2025 | One-step diffusion cho nén cực thấp | Có lý thuyết D–P, núm tradeoff liên tục |
| [One-Step Diffusion w/ Semantic Distillation (OneDC)](https://arxiv.org/abs/2505.16687) | arXiv 2025 | One-step + distill semantic | Không cần distill từ teacher nhiều bước |
| [DiffO](https://arxiv.org/abs/2506.16572) | WACV 2026 | Single-step diffusion, bitrate cực thấp | Flow matching thay vì diffusion; OT coupling |
| [AEIC (Shallow Encoder)](https://arxiv.org/abs/2512.12229) | CVPR 2026 | Encoder nông + decoder diffusion một bước | Trực giao: có thể kết hợp encoder nông của họ |
| [VoRTeC](https://pith.science/paper/2609.02291) | arXiv 09/2026 | Video, Wan2.1 frozen, vị trí trên quỹ đạo theo công thức đóng | Là video, nhưng phải trích dẫn và phân biệt ý "vị trí trên quỹ đạo" |

Nền tảng lý thuyết: [Freirich, Michaeli & Meir (NeurIPS 2021)](https://arxiv.org/abs/2107.02555) chứng minh hàm D–P với MSE + W₂ luôn là bậc hai và các estimator tối ưu nằm trên một geodesic Wasserstein giữa estimator MMSE và estimator perception hoàn hảo.

## Phương pháp

Codec gồm một latent codec học được trên VAE frozen, và một flow bridge một bước đưa latent MMSE về latent thật. Bước lượng tử Δ (tức bitrate) quyết định bridge bắt đầu từ thời điểm nào.

&#91;embedded content: pipeline codec · encoder → dither → bridge một bước → decoder\]

**Các thành phần:**

1. **Latent codec.** Analysis/synthesis transform + hyperprior (kiểu Ballé / ELIC) trên latent **DC-AE f32c32 của SANA** (nén 32×, 32 kênh; ảnh 512² cho latent 16×16×32 = 8192 phần tử). *(Đổi backbone 27/9: SD3.5-medium → SANA.)* Ở 0.03 bpp với ảnh 512², ngân sách khoảng 7.900 bit, tức khoảng 1 bit trên mỗi phần tử latent. Đây là vùng mà toy cho thấy mức phạt rate của dither nhỏ, trong khi với latent 64×64×16 của SD3.5 chỉ được khoảng 0.12 bit mỗi phần tử. Nhiều mức Δ dùng chung một mạng qua gain vector (variable-rate).
2. **Subtractive dither.** Encoder và decoder dùng chung seed u \~ U(−Δ/2, Δ/2). Khi đó ŷ = y + e với e \~ U(−Δ/2, Δ/2) độc lập với y, nên source của bridge được biết chính xác.
3. **Rate-as-Time.** Hàm đơn điệu t₀ = τ(Δ), khởi tạo từ lý thuyết (mục Lý thuyết) rồi tinh chỉnh bằng học. Δ lớn → t₀ nhỏ (đi xa hơn trên quỹ đạo).
4. **Mean-Flow bridge.** Mạng u\_θ(z, t₀, 1 | ŷ) học vận tốc trung bình từ t₀ đến 1; decode là ẳ = z̄ + (1 − t₀)·u\_θ. Khởi tạo từ **SANA-1.6B** (Linear DiT, flow matching), điều kiện qua ŷ (concat kênh hoặc ControlNet nhẹ). Không nạp text encoder Gemma-2: dùng một embedding học được hoặc prompt rỗng tính sẵn, giống OSCAR và StableCodec, để tiết kiệm bộ nhớ. **Tùy chọn cho S3:** khởi tạo từ **SANA-Sprint** (bản distill 1–4 bước, 0.6B/1.6B). Lưu ý SANA-Sprint dùng tham số hóa TrigFlow chứ không phải nội suy tuyến tính, nên cần kiểm tra phép chuyển đổi trước khi dùng.
5. **Stochasticity.** Thêm nhiễu σ(t₀)·ε vào điểm bắt đầu để decoder có thể đạt perception hoàn hảo. *Cập nhật 27/9:* khi analysis transform giảm chiều (k < d, trường hợp thường gặp), p\_X\* nằm trên một tập có độ đo 0 nên không tồn tại Brenier map; Định lý 3 dạng tổng quát cần một OT *plan* cộng với ngẫu nhiên phía decoder, và nhiễu σ(t₀)ε chính là thành phần đó. Như vậy đây là thành phần cần thiết về mặt lý thuyết, không chỉ là tùy chọn.

**Training 3 giai đoạn:**

| Giai đoạn | Train gì | Loss | Frozen |
| --- | --- | --- | --- |
| S1 — Codec | g\_a, g\_s, hyperprior | R + λ·MSE trên latent (cho z̄ xấp xỉ MMSE) | DC-AE |
| S2 — Bridge nhiều bước | SANA DiT (full fine-tune với 1.6B, hoặc LoRA rộng) | Conditional FM trên bridge z̄ → z, dùng **coupling tự nhiên kèm nhiễu khởi đầu σ(t₀)ε** (đã duyệt 27/9); minibatch-OT chỉ còn là ablation, vì ở số chiều cao nó suy biến về coupling tự nhiên (xem `refs/code_notes.md`, mục 3.7) | DC-AE, codec |
| S3 — Một bước | DiT khởi tạo từ S2 (hoặc SANA-Sprint) | α-Flow curriculum (target sai phân hữu hạn, không cần JVP, chạy được trên TPU) + LPIPS/DISTS + **GAN vision-aided (DINOv2)** như StableCodec/AEIC (đã duyệt 27/9), vì đối thủ nhờ nó mà có FID/DISTS tốt | DC-AE, codec |

S2 là mốc tối thiểu để nộp; S3 là mở rộng. Nếu thời gian cho phép, fine-tune S1+S2 chung ở cuối.

**S1.5 — Nhánh residual (tùy chọn, thêm ngày 30/9, chỉ bật khi đạt điều kiện ở go/no-go 2):**
- Mạng: một codec pixel nhẹ kiểu ELIC, đồng thời dùng ảnh base x̂ = DC-AE-decode(ẑ) làm điều kiện.
- Mã hóa phần dư x − x̂, với ngân sách bằng một phần nhỏ của tổng bpp.
- Loss: R + λ·(MSE + LPIPS).
- Lý do: trần DC-AE thấp hơn trần f8 1.9 dB PSNR và làm hỏng chữ nhỏ (xem mục Rủi ro).
- Với CLIC 2027 thì nhánh này là **bắt buộc**. Theo phép đo cận dưới trong B6, DC-AE + residual (mã hóa bằng VTM) ngang VTM về PSNR ở 0.075 bpp, mà LPIPS tốt hơn 4 lần. Nhánh này sẽ được làm sau 16/11, kể cả khi CVPR không dùng.
- Thiết kế S1 nên chừa sẵn chỗ cho nhánh này: lưu x̂ và ngân sách rate theo từng ảnh.

## Lý thuyết

Phần lý thuyết gồm một chuỗi 4 kết quả nối với nhau: dither làm phân phối MMSE liên tục → bridge có thể là một map tất định → với OT coupling, map đó vừa thẳng (một bước là đủ) vừa tối ưu D–P → coupling không phải OT thì mất tối đa một lượng có thể chặn được.

**Ký hiệu.** X là latent sạch, Y là thông tin decoder có (bitstream + seed dither), X\* = E\[X | Y\] là ước lượng MMSE, D\* = E‖X − X\*‖², W = W₂(p\_X\*, p\_X). Interpolation tuyến tính x\_t = (1 − t)x₀ + t·x₁.

### Mệnh đề 1 — Dither cho nguồn chính xác (đã biết, chỉ cần phát biểu lại)

Với lượng tử đều bước Δ và subtractive dither u \~ U(−Δ/2, Δ/2) dùng chung, sai số e = round(y + u) − u − y có phân phối U(−Δ/2, Δ/2)^d và độc lập với y (Schuchman 1964; Gray & Stockham 1993).

- **Hệ quả (đã chứng minh, xem `paper/proofs/theory_appendix.tex`):** ŷ luôn có mật độ trên R^k, dù y có phân phối thế nào. p\_X\* liên tục tuyệt đối khi k = d và m(v) = E\[X | ŷ = v\] là Lipschitz địa phương với Jacobian khác 0 hầu khắp nơi. Không có dither, Y rời rạc nên p\_X\* là phân phối rời rạc, và không map tất định nào đưa nó về p\_X liên tục được.
- **Lưu ý (27/9):** khi k < d, p\_X\* *không* liên tục tuyệt đối, xem Remark "Dimension reduction" trong appendix và mục Phương pháp, thành phần 5.
- **Ý nghĩa:** dither đóng vai trò common randomness, điều mà lý thuyết RDP đã biết là có lợi cho perception. **Lưu ý về story:** bản thân đường D–P không đòi hỏi dither, vì không có dither thì một decoder ngẫu nhiên vẫn đạt được đường này. Dither mang lại hai thứ: (1) mô hình nhiễu chính xác và (2) sự tồn tại của một decoder *tất định*, một bước và không cần nhiễu. Framing "quantization noise is not diffusion noise" nên nhấn vào hai điểm này.

### Mệnh đề 2 — Một bước Euler không làm gì cả (mới, dễ)

Với coupling tự nhiên (X₀, X₁) = (X\*, X), vận tốc marginal tại t = 0 là:

```latex
v(x, 0) = \mathbb{E}[X_1 - X_0 \mid X_0 = x] = \mathbb{E}[X \mid X^*] - X^* = 0
```

vì X\* đo được theo σ(Y) và tính chất tháp (tower property) cho E\[X | X\*\] = X\*. Tổng quát hơn, với coupling bất kỳ, một bước Euler từ t = 0 cho E\[X₁ | X₀\], tức một trung bình có điều kiện.

- **Bổ sung (27/9, đã chứng minh):** với X₁ \~ p\_X, một bước Euler đầy đủ đạt W₂ = 0 *khi và chỉ khi* coupling là tất định (X₁ = f(X₀)). Coupling độc lập cho ra hằng số E\[X\]. Toy xác nhận cả hai trường hợp (Mệnh đề 2(b) và 2(c)).
- **Hệ quả:** decoder một bước kiểu Euler hoặc là vô dụng, hoặc sinh ảnh mờ. Muốn một bước mà vẫn sắc nét cần học flow map (MeanFlow), hoặc dùng coupling làm quỹ đạo thẳng (Định lý 3). Đây là lý do có cơ sở cho thiết kế ở mục Phương pháp.

### Định lý 3 — OT bridge đi đúng đường cong D–P tối ưu (đóng góp chính)

*Dạng tổng quát (27/9):* không cần giả thiết liên tục tuyệt đối. Lấy γ\* là một OT plan tối ưu, Z là partner của X\* rút theo γ\* bằng ngẫu nhiên riêng của decoder, và x̂\_t = (1 − t)X\* + tZ; kết luận giữ nguyên. Dạng dưới đây là trường hợp tất định. Giả sử p\_X\* liên tục tuyệt đối (Mệnh đề 1) và có moment bậc hai hữu hạn. Gọi T là OT map Brenier từ p\_X\* sang p\_X, và đặt x̂\_t = (1 − t)X\* + t·T(X\*). Khi đó với mọi t ∈ \[0, 1\]:

```latex
\mathbb{E}\|\hat{x}_t - X\|^2 = D^* + t^2 W^2, \qquad W_2(p_{\hat{x}_t}, p_X) = (1-t)\,W
```

Điểm này nằm đúng trên hàm D–P tối ưu D(P) = D\* + (W − P)² của Freirich et al.

- **Phác thảo chứng minh:** (i) x̂\_t độc lập có điều kiện với X khi biết Y, nên E‖x̂\_t − X‖² = D\* + E‖x̂\_t − X\*‖² (trực giao của kỳ vọng có điều kiện). (ii) E‖x̂\_t − X\*‖² = t²·E‖Z − X\*‖² = t²W². (iii) W₂(p\_x̂t, p\_X) = (1 − t)W: cận trên lấy từ coupling (x̂\_t, Z), cận dưới từ bất đẳng thức tam giác. Bước này không cần lý thuyết geodesic McCann. (iv) Cận dưới D(P) ≥ D\* + (W − P)₊² cũng chỉ cần bổ đề trực giao và bất đẳng thức tam giác. Như vậy phần chứng minh D(P) **tự đứng được**, không phụ thuộc vào điều kiện của Freirich et al., và chỉ trích dẫn họ để ghi công.
- **Liên kết với flow matching (đã chứng minh):** với OT coupling, v(x, 0) = T(x) − x, nên **một bước Euler cỡ t tính từ t = 0 cho đúng x̂\_t**. Với t < 1, map x ↦ (1 − t)x + tT(x) là đơn ánh, nên các quỹ đạo không cắt nhau và straightness gap bằng 0.
- **Đính chính (27/9):** câu cũ "Thẳng và tối ưu D–P là cùng một điều kiện" nói quá. Điều chứng minh được chỉ là: OT ⇒ thẳng và tối ưu D–P. Chiều ngược lại sai khi d ≥ 2. Thêm nữa, một bước Euler chính xác với *mọi* coupling tất định, và cái làm OT nổi bật là distortion: trong số các decoder một bước f(X\*) có perception hoàn hảo, D = D\* + E‖f(X\*) − X\*‖² ≥ D\* + W², dấu bằng khi và chỉ khi f = T (Hệ quả "Why the OT coupling").
- **Hệ quả cho thiết kế:** thời gian t chính là núm tradeoff tối ưu, dừng ở t < 1 cho ảnh ít "bịa" hơn.
- **Còn phải kiểm tra:** số hiệu định lý trong Liu et al. 2023 (bảo toàn marginal, giảm chi phí vận chuyển lồi), cùng các chỗ đánh dấu `% CHECK:` trong appendix.

### Định lý 4 — Coupling không phải OT mất bao nhiêu (mới; phần dễ + phần mở)

**Phần dễ.** Gọi X̂ = Φ(X\*) là điểm cuối ODE của bridge với coupling tự nhiên (rectified flow). Theo Liu et al. (2023), rectification không làm tăng chi phí vận chuyển lồi, nên E‖X̂ − X\*‖² ≤ E‖X − X\*‖² = D\*. Kết hợp với bước (i) ở trên:

```latex
W_2(p_{\hat{X}}, p_X) = 0, \qquad D^* + W^2 \;\le\; \mathbb{E}\|\hat{X} - X\|^2 \;\le\; 2D^*
```

Nghĩa là decoder rectified đạt perception hoàn hảo với distortion không quá 2 lần MMSE. Đây là một cách xây dựng cụ thể cho cận "2×" nổi tiếng của Blau & Michaeli.

**Phần mở (mục tiêu mở rộng 1(c)).** Chặn khoảng cách Δ\_gap = E‖X̂ − X\*‖² − W² theo độ thẳng S = E∫‖v(x\_t, t) − (x₁ − x₀)‖²dt, ví dụ Δ\_gap ≤ C·S hoặc ≤ C·√S. Nếu làm được, nó biện minh việc dùng minibatch / semi-discrete OT trong training.

### Heuristic — khởi tạo τ(Δ)

Khớp SNR: sai số dither có phương sai Δ²/12 mỗi chiều. Chọn t₀ sao cho nhiễu hiệu dụng của x\_t₀ có cùng phương sai, rồi để mạng học hiệu chỉnh. Ghi rõ đây là heuristic, không phải định lý.

### Kiểm chứng toy Gaussian

X \~ N(0, Σ) trong R^d (d = 2 để vẽ, d = 64 để đo), lượng tử có dither. **Đính chính (27/9):** với dither đều, p\_X\* *không* Gaussian nên không dùng được Bures. Thay vào đó, lượng tử trong cơ sở KLT để mọi thứ tách theo từng tọa độ: MMSE là trung bình Gaussian cắt cụt (dạng đóng), còn OT map chính xác là phép sắp xếp lại đơn điệu 1D, T(y) = σΦ⁻¹(F\_Y(y)). So sánh 4 coupling: tự nhiên, độc lập, minibatch OT, OT chính xác. Code nằm ở `experiments/toy_gaussian/`.

**Kết quả vòng 1 (27/9, CPU):**

- **Lý thuyết khớp chính xác.** D\* và W từ công thức đóng khớp với Monte Carlo. Đi theo đường OT chính xác, D = D\* + t²W² tới 3 chữ số và P giảm tuyến tính theo (1 − t)W. W²/D\* ≈ 0.3–0.4, tức perception hoàn hảo tối ưu chỉ tốn thêm khoảng 30–40% distortion so với MMSE, so với +100% của posterior sampling.
- **Mệnh đề 2 đúng.** Coupling tự nhiên với 1 bước Euler trả lại X\*; coupling độc lập với 1 bước co về E\[X\].
- **d = 2 (Δ = 4).** Minibatch OT và OT chính xác chỉ với **1 bước Euler** đạt D ≈ 0.96–0.97·(D\* + W²) và P ≈ floor. Các coupling khác cần ODE nhiều bước. Tuy nhiên endpoint ODE của cả 4 coupling gần như trùng nhau, vì toy tách được theo tọa độ quá dễ.
- **d = 64 (Δ = 2).** Chưa flow học được nào đạt P ≈ 0: kể cả OT chính xác, cận dưới của P vẫn kẹt ở khoảng 0.73. Sai số tập trung ở các chiều có phương sai nhỏ, bị nén mạnh. Minibatch OT với batch 1024 không nhúc nhích khỏi X\*, vì chi phí ghép cặp của nó bằng khoảng 2.5 lần W² thật.

**Vòng 2 (đã duyệt 27/9):**

- **A (A6000):** exact OT ở d = 64, gồm bản gốc, bản tăng capacity (24k bước, width 1024), bản chuẩn hóa từng tọa độ, và bản kết hợp cả hai. Mục tiêu là tách "train chưa đủ" với "map khó học".
- **B (A6000):** minibatch OT theo kích thước pool 256 → 16384 (Sinkhorn trên GPU) ở d ∈ {16, 64}, kèm chỉ số chi phí ghép cặp / W².
- **C (CPU):** toy 2D không tách được (Σ tương quan, lượng tử trong cơ sở chuẩn), dùng cho Hình 3. **Đã xong (27/9):** chỉ các coupling OT decode tới đích trong 1 bước (D ≈ 0.97–0.99·(D\* + W²), P ≈ floor). Endpoint ODE của cả 4 coupling lại trùng nhau: rectified flow thực tế gần như tối ưu OT, nên cận 2D\* của Định lý 4 đúng nhưng lỏng. Output của mọi flow học được còn "vệt" của X\*, vì X\* gần suy biến.
- **A5–A7 (thêm 27/9):** nhiễu khởi đầu X\* + ε với ε\_i \~ N(0, (c·s\_i)²), trong đó s\_i² = E Var(X\_i | Y) và c ∈ {0.25, 0.5, 1.0}. Chạy thử nhanh (300 bước): với c = 0.5, map oracle có D = 1.18·(D\* + W²); flow học được đã đưa cận dưới của P xuống 0.23, trong khi bản không nhiễu sau 6000 bước vẫn ở 0.73. Nhiễu đổi distortion lấy khả năng học được map, và c là núm điều chỉnh sự đánh đổi đó.
- **Đo sai số lượng tử** (`experiments/quant_noise/measure_quant_noise.py`): mã hóa ảnh bằng DC-AE của SANA (mặc định; vẫn hỗ trợ VAE KL như SD3.5), rồi dùng transform coder làm proxy (direct, KLT 2×2), dò Δ để đạt 0.01–0.05 bpp, và đo kurtosis, corr(e, y), tương quan giữa các kênh, tương quan không gian, độ phụ thuộc nội dung, so với tham chiếu Gaussian. Pipeline đã kiểm tra trên latent tổng hợp; cần chạy trên Kodak/CLIC.

### Trạng thái các kết quả

| Kết quả | Độ khó | Cần cho CVPR? | Việc còn lại |
| --- | --- | --- | --- |
| Mệnh đề 1 (dither) | Thấp | Có | **Bản nháp LaTeX xong (27/9).** Cần review; giả thiết k = d được nêu rõ |
| Mệnh đề 2 (Euler = identity) | Thấp | Có | **Bản nháp LaTeX xong (27/9)**, thêm phần (d) "iff coupling tất định" |
| Định lý 3 (OT = tối ưu D–P) | Trung bình | Có | **Bản nháp LaTeX xong (27/9)**: dạng tổng quát dùng OT plan, chứng minh D(P) tự đứng được, phần FM có Hệ quả "Why OT" |
| Định lý 4, phần dễ (≤ 2D\*) | Trung bình | Có | **Bản nháp LaTeX xong (27/9)**. Còn đối chiếu số hiệu định lý của Liu et al. |
| Định lý 4, phần mở (theo S) | Cao | Không | **Bỏ khỏi bản CVPR (27/9, do làm một mình)**; để dành cho ICLR 2028 |

## Thiết kế thí nghiệm

Mọi thí nghiệm phải trả lời một trong ba câu: lý thuyết có đúng trên thực tế không, codec có thắng đối thủ không, và thành phần nào tạo ra mức thắng đó.

**Dữ liệu & metric**

| Hạng mục | Lựa chọn |
| --- | --- |
| Train | DF2K + LSDIR (giống OSCAR), crop 512×512 (latent DC-AE 16×16×32), cache latent trước |
| Test | Kodak (24), CLIC2020 test (428), DIV2K-val (100) |
| Bitrate | 4 mức trong khoảng \~0.01–0.05 bpp, tính bpp thật từ bitstream |
| Perception | LPIPS, DISTS, FID và KID trên patch 256 của CLIC2020 (theo HiFiC) |
| Fidelity | PSNR, MS-SSIM (báo cáo đầy đủ để reviewer compression không bắt bẻ) |
| Tổng hợp | BD-rate theo LPIPS và DISTS; thời gian encode/decode trên A6000; số tham số |

**Baseline:** OSCAR, StableCodec, OneDC, DiffO, AEIC (các codec có code công khai), FlowCodec nếu có code, PerCo, MS-ILLM, VTM, DCVC-RT (image mode nếu áp dụng được). Codec không có code thì lấy số liệu từ paper ở cùng dataset và bpp.

**Ablation (chạy trên SANA-0.6B để tiết kiệm, xác nhận lại 2–3 cái trên SANA-1.6B):**

1. Dither vs. không dither vs. ép nhiễu Gaussian kiểu OSCAR → kiểm chứng Mệnh đề 1.
2. Coupling độc lập vs. minibatch-OT vs. semi-discrete OT → kiểm chứng Định lý 3–4.
3. Bridge từ z̄ vs. từ noise thuần → giá trị của việc xuất phát từ MMSE.
4. τ(Δ) cố định (heuristic SNR) vs. học được.
5. Euler 1 bước vs. MeanFlow 1 bước vs. nhiều bước → minh họa Mệnh đề 2.
6. Quét t từ t₀ đến 1 → đường cong distortion–perception thực nghiệm.

**Bảng và hình dự kiến trong paper:**

- Hình 1: ảnh so sánh ở \~0.02 bpp (mình vs. OSCAR vs. VTM).
- Hình 2: pipeline (như sơ đồ ở trên).
- Hình 3: toy Gaussian — đường cong D–P lý thuyết và điểm của 3 coupling.
- Hình 4: RD curve LPIPS/DISTS/FID theo bpp trên Kodak và CLIC.
- Hình 5: đường D–P thực nghiệm khi quét t (ablation 6).
- Bảng 1: BD-rate + tốc độ + số tham số so với baseline.
- Bảng 2: ablation 1–5.

## Compute

8×A6000 trong 7 tuần cho tối đa khoảng 9.400 GPU-giờ; plan dưới đây dùng khoảng 60% để chừa chỗ cho lỗi và chạy lại. Các con số là ước lượng, cần đo lại throughput thật ở tuần 2.

**Cập nhật 27/9: tài nguyên chưa chắc chắn.** Máy 8×A6000 hiện có khả năng chưa dùng được và đang được lấy lại. Phương án thay thế:

- **Kaggle** (2×T4 hoặc P100, khoảng 30 GPU-giờ/tuần): toy vòng 2 A/B, script đo sai số lượng tử, eval.
- **Google TPU Research Cloud** (30 ngày miễn phí, tính từ lúc gửi project number): dùng cho S1–S3. Tạo project ngay, nhưng *chỉ gửi form khoảng 8–10/10*, sau khi code đã chạy được trên Colab/Kaggle TPU, để 30 ngày phủ khoảng 10/10–9/11. Code viết bằng PyTorch, device-agnostic (cuda/xla/cpu). Rủi ro: JVP của MeanFlow trên XLA; phương án dự phòng là sai phân hữu hạn.
- Nếu lấy lại được A6000, TRC trở thành tài nguyên bổ sung cho ablation.

**Backbone (đổi 27/9):** **SANA-1.6B** là chính, **SANA-0.6B** cho ablation; cả hai dùng chung DC-AE f32c32, nên chỉ cache latent một lần. Lý do đổi:

- **Compute chưa chắc chắn.** Các codec một bước của đối thủ đều dùng UNet khoảng 0.9B, và AEIC chỉ train trên ≤ 4×3090. SD3.5-medium (2.5B) nặng hơn tất cả.
- **SANA là flow matching**, nên khớp với phần lý thuyết.
- **Latent 32× có ít token** (256 token cho ảnh 512²), nên train rẻ, và rate trên mỗi phần tử latent rơi vào vùng dither ít bị phạt.

Rủi ro mới: xem mục Rủi ro. Không dùng SD3.5 hay FLUX.

| Hạng mục | GPU | Thời gian ước tính | Ghi chú |
| --- | --- | --- | --- |
| Cache latent DC-AE cho DF2K + LSDIR | 8 | \~0.5 ngày | Làm một lần, dùng cho cả SANA-0.6B và 1.6B |
| Toy Gaussian | 1 | vài giờ | Có thể chạy CPU |
| Chạy lại baseline + eval | 2 | \~1–2 ngày | *(27/9)* Chỉ chạy lại StableCodec/AEIC, vì có checkpoint và code eval. Với các baseline khác, lấy số từ paper: StableCodec có sẵn `results/results.txt` cho Kodak/CLIC/DIV2K. Số của OSCAR phải ghi chú (họ resize ảnh, dùng FID full-res, bpp danh nghĩa) |
| S1 — codec | 6 | \~2–3 ngày | Nhẹ, chỉ train transform + hyperprior |
| S2 — bridge nhiều bước | 6 | \~4–5 ngày | bf16 + gradient checkpointing + FSDP/ZeRO-2 |
| S3 — MeanFlow một bước | 6 | \~4–5 ngày | JVP nặng bộ nhớ, xem rủi ro |
| Ablation trên SANA-0.6B | 2 | \~1 ngày mỗi lần, 6 lần | Chạy song song với S2/S3 |

**Quy ước vận hành:** GPU 0–5 dành cho train chính, GPU 6–7 cho baseline/ablation/eval. Mọi run log lên W&B với config + commit hash. Checkpoint mỗi 2 giờ và có script resume.

## Timeline 7 tuần

Lý thuyết chạy song song với thí nghiệm ngay từ tuần 1; hai mốc go/no-go chặn rủi ro lớn nhất trước khi tốn nhiều compute.

&#91;embedded content: timeline 7 tuần · thí nghiệm, lý thuyết, viết, 2 mốc go/no-go\]

**Tiêu chí hai mốc:**

| Mốc | Đi tiếp nếu | Nếu không đạt |
| --- | --- | --- |
| Go/no-go 1 (4/10) | *(tiêu chí đổi 27/9)* **(a)** Trên latent DC-AE thật (Kodak + CLIC, transform coder proxy KLT 2×2, 4 mức 0.01–0.05 bpp), một denoiser nhỏ được train với **mô hình nhiễu chính xác** (subtractive dither, nhiễu đều đã biết) đạt MSE latent thấp hơn **≥ 5%** so với cùng kiến trúc được train với **giả định Gaussian** kiểu OSCAR (ŷ = y + N(0, Δ²/12)), ở cùng rate, tại ít nhất 2/4 mức bpp. **VÀ (b)** toy Gaussian cho thấy decode một bước với OT coupling sát đường D–P (đã đạt ở d = 2, xem kết quả vòng 1 và C) | Đổi trọng tâm sang Lý thuyết D–P + decoder nhẹ (Mệnh đề 2, Định lý 3–4 vẫn giữ), bỏ story "quantization noise" |
| Go/no-go 2 (25/10) | S2 nhiều bước vượt OSCAR về LPIPS hoặc DISTS ở ít nhất 2/4 mức bpp | Nộp bản "bridge nhiều bước + lý thuyết"; S3 một bước dời sang bản nộp ICCV 2027 |
| *(thêm 30/9)* Kiểm tra trần, cùng lúc với go/no-go 2 | LPIPS/DISTS của S2 ở 0.05 bpp cách trần DC-AE **> 15%**, và không thua StableCodec chỉ ở mức bpp cao nhất → trần chưa phải nút thắt, **không làm S1.5** | Bật S1.5 (residual): ~3–4 ngày, lấy thời gian từ ablation 4–6 hoặc từ S3 |

**Việc cụ thể theo tuần:**

- **T1 (28/9–4/10):** dựng repo, cache latent; toy Gaussian; đo phân phối sai số lượng tử; viết sạch Mệnh đề 1–2. *Tiến độ 27/9:* toy vòng 1 xong; bản nháp LaTeX của Mệnh đề 1–2, Định lý 3 và Định lý 4 (phần dễ) xong. Toy vòng 2: chạy A và B trên A6000 ngày 28–29/9, C trên CPU; phân tích ngày 30/9–1/10; chốt dữ liệu go/no-go 1 ngày 2–3/10. Script đo sai số lượng tử đã viết, mặc định dùng DC-AE của SANA; còn phải chạy trên Kodak/CLIC. **Việc mới cho go/no-go 1:** so sánh denoiser dùng mô hình nhiễu chính xác với denoiser giả định Gaussian trên latent DC-AE. Train nhỏ, khoảng 1 GPU-giờ trên Kaggle. Backbone đổi sang SANA (27/9).
- **T2 (5–11/10):** *(thêm 30/9)* **trước khi train S1, chạy `notebooks/parity.ipynb`**: DC-AE/SANA viết lại bằng torch thuần (`src/ratflow/nn`) phải khớp diffusers, entropy số nguyên (`src/ratflow/entropy`) phải khớp từng bit giữa CPU và GPU. S1 dùng `QConv2d`/`ScaleMeanHead`/`GaussianConditional`/`DiscretePrior`, nên bpp đo trên bitstream thật và decoder dùng lại được cho CLIC. Sau đó train S1 *(parity đạt 9/9 ngày 01/10; code S1 viết xong ngày 01/10: `src/ratflow/codec/latent_codec.py`, `experiments/s1/`, notebook `s1_cache` rồi `s1_train`. Thiết kế: g_a/g_s ResBlock N=192, y stride 2 (M=128), hyperprior Z=96 với h_s số nguyên có bias theo rate, 8 mức rate qua gain vector, **subtractive dither nằm trong bitstream** (Prop. 1), rate khi train là độ dài mã chính xác của q khi biết u, bảng chọn theo scale × 16 mức phần lẻ của tâm; λ ∈ [0.03, 4] cần chỉnh lại sau lần chạy đầu để phủ 0.01–0.05 bpp (CVPR) và tới 0.3 bpp (CLIC))*; chạy lại StableCodec/AEIC (dùng checkpoint có sẵn); đo throughput thật để chốt ngân sách compute; chứng minh Định lý 3. *(Thêm 30/9)* Đo trần DC-AE và trần f8 trên Kodak/CLIC2020/DIV2K-val bằng notebook `notebooks/ceiling.ipynb` (script `experiments/ceiling/ceiling.py`; notebook đã làm sẵn ngày 30/9, ~1.5–2.5 giờ trên Kaggle T4, có cả FID/KID trên patch), để có đường trần cho các RD plot.
- **Kết quả S1 lần 1 (02/10, Kaggle T4, DIV2K 800 ảnh, 200k bước trong ~2.5 giờ; `results/s1/`):**
  - **Bitstream đúng như thiết kế:** selftest đạt. ŷ do decoder dựng lại khớp từng bit với ŷ mà encoder dự kiến; số byte thật lệch số ước lượng lúc train dưới 0.5%.
  - **Đã hội tụ:** bpp và MSE gần như không đổi từ khoảng 100k bước.
  - **Kodak, bitstream thật:**

    | Mức rate | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
    |---|---|---|---|---|---|---|---|---|
    | bpp | 0.0155 | 0.0204 | 0.0342 | 0.0531 | 0.0710 | 0.0885 | 0.1057 | 0.1215 |
    | MSE latent | 0.367 | 0.286 | 0.169 | 0.087 | 0.045 | 0.023 | 0.011 | 0.006 |
    | PSNR (dB) | 17.83 | 18.57 | 19.62 | 20.71 | 21.70 | 22.45 | 22.99 | 23.29 |
    | LPIPS | 0.523 | 0.449 | 0.309 | 0.206 | 0.152 | 0.122 | 0.105 | 0.098 |

    Trần DC-AE trên Kodak: 23.66 dB, LPIPS 0.091.
  - **Bão hòa sớm:** từ khoảng 0.1 bpp, S1 đã sát trần DC-AE, nên thêm bit cho latent gần như vô ích. Điều này xác nhận nhánh residual là bắt buộc cho CLIC ở 0.15/0.3 bpp.
  - **Dải rate chưa phủ đúng:** mức thấp nhất mới tới 0.0155 bpp, trong khi CVPR cần xuống 0.01. Ba mức trên cùng lại phí vì đã bão hòa.
  - **Thời gian (Kodak):** encode S1 mất 0.05 s/ảnh, decode S1 mất 0.40 s/ảnh (rANS numpy + h_s số nguyên trên CPU). Với CVPR thì chấp nhận được; với CLIC cần tối ưu (ghi vào A4).
  - **Việc tiếp:** train lại S1 với λ = geomspace(0.015, 0.6, 8), ước tính phủ khoảng 0.010–0.08 bpp (rẻ: ~2.5 giờ T4). Phải xong trước khi bắt đầu S2, vì S2 train trên đầu ra của S1. Cân nhắc thêm Flickr2K, vì bpp trên Kodak cao hơn bpp lúc train ~20% ở r0.
- **T3–T4 (12–25/10):** train S2 với coupling tự nhiên cộng nhiễu khởi đầu (hệ số c chọn theo toy A5–A7); ablation 1–3 trên SANA; Định lý 4 phần dễ + thử phần mở.
- **T5–T6 (26/10–8/11):** S3 α-Flow cộng GAN DINOv2; ablation 4–6; eval đầy đủ + BD-rate; bắt đầu viết từ 2/11.
- **T7 (9–16/11):** đăng ký abstract 10/11; hoàn thiện paper, appendix chứng minh; nộp 16/11 (AoE, tức khoảng 19:00 ngày 17/11 giờ Việt Nam). Supplementary trước 23/11.

## Rủi ro & phương án dự phòng

Rủi ro lớn nhất là bị "scoop" và thiếu thời gian cho S3; cả hai đã có phương án lùi trong timeline.

| Rủi ro | Khả năng | Tác động | Phương án |
| --- | --- | --- | --- |
| Nhóm khác ra paper gần giống (FlowCodec, VoRTeC đang đi rất nhanh) | Cao | Cao | Theo dõi arXiv hàng tuần; phần lý thuyết D–P khó bị trùng; đăng arXiv ngay sau khi nộp |
| Sai số lượng tử gần Gaussian (Go/no-go 1 thất bại) | Trung bình | Trung bình | Chuyển trọng tâm sang Định lý 3–4 + decoder nhẹ |
| JVP của MeanFlow tràn bộ nhớ hoặc không chạy được trên XLA | Trung bình | Trung bình | Dùng target sai phân hữu hạn của AlphaFlow (`_compute_mean_velocity_d`, không cần JVP); hoặc chạy S3 trên SANA-0.6B |
| *(mới 27/9; **đã xảy ra một phần**, đo ngày 30/9)* **Trần chất lượng của DC-AE f32.** Nén 32× có thể giới hạn PSNR/chi tiết, và đối thủ dùng VAE 8× | ~~Trung bình~~ **Cao** (PSNR, chữ) / **Thấp–TB** (LPIPS/DISTS) | Trung bình | **Kết quả đo** trên 30 ảnh CLIC 2025-test (`results/clic_b/b_results/`), khi chỉ encode rồi decode, chưa lượng tử:<br>• DC-AE f32: **PSNR 22.86 dB**, LPIPS 0.082, DISTS 0.060<br>• VAE f8 của sd-turbo (giống StableCodec): **24.73 dB**, LPIPS 0.073, DISTS 0.055<br>→ **Kém 1.9 dB PSNR nhưng chỉ kém ~10% LPIPS/DISTS.**<br>• **Chữ nhỏ bị biến thành chữ ảo** ngay ở mức trần (text PSNR 16.7 dB trên ảnh màn hình), và khuôn mặt nhỏ bị trôi danh tính.<br><br>**Phương án giảm thiểu, kích hoạt có điều kiện:**<br>(1) T2: đo trần DC-AE trên chính Kodak/CLIC2020/DIV2K-val (`notebooks/ceiling.ipynb`), rồi **vẽ đường trần** trên mọi RD plot, để reviewer thấy giới hạn này là của backbone chứ không phải của phương pháp.<br>(2) Tại go/no-go 2: nếu S2 ở 0.05 bpp đã **sát trần** (LPIPS/DISTS cách trần ≤ 15%), hoặc thua StableCodec *chỉ ở mức bpp cao nhất*, thì thêm **nhánh residual nhẹ ở pixel, có điều kiện theo ảnh base** (như `res` của StableCodec) làm giai đoạn S1.5. Nếu không thì để dành cho CLIC (sau 16/11).<br>(3) Không dùng ảnh có chữ trong các hình so sánh định tính, và nêu giới hạn về chữ ở mục Limitations |
| *(mới 27/9)* Linear attention của SANA với JVP/XLA, và việc bỏ text encoder | Thấp | Thấp | Kiểm tra forward và backward trên Colab TPU trước khi gửi form TRC |
| S2 không vượt OSCAR | Trung bình | Cao | Kiểm tra S1 trước (z̄ có đủ tốt không); tăng điều kiện qua ControlNet; thêm perceptual loss nhẹ |
| Giả thiết Định lý 3–4 không khớp với thực tế (latent không liên tục tuyệt đối, OT xấp xỉ) | Thấp | Trung bình | Phát biểu định lý cho trường hợp lý tưởng, kiểm chứng bằng toy, và nói rõ khoảng cách trong phần Limitations |
| Chạy lại baseline không ra số như paper | Trung bình | Thấp | Dùng số của paper gốc, ghi chú nguồn; chỉ so sánh trực tiếp với baseline chạy được |
| Reviewer compression chê PSNR thấp | Cao | Trung bình | Báo cáo đầy đủ PSNR; dùng Định lý 3 để biện luận: núm t cho phép chọn điểm fidelity cao hơn |
| *(mới 27/9)* **Dither tốn rate ở bitrate cực thấp.** Toy Gaussian 1D so ở cùng rate: tại 0.01–0.1 bit/chiều, MMSE có dither chỉ giữ được khoảng 50–75% mức giảm distortion của lượng tử không dither, tức cần thêm khoảng 30–50% bit cho cùng D. Tại ≥ 1 bit/chiều thì không còn khác biệt. | Trung bình | Cao | Đo trên codec thật ở S1: so D\* có và không có dither ở cùng rate. Nếu mức phạt lớn, dùng dither chỉ cho các hệ số có rate cao, hoặc dùng dither lúc train rồi lượng tử tất định kèm MMSE lúc test, và sửa Mệnh đề 1 cho phù hợp |
| *(mới 27/9, **đã xử lý**: tiêu chí đã đổi)* **Tiêu chí go/no-go 1 cũ gần như chắc chắn đạt vì lý do tầm thường.** Ở ≤ 0.05 bpp, 95–99% chỉ số lượng tử bằng 0, nên sai số không dither ≈ −y (corr(e, y) ≈ −1), mặc nhiên "không Gaussian" | Cao | Trung bình | Đổi tiêu chí sang câu hỏi thực sự quan trọng: residual X − X\* (thứ bridge phải bù) có lệch khỏi Gaussian đẳng hướng hay không, và decoder biết mô hình nhiễu chính xác có thắng decoder giả định Gaussian (kiểu OSCAR) ở cùng rate hay không |
| *(mới 27/9)* Minibatch OT mất tác dụng ở số chiều cao: ở toy d = 64, batch 1024 cho chi phí ghép cặp ≈ 2.5·W², và 1 bước decode không dịch chuyển | Cao | Cao | Thí nghiệm B đo xu hướng theo batch. Nếu không đạt thì chuyển sang semi-discrete OT, hoặc ghép cặp trong một không gian đặc trưng ít chiều |
| *(mới 27/9)* Map OT khó học ở các chiều bị nén mạnh: ở toy d = 64, kể cả coupling chính xác vẫn không đạt P ≈ 0 | Trung bình | Cao | Thí nghiệm A. Nếu nguyên nhân là conditioning thì chuẩn hóa từng kênh cho ŷ/z̄; nếu là bản chất thì dùng nhiễu khởi đầu σ(t₀)ε, theo Định lý 3 dạng plan |

## Quyết định cần bạn duyệt

Tích vào ô khi đồng ý, hoặc để lại comment ở dòng muốn đổi.

- [x] **Venue:** nhắm CVPR 2027 (nộp 16/11), bản dự phòng là ICCV 2027.
- [x] **Story chính:** "Quantization noise is not diffusion noise" + lý thuyết D–P, thay vì chỉ đua số liệu.
- [x] **Backbone (đổi 27/9):** SANA-1.6B làm chính, SANA-0.6B cho ablation (thay cho SD3.5-medium).
- [x] **Mục tiêu tối thiểu:** Mệnh đề 1–2 + Định lý 3 + Định lý 4 phần dễ, cùng S2 nhiều bước; S3 một bước và phần mở của Định lý 4 là mở rộng.
- [x] **Hai mốc go/no-go** và tiêu chí như bảng ở mục Timeline.
- [x] **Phân bổ GPU:** 6 GPU train chính, 2 GPU baseline/ablation.
- [x] **Đăng arXiv** ngay sau khi nộp (CVPR cho phép; cần kiểm tra lại quy định anonymity 2027).
- [x] *(mới 30/9, **đã duyệt 30/9**)* **Trần DC-AE:** giữ DC-AE f32 làm backbone (không quay lại dùng VAE f8), vì trên thước đo cảm nhận chỉ kém ~10%. Thêm đường trần vào các RD plot, và chỉ bật S1.5 (residual) khi đạt điều kiện ở mốc kiểm tra trần. Phương án khác là đổi sang backbone f8, nhưng như vậy mất lợi thế ~1 bit trên mỗi phần tử latent của DC-AE (mục Phương án, ý 1), và phải cache lại latent.

**Câu hỏi mở:**

- ~~Ai phụ trách chứng minh, ai phụ trách training?~~ **Trả lời 27/9: một người làm tất cả.** Hệ quả cho plan:
  - Phần chứng minh đã có bản nháp LaTeX (Mệnh đề 1–2, Định lý 3, Định lý 4 phần dễ). Việc còn lại chủ yếu là *đọc duyệt và sửa*, khoảng 2–3 buổi, làm vào những lúc GPU đang chạy.
  - Cắt phạm vi: bỏ Định lý 4 phần mở; chỉ chạy lại 2 baseline; ablation giữ 1, 2, 5, 6 (ablation 3 và 4 chỉ làm nếu còn thời gian).
  - Mỗi tuần chỉ theo một luồng compute chính. Claude viết code, người chạy job và duyệt kết quả.
  - S3 một bước vẫn là mở rộng; go/no-go 2 (25/10) quyết định có làm hay không.
- Có muốn thêm một người có nền OT/xác suất để review chứng minh trước khi nộp không? *Khuyến nghị:* nhờ một người quen đọc appendix khoảng 1–2 giờ trong tuần 2–3/11. Chi phí thấp, và giảm rủi ro reviewer bắt lỗi toán.

## Tài liệu tham khảo

**Lý thuyết (đọc trước tuần 2):**

- [Freirich, Michaeli & Meir — A Theory of the Distortion-Perception Tradeoff in Wasserstein Space (NeurIPS 2021)](https://arxiv.org/abs/2107.02555)
- Blau & Michaeli — The Perception-Distortion Tradeoff (CVPR 2018); Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff (ICML 2019)
- Liu, Gong & Liu — Flow Straight and Fast: Rectified Flow (ICLR 2023)
- Albergo, Boffi & Vanden-Eijnden — Stochastic Interpolants
- Schuchman (1964); Gray & Stockham (1993) — dithered quantization
- Theis et al. — Lossy Compression with Gaussian Diffusion (2022)
- [Flow Matching with Semidiscrete Couplings (ICLR 2026)](https://iclr.cc/virtual/2026/poster/10011571)

**Codec đối thủ:** [FlowCodec](https://arxiv.org/html/2606.21030v1) · [OSCAR](https://arxiv.org/html/2505.16091) · [StableCodec](https://arxiv.org/abs/2506.21977) · [OneDC](https://arxiv.org/abs/2505.16687) · [DiffO](https://arxiv.org/abs/2506.16572) · [AEIC](https://arxiv.org/abs/2512.12229) · [VoRTeC](https://pith.science/paper/2609.02291) · [GVCC](https://arxiv.org/abs/2603.26571)

**One-step flow:** [Improved Mean Flows](https://arxiv.org/abs/2512.02012) · [AlphaFlow](https://iclr.cc/virtual/2026/poster/10008684) · [Stable Mean Flow](https://cvpr.thecvf.com/virtual/2026/poster/36747)

**Deadline:** [CVPR 2027 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)

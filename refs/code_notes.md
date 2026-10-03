# Ghi chú đọc code các paper tham khảo (27/9/2026)

Các repo được clone shallow vào `refs/repos/`. Đường dẫn file trong ghi chú này tính từ thư mục đó.

| Paper | Repo | Trạng thái |
|---|---|---|
| OSCAR (NeurIPS'25) | [jp-guo/OSCAR](https://github.com/jp-guo/OSCAR) | Đã đọc model, train, test |
| StableCodec (ICCV'25) | [LuizScarlet/StableCodec](https://github.com/LuizScarlet/StableCodec) | Đã đọc model, codec, loss, eval, kết quả báo cáo |
| AEIC (CVPR'26) | [LuizScarlet/AEIC](https://github.com/LuizScarlet/AEIC) | Đọc lướt: cùng họ với StableCodec |
| DiffO (WACV'26) | [Freemasti/DiffO](https://github.com/Freemasti/DiffO) | Đọc lướt: dựa trên ResShift |
| OneDC (NeurIPS'25) | [onedc-codec/onedc](https://github.com/onedc-codec/onedc) | Đọc lướt cấu trúc |
| MeanFlow / iMF (PyTorch, không chính thức) | [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow) | Đã đọc phần loss |
| AlphaFlow (ICLR'26) | [snap-research/alphaflow](https://github.com/snap-research/alphaflow) | Đã đọc phần loss |
| torchcfm (OT-CFM) | [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) | Đã đọc OT sampler |
| Rectified Flow | [gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) | Mới clone, chưa đọc |
| FlowCodec, VoRTeC | chưa thấy code công bố | — |
| Semidiscrete FM (Apple) | chưa thấy repo chính thức | — |
| iMF chính thức (JAX, TPU) | [Lyy-iiis/imeanflow](https://github.com/Lyy-iiis/imeanflow) | Chưa clone. Hữu ích nếu chuyển sang JAX/TPU |

---

## 1. Đối thủ làm gì thực sự (khác với cách paper mô tả)

### OSCAR (`diffusion/oscar.py`, `diffusion/models/hyper_encoder.py`)

- **Backbone:** UNet SD-2.1 với LoRA; VAE 4 kênh dùng `latent_dist.sample()`.
- **Lượng tử hóa:** **VQ với codebook cố định** (`vector_quantize_pytorch`, cosine sim), mượn hyper-encoder của PerCo. **Không có entropy coding.** bpp là giá trị danh nghĩa, tính bằng log₂(codebook) × số token / số pixel (xem `utils/rate_config.py`).
- **"Một model cho mọi bitrate" chỉ đúng một phần:** UNet dùng chung, nhưng có **8 hyper-encoder riêng**, mỗi bitrate một cái (`nn.ModuleList`).
- **Ánh xạ bitrate sang timestep:** là một **bảng cứng**, `timesteps = [500, 400, 310, 280, 210, 190, 170, 150]`. Code không tính theo công thức nào lúc chạy.
- **Chuẩn hóa norm:** `new_z_hat = z_hat · ‖z‖/‖z_hat‖` theo từng kênh, trong đó ‖z‖ lấy **từ latent của encoder**, kể cả lúc test. Đây là thông tin phụ (4 số thực mỗi ảnh) không được tính vào bpp. Về số bit thì không đáng kể, nhưng cần nhớ khi so sánh.
- **Decode:** một bước DDPM `pred_original_sample` tại timestep giả.
- **Loss:** L1 + DISTS (pyiqa) + cosine latent (trọng số 2) + GAN tùy chọn. Discriminator hoạt động trong latent space và xây trên UNet.
- **Eval** (`main_test.py`): ảnh GT bị **resize LANCZOS về bội số của 128** trước khi đo; FID dùng pyiqa trên ảnh full-res. Số liệu vì vậy **không so trực tiếp được** với giao thức patch-FID.

### StableCodec (`src/StableCodec.py`, `src/latent_codec.py`) và AEIC

- **Backbone:** UNet **SD-Turbo** (SD-2.1 đã distill), gọi ở **t = 999 cố định**. `conv_in` được thay để nhận **feature 320 kênh** từ codec (AEIC dùng 256 kênh), chứ không phải latent nhiễu 4 kênh. Thêm vào đó là một residual `res` cộng sau bước denoise.
  → Diễn giải "latent nhiễu là trạng thái diffusion" gần như bị bỏ. Về bản chất đây là một generator feed-forward khởi tạo từ SD-Turbo.
- **Codec:** kiểu ELIC. Có hyperprior (EntropyBottleneck + GaussianConditional), context checkerboard 4 phần, và LRP.
  - Lượng tử bằng **`ste_round(y − mean) + mean`**: tất định, **không dither**.
  - Rate lúc train là likelihood của CompressAI với nhiễu đều. Có entropy coding thật (`compress`/`decompress`).
  - Nhánh phụ dùng `g_a` của ELIC pretrained, đóng băng.
- **Mỗi bitrate là một checkpoint riêng**, fine-tune theo λ (ft2, ft3, ft4, ft6, ft8).
- **Loss:**
  - Giai đoạn 1: rate + L2 + LPIPS-VGG + CLIP.
  - Giai đoạn 2: thêm **GAN vision-aided với backbone DINOv2** (`vision_aided_loss`).
- **Eval** (`src/evaluate.py`): pyiqa (PSNR, DISTS, MS-SSIM), cùng **torchmetrics FID/KID** với **`neuralcompression.metrics.update_patch_fid`**, tức giao thức patch của HiFiC/MS-ILLM. bpp lấy từ bitstream thật.
- **Số liệu báo cáo có sẵn** trong `StableCodec/results/results.txt` cho Kodak, CLIC2020 và DIV2K-val, từ 0.004 đến 0.036 bpp. Có thể dùng trực tiếp làm đường baseline. Ví dụ CLIC2020 ở 0.0216 bpp: DISTS 0.073, FID 3.73.
- **Compute của AEIC:** "tối đa 4× RTX 3090 (24 GB)". Dữ liệu train gồm DIV2K + CLIC train + 10K ảnh LSDIR.

### DiffO

- Xây trên **ResShift**: một bridge có dạng x_t = x₀ + η_t(y − x₀) + κ√η_t·ε, tức nội suy từ ảnh chất lượng thấp sang ảnh chất lượng cao kèm nhiễu. Lượng tử bằng VQ (taming).
- Decode một bước bằng `pred_xstart` ở bước đầu (`models/gaussian_diffusion.py`, khoảng dòng 831).
- **Ý tưởng "bridge từ tín hiệu đã lượng tử" đã có trong DiffO/ResShift.** Cần trích dẫn và phân biệt: của họ là bridge Gaussian với lịch nhiễu thủ công, còn của mình dùng OT coupling, có lý thuyết D–P và mô hình nhiễu chính xác.

### OneDC

- SD-1.5 + entropy model kiểu DCVC + tokenizer MaskGIT-VQGAN để distill semantic vào hyperprior. Giai đoạn 2 dùng **DMD** (`src/modules/dmd/sd_guidance.py`).

---

## 2. Code flow một bước

### MeanFlow / iMF (`MeanFlow/meanflow.py`)

- Model có **hai head, (u, v)**. Mục tiêu JVP dùng **v do chính model dự đoán** (`v_c`) làm tangent, chứ không dùng velocity có điều kiện x₁ − x₀. Tức là đúng tinh thần iMF.
  → Điều này quan trọng cho bridge của mình: theo Mệnh đề 2, velocity có điều kiện ≠ velocity marginal, và dùng v đã học làm tangent là cách xử lý đúng.
- JVP được tính bằng `torch.autograd.functional.jvp` trong `no_grad`, **phải tắt flash-attention**. Hàm này dùng double-backward nên tốn bộ nhớ.
- Quy ước: **t = 1 là nhiễu, t = 0 là data**, ngược với quy ước trong plan. Cần cẩn thận khi port.
- Thời gian lấy mẫu theo phân phối logit-normal(−0.4, 1.0); dùng adaptive L2 loss.

### AlphaFlow (`alphaflow/src/training/loss.py`, `_compute_mean_velocity_d`)

- Khi α < 1, target **không cần JVP**: nó dùng thêm **một forward pass** tại t − dt:
  `u_tgt = (dt·v + (t − dt − r)·u_θ(x_t − dt·v, t − dt, r)) / (t − r)`, với dt = α(t − r).
  → **Chạy được trên XLA/TPU**, và đây là phương án thay JVP đã ghi trong bảng rủi ro. Curriculum đi từ trajectory FM (α nhỏ) sang MeanFlow.

### torchcfm (`torchcfm/optimal_transport.py`)

- `OTPlanSampler` dùng `pot.emd` (chính xác) hoặc `pot.sinkhorn` trên minibatch, rồi **lấy mẫu cặp từ plan có hoàn lại**. Thư viện có sẵn `SchrodingerBridgeConditionalFlowMatcher`, tức OT entropic cộng nhiễu Brown: gần với biến thể "nhiễu khởi đầu" của mình.

---

## 3. Hệ quả cho plan

1. **Backbone.** Mọi codec một bước kể trên đều dùng UNet cỡ khoảng 0.9B (SD-1.5, SD-2.1 hoặc SD-Turbo) với LoRA, và AEIC chỉ train trên ≤ 4×3090. SD3.5-medium (2.5B) nặng hơn mọi đối thủ. Với compute chưa chắc chắn (Kaggle/TRC), có hai hướng:
   - Dùng **SANA-0.6B hoặc 1.6B** (flow matching, khớp với lý thuyết), hoặc
   - Dùng **SD-Turbo** như StableCodec/AEIC. Hướng này dễ so sánh công bằng, nhưng SD-Turbo không phải flow nên phần lý thuyết phải "diễn giải lại".
   → **Cần bạn quyết định.**
2. **"Một model cho mọi bitrate" là điểm khác biệt thật.** StableCodec và AEIC cần một checkpoint cho mỗi λ; OSCAR cần một hyper-encoder cho mỗi rate cùng bảng timestep cứng.
3. **Chưa đối thủ nào dùng dither, và chưa ai liên tục hóa rate.** Tuy vậy, nhớ rủi ro "dither tốn rate" từ toy (mục Rủi ro trong plan).
4. **Chất lượng một bước của đối thủ đến phần lớn từ GAN (DINOv2) hoặc DMD**, không từ cấu trúc bridge. S3 chỉ với "LPIPS/DISTS nhẹ" khó thắng về FID/DISTS. Nên dự trù một GAN vision-aided trong S3 (chi phí thêm khoảng một backbone DINOv2-S).
5. **Kiến trúc điều kiện.** StableCodec/AEIC đưa feature giàu thông tin (256–320 kênh) vào `conv_in`. Bridge của mình nên nối (concat) cả z̄ lẫn feature từ ŷ, như plan đã ghi "concat kênh".
6. **Giao thức eval:** dùng lại `StableCodec/src/evaluate.py`, gồm pyiqa cùng `neuralcompression.update_patch_fid` và torchmetrics FID/KID. Lấy đường baseline từ `results.txt`. Nếu so với số của OSCAR phải ghi chú, vì họ resize ảnh và dùng FID full-res.
7. **Nhận xét quan trọng về coupling, rút ra khi đối chiếu với toy B.**
   - Ở số chiều cao, minibatch OT giữa X\* và X gần như trùng với **coupling tự nhiên**. Trong toy d = 64, chi phí ghép cặp của minibatch OT (≈ 2.85) ≈ D\* (2.78), vì X\*_i gần X_i hơn mọi X_j khác.
   - Hệ quả: với ảnh thật, minibatch OT có thể **suy biến thành coupling tự nhiên**, và theo Mệnh đề 2, khi đó một bước Euler không làm gì cả.
   - Đây là lý do mạnh để dùng **tham số hóa flow-map (MeanFlow/AlphaFlow)** thay vì Euler một bước, và/hoặc **nhiễu khởi đầu** (kết quả A5–A7), thay vì trông cậy vào minibatch OT.
   - Nên đưa nhận xét này vào paper, như một phân tích vì sao OT-CFM không giúp gì trong nén ảnh.
8. **Tránh 2 lỗi mà đối thủ mắc phải:** chuẩn hóa bằng thông tin phía encoder mà không truyền đi (OSCAR); và eval trên ảnh đã resize.

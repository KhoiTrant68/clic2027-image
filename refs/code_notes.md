# Ghi chú đọc code các codec tham khảo (27/9/2026)

Các repo được clone shallow vào `refs/repos/`. Đường dẫn file trong ghi chú này tính từ thư mục đó.

| Paper | Repo | Trạng thái |
|---|---|---|
| OSCAR (NeurIPS'25) | [jp-guo/OSCAR](https://github.com/jp-guo/OSCAR) | Đã đọc model, train, test |
| StableCodec (ICCV'25) | [LuizScarlet/StableCodec](https://github.com/LuizScarlet/StableCodec) | Đã đọc model, codec, loss, eval, kết quả báo cáo |
| AEIC (CVPR'26) | [LuizScarlet/AEIC](https://github.com/LuizScarlet/AEIC) | Đọc lướt: cùng họ với StableCodec |
| DiffO (WACV'26) | [Freemasti/DiffO](https://github.com/Freemasti/DiffO) | Đọc lướt: dựa trên ResShift |
| OneDC (NeurIPS'25) | [onedc-codec/onedc](https://github.com/onedc-codec/onedc) | Đọc lướt cấu trúc |
| FlowCodec, VoRTeC | chưa thấy code công bố | — |

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

### OneDC

- SD-1.5 + entropy model kiểu DCVC + tokenizer MaskGIT-VQGAN để distill semantic vào hyperprior. Giai đoạn 2 dùng **DMD** (`src/modules/dmd/sd_guidance.py`).


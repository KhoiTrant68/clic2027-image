# Đo sai số lượng tử trên latent SD3.5 (go/no-go 1)

`measure_quant_noise.py` gồm hai bước:
- `encode` chạy trên GPU, TPU hoặc CPU, và lưu latent đã chuẩn hóa vào `<out>/latents/`.
- `analyze` chạy trên CPU và tạo ra `summary.md`, `quant_noise_stats.json` và `quant_noise.png`.

`runs/synthetic/` là kết quả chạy trên latent tổng hợp, **chỉ dùng để kiểm tra pipeline**. Không dùng các số trong đó để kết luận.

## Kaggle (GPU T4/P100, bật Internet)

Mặc định dùng DC-AE của SANA (`Efficient-Large-Model/Sana_1600M_1024px_diffusers`, thư mục `vae`), repo này không bị gated nên không cần token. Chỉ khi thử VAE của SD3.5 (bị gated) mới cần thêm `HF_TOKEN` vào Kaggle Secrets.

```bash
%%bash
pip install -q diffusers transformers accelerate
cp /kaggle/input/<dataset>/measure_quant_noise.py /kaggle/working/ && cd /kaggle/working
python measure_quant_noise.py encode  --images /kaggle/input/<kodak-dataset> --out runs/kodak --device cuda
python measure_quant_noise.py analyze --out runs/kodak
```

Nên chạy thêm CLIC2020 hoặc DIV2K-val, với `--max_side 1024`. Phần `analyze` cho khoảng 100 ảnh 2K mất vài phút đến vài chục phút trên CPU.

## Go/no-go 1 (a): denoiser dùng mô hình nhiễu chính xác so với giả định Gaussian (`denoiser_gonogo.py`)

Cần latent của một tập **train riêng** (không được là tập test), ví dụ DIV2K train 800 ảnh, cùng latent của các tập test. Chạy trên Kaggle GPU:

```bash
%%bash
pip install -q diffusers transformers accelerate
cp /kaggle/input/<dataset>/*.py /kaggle/working/ && cd /kaggle/working
python measure_quant_noise.py encode --images /kaggle/input/<div2k-train> --out runs/div2k --device cuda
python measure_quant_noise.py encode --images /kaggle/input/<kodak>       --out runs/kodak --device cuda
python measure_quant_noise.py encode --images /kaggle/input/<clic2020>    --out runs/clic  --device cuda
python denoiser_gonogo.py --train runs/div2k --test runs/kodak runs/clic --out runs/gonogo1 --device cuda
```

- Script train 3 denoiser (`exact`, `gaussian`, `nodither`), mỗi cái 20k bước. Mạng CNN khoảng 2.4M tham số, train trên crop latent 16×16. Thời gian chưa đo; ước chừng dưới 1 giờ trên T4.
- Kết quả nằm trong `runs/gonogo1/summary.md`: bảng MSE latent theo bpp, và **PASS/FAIL** cho từng tập test. Tiêu chí PASS: `exact` tốt hơn `gaussian` ít nhất 5% ở ít nhất 2/4 mức bpp.
- `runs/gonogo1_smoke/` là kết quả chạy thử trên latent tổng hợp với 300 bước. **Không dùng số trong đó.**

## Đọc kết quả

- **nd / d / g:** không dither, có dither, và tham chiếu Gaussian (giả định "nhiễu diffusion").
- **Lưu ý:** ở ≤ 0.05 bpp, 95–99% chỉ số lượng tử bằng 0, nên sai số không dither ≈ −y và mặc nhiên "không Gaussian". Các cột đáng xem hơn:
  - kurtosis và tương quan kênh trong *latent domain*;
  - `content ρ`, độ phụ thuộc nội dung của năng lượng sai số;
  - MSE latent của nd so với d. Bước lượng tử lớn làm dither tốn distortion; xem mục rủi ro trong plan.

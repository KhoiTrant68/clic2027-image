# Thử nghiệm ELIC trên CLIC30

Đánh giá checkpoint MSE có sẵn của ELIC trên cùng 30 ảnh và cùng giao thức với [thử nghiệm MLIC++](../mlicpp/README.md), để so trực tiếp hai codec. Không huấn luyện lại.

## Nguồn

- ELIC chính thức (He và cộng sự, CVPR 2022) không công bố mã/trọng số. Dùng bản tái lập không chính thức [VincentChandelier/ELiC-ReImplemetation](https://github.com/VincentChandelier/ELiC-ReImplemetation), pin commit `92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d`. **Repo không có tệp license**; chỉ dùng để đánh giá nội bộ.
- Năm checkpoint MSE λ = 0,004 / 0,008 / 0,016 / 0,032 / 0,15 (q1–q5) từ link Google Drive trong README của repo; q5 được chạy bổ sung sau (`elic/run_q5.sh` trên lab09, cùng giao thức). README còn có λ = 0,45, rate quá cao nên không dùng. SHA-256 được ghi trong `qN-encode.json`.
- Môi trường: Python 3.10, PyTorch `2.0.1+cu118`, CompressAI **`1.1.5`** (đúng bản tác giả ghi), timm 0.6.13, thop, ptflops; metric dùng lpips 0.1.4, piq 0.8.0 như MLIC++.

## Chạy trên lab09 (GPU 6, mọi thứ trong `/mnt/data/khoitm8/clic2027`)

```bash
source /mnt/data/khoitm8/env.sh
cd /mnt/data/khoitm8/clic2027
ELIC_ENV=$PWD/envs/elic CUDA_VISIBLE_DEVICES=6 bash clic2027-image/experiments/elic/run_server.sh \
  --upstream refs/ELiC-ReImplemetation --weights elic/weights --data data/valid30 \
  --out elic/full30 --qualities 1 2 3 4
envs/elic/bin/python clic2027-image/experiments/elic/compare.py --out elic/full30
```

- Dữ liệu tạo bằng `experiments/mlicpp/prepare_data.py` từ ZIP chính thức; hash 30 ảnh trùng với lần chạy MLIC++.
- Giao thức giống MLIC++: encode và decode ở hai tiến trình riêng, decoder chỉ đọc stream; padding zero đến bội số 64; fp32, tắt TF32, deterministic, `TORCH_CUDNN_V8_API_DISABLED=1`.
- Stream (`stream.py`): header 12 byte + 11 payload (1 chuỗi z, 5 slice × anchor/non-anchor của y), độ dài mã varint. Mọi byte đều tính vào rate.
- GPU 6 dùng chung, nên wrapper bật `AEIC_SHARED_SMALL=1`: allocator 8 GiB, tự dừng khi tiến trình vượt 10 GiB.
- `compare.py` đọc thêm `results/2026-11-02_mlicpp/full30` để tính BD-rate ELIC so với MLIC++ (đường 4 checkpoint, khoảng chồng lấn), phân bổ theo ngân sách, và mix khi thêm ELIC. Số MLIC++ đo trên A30, ELIC đo trên H100 dùng chung: **không so trực tiếp thời gian giải mã giữa hai lần chạy**.

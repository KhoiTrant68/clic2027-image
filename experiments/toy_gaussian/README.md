# Toy Gaussian — kiểm chứng Mệnh đề 1–2, Định lý 3–4

| Script | Nội dung | Chạy ở đâu |
|---|---|---|
| `toy_dp.py` | Vòng 1: toy tách được (KLT), d = 2 và d = 64, 4 coupling | CPU (đã chạy, xem `results/`) |
| `plot_dp.py` | Hình và `results/summary.md` cho vòng 1 | CPU |
| `toy_ab.py` | Vòng 2, thí nghiệm A (capacity / preconditioning) và B (kích thước pool của minibatch OT) | **A6000** |
| `toy_corr2d.py` | Vòng 2, thí nghiệm C: toy 2D không tách được, dùng cho Hình 3 | CPU (`results_C/`) |

## Chạy A và B trên máy 8×A6000 (GPU 6–7)

Cần `uv` (https://docs.astral.sh/uv/). Trên Linux, gói torch từ PyPI đã kèm CUDA. Copy cả thư mục `toy_gaussian/` sang máy đó, rồi chạy:

```bash
cd toy_gaussian
CUDA_VISIBLE_DEVICES=6 nohup uv run toy_ab.py --exp A --device cuda --out results_A > results_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup uv run toy_ab.py --exp B --device cuda --out results_B > results_B.log 2>&1 &
```

Chạy thử nhanh trước (khoảng 1 phút), để chắc môi trường ổn:

```bash
uv run toy_ab.py --exp B --device cuda --out smoke_B --steps_scale 0.01 --n_eval 2000 --only B_d64_sk16384,B_d64_exact_ot
```

Thời gian ước tính: A mất khoảng 30–60 phút (2 run có 24k bước). B gồm 14 run, mất khoảng 1–2 giờ; phần lớn thời gian nằm ở Sinkhorn với pool 16384 và ở bài toán gán trên CPU với pool 1024. Các con số này chưa được đo trên A6000.

Kết quả cần gửi lại: `results_A/summary.md`, `results_B/summary.md` (và các file `*.json` nếu tiện).

## Chạy A và B trên Kaggle (2×T4 hoặc P100)

1. Upload `toy_dp.py` và `toy_ab.py` thành một Kaggle Dataset, ví dụ tên `toy-gaussian`.
2. Tạo notebook mới, chọn Accelerator **GPU T4 x2** và Add Input là dataset ở bước 1. Không cần bật Internet, vì numpy, scipy và torch đã có sẵn.
3. Chạy một cell duy nhất:

```bash
%%bash
cp /kaggle/input/toy-gaussian/*.py /kaggle/working/ && cd /kaggle/working
CUDA_VISIBLE_DEVICES=0 python toy_ab.py --exp A --device cuda --out results_A > A.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python toy_ab.py --exp B --device cuda --out results_B > B.log 2>&1 &
wait
```

4. Chọn **Save Version → Save & Run All (Commit)** để notebook chạy nền, không cần giữ tab mở. Kết quả nằm trong tab Output.

T4 chậm hơn A6000 khoảng 3–5 lần. Nếu sợ vượt giới hạn 12 giờ mỗi session, chia B làm hai session: session đầu thêm `--only` với 7 run `B_d16_*`, session sau với 7 run `B_d64_*`.

## Đọc kết quả

- **A:** so `1-step P_lb` và nhóm `low_var` giữa A1, A2, A3 và A4.
  - A2 đưa P_lb xuống dưới khoảng 0.2: nguyên nhân là train chưa đủ.
  - Chỉ A3 hoặc A4 sửa được: vấn đề nằm ở conditioning, cần chuẩn hóa từng kênh cho ŷ/z̄.
  - Không run nào sửa được: map tất định về bản chất khó học, nên cần nhiễu khởi đầu σ(t₀)ε.
- **B:** cột `pair cost / W²` cho biết coupling minibatch còn cách OT thật bao xa (1.0 là OT thật). So nó với `1-step D/(D*+W²)` và `1-step P_lb` theo pool, ở d = 16 và d = 64.

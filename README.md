# ratflow: Rate-as-Time flow-bridge image codec

Repo này phục vụ hai mục tiêu:
- Bản nộp **CVPR 2027**, hạn 16/11/2026.
- Bài dự thi **CLIC 2027** (image, GPU track), hạn validation 01/03/2027.

Codec của CVPR chính là "mode A" trong bài dự thi CLIC.

| Thư mục | Nội dung |
|---|---|
| `plans/` | Kế hoạch CVPR và CLIC (nguồn chuẩn cho quyết định, mốc thời gian, rủi ro) |
| `src/ratflow/` | Package dùng chung. Hiện có `eval/metrics.py`: **mọi con số trong paper và trong phân tích CLIC đều phải đi qua module này** |
| `experiments/` | `toy_gaussian/`, `quant_noise/` (go/no-go 1), `ceiling/` (trần AE), `clic_b/` (nhánh B của CLIC), `determinism/` (A0) |
| `clic/` | Phần riêng của CLIC. Hiện có `l4/` (giả lập server trên L4). Sau này thêm `submission/`, `encoder/`, `tools/` |
| `notebooks/` | Notebook Kaggle **được sinh tự động** bằng `python notebooks/build.py`. Không sửa tay |
| `paper/` | `proofs/` (appendix LaTeX), `figures/` (script vẽ hình, style figures4papers) |
| `results/` | Kết quả chạy trên Kaggle. Chỉ commit `summary.md`, csv, json; zip, recon, npz và crops bị bỏ qua |
| `refs/` | `code_notes.md`, `README.md` (URL và commit của các repo đối thủ và devkit CLIC). Các bản clone nằm trong `refs/repos/`, không commit |
| `tests/` | `python tests/test_metrics.py` hoặc `pytest`, chạy trên CPU, không cần torch |

## Quy trình

1. Sửa code trong `src/` hoặc `experiments/`.
2. Chạy `python tests/test_metrics.py`.
3. Chạy `python notebooks/build.py`. Notebook ghi lại commit hash, nên **commit trước khi build**.
4. Upload notebook lên Kaggle và chạy.
5. Đặt zip kết quả vào `results/<exp>/`, giải nén rồi commit `summary.md` và các csv.

Máy local không chạy được torch (Windows Smart App Control chặn DLL), nên mọi thứ cần torch đều chạy trên Kaggle hoặc L4.

# ratflow: Rate-as-Time flow-bridge image codec

Repo này phục vụ hai mục tiêu:
- Bản nộp **CVPR 2027**, hạn 16/11/2026.
- Bài dự thi **CLIC 2027** (image, GPU track), hạn validation 01/03/2027.

Codec của CVPR chính là "mode A" trong bài dự thi CLIC.

| Thư mục | Nội dung |
|---|---|
| `plans/` | Kế hoạch CVPR và CLIC (nguồn chuẩn cho quyết định, mốc thời gian, rủi ro) |
| `src/ratflow/` | Package dùng chung:<br>• `eval/metrics.py`: **mọi con số trong paper và trong phân tích CLIC đều phải đi qua module này**<br>• `nn/`: DC-AE và SANA DiT viết lại bằng torch thuần, không cần diffusers, nạp được weights của diffusers (`.safetensors`) hoặc `.pt` (dùng cho decoder CLIC)<br>• `codec/latent_codec.py`: **S1**, codec đa rate trên latent DC-AE, có subtractive dither trong bitstream<br>• `entropy/`: rANS bằng numpy, dither tất định (`dither.py`), bảng CDF số nguyên, mạng số nguyên `QConv2d` (train bằng fake-quant, suy luận chính xác từng bit), `GaussianConditional`, `DiscretePrior` |
| `experiments/` | `s1/` (cache latent, train, eval S1), `parity/`, `toy_gaussian/`, `quant_noise/` (go/no-go 1), `ceiling/` (trần AE), `clic_b/` (nhánh B của CLIC), `determinism/` (A0) |
| `clic/` | Phần riêng của CLIC. Hiện có `l4/` (giả lập server trên L4). Sau này thêm `submission/`, `encoder/`, `tools/` |
| `notebooks/` | Notebook Kaggle **được sinh tự động** bằng `python notebooks/build.py`. Không sửa tay |
| `paper/` | `proofs/` (appendix LaTeX), `figures/` (script vẽ hình, style figures4papers) |
| `results/` | Kết quả chạy trên Kaggle. Chỉ commit `summary.md`, csv, json; zip, recon, npz và crops bị bỏ qua |
| `refs/` | `code_notes.md`, `README.md` (URL và commit của các repo đối thủ và devkit CLIC). Các bản clone nằm trong `refs/repos/`, không commit |
| `tests/` | `python tests/test_metrics.py`, `test_entropy.py`, `test_dither.py` (hoặc `pytest`), chạy trên CPU, không cần torch. Phần cần torch được kiểm bằng `notebooks/parity.ipynb` |

## Quy trình

**Mọi việc cần GPU đều chạy qua một script:** `experiments/pipeline.py`, đóng gói thành một file `dist/ratflow_run.py` bằng `python tools/bundle.py`.

1. Sửa code trong `src/` hoặc `experiments/`, chạy `python tests/test_*.py`, rồi commit.
2. `python tools/bundle.py` tạo `dist/ratflow_run.py` (và `dist/ratflow_run.ipynb`, notebook một cell dành cho Kaggle).
3. Chạy `python ratflow_run.py [stages] [--hours H]` trên Kaggle (bật GPU và Internet) hoặc trên máy GPU thuê.
   - Chạy lại thì tự tiếp tục. Trên Kaggle, gắn Output của lần trước làm Input.
   - `--dry-run` chỉ in ra các lệnh sẽ chạy.
4. Đặt `results.zip` vào `results/<ngày>/`, giải nén, rồi commit `summary.md` và các file csv/json.

`notebooks/` (sinh bởi `notebooks/build.py`) là cách cũ, chỉ giữ lại để đối chiếu.

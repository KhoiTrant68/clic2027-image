# ratflow: Rate-as-Time flow-bridge image codec

Repo này phục vụ hai mục tiêu:
- Bản nộp **CVPR 2027**, hạn 16/11/2026.
- Bài dự thi **CLIC 2027** (image, GPU track), hạn validation 01/03/2027.

Codec của CVPR chính là "mode A" trong bài dự thi CLIC.

| Thư mục | Nội dung |
|---|---|
| `plans/` | Kế hoạch CVPR và CLIC (nguồn chuẩn cho quyết định, mốc thời gian, rủi ro) |
| `src/ratflow/` | Package dùng chung:<br>• `eval/metrics.py`: **mọi con số trong paper và trong phân tích CLIC đều phải đi qua module này**<br>• `nn/`: DC-AE và SANA DiT viết lại bằng torch thuần, không cần diffusers, nạp được weights của diffusers (`.safetensors`) hoặc `.pt` (dùng cho decoder CLIC)<br>• `codec/latent_codec.py`: **S1**, codec đa rate trên latent DC-AE, có subtractive dither trong bitstream<br>• `entropy/`: rANS bằng numpy, dither tất định (`dither.py`), bảng CDF số nguyên, mạng số nguyên `QConv2d` (train bằng fake-quant, suy luận chính xác từng bit), `GaussianConditional`, `DiscretePrior` |
| `experiments/` | `s1/` (cache latent, train, eval S1), `parity/`, `toy_gaussian/`, `quant_noise/` (go/no-go 1), `ceiling/` (trần AE), `clic_b/` (nhánh B của CLIC), `determinism/` (A0), `qhat/` (Q̂: thước đo thay người chấm, fit trên dữ liệu chấm theo cặp của CLIC), `bakeoff/` (so sánh base C0–C3 trên 30 ảnh validation, chia bit theo Q̂) |
| `clic/` | Phần riêng của CLIC. Hiện có `l4/` (giả lập server trên L4). Sau này thêm `submission/`, `encoder/`, `tools/` |
| `paper/` | `proofs/` (appendix LaTeX), `figures/` (script vẽ hình matplotlib theo style figures4papers; `figures/drawio/` chứa sơ đồ kiến trúc, pipeline, huấn luyện, lộ trình và codec C1, sinh bằng `make_drawio.py` / `make_codec_c1.py`) |
| `results/` | Kết quả chạy trên Kaggle. Chỉ commit `summary.md`, csv, json; zip, recon, npz và crops bị bỏ qua |
| `refs/` | `code_notes.md`, `README.md` (URL và commit của các repo đối thủ và devkit CLIC). Các bản clone nằm trong `refs/repos/`, không commit |
| `tests/` | `python tests/test_metrics.py`, `test_entropy.py`, `test_dither.py` (hoặc `pytest`), chạy trên CPU, không cần torch. Phần cần torch được kiểm bằng stage `parity` của pipeline |

## Quy trình

**Mọi việc cần GPU đều chạy qua một script:** `experiments/pipeline.py`, đóng gói thành một file `ratflow_run.py` bằng `python tools/bundle.py`.

1. Sửa code trong `src/` hoặc `experiments/`, chạy `python tests/test_*.py`, rồi commit.
2. `python tools/bundle.py` ghi `ratflow_run.py`, `clic_qhat_bakeoff.ipynb` (notebook Kaggle) và `run_clic.sh` (máy GPU thuê) ra **ngoài repo**, vào `../ratflow-artifacts/run/` (đổi bằng `--out`).
3. Chạy `python ratflow_run.py [stages] [--hours H]` trên Kaggle (bật GPU và Internet) hoặc trên máy GPU thuê.
   - Chạy lại thì tự tiếp tục. Trên Kaggle, gắn Output của lần trước làm Input.
   - `--dry-run` chỉ in ra các lệnh sẽ chạy.
4. Đặt `results.zip` vào `results/<ngày>/`, giải nén, rồi commit `summary.md` và các file csv/json.


## CLIC: Q̂ và so sánh base (plan v2, tuần 1–3)

Một lần chạy trên Kaggle (GPU T4 + Internet), tự tiếp tục khi chạy lại:

```
python ratflow_run.py qhat bakeoff --hours 11.5
```

- **Input cần gắn:** checkpoint S1 lần 1 (`s1_out/last.pt`, λ 0.03–4, phủ 0.02–0.12 bpp). Lần 2 (`s1/last.pt`) chỉ tới 0.067 bpp nên không dùng.
- `qhat`: lấy mẫu khoảng 16k câu hỏi từ dữ liệu chấm CLIC 2021/2022/2024, đọc crop thẳng từ zip trên mạng (không tải 46–135 GiB), fit Bradley–Terry. Ra `qhat/summary.md` và `qhat/qhat_v0.json`.
- `bakeoff`: build VTM 23.8, tạo điểm vận hành cho VTM 4:2:0, VTM-SCC, DC-AE+S1+residual VTM, MS-ILLM, mbt2018-mean; chia bit cho cả bộ ảnh theo Q̂ (quy hoạch động, đúng ngân sách byte). Ra `bakeoff/summary.md` và `bakeoff/crops/*.jpg`.
- Chỉ chạy một phần: `--bakeoff-args="--cands msillm mbt"`; đổi cỡ mẫu Q̂: `--qhat-args="--n 2024t=8000"`.

## Dữ liệu và artifact lớn (ngoài repo)

`../ratflow-artifacts/` chứa những thứ không đưa vào git:
- `data/`: ảnh DIV2K và latent.
- `results/`: các zip từ Kaggle, ảnh recon, npz, và **checkpoint S1**:
  - `results/s1/s1_out/last.pt`: lần 1, λ 0.03–4, dùng cho bake-off CLIC.
  - `results/2026-10-03/s1/last.pt`: lần 2, λ 0.015–0.6.
- `run/`: output của `tools/bundle.py`.

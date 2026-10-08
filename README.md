# ratflow-codec: bài dự thi CLIC 2027

Mục tiêu duy nhất: **thắng track Image GPU của CLIC 2027** (https://clic2027.compression.cc/). Hạn nộp decoder vòng validation: 01/03/2027.
Repo không phục vụ paper nào.

| Thư mục | Nội dung |
|---|---|
| `plans/` | Kế hoạch CLIC: `Plan CLIC 2027 v2 - tap trung de thang.md` (lịch và quyết định hiện hành), `Plan CLIC 2027.md` (luật thi, phần kỹ thuật A1–A6 và nhánh B) |
| `src/ratflow/` | Package dùng chung:<br>• `eval/metrics.py`: **mọi con số trong phân tích CLIC đều đi qua module này**<br>• `nn/`: DC-AE (và SANA DiT) viết lại bằng torch thuần, không cần diffusers (server không có diffusers)<br>• `codec/latent_codec.py`: S1, codec đa rate trên latent DC-AE (ứng viên C1)<br>• `entropy/`: rANS bằng numpy, bảng CDF số nguyên, mạng số nguyên `QConv2d` (khớp từng bit giữa các máy), `GaussianConditional`, `DiscretePrior` |
| `experiments/` | `qhat/` (Q̂: thước đo thay người chấm, fit trên dữ liệu chấm theo cặp của CLIC), `bakeoff/` (so sánh các base trên 30 ảnh validation, chia bit theo Q̂; `merge_sessions.py` gộp nhiều session), `clic_b/` (trần AE, VTM/VTM-SCC trên ảnh màn hình), `determinism/` (float vs số nguyên giữa các máy), `parity/` (torch thuần vs diffusers), `s1/` (train/eval S1) |
| `clic/` | `l4/`: dựng môi trường giống server trên máy L4 thuê. Sau này thêm `submission/`, `encoder/`, `tools/` |
| `reports/` | Báo cáo theo ngày (tình trạng, rà soát codec diffusion) |
| `results/` | Kết quả chạy trên Kaggle theo ngày. Chỉ commit `summary.md`, csv, json; zip, recon, npz và crops bị bỏ qua |
| `refs/` | `code_notes.md` (đọc code các codec đối thủ), `README.md` (URL và commit của các repo tham khảo và devkit CLIC). Bản clone nằm trong `refs/repos/`, không commit |
| `tools/` | `bundle.py`: đóng gói code thành một file `ratflow_run.py`; `clic_notebook.py`: notebook Kaggle `clic_qhat_bakeoff.ipynb` và `run_clic.sh` (được `bundle.py` gọi) |
| `tests/` | `python tests/test_metrics.py`, `test_entropy.py`, `test_dither.py` (hoặc `pytest`), chạy trên CPU, không cần torch |

## Quy trình

**Mọi việc cần GPU đều chạy qua một script:** `experiments/pipeline.py`, đóng gói thành một file `ratflow_run.py` bằng `python tools/bundle.py`.

1. Sửa code trong `src/` hoặc `experiments/`, chạy `python tests/test_*.py`, rồi commit.
2. `python tools/bundle.py` ghi `ratflow_run.py`, `ratflow_run.ipynb`, `clic_qhat_bakeoff.ipynb` (notebook Kaggle) và `run_clic.sh` (máy GPU thuê) ra **ngoài repo**, vào `../ratflow-artifacts/run/` (đổi bằng `--out`).
3. Chạy `python ratflow_run.py [stages] [--hours H]` trên Kaggle (bật GPU và Internet) hoặc trên máy GPU thuê.
   - Không ghi stage thì chạy mặc định `check → qhat → bakeoff → pack`. Các stage `data`, `cache`, `s1`, `s1_eval`, `parity` là tùy chọn (chỉ dùng cho ứng viên S1).
   - Chạy lại thì tự tiếp tục. Trên Kaggle, gắn Output của lần trước làm Input.
   - `--dry-run` chỉ in ra các lệnh sẽ chạy.
4. Giải nén `results.zip` vào `results/<ngày>_<tên>/`, commit `summary.md` và các file csv/json, rồi chuyển zip gốc sang `../ratflow-artifacts/results/` (git bỏ qua mọi file `.zip`).


## Q̂ và so sánh base (bake-off)

Một lần chạy trên Kaggle (GPU T4 + Internet), tự tiếp tục khi chạy lại:

```
python ratflow_run.py qhat bakeoff --hours 11.5
```

- **Q̂ đã có sẵn:** Q̂ v0 đã fit (`results/2026-10-06_qhat_v0/`). Để khỏi fit lại (~1.9 giờ), gắn thư mục đó làm Kaggle Input rồi chạy `python ratflow_run.py bakeoff ...`; pipeline tự tìm `qhat_v0.json` trong work dir và các input.
  **Luôn kiểm tra dòng `bakeoff: ... | Q-hat <đường dẫn>` trong log:** nếu là `None`, bộ chia bit lặng lẽ tối ưu theo −LPIPS thay vì Q̂.
- **Checkpoint S1 lần 1** (`s1_out/last.pt`, λ 0.03–4, phủ 0.02–0.12 bpp) chỉ cần khi chạy ứng viên `s1res`. Lần 2 (`s1/last.pt`) chỉ tới 0.067 bpp nên không dùng. Ứng viên này đã thua xa trong bake-off 08/10.
- `qhat`: lấy mẫu khoảng 16k câu hỏi từ dữ liệu chấm CLIC 2021/2022/2024, đọc crop thẳng từ zip trên mạng (không tải 46–135 GiB), fit Bradley–Terry. Ra `qhat/summary.md` và `qhat/qhat_v0.json`.
- `bakeoff`: build VTM 23.8, tạo điểm vận hành cho VTM 4:2:0, VTM-SCC, DC-AE+S1+residual VTM, MS-ILLM, mbt2018-mean, CoD-Lite, refiner SD-Turbo; chia bit cho cả bộ ảnh theo Q̂ (quy hoạch động, đúng ngân sách byte). Ra `bakeoff/summary.md` và `bakeoff/crops/*.jpg`.
- Chỉ chạy một phần: `--bakeoff-args="--cands msillm mbt"`; đổi cỡ mẫu Q̂: `--qhat-args="--n 2024t=8000"`.
- VTM trên phiên chỉ có CPU (không tốn quota GPU): `python ratflow_run.py bakeoff --bakeoff-step points --bakeoff-args="--cands vtmscc --only screen texture"`. `--only` nhận nhãn (`screen`, `texture`, `natural`) hoặc tiền tố tên ảnh; `--qps` đổi lưới QP.
- Gộp nhiều session (mỗi session chạy một phần ứng viên): `python experiments/bakeoff/merge_sessions.py --out results/<ngày>_bakeoff_merged --qhat results/2026-10-06_qhat_v0/qhat_v0.json results/<session>/bakeoff ...`

## Dữ liệu và artifact lớn (ngoài repo)

`../ratflow-artifacts/` chứa những thứ không đưa vào git:
- (ảnh DIV2K và latent không giữ lại: pipeline tự tải và cache lại khi cần)
- `results/`: các zip từ Kaggle, ảnh recon của nhánh B, và **checkpoint S1** (không có bản sao nào khác, nên sao lưu):
  - `results/s1/s1_out/last.pt`: lần 1, λ 0.03–4, dùng cho bake-off CLIC.
  - `results/2026-10-03/s1/last.pt`: lần 2, λ 0.015–0.6.
- `run/`: output của `tools/bundle.py`.

# Tình hình repo CLIC 2027 (Image, GPU track), ngày 14/10/2026

Nguồn: các file `results/*`, `reports/*`, `plans/Plan CLIC 2027 v2 - tap trung de thang.md` và git log của nhánh
`claude/what-should-i-do-now-f6f4dd`. Mọi con số được chép từ các file kết quả. Phần ước lượng Elo là suy luận và được ghi rõ.

## Tóm tắt

- **Đã chọn được hướng chính:**
  - **0.075 bpp:** DiT-IC cho ảnh tự nhiên.
  - **0.15 và 0.3 bpp:** MS-ILLM, trộn với DiT-IC theo từng ảnh.
  - **Ảnh chữ và màn hình:** VTM-SCC.

  Phương án này thắng mọi ứng viên khác trong bake-off. Trong lần chấm mù, DiT-IC được chọn 69% so với MS-ILLM.
- **Chưa đủ để thắng.** So với đội nhất GPU năm 2025, ước lượng thô:
  - ở 0.075 mình gần bằng, thiếu khoảng 30–55 Elo;
  - ở 0.15 thiếu khoảng **95 Elo**;
  - ở 0.3 thiếu khoảng 40 Elo.
- **Chưa có gì nộp được.** Ba việc còn thiếu:
  - decoder chưa chạy được trong môi trường CLIC (không có diffusers và compressai);
  - chưa đo trên L4;
  - chưa kiểm chứng bitstream giải đúng từng bit giữa hai máy L4.
- **Rủi ro lớn nhất là tốc độ trên L4.** Trên L40 của server, cả decoder mất 17.5 s cho 30 ảnh với decoder gốc và 11.4 s với
  decoder chưng cất. Suy từ thông số, L4 chậm khoảng 3 lần, tức khoảng 50 s và 33 s. Đội nhất 2025 mất 24 s, và
  25% bài chậm nhất mất quyền nhận giải.
- **Code chưa được sao lưu.** Toàn bộ việc từ 09/10 đã merge vào `main` trên máy (14/10) nhưng chưa push lên GitHub.
  Checkpoint của decoder chưng cất chỉ nằm trên server.

---

## 1. Đối chiếu với kế hoạch v2

Năm câu hỏi của tháng 10, theo mục 3 của kế hoạch:

| # | Câu hỏi | Hạn | Trạng thái |
|---|---|---|---|
| Q1 | Base nào cho Q̂ cao nhất ở từng mức? | 26/10 | ✅ Gần xong: DiT-IC ở 0.075, MS-ILLM (trộn DiT-IC) ở 0.15/0.3. Còn thiếu số tốc độ L4 |
| Q2 | Q̂ dự đoán người chấm tốt đến đâu? | 12/10 | ✅ Q̂ v0: 0.812 khi thử trên năm khác. Đã kiểm bằng chấm mù. Q̂ v1 bị loại |
| Q3 | Các đội đầu 2024–2025 đã làm gì? | 12/10 | ⛔ Bị chặn: OpenReview bắt qua bước xác minh chống bot. Mới đọc được bảng xếp hạng và slide |
| Q4 | Mỗi ứng viên giải mã nhanh đến đâu trên L4? | 26/10 | 🟡 Kit đo đã sẵn và chạy thử trên L40. **Chưa thuê L4** |
| Q5 | Server CLIC tính byte và chạy decoder ra sao? | khi server mở | ⬜ Chưa nộp thử lần nào |

**Thay đổi so với kế hoạch:**
- **Compute:** Kaggle không còn là nơi chạy chính. Từ 09/10, mọi thứ chạy trên server 2 × L40 (`ssh guess`).
- **Phạm vi model:** kế hoạch v2 chỉ nhắm tới các codec C0–C4. Nay đã thêm CoD-Lite, Cool-chic, DiT-IC và CRDR.

## 2. Các ứng viên đã thử

| Ứng viên | Kết quả chính | Trạng thái |
|---|---|---|
| VTM 4:2:0, VTM-SCC | Mốc chuẩn. SCC tốt nhất cho ảnh màn hình | Giữ VTM-SCC làm mode cho ảnh chữ/màn hình |
| S1: DC-AE + S1 + residual VTM (C1) | Thua MS-ILLM 1–2.3 Q̂ ở cả 3 mức, giải mã 9 s/ảnh trên T4 | Loại (06/10) |
| mbt2018-mean | Q̂ −2.0 ở 0.15 | Loại |
| SD-Turbo refiner trên MS-ILLM | Kém hơn MS-ILLM | Loại (08/10) |
| CoD-Lite (pixel-space, một bước) | 0.075: −0.38 (MS-ILLM −0.45). 0.15/0.3: kém MS-ILLM. Nhanh nhất: 0.15 s/ảnh | Không làm base. Thỉnh thoảng được chọn khi trộn |
| Cool-chic (tag `clic2025`, decoder C trên CPU) | Kém MS-ILLM khoảng 0.4–0.6 Q̂. Encode khoảng 1.5 giờ/ảnh/λ | Tạm dừng theo yêu cầu của bạn (09/10) |
| CRDR | Kém MS-ILLM theo Q̂ v0, giải mã 5 s/ảnh | Loại (10/10) |
| **MS-ILLM** | Base mạnh nhất ở 0.15 và 0.3 | **Chọn cho 0.15/0.3.** Trọng số giấy phép CC-BY-NC |
| **DiT-IC** (SANA-600M + DC-AE + codec kiểu ELIC) | 0.075: Q̂ +0.164, hơn MS-ILLM 0.62 | **Chọn cho ảnh tự nhiên ở 0.075.** Làm hỏng chữ |
| DiT-IC với decoder chưng cất | Kém decoder gốc khoảng 0.16 Q̂, nhanh hơn 1.5 lần | Dự phòng nếu L4 quá chậm |

## 3. Kết quả chính

### 3.1 Q̂: thước đo thay người chấm

- **Q̂ v0** là mô hình Bradley–Terry trên các thước đo LPIPS, DISTS, PSNR, MS-SSIM…, fit trên dữ liệu chấm theo cặp của
  CLIC 2021/2022/2024.
  - Dự đoán đúng 0.812 khi thử trên năm không dùng để fit.
  - Mức trần thực tế là 72–78%: đó là tỷ lệ người chấm tự đồng ý với chính mình khi gặp lại cùng câu hỏi.
  - PSNR đơn lẻ chỉ đúng 57%.
- **Q̂ v1** (thêm Wasserstein Distortion) có điểm cao hơn (0.833), nhưng đặt VTM ngang CoD-Lite ở 0.075, trái với Elo 2025.
  Lần chấm mù cũng bác nó. **Bỏ v1, dùng v0.**
- **Quy đổi:** 1.0 Q̂ ≈ 174 Elo.

### 3.2 Bake-off ngày 13/10

Tập: 30 ảnh validation (cũng là bộ test của 2025). Mỗi ứng viên được chia bit cho cả bộ ảnh theo Q̂ v0, đúng ngân sách byte.
File: `results/2026-10-13_bakeoff_v0`.

| ứng viên | Q̂ ở 0.075 | Q̂ ở 0.15 | Q̂ ở 0.3 | giải mã (s/ảnh, L40) |
|---|---|---|---|---|
| VTM 4:2:0 | −2.515 | −1.163 | −0.152 | |
| VTM-SCC | −2.404 | −1.102 | −0.004 | |
| MS-ILLM | −0.451 | +0.545 | +1.383 | 0.7–0.8 |
| CoD-Lite | −0.382 | +0.285 | +0.906 | 0.15 |
| DiT-IC | +0.164 | (chỉ tới 0.096 bpp) | | 0.55 |
| DiT-IC chưng cất | −0.000 | | | 0.36 |
| **trộn theo ảnh** | **+0.184** | **+0.758** | **+1.421** | |

**Khi trộn, bộ chia bit chọn:**

| rate | DiT-IC | MS-ILLM | VTM-SCC | CoD-Lite |
|---|---|---|---|---|
| 0.075 | 29 ảnh (q2: 4, q3: 24, q4: 1) | | 1 | |
| 0.15 | 16 ảnh (q3: 10, q4: 6) | 11 | 2 | 1 |
| 0.3 | 3 ảnh (q4) | 24 | 2 | 1 |

DiT-IC chỉ có điểm vận hành tới khoảng 0.10 bpp. Ở 0.15, nó tiết kiệm bit ở các ảnh nó nhận để chuyển sang các ảnh của MS-ILLM.

### 3.3 Chấm mù (11/10)

Chi tiết trong `results/2026-10-11_rating_ditic_msillm`.

- **Thiết kế:** 90 cặp crop 512×512 ở 0.075 bpp, DiT-IC so với MS-ILLM, thêm 10 cặp lặp lại có đảo bên.
- **Người chấm:** 1 người, là bạn.
- **Kết quả chung:** DiT-IC được chọn 62/90 lần, tức **69%** (khoảng tin cậy 59–78%), tương đương khoảng **+138 Elo**.
  - Ảnh tự nhiên: 73%.
  - Ảnh chữ/màn hình: chỉ 4/12. DiT-IC thua ở đây, nên cần mode khác cho loại ảnh này.
- **Đối chiếu với Q̂ v0:** Q̂ v0 dự đoán +111 Elo, và khớp đa số phiếu ở 26/30 ảnh.
- **Độ nhất quán:** 7/10 cặp lặp lại được chọn giống lần đầu.

### 3.4 Tốc độ decoder DiT-IC

| bước | 30 ảnh | ghi chú |
|---|---|---|
| Bản gốc, fp32 | ~42 s (1.4 s/ảnh) | L40 |
| Trọng số bf16, RMSNorm NCHW, chia tile, cắt latent theo ảnh | 2.94 → 1.10 s/ảnh | L40 lúc bận |
| Giải cả khung hình, gom latent theo bội số 16 | 91 → 58.5 s | L40 lúc bận; đo như CLIC: mỗi ảnh giải một lần |
| **Bản hiện tại, decoder gốc** | **17.5 s** (0.57 s/ảnh) | L40 rảnh, môi trường giống CLIC (torch 2.6) |
| **Bản hiện tại, decoder chưng cất** | **11.4 s** (0.37 s/ảnh) | như trên |
| Nạp mạng (chưa tính ở trên) | ~18 s | Dựng từ checkpoint fp32 4.3 GB. Bản nộp sẽ gói sẵn trọng số bf16 |

- **Đã thử, không có lợi:** channels_last, `cudnn.benchmark`, fp16, tách up-conv thành sub-pixel (đúng nhưng chậm hơn).
- **`torch.compile`:** nhanh hơn 1.45 lần. Muốn dùng cho bài nộp phải build bằng AOTInductor trên torch 2.6, kiến trúc sm_89.
  Chưa làm.

### 3.5 Chưng cất decoder DC-AE

Chi tiết trong `results/2026-10-11_ditic_distill.md`.

- **Cấu trúc:** decoder trò có ba tầng độ phân giải cao còn một nửa số kênh. Phần đầu được chép từ decoder gốc và đóng băng.
- **Huấn luyện:** 20k bước L1 + LPIPS, sau đó 30k bước có thêm GAN (trọng số 0.1).
- **Chất lượng:** kém decoder gốc 0.13 / 0.17 / 0.16 Q̂ ở q2 / q3 / q4, tức khoảng −29 Elo. Vẫn hơn MS-ILLM khoảng 0.45 Q̂.
- **Lưu trữ:** checkpoint `~/clic/distill/students/{q2,q3,q4}.pt` **chỉ có trên server**.

### 3.6 Tất định của entropy coding

Chi tiết trong `results/2026-10-12_ditic_determinism`.

| hai bên | precision | kết quả |
|---|---|---|
| CPU ↔ GPU | fp32 | 15–60% chỉ số lệch → ảnh hỏng |
| CPU ↔ GPU | fp64 | khớp từng bit, nhưng phần entropy mất 1.5 s/ảnh |
| GPU 0 ↔ GPU 1 (cùng máy, cùng L40) | fp32 | khớp từng bit, 0.14 s/ảnh |
| L40 → L40, môi trường giống CLIC (kit L4, 13/10) | fp32 | khớp từng bit trên cả 30 ảnh |

- **Phạm vi:** vấn đề này áp dụng cho mọi codec có hyperprior số thực, **kể cả MS-ILLM**.
- **Hướng A:** nén trên đúng loại máy và phần mềm như decoder của CLIC, tức L4 với torch 2.6 và cuDNN deterministic.
  Chưa kiểm chứng giữa hai máy L4.
- **Hướng B (dự phòng):** chuỗi tham số entropy bằng số nguyên int8. Phải fine-tune có lượng tử hoá, ước khoảng một tuần.

## 4. Vị trí ước lượng so với 2025

Lấy VTM 2025 làm mốc ở từng rate, cộng chênh Q̂ v0 × 174. Đây là suy luận, chỉ đúng về cỡ độ lớn.

| rate | phương án | Elo ước lượng | GPU nhất 2025 | còn thiếu |
|---|---|---|---|---|
| 0.075 | trộn (DiT-IC) | ~1875 theo Q̂, ~1900 theo chấm mù | 1929 (Vcoder) | ~30–55 |
| 0.15 | trộn (DiT-IC + MS-ILLM) | ~2030 | 2127 | **~95** |
| 0.3 | trộn (chủ yếu MS-ILLM) | ~2230 | 2269 | ~40 |

- **Cách xếp hạng chung:** cộng thứ hạng ở từng rate. Năm 2025, nhóm đầu ở 0.15 và 0.3 chỉ cách nhau khoảng 20 Elo.
  Với mức thiếu trên, mình nằm ngoài top ở hai rate này.
- **Đây chỉ là mức sàn.** Các đội 2027 cũng sẽ mạnh lên: năm 2025 không có bài diffusion vì luật tốc độ, năm 2027 thì có thể có.
- **Giới hạn của ước lượng:**
  - Q̂ v0 chỉ đúng khoảng 81%.
  - Phiếu chấm mù mới có một người.
  - Bộ test 2027 là ảnh mới, không phải 30 ảnh này.

## 5. Còn thiếu để nộp được bài

| việc | trạng thái | ghi chú |
|---|---|---|
| Decoder chạy trong môi trường CLIC (torch 2.6, **không có** diffusers, compressai, timm, peft) | 🟡 Có nền | `src/clic27/nn` đã có DC-AE và SANA bằng torch thuần. Còn thiếu: phần codec của DiT-IC, prompter, bước scheduler, rANS (hoặc chuyển sang `range-coder` 1.1 có sẵn), MS-ILLM. Đã có thẻ đề xuất session riêng |
| Decoder ≤ 4 GB | ⚠️ Đang vượt | Mỗi mức q của DiT-IC khoảng 1.8 GB ở bf16, và bộ chia bit dùng cả q2, q3, q4 (khoảng 5.3 GB) cộng MS-ILLM. Phải dùng chung trọng số SANA, chỉ lưu phần chênh cho từng q, hoặc bớt số mức q |
| RAM 12 GB, 2 CPU | ⬜ Chưa đo | `l4_check.py speed` nay ghi RAM đỉnh và số CPU. File kit trên server chưa đóng gói lại với bản này |
| Tốc độ trên L4 | ⬜ Chưa đo | Kit `experiments/l4_kit` sẵn sàng. Thuê một L4 khoảng 30 phút, chi phí khoảng 1 USD. Giả lập L4 trên L40 bằng MPS đã thử và cho kết quả không tin được |
| Tất định giữa hai L4 | ⬜ Chưa đo | Cùng kit, cần lần thuê thứ hai trên một máy khác |
| Phát hiện ảnh chữ/màn hình để đổi mode | ⬜ Chưa làm | Bộ chia bit chọn mode theo Q̂, mà Q̂ v0 không bắt được lỗi chữ: ở 0.075 nó giao 29/30 ảnh cho DiT-IC, kể cả ảnh màn hình. Trong khi đó người chấm chỉ chọn DiT-IC 4/12 lần trên crop có chữ. Cần quy tắc riêng phía encoder |
| Định dạng bitstream, `check_submission.py`, nộp thử | ⬜ Chưa làm | Kế hoạch v2 hẹn tháng 11. Chưa biết server validation đã mở chưa |

## 6. Còn thiếu để thắng

1. **0.15 bpp (~95 Elo, đòn bẩy lớn nhất).** DiT-IC dừng ở khoảng 0.10 bpp. Cần thêm điểm vận hành 0.15–0.3, ví dụ
   fine-tune codec của DiT-IC với λ thấp hơn, bắt đầu từ q4, trên server. Đây là việc train, tính bằng tuần.
2. **0.3 bpp (~40 Elo).** Fine-tune MS-ILLM với GAN/LPIPS hướng theo Q̂, hoặc dùng DiT-IC sau khi đã mở rộng rate.
3. **Tốc độ.** Nếu L4 chậm như ước tính, cần decoder chưng cất, AOTInductor, hoặc chưng cất mạnh hơn.
   Mục tiêu trong kế hoạch: ≤ 20 s cho 30 ảnh.
4. **Thêm người chấm.** Một người chấm là quá ít trước mốc quyết định. Trang chấm mù đã có sẵn và chia sẻ được.
5. **Whitepaper 2025.** Cần bạn mở OpenReview và qua bước xác minh, để mình đọc cách các đội đầu làm.

## 7. Rủi ro

| rủi ro | mức | cách xử lý |
|---|---|---|
| DiT-IC quá chậm trên L4, rơi vào 25% chậm nhất | **cao** | Đo ngay. Dùng decoder chưng cất, AOTInductor. Nếu vẫn không đạt, quay về MS-ILLM ở 0.075 (mất khoảng 110 Elo) |
| Bitstream không giải đúng trên L4 của CLIC | cao | Đo giữa hai L4, nộp thử lên server validation. Dự phòng: entropy số nguyên (khoảng một tuần) |
| Decoder vượt 4 GB | trung bình | Dùng chung trọng số SANA cho các mức q, lưu phần chênh; hoặc lượng tử trọng số |
| Giấy phép: trọng số MS-ILLM là CC-BY-NC; giấy phép của SANA và DiT-IC chưa kiểm tra | thấp–trung bình | Kiểm tra giấy phép và luật CLIC về trọng số bên thứ ba trước khi nộp |
| Mất code hoặc checkpoint | trung bình | Push `main` lên GitHub, sao lưu `~/clic/distill/students` |
| Q̂ dẫn sai hướng | trung bình | Thêm người chấm. Đã thấy Q̂ v0 khớp người ở 26/30 ảnh |

## 8. Đề xuất cho 2 tuần tới (đến mốc 27/10)

1. **Thuê L4 lần 1 (khoảng 30 phút):** lấy số tốc độ thật của decoder gốc và decoder chưng cất, cùng kết quả kiểm L40 → L4.
   Số này quyết định DiT-IC còn dùng được không.
2. **Thuê L4 lần 2, trên máy khác:** kiểm bitstream giữa hai L4 (hướng A).
3. **Song song: đóng gói decoder DiT-IC chạy được trong môi trường CLIC.** Việc này cần làm dù chọn model nào, gồm cả bài
   toán dung lượng 4 GB.
4. **Thêm 2–3 người chấm** bằng trang chấm mù hiện có.
5. **Mốc 27/10:** chốt decoder (gốc hay chưng cất) và kế hoạch mở rộng DiT-IC lên 0.15.

## 9. Tình trạng repo

**Bản đồ code:**

| thư mục | nội dung |
|---|---|
| `src/clic27/` | `nn/`: DC-AE, SANA bằng torch thuần. `entropy/`: rANS numpy, bảng CDF số nguyên, mạng số nguyên. `eval/metrics.py`. `codec/`: S1 (đã loại) |
| `experiments/bakeoff/` | So sánh các ứng viên và chia bit theo Q̂. DiT-IC, CRDR, Cool-chic chạy trong tiến trình riêng |
| `experiments/qhat/` | Fit Q̂ v0/v1 trên dữ liệu chấm của CLIC |
| `experiments/rating/` | Trang chấm mù (artifact) và các cặp ảnh |
| `experiments/ditic_speed/`, `ditic_distill/` | Tăng tốc decoder DiT-IC, chưng cất decoder DC-AE |
| `experiments/determinism/` | Kiểm tất định entropy coding giữa các thiết bị |
| `experiments/l4_kit/` | Kit đo trên L4 thuê: dựng môi trường giống CLIC, đo tốc độ, kiểm bitstream giữa hai máy |
| `plans/`, `reports/`, `results/` | Kế hoạch, báo cáo, kết quả theo ngày |

**Việc dọn dẹp:**
- **Git:** ngày 14/10, nhánh `claude/what-should-i-do-now-f6f4dd` (từ Cool-chic `clic2025` tới báo cáo này) đã được merge
  fast-forward vào `main`. `origin/main` vẫn dừng ở bake-off 09/10, cần push.
- **Giả lập L4:** `experiments/l4_kit/l4_emu.py` và `emulate_l4.sh` (giả lập L4 trên L40 bằng MPS) được giữ lại để ghi nhận.
  Số hiệu chỉnh ghi ở đầu `emulate_l4.sh`. Không dùng số đo từ hai file này.
- **Tài liệu cũ:**
  - `README.md` vẫn mô tả quy trình Kaggle và chưa nhắc tới DiT-IC, server, chưng cất, kit L4.
  - `plans/Plan CLIC 2027 v2` vẫn ghi compute là Kaggle cộng L4.
- **Chỉ có trên server, chưa sao lưu:** checkpoint chưng cất, latent DIV2K dùng để chưng cất, các file streams.

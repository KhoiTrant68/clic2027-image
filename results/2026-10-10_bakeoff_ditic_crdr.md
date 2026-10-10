# Bake-off 10/10: thêm DiT-IC (C7) và CRDR (C8)

Chạy trên server L40, 30 ảnh validation. Code: `experiments/bakeoff/bakeoff.py` (`--cands ditic crdr`), mỗi codec chạy trong một
tiến trình Python riêng; bytes là bitstream rANS thật.
- `2026-10-10_server_ditic_crdr`: DiT-IC q2/q3/q4 (fp32, DC-AE chia tile 1024); CRDR q 0–3, β 3.84.
- `2026-10-10_server_ditic_bf16`: DiT-IC chạy lại với autocast bf16 để đo tốc độ.

Gộp với hai session ngày 09/10 (`server_full_gpu`, `server_full_vtm`), rồi chia bit lại:
`2026-10-10_bakeoff_v0` (Q̂ v0, mặc định) và `2026-10-10_bakeoff_v1` (Q̂ v1, chỉ tham khảo).
Ảnh cắt ở 0.075 bpp (gốc, MS-ILLM, CoD-Lite, DiT-IC): `2026-10-10_bakeoff_v0/crops/` (không commit).

## Kết quả (tất cả 30 ảnh)

| rate | ứng viên | Q̂ v0 | Q̂ v1 | bpp | PSNR | DISTS↓ | giải mã L40 (s/ảnh) |
|---|---|---|---|---|---|---|---|
| 0.075 | MS-ILLM | −0.45 | 2.15 | 0.075 | 25.2 | 0.094 | 0.8 |
| 0.075 | CoD-Lite | −0.38 | 1.85 | 0.075 | 22.0 | 0.093 | 0.15 |
| 0.075 | **DiT-IC** | **+0.19** | **2.19** | 0.075 | 22.4 | **0.068** | 1.4 |
| 0.075 | mix | +0.21 | 2.31 | 0.075 | | | |
| 0.15 | MS-ILLM | **+0.55** | **2.62** | 0.150 | 27.2 | 0.061 | 0.8 |
| 0.15 | DiT-IC (chỉ tới 0.096 bpp) | +0.31 | 2.26 | 0.096 | 22.7 | 0.063 | 1.4 |
| 0.15 | CRDR | +0.30 | 2.59 | 0.150 | 27.7 | 0.074 | 5.3 |
| 0.15 | mix | +0.78 | 2.78 | 0.150 | | | |
| 0.3 | MS-ILLM | **+1.38** | 3.17 | 0.300 | 29.4 | 0.036 | 0.7 |
| 0.3 | CRDR | +1.27 | **3.25** | 0.300 | 30.0 | 0.044 | 5.1 |
| 0.3 | mix | +1.46 | 3.31 | 0.300 | | | |

`mix` (mỗi ảnh chọn codec tốt nhất) giao cho DiT-IC:
- theo Q̂ v0: 28/30 ảnh ở 0.075, 17/30 ở 0.15, 4/30 ở 0.3;
- theo Q̂ v1: 22/30, 13/30, 3/30.

## DiT-IC

- **0.075 bpp: hơn MS-ILLM 0.64 Q̂ v0** (≈ 110 Elo theo quy đổi 1 Q̂ ≈ 174 Elo), gần đúng bằng khoảng cách ước lượng tới Vcoder (≈ 120 Elo).
  Q̂ v1 chỉ cho +0.04. v1 gần với PSNR (wd3 tương quan 0.96 với PSNR) nên phạt nặng PSNR thấp của DiT-IC. Hôm qua v1 đã trượt
  bước kiểm tra hợp lý với Elo 2025, nên tin v0 hơn. Dù vậy, hai bản lệch nhau đúng ở loại ảnh này, nên **phải tự chấm bằng mắt**.
- **Nhìn bằng mắt:** trên ảnh tự nhiên, DiT-IC nét nhất và giữ cấu trúc tốt nhất (lông thú, cành cây, chữ ký khắc trên tượng, mặt người).
  MS-ILLM bị nhòe, CoD-Lite làm méo mắt ở rate thấp. Màu và độ tương phản của DiT-IC đôi khi hơi lệch so với ảnh gốc.
- **Hỏng với chữ:** ảnh trang Wikipedia (`ebfd571f`) ở 0.041 bpp bị DiT-IC biến thành chữ vô nghĩa, trong khi MS-ILLM vẫn đọc được.
  Q̂ không bắt được lỗi này (Q̂ v0 trên 4 ảnh màn hình: DiT-IC 1.17, MS-ILLM 1.21). Ảnh có chữ phải đi qua VTM-SCC hoặc MS-ILLM.
- **PSNR thấp:** 22.4 dB ở 0.075, so với 24.0 dB của Vcoder, đội thắng 2025 vốn đã có PSNR thấp nhất. Phần lớn do trần của DC-AE f32
  (đo được 22.9 dB hôm 03/10).
- **Tốc độ là rủi ro lớn nhất.** Giải mã trung bình 1.4 s/ảnh trên L40 (fp32), tức ≈ 41 s cho 30 ảnh. L4 còn chậm hơn, trong khi đội top
  GPU 2025 chỉ mất 21–29 s cho cả bộ. Đo từng bước (lúc server bận, tổng ≈ 3 s/ảnh): entropy decode 0.08 s, DiT 0.34 s,
  **DC-AE decode 2.5 s (≈ 85%)**. bf16 không nhanh hơn (1.77 so với 1.40 s/ảnh), chất lượng như nhau (DISTS 0.0734 so với 0.0733).
- **Dung lượng decoder:** 3.56 GB ở fp32 (sát giới hạn 4 GB), 1.78 GB ở bf16. bf16 không làm giảm chất lượng, nên lưu weights bằng bf16.
- **Chỉ tới 0.1 bpp** (q4). Ở 0.15 bpp, `mix` = MS-ILLM + DiT-IC đạt +0.78, so với +0.55 của MS-ILLM: có thêm checkpoint ~0.2 bpp thì
  DiT-IC cũng có ích ở mức 0.15.

## CRDR

- Lần chạy đầu sai: CompressAI ≥ 1.2.5 đổi tên tham số của factorized prior (`_matrix0` → `matrices.0`). Hàm nạp của CRDR lặng lẽ bỏ
  qua 14 khóa này, nên ảnh demo trong README ra 0.2697 bpp thay vì 0.0641. Đã đổi tên khi nạp và nạp ở chế độ strict
  (demo ra 0.0638 bpp, khớp README). Các điểm sai đã xóa và chạy lại.
- Theo Q̂ v0, CRDR kém MS-ILLM ở cả 0.15 (+0.30 so với +0.55) và 0.3 (+1.27 so với +1.38). Theo v1, nhỉnh hơn một chút ở 0.3.
  Không xuống tới 0.075. Giải mã 5 s/ảnh vì hyperprior và context model chạy trên CPU. **Không theo tiếp.**

## Việc tiếp theo

1. **Tự chấm DiT-IC so với MS-ILLM ở 0.075** (ảnh cắt 512×512, có ảnh gốc, giống cách CLIC chấm), khoảng 100 cặp.
   Việc này quyết định có đổi base cho mức 0.075 hay không, vì Q̂ v0 và v1 lệch nhau.
   **Đã làm 11/10** (`2026-10-11_rating_ditic_msillm/`): DiT-IC thắng 69% (ảnh tự nhiên 73%), ≈ +138 Elo, khớp Q̂ v0.
2. **Chọn mode theo nội dung:** ảnh có chữ đi VTM-SCC hoặc MS-ILLM. Cần một bộ phát hiện chữ, có thể dùng OCR CER của nhánh B.
3. **Tăng tốc DC-AE decoder 3–5×:**
   - trước hết thử giải mã cả khung hình không chia tile ở bf16 (L4 có 24 GB) và channels_last;
   - nếu chưa đủ, chưng cất (distill) một decoder nhẹ hơn trên latent của DiT-IC.
4. Nếu DiT-IC qua được bước 1: train thêm checkpoint ~0.15–0.2 bpp cho mức 0.15.

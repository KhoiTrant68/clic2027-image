# Tự chấm mù DiT-IC so với MS-ILLM ở 0.075 bpp, 11/10/2026

Trang chấm: https://claude.ai/artifact/FWEg1xHKgwLDqwCzSuWZwr (nguồn `experiments/rating/`).
- **Bộ cặp:** 90 cặp từ 30 ảnh validation, mỗi ảnh 3 vùng cắt 512×512 ngẫu nhiên (bỏ vùng gần như phẳng).
  Cộng 10 cặp lặp lại có đổi bên.
- **Điểm so sánh:** mỗi codec dùng điểm vận hành do bộ chia bit theo Q̂ v0 chọn ở 0.075 bpp
  (`2026-10-10_bakeoff_v0/allocation.csv`), nên tổng byte cả bộ ảnh bằng nhau.
- **Cách chấm:** mù, A/B đổi ngẫu nhiên; có ảnh gốc để so; bắt buộc chọn một bên.
- **Người chấm:** 1 người (r1). Phiếu thô: `votes.json`.

## Kết quả

| nhóm | phiếu | DiT-IC thắng | 95% CI (Wilson) | Elo DiT-IC − MS-ILLM |
|---|---|---|---|---|
| tất cả | 90 | **62 (69%)** | 59–78% | **+138** (+61 … +215) |
| ảnh tự nhiên | 75 | 55 (73%) | 62–82% | +176 (+88 … +264) |
| ảnh màn hình / chữ | 12 | 4 (33%) | 14–61% | −120 (−318 … +77) |
| texture | 3 | 3 | | |

- **Độ ổn định:** cặp lặp lại chọn giống lần đầu 7/10. Người chấm CLIC lặp lại khớp 72–78% (đo trên dữ liệu chấm CLIC).
  Hai trong ba lần đổi ý rơi vào ảnh `50d13125`, nơi hai bản rất giống nhau.
- **Thiên lệch bên:** chọn bên A 57/100 lần, không có ý nghĩa thống kê.
- **Thời gian:** trung vị 6.5 s mỗi cặp. Số lần lật trung vị là 0, nên phần lớn có lẽ chấm ở chế độ xem cạnh nhau.

## So với Q̂

- **Q̂ v0 dự đoán đúng.** Nó cho DiT-IC hơn 0.64 điểm (≈ +111 Elo), người chấm cho ≈ +138 Elo.
  Q̂ v1 dự đoán +0.04 (≈ +7 Elo) là sai. **Giữ Q̂ v0 làm mục tiêu chia bit, bỏ v1.**
- **Theo từng ảnh:** dấu chênh Q̂ v0 trùng với lựa chọn đa số (≥ 2/3 phiếu) ở **26/30 ảnh**.
- **Ảnh DiT-IC thua:** ở những ảnh này bộ chia bit đã dồn bớt byte của DiT-IC sang ảnh khác (bpp DiT-IC/MS-ILLM
  trung bình 0.74, ở ảnh DiT-IC thắng là 1.39). Q̂ v0 cũng đã dự đoán thua ở phần lớn các ảnh đó:
  `50d13125` 0.034 so với 0.074 bpp, `2397c73f` 0.071 so với 0.130, cộng các ảnh chữ `ebfd571f` và `bb7344a2`.
- **Bốn ảnh lệch dấu:** `4ebc0e6c`, `a365e654`, `da52c4f6` (Q̂ nói DiT-IC hơn, người chấm 1/3) và `86127fbd`
  (ngược lại). Mỗi ảnh chỉ có 3 phiếu nên còn nhiễu.

## Kết luận

1. **Đổi base cho 0.075 bpp sang DiT-IC trên ảnh tự nhiên.** Mức lợi ≈ +140 Elo cỡ bằng khoảng cách tới Vcoder đã ước lượng
   (≈ 120 Elo). Elo của một người chấm ở đây và Elo của CLIC không cùng thang, nên không cộng thẳng được.
2. **Ảnh màn hình / chữ giữ MS-ILLM hoặc VTM-SCC.** DiT-IC thua 4/12, và làm hỏng chữ ở `ebfd571f`.
3. **Hạn chế:** mới có một người chấm, 90 phiếu. Nên mời thêm 2–3 người (quyền Contributor trên trang) trước mốc quyết định 27/10.
4. **Rủi ro lớn nhất bây giờ là tốc độ giải mã**, không còn là chất lượng: DC-AE decode chiếm ≈ 85% của 1.4 s/ảnh trên L40.

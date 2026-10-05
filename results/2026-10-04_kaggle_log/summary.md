# Kaggle 2026-10-04: `qhat bakeoff` (commit 1ebc81b, T4, --hours 11.5)

**Output bị mất:** Kaggle giết tiến trình ở mốc 12 giờ, giữa lúc đang tạo điểm VTM. Pipeline không kịp đóng gói, nên Kaggle không lưu Output. Phần dưới chép lại từ log người dùng gửi, và chưa có `qhat_v0.json` (chỉ có trọng số đã chuẩn hóa). Đã sửa ở commit 98a9928: pipeline dừng stage đang chạy khi hết `--hours`, vẫn đóng gói và thoát bình thường.

## Q̂ v0: dự đoán lựa chọn của người chấm CLIC (stage qhat: 3.5 giờ, 1.3 dòng/giây)

16218 câu hỏi có đủ thước đo: psnr, msssim, lpips_alex, lpips_vgg, dists, clipiqa.

Độ chính xác là tỉ lệ đoán đúng ảnh mà người chấm chọn.

| tập | n | sign_psnr | sign_msssim | sign_lpips_alex | sign_lpips_vgg | sign_dists | sign_clipiqa | **qhat_cv** |
|---|---|---|---|---|---|---|---|---|
| 2021t | 3000 | 0.510 | 0.504 | 0.752 | 0.733 | 0.746 | 0.641 | **0.756** |
| 2021v | 5218 | 0.573 | 0.612 | 0.736 | 0.744 | 0.751 | 0.596 | **0.757** |
| 2022t | 3000 | 0.640 | 0.708 | 0.779 | 0.778 | 0.775 | 0.670 | **0.798** |
| 2024t | 5000 | 0.702 | 0.754 | 0.811 | 0.805 | 0.800 | 0.654 | **0.838** |
| all | 16218 | 0.613 | 0.654 | 0.770 | 0.767 | 0.770 | 0.636 | **0.783** |

**Train trên 2021t, 2021v, 2022t, test trên 2024t: 0.813**, là con số đáng tin nhất cho CLIC 2027.

Tỉ lệ hai người chấm đồng ý với nhau (trần thực tế của mọi Q̂):
- 2021v: 0.777 (n = 757)
- 2021t: 0.720 (n = 5319)

Câu kiểm tra, tức tỉ lệ chọn đúng ảnh gốc:
- 2022t: 0.924 (n = 1303)
- 2024t: 0.970 (n = 1325)

Trọng số đã chuẩn hóa:

| thước đo | trọng số |
|---|---|
| psnr | -0.594 |
| msssim | +0.853 |
| lpips_alex | +0.259 |
| lpips_vgg | +0.217 |
| dists | +0.947 |
| clipiqa | +0.349 |

Bias (thiên lệch vị trí A/B): +0.098.

## Bake-off (đã dừng ở bước points)

- **msillm, mbt, s1res: FAILED.** `pip install compressai` đã hạ numpy 2 xuống 1.26, khiến các import sau đó hỏng (`numpy.dtype size changed`, `torch_geometric`). Đã sửa ở commit 98a9928: pin numpy/torch theo bản đang cài.
- **VTM: 211/420 điểm** (vtm420 đủ 7 QP × 30 ảnh, vtmscc mới bắt đầu). Mỗi lần encode mất 2–8 phút trên 4 CPU. Các điểm này mất theo Output.

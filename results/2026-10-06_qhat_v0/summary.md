# Q̂ v0: dự đoán lựa chọn của người chấm CLIC

8000 câu hỏi có đủ thước đo; thước đo: psnr, msssim, lpips_alex, lpips_vgg, dists, clipiqa.

Độ chính xác (tỉ lệ đoán đúng ảnh được chọn). `sign_*`: chỉ dựa vào dấu hiệu số của một thước đo; `qhat_cv`: hồi quy logistic trên mọi thước đo, kiểm định chéo 5 phần.

| tập | n | sign_psnr | sign_msssim | sign_lpips_alex | sign_lpips_vgg | sign_dists | sign_clipiqa | **qhat_cv** |
|---|---|---|---|---|---|---|---|---|
| 2021t | 1500 | 0.504 | 0.497 | 0.750 | 0.728 | 0.744 | 0.633 | **0.753** |
| 2021v | 2000 | 0.583 | 0.624 | 0.738 | 0.753 | 0.753 | 0.592 | **0.770** |
| 2022t | 1500 | 0.642 | 0.713 | 0.768 | 0.772 | 0.769 | 0.673 | **0.795** |
| 2024t | 3000 | 0.705 | 0.761 | 0.807 | 0.808 | 0.801 | 0.650 | **0.837** |
| all | 8000 | 0.625 | 0.668 | 0.772 | 0.773 | 0.773 | 0.636 | **0.788** |

**Train trên 2021t, 2021v, 2022t, test trên 2024t: 0.812** (con số đáng tin nhất cho CLIC 2027).

| tập | số câu | chọn đúng ảnh gốc (câu kiểm tra) | câu hỏi lặp lại: tỉ lệ hai người đồng ý |
|---|---|---|---|
| 2021v | 5220 | nan (n=0) | 0.777 (n=757) |
| 2021t | 122107 | nan (n=0) | 0.720 (n=5319) |
| 2022t | 57300 | 0.924 (n=1303) | nan (n=0) |
| 2024t | 24806 | 0.970 (n=1325) | 0.000 (n=1) |

Tỉ lệ đồng ý giữa hai người chấm là trần thực tế của mọi Q̂.

Trọng số chuẩn hóa (độ lớn so sánh được giữa các thước đo):

| thước đo | trọng số |
|---|---|
| psnr | -0.572 |
| msssim | +0.920 |
| lpips_alex | +0.256 |
| lpips_vgg | +0.265 |
| dists | +0.916 |
| clipiqa | +0.341 |

Bias (thiên lệch vị trí A/B): +0.096. Model: `qhat_v0.json`.

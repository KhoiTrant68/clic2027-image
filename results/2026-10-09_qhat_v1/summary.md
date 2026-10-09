# Q̂ (qhat_v1): dự đoán lựa chọn của người chấm CLIC

8000 câu hỏi có đủ thước đo; thước đo: msssim, wd3.

Độ chính xác (tỉ lệ đoán đúng ảnh được chọn). `sign_*`: chỉ dựa vào dấu hiệu số của một thước đo; `qhat_cv`: hồi quy logistic trên mọi thước đo, kiểm định chéo 5 phần.

| tập | n | sign_msssim | sign_wd3 | **qhat_cv** |
|---|---|---|---|---|
| 2021t | 1500 | 0.497 | 0.749 | **0.665** |
| 2021v | 2000 | 0.624 | 0.779 | **0.753** |
| 2022t | 1500 | 0.713 | 0.790 | **0.789** |
| 2024t | 3000 | 0.761 | 0.825 | **0.840** |
| all | 8000 | 0.668 | 0.793 | **0.790** |

**Train trên 2021t, 2021v, 2022t, test trên 2024t: 0.833** (con số đáng tin nhất cho CLIC 2027).

Cùng phép thử nhưng bỏ hai đặc trưng Wasserstein (bộ đặc trưng v0): 0.752. Chỉ dùng dấu của một đặc trưng: wd3 0.825.

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
| msssim | +0.835 |
| wd3 | +1.656 |

Bias (thiên lệch vị trí A/B): +0.018. Model: `qhat_v1.json`.

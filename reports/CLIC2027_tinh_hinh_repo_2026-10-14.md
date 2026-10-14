# Báo cáo nghiên cứu tiến độ CLIC 2027 — Image GPU Track

**Ngày chốt:** 14/10/2026  
**Phạm vi:** lựa chọn codec, chất lượng cảm nhận, độ tin cậy của bitstream, tốc độ giải mã và mức độ sẵn sàng nộp bài.  
**Nguồn nội bộ:** các artifact trong **results/**, **reports/**, **plans/Plan CLIC 2027 v2 - tap trung de thang.md** và Git log của nhánh **claude/what-should-i-do-now-f6f4dd**.  
**Trạng thái bằng chứng:** báo cáo tổng hợp các kết quả đã ghi trong repo; không có nghĩa là toàn bộ thí nghiệm đã được chạy lại độc lập.

## Tóm tắt điều hành

### Quyết định kỹ thuật hiện tại

| Mức bitrate | Hướng đang chọn | Vai trò | Điểm cần lưu ý |
|---|---|---|---|
| 0.075 bpp | **DiT-IC** cho ảnh tự nhiên | Codec chính | Chất lượng chữ/màn hình kém; cần bộ chọn mode riêng |
| 0.15 bpp | **Trộn theo ảnh: DiT-IC + MS-ILLM**, thêm mode dự phòng | Codec chính | DiT-IC hiện chỉ có operating point đến khoảng 0.10 bpp |
| 0.3 bpp | **Trộn theo ảnh, chủ yếu MS-ILLM** | Codec chính | Cần cải thiện thêm chất lượng và xác minh chi phí giải mã |
| Ảnh chữ / giao diện / màn hình | **VTM-SCC** | Mode chuyên biệt | Bộ phát hiện ảnh chữ chưa được triển khai |

Bake-off nội bộ ngày 13/10 cho thấy phương án trộn theo ảnh đạt Q̂ v0 cao nhất trong các phương án được báo cáo ở cả ba mức bitrate. Trong phép chấm mù ngày 11/10, DiT-IC được chọn 62/90 lần so với MS-ILLM trên crop 512×512 ở 0.075 bpp. Đây là tín hiệu thuận lợi, nhưng phép chấm mù hiện chỉ có **một người chấm** và bake-off 30 ảnh chưa đại diện cho tập test CLIC 2027.

### Mức độ sẵn sàng

**Hệ thống chưa sẵn sàng nộp.** Các chặn chính:

1. Chưa chứng minh decoder chạy được trong môi trường CLIC tối giản.
2. Chưa đo tốc độ trên L4 thật.
3. Chưa kiểm tra bitstream giữa hai máy L4 độc lập.
4. Decoder DiT-IC đang vượt ngân sách trọng số 4 GB nếu lưu riêng các mức q2/q3/q4.
5. Chưa có bộ phát hiện ảnh chữ/màn hình để chọn mode phù hợp.
6. Các thay đổi mới nhất và checkpoint decoder chưng cất chưa được sao lưu đầy đủ.

**Rủi ro quyết định nhất là tốc độ và tính tái lập của giải mã trên L4.** Trên L40, decoder DiT-IC hiện tại mất 17.5 giây cho 30 ảnh ở bản gốc và 11.4 giây ở bản chưng cất, chưa tính khoảng 18 giây nạp mạng trong phép đo được ghi nhận. Quy đổi sang L4 chưa được xác minh và không nên dùng làm số liệu chính thức.

## 1. Quy ước về bằng chứng

- **Đã đo / đã quan sát:** số liệu có ghi trong artifact hoặc báo cáo thí nghiệm nội bộ.
- **Ước lượng:** giá trị suy ra từ Q̂, kết quả năm trước hoặc mô hình quy đổi phần cứng.
- **Chưa xác minh:** chưa chạy trên đúng phần cứng, môi trường hoặc tập dữ liệu đích.
- **Giả thuyết / đề xuất:** hướng kỹ thuật cần được kiểm nghiệm.

Q̂ và Elo trong báo cáo là chỉ số nội bộ. Chúng không phải điểm chính thức của CLIC và không bảo đảm thứ hạng trên tập test 2027.

## 2. Mục tiêu và pipeline nghiên cứu

Mục tiêu không chỉ là tìm codec có điểm cảm nhận cao, mà là tìm cấu hình đồng thời thỏa mãn chất lượng, ngân sách bitrate, giới hạn tài nguyên, tính tất định của bitstream và yêu cầu đóng gói của CLIC.

~~~mermaid
flowchart TD
    A["Tập ảnh validation"] --> B["Chạy các codec ứng viên"]
    B --> C["Phân bổ byte theo từng bitrate"]
    C --> D["Giải mã và tính metric"]
    D --> E["Q̂ v0: ước lượng preference"]
    E --> F["Bake-off 0.075 / 0.15 / 0.3 bpp"]
    F --> G["Chọn codec theo ảnh và theo rate"]
    G --> H["Chấm mù với người"]
    H --> I["Kiểm tra chéo Q̂ và preference"]
    G --> J["Benchmark decoder"]
    J --> K["Đo L4, RAM, CPU, thời gian nạp"]
    K --> L["Kiểm tra bitstream giữa hai máy"]
    L --> M["Đóng gói môi trường CLIC"]
    M --> N["Submission smoke test"]
~~~

### Pipeline triển khai và giải mã

~~~mermaid
flowchart LR
    A["Bitstream + metadata"] --> B["Đọc header / xác định mode"]
    B --> C{"Loại codec"}
    C -->|DiT-IC| D["Entropy decode latent"]
    D --> E["DC-AE decoder / SANA pipeline"]
    C -->|MS-ILLM| F["Entropy decode + neural synthesis"]
    C -->|Ảnh chữ / màn hình| G["VTM-SCC decoder"]
    E --> H["Ảnh RGB đầu ra"]
    F --> H
    G --> H
    H --> I["Kiểm tra tên file, kích thước, số lượng"]
~~~

Sơ đồ thứ hai mô tả luồng logic ở mức hệ thống; nó **không khẳng định** mọi nhánh đã được tích hợp hoàn chỉnh vào một decoder duy nhất. Tích hợp DiT-IC, MS-ILLM và VTM-SCC vào môi trường submission vẫn là hạng mục phải hoàn thành.

## 3. Đối chiếu kế hoạch tháng 10

| ID | Câu hỏi nghiên cứu / kỹ thuật | Hạn kế hoạch | Trạng thái ngày 14/10 | Bằng chứng còn thiếu |
|---|---|---:|---|---|
| Q1 | Codec nào có Q̂ cao nhất ở từng rate? | 26/10 | **Gần hoàn thành** | Tốc độ L4 và kiểm tra trên tập ảnh lớn hơn |
| Q2 | Q̂ dự đoán preference của người tốt đến đâu? | 12/10 | **Có kết quả sơ bộ** | Nhiều người chấm và tập đánh giá lớn hơn |
| Q3 | Các đội đầu 2024–2025 đã làm gì? | 12/10 | **Bị chặn một phần** | Đọc đầy đủ paper/whitepaper; OpenReview yêu cầu xác minh |
| Q4 | Decoder chạy nhanh đến đâu trên L4? | 26/10 | **Chưa đo trên L4 thật** | Benchmark trên L4 bằng kit hiện có |
| Q5 | Server CLIC đóng gói và tính byte thế nào? | Khi server mở | **Chưa thử submission** | Chạy checker và submission validation thực tế |

Thay đổi vận hành so với kế hoạch v2: từ 09/10, server có 2×L40 trở thành môi trường chạy chính thay cho Kaggle. Phạm vi ứng viên mở rộng ngoài C0–C4, gồm CoD-Lite, Cool-chic, DiT-IC và CRDR.

## 4. Phương pháp đánh giá

### 4.1 Q̂ v0 và Q̂ v1

Q̂ v0 là mô hình preference Bradley–Terry được fit từ kết quả chấm theo cặp của CLIC 2021/2022/2024, sử dụng các đặc trưng chất lượng như LPIPS, DISTS, PSNR và MS-SSIM cùng các metric liên quan.

| Phép đánh giá | Kết quả được báo cáo | Diễn giải |
|---|---:|---|
| Q̂ v0, thử trên năm không dùng để fit | 0.812 accuracy | Khả năng dự đoán preference sơ bộ; cần xác minh lại từ artifact gốc |
| PSNR đơn lẻ | 0.57 accuracy | Yếu hơn Q̂ v0 trong phép so sánh nội bộ |
| Mức đồng thuận tự lặp của người chấm | 72–78% | Trần tham khảo được báo cáo, không phải hằng số phổ quát |
| Q̂ v1, có Wasserstein Distortion | 0.833 accuracy | Điểm cao hơn nhưng không phù hợp với một số đối chiếu ngoài mẫu và kết quả chấm mù |
| Quy đổi Q̂ sang Elo | 1.0 Q̂ ≈ 174 Elo | Quy đổi nội bộ, không phải chuẩn chính thức |

**Quyết định hiện tại:** dùng Q̂ v0 để chọn codec; không dùng Q̂ v1 làm tiêu chí chính. Q̂ v1 bị loại vì tạo thứ tự codec đáng ngờ ở 0.075 bpp và không được kết quả chấm mù ủng hộ.

**Giới hạn:** accuracy 0.812 không đồng nghĩa điểm Q̂ tuyệt đối được hiệu chuẩn chính xác cho tập test 2027. Kết quả 26/30 ảnh khớp đa số phiếu trong phép chấm mù cũng là mẫu nhỏ.

### 4.2 Thiết lập bake-off

- **Ngày:** 13/10/2026.
- **Tập:** 30 ảnh validation, được báo cáo là cùng bộ ảnh test dùng trong CLIC 2025.
- **Bitrate mục tiêu:** 0.075, 0.15 và 0.3 bpp.
- **Phân bổ byte:** trên toàn bộ tập ảnh theo Q̂ v0.
- **Artifact:** **results/2026-10-13_bakeoff_v0**.

Cần giữ nguyên cấu hình, commit, file Q̂, quy tắc phân bổ byte và đầu ra đã giải mã để có thể tái lập phép đo.

## 5. Kết quả chất lượng

### 5.1 Bake-off ngày 13/10

| Ứng viên | Q̂ v0 @ 0.075 bpp | Q̂ v0 @ 0.15 bpp | Q̂ v0 @ 0.3 bpp | Tốc độ giải mã trên L40 |
|---|---:|---:|---:|---:|
| VTM 4:2:0 | −2.515 | −1.163 | −0.152 | Chưa có số tổng hợp |
| VTM-SCC | −2.404 | −1.102 | −0.004 | Chưa có số tổng hợp |
| MS-ILLM | −0.451 | +0.545 | +1.383 | 0.7–0.8 s/ảnh |
| CoD-Lite | −0.382 | +0.285 | +0.906 | 0.15 s/ảnh |
| DiT-IC | +0.164 | Chỉ có operating point tới khoảng 0.096 bpp | — | 0.55 s/ảnh |
| DiT-IC chưng cất | −0.000 | — | — | 0.36 s/ảnh |
| **Trộn theo ảnh** | **+0.184** | **+0.758** | **+1.421** | Phụ thuộc mode từng ảnh |

Đây là số liệu nội bộ được ghi trong repo, chưa phải kết quả chính thức của CLIC. Không so sánh trực tiếp các điểm từ những phiên bản Q̂ khác nhau.

### 5.2 Phân bổ codec khi trộn theo ảnh

| Bitrate | DiT-IC | MS-ILLM | VTM-SCC | CoD-Lite |
|---|---|---|---|---|
| 0.075 bpp | 29 ảnh (q2: 4, q3: 24, q4: 1) | — | 1 ảnh | — |
| 0.15 bpp | 16 ảnh (q3: 10, q4: 6) | 11 ảnh | 2 ảnh | 1 ảnh |
| 0.3 bpp | 3 ảnh (q4) | 24 ảnh | 2 ảnh | 1 ảnh |

DiT-IC hiện chỉ có operating point đến khoảng 0.10 bpp. Ở mức 0.15 bpp, phương án trộn dùng DiT-IC trên một phần ảnh và dành ngân sách bit còn lại cho các ảnh được mã hóa bằng codec khác.

### 5.3 Chấm mù DiT-IC so với MS-ILLM

| Thuộc tính | Thiết lập / kết quả |
|---|---|
| Ngày | 11/10/2026 |
| Artifact | **results/2026-10-11_rating_ditic_msillm** |
| Thiết lập | 90 cặp crop 512×512, 0.075 bpp; thêm 10 cặp lặp lại có đảo bên |
| Người chấm | 1 |
| DiT-IC được chọn | 62/90 = 68.9% (làm tròn 69%) |
| Khoảng tin cậy được báo cáo | 59–78% |
| Preference trên ảnh tự nhiên | 73% |
| Ảnh chữ/màn hình | DiT-IC được chọn 4/12 lần |
| Q̂ v0 dự đoán | +111 Elo; khớp đa số phiếu trên 26/30 ảnh |
| Độ nhất quán của 10 cặp lặp | 7/10 cặp |

Kết quả ủng hộ DiT-IC trên ảnh tự nhiên ở 0.075 bpp nhưng đồng thời cho thấy cần xử lý riêng ảnh chữ/màn hình. Vì chỉ có một người chấm và số crop chữ ít, cần xác nhận với thêm người chấm và nhiều ảnh hơn.

## 6. Các codec đã khảo sát

| Codec / ứng viên | Kết quả được ghi nhận | Quyết định hiện tại |
|---|---|---|
| VTM 4:2:0 | Baseline cổ điển | Giữ làm mốc so sánh |
| VTM-SCC | Tốt nhất trong các lựa chọn đã thử cho ảnh chữ/màn hình | Giữ làm mode chuyên biệt |
| S1: DC-AE + S1 + residual VTM | Thấp hơn MS-ILLM khoảng 1–2.3 Q̂ ở ba rate; khoảng 9 s/ảnh trên T4 | Loại ngày 06/10 |
| mbt2018-mean | Q̂ khoảng −2.0 ở 0.15 bpp | Loại |
| SD-Turbo refiner trên MS-ILLM | Kém MS-ILLM | Loại ngày 08/10 |
| CoD-Lite | −0.382 ở 0.075 bpp so với −0.451 của MS-ILLM; khoảng 0.15 s/ảnh; kém MS-ILLM ở 0.15/0.3 | Không dùng làm base; có thể được bộ trộn chọn |
| Cool-chic (tag clic2025, decoder C trên CPU) | Kém MS-ILLM khoảng 0.4–0.6 Q̂; encode khoảng 1.5 giờ/ảnh/λ | Tạm dừng ngày 09/10 |
| CRDR | Kém MS-ILLM theo Q̂ v0; khoảng 5 s/ảnh để giải mã | Loại ngày 10/10 |
| MS-ILLM | Base mạnh ở 0.15 và 0.3 bpp | Chọn; cần rà soát giấy phép trọng số CC-BY-NC |
| DiT-IC (SANA-600M + DC-AE + codec kiểu ELIC) | +0.164 ở 0.075 bpp, cao hơn MS-ILLM khoảng 0.62 Q̂; kém trên chữ/màn hình | Chọn cho ảnh tự nhiên ở 0.075 bpp |
| DiT-IC chưng cất | Kém decoder gốc khoảng 0.16 Q̂; nhanh hơn khoảng 1.5× trong phép đo được báo cáo | Dự phòng nếu tốc độ L4 không đạt |

## 7. Tốc độ, dung lượng và tính tất định

### 7.1 Tốc độ decoder DiT-IC trên L40

| Cấu hình / phép đo | Thời gian | Diễn giải |
|---|---:|---|
| Decoder gốc fp32 | Khoảng 42 s / 30 ảnh (1.4 s/ảnh) | Đo ban đầu trên L40 |
| Tối ưu bf16, RMSNorm NCHW, tile và cắt latent theo ảnh | 2.94 → 1.10 s/ảnh | Server đang bận |
| Giải toàn khung hình, gom latent theo bội số 16 | 91 → 58.5 s / 30 ảnh | Server đang bận; mỗi ảnh giải một lần |
| **Decoder gốc hiện tại** | **17.5 s / 30 ảnh (0.57 s/ảnh)** | L40 rảnh, môi trường gần CLIC, torch 2.6 |
| **Decoder chưng cất hiện tại** | **11.4 s / 30 ảnh (0.37 s/ảnh)** | Cùng điều kiện nêu trên |
| Nạp mạng | Khoảng 18 s | Không nằm trong thời gian trên; dựng từ checkpoint fp32 4.3 GB |

Các thử nghiệm channels_last, cudnn.benchmark, fp16 và tách up-convolution thành sub-pixel không đem lại lợi ích trong cấu hình đã thử. torch.compile nhanh hơn khoảng 1.45×, nhưng chưa tích hợp vào artifact nộp; hướng được đề xuất là build AOTInductor trên torch 2.6, kiến trúc sm_89.

**Chưa được kết luận decoder đạt mục tiêu tốc độ trên L4.** Các phép đo hiện có là trên L40. Mô phỏng L4 bằng MPS đã được ghi nhận là không đáng tin cậy.

### 7.2 Chưng cất decoder DC-AE

- **Artifact:** **results/2026-10-11_ditic_distill.md**.
- Kiến trúc student giảm một nửa số kênh ở ba tầng độ phân giải cao; phần đầu được sao chép từ decoder gốc và đóng băng.
- Huấn luyện 20k bước với L1 + LPIPS, sau đó 30k bước bổ sung GAN với trọng số 0.1.
- Chất lượng kém decoder gốc 0.13 / 0.17 / 0.16 Q̂ ở q2 / q3 / q4, tương đương khoảng −29 Elo theo quy đổi nội bộ.
- Vẫn cao hơn MS-ILLM khoảng 0.45 Q̂ trong phép so sánh được báo cáo.
- Checkpoint **~/clic/distill/students/{q2,q3,q4}.pt** hiện chỉ có trên server.

### 7.3 Tính tất định của entropy coding

| Thiết bị / precision | Kết quả | Ý nghĩa |
|---|---|---|
| CPU ↔ GPU, fp32 | 15–60% chỉ số lệch; ảnh hỏng | Không dùng cấu hình này |
| CPU ↔ GPU, fp64 | Khớp từng bit; phần entropy mất khoảng 1.5 s/ảnh | Đúng nhưng có chi phí lớn |
| GPU 0 ↔ GPU 1 trên cùng máy L40, fp32 | Khớp từng bit; 0.14 s/ảnh | Bằng chứng trong cùng máy |
| L40 ↔ L40, môi trường giống CLIC, kit L4 ngày 13/10, fp32 | Khớp từng bit trên 30 ảnh | Chưa phải kiểm tra L4 ↔ L4 |

Vấn đề có thể ảnh hưởng mọi codec dùng hyperprior số thực, bao gồm MS-ILLM. Hướng ưu tiên là kiểm tra giải mã trên đúng phần cứng và phần mềm của môi trường đích. **Hai máy L4 độc lập vẫn chưa được kiểm tra.** Phương án dự phòng là entropy coding số nguyên/int8 và fine-tune có lượng tử hóa; ước lượng công sức khoảng một tuần, chưa phải cam kết thời gian.

### 7.4 Ngân sách tài nguyên

| Ràng buộc | Trạng thái |
|---|---|
| Decoder ≤ 4 GB | **Chưa đạt / chưa chứng minh**: q2, q3, q4 của DiT-IC khoảng 1.8 GB mỗi mức ở bf16, tổng khoảng 5.3 GB trước khi cộng MS-ILLM |
| RAM 12 GB, 2 CPU | Chưa đo trong cấu hình submission cuối |
| L4 24 GB | Chưa benchmark trên L4 thật |
| Thời gian giải mã | Mục tiêu nội bộ ≤20 s / 30 ảnh; chưa xác nhận trên L4 |
| Tất định giữa hai máy L4 | Chưa kiểm tra |

Tài liệu CLIC nêu môi trường GPU track có 2 CPU, 12 GB RAM và L4 24 GB; đồng thời 25% submission chậm nhất có thể không được xét giải. Cần kiểm tra lại quy định chính thức và môi trường của đúng mùa thi trước khi đóng băng decoder: https://clic2027.compression.cc/faq/

## 8. Ước lượng khoảng cách so với kết quả 2025

Ước lượng nội bộ lấy VTM 2025 làm mốc theo từng rate và cộng chênh lệch Q̂ v0 × 174 Elo. Đây là quy đổi định hướng, không phải dự đoán thứ hạng đáng tin cậy.

| Rate | Phương án hiện tại | Elo ước lượng trong báo cáo | Mốc GPU nhất 2025 | Khoảng cách ước lượng |
|---|---|---:|---:|---:|
| 0.075 bpp | Trộn, chủ yếu DiT-IC | ~1875 theo Q̂; ~1900 theo chấm mù | 1929 (Vcoder) | Thiếu khoảng 30–55 |
| 0.15 bpp | DiT-IC + MS-ILLM | ~2030 | 2127 | **Thiếu khoảng 95** |
| 0.3 bpp | Trộn, chủ yếu MS-ILLM | ~2230 | 2269 | Thiếu khoảng 40 |

### Cách diễn giải

- Đây là **ước lượng thô**, không phải bảng xếp hạng CLIC 2027.
- Q̂ v0 có accuracy được báo cáo khoảng 0.812; chấm mù mới có một người; bake-off chỉ có 30 ảnh.
- Tập test 2027 sẽ khác bộ ảnh đã dùng.
- Các đội khác có thể cải thiện đáng kể so với năm 2025.
- Theo ước lượng hiện tại, 0.15 bpp là khoảng trống lớn nhất; đây là ưu tiên nghiên cứu về chất lượng sau khi giải quyết các rủi ro submission.

## 9. Việc cần làm trước khi nộp

| Ưu tiên | Việc | Tiêu chí hoàn thành | Trạng thái |
|---|---|---|---|
| **P0** | Push main lên GitHub | Remote chứa đầy đủ code và báo cáo đến 14/10 | Chưa hoàn thành theo ghi chú repo |
| **P0** | Sao lưu checkpoint chưng cất, latent DIV2K và streams | Có bản sao kiểm tra được ngoài server hiện tại | Chưa hoàn thành |
| **P0** | Chạy kit trên L4 thật | Có log thời gian giải mã, thời gian nạp, RAM đỉnh và phiên bản môi trường | Chưa chạy |
| **P0** | Kiểm tra bitstream L4 ↔ L4 | Encode trên máy A, decode trên máy B; pixel đầu ra và checksum khớp | Chưa chạy |
| **P0** | Decoder chạy trong môi trường CLIC tối giản | Không phụ thuộc diffusers, compressai, timm, peft; giải mã đủ file và đúng kích thước | Mới có nền, chưa tích hợp hoàn chỉnh |
| **P0** | Giải quyết ngân sách 4 GB | Tổng artifact decoder và trọng số trong giới hạn; có đo chính thức | Chưa đạt / chưa chứng minh |
| **P1** | Thêm bộ phát hiện ảnh chữ/màn hình | Mode VTM-SCC được chọn đúng trên validation; đo precision/recall và ảnh hưởng chất lượng | Chưa làm |
| **P1** | Mở rộng chấm mù | Thêm 2–3 người chấm, giữ cặp lặp và phân tích theo loại ảnh | Chưa làm |
| **P1** | Chạy checker và submission thử | Không thiếu file; tên, kích thước và định dạng đầu ra đúng | Chưa làm |
| **P1** | Rà soát giấy phép | Xác minh code, pretrained weights và quyền sử dụng trong cuộc thi | Chưa hoàn thành |
| **P2** | Mở rộng DiT-IC lên 0.15–0.3 bpp | Có operating point mới và bake-off công bằng ở cùng bitrate | Chưa làm |
| **P2** | Cải thiện 0.3 bpp | Thử fine-tune MS-ILLM theo GAN/LPIPS hoặc mục tiêu preference; xác nhận bằng chấm mù | Đề xuất |
| **P2** | Thử AOTInductor / chưng cất thêm | Cải thiện tốc độ thực trên L4 mà không làm giảm Q̂ quá mức | Chưa hoàn thành |
| **P2** | Đọc paper/whitepaper các đội đầu 2025 | Ghi lại phương pháp, bitrate, tốc độ và tính khả thi với luật 2027 | Bị chặn một phần bởi OpenReview |

### Kế hoạch hành động đến 27/10/2026

1. **L4 run #1:** chạy decoder gốc và chưng cất; đo end-to-end, memory và kiểm tra bitstream L40 → L4.
2. **L4 run #2:** dùng một máy L4 khác để xác nhận bitstream giữa hai máy.
3. **Đóng gói decoder DiT-IC:** loại bỏ phụ thuộc không có trong môi trường đích; xử lý chia sẻ trọng số SANA giữa các mức q hoặc giảm số mức q để đạt giới hạn 4 GB.
4. **Chạy thêm chấm mù:** mời 2–3 người chấm; phân tích riêng ảnh tự nhiên và ảnh chữ/màn hình.
5. **Cổng quyết định ngày 27/10:** chốt decoder gốc hay chưng cất và quyết định có tiếp tục mở rộng DiT-IC lên 0.15 bpp hay không.

Nếu bất kỳ kiểm tra P0 nào thất bại, ưu tiên chuyển sang phương án đơn giản hơn có thể giải mã tin cậy thay vì tối ưu thêm Q̂ trên một pipeline chưa thể nộp.

## 10. Rủi ro và phương án giảm thiểu

| Rủi ro | Mức | Giảm thiểu / phương án dự phòng |
|---|---|---|
| DiT-IC quá chậm trên L4, rơi vào nhóm 25% chậm nhất | **Cao** | Đo L4 ngay; thử decoder chưng cất và AOTInductor. Nếu không đạt, cân nhắc MS-ILLM ở 0.075 bpp, với tổn thất chất lượng ước khoảng 110 Elo theo ghi chú nội bộ |
| Bitstream sai giữa môi trường encode/decode | **Cao** | Kiểm tra hai máy L4; cố định môi trường. Dự phòng entropy coding số nguyên |
| Decoder vượt 4 GB | **Cao** | Chia sẻ trọng số SANA giữa các mức q, lưu phần delta hoặc giảm số mức q; xác nhận bằng artifact thực tế |
| Q̂ chọn sai loại ảnh | **Trung bình–cao** | Phát hiện ảnh chữ/màn hình; đánh giá bằng chấm mù theo nhóm ảnh |
| Q̂ không khái quát sang test 2027 | **Trung bình** | Thêm người chấm, tăng số ảnh, báo cáo độ bất định |
| Vi phạm giấy phép / điều kiện sử dụng pretrained weights | **Trung bình** | Rà soát giấy phép MS-ILLM, SANA, DiT-IC và quy định cuộc thi trước khi nộp |
| Mất code hoặc checkpoint | **Cao** | Push remote; sao lưu checkpoint, latent và streams; kiểm tra khả năng khôi phục |

## 11. Bản đồ code và trạng thái repo

| Đường dẫn | Vai trò |
|---|---|
| **src/clic27/nn/** | DC-AE và SANA triển khai bằng PyTorch thuần |
| **src/clic27/entropy/** | rANS bằng NumPy, bảng CDF số nguyên và mạng số nguyên |
| **src/clic27/eval/metrics.py** | Các metric đánh giá |
| **src/clic27/codec/** | Codec S1, hiện đã bị loại khỏi lựa chọn chính |
| **experiments/bakeoff/** | Chạy ứng viên, so sánh codec và phân bổ byte theo Q̂; một số codec chạy trong tiến trình riêng |
| **experiments/qhat/** | Fit và đánh giá Q̂ v0/v1 |
| **experiments/rating/** | Artifact trang chấm mù và các cặp ảnh |
| **experiments/ditic_speed/** | Thử nghiệm tối ưu tốc độ DiT-IC |
| **experiments/ditic_distill/** | Chưng cất decoder DC-AE |
| **experiments/determinism/** | Kiểm tra tính tất định entropy coding |
| **experiments/l4_kit/** | Kit benchmark trên L4 thuê: môi trường, tốc độ và kiểm tra bitstream |
| **plans/**, **reports/**, **results/** | Kế hoạch, báo cáo và kết quả thí nghiệm |

### Vệ sinh Git và artifact

- Theo ghi chú ngày 14/10, nhánh **claude/what-should-i-do-now-f6f4dd** đã được merge fast-forward vào **main** trên máy local. **origin/main** vẫn dừng ở bake-off ngày 09/10 và cần push.
- **experiments/l4_kit/l4_emu.py** và **emulate_l4.sh** được giữ để ghi lại thử nghiệm mô phỏng L4 trên L40 bằng MPS; **không dùng kết quả mô phỏng làm benchmark L4**.
- **README.md** vẫn mô tả quy trình Kaggle và chưa phản ánh DiT-IC, server, decoder chưng cất và kit L4.
- **plans/Plan CLIC 2027 v2** vẫn ghi compute là Kaggle cộng L4.
- Checkpoint chưng cất, latent DIV2K dùng để chưng cất và các file streams hiện chỉ có trên server theo ghi chú nguồn.

## 12. Kết luận

Hướng nghiên cứu hiện tại có bằng chứng ban đầu ủng hộ DiT-IC cho ảnh tự nhiên ở bitrate thấp, MS-ILLM ở bitrate trung bình/cao và VTM-SCC cho ảnh chữ/màn hình. Phương án trộn theo ảnh đạt Q̂ v0 tốt nhất trong bake-off 30 ảnh đã báo cáo. Tuy nhiên, đây chưa phải bằng chứng rằng hệ thống sẽ thắng CLIC 2027.

Trong ngắn hạn, **không nên tối ưu chất lượng thêm trước khi khóa được khả năng giải mã**. Ba cổng bắt buộc là: (1) chạy được trong môi trường CLIC, (2) giải mã nhất quán giữa hai máy L4, và (3) đáp ứng giới hạn thời gian và dung lượng. Sau khi vượt qua các cổng này, cần mở rộng đánh giá người chấm và tập ảnh, rồi mới đầu tư vào việc mở rộng operating point DiT-IC ở 0.15–0.3 bpp.

## Tài liệu tham khảo

- CLIC 2027 FAQ — môi trường đánh giá và quy định hiệu năng: https://clic2027.compression.cc/faq/
- CLIC 2027 tasks: https://clic2027.compression.cc/tasks/
- MS-ILLM, *Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models*: https://arxiv.org/abs/2301.11189
- DC-AE: https://arxiv.org/abs/2410.10733
- SANA: https://arxiv.org/abs/2410.10629
- CoD-Lite: https://arxiv.org/abs/2604.12525
- COOL-CHIC: https://openaccess.thecvf.com/content/ICCV2023/html/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.html
- MBT2018, *Joint Autoregressive and Hierarchical Priors for Learned Image Compression*: https://arxiv.org/abs/1809.02736

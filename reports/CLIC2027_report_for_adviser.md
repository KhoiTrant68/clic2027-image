# CLIC 2027 Image GPU Track — Báo cáo tiến độ

**Ngày cập nhật:** 15/10/2026\
**Repo:** [clic2027-image](https://github.com/KhoiTrant68/clic2027-image)\
**Mục tiêu:** xây dựng bộ nén ảnh có chất lượng cảm nhận cao ở các mức
bitrate thấp, đồng thời đáp ứng giới hạn thời gian giải mã, dung lượng
decoder và tính tái lập của môi trường CLIC 2027.

**Gợi ý đọc:** mục 1 tóm tắt quyết định; mục 5 trình bày bằng chứng thực nghiệm; mục 9–10 là hướng nghiên cứu và nội dung xin ý kiến. Mục 3 giải thích pipeline, còn Phụ lục A trình bày nền tảng MS-ILLM.

## 1. Tóm tắt dự án

**Cập nhật sau vòng tích hợp ngày 15/10:** PULSE-L đã chạy qua CLI bake-off chính với **150 điểm (30 ảnh × q0–q4)**, có header truyền thật. Mix đạt **Q̂ 1.4978 tại 0.3 bpp**, thay cho 1.4912 của vòng sàng lọc dưới đây; tại 0.15 bpp vẫn đạt 0.7891. Gói decoder PULSE-L riêng giải mã 30/30 ảnh khớp pixel trong **12.235 giây trên L40**, tính cả giải nén gói, khởi động, nạp model và ghi PNG. Gói ZIP 74.7 MB; VRAM toàn thiết bị cao nhất trong vòng tích hợp là 5,920 MiB, thấp hơn giới hạn 30 GB/GPU. Đây chưa phải benchmark toàn bộ mix hoặc trên L4. Chi tiết và giới hạn phép đo: [báo cáo tích hợp PULSE-L](CLIC2027_pulse_integration_2026-10-15.md). Các số 204 lượt và 1.4912 bên dưới mô tả vòng sàng lọc trước đó; 120 payload lặp lại đã được xác nhận giống hệt, không tính là điểm chất lượng mới.

Đến ngày 15/10/2026, dự án đã xây dựng thước đo preference Q̂ v0, đánh giá nhiều codec bằng bitstream thật, thử chưng cất decoder DiT-IC và kiểm tra một phần tính tất định. Vòng mới hoàn tất **204 lượt nén–giải nén**, gồm AFP-GIC, DC-VIC, PULSE và PULSE-L; riêng PULSE-L được đánh giá trên đủ 30 ảnh ở bốn mức chất lượng.

Kết quả mới đáng chú ý là **thêm PULSE-L vào lựa chọn theo ảnh**: Q̂ tăng từ **0.7582 lên 0.7891 ở 0.15 bpp**, và từ **1.4215 lên 1.4912 ở 0.3 bpp**. Đây là mức tăng của metric đại diện trên validation, chưa được xác nhận bằng chấm mù. Decoder PULSE-L có trung vị 57.2 ms/ảnh trên L40 trong phép đo API; chưa phải thời gian bài nộp trên L4.

Đề xuất hiện tại là giữ DiT-IC cho bitrate thấp, bổ sung PULSE-L vào hệ thống hỗn hợp và nghiên cứu cách lựa chọn codec có tính cả thời gian giải mã, chi phí nạp model và độ tin cậy của đánh giá chất lượng. Các phần tiếp theo phân biệt **kết quả đã đo**, **diễn giải** và **thí nghiệm đề xuất**.

### Đọc nhanh bằng hình — cập nhật 15/10

**Kiểm deployment mới nhất:** nhánh MS-ILLM của mix đã được kiểm bằng stream lưu trên đĩa. Cấu hình mặc định lệch pixel nhẹ giữa tiến trình; cấu hình deterministic đạt **5/5 ảnh khớp pixel**, metadata thực tăng tổng 120 byte và vẫn vừa budget. Output khác nhẹ baseline nên cần chấm lại trước khi gán điểm Q̂ cho decoder mới. Xem [báo cáo kiểm MS-ILLM](CLIC2027_msillm_stream_audit_2026-10-15.md). Các biểu đồ dưới đây vẫn mô tả bake-off trước hiệu chỉnh này.

**Chất lượng tăng bao nhiêu?** Hình trái so sánh cùng budget trên cùng 30 ảnh; hình phải tách mức tăng để thấy cải thiện nhỏ ở 0.075 bpp. Q̂ là metric đại diện, chưa phải điểm cuộc thi hoặc preference đã xác nhận bằng chấm mù.

![Chất lượng mix trước và sau tích hợp PULSE-L](figures/pulse_2026-10-15/quality.png)

**PULSE-L đóng vai trò gì?** Tại 0.3 bpp, allocator chọn PULSE-L cho 22/30 ảnh; các codec khác vẫn đóng góp. Thanh thể hiện số ảnh, không phải tỷ trọng byte.

![Phân bố lựa chọn codec tại 0.3 bpp](figures/pulse_2026-10-15/allocation.png)

**Vì sao thời gian giảm?** Phần lớn chênh lệch đến từ ghi PNG đầu ra, trong khi 30/30 ảnh vẫn khớp pixel. VRAM dùng GB thập phân để so cùng đơn vị với hạn mức 30 GB/GPU; đây là tổng bộ nhớ thiết bị quan sát, gồm cả tiến trình khác.

![Thời gian offline và bộ nhớ GPU](figures/pulse_2026-10-15/runtime_memory.png)

Thời gian trên chỉ đo **PULSE-L riêng trên L40**, một lượt mỗi cấu hình. Các nhánh khác chưa được đo chung trong gói decoder toàn mix. Xem [báo cáo tích hợp](CLIC2027_pulse_integration_2026-10-15.md) để đối chiếu nguồn và phạm vi đo. Hình có bản SVG và [script tái tạo từ artifact](figures/pulse_2026-10-15/make_figures.py).

**Từ bằng chứng hiện tại đến quyết định training:** ô xanh là kết quả đã đo, ô vàng là việc còn cần thực hiện.

```mermaid
flowchart LR
    A["ĐÃ ĐO: 150 điểm PULSE-L<br/>30 ảnh × 5 quality"] --> B["ĐÃ ĐO: mix đạt Q̂ 1.4978<br/>tại 0.3 bpp"]
    A --> C["ĐÃ ĐO: PULSE-L offline trên L40<br/>12.235 s; 30/30 ảnh khớp pixel"]
    B --> D["CẦN LÀM: dispatcher và gói toàn mix"]
    C --> D
    D --> E["CẦN KIỂM: tốc độ và parity trên L4"]
    B --> F["CẦN KIỂM: chấm mù mix cũ / mới"]
    E --> G["QUYẾT ĐỊNH: ngân sách fine-tune<br/>và hướng rate thấp"]
    F --> G
    classDef done fill:#e4f3ef,stroke:#008577,color:#163c35;
    classDef pending fill:#fff3df,stroke:#d87826,color:#613700;
    class A,B,C done;
    class D,E,F,G pending;
```

### 1.1 Vấn đề nghiên cứu

PSNR đo sai số pixel, nhưng không phải lúc nào cũng phản ánh ảnh nào được con người ưa thích hơn. Các codec học sâu có thể tạo ảnh trông tự nhiên hơn, song thường đòi hỏi decoder nặng, nhiều trọng số, và có thể gặp vấn đề tái lập bitstream khi entropy coding phụ thuộc phép toán float.

**Dự án tập trung vào bốn câu hỏi:**

1.  Codec nào cho chất lượng cảm nhận tốt nhất tại **0.075, 0.15 và 0.3
    bpp**?
2.  Có thể dùng dữ liệu preference có sẵn để xây dựng thước đo tự động
    Q̂, giảm chi phí so sánh hàng chục codec hay không?
3.  Làm thế nào kết hợp codec theo từng ảnh hoặc từng loại ảnh, thay vì
    buộc mọi ảnh dùng cùng một mô hình?
4.  Làm thế nào bảo đảm decoder chạy đủ nhanh, trong giới hạn bộ nhớ, và
    tạo kết quả nhất quán trên môi trường đánh giá?

### 1.2 Quyết định kỹ thuật hiện tại

| Tình huống | Hướng đang ưu tiên | Lý do | Mức độ chắc chắn |
| --- | --- | --- | --- |
| **0.075 bpp, ảnh tự nhiên** | DiT-IC | Bake-off và chấm mù sơ bộ ủng hộ DiT-IC so với MS-ILLM | Bằng chứng ban đầu; mẫu nhỏ |
| **0.15 bpp** | Mix DiT-IC, MS-ILLM, PULSE-L và VTM-SCC | PULSE-L được chọn cho 10/30 ảnh trong allocation mới | Q̂ trên 30 ảnh; chưa có chấm mù mix mới |
| **0.3 bpp** | Mix có PULSE-L, giữ MS-ILLM và các mode bổ sung | PULSE-L được chọn cho 20/30 ảnh; mix đạt Q̂ 1.4912 | Kết quả nội bộ; cần holdout và benchmark toàn gói |
| **Ảnh chữ, giao diện, màn hình** | VTM-SCC | Codec hướng screen content giữ cấu trúc và chữ tốt hơn trong thử nghiệm trước đó | Có tín hiệu tốt; bộ phát hiện ảnh chưa triển khai |

**Vấn đề chính:** Các rủi ro lớn nhất hiện nay là benchmark trên L4 thật, tính tất định giữa hai máy L4, đóng gói decoder trong giới hạn 4 GB và tích hợp các nhánh codec thành một pipeline decode hoàn chỉnh.

## 2. Mục tiêu, giả thuyết và tiêu chí thành công

### 2.1 Mục tiêu

-   Tối ưu chất lượng cảm nhận theo ngân sách bitrate của toàn bộ tập ảnh.
-   Cho phép chọn codec/operating point khác nhau theo ảnh và theo  bitrate.
-   Đo riêng chất lượng, tốc độ, bộ nhớ, kích thước artifact và độ tin cậy của giải mã.
-   Giữ pipeline thí nghiệm có thể chạy lại, lưu log và tiếp tục từ trạng thái trước.

### 2.2 Giả thuyết nghiên cứu

-   **H1 --- Preference thay cho PSNR đơn lẻ:** một mô hình $\hat{Q}$ kết hợp nhiều metric sẽ dự đoán lựa chọn của người chấm tốt hơn PSNR riêng. **Giải thích**: $\hat{Q}$ là một mô hình dự đoán chất lượng; điểm số do mô hình tạo ra mới là metric dùng để đánh giá và so sánh codec.
-   **H2 --- Codec chuyên biệt theo loại ảnh:** codec diffusion/latent có thể có lợi trên ảnh tự nhiên ở bitrate rất thấp, trong khi VTM-SCC có thể phù hợp hơn cho ảnh chữ/màn hình.
-   **H3 --- Trộn theo ảnh tốt hơn một codec duy nhất:** phân bổ ngân sách byte cho codec/operating point phù hợp từng ảnh có thể tăng tổng preference dưới cùng ngân sách byte.
-   **H4 --- Decoder chưng cất là phương án đánh đổi:** có thể giảm thời gian giải mã với tổn thất chất lượng cảm nhận có giới hạn.
-   **H5 --- Tính tất định là điều kiện bắt buộc:** chất lượng cao không có ý nghĩa nếu entropy decode tạo latent/ảnh khác nhau giữa môi trường encode và decode.

### 2.3 Tiêu chí thành công

Luật hiện tại yêu cầu đủ ba rate, decoder không quá **4 GB**, chạy với **2 CPU, RAM 12 GB và GPU L4 24 GB**. Chất lượng được đánh giá bằng người; nhóm **25% bài có thời gian giải mã cao nhất** không đủ điều kiện nhận giải. Decoder test phải khớp hash đã đăng ký ở validation. Có thể dùng decoder khác nhau giữa các rate, nhưng mỗi decoder phải được đăng ký trước. [Nguồn: CLIC 2027 tasks](https://clic2027.compression.cc/tasks/).

Mục tiêu kỹ thuật nội bộ là hướng tới **≤20 giây/30 ảnh trên L4**, đồng thời báo riêng decode và tổng wall time từ tiến trình mới. Mốc 20 giây chưa được xác nhận, cũng không bảo đảm tránh nhóm 25% chậm nhất. Kết quả nghiên cứu chỉ được xem là sẵn sàng triển khai khi qua kiểm tra bitstream, byte-accounting và chạy trọn gói trong môi trường đích.

Tài nguyên nghiên cứu hiện có là **2 GPU L40** trên server. Giới hạn được chủ dự án cho phép là **30 GB trên mỗi GPU**, hiểu bảo thủ là 30,000,000,000 byte tổng bộ nhớ thiết bị, gồm cả dịch vụ đang chạy. Giới hạn này dành cho nghiên cứu trên server; không thay đổi giới hạn phần cứng bài nộp L4.

## 3. Pipeline nghiên cứu tổng thể

``` mermaid
flowchart TD
    A["Dữ liệu preference CLIC 2021/2022/2024"] --> B["Tính metric cho cặp ảnh"]
    B --> C["Fit Q-hat: Bradley–Terry"]
    D["30 ảnh validation"] --> E["Encode bằng các codec ứng viên"]
    E --> F["Decode và đo metric / thời gian"]
    F --> G["Chia ngân sách byte toàn tập"]
    C --> G
    G --> H["Bake-off tại 0.075 / 0.15 / 0.3 bpp"]
    H --> I["Phân tích chất lượng và trade-off"]
    I --> J["Chấm mù với người"]
    I --> K["Benchmark decoder và bộ nhớ"]
    K --> L["Kiểm tra bitstream trên phần cứng khác"]
    L --> M["Đóng gói và smoke test trong môi trường CLIC"]

```

Sơ đồ này mô tả toàn bộ quy trình nghiên cứu để tìm ra một hệ thống nén ảnh phù hợp với CLIC 2027 GPU track. Nó không chỉ nhằm tìm codec có chất lượng ảnh tốt, mà còn phải đánh giá chất lượng đáng tin cậy, phân bổ bitrate, giải mã đủ nhanh, chạy ổn định trên phần cứng khác và đóng gói đúng yêu cầu cuộc thi.

Có thể chia pipeline thành 5 giai đoạn:

1. Xây dựng thước đo chất lượng $\hat{Q}$.
2. Đánh giá các codec ứng viên trên cùng tập ảnh.
3. So sánh chất lượng tại cùng ngân sách bitrate.
4. Xác nhận chất lượng chủ quan và khả năng triển khai.
5. Kiểm tra tính tái lập và đóng gói nộp bài.

### 3.1 Xây dựng thước đo và so sánh codec

#### Bước A — Dữ liệu preference CLIC 2021/2022/2024

**Mục tiêu: Thu thập dữ liệu phản ánh cách con người đánh giá chất lượng ảnh nén.**

Dữ liệu preference thường chứa các cặp ảnh được so sánh: người đánh giá chọn ảnh A tốt hơn ảnh B, hoặc dữ liệu tương đương được suy ra từ các đánh giá có sẵn. Trong nghiên cứu này, dữ liệu đến từ các năm CLIC 2021, 2022 và 2024.

**Vì sao cần nhiều năm dữ liệu?**

- Mỗi năm có thể có phân bố ảnh khác nhau: ảnh tự nhiên, cảnh vật, chân dung, ảnh có nhiều chi tiết hoặc vùng phẳng.
- Dùng nhiều năm giúp kiểm tra liệu mô hình đánh giá chất lượng có tổng quát hóa hay chỉ phù hợp với một tập ảnh cụ thể.
- Có thể dùng một số năm để huấn luyện và năm còn lại để kiểm tra khả năng dự đoán trên dữ liệu chưa thấy.

**Đầu ra: Tập dữ liệu các so sánh chất lượng, ví dụ:**

- Ảnh A được đánh giá tốt hơn ảnh B.
- Ảnh B được đánh giá tốt hơn ảnh C.
- Nếu có dữ liệu gốc, mỗi so sánh được liên kết với ảnh, codec, bitrate và các metric tương ứng.

**Điểm cần chú ý:**
Cần kiểm tra nguồn gốc nhãn preference, cách chia train/validation/test và nguy cơ trùng ảnh giữa các tập. Nếu nhãn đến từ nguồn khác nhau, cũng cần xem chúng có cùng quy ước đánh giá hay không.

#### Bước B — Tính metric cho cặp ảnh

**Mục tiêu: Chuyển mỗi ảnh nén thành một vector đặc trưng định lượng để mô hình $\hat{Q}$ học mối quan hệ giữa metric và preference.**

Với mỗi ảnh nén, so sánh ảnh đó với ảnh gốc và tính các metric như:

| Metric   | Ý nghĩa                                                                      |
| -------- | ---------------------------------------------------------------------------- |
| PSNR     | Đo sai số pixel; cao thường là tốt hơn.                                      |
| MS-SSIM  | Đo độ tương đồng về cấu trúc ở nhiều thang đo.                               |
| LPIPS    | Đo khoảng cách đặc trưng perceptual từ mạng học sâu; thấp thường là tốt hơn. |
| DISTS    | Đánh giá khác biệt về cấu trúc và kết cấu; thấp thường là tốt hơn.           |
| CLIP-IQA | Ước lượng chất lượng cảm nhận dựa trên đặc trưng học được.                   |

Các metric có thang đo và chiều tốt/xấu khác nhau. Vì vậy, trước khi fit mô hình, cần chuẩn hóa hoặc biểu diễn chúng theo cách nhất quán.

**Đầu ra:** Một vector metric cho mỗi ảnh nén, sau đó ghép thành đặc trưng so sánh cho từng cặp ảnh.

#### Bước C — Fit $\hat{Q}$: Bradley–Terry

**Mục tiêu: Học một hàm điểm số tổng hợp để dự đoán ảnh nào có khả năng được đánh giá tốt hơn.**

Mô hình Bradley–Terry thường biểu diễn xác suất ảnh $i$ được ưa thích hơn ảnh $j$ như sau:

$$
P(i \succ j)=\sigma(\beta+s_i-s_j),\qquad
\sigma(t)=\frac{1}{1+\exp(-t)}.
$$

Trong đó:

- $s_i$, $s_j$: điểm chất lượng mà mô hình gán cho hai ảnh.
- $\sigma$: hàm sigmoid.
- $P(i \succ j)$: xác suất ảnh $i$ được ưa thích hơn ảnh $j$.
- $\beta$: bias vị trí A/B trong implementation; bằng 0 thì thu về dạng Bradley–Terry đối xứng.

Trong cách triển khai dựa trên metric, điểm $s_i$ có thể được tính từ vector metric của ảnh:

$$
s_i=\hat{Q}(x_i,\hat{x}_i)
=\sum_{k=1}^{K}w_k\,T_k\!\left(\phi_k(x_i,\hat{x}_i)\right).
$$

Ở đây $x_i$ là ảnh gốc, $\hat{x}_i$ là ảnh tái tạo, $\phi_k$ là metric thứ $k$, $T_k$ là phép biến đổi metric và $w_k$ là trọng số đã fit. Trong repo, `qhat.transform` biến đổi đặc trưng và `bakeoff.qhat_score` tính tổng có trọng số. Bias A/B dùng khi dự đoán preference, không cộng vào utility từng ảnh của allocator. Điểm Q̂ không phải phần trăm chất lượng và không có mốc 0 mang ý nghĩa tuyệt đối.

Mô hình học tham số $\theta$ sao cho thứ tự dự đoán khớp với preference trong dữ liệu huấn luyện.

**Cách đánh giá mô hình:**

- Độ chính xác dự đoán preference trên tập chưa dùng để huấn luyện.
- Khả năng tổng quát hóa qua các năm CLIC.
- Mức độ tương quan với đánh giá của con người.

**Đầu ra:** Mô hình $\hat{Q}$ đã huấn luyện, dùng làm thước đo chất lượng trong các thí nghiệm codec.

**Điểm quan trọng:** $\hat{Q}$ không tự động trở thành một metric chuẩn chỉ vì nó được fit trên preference. Nó cần được xác thực trên dữ liệu độc lập. Nếu dữ liệu huấn luyện thiên về một loại ảnh hoặc một metric, điểm số có thể thiên lệch theo.

#### Bước D — 30 ảnh validation

**Mục tiêu:** Chọn một tập ảnh nhỏ để chạy thử nghiệm codec nhanh và có thể lặp lại nhiều lần.

Tập 30 ảnh giúp giảm chi phí tính toán khi thử nhiều cấu hình, chẳng hạn:

- Nhiều codec.
- Nhiều mức bitrate.
- Nhiều phiên bản model.
- Các phương pháp phân bổ bitrate.
- Các thay đổi ở encoder hoặc decoder.

Đây là tập validation CLIC 2027 gồm 30 ảnh, trùng tập test CLIC 2025, có ảnh tự nhiên, game và screen content. Không phải một tập con tự chọn từ validation lớn hơn. [Nguồn dữ liệu chính thức](https://clic2027.compression.cc/tasks/).

**Phân biệt validation chính thức với pilot và holdout**

Vòng mới dùng pilot sáu ảnh lấy từ tập này để phát hiện lỗi và loại ứng viên yếu, sau đó đánh giá PULSE-L trên đủ 30 ảnh. Cần bổ sung dev/holdout độc lập ngoài validation để kiểm tra khả năng tổng quát hóa; không dùng 30 ảnh validation để train codec.

Tuy nhiên, 30 ảnh chỉ phù hợp để sàng lọc và phân tích sơ bộ. Kết quả có thể thay đổi đáng kể theo cách chọn ảnh, nhất là khi chỉ có vài ảnh đại diện cho một loại nội dung.

**Đầu ra:** Một tập validation cố định, được sử dụng nhất quán cho các codec so sánh.

#### Bước E — Encode bằng các codec ứng viên

**Mục tiêu: Nén cùng một tập ảnh bằng các phương pháp khác nhau để tạo ra các bitstream có thể so sánh.**

Các ứng viên trong dự án bao gồm:

- VTM / VTM-SCC: Codec dựa trên công nghệ video coding, đóng vai trò baseline truyền thống.
- MS-ILLM: Phương pháp nén ảnh dựa trên mô hình học sâu.
- MBT2018: Một codec nén ảnh học sâu có entropy model.
- CoD-Lite: Một phương pháp nén ảnh học sâu khác.
- DC-AE + latent codec: Mã hóa ảnh thành latent rồi lượng tử hóa và entropy-code latent.
- DiT-IC: Hướng nghiên cứu kết hợp biểu diễn latent với mô hình sinh hoặc xử lý dựa trên kiến trúc DiT, theo thiết kế cụ thể của dự án.

Khi encode, mỗi codec chuyển ảnh thành bitstream. Bitstream có thể chứa latent đã lượng tử hóa, thông tin phụ, header và dữ liệu entropy-coded.

**Điều cần kiểm soát:**
1. Dùng cùng ảnh gốc.
2. Chọn cấu hình tương ứng với mục tiêu bitrate.
3. Tính bitrate từ số byte thực tế của bitstream, không chỉ dựa trên tham số rate của model.
4. Ghi lại phiên bản code, checkpoint, cấu hình và thời gian encode.
5. Không để một codec được dùng thông tin bổ sung mà codec khác không được phép sử dụng.

**Đầu ra:** Bitstream của mỗi codec, cùng metadata về cấu hình, số byte và thời gian encode.

#### Bước F — Decode và đo metric / thời gian

**Mục tiêu: Đánh giá chất lượng ảnh khôi phục và chi phí tính toán thực tế.**

Mỗi bitstream được giải mã thành ảnh tái tạo $\hat{x}$. Sau đó, ảnh tái tạo được so sánh với ảnh gốc $x$.

**Các đại lượng chính:**

- $\hat{Q}$: Điểm chất lượng cảm nhận tổng hợp.
- PSNR, MS-SSIM, LPIPS, DISTS: Các metric bổ trợ giúp giải thích kết quả.
- Thời gian decode: Chi phí giải mã.
- Bộ nhớ GPU: Lượng VRAM cần thiết.
- Kích thước bitstream: Dữ liệu cần truyền hoặc lưu trữ.

Ví dụ, một codec có thể đạt $\hat{Q}$ cao nhưng cần một decoder rất lớn hoặc mất quá nhiều thời gian để tạo ảnh. Codec khác có điểm thấp hơn đôi chút nhưng nhanh và ổn định hơn.

**Vì sao đo thời gian decode ở đây?**

Tốc độ giải mã rất quan trọng. Cần phân biệt:

- Thời gian nạp model.
- Thời gian giải mã thực tế.
- Thời gian xử lý phụ trợ, chẳng hạn entropy decoding hoặc chuyển đổi tensor.
- Peak VRAM và chi phí khởi tạo.

Để so sánh công bằng, cần thống nhất phần cứng, số ảnh, batch size, warm-up và cách đo thời gian. Không nên so sánh số liệu thu từ các điều kiện khác nhau như thể chúng tương đương. Các phép đo hiện tại cố định cách đo trên L40; kết quả L4 phải được đo riêng. Không giả định hai GPU có hiệu năng tương đương hoặc dùng hệ số quy đổi chưa được kiểm chứng.

**Đầu ra:** Ảnh tái tạo, metric chất lượng, thời gian decode và thống kê tài nguyên cho mỗi codec.

#### Bước G — Chia ngân sách byte toàn tập

Đây là bước quan trọng nhất để tránh so sánh codec không công bằng.

Mục tiêu: Phân bổ tổng số byte cho các ảnh sao cho tổng bitrate của cả tập nằm trong ngân sách quy định.

Nếu có $N$ ảnh, với kích thước ảnh $H_i \times W_i$, tổng số bitstream là $B_{\text{total}}$ byte, bitrate trung bình là:

$$
\mathrm{bpp}_{\mathrm{avg}}=\frac{8B_{\mathrm{total}}}{\sum_{i=1}^{N}H_iW_i}.
$$

Trong đó $B_{\text{total}}$ phải bao gồm các thành phần bitstream thực sự được tính vào ngân sách cuộc thi, chẳng hạn header và side information nếu chúng thuộc phần dữ liệu được chấm.

**Tại sao phải phân bổ ngân sách?**

Giả sử codec A có thể dành nhiều byte cho ảnh khó và ít byte cho ảnh dễ, còn codec B dùng cùng một bitrate cho mọi ảnh. Nếu A được tối ưu trên toàn tập, nó có thể tạo ra chất lượng trung bình cao hơn bằng cách đặt thêm bit vào những ảnh mà các bit bổ sung mang lại nhiều cải thiện nhất.

Một cách diễn đạt bài toán phân bổ bitrate là:

$$
\max_{j_1,\ldots,j_N}\sum_{i=1}^{N}\hat{Q}_{i,j_i}
$$

với ràng buộc ngân sách:

$$
\sum_{i=1}^{N}B_{i,j_i}\leq
B_{\mathrm{budget}}=
\left\lfloor\frac{r}{8}\sum_{i=1}^{N}H_iW_i\right\rfloor.
$$

Trong đó:

- $N$: số ảnh; $H_i,W_i$: kích thước ảnh $i$.
- $j_i$: lựa chọn codec và mức chất lượng cho ảnh $i$ trong các điểm đã đo.
- $\hat{Q}_{i,j_i}$: utility dự đoán của lựa chọn đó.
- $B_{i,j_i}$: số byte thực tế của bitstream; $r$: budget bpp toàn tập.

Repo dùng multiple-choice knapsack: chọn đúng một điểm cho mỗi ảnh, tối đa tổng Q̂ với trọng số ảnh bằng nhau. Chi phí byte được làm tròn lên khi giải bằng quy hoạch động; kết quả luôn phải kiểm lại bằng byte thật. Trong khi bpp được gộp theo tổng pixel, Q̂ báo cáo là trung bình theo ảnh.

Nếu cuộc thi đánh giá theo chất lượng trung bình không trọng số, mục tiêu cần phản ánh đúng quy tắc đó; không nên tự đặt trọng số làm thay đổi mục tiêu đánh giá.

**Đầu ra:** Các bitstream hoặc cấu hình bitrate đã được chọn sao cho tổng số byte nằm trong ngân sách tương ứng.

Lưu ý quan trọng: Không phải lúc nào codec cũng có thể được cấp chính xác số byte mong muốn. Phân bổ bitrate cần dùng các điểm rate–distortion đã đo, nội suy hoặc thử nghiệm thêm, rồi kiểm tra lại bằng kích thước bitstream thực tế.

#### Bước H — Từ $\hat{Q}$ đến bake-off tại 0.075 / 0.15 / 0.3 bpp

Trong sơ đồ, bước C tạo ra thước đo $\hat{Q}$, còn bước H sử dụng thước đo đó để so sánh codec. Đây là hai vai trò khác nhau.

**Mục tiêu của bake-off: So sánh các codec ở ba mức bitrate đại diện cho những chế độ nén khác nhau.**

| Bitrate   | Ý nghĩa trực quan                                          |
| --------- | ---------------------------------------------------------- |
| 0.075 bpp | Nén rất mạnh; chỉ có ít bit để biểu diễn ảnh.              |
| 0.15 bpp  | Mức trung gian.                                            |
| 0.3 bpp   | Ngân sách lớn hơn, thường cho phép giữ nhiều chi tiết hơn. |

Ở mỗi mức, các codec được so sánh dựa trên:

1. Tổng byte toàn tập.
2. Chất lượng theo $\hat{Q}$.
3. Các metric bổ trợ.
4. Thời gian decode và bộ nhớ.
5. Sự khác biệt giữa các nhóm nội dung ảnh.

Đo nhiều mức bitrate cho phép phân biệt một codec chỉ mạnh ở một miền rate với một codec có lợi trên toàn dải mục tiêu.

**Đầu ra:** Bảng kết quả bake-off, gồm chất lượng và chi phí của từng codec tại mỗi bitrate.

### 3.2 Phân tích sau bake-off

#### Bước I — Phân tích chất lượng và trade-off

**Mục tiêu: Xác định phương án nào tốt nhất theo mục tiêu thực tế, thay vì chỉ chọn codec có một điểm số cao nhất.**

Các phân tích cần làm:

- Rate–quality: Chất lượng thay đổi như thế nào khi bitrate tăng?
- So sánh với baseline: Codec mới có cải thiện rõ ràng so với VTM, VTM-SCC hoặc MS-ILLM không?
- Theo loại ảnh: Kết quả trên ảnh tự nhiên có khác ảnh chứa chữ hoặc screenshot không?
- Độ ổn định: Kết quả có bị chi phối bởi một vài ảnh ngoại lệ không?
- Độ phức tạp: Lợi ích chất lượng có đáng với chi phí decode và bộ nhớ không?

Có thể biểu diễn kết quả bằng đường cong rate–quality: trục ngang là bpp, trục dọc là $\hat{Q}$. Một codec tốt hơn thường có điểm cao hơn ở cùng bitrate hoặc đạt cùng chất lượng với ít bit hơn.

Nếu sử dụng BD-rate hoặc một đại lượng tổng hợp tương tự, cần xác định rõ metric, miền bitrate và phương pháp nội suy. Không nên tính hoặc diễn giải một con số tổng hợp khi các đường cong không có miền bitrate phù hợp.

**Đầu ra:** Danh sách codec được ưu tiên, các điểm mạnh/yếu và giả thuyết cần kiểm tra tiếp.

#### Bước J — Blind grading với người

Mục tiêu: Xác nhận rằng cải thiện theo $\hat{Q}$ thực sự tương ứng với chất lượng cảm nhận của con người.

Trong đánh giá mù, người đánh giá xem hai ảnh tái tạo mà không được biết codec nào tạo ra ảnh nào. Họ chọn ảnh tốt hơn hoặc cho điểm theo giao thức đã định trước.

Kết quả có thể tính thành tỷ lệ thắng, kèm số lượng cặp, khoảng tin cậy và phân tích theo loại ảnh.

**Tại sao cần bước này dù đã có $\hat{Q}$?**

Vì $\hat{Q}$ chỉ là mô hình dự đoán từ dữ liệu preference. Nếu mô hình ưu tiên một loại đặc trưng không phù hợp với người xem, bake-off có thể chọn nhầm codec.

Đánh giá mù đặc biệt hữu ích khi so sánh codec truyền thống với codec sinh ảnh. Hai ảnh có thể có chất lượng cảm nhận tương tự nhưng khác độ trung thực với ảnh gốc. Do đó, cần quyết định rõ người đánh giá được hướng dẫn ưu tiên “giống ảnh gốc” hay “trông đẹp hơn”.

**Đầu ra:** Bằng chứng độc lập để xác nhận hoặc bác bỏ thứ hạng theo $\hat{Q}$.

**Lưu ý:** Kết quả từ một người đánh giá hoặc một số ít cặp ảnh chỉ nên coi là sơ bộ. Muốn có kết luận đáng tin cậy, cần nhiều người đánh giá, thứ tự hiển thị ngẫu nhiên, giao thức nhất quán và phân tích bất định.

#### Bước K — Benchmark decoder và bộ nhớ

**Mục tiêu: Xác định codec có thể chạy trong giới hạn triển khai thực tế hay không.**

Những chỉ số nên ghi nhận:

| Chỉ số                | Câu hỏi cần trả lời                           |
| --------------------- | --------------------------------------------- |
| Tổng thời gian decode | Có hoàn thành trong thời gian cho phép không? |
| Thời gian mỗi ảnh     | Chi phí trung bình là bao nhiêu?              |
| Peak VRAM             | Có vượt giới hạn GPU không?                   |
| Thời gian khởi tạo    | Nạp model có tốn đáng kể thời gian không?     |
| Độ ổn định            | Có OOM, timeout hoặc lỗi ngẫu nhiên không?    |

Cần benchmark trên phần cứng phù hợp với môi trường cuộc thi, hoặc ghi rõ khi đang sử dụng phần cứng thay thế.

Đặc biệt, nên đo toàn bộ đường đi từ bitstream đến ảnh đầu ra, không chỉ đo một phép forward của neural network. Entropy decoding, chuyển dữ liệu, hậu xử lý và khởi tạo đều có thể làm tăng thời gian thực tế.

**Đầu ra:** Bảng benchmark và kết luận codec nào đáp ứng các giới hạn thời gian/bộ nhớ.

#### Bước L — Kiểm tra bitstream trên phần cứng khác

**Mục tiêu: Đảm bảo bitstream có thể được giải mã ổn định và cho kết quả đúng trên môi trường CLIC.**

Các phép kiểm tra nên gồm:

1. Encode trên máy A, decode trên máy B.
2. Decode cùng bitstream nhiều lần.
3. So sánh ảnh đầu ra bằng checksum hoặc kiểm tra từng pixel.
4. Kiểm tra các trường hợp biên: ảnh nhỏ, kích thước lẻ, bitrate thấp và bitstream không hợp lệ.
5. Kiểm tra tính tương thích của phiên bản thư viện, CUDA, driver và checkpoint.

Nếu decoder dùng xác suất để tạo CDF cho entropy coding, những sai khác số học nhỏ có thể làm thay đổi cách giải mã bitstream. Vì vậy, cần đặc biệt cẩn thận với các phép tính phải tái lập chính xác.

Bitwise determinism nghĩa là kết quả phải khớp chính xác từng bit khi điều kiện kiểm tra yêu cầu như vậy. Tuy nhiên, tiêu chí cuối cùng phải tuân theo giao thức và cách chấm của CLIC; không nên mặc định rằng mọi phép tính trung gian đều bắt buộc bitwise-identical nếu cuộc thi không yêu cầu điều đó.

**Đầu ra:** Bằng chứng decoder tái lập được trên môi trường mục tiêu, cùng danh sách giới hạn tương thích nếu có.

#### Bước M — Đóng gói và smoke test trong môi trường CLIC

Tạo artifact decoder độc lập, bao gồm executable/script, dependency cần thiết và đúng trọng số. Chạy từ thư mục sạch, offline, chỉ với stream đầu vào; kiểm đúng số ảnh, tên file, kích thước, màu và định dạng. Đo tổng wall time kể cả unpack/load/I/O, RAM và VRAM, rồi kiểm giới hạn dung lượng của từng gói đăng ký.

Mode ID, quality index và mọi thông tin decoder cần biết phải có trong dữ liệu được truyền hoặc định nghĩa rõ trong format. Khả năng chạy từng codec riêng trong bake-off chưa chứng minh khả năng chạy cả mix. Sau khi gói đạt kiểm tra, lưu hash và môi trường để đối chiếu lần chạy sau; việc gửi bài chính thức là bước riêng.

### 3.3 Quan hệ phụ thuộc giữa các bước

Có thể nhìn toàn bộ quy trình như ba nhánh kết hợp ở giai đoạn quyết định codec:

- Nhánh chất lượng: Dữ liệu preference → Metric → Fit $\hat{Q}$

- Nhánh codec: Ảnh validation → Encode → Decode → Đo chất lượng và chi phí

- Nhánh ngân sách: Phân bổ byte → Cùng tổng bitrate → Bake-off

**Chọn ứng viên** : Phân tích trade-off → Đánh giá mù → Benchmark → Kiểm tra tính tái lập → Smoke test

**Ba nhánh giải quyết ba câu hỏi khác nhau:**
- Chất lượng: Làm thế nào đo được ảnh nào tốt hơn?
- Công bằng: Làm thế nào so sánh codec khi tổng ngân sách byte giống nhau?
- Khả năng triển khai: Codec có chạy đúng, đủ nhanh và ổn định trong môi trường đích không?

Một codec chỉ thực sự là ứng viên tốt khi đáp ứng cả ba nhóm yêu cầu.

### 3.4 Phạm vi kết luận của từng thí nghiệm

| Thí nghiệm                    | Có thể hỗ trợ kết luận                                                | Không đủ để kết luận                                 |
| ----------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------- |
| Validation $\hat{Q}$ | Dự đoán preference tốt trên dữ liệu kiểm tra hay không    | $\hat{Q}$ luôn đúng với mọi loại ảnh        |
| Bake-off                      | Codec nào tốt hơn trong tập ảnh, bitrate và cấu hình đã thử           | Codec đó chắc chắn tốt nhất trên test chính thức     |
| Đánh giá mù                   | Người đánh giá thích ảnh của codec nào hơn trong các model đã dùng | Ai cũng thích như vậy                  |
| Benchmark                     | Thời gian và bộ nhớ trong môi trường đo                               | Hiệu năng tương tự trên mọi GPU                      |
| Smoke test                    | Artifact chạy được trong các trường hợp thử                           | Artifact chắc chắn vượt qua toàn bộ hệ thống chấm    |

## 4. Các mô hình và codec đang sử dụng

### 4.1 Q̂ — thước đo preference tự động

**Vai trò:** Q̂ không phải codec nén ảnh. Nó là mô hình xếp hạng dùng để
sàng lọc codec và phân bổ ngân sách byte theo chất lượng cảm nhận dự
đoán.

**Cách làm:**
1. Lấy dữ liệu đánh giá theo cặp từ CLIC 2021, 2022 và 2024.
2. Tính các đặc trưng chất lượng như PSNR, MS-SSIM, LPIPS, DISTS và các metric liên quan cho các ảnh trong cặp.
3. Fit mô hình preference Bradley--Terry để ước lượng xác suất một ảnh được ưa thích hơn ảnh kia.
4. Đánh giá trên phần dữ liệu giữ lại test so với baseline.
5. Dùng Q̂ v0 trong bake-off để đánh giá và tối ưu phân bổ byte.

**Ý nghĩa:** Q̂ cho phép đánh giá hàng loạt ứng viên nhanh hơn việc con
người phải chấm mọi cặp. Tuy nhiên, Q̂ chỉ là proxy; không thay thế đánh
giá người thật và có thể khái quát kém sang ảnh hoặc codec mới.

### 4.2 MS-ILLM — codec học sâu làm base ở bitrate trung bình/cao

**Paper:** *Improving Statistical Fidelity for Neural Image Compression
with Implicit Local Likelihood Models* (ICML 2023).\
Paper: https://arxiv.org/abs/2301.11189

**Vai trò trong dự án:** ứng viên codec dùng làm base mạnh ở 0.15 và 0.3 bpp, đồng thời là baseline quan trọng để so sánh với DiT-IC ở 0.075 bpp.

**Kiến trúc và cơ chế của model:**
Paper kết hợp kiến trúc nén ảnh Mean-Scale Hyperprior với một bộ phân biệt mới gọi là ILLM (Implicit Local Likelihood Model).

Phần giải thích chi tiết về rate–distortion–perception, VQ-VAE, discriminator và loss được đặt ở Phụ lục A để tách nền tảng phương pháp khỏi báo cáo kết quả của dự án.

**Tại sao giữ lại:** kết quả bake-off được báo cáo cho thấy MS-ILLM có Q̂
v0 tốt hơn nhiều codec cổ điển ở 0.15 và 0.3 bpp; tốc độ được ghi nhận
khoảng 0.7--0.8 giây/ảnh trên L40. Chưa có kết luận về tốc độ L4 hoặc độ
tương thích hoàn chỉnh với môi trường CLIC.

**Rủi ro:** cần xác minh giấy phép code và pretrained weights, cũng như
tính tất định của mọi thành phần entropy coding phụ thuộc số thực.

### 4.3 DiT-IC — hướng diffusion/latent cho bitrate thấp

**Thành phần được ghi trong repo:** SANA-600M, DC-AE và codec latent
theo hướng ELIC/entropy coding.\
Tham khảo nền tảng: [SANA](https://arxiv.org/abs/2410.10629),
[DC-AE](https://arxiv.org/abs/2410.10733).

**Vai trò:** ứng viên ưu tiên cho ảnh tự nhiên ở 0.075 bpp. Trong
bake-off nội bộ ngày 13/10, DiT-IC đạt Q̂ v0 +0.164 so với −0.451 của
MS-ILLM ở rate này.

**Vì sao có thể hiệu quả:** ở bitrate rất thấp, mục tiêu không chỉ là
giữ từng pixel mà còn tái tạo nội dung có vẻ tự nhiên. Mô hình
sinh/decoder latent có thể tạo cấu trúc thị giác hợp lý với ít bit hơn,
nhưng lợi thế này cần được kiểm tra bằng preference người thật và các
nhóm ảnh khác nhau.

**Giới hạn quan sát được:** DiT-IC kém hơn trên ảnh chữ/màn hình trong
phép chấm mù sơ bộ; operating point đang có chỉ tới khoảng 0.096--0.10
bpp. Vì vậy, không thể coi DiT-IC là codec duy nhất cho toàn bộ dải
bitrate.

**Rủi ro kỹ thuật:** decoder nặng; nhiều bộ trọng số q2/q3/q4 làm tăng
dung lượng; tốc độ hiện đo trên L40 chứ chưa xác nhận trên L4; cần bảo
đảm pipeline decoder chạy được bằng các dependency có trong môi trường
đích.

### 4.4 DiT-IC chưng cất — phương án tăng tốc

**Mục tiêu:** giảm chi phí giải mã DC-AE bằng cách huấn luyện student
bắt chước đầu ra decoder teacher.

**Cách làm được báo cáo:**

- Teacher là decoder DC-AE đã gộp LoRA từ checkpoint DiT-IC q3.
- Student giảm một nửa số kênh ở ba tầng ResBlock độ phân giải cao; phần đầu sao chép từ teacher và đóng băng.
- Dữ liệu gồm latent từ 800 ảnh DIV2K train. Student học đầu ra teacher, không trực tiếp học ảnh gốc.
- Đợt đầu dùng L1 + LPIPS trong 20.000 bước; đợt tiếp thêm PatchGAN với trọng số 0.1. Các bảng snapshot ghi mốc bước riêng của từng stage, không suy tổng số update chỉ từ tên `snap_30000`.

Trainer hiện có chỉ phục vụ phần decoder này; chưa phải trainer rate–distortion hoàn chỉnh cho toàn codec. File student khoảng 24 MB chứa phần trọng số cần thay, không phải toàn bộ decoder chạy độc lập.

**Kết quả nội bộ q3:** Q̂ v0 của teacher +0.157, student −0.012; chênh
lệch khoảng −0.17 Q̂ (xấp xỉ −29 Elo theo quy đổi nội bộ). Thời gian
DC-AE trên L40 giảm từ 12.9 giây xuống 7.3 giây cho 30 ảnh ở phép đo
riêng; toàn decoder được báo cáo khoảng 17--18 giây với teacher và
khoảng 11 giây với student.

**Ý nghĩa:** chưng cất tạo phương án dự phòng nếu teacher không đạt tốc
độ trên L4. Đây là trade-off chất lượng/tốc độ, không phải cải thiện
chất lượng. Cần benchmark lại trên L4 thật và kiểm tra checkpoint được
lưu trữ, dung lượng tổng và tính nhất quán.

### 4.5 VTM 4:2:0 và VTM-SCC — baseline cổ điển / ảnh chữ

-   **VTM 4:2:0:** baseline VVC thông thường, giúp định vị hiệu quả của
    codec học sâu so với codec truyền thống.
-   **VTM-SCC:** hướng screen-content coding, phù hợp để thử nghiệm trên
    ảnh có chữ, giao diện hoặc vùng phẳng.
-   Các kết quả ban đầu trong kế hoạch cho thấy VVC-SCC vượt VTM 4:2:0
    khoảng 1.1--3.3 dB trên ảnh màn hình ở phép đo PSNR cũ; đây là bằng
    chứng thuộc thí nghiệm đó, không phải kết quả bake-off Q̂ ngày 13/10.

**Vai trò dự kiến:** giữ làm baseline và làm mode chuyên biệt. Cần phát
triển bộ phát hiện ảnh chữ/màn hình hoặc bộ chọn mode, rồi đo xem lựa
chọn đó có cải thiện tổng preference dưới cùng ngân sách byte không.

### 4.6 Các ứng viên khác

| Ứng viên | Mục đích trong nghiên cứu | Kết quả/ quyết định theo ghi chú hiện có |
| --- | --- | --- |
| **CoD-Lite** | Codec học sâu cạnh tranh, tốc độ nhanh | Q̂ v0 −0.382 ở 0.075 bpp; khoảng 0.15 s/ảnh; kém MS-ILLM ở 0.15/0.3 theo bake-off |
| **MBT2018-mean** | Baseline learned compression cổ điển | Q̂ khoảng −2.0 tại 0.15 bpp; loại khỏi lựa chọn chính |
| **S1: DC-AE + S1 + residual VTM** | Kiểm tra hướng latent codec đa rate kết hợp residual | Kém MS-ILLM khoảng 1--2.3 Q̂ trong phép so sánh được ghi nhận; loại khỏi hướng chính |
| **SD-Turbo refiner** | Kiểm tra khả năng hậu xử lý/refinement | Không vượt MS-ILLM trong thử nghiệm nội bộ; dừng |
| **COOL-CHIC** | Codec tọa độ/implicit representation, tham khảo đối thủ | Theo ghi chú nội bộ, kém MS-ILLM khoảng 0.4--0.6 Q̂ và encode chậm; tạm dừng |
| **CRDR** | Ứng viên learned codec khác | Kém MS-ILLM theo Q̂ v0 và khoảng 5 s/ảnh giải mã trong phép đo ghi nhận; loại khỏi lựa chọn chính |

Các kết luận loại/giữ chỉ áp dụng cho cấu hình, checkpoint, bitrate và
tập ảnh đã thử; không chứng minh các phương pháp đó kém trong mọi điều
kiện.

### 4.7 PULSE-L — ứng viên decoder nhẹ mới

PULSE kết hợp xử lý đặc trưng ở độ phân giải thấp, pixel renderer và entropy path có thành phần số nguyên. Bản phát hành có checkpoint perceptual base/L và nhiều mức q. Dự án đã thử cả hai, sau đó chọn PULSE-L để đánh giá đầy đủ vì vùng rate và khả năng bổ sung cho mix phù hợp hơn. [Implementation chính thức](https://github.com/microsoft/GenCodec/tree/main/PULSE).

Vai trò hiện tại của PULSE-L là bổ sung cho các codec đã có, chưa thay chúng ở mọi rate. Các con số chất lượng và tốc độ do dự án đo được trình bày tại mục 5.8–5.10; không dùng benchmark phần cứng của tác giả thay cho benchmark L40/L4 của dự án. Entropy số nguyên là đặc tính đáng quan tâm, nhưng parity của toàn pipeline vẫn cần kiểm thực tế.

## 5. Thiết kế thí nghiệm và ý nghĩa

### 5.1 Thí nghiệm A — xây dựng và kiểm định Q̂

**Câu hỏi:** Q̂ có dự đoán preference của người tốt hơn các metric đơn lẻ
không?

**Thiết kế:** lấy dữ liệu pairwise CLIC 2021/2022/2024, tính metric cho
từng phương án trong cặp, fit Bradley--Terry và đánh giá trên dữ liệu
giữ lại/ngoài năm huấn luyện.

| Kết quả được báo cáo | Ý nghĩa | Giới hạn |
| --- | --- | --- |
| **Q̂ v0: cross-year accuracy 0.812** | Train trên 2021t/2021v/2022t, test 2024t | Không phải accuracy trên ảnh nén CLIC 2027 |
| **Trên 2024t: dấu PSNR 0.705; Q̂ v0 cross-year 0.812** | So trên cùng miền test, Q̂ có tín hiệu tốt hơn PSNR đơn lẻ | Cần kiểm lại split theo ảnh và leakage |
| **Trên 8.000 cặp: Q̂ v0 5-fold CV 0.788; dấu PSNR 0.625** | Kết quả gộp khác với phép kiểm tra cross-year | Không trộn hai protocol thành một con số |
| **Q̂ v1: cross-year 0.833** | Accuracy cao hơn chưa đủ để chọn khi ranking ở miền rate thấp không phù hợp | Tiếp tục dùng v0 cho bảng codec hiện tại; v1 chỉ làm phân tích độ nhạy |

Nguồn: [Q̂ v0](D:/AwesomeCV/ratflow-codec/results/2026-10-06_qhat_v0/summary.md) và [Q̂ v1](D:/AwesomeCV/ratflow-codec/results/2026-10-09_qhat_v1/summary.md). Các cặp cùng ảnh có thể phụ thuộc nhau; accuracy cross-year không loại bỏ mọi dạng domain shift. Tỉ lệ đồng ý giữa hai người là chỉ báo độ nhiễu, không phải một trần toán học tuyệt đối của mọi mô hình dự đoán.

**Ý nghĩa khoa học:** thí nghiệm kiểm tra khả năng dùng dữ liệu
preference lịch sử để xây dựng surrogate metric cho quá trình nghiên
cứu.\
**Việc tiếp theo:** tái lập split, kiểm tra rò rỉ dữ liệu, báo cáo
confidence interval và calibration; đánh giá trên nhiều người chấm.

### 5.2 Thí nghiệm B — bake-off nhiều codec trên 30 ảnh

**Câu hỏi:** codec nào có chất lượng cảm nhận tốt nhất tại các rate mục
tiêu, khi ngân sách byte của cả tập bị giới hạn?

**Thiết kế được ghi:** 30 ảnh validation, các mức 0.075/0.15/0.3 bpp,
encode/decode bằng nhiều codec; phân bổ ngân sách byte toàn tập theo Q̂
v0. Kết quả trong `results/2026-10-13_bakeoff_v0`.

| Ứng viên | 0.075 bpp | 0.15 bpp | 0.3 bpp |
| --- | --- | --- | --- |
| **VTM 4:2:0** | −2.515 | −1.163 | −0.152 |
| **VTM-SCC** | −2.404 | −1.102 | −0.004 |
| **MS-ILLM** | −0.451 | +0.545 | +1.383 |
| **CoD-Lite** | −0.382 | +0.285 | +0.906 |
| **DiT-IC** | +0.164 | Chưa có operating point phù hợp tới 0.15 | --- |
| **DiT-IC chưng cất** | −0.000 | --- | --- |
| **Trộn theo ảnh** | **+0.184** | **+0.758** | **+1.421** |

*Lưu ý:* đây là Q̂ v0 nội bộ; ô "---" nghĩa là không có kết quả được báo
cáo cho cấu hình đó, không có nghĩa là điểm bằng 0.

**Ý nghĩa:** kết quả ủng hộ một hệ thống hỗn hợp thay vì codec duy nhất.
Ở 0.075 bpp, DiT-IC có lợi thế trên ảnh tự nhiên; ở 0.15/0.3 bpp,
MS-ILLM bổ sung operating point hữu ích. Tuy nhiên, 30 ảnh là tập nhỏ và
Q̂ có thể sai; cần mở rộng mẫu và xác nhận bằng người.

### 5.3 Thí nghiệm C — chấm mù DiT-IC so với MS-ILLM

**Câu hỏi:** lợi thế Q̂ của DiT-IC có xuất hiện trong đánh giá trực quan
của con người không?

**Thiết kế được báo cáo:** 90 cặp crop 512×512 ở 0.075 bpp; một người
chấm; thêm 10 cặp lặp lại có đảo bên để kiểm tra độ nhất quán.

| Kết quả | Giá trị |
| --- | --- |
| **DiT-IC được chọn** | 62/90 = 68.9% |
| **Preference trên ảnh tự nhiên** | 55/75 = 73.3% |
| **DiT-IC được chọn ở ảnh chữ/màn hình** | 4/12 |
| **Độ nhất quán của 10 cặp lặp** | 7/10 |
| **Q̂ v0 khớp đa số lựa chọn của ba crop/ảnh** | 26/30 ảnh, cùng một người chấm |

Nguồn: [protocol và kết quả chấm mù](D:/AwesomeCV/ratflow-codec/results/2026-10-11_rating_ditic_msillm/summary.md). Các operating point được chọn từ allocation Q̂ v0 ở 0.075 bpp; cùng budget toàn tập không có nghĩa từng cặp ảnh có cùng bpp.

**Ý nghĩa:** đây là bằng chứng độc lập tương đối với Q̂ rằng DiT-IC có
lợi trên ảnh tự nhiên ở rate thấp, đồng thời chỉ ra thất bại có cấu trúc
trên ảnh chữ/màn hình.\
**Giới hạn:** một người chấm không đại diện cho phân bố preference; cặp
crop có thể không độc lập; khoảng tin cậy và cách lấy mẫu cần được rà
soát.\
**Bước tiếp theo:** thêm 2--3 người chấm hoặc nhiều hơn, cân bằng loại
ảnh, ngẫu nhiên hóa thứ tự trái/phải, ghi lại bất đồng và độ nhất quán
liên-chấm.

### 5.4 Thí nghiệm D — chưng cất decoder DiT-IC

**Câu hỏi:** có thể tăng tốc decoder mà không làm mất quá nhiều chất
lượng cảm nhận?

**Thiết kế:** teacher--student; student giảm kênh ở các tầng decoder độ
phân giải cao; huấn luyện bắt chước output teacher bằng L1 + LPIPS, sau
đó thêm GAN loss.

| Cấu hình q | Teacher Q̂ v0 | Student Q̂ v0 | Student − teacher |
| --- | --- | --- | --- |
| **q2** | −0.450 | −0.584 | −0.13 |
| **q3** | +0.157 | −0.012 | −0.17 |
| **q4** | +0.272 | +0.115 | −0.16 |

**Ý nghĩa:** chưng cất giảm độ phức tạp và tăng tốc, nhưng làm giảm Q̂ ở
cả ba operating point. Quyết định dùng teacher hay student phải dựa trên
tốc độ L4 thật, không phải L40.\
**Cần làm:** đánh giá Pareto chất lượng--thời gian; không chọn student
chỉ vì nhanh hơn nếu teacher vẫn qua được giới hạn tốc độ.

### 5.5 Thí nghiệm E — benchmark tốc độ

**Câu hỏi:** toàn bộ đường giải mã có nằm trong giới hạn thời gian
không?

| Phép đo trên L40 được báo cáo | Thời gian |
| --- | --- |
| **DiT-IC teacher, toàn decoder** | 17.5 s / 30 ảnh |
| **DiT-IC student, toàn decoder** | 11.4 s / 30 ảnh |
| **Nạp mạng** | Khoảng 18 s, đo riêng |
| **MS-ILLM** | 0.7--0.8 s/ảnh |
| **CoD-Lite** | Khoảng 0.15 s/ảnh |

Các phép đo trên không cùng điều kiện đo ở mọi hàng, vì vậy không nên
xếp hạng tốc độ chính xác nếu thiếu log và quy trình chuẩn hóa. Đặc
biệt, L40 không phải L4.

**Ý nghĩa:** tốc độ là ràng buộc thiết kế chứ không phải chỉ số phụ.
Benchmark cần báo cả cold start, warm decode, thời gian encode nếu luật
yêu cầu, thời gian giải mã toàn bộ tập, peak GPU memory, RAM và số CPU.

### 5.6 Thí nghiệm F — tính tất định của entropy coding

**Câu hỏi:** cùng một bitstream có tạo ra cùng latent/ảnh trên phần cứng
khác nhau không?

| Thiết lập | Kết quả nội bộ được ghi |
| --- | --- |
| **CPU ↔ GPU, fp32** | Lệch đáng kể, ảnh hỏng |
| **CPU ↔ GPU, fp64** | Khớp từng bit, nhưng entropy decode khoảng 1.5 s/ảnh |
| **GPU 0 ↔ GPU 1 cùng máy L40, fp32** | Khớp từng bit |
| **L40, tiến trình mới trên cùng máy, môi trường dựng theo devkit** | 0 index/symbol lệch trên 30 ảnh q3; chưa phải hai máy độc lập |

Nguồn: [kiểm entropy sáu ảnh](D:/AwesomeCV/ratflow-codec/results/2026-10-12_ditic_determinism/summary.md) và [kit chạy trên L40](D:/AwesomeCV/ratflow-codec/results/2026-10-13_l4_kit_l40/summary.md). Kết quả fp64 chỉ áp dụng cho cấu hình đã đo; chưa chứng minh an toàn trên mọi thiết bị. Khớp index/symbol entropy, khớp bitstream và khớp pixel PNG là ba phép kiểm khác nhau.

**Ý nghĩa:** các phép toán floating point có thể khác nhau giữa phần
cứng/precision và làm xác suất hoặc symbol entropy decode lệch, dẫn đến
giải mã sai. Mạng số nguyên và CDF số nguyên có thể cải thiện tính tất
định, nhưng phải kiểm tra toàn bộ đường đi của từng codec.

**Bước tiếp theo:** encode trên máy L4 A, decode trên máy L4 B; so sánh
checksum của latent và ảnh đầu ra, kích thước ảnh, số file, log lỗi và
version môi trường. Nếu không đạt, cân nhắc entropy coding số nguyên
hoặc cố định một đường giải mã tương thích.

### 5.7 Thí nghiệm G — so sánh ảnh màn hình và baseline cổ điển

Các phép thử ban đầu ghi nhận VVC-SCC tốt hơn VTM 4:2:0 khoảng 1.1--3.3
dB trên ảnh màn hình; DC-AE có thể làm hỏng chữ nhỏ (CER được ghi là
0.58 trong một phép đo trước). Đây là lý do thiết kế mode chuyên biệt
cho screen content.

**Ý nghĩa:** đánh giá trung bình toàn tập có thể che giấu thất bại trên
một nhóm ảnh. Cần báo cáo metric theo nhóm: ảnh tự nhiên, chữ/màn hình,
texture, ảnh có cạnh mảnh; sau đó đánh giá bộ chọn mode bằng confusion
matrix, precision/recall và preference cuối cùng.

### 5.8 Thí nghiệm H — sàng lọc model mới trên sáu ảnh

**Câu hỏi:** model mới công khai có bổ sung chất lượng hoặc tốc độ hữu ích cho các codec hiện có không?

Pilot cố định trước sáu ảnh: ba ảnh tự nhiên (`02809272`, `0369d229`, `07b9f93f`), texture (`937476dd`), hai ảnh screen (`bb7344a2`, `ebfd571f`). Chạy nguyên kích thước, lưu bitstream thật, đọc lại bitstream để decode, kiểm kích thước ảnh và chấm bằng metric chung của repo. Bộ sáu ảnh chỉ dùng sàng lọc, không đại diện đủ phân phối validation.

| Ứng viên | Số điểm | Q̂ @ 0.075 | Q̂ @ 0.15 | Q̂ @ 0.3 | Trung vị decode/ảnh |
|---|---:|---:|---:|---:|---:|
| AFP-GIC, q0/1/4 | 18 | -0.7304 | 0.3158 | 0.8712 | 7.00 s |
| DC-VIC PatchGAN, q0/1/4 | 18 | Không vừa | 0.4599 | 1.0237 | 9.49 s |
| PULSE perceptual, q0/2/4/7 | 24 | Không vừa | Không vừa | Không vừa | 51.3 ms |
| PULSE-L perceptual, q0/2/4/7 | 24 | Không vừa | Không vừa | 1.6057 | 63.0 ms |
| MS-ILLM trên cùng sáu ảnh | Baseline đã có | -0.3030 | 0.6423 | 1.7323 | Không đo lại |
| Mix baseline trên cùng sáu ảnh | Baseline đã có | 0.3806 | 1.0409 | 1.8629 | Chưa đo toàn mix |
| Mix thêm cả bốn model | Allocation mới | 0.3806 | 1.0560 | 1.9194 | Chưa đo toàn mix |

“Không vừa” nghĩa là ngay tổ hợp điểm ít byte nhất đang có cũng vượt ngân sách, không phải lỗi chạy hoặc điểm bằng 0. Q̂ tối ưu dưới budget không yêu cầu dùng hết budget; AFP/DC-VIC chưa có đủ điểm cao để tận dụng toàn bộ 0.3. Không so trực tiếp bảng sáu ảnh với bảng 30 ảnh.

AFP/DC-VIC được chạy trên Torch 2.11.0+cu128, CompressAI 1.2.8; PULSE dùng Torch 2.6.0+cu124. Do khác runtime và chưa tối ưu cùng điều kiện, thời gian trên chỉ để sàng lọc. Các phép thử không được gọi là tái lập đầy đủ kết quả paper.

**Kiểm soát lỗi:** checkpoint AFP/DC-VIC cần đổi tên các entropy parameters của CompressAI cũ, sau đó load bằng `strict=True`. Một smoke AFP trước khi sửa đã bỏ sót trọng số; toàn bộ kết quả smoke đó bị loại. Bảng này chỉ dùng các run `pilot_afp_strict` và `pilot_dcvic_strict` hợp lệ.

**Quyết định:** chưa mở rộng AFP/DC-VIC vì chất lượng pilot chưa vượt baseline và decoder chậm; PULSE base có vùng rate chưa phù hợp. PULSE-L có lợi thế tốc độ và khả năng bổ sung cho mix, nên được chọn chạy đủ 30 ảnh. [Bảng pilot gốc](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/pilot_scores/summary.csv).

### 5.9 Thí nghiệm I — PULSE-L trên đủ 30 ảnh

Chạy q0/1/2/4, tổng cộng **120 điểm**. So sánh cùng bộ ảnh, cùng Q̂ v0, cùng allocator và cùng ngân sách byte với baseline. Mix mở rộng trong bảng này chỉ thêm **PULSE-L**, không thêm AFP/DC-VIC/PULSE base vì chúng mới được chấm trên sáu ảnh.

| Ứng viên | 0.075 bpp | 0.15 bpp | 0.3 bpp |
|---|---:|---:|---:|
| DiT-IC hiện có | 0.1644 | 0.2788* | 0.2788* |
| MS-ILLM | -0.4513 | 0.5447 | 1.3831 |
| PULSE-L perceptual | Không vừa budget | Không vừa budget | 1.4030 |
| Mix baseline | 0.1836 | 0.7582 | 1.4215 |
| **Mix baseline + PULSE-L** | **0.1841** | **0.7891** | **1.4912** |
| **Chênh lệch mix mới − cũ** | **+0.0006** | **+0.0309** | **+0.0697** |

*DiT-IC dùng các điểm hiện có và chỉ tiêu thụ khoảng 0.0958 bpp, dù budget lớn hơn. Đây không phải checkpoint DiT-IC mới đạt 0.15/0.3 bpp.*

| Budget | DiT-IC | MS-ILLM | PULSE-L | VTM-SCC | Bpp thực của mix mới |
|---|---:|---:|---:|---:|---:|
| 0.075 | 28 | 0 | 1 | 1 | 0.074915 |
| 0.15 | 14 | 4 | 10 | 2 | 0.149904 |
| 0.3 | 1 | 7 | 20 | 2 | 0.299591 |

Số trong bốn cột codec là số ảnh được allocator chọn. Đây là chọn theo điểm đã encode/chấm, chưa phải một bộ phân loại nội dung học được. Ở encoder có ảnh gốc để tính metric và chọn mode; decoder chỉ nhận bitstream và metadata cần thiết.

PULSE-L q0 trên cả tập đã tốn **0.1921 bpp**, nên checkpoint hiện tại chưa dùng độc lập được ở 0.075/0.15. Lợi ích ở 0.075 chỉ +0.0006 Q̂, chưa đáng thêm một họ decoder chỉ vì điểm số này. Ở 0.15/0.3, việc PULSE-L được chọn cho nhiều ảnh là cơ sở để tiếp tục kiểm tra bằng người và benchmark triển khai.

**Giới hạn:** allocation tối ưu trên cùng proxy đang được dùng để báo cáo; chưa có holdout mới, bootstrap CI hoặc chấm mù mix mới. Byte từng stream đã được tính, nhưng metadata chọn codec chung và kích thước/tải toàn bộ decoder chưa được đo trong một gói submission. Không diễn giải ΔQ̂ thành phần trăm tăng chất lượng hay phần trăm thắng người chấm.

Nguồn: [summary 30 ảnh](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/full_scores/summary.csv), [allocation](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/full_scores/allocation.csv), [metric từng điểm](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/full_scores/metrics.csv).

### 5.10 Thí nghiệm J — tốc độ, VRAM và decode độc lập của PULSE-L

| Đại lượng | Kết quả | Phạm vi |
|---|---:|---|
| Trung vị decode | 57.2 ms/ảnh | 120 điểm trên L40, hai CPU thread |
| Khoảng decode | 30.8–121.4 ms/ảnh | API decode, chưa gồm I/O và load |
| Tổng decode PULSE-L riêng tại 0.3 | 1.762 s/30 ảnh | Tổng các điểm được chọn, không phải wall time bài nộp |
| Nạp model | 0.730 s | Run full trên Torch 2.6 |
| Đỉnh tổng VRAM quan sát | 4,304 MiB | Có cả dịch vụ đang dùng GPU |
| Decode từ tiến trình mới | 3/3 ảnh khớp pixel | Cùng server L40; decoder không dùng ảnh gốc |

Phép đo tốc độ trong run full xảy ra sau encode trong cùng tiến trình, không phải cold-start. Khi kiểm lại ba stream trong tiến trình mới, thời gian decode lần lượt khoảng 402/131/110 ms. Không cộng cơ học thời gian từ các protocol khác nhau để tuyên bố đạt tốc độ bài thi.

Đỉnh VRAM các run pilot lần lượt là AFP-GIC 11,264 MiB, DC-VIC 7,718 MiB, PULSE 3,130 MiB và PULSE-L 4,140 MiB. Tất cả giữ guard nửa thiết bị, chặt hơn mức 30 GB/GPU mới được cho phép. Guard theo dõi tổng bộ nhớ và dừng job của mình trước ngưỡng, nhưng không phải phân vùng GPU cứng.

Nguồn: [manifest](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/manifest.json) và [kiểm decode từ tiến trình mới](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/fresh_decode/checks.json). Ba phép kiểm này chưa chứng minh parity CPU↔GPU hoặc L40↔L4. Tổng vòng mới có 204 điểm encode/decode hợp lệ và ba phép kiểm bổ sung; chưa train checkpoint mới.

## 6. Kết quả tổng hợp và cách diễn giải

### 6.1 Các phát hiện hiện tại

1.  **Trộn theo ảnh có triển vọng:** Q̂ v0 của phương án trộn được báo
    cáo cao nhất tại cả ba rate.
2.  **DiT-IC có tín hiệu tốt ở bitrate thấp:** bake-off và chấm mù đều
    ủng hộ trên ảnh tự nhiên.
3.  **MS-ILLM vẫn là base quan trọng:** ở 0.15/0.3 bpp, đây là thành
    phần mạnh trong phương án hiện tại.
4.  **Một codec không phù hợp mọi ảnh:** VTM-SCC có vai trò tiềm năng
    cho ảnh chữ/màn hình.
5.  **Chưng cất tạo đánh đổi rõ ràng:** student nhanh hơn nhưng Q̂ thấp
    hơn teacher.
6.  **Tính tất định và đóng gói là blockers:** chưa được xác nhận trên
    hai máy L4 độc lập và môi trường submission.
7.  **PULSE-L bổ sung được cho mix:** ΔQ̂ là +0.0309/+0.0697 ở 0.15/0.3 trên 30 ảnh; decoder nhẹ là lý do tiếp tục đầu tư, nhưng chưa có bằng chứng người chấm cho mix mới.
8.  **Model mới không tự động tốt hơn:** AFP-GIC/DC-VIC chưa vượt baseline trong pilot; PULSE base chưa đáp ứng budget. Sàng lọc bằng byte thật giúp tránh dành train dài cho cấu hình chưa phù hợp.

### 6.2 Những điều chưa thể kết luận

-   Chưa thể nói hệ thống sẽ thắng hoặc đạt thứ hạng cụ thể tại CLIC
    2027.
-   Chưa thể kết luận thời gian decode đạt mục tiêu trên L4 từ benchmark
    L40.
-   Chưa thể coi Q̂ là thước đo thay thế hoàn toàn cho người chấm.
-   Chưa thể coi các nhánh codec đã được tích hợp chỉ vì mỗi ứng viên
    chạy được riêng trong bake-off.
-   Chưa thể xem giới hạn 4 GB là đạt khi tổng trọng số q2/q3/q4 và các
    codec khác chưa được kiểm kê trong artifact cuối.

## 7. Pipeline phần mềm và tái lập thí nghiệm

Repo tổ chức pipeline GPU trong `experiments/pipeline.py`. Pipeline có
các stage chính:

-   `check`: kiểm tra GPU, Internet và package.
-   `data`, `cache`: tùy chọn tải dữ liệu DIV2K và cache latent.
-   `s1`, `s1_eval`: train/evaluate hướng S1 cũ.
-   `parity`: kiểm tra parity giữa implementation và dependency tham
    khảo.
-   `qhat`: fit Q̂ từ dữ liệu preference.
-   `bakeoff`: chạy các ứng viên trên 30 ảnh và phân bổ byte theo Q̂.
-   `pack`: gom summary và artifact vào file kết quả.

Pipeline có `state.json` để bỏ qua stage đã hoàn thành khi chạy lại, và
đóng gói kết quả thành `results.zip`. Repo cũng có module metric chung
(`src/clic27/eval/metrics.py`), entropy coding (`src/clic27/entropy/`),
mô hình (`src/clic27/nn/`) và test CPU.

``` mermaid
flowchart LR
    A["Code + tests"] --> B["tools/bundle.py"]
    B --> C["clic27_run.py / notebook"]
    C --> D["check"]
    D --> E["qhat"]
    E --> F["bakeoff"]
    F --> G["pack results.zip"]
    G --> H["Lưu summary / CSV / JSON"]
    H --> I["Phân tích và quyết định vòng tiếp theo"]
```

Để báo cáo kết quả có thể tái lập, mỗi run nên lưu: commit SHA, phiên
bản Q̂ và file model, danh sách ảnh, cấu hình bitrate, lệnh chạy,
checkpoint, phiên bản Python/PyTorch/CUDA, tên GPU, thời gian nạp/giải
mã, peak memory, checksum bitstream và ảnh đầu ra.

Vòng 15/10 dùng script riêng trong `experiments/new_codecs`, chưa tích hợp vào CLI bake-off chính. Đã lưu revision của code tác giả, hash trọng số, JSON mỗi điểm, trace VRAM, metric và allocation tại `results/2026-10-15_new_codecs`. Model, ảnh tái tạo và stream lớn còn trên server tại `/mnt/home/guess03/clic/research_20261015`. Không coi file student hoặc thư mục checkpoint riêng là gói decoder hoàn chỉnh.

Các script đã qua kiểm tra cú pháp và guard bộ nhớ; bằng chứng chức năng là round-trip thực cùng kiểm decode độc lập. Khi đổi code/checkpoint/config phải tạo tên run mới, không resume vào thư mục kết quả cũ. Cần thêm kiểm toàn gói offline và phục hồi artifact trước khi khóa bài nộp.

## 8. Rủi ro, giới hạn và kế hoạch giảm thiểu

| Rủi ro | Mức | Hành động giảm thiểu |
| --- | --- | --- |
| **Decoder/mix quá chậm trên L4** | Cao | Đo cold-start và tổng wall time từng họ; so teacher/student/PULSE-L, không mặc định MS-ILLM nhanh hơn |
| **Bitstream khác nhau giữa máy** | Cao | Kiểm tra L4↔L4; cố định environment; thay entropy path bằng phép toán số nguyên khi cần |
| **Tổng decoder vượt 4 GB** | Cao | Chia sẻ trọng số giữa q-level, lưu delta hoặc giảm số mức; đo tổng artifact cuối |
| **Allocator tối ưu proxy nhưng người không thích hơn** | Trung bình--cao | Chấm mù toàn tập và theo nhóm; nếu phát triển bộ chọn nhanh thì đo sai số so với lựa chọn từ RD đã biết |
| **Q̂ khái quát kém** | Trung bình--cao | Thêm người chấm, mở rộng ảnh, kiểm tra calibration và confidence interval |
| **Không tích hợp được các codec vào một decoder** | Cao | Định nghĩa format/header và mode ID; kiểm thử end-to-end từ bitstream đến ảnh |
| **Thiếu artifact hoặc mất checkpoint** | Cao | Sao lưu checkpoint, latent và streams; kiểm tra khôi phục từ bản sao |
| **Giấy phép pretrained weights không phù hợp** | Trung bình--cao | Rà soát code, weights, model card và điều khoản cuộc thi trước khi nộp |

## 9. Hướng nghiên cứu và kế hoạch tiếp theo

### 9.1 Các phương pháp mới và bài CLIC cũ

| Nguồn chính thức | Hướng tiếp cận | Trạng thái và quyết định |
|---|---|---|
| [PULSE](https://github.com/microsoft/GenCodec/tree/main/PULSE) | Decoder nhẹ, variable-rate, entropy số nguyên | Đã thử base/L; ưu tiên tích hợp L và nghiên cứu vùng rate thấp |
| [AFP-GIC](https://github.com/yifeipet/AFP_GIC) | Kết hợp prior sinh ảnh, một checkpoint nhiều rate | Pilot chưa vượt baseline; chưa mở train dài |
| [DC-VIC](https://github.com/iwa-shi/DC_VIC) | Khai thác codebook sinh ảnh bằng dual conditioning | Đã thử PatchGAN; giữ làm tham khảo phương pháp |
| [PICO](https://github.com/apple-aiml-research/ml-pico) | Tối ưu cảm nhận người xem cùng hiệu năng triển khai | Đã rà repo: viewer/ảnh kết quả, chưa tìm được runtime và weights để chạy trên tập của dự án |
| [ProGIC](https://arxiv.org/abs/2603.02897) | Progressive RVQ và backbone nhẹ | Có paper; chưa xác minh được checkpoint chạy được |
| [AEIC](https://github.com/LuizScarlet/AEIC) | Ultra-low bitrate với shallow encoder | Có repo; chưa thử, cần kiểm chi phí decoder trước |
| [PORT — CLIC 2025](https://www.jdl.link/doc/2011/20251221_232_32_18.PORT%20Perception-Oriented%20Image%20Compression.pdf) | Decoder nhanh, loss semantic/patch và phân bổ theo ROI | Tìm được paper của PKUSZ-AliMerlin; chưa tìm checkpoint công khai để thử |
| [Cool-Chic — tag CLIC 2025](https://github.com/Orange-OpenSource/Cool-Chic/releases) | Tối ưu từng ảnh, decoder C nhẹ | Đã có phép thử trong repo ngày 09/10; chưa chạy lại trong vòng mới |

Cool-Chic CLIC2025 trên ảnh `2684452d` mất khoảng 1 giờ 39 phút encode, đạt Q̂ v0 1.33 tại 0.557 bpp; MS-ILLM cùng ảnh đạt 1.46 tại 0.267 bpp. Đây là bằng chứng trên một ảnh và cấu hình cụ thể, đủ để chưa ưu tiên quét hàng loạt, không đủ để loại mọi cấu hình Cool-Chic. Nếu thử tiếp, giới hạn ở screen/texture và lambda phù hợp 0.075–0.3. [Kết quả Cool-Chic cũ](D:/AwesomeCV/ratflow-codec/results/2026-10-09_coolchic2025_probe/summary.md).

Tìm được paper không đồng nghĩa có thể reproduce. Các mục “chưa tìm checkpoint” phản ánh phạm vi tìm kiếm hiện tại, không khẳng định checkpoint không tồn tại. Việc đánh giá trên bộ dữ liệu CLIC cũng không tự chứng minh một model là bài dự thi CLIC.

### 9.2 Hướng A — lựa chọn codec có tính chi phí triển khai

**Giả thuyết:** lựa chọn theo từng ảnh có thể đạt chất lượng tốt hơn tại cùng ngân sách byte và thời gian nếu tính cả chi phí nạp từng họ decoder. Bằng chứng khởi đầu là PULSE-L được chọn trên 10/30 và 20/30 ảnh ở hai budget cao.

Giữ các lựa chọn $j_i$ ở mục 3 và bổ sung ràng buộc thời gian:

$$
\sum_{i=1}^{N}t_{i,j_i}+\sum_{c\in\mathcal C}a_c\ell_c
\leq T_{\max}.
$$

Trong đó $t_{i,j_i}$ là thời gian decode ảnh, $\ell_c$ là chi phí nạp codec $c$, và $a_c$ bằng 1 khi có ít nhất một ảnh dùng codec đó. Đây là mô hình xấp xỉ để chọn phương án; quyết định cuối vẫn phải đo wall time của gói thật, kể cả I/O và khởi động. Có thể bổ sung hình phạt độ bất định chất lượng sau khi có dữ liệu calibration độc lập.

**Ablation:** A0 chỉ tối ưu byte; A1 thêm thời gian decode; A2 thêm chi phí load dùng chung; A3 thêm độ bất định của Q̂ nếu đủ dữ liệu. Giữ nguyên checkpoint, tập ảnh và tập điểm RD. Báo đường Pareto chất lượng–thời gian, kích thước decoder và preference người thật.

**Tính mới cần kiểm chứng:** routing và knapsack đã có từ trước. Đóng góp tiềm năng nằm ở cách phối hợp chi phí dùng chung, sai số của surrogate và điều kiện triển khai; chưa đủ cơ sở gọi đây là phương pháp mới trước khi rà related work và có ablation.

### 9.3 Hướng B — PULSE-L ở rate thấp với distillation theo độ tin cậy

**Giả thuyết:** fine-tune codec nhanh tới vùng 0.075/0.15, kết hợp teacher cho texture và loss trung thực ảnh gốc cho chữ/mặt, có thể giữ chất lượng tốt mà tránh decoder sinh ảnh nặng.

Bắt đầu từ PULSE-L đã chạy được, fine-tune RD cùng gain/rate conditioning và entropy model phù hợp. Không ngoại suy q âm rồi xem đó là operating point hợp lệ. Teacher nên cung cấp mục tiêu trong image space; latent của DiT-IC và PULSE không tương thích mặc định. Nếu dùng projection giữa latent thì projection đó phải được thiết kế, train và tính vào chi phí.

| Nhánh | Thay đổi duy nhất | Câu hỏi kiểm tra |
|---|---|---|
| B0 | Fine-tune RD từ PULSE-L | Có đạt rate thấp bằng stream thật và giữ decoder nhanh không? |
| B1 | B0 + distillation đồng đều | Teacher có giúp hơn fine-tune RD đơn thuần không? |
| B2 | B1 + trọng số theo vùng/độ tin cậy | Có giảm lỗi chữ/mặt mà giữ lợi ích texture không? |

So cùng số update, dữ liệu, crop và seed trong pilot; đo GPU-giờ riêng vì chi phí mỗi bước khác nhau. Cache teacher offline nếu giúp giữ giới hạn 30 GB/GPU. Smoke 200–500 bước trước, rồi tối đa 6 GPU-giờ/nhánh cho vòng sàng lọc. Đây là ngân sách đề xuất, chưa phải các run đã khởi chạy. Dùng nhiều seed khi xác nhận finalist.

ROI và điều khiển hallucination đã có trong [PORT](https://www.jdl.link/doc/2011/20251221_232_32_18.PORT%20Perception-Oriented%20Image%20Compression.pdf), [Conditional Hallucinations](https://arxiv.org/abs/2410.19493) và [Region-Adaptive Generative Compression](https://arxiv.org/abs/2604.01122). Vì vậy, thêm ROI mask chưa đủ là đóng góp mới. Cần kiểm chứng cụ thể lợi ích của độ tin cậy teacher khi chuyển chất lượng xuống decoder nhanh, đặc biệt ở bitrate thấp.

### 9.4 Hướng C — refinement theo vùng với byte thật

**Giả thuyết:** dành residual/refinement cho vùng bị sai có ý nghĩa với người xem sẽ hiệu quả hơn phân bổ đồng đều. Ưu tiên encoder-side refinement để hạn chế tăng chi phí decoder; mọi mask, vị trí patch, gain hoặc latent sửa phải truyền và tính byte.

Ablation gồm không refinement, refinement đồng đều và refinement theo vùng dưới cùng ngân sách byte và encode time. Chỉ giữ nhánh khi có lợi trên holdout/chấm mù và không phá tất định. Adaptive residual/ROI không mới tự thân; đóng góp cần ở tiêu chí chọn vùng và mức cải thiện rate–quality–time. Nhánh này đứng sau A/B để tránh phân tán tài nguyên.

### 9.5 Lộ trình thực hiện và điều kiện chuyển giai đoạn

Các khoảng ngày dưới đây là **mục tiêu nội bộ**, không phải deadline cuộc thi. Việc có máy L4 là một phụ thuộc thực tế cần được bố trí riêng.

| Giai đoạn | Công việc | Sản phẩm và cổng quyết định |
|---|---|---|
| 16–18/10 | Tích hợp PULSE-L vào bake-off; khóa manifest baseline; lưu checkpoint/stream và thử phục hồi | Chạy lại bảng 30 ảnh; không ghi đè kết quả cũ; kiểm đủ byte/header/mode |
| 19–23/10 | Đóng gói decoder offline, profile load/decode/I/O/RAM; chuẩn bị hoặc chạy L4 A/B khi có máy | Report tổng wall time, kích thước artifact; tách entropy parity và PNG parity |
| 19–27/10 | Chấm mù mix mới với mix cũ, thêm 2–3 người; tạo dev/holdout độc lập | Preference theo ảnh/nhóm; bootstrap theo ảnh, không coi crop là mẫu độc lập |
| 24–31/10 | Thử A0/A1/A2; smoke B0/B1/B2 dưới 30 GB/GPU | Đường Pareto, actual-rate, chi phí GPU-giờ; chọn tối đa hai nhánh để mở rộng |
| Tháng 11 | Train finalist trên train/dev; kiểm lại holdout và gói triển khai | Chỉ mở train dài khi chất lượng, tốc độ và khả năng decode có bằng chứng |

Chấm mù cần ngẫu nhiên hóa trái/phải, thêm cặp lặp, có ảnh gốc, kết hợp xem toàn ảnh và crop. Tách một phân tích tập trung vào ảnh đổi codec để chẩn đoán với phân tích trên toàn tập để ước lượng lợi ích tổng thể; không chỉ báo nhóm ảnh có lợi. Kiểm split Q̂ theo ảnh/nguồn codec và giữ validation chính thức ngoài training.

Không cần chờ L4 để làm mọi pilot có giới hạn trên L40. Tuy nhiên, không mở rộng train dài hoặc khóa bài nộp chỉ dựa vào tốc độ L40. Nếu phương án chất lượng cao không đáp ứng triển khai, chọn điểm Pareto tốt hơn thay vì mặc định dùng student. Kiểm kê trọng số theo từng gói decoder/rate, rà giấy phép và chạy end-to-end offline trước khi đăng ký validation.

## 10. Các vấn đề xin ý kiến người hướng dẫn

1. **Đóng góp trọng tâm:** ưu tiên bài toán lựa chọn codec có chi phí load/thời gian (A), hay ưu tiên một codec nhanh mới ở bitrate thấp (B)? Đề xuất: A tạo kết quả hệ thống sớm; B là nhánh nghiên cứu sâu.
2. **Bằng chứng chất lượng:** cần bao nhiêu người chấm và bao nhiêu ảnh holdout để quyết định mở train dài? Đề xuất khởi đầu thêm 2–3 người, phân tầng nội dung và báo bất định theo ảnh.
3. **Trade-off tốc độ:** có chấp nhận giảm một phần Q̂ để giảm cold-start và dung lượng decoder không? Quyết định nên dựa trên đường Pareto và kết quả L4, không đặt một mức mất chất lượng tùy ý.
4. **Nguồn lực:** bố trí thời gian trên hai máy L4 độc lập để kiểm deployment và parity; nghiên cứu chính tiếp tục trên 2 L40, tối đa 30 GB/GPU.
5. **Mức độ mới:** phần nào đủ để phát triển thành đóng góp nghiên cứu ngoài một hệ thống dự thi tốt? Cần rà prior art cho routing, teacher confidence và adaptive residual trước khi chốt tuyên bố novelty.

## 11. Kết luận và tài liệu đối chiếu

Dự án đã có bằng chứng ủng hộ hệ thống nhiều codec: DiT-IC có tín hiệu tốt ở rate thấp; MS-ILLM và PULSE-L bổ sung ở rate cao hơn; VTM-SCC giữ vai trò trên một số nội dung screen. Kết quả mới nhất nâng Q̂ mix lên 0.7891/1.4912 tại 0.15/0.3 bpp, nhưng chưa xác nhận bằng người. Ưu tiên tiếp theo là tích hợp PULSE-L, chấm mù và đo gói decoder thật, đồng thời sàng lọc có giới hạn các giả thuyết A/B trên server.

### 11.1 Bằng chứng trong repo

| Nội dung | Artifact |
|---|---|
| Q̂ v0 và protocol cross-year/CV | [Summary Q̂ v0](D:/AwesomeCV/ratflow-codec/results/2026-10-06_qhat_v0/summary.md) |
| Q̂ v1 | [Summary Q̂ v1](D:/AwesomeCV/ratflow-codec/results/2026-10-09_qhat_v1/summary.md) |
| Baseline trước vòng model mới | [Bake-off 13/10](D:/AwesomeCV/ratflow-codec/results/2026-10-13_bakeoff_v0/summary.md) |
| Chấm mù một người | [Rating 11/10](D:/AwesomeCV/ratflow-codec/results/2026-10-11_rating_ditic_msillm/summary.md) |
| Chưng cất q3 | [Snapshot GAN 30k](D:/AwesomeCV/ratflow-codec/results/2026-10-11_ditic_distill/stage2_gan01/snap_30000/summary.md) |
| Entropy determinism | [Kiểm tra 12/10](D:/AwesomeCV/ratflow-codec/results/2026-10-12_ditic_determinism/summary.md) |
| Kit trong stack theo devkit, phần cứng L40 | [Kit 13/10](D:/AwesomeCV/ratflow-codec/results/2026-10-13_l4_kit_l40/summary.md) |
| Model mới, checksum, VRAM, số điểm | [Manifest 15/10](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/manifest.json) |
| Bảng cập nhật 30 ảnh | [Summary CSV](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/full_scores/summary.csv) |
| Nghiên cứu và pilot chi tiết | [Report model mới](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_new_models_2026-10-15.md) |
| Kế hoạch toàn chiến dịch | [Plan v3](<D:/AwesomeCV/ratflow-codec/plans/Plan CLIC 2027 v3 - execution 2026-10-15.md>) |

Các bảng cập nhật lấy từ artifact đã lưu đến ngày 15/10/2026. Các thí nghiệm A/B/C trong kế hoạch chưa được thực hiện; số liệu của chúng không nằm trong các bảng kết quả hiện tại.

### 11.2 Nguồn bên ngoài

- [Repo dự án](https://github.com/KhoiTrant68/clic2027-image).
- [Luật và tập dữ liệu CLIC 2027](https://clic2027.compression.cc/tasks/).
- [MS-ILLM — ICML 2023, PMLR](https://proceedings.mlr.press/v202/muckley23a.html); [paper đầy đủ](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf).
- [DiT-IC — implementation chính thức](https://github.com/Eric-qi/DiT-IC), [SANA](https://arxiv.org/abs/2410.10629), [DC-AE](https://arxiv.org/abs/2410.10733).
- [PULSE — code/checkpoint/training](https://github.com/microsoft/GenCodec/tree/main/PULSE).
- [MBT2018](https://arxiv.org/abs/1809.02736).
- Các nguồn AFP-GIC, DC-VIC, PICO, ProGIC, AEIC, PORT và Cool-Chic được liên kết tại mục 9.1; prior art liên quan ROI/hallucination tại mục 9.3.

## Phụ lục A. Nền tảng và phương pháp MS-ILLM

Phụ lục này giữ phần giải thích phương pháp của bản báo cáo ban đầu, đã chuẩn hóa ký hiệu và bố cục. Đây là mô tả **phương pháp của tác giả MS-ILLM**, không phải kiến trúc mới do dự án đề xuất hoặc một lần training dự án đã thực hiện. Tham chiếu: [paper MS-ILLM, mục 3–4.5](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf).

### A.1 Rate, distortion và statistical fidelity

Nén ảnh cần cân bằng số bit với mức độ giữ lại thông tin thị giác. Với ảnh gốc $x$ và ảnh tái tạo $\hat{x}$, ba khái niệm cần phân biệt là:

- **Rate $R$:** lượng thông tin để biểu diễn ảnh; bpp thực tính từ số bit của stream chia số pixel.
- **Distortion $D$:** sai khác với chính ảnh gốc. Ví dụ MSE trên $M$ giá trị pixel/kênh là $D(x,\hat{x})=M^{-1}\sum_{m=1}^{M}(x_m-\hat{x}_m)^2$.
- **Statistical fidelity:** mức độ gần nhau giữa phân phối ảnh tái tạo $P_{\hat X}$ và phân phối ảnh thật $P_X$.

Ảnh có texture tự nhiên chưa chắc có các chi tiết khớp ảnh gốc. Ngược lại, tối ưu sai số pixel có thể tạo ảnh mượt, ít texture. Vì vậy cần đánh giá cả độ trung thực với ảnh gốc và chất lượng cảm nhận, nhất là khi ảnh chứa chữ hoặc khuôn mặt.

Một cách biểu diễn bài toán lý thuyết là:

$$
R(\tau,\delta)=\min_{P_{\hat X\mid X}} I(X;\hat X)
\quad\text{với}\quad
\mathbb E[\rho(X,\hat X)]\leq\tau,\qquad
d(P_{\hat X},P_X)\leq\delta.
$$

$\rho$ là hàm distortion, $d$ là độ khác biệt phân phối, $\tau$ và $\delta$ là các ngưỡng. Đây là mô tả lý thuyết; rate ước lượng lúc train và số byte stream thực lúc đánh giá không phải cùng một phép đo.

Trong training neural codec, có thể diễn đạt mục tiêu tổng quát bằng:

$$
\mathcal L_{\mathrm{codec}}=
\lambda_R R+\lambda_D D+\lambda_G\mathcal L_{\mathrm{adv}}.
$$

Các trọng số điều khiển trade-off. Tín hiệu đối kháng là cách học gián tiếp để cải thiện thống kê ảnh; không phải phép tính trực tiếp một divergence đầy đủ giữa hai phân phối, cũng không bảo đảm mọi chi tiết sinh ra đều đúng.

### A.2 Mean-Scale Hyperprior làm nhiệm vụ nén

Encoder chuyển ảnh $x$ thành latent $y$; lượng tử hóa tạo latent rời rạc $\hat y$. Hyperprior cung cấp thông tin để dự đoán trung bình và scale của phân phối latent, từ đó entropy coder tạo bitstream. Decoder tái tạo ảnh từ latent đã giải mã.

Nếu phân phối xác suất dự đoán phù hợp hơn với latent, entropy coding có thể dùng ít bit hơn. ILLM không thay thế đường nén này: nó bổ sung tín hiệu huấn luyện để cải thiện ảnh đầu ra. Khi triển khai, discriminator và bộ tạo nhãn dùng trong training không mặc nhiên phải đi cùng decoder.

### A.3 VQ-VAE tạo nhãn cục bộ

VQ-VAE có codebook gồm $C$ vector $\{m_1,\ldots,m_C\}$. Tại vị trí không gian $(u,v)$, encoder tạo đặc trưng $e^{(u,v)}$ và chọn code gần nhất:

$$
c^*(u,v)=\arg\min_{c\in\{1,\ldots,C\}}
\left\|e^{(u,v)}-m_c\right\|_2^2.
$$

Các chỉ số codebook tạo thành bản đồ nhãn $u(x)$ của ảnh thật. Chúng là nhãn đặc trưng được học, không phải các lớp ngữ nghĩa do người gán như “mèo”, “cây” hay “chữ”. Trong thiết lập mặc định được mô tả ở bản báo cáo, crop $256\times256$ tạo bản đồ $32\times32$, với codebook 1024 vector.

### A.4 Discriminator U-Net và hai hàm loss

Discriminator $D_\phi$ dự đoán $C+1$ lớp ở mỗi vị trí: $C$ lớp codebook và một lớp fake. Với ảnh thật, mục tiêu là nhãn cục bộ $u(x)$; với ảnh tái tạo, mục tiêu của discriminator là bản đồ fake $b_0$. Dùng biểu diễn one-hot, có thể viết gọn:

$$
\mathcal L_D=
-\mathbb E_x\!\left[\langle u(x),\log D_\phi(x)\rangle\right]
-\mathbb E_x\!\left[\langle b_0,\log D_\phi(\hat x)\rangle\right].
$$

Phần codec đóng vai trò generator, muốn ảnh tái tạo được dự đoán theo nhãn cục bộ của ảnh gốc:

$$
\mathcal L_G=
-\mathbb E_x\!\left[\langle u(x),\log D_\phi(\hat x)\rangle\right].
$$

Tích vô hướng ở đây gộp theo lớp và các vị trí không gian; công thức nhằm giải thích mục tiêu, không ấn định cách chuẩn hóa tổng/trung bình của implementation. $\mathcal L_G$ cung cấp thành phần đối kháng trong loss codec, không thay thế mọi thành phần rate và distortion.

U-Net kết hợp ngữ cảnh nhiều độ phân giải với thông tin không gian qua skip connections. Biến thể trong MS-ILLM dùng các residual block, LeakyReLU và đầu ra phù hợp độ phân giải bản đồ nhãn; bản thiết kế được mô tả không dùng normalization ở các lớp convolution của discriminator.

Điểm khác với discriminator nhị phân là yêu cầu dự đoán cấu trúc nhãn cục bộ được đưa vào loss. Nhãn VQ-VAE hướng tín hiệu học về các vùng đặc trưng, song vẫn không bảo đảm chi tiết ảnh tái tạo đúng tuyệt đối với ảnh gốc.

### A.5 Hai giai đoạn training

Giai đoạn đầu huấn luyện encoder, entropy model và decoder theo mục tiêu nén, chưa dùng discriminator. Giai đoạn sau fine-tune phần decoder với tín hiệu đối kháng. Các thông số dưới đây là **recipe của paper**, không phải log một run MS-ILLM mới của dự án.

| Tham số | Pretraining | Fine-tuning đối kháng |
|---|---|---|
| Số bước pretraining | 1.000.000 | Không suy từ số bước pretraining |
| Learning rate codec/generator | Peak $3\times10^{-4}$ | $1\times10^{-4}$ |
| Learning rate discriminator | Không dùng | $4\times10^{-4}$ |
| Optimizer | AdamW | AdamW |
| Warmup pretraining | 10.000 bước | Không suy cùng lịch từ giai đoạn đầu |
| Weight decay | $5\times10^{-5}$ | $5\times10^{-5}$ |

Khi reproduce hoặc fine-tune trong dự án, cần đối chiếu config, checkpoint và thành phần thực sự được cập nhật, thay vì chỉ sao chép vài hyperparameter trong bảng.

### A.6 Liên hệ với hướng nghiên cứu hiện tại

MS-ILLM cho thấy việc thiết kế tín hiệu đối kháng cục bộ có thể quan trọng bên cạnh kiến trúc entropy. Điều dự án cần kiểm tra tiếp là liệu cách học chất lượng này hoặc distillation từ teacher có chuyển được sang decoder nhanh ở bitrate thấp mà vẫn giữ chữ/mặt hay không. Phụ lục cung cấp nền tảng cho câu hỏi đó; bằng chứng hiện tại của dự án vẫn là các phép đo và giới hạn tại mục 5, không phải một phương pháp training mới đã được xác nhận.

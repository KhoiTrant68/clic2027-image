# CLIC 2027 Image GPU Track — Báo cáo tổng hợp

**Ngày biên tập tổng hợp:** 21/10/2026 — số liệu đến 20/10/2026\
**Repo:** [clic2027-image](https://github.com/KhoiTrant68/clic2027-image)\
**Mục tiêu:** xây dựng bộ nén ảnh có chất lượng cảm nhận cao ở các mức
bitrate thấp, đồng thời đáp ứng giới hạn thời gian giải mã, dung lượng
decoder và tính tái lập của môi trường CLIC 2027.

**Bản tổng hợp chính:** kiến trúc model, kết quả thực nghiệm, bảng metric và hình minh họa được đặt trong cùng tài liệu. Các báo cáo riêng giữ vai trò lưu vết nguồn.

**Đọc nhanh:** [metric và thang Q̂](#metrics-qhat) → [bảng + hình 30 ảnh](#comparison-30) → [pilot 6 ảnh](#pilot-six) → [AEIC, VRAM và tốc độ](#aeic-full30). Mục 1 là quyết định hiện tại; mục 9–10 là hướng nghiên cứu; Phụ lục A–C giữ nền tảng và cơ chế model.

**Đọc kiến trúc model:** [Phụ lục B](#model-mechanisms) bổ sung cơ chế encode/decode, thành phần dùng khi train, sơ đồ kiến trúc và đối chiếu với code đã chạy. Nguồn là paper gốc và implementation chính thức, đọc/đối chiếu ngày 15/10/2026.

## 1. Tóm tắt dự án

**Trạng thái theo dữ liệu đến 20/10:** PULSE-L đã tích hợp vào bake-off với **150 điểm (30 ảnh × q0–q4)**. Mix có PULSE-L đạt Q̂ **0,1841 / 0,7891 / 1,4978** dưới các ngân sách **0,075 / 0,15 / 0,3 bpp**. PULSE-L được chọn cho 22/30 ảnh ở ngân sách cao nhất. Đây là kết quả metric đại diện, chưa phải preference được người chấm xác nhận.

**Vòng AEIC đã hoàn tất:** hai checkpoint ME-ft2/SE-ft2 cho **60/60 stream thật và 60/60 ảnh khớp pixel** khi decode ở tiến trình mới trên cùng L40. Thêm AEIC nâng mix ở 0,075 bpp từ **0,1841 lên 0,1903**; hai ngân sách cao không đổi. Mức tăng **+0,00617** thấp hơn ngưỡng sàng lọc +0,02; do đó **chưa fine-tune hoặc đưa AEIC vào decoder chính**. Đỉnh VRAM riêng tiến trình khoảng **6,84 GiB (SE) / 7,32 GiB (ME)**. Phân tích đầy đủ cùng ảnh crop nằm tại [mục 5.11](#aeic-full30).

**Khả năng triển khai:** gói PULSE-L riêng đã decode 30/30 ảnh khớp pixel trong **12,235 giây trên L40**, gồm giải nén gói, khởi động, load và ghi PNG. ZIP 74,7 MB; đỉnh tổng VRAM thiết bị trong vòng tích hợp 5.920 MiB. Đây chưa phải benchmark toàn mix hoặc trên L4. [Bằng chứng tích hợp](CLIC2027_pulse_integration_2026-10-15.md).

**Ưu tiên hiện tại:** giữ mix có PULSE-L làm phương án chính; kiểm toàn gói decoder, chấm mù và holdout trước khi mở rộng huấn luyện. DiT-IC vẫn có tín hiệu tốt ở bitrate thấp; MS-ILLM và PULSE-L bổ sung ở rate cao. Các vòng sàng lọc trước tích hợp được giữ tại mục 5.8–5.10 và ghi rõ phạm vi lịch sử.

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
| **0.3 bpp** | Mix có PULSE-L, giữ MS-ILLM và các mode bổ sung | PULSE-L được chọn cho 22/30 ảnh; mix đạt Q̂ 1.4978 | Kết quả nội bộ; cần holdout và benchmark toàn gói |
| **AEIC-ME/SE ft2** | Giữ kết quả nghiên cứu, chưa fine-tune | Tăng Q̂ mix +0,00617 tại 0,075 bpp, không đạt ngưỡng +0,02; chi phí decode cao | Chỉ áp dụng recipe/checkpoint đã thử |
| **Ảnh chữ, giao diện, màn hình** | VTM-SCC | Codec hướng screen content giữ cấu trúc và chữ tốt hơn trong thử nghiệm trước đó | Có tín hiệu tốt; bộ phát hiện ảnh chưa triển khai |

**Vấn đề chính:** Benchmark trên L4 thật, tính tất định giữa các máy L4(mình có thể thuê runpod để chạy evaluation nhưng cần đảm bảo kết quả chạy trên các máy L4 là giống nhau), đóng gói decoder trong giới hạn 4 GB và tích hợp các nhánh codec thành một pipeline decode hoàn chỉnh.

## 2. Mục tiêu, giả thuyết và tiêu chí thành công

### 2.1 Mục tiêu

-   Tối ưu chất lượng cảm nhận theo ngân sách bitrate của toàn bộ tập ảnh.
-   Cho phép chọn codec/operating point khác nhau theo ảnh và theo  bitrate.
-   Đo riêng chất lượng, tốc độ, bộ nhớ, kích thước artifact và độ tin cậy của decoder.

### 2.2 Giả thuyết nghiên cứu

-   **H1 --- Preference thay cho PSNR đơn lẻ:** một mô hình $\hat{Q}$ kết hợp nhiều metric sẽ dự đoán lựa chọn của người chấm tốt hơn PSNR riêng. **Giải thích**: $\hat{Q}$ là một mô hình dự đoán chất lượng; điểm số do mô hình tạo ra mới là metric dùng để đánh giá và so sánh codec.
-   **H2 --- Codec chuyên biệt theo loại ảnh:** codec diffusion/latent có thể có lợi trên ảnh tự nhiên ở bitrate rất thấp, trong khi VTM-SCC có thể phù hợp hơn cho ảnh chữ/màn hình.
-   **H3 --- Trộn theo ảnh tốt sẽ hơn một codec duy nhất:** phân bổ ngân sách byte cho codec/operating point phù hợp từng ảnh có thể tăng tổng preference dưới cùng ngân sách byte.
-   **H4 --- Decoder chưng cất là phương án có thể suy nghĩ tuy nhiên sẽ có đánh đổi:** có thể giảm thời gian giải mã với tổn thất chất lượng cảm nhận có giới hạn.
-   **H5 --- Tính tất định là điều kiện bắt buộc:** chất lượng cao không có ý nghĩa nếu entropy decode tạo latent/ảnh khác nhau giữa môi trường encode và decode.

### 2.3 Tiêu chí thành công

Luật hiện tại yêu cầu đủ ba rate, decoder không quá **4 GB**, chạy với **2 CPU, RAM 12 GB và GPU L4 24 GB**. Chất lượng được đánh giá bằng người; nhóm **25% bài có thời gian giải mã lâu nhất** không đủ điều kiện nhận giải. Decoder test phải khớp hash đã đăng ký ở validation. Có thể dùng decoder khác nhau giữa các rate, nhưng mỗi decoder phải được đăng ký trước. [Nguồn: CLIC 2027 tasks](https://clic2027.compression.cc/tasks/).

Mục tiêu kỹ thuật nội bộ là hướng tới **≤20 giây/30 ảnh trên L4**, đồng thời có báo cáo số liệu riêng cho decode và tổng time xử lý. Mốc 20 giây chưa được xác nhận là mốc đáng tin cậy, cũng không bảo đảm tránh nhóm 25% chậm nhất nhưng là con số an toàn (theo số liệu của challenge đợt trước 2025). 

Lưu ý quan trọng: Kết quả nghiên cứu chỉ được xem là sẵn sàng nộp khi qua kiểm tra bitstream, byte-accounting và chạy trọn gói trong môi trường L4.

Tài nguyên nghiên cứu hiện có là **2 GPU L40** trên server. Giới hạn được cho phép là **30 GB trên mỗi GPU**. Giới hạn này dành cho nghiên cứu trên server; không thay đổi giới hạn phần cứng bài nộp L4.

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
- Có thể dùng một số năm để huấn luyện và năm còn lại để kiểm tra khả năng dự đoán trên dữ liệu chưa thấy. Hiện tại đang dùng 2021+2022 train còn 2024 test

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
| MS-SSIM  | Đo độ tương đồng về cấu trúc ở nhiều scale.                               |
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

Ở đây $x_i$ là ảnh gốc, $\hat{x}_i$ là ảnh tái tạo, $\phi_k$ là metric thứ $k$, $T_k$ là phép biến đổi metric và $w_k$ là trọng số đã fit. Bias A/B dùng khi dự đoán preference, không cộng vào utility từng ảnh của allocator.

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

Đây là tập validation CLIC 2027 gồm 30 ảnh, trùng tập test CLIC 2025, có ảnh tự nhiên, game và screen content. [Nguồn dữ liệu chính thức](https://clic2027.compression.cc/tasks/).

**Đầu ra:** Một tập validation cố định, được sử dụng nhất quán cho các codec so sánh.

#### Bước E — Encode bằng các codec ứng viên

**Mục tiêu: Nén cùng một tập ảnh bằng các phương pháp khác nhau để tạo ra các bitstream có thể so sánh.**

Các ứng viên trong dự án bao gồm:

- VTM / VTM-SCC: Codec đóng vai trò baseline truyền thống.
- MS-ILLM 
- MBT2018 
- CoD-Lite
- DC-AE + latent codec: Mã hóa ảnh thành latent rồi lượng tử hóa và entropy-code latent.
- DiT-IC
Khi encode, mỗi codec chuyển ảnh thành bitstream. Bitstream có thể chứa latent đã lượng tử hóa, thông tin phụ, header và dữ liệu entropy-coded.

**Điều cần kiểm soát:**
1. Dùng cùng ảnh gốc.
2. Chọn cấu hình tương ứng với mục tiêu bitrate.
3. Tính bitrate từ số byte thực tế của bitstream, không chỉ dựa trên tham số rate của model.
4. Ghi lại phiên bản code, checkpoint, cấu hình và thời gian encode.
5. Inference công bằng trên các codec.

**Đầu ra:** Bitstream của mỗi codec, cùng metadata về cấu hình, số byte và thời gian encode.

#### Bước F — Decode và đo metric + thời gian

**Mục tiêu: Đánh giá chất lượng ảnh khôi phục và chi phí tính toán thực tế.**
FTuwj
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

Để so sánh công bằng, cần thống nhất phần cứng, số ảnh, batch size, warm-up và cách đo thời gian. Các phép đo hiện tại cố định cách đo trên L40; kết quả L4 phải được đo riêng.

**Đầu ra:** Ảnh tái tạo, metric chất lượng, thời gian decode và thống kê tài nguyên cho mỗi codec.

#### Bước G — Chia ngân sách byte toàn tập

Đây là bước quan trọng nhất để tránh so sánh codec không công bằng.

Mục tiêu: Phân bổ tổng số byte cho các ảnh sao cho tổng bitrate của cả tập nằm trong ngân sách quy định.

**Tại sao phải phân bổ ngân sách?**

Giả sử bạn có 10.000.000 byte (khoảng 10 MB) để lưu trữ một bộ tập hợp 100 bức ảnh. 
Bạn có hai lựa chọn:
- Chia đều: Mỗi bức ảnh nhận đúng 100.000 byte. Nhưng cách này kém hiệu quả
- Chia thông minh: Ảnh đơn giản nhận 30.000 byte, ảnh phức tạp nhận 250.000 byte.

**Tại sao "Chia đều Bitrate" lại kém hiệu quả?**

Hiệu suất biểu diễn hình ảnh tuân theo quy luật hiệu suất giảm dần (Diminishing Returns): 
- Ảnh dễ (bầu trời xanh, tường phẳng, phông nền mờ) ---> Chỉ cần lượng byte rất nhỏ để đạt chất lượng tiệm cận hoàn hảo. Nếu rót thêm byte vào đây, điểm chất lượng $\hat{Q}$ gần như không tăng thêm đáng kể. 
- Trong khi ảnh khó (văn bản, chi tiết tóc, kết cấu phức tạp, khung cảnh nhiều vật thể), nếu thiếu byte, chất lượng vỡ nát nghiêm trọng làm giảm điểm $\hat{Q}$ rất sâu. Nhưng nếu cấp thêm byte, điểm chất lượng lại gia tăng cực kỳ rõ rệt.

Một codec tối ưu toàn cục sẽ rút bớt byte từ các ảnh dễ để viện trợ cho các ảnh khó, giúp tổng điểm trung bình toàn tập nâng lên rõ rệt so với việc cào bằng.

**Cụ thể thì bước này làm gì**
Bước này giải quyết lựa chọn thứ hai để tối đa hóa chất lượng cảm nhận trung bình ($\hat{Q}$) trên toàn bộ tập dữ liệu, đảm bảo tổng dung lượng không vượt quá hằng số ngân sách $B_{\text{budget}}$. Một cách diễn đạt bài toán phân bổ bitrate là:

Bitrate trung bình toàn tập tính theo bits per pixel ($\text{bpp}_{\text{avg}}$):

$$\text{bpp}_{\text{avg}} = \frac{8 \cdot B_{\text{total}}}{\sum_{i=1}^{N} H_i W_i}$$

Để khống chế bitrate không vượt quá mức target $r$, tổng số byte tối đa cho phép toàn tập là:

$$B_{\text{budget}} = \left\lfloor \frac{r}{8} \sum_{i=1}^{N} H_i W_i \right\rfloor$$

**Bài toán Multiple-Choice Knapsack (MCKP)**

* **Vật phẩm:** $N$ bức ảnh.
* **Thực đơn Lựa chọn:** Với mỗi ảnh $i$, ta có danh sách các codec nén $j_i$ (ví dụ: $q = VTM, COD-Lite,\dots$).
* **Chi phí:** Lựa chọn $j_i$ tốn $B_{i, j_i}$ byte thực tế.
* **Giá trị thu về:** Lựa chọn $j_i$ đem lại điểm chất lượng dự đoán $\hat{Q}_{i, j_i}$.

Mục tiêu là tìm bộ chỉ số $(j_1, j_2, \dots, j_N)$ giải quyết bài toán:

$$\max \sum_{i=1}^{N} \hat{Q}_{i, j_i} \quad \text{với điều kiện} \quad \sum_{i=1}^{N} B_{i, j_i} \le B_{\text{budget}}$$

**Đầu ra:** Các bitstream hoặc cấu hình bitrate đã được chọn sao cho tổng số byte nằm trong ngân sách tương ứng.

Lưu ý quan trọng: Không phải lúc nào codec cũng có thể được cấp chính xác số byte mong muốn. Phân bổ bitrate cần dùng các điểm rate–distortion đã đo, nội suy hoặc thử nghiệm thêm, rồi kiểm tra lại bằng kích thước bitstream thực tế. (Giải thích cụ thể ở phần phục lục C)

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

Cần benchmark trên phần cứng phù hợp với môi trường cuộc thi (hiện đang sử dụng phần cứng thay thế)

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

<a id="metrics-qhat"></a>

### 4.1 Metrics và Q̂ — cách đọc thang đo

**Vai trò:** Q̂ là mô hình preference để sàng lọc và phân bổ byte, không phải codec. Phần này thống nhất cách đọc tất cả bảng và hình trong báo cáo.

#### Các metric và quy ước gộp toàn tập

| Metric | Đơn vị / thang đo | Chiều tốt | Ý nghĩa và giới hạn |
|---|---|---|---|
| Bitrate — bpp | Bit trên một pixel; `8 × tổng byte / tổng pixel` | Thấp hơn khi giữ chất lượng tương đương | Tính byte stream thật. Không chỉ tính entropy ước lượng. Cần đọc cùng chất lượng. |
| PSNR | dB, không phải phần trăm | ↑ | Đo sai số RGB; nhạy với lệch màu/vị trí, có thể ưu ái ảnh mịn. Bảng dùng **pooled PSNR** từ MSE gộp có trọng số pixel. |
| MS-SSIM | Thường 0–1; 1 là giống hoàn toàn theo metric | ↑ | Độ giống cấu trúc ở nhiều tỉ lệ. Không bảo đảm chữ và chi tiết ngữ nghĩa đúng. |
| LPIPS-Alex | Khoảng cách đặc trưng AlexNet; 0 là khớp | ↓ | Dùng bản Alex trong cột LPIPS của báo cáo cũ. Không phải thang phần trăm, không mặc định bị chặn ở 1. |
| LPIPS-VGG | Khoảng cách đặc trưng VGG; 0 là khớp | ↓ | Khác backbone với LPIPS-Alex; không trộn hai giá trị vào một cột. |
| DISTS | Sai khác cấu trúc/texture; gần 0 tốt hơn | ↓ | Hữu ích cho ảnh cảm nhận; vẫn có thể bỏ sót lỗi chữ hoặc texture bịa. |
| CLIP-IQA | Điểm chất lượng ảnh không cần ảnh gốc, thường 0–1 | ↑ | Ảnh trông tự nhiên có thể được điểm cao dù không trung thực với ảnh gốc. |
| **Q̂ v0** | **Tiện ích thực, không có thang 0–100 hay mốc “đạt” cố định** | ↑ | Kết hợp sáu metric để dự đoán preference theo cặp. Giá trị âm hợp lệ; không đồng nghĩa model hỏng. |
| Decode time | Giây/ảnh hoặc giây/toàn bộ tập | ↓ | Phải phân biệt API với toàn gói gồm khởi động/load/I/O, và giữ phần cứng/tải giống nhau. |
| VRAM | MiB/GiB; ghi rõ theo PID hay toàn GPU | ↓ khi chất lượng/tốc độ tương đương | Số toàn GPU có thể bao gồm job khác. Không so thẳng với số riêng tiến trình. |
| Pixel parity | Số ảnh giải mã khớp pixel / tổng số | Đủ toàn bộ | Kiểm tính tái lập, không phải điểm chất lượng. Khớp trên L40 chưa chứng minh khớp trên L4. |
| OCR CER | Tỷ lệ lỗi ký tự, có thể >1 khi nhiều ký tự chèn | ↓ | Có trong probe chữ của repo; chưa đo đồng nhất cho mọi codec. |
| FID/KID, WD3/WD5 | Các khoảng cách phân phối/đặc trưng | ↓ ở dạng khoảng cách gốc | Có công cụ trong repo nhưng chưa phủ tất cả phương pháp ở bảng chính; **không nằm trong Q̂ v0**. |

Quy ước tính ảnh của repo: PSNR từ RGB; MS-SSIM trên ảnh đầy đủ; LPIPS và DISTS theo tile 512×512, gộp theo diện tích tile, bỏ tile có cạnh nhỏ hơn 64. Bpp gộp theo pixel; MS-SSIM, LPIPS, DISTS, CLIP-IQA và Q̂ của tập là trung bình đều trên các ảnh. Nguồn: [metrics.py](../src/clic27/eval/metrics.py), [Features và transform](../experiments/qhat/qhat.py), [bake-off summary](../experiments/bakeoff/bakeoff.py).

#### Phân biệt Q̂ với quality index

**Q̂ khác `q0/q1/q2/q3/q4`.** Các tên `q…` là chỉ số cấu hình/checkpoint chất lượng của từng codec; không có thang chung giữa các model. Q̂ là điểm được tính từ ảnh tái tạo và các metric bên dưới.

#### Công thức thực tế

Với ảnh gốc \(O\) và ảnh tái tạo \(X\), repo tính Q̂ v0 **riêng cho từng ảnh**:

```text
Q̂(O, X) =
  -0.034319 × PSNR
  +1.346040 × [-log10(max(1 - MS-SSIM, 0.000001))]
  -2.096177 × LPIPS-Alex
  -2.379560 × LPIPS-VGG
  -13.931139 × DISTS
  +1.720847 × CLIP-IQA
```

Các hệ số trên được làm tròn để trình bày; script dùng đủ precision trong [qhat_v0.json](../results/2026-10-15_new_codecs/qhat_v0.json). Trọng số `weights` đã được đổi khỏi chuẩn hóa khi xuất model; **không chia thêm cho `scale`**. Hệ số PSNR âm là kết quả fit đồng thời các đặc trưng tương quan; không diễn giải thành quy luật “PSNR càng thấp càng tốt”, hoặc lý do cố tình làm hỏng ảnh.

**Không thế trực tiếp các số trung bình trong bảng vào công thức để tái tạo Q̂ trung bình.** Phép biến đổi MS-SSIM là phi tuyến, còn PSNR trong bảng là pooled PSNR, khác trung bình PSNR từng ảnh. Script tính Q̂ từng ảnh trước rồi mới lấy trung bình.

#### Chênh lệch Q̂ có nghĩa gì?

Mô hình preference lưu trong repo:

```text
P(A được chọn hơn B | cùng ảnh gốc) = sigmoid(0.095620 + Q̂(A) - Q̂(B))
sigmoid(z) = 1 / (1 + exp(-z))
```

Bias 0,095620 phản ánh thiên lệch vị trí A/B trong phép fit, **không cộng vào Q̂ riêng của codec**. Bảng dưới minh họa logistic khi bỏ bias vị trí để dễ đọc:

| ΔQ̂ = Q̂(A) − Q̂(B) trên cùng ảnh | Xác suất mô hình, giả sử bias = 0 |
|---:|---:|
| 0 | 50,0% |
| +0,25 | 56,2% |
| +0,5 | 62,2% |
| +1 | 73,1% |
| +2 | 88,1% |

Đây là **diễn giải của mô hình**, không phải xác suất đã được kiểm chứng trên CLIC 2027. Tăng 1 Q̂ làm odds dự đoán nhân khoảng \(e ≈ 2,72\) trong mô hình. Q̂ = 1 không có nghĩa chất lượng “100%”; Q̂ = 0 cũng không phải ảnh ngẫu nhiên hay mốc đạt chuẩn.

**Không lấy sigmoid của ΔQ̂ trung bình toàn tập để tuyên bố tỷ lệ thắng của codec.** Mỗi ảnh có chênh lệch khác nhau; ngay trung bình xác suất theo từng ảnh cũng chỉ là dự đoán, chưa phải phiếu người. Ví dụ AEIC làm mix tăng +0,00617 trung bình nhờ đổi cách chia bit, trong khi cả ba ảnh chuyển sang AEIC đều giảm Q̂ riêng.

**Hình 7 — Diễn giải chênh lệch Q̂ của cùng một cặp ảnh.** Đường liền bỏ bias vị trí; đường đứt dùng bias trong file fit. Trục phần trăm là xác suất của mô hình preference, không phải chuyển Q̂ thành điểm chất lượng 0–100.

![Đường sigmoid minh họa quan hệ giữa chênh lệch Qhat và xác suất mô hình chọn A](figures/metrics_2026-10-20/07_qhat_scale.png)

#### Mức tin cậy hiện có

| Kiểm tra Q̂ | Q̂ v0 | Q̂ v1 | Cách đọc |
|---|---:|---:|---|
| Số câu hỏi có features dùng để fit | 8.000 | 8.000 | Dữ liệu preference CLIC cũ; không phải 8.000 ảnh độc lập |
| Accuracy 5-fold đã lưu | 78,8% | 79,0% | Fold theo câu hỏi; chưa bảo đảm độc lập theo ảnh |
| Train 2021/2022, test 2024 | 81,2% | 83,3% | Accuracy cross-year đã lưu; không phải accuracy trên tập 2027 |
| Đặc trưng | PSNR, MS-SSIM, LPIPS-A/V, DISTS, CLIP-IQA | MS-SSIM, WD3 | **Hai thang điểm khác nhau; không so trị số v0 với v1** |
| Vai trò trong bảng này | **Dùng thống nhất** | Không dùng để xếp hạng | Giữ cùng objective với allocator hiện tại |

Cross-year score đánh giá model fit trên các năm cũ; file weights triển khai được fit lại trên đủ dữ liệu. Repo còn có kiểm tra một người chấm DiT-IC với MS-ILLM: 62/90 crop nghiêng về DiT-IC, dấu ΔQ̂ v0 phù hợp đa số phiếu ở 26/30 ảnh. Bằng chứng này chỉ thuộc cặp model/cấu hình cũ, không xác nhận AEIC hay PULSE, và không thay thế nghiên cứu nhiều người. Không gọi mức đồng ý giữa hai người là một trần toán học tuyệt đối.

Nguồn: [fit v0](../results/2026-10-06_qhat_v0/fit.json), [fit v1](../results/2026-10-09_qhat_v1/fit.json), [phiếu so DiT-IC/MS-ILLM](../results/2026-10-11_rating_ditic_msillm/summary.md).

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

**Paper gốc:** Junqi Shi và cộng sự, *DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression*, CVPR 2026. [Bản proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_DiT-IC_Aligned_Diffusion_Transformer_for_Efficient_Image_Compression_CVPR_2026_paper.html). Xem [B.3](#model-ditic) về ba cơ chế alignment và sự khác biệt với student nội bộ.

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
| **S1: DC-AE + S1 + residual VTM** | Kiểm tra hướng latent codec đa rate kết hợp residual | Artifact local chỉ có metric cho 28/30 ảnh; chưa xếp hạng toàn tập. Các probe Kodak được ghi riêng ở mục 5.12 |
| **SD-Turbo refiner** | Kiểm tra khả năng hậu xử lý/refinement | Không vượt MS-ILLM trong thử nghiệm nội bộ; dừng |
| **COOL-CHIC** | Codec tọa độ/implicit representation, tham khảo đối thủ | Theo ghi chú nội bộ, kém MS-ILLM khoảng 0.4--0.6 Q̂ và encode chậm; tạm dừng |
| **CRDR** | Ứng viên learned codec khác | Kém MS-ILLM theo Q̂ v0 và khoảng 5 s/ảnh giải mã trong phép đo ghi nhận; loại khỏi lựa chọn chính |

Các kết luận loại/giữ chỉ áp dụng cho cấu hình, checkpoint, bitrate và
tập ảnh đã thử; không chứng minh các phương pháp đó kém trong mọi điều
kiện.

### 4.7 PULSE-L — ứng viên decoder nhẹ mới

**Phân biệt phiên bản:** PULSE base khoảng 5.2 kMAC/pixel, PULSE-L perceptual khoảng 20 kMAC/pixel theo [bảng model chính thức](https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md). Không gán độ phức tạp hay tốc độ CPU của base cho checkpoint L mà repo đang dùng. Xem [B.4](#model-pulse) về pixel renderer, CDF số nguyên và Meta Prior.

PULSE kết hợp xử lý đặc trưng ở độ phân giải thấp, pixel renderer và entropy path có thành phần số nguyên. Bản phát hành có checkpoint perceptual base/L và nhiều mức q. Dự án đã thử cả hai, sau đó chọn PULSE-L để đánh giá đầy đủ vì vùng rate và khả năng bổ sung cho mix phù hợp hơn. [Implementation chính thức](https://github.com/microsoft/GenCodec/tree/main/PULSE).

Vai trò hiện tại của PULSE-L là bổ sung cho các codec đã có, chưa thay chúng ở mọi rate. Các con số chất lượng và tốc độ do dự án đo được trình bày tại mục 5.8–5.10; không dùng benchmark phần cứng của tác giả thay cho benchmark L40/L4 của dự án. Entropy số nguyên là đặc tính đáng quan tâm, nhưng parity của toàn pipeline vẫn cần kiểm thực tế.

### 4.8 Bản đồ cơ chế của các model đã chạy

| Model / nhóm | Khác biệt cần hiểu | Phần đọc chi tiết |
|---|---|---|
| MS-ILLM | Discriminator tạo tín hiệu học cục bộ; không nằm trong decoder triển khai | Phụ lục A và [B.2](#model-msillm) |
| DiT-IC; student nội bộ | Diffusion một bước trong latent; student của repo chỉ thay DC-AE decoder | [B.3](#model-ditic) |
| PULSE / PULSE-L | Decoder bất đối xứng; tách điều khiển entropy khỏi dựng ảnh | [B.4](#model-pulse) |
| CoD-Lite | Mã hóa condition bằng VQ; dựng ảnh trực tiếp bằng diffusion convolution | [B.5](#model-codlite) |
| DC-VIC / AFP-GIC | Dự đoán prior từ thông tin đã truyền; không có ảnh gốc ở decoder | [B.6–B.7](#model-dcvic) |
| CRDR | Điều khiển bitrate và realism riêng | [B.8](#model-crdr) |
| Cool-Chic | Tối ưu riêng cho mỗi ảnh; phải truyền cả tham số mạng | [B.9](#model-coolchic) |
| MBT2018-mean | Mean-scale hyperprior; phân biệt với bản autoregressive đầy đủ | [B.10](#model-mbt) |
| VTM / VTM-SCC | Dự đoán khối và residual; công cụ riêng cho screen content | [B.11](#model-vtm) |
| S1 / SD-Turbo refiner | Hai pipeline thí nghiệm của repo; không coi là paper codec mới | [B.12](#model-internal) |

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
| **Trên 2024t: PSNR 0.705; Q̂ v0 cross-year 0.812** | So trên cùng miền test, Q̂ có tín hiệu tốt hơn PSNR đơn lẻ | Cần kiểm lại split theo ảnh và leakage |
| **Trên 8.000 cặp: Q̂ v0 5-fold CV 0.788; PSNR 0.625** | Kết quả gộp khác với phép kiểm tra cross-year | Không trộn hai protocol thành một con số |
| **Q̂ v1: cross-year 0.833** | Accuracy cao hơn chưa đủ để chọn khi ranking ở miền rate thấp không phù hợp | Tiếp tục dùng v0 cho bảng codec hiện tại; v1 chỉ làm phân tích độ nhạy. Nguyên nhân không chọn v1: Trong bài toán nén ảnh, việc duy trì thứ tự xếp hạng (ranking) ở miền bitrate thấp (low rate regime) cực kỳ quan trọng (nơi ảnh dễ bị nhòe/artefact nặng). $\hat{Q}\text{ v1}$ tuy nâng accuracy tổng thể nhưng lại làm sai lệch thứ tự xếp hạng ở dải bitrate thấp này.  |

Nguồn: [Q̂ v0](D:/AwesomeCV/ratflow-codec/results/2026-10-06_qhat_v0/summary.md) và [Q̂ v1](D:/AwesomeCV/ratflow-codec/results/2026-10-09_qhat_v1/summary.md). Các cặp cùng ảnh có thể phụ thuộc nhau; accuracy cross-year không loại bỏ mọi dạng domain shift. Tỉ lệ đồng ý giữa hai người là chỉ báo độ nhiễu, không phải một trần toán học tuyệt đối của mọi mô hình dự đoán.

**Ý nghĩa khoa học:** thí nghiệm kiểm tra khả năng dùng dữ liệu
preference lịch sử để xây dựng surrogate metric cho quá trình nghiên
cứu.\
**Việc tiếp theo:** tái lập split, kiểm tra rò rỉ dữ liệu, báo cáo
confidence interval và calibration; đánh giá trên nhiều người chấm.

<a id="comparison-30"></a>

### 5.2 Thí nghiệm B — bảng và đồ thị tổng hợp trên 30 ảnh

**Cập nhật dữ liệu đến 20/10/2026.** Các bảng dưới đây thay bảng tổng quan 13/10, giữ rõ nguồn snapshot của từng phương pháp và bổ sung AEIC. Không chạy lại model trong bước tổng hợp.

- Bảng 30 ảnh dùng cùng danh sách ảnh và tổng 86.245.376 pixel. Các ngân sách byte là 808.550 / 1.617.100 / 3.234.201.
- Mỗi phương pháp chọn một điểm đã đo cho từng ảnh để tối đa tổng Q̂ dưới **ngân sách toàn tập**. Đây không phải kết quả mọi ảnh đều đúng cùng bpp.
- VTM-SCC, MS-ILLM, CoD-Lite, DiT-IC, student và PULSE-L lấy từ snapshot tích hợp 15/10. VTM 4:2:0 và MBT lấy từ snapshot 13/10; CRDR từ 10/10; SD-Turbo refiner từ 08/10. Đó là các evaluation đã hoàn tất, **không phải so tiến độ huấn luyện ngang số bước**.
- Mix có PULSE gồm sáu họ trong snapshot tích hợp; mix + AEIC thêm hai biến thể. **Không phải mix hợp nhất toàn bộ các phương pháp lịch sử** ở bảng này.
- Pixel parity và header của tất cả codec lịch sử chưa được kiểm cùng một quy trình triển khai. Đặc biệt MS-ILLM có [audit riêng](CLIC2027_msillm_stream_audit_2026-10-15.md); bảng giữ output/metric snapshot đã đo, chưa đại diện một gói decoder cuối cùng.
- S1 + residual có point cho 30 ảnh nhưng chỉ **364 metric rows phủ 28/30 ảnh** trong artifact local. Không điền một Q̂ “toàn 30 ảnh” từ tập thiếu.
- Nhãn `Không vừa` chỉ nói về các checkpoint/điểm đã thử, không nói kiến trúc đó không thể đạt rate thấp hơn sau huấn luyện.

#### Q̂ theo ba ngân sách

Mỗi ô là **Q̂ / bpp thực** dưới ngân sách cột. `Không vừa` nghĩa là các điểm đã có không đủ thấp để nén toàn tập. Bitrate thấp hơn ngân sách không được nội suy thành một điểm mới.

| Phương pháp | ≤0,075 bpp | ≤0,15 bpp | ≤0,3 bpp |
|---|---:|---:|---:|
| VTM 4:2:0 | -2.5149 / 0.0749 | -1.1633 / 0.1498 | -0.1516 / 0.2998 |
| VTM-SCC 4:4:4 | -2.4038 / 0.0749 | -1.1020 / 0.1499 | -0.0044 / 0.2990 |
| DC-AE + S1 + residual VTM | Thiếu metric (28/30 ảnh) | Thiếu metric (28/30 ảnh) | Thiếu metric (28/30 ảnh) |
| MBT2018-mean | Không vừa | -2.0078 / 0.1499 | -0.4737 / 0.2995 |
| MS-ILLM + SD-Turbo refiner | -1.6089 / 0.0749 | -1.3315 / 0.1463 | -1.3315 / 0.1463 |
| MS-ILLM | -0.4513 / 0.0749 | 0.5447 / 0.1498 | 1.3831 / 0.2997 |
| CoD-Lite | -0.3825 / 0.0749 | 0.2846 / 0.1497 | 0.9055 / 0.2996 |
| CRDR | Không vừa | 0.2953 / 0.1498 | 1.2716 / 0.2997 |
| DiT-IC | 0.1644 / 0.0749 | 0.2788 / 0.0958 | 0.2788 / 0.0958 |
| DiT-IC student | -0.0002 / 0.0748 | 0.1151 / 0.0984 | 0.1151 / 0.0984 |
| PULSE-L perceptual | Không vừa | Không vừa | 1.4234 / 0.2998 |
| Mix có PULSE-L | 0.1841 / 0.0749 | 0.7891 / 0.1499 | 1.4978 / 0.2997 |
| Mix có PULSE-L + AEIC | 0.1903 / 0.0749 | 0.7891 / 0.1499 | 1.4978 / 0.2997 |

**Hình 1— Q̂ theo ngân sách.** Đọc theo cột để so dưới cùng trần byte, đọc theo hàng để xem phương pháp thay đổi thế nào khi tăng ngân sách. Mỗi ô ghi cả Q̂ và bitrate thực. Ví dụ DiT-IC chỉ đạt khoảng 0,096 bpp trong các điểm đã thử; hai cột ngân sách cao không phải hai checkpoint mới.

![Qhat theo ba ngân sách trên cùng 30 ảnh, mỗi ô kèm bitrate thực](figures/metrics_2026-10-20/01_qhat_overview.png)

**Hình 2— Hai tiêu chí chất lượng theo bitrate thực.** Q̂ và PSNR có thể dẫn đến thứ tự khác nhau. AEIC nằm ở khoảng 0,03 bpp, vì vậy không xem nó như một điểm đã đo tại 0,075 bpp. Đường nối chỉ giúp theo dõi các điểm có sẵn, không chứng minh chất lượng giữa các điểm.

![Qhat và PSNR theo bitrate thực của các phương pháp tiêu biểu](figures/metrics_2026-10-20/02_rate_quality.png)

#### Đầy đủ metrics — ngân sách 0.075 bpp

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VTM 4:2:0 | 0.07488 | 26.01 | 0.9213 | 0.2961 | 0.3756 | 0.1782 | 0.4652 | -2.5149 |
| VTM-SCC 4:4:4 | 0.07489 | 26.15 | 0.9256 | 0.2867 | 0.3698 | 0.1738 | 0.4606 | -2.4038 |
| MS-ILLM + SD-Turbo refiner | 0.07488 | 22.22 | 0.8422 | 0.1972 | 0.3010 | 0.1492 | 0.7098 | -1.6089 |
| MS-ILLM | 0.07494 | 25.15 | 0.9143 | 0.0943 | 0.2664 | 0.0939 | 0.5940 | -0.4513 |
| CoD-Lite | 0.07491 | 22.02 | 0.8579 | 0.1091 | 0.2301 | 0.0934 | 0.6798 | -0.3825 |
| DiT-IC | 0.07494 | 22.27 | 0.8795 | 0.0831 | 0.2028 | 0.0687 | 0.7034 | 0.1644 |
| DiT-IC student | 0.07478 | 22.24 | 0.8768 | 0.0944 | 0.2070 | 0.0762 | 0.6932 | -0.0002 |
| Mix có PULSE-L | 0.07492 | 22.38 | 0.8758 | 0.0847 | 0.2078 | 0.0711 | 0.7261 | 0.1841 |
| Mix có PULSE-L + AEIC | 0.07494 | 22.10 | 0.8735 | 0.0883 | 0.2086 | 0.0699 | 0.7240 | 0.1903 |

![Sáu metric tại ngân sách 0.075 bpp](figures/metrics_2026-10-20/03_metrics_0075.png)

Ở ngân sách thấp, Q̂ cao không đồng nghĩa PSNR, MS-SSIM và mọi khoảng cách cảm nhận đều tốt hơn. Hai hàng mix thể hiện rõ đánh đổi khi thêm AEIC.

#### Đầy đủ metrics — ngân sách 0.15 bpp

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VTM 4:2:0 | 0.14977 | 28.29 | 0.9549 | 0.2141 | 0.3153 | 0.1351 | 0.6030 | -1.1633 |
| VTM-SCC 4:4:4 | 0.14986 | 28.10 | 0.9529 | 0.2109 | 0.3099 | 0.1334 | 0.5899 | -1.1020 |
| MBT2018-mean | 0.14988 | 27.42 | 0.9500 | 0.2611 | 0.3501 | 0.1649 | 0.4798 | -2.0078 |
| MS-ILLM + SD-Turbo refiner | 0.14630 | 22.13 | 0.8475 | 0.1779 | 0.2828 | 0.1402 | 0.7366 | -1.3315 |
| MS-ILLM | 0.14979 | 27.19 | 0.9490 | 0.0580 | 0.2143 | 0.0608 | 0.6550 | 0.5447 |
| CoD-Lite | 0.14971 | 23.87 | 0.9206 | 0.0798 | 0.1857 | 0.0697 | 0.6710 | 0.2846 |
| CRDR | 0.14983 | 27.66 | 0.9470 | 0.0637 | 0.2354 | 0.0737 | 0.6855 | 0.2953 |
| DiT-IC | 0.09577 | 22.57 | 0.8879 | 0.0784 | 0.1902 | 0.0643 | 0.6946 | 0.2788 |
| DiT-IC student | 0.09843 | 22.62 | 0.8875 | 0.0911 | 0.1943 | 0.0722 | 0.6888 | 0.1151 |
| Mix có PULSE-L | 0.14991 | 24.59 | 0.9257 | 0.0653 | 0.1774 | 0.0536 | 0.7170 | 0.7891 |
| Mix có PULSE-L + AEIC | 0.14991 | 24.59 | 0.9257 | 0.0653 | 0.1774 | 0.0536 | 0.7170 | 0.7891 |

![Sáu metric tại ngân sách 0.15 bpp](figures/metrics_2026-10-20/04_metrics_015.png)

DiT-IC và student dùng dưới 0,1 bpp; đây chưa phải các checkpoint khớp rate 0,15 bpp. Hai mix trùng kết quả vì AEIC không được chọn.

#### Đầy đủ metrics — ngân sách 0.3 bpp

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VTM 4:2:0 | 0.29980 | 31.62 | 0.9733 | 0.1496 | 0.2594 | 0.1046 | 0.6872 | -0.1516 |
| VTM-SCC 4:4:4 | 0.29900 | 31.68 | 0.9757 | 0.1449 | 0.2535 | 0.0994 | 0.6711 | -0.0044 |
| MBT2018-mean | 0.29945 | 29.80 | 0.9726 | 0.1719 | 0.2790 | 0.1160 | 0.6187 | -0.4737 |
| MS-ILLM + SD-Turbo refiner | 0.14630 | 22.13 | 0.8475 | 0.1779 | 0.2828 | 0.1402 | 0.7366 | -1.3315 |
| MS-ILLM | 0.29967 | 29.41 | 0.9731 | 0.0335 | 0.1621 | 0.0362 | 0.6909 | 1.3831 |
| CoD-Lite | 0.29959 | 25.77 | 0.9518 | 0.0583 | 0.1453 | 0.0536 | 0.6804 | 0.9055 |
| CRDR | 0.29968 | 29.99 | 0.9701 | 0.0352 | 0.1712 | 0.0436 | 0.7343 | 1.2716 |
| DiT-IC | 0.09577 | 22.57 | 0.8879 | 0.0784 | 0.1902 | 0.0643 | 0.6946 | 0.2788 |
| DiT-IC student | 0.09843 | 22.62 | 0.8875 | 0.0911 | 0.1943 | 0.0722 | 0.6888 | 0.1151 |
| PULSE-L perceptual | 0.29977 | 29.43 | 0.9702 | 0.0353 | 0.1216 | 0.0345 | 0.6760 | 1.4234 |
| Mix có PULSE-L | 0.29973 | 29.34 | 0.9707 | 0.0364 | 0.1281 | 0.0338 | 0.6932 | 1.4978 |
| Mix có PULSE-L + AEIC | 0.29973 | 29.34 | 0.9707 | 0.0364 | 0.1281 | 0.0338 | 0.6932 | 1.4978 |

![Sáu metric tại ngân sách 0.3 bpp](figures/metrics_2026-10-20/05_metrics_03.png)

VTM-SCC nổi bật về PSNR/MS-SSIM; PULSE-L có Q̂ cao nhất trong các codec đơn đã thử ở gần budget. Chênh lệch nhỏ chưa được kiểm định ý nghĩa thống kê.

Mỗi hình giữ cùng thứ tự hàng trong sáu panel, sắp theo Q̂ giảm dần. Số sau tên là bpp thực; các trục dùng giá trị metric gốc. ↑ tốt hơn khi tăng, ↓ tốt hơn khi giảm. Khoảng cách giữa các panel không cùng đơn vị nên không so trực tiếp. Các đường RD chỉ nối các điểm đã đo để hướng dẫn mắt, không ngoại suy checkpoint mới.

Nguồn: [CSV 30 ảnh](../results/2026-10-20_metrics_comparison/full30_budget_metrics.csv), [script tính lại](../results/2026-10-20_metrics_comparison/build_comparison.py), [hash dữ liệu](../results/2026-10-20_metrics_comparison/sources.sha256.json).

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

<a id="pilot-six"></a>

### 5.8 Thí nghiệm H — sàng lọc model mới trên sáu ảnh

**Câu hỏi:** model mới công khai có tăng chất lượng hoặc tốc độ hữu ích so với các codec hiện có không?

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


**Quyết định:** chưa mở rộng AFP/DC-VIC vì chất lượng pilot chưa vượt baseline và decoder chậm; PULSE base có vùng rate chưa phù hợp. PULSE-L có lợi thế tốc độ và khả năng bổ sung cho mix, nên được chọn chạy đủ 30 ảnh. [Bảng pilot gốc](D:/AwesomeCV/ratflow-codec/results/2026-10-15_new_codecs/pilot_scores/summary.csv).



#### Đồ thị và đầy đủ metrics trên cùng pilot

![Q̂ và bitrate thực trên pilot 6 ảnh](figures/metrics_2026-10-20/06_pilot6.png)

Màu cùng thang với hình 30 ảnh, nhưng tập ảnh khác nhau. Chỉ so các hàng trong đúng pilot. `PULSE(-L)` trong tên mix chỉ tập ứng viên chứa PULSE base và PULSE-L, không nói mọi họ đều được chọn.

AFP-GIC/DC-VIC/PULSE-L dưới đây là pilot 6 ảnh. Baseline pilot chưa có PULSE-L; khác baseline tích hợp của bảng 30 ảnh. PULSE base đã chạy 24 điểm nhưng không vừa cả ba budget, nên không có hàng allocation hợp lệ.

##### Ngân sách 0.075 bpp

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AFP-GIC | 0.07470 | 23.20 | 0.8956 | 0.0901 | 0.2323 | 0.1190 | 0.5922 | -0.7304 |
| MS-ILLM | 0.07496 | 23.40 | 0.9020 | 0.0883 | 0.2119 | 0.1005 | 0.5833 | -0.3030 |
| Mix baseline pilot | 0.07388 | 20.10 | 0.8324 | 0.0757 | 0.1676 | 0.0756 | 0.7221 | 0.3806 |
| Mix pilot + AFP/DC-VIC/PULSE/PULSE-L | 0.07388 | 20.10 | 0.8324 | 0.0757 | 0.1676 | 0.0756 | 0.7221 | 0.3806 |

##### Ngân sách 0.15 bpp

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AFP-GIC | 0.14150 | 24.30 | 0.9191 | 0.0649 | 0.1841 | 0.0860 | 0.6910 | 0.3158 |
| DC-VIC PatchGAN | 0.14657 | 24.11 | 0.9183 | 0.0653 | 0.1693 | 0.0805 | 0.6945 | 0.4599 |
| MS-ILLM | 0.14941 | 25.49 | 0.9528 | 0.0501 | 0.1742 | 0.0627 | 0.6128 | 0.6423 |
| Mix baseline pilot | 0.14738 | 21.86 | 0.8825 | 0.0563 | 0.1446 | 0.0585 | 0.7162 | 1.0409 |
| Mix pilot + AFP/DC-VIC/PULSE/PULSE-L | 0.14938 | 21.86 | 0.8825 | 0.0564 | 0.1406 | 0.0581 | 0.7135 | 1.0560 |

##### Ngân sách 0.3 bpp

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AFP-GIC | 0.22538 | 26.65 | 0.9594 | 0.0393 | 0.1693 | 0.0660 | 0.7165 | 0.8712 |
| DC-VIC PatchGAN | 0.22920 | 26.69 | 0.9601 | 0.0378 | 0.1505 | 0.0577 | 0.6975 | 1.0237 |
| PULSE-L pilot | 0.29426 | 27.06 | 0.9668 | 0.0309 | 0.0987 | 0.0348 | 0.6743 | 1.6057 |
| MS-ILLM | 0.29972 | 28.06 | 0.9761 | 0.0255 | 0.1283 | 0.0339 | 0.7154 | 1.7323 |
| Mix baseline pilot | 0.29979 | 27.45 | 0.9628 | 0.0385 | 0.1051 | 0.0342 | 0.7653 | 1.8629 |
| Mix pilot + AFP/DC-VIC/PULSE/PULSE-L | 0.29975 | 28.44 | 0.9734 | 0.0421 | 0.1012 | 0.0310 | 0.7204 | 1.9194 |

Nguồn: [CSV pilot 6 ảnh](../results/2026-10-20_metrics_comparison/pilot6_budget_metrics.csv).

### 5.9 Thí nghiệm I — PULSE-L trên đủ 30 ảnh

**Vòng sàng lọc lịch sử 15/10, trước tích hợp q0–q4.** Các số 1,4912 và 20/30 dưới đây thuộc vòng bốn quality; kết quả sau tích hợp là 1,4978 và 22/30 tại 0,3 bpp, xem mục 1 và 5.2.

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

**Phép đo lịch sử của vòng sàng lọc 120 điểm.** Bảng dưới giữ thời gian API và kiểm parity 3 ảnh của vòng này. Phép kiểm toàn gói PULSE-L sau tích hợp đã đạt 30/30 ảnh khớp pixel và 12,235 giây tổng wall time trên L40, được trình bày cùng đồ thị tại mục 1. Không so trực tiếp hai protocol như một mức tăng tốc của model.

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

<a id="aeic-full30"></a>

### 5.11 Thí nghiệm K — AEIC full30: chất lượng, chi phí và quyết định

Ngày thực nghiệm: 20/10/2026, múi giờ Asia/Bangkok.

#### Metrics của hai checkpoint cố định

| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AEIC-SE ft2 | 0.03111 | 20.71 | 0.8287 | 0.1441 | 0.2807 | 0.1097 | 0.7081 | -0.9016 |
| AEIC-ME ft2 | 0.03260 | 21.48 | 0.8434 | 0.1352 | 0.2640 | 0.0988 | 0.7087 | -0.6635 |

Cả hai checkpoint dùng khoảng 0,03 bpp; không so chúng như các điểm đã đo tại 0,075 bpp.

#### Kết quả và quyết định

**Hoàn tất lúc 06:49:43: 60/60 điểm có bitstream thật, 60/60 ảnh khớp pixel khi giải mã ở tiến trình mới, scoring và phân bổ ngân sách hoàn tất. Không fine-tune AEIC ở vòng này**, vì mức cải thiện không đạt ngưỡng thực dụng đã định trước. Các tiến trình AEIC và scorer đã kết thúc.

| Ngân sách toàn tập | Q̂ mix có PULSE | Q̂ thêm AEIC | ΔQ̂ trung bình | Ảnh dùng AEIC |
|---|---:|---:|---:|---:|
| 0,075 bpp | 0,184142 | 0,190313 | **+0,006171** | 3/30, đều ME |
| 0,15 bpp | 0,789137 | 0,789137 | 0 | 0/30 |
| 0,3 bpp | 1,497811 | 1,497811 | 0 | 0/30 |

SE không được chọn ở cả ba ngân sách. Baseline gồm 780 điểm của sáu codec, bao gồm 150 điểm PULSE-L; thêm AEIC tạo tổng 840 điểm. Q̂ baseline tái lập khớp số đã lưu trong sai số 1e-6. Bitrate thực của mix mở rộng lần lượt là 0,0749355 / 0,1499080 / 0,2997301 bpp, đều không vượt ngân sách.

![Chất lượng và số ảnh được chọn theo codec](../results/2026-10-20_aeic_full30/full30_20261020/quality_and_selection.png)

Nguồn số liệu: [comparison.json](../results/2026-10-20_aeic_full30/full30_20261020/comparison.json), [allocation.csv](../results/2026-10-20_aeic_full30/full30_20261020/scores/allocation.csv), [baseline_preflight.json](../results/2026-10-20_aeic_full30/full30_20261020/baseline_preflight.json). Hình có [bản SVG](../results/2026-10-20_aeic_full30/full30_20261020/quality_and_selection.svg) và [script tái tạo](../experiments/new_codecs/summarize_aeic.py).

##### Lợi ích đến từ đâu?

Ở 0,075 bpp, có sáu ảnh đổi lựa chọn:

| Ảnh (prefix) | Trước → sau | Thay đổi byte | ΔQ̂ riêng ảnh |
|---|---|---:|---:|
| `2684452d` — tượng và bảng chữ | DiT-IC q4 → AEIC-ME ft2 | −25.497 | −0,84497 |
| `937476dd` — texture vải | DiT-IC q3 → AEIC-ME ft2 | −16.047 | −0,38640 |
| `d0208a1e` — thác nước | DiT-IC q2 → AEIC-ME ft2 | −3.846 | −0,10844 |
| `8426ed22` — chân dung | DiT-IC q2 → q3 | +13.920 | +0,46328 |
| `a365e654` — đường phố | DiT-IC q2 → q3 | +14.904 | +0,48898 |
| `eda6d427` — phố và xe đạp | DiT-IC q2 → q3 | +16.780 | +0,57266 |

Ba ảnh chuyển sang AEIC tiết kiệm **45.390 byte**, nhưng tổng Q̂ của chúng giảm 1,33980. Ba ảnh còn lại dùng thêm 45.604 byte và tăng tổng Q̂ 1,52492. Chênh lệch tổng là 0,18512; chia cho 30 ảnh được **+0,00617 Q̂ trung bình**. Cả tập chỉ dùng thêm 214 byte so với baseline, vẫn dưới budget.

Do đó, việc allocator chọn AEIC **không chứng minh AEIC cho ảnh đẹp hơn lựa chọn cũ**. Trong phép thử này, AEIC đóng vai trò lựa chọn ít bit để nhường ngân sách. Nguồn đầy đủ: [changed_images.csv](../results/2026-10-20_aeic_full30/full30_20261020/changed_images.csv).

##### Bộ nhớ và thời gian

| Model | Bpp gộp 30 ảnh | Q̂ đơn model | Decode độc lập TB/ảnh | Tổng API decode 30 ảnh | Peak VRAM riêng tiến trình |
|---|---:|---:|---:|---:|---:|
| AEIC-SE ft2 | 0,03111 | −0,90158 | 24,26 s | 727,78 s | 7.006 MiB = 6,84 GiB |
| AEIC-ME ft2 | 0,03260 | −0,66348 | 21,90 s | 656,90 s | 7.496 MiB = 7,32 GiB |

Scorer đạt peak 4.072 MiB. Tất cả thấp hơn ngưỡng dừng riêng tiến trình 10 GiB. ME và SE chạy trên hai GPU có tải dùng chung khác nhau, nên **không suy ra ME nhanh hơn SE về mặt kiến trúc** từ bảng này. Q̂ đơn model cũng không phải so sánh ở cùng bitrate với mix.

Riêng ba ảnh AEIC-ME được chọn tại 0,075 bpp đã cần tổng **65,20 giây API decode**, chưa gồm nạp model và I/O. Đây là chi phí đáng cân nhắc cho một mức tăng Q̂ nhỏ; chưa đo thời gian toàn mix trên L4. Bộ file weights nguồn cần nạp có tổng 4,792 GB cho ME và 4,633 GB cho SE (GB thập phân), gồm cả tensor sau đó bị cắt bỏ hoặc thay thế. **Đây chưa phải kích thước gói decoder tối ưu**, cũng chưa tính dependencies. Số đo: [resource_summary.json](../results/2026-10-20_aeic_full30/full30_20261020/resource_summary.json), [weight_inventory.json](../results/2026-10-20_aeic_full30/full30_20261020/weight_inventory.json).

##### Quan sát trực quan và bộ chấm A/B

Đã tạo **18 cặp crop × 3 ảnh gốc/A/B = 54 ảnh lossless**, phủ đủ sáu lựa chọn thay đổi: mỗi ảnh gồm một crop giữa và hai crop ngẫu nhiên có texture, seed cố định. Tên model không có trong trang chấm; khóa đáp án được giữ riêng. Phiếu người xem chưa được điền.

- [Mở bộ chấm HTML](../results/2026-10-20_aeic_full30/full30_20261020/blind_review/index.html): chọn A/B/ngang nhau/khó đánh giá rồi tải phiếu JSON. Mở file bằng trình duyệt trên máy; giữ nguyên thư mục `img` bên cạnh.
- [Ghi chú trực quan của AI trước khi mở khóa A/B](../results/2026-10-20_aeic_full30/full30_20261020/visual_audit/AI_NOTES_BLIND.md): đã xem toàn bộ 18 crop, không tính thành phiếu người thật.
- Ảnh trong bộ chấm đã được kiểm tra đủ file và đúng kích thước 512×512. Trình duyệt tích hợp chặn URL `file://`, nên thao tác chấm/lưu phiếu trong trình duyệt chưa được xác nhận bằng UI.

Sau khi đối chiếu khóa: các crop chữ và vải chuyển sang AEIC có những sai lệch dễ thấy; crop thác nước có chi tiết được giữ tốt hơn ở một số vị trí nhưng không nhất quán. Các ảnh được thêm bit cũng không thể hiện ưu thế rõ ở mọi crop. Đây là nhận xét định tính của AI, không phải kết quả nghiên cứu preference.

![Ví dụ chữ và vải, ảnh gốc cùng A/B ẩn tên](../results/2026-10-20_aeic_full30/full30_20261020/visual_audit/examples.png)

Hai hàng là p002 (chữ) và p006 (vải); ảnh gốc ở bên trái, A/B ở giữa/phải. Các crop được giữ nguyên pixel, không tăng nét hay chỉnh màu để minh họa.

##### Hành động sau vòng này

1. **Giữ mix có PULSE làm phương án chính**, lưu AEIC là candidate nghiên cứu, chưa tích hợp vào decoder nộp bài.
2. **Không chạy fine-tune dài hoặc pilot fine-tune ở vòng này**: +0,00617 thấp hơn ngưỡng +0,02, chưa có bằng chứng trực quan bù lại chi phí decoder. Kết luận này chỉ áp dụng hai checkpoint ft2 và recipe tile đã thử; không kết luận mọi cấu hình AEIC đều không hữu ích.
3. Đã chuẩn bị kiểm tra khả năng fine-tune để dùng lại nếu có giả thuyết mới đủ cụ thể. Muốn mở lại hướng này cần đánh giá thêm trên held-out, cải thiện có ý nghĩa và phép đo chi phí triển khai; không dùng việc huấn luyện dài để thay thế các bằng chứng còn thiếu.

Các bước đánh giá, phân tích, crop và báo cáo đã hoàn tất. Chấm bởi người là phần còn mở nếu muốn thu thập preference; không cần chờ phiếu để thực hiện quyết định không fine-tune theo ngưỡng đã định.

#### Câu hỏi thực nghiệm

AEIC-ME/SE có giúp hệ thống hiện tại phân bổ ngân sách tốt hơn ở 0,075 / 0,15 / 0,3 bpp hay không? Baseline là **mix đã tích hợp PULSE-L**, không phải mix cũ của pilot ba ảnh.

Đơn vị so sánh chính là phương án nén **cùng 30 ảnh dưới cùng ngân sách byte toàn tập**. Một ảnh có thể được nén rẻ hơn để nhường bit cho ảnh khác. Vì vậy, không so trực tiếp AEIC-ft2 ở khoảng 0,02–0,03 bpp với một codec ở 0,075 bpp rồi kết luận ai tốt hơn.

Protocol và ngưỡng quyết định được ghi trước khi scoring trong [PLAN.md](../results/2026-10-20_aeic_full30/PLAN.md). Ngưỡng sàng lọc thực dụng: tăng ít nhất 0,02 Q̂ trung bình và AEIC được chọn cho ít nhất 3/30 ảnh ở một ngân sách. Đây không phải kiểm định ý nghĩa thống kê hay ngưỡng thắng cuộc thi.

#### Cấu hình và tính hợp lệ

- Hai checkpoint chính thức ME-ft2 và SE-ft2; mã tác giả pin tại [`7222c2c`](https://github.com/LuizScarlet/AEIC/tree/7222c2c85e8db7ea0b829a139313ab7f4fdea88c).
- Giữ nguyên kích thước ảnh, dùng entropy coder thật. Số byte bao gồm header AEIC 4 byte và metadata kích thước trong stream tác giả.
- `q2` trong thư mục kết quả chỉ định danh checkpoint ft2; không cùng mức bitrate/chất lượng với q2 của PULSE hoặc codec khác.
- Tắt color correction cần thống kê từ ảnh gốc. Tiến trình giải mã độc lập không được truyền đường dẫn tập ảnh gốc; so SHA256 RGB với lần giải mã reference.
- FP32, seed 0, cuDNN deterministic, tắt TF32, tắt compile, bật xformers. Latent tile 64, overlap 16, VAE tile 64. Đây là cấu hình ít VRAM khác tile mặc định tác giả; không coi kết quả này là tái lập nguyên vẹn benchmark trong paper.
- SE chạy GPU 0, ME chạy GPU 1 trên hai L40 dùng chung. Allocator 8 GiB, dừng nếu bộ nhớ riêng tiến trình đạt 10 GiB hoặc thiết bị còn dưới 1,5 GiB trống. Guard là giám sát phần mềm, không phải phân vùng GPU cứng.
- Chỉ chấm điểm sau khi cả 60 stream vượt kiểm giải mã độc lập. Q̂ baseline phải khớp kết quả mix đã lưu trong sai số 1e-6.

#### Cơ chế liên quan đến kết quả

Đường giải mã trong [`AEIC_practical.py`](https://github.com/LuizScarlet/AEIC/blob/7222c2c85e8db7ea0b829a139313ab7f4fdea88c/src/AEIC_practical.py) khôi phục latent qua entropy decoder, dùng UNet một bước rồi cộng residual trong không gian latent trước khi đưa vào VAE decoder. [`base_model.py`](https://github.com/LuizScarlet/AEIC/blob/7222c2c85e8db7ea0b829a139313ab7f4fdea88c/src/base_model.py) bỏ time embedding và cross-attention của UNet, thay VAE decoder bằng bản thu hẹp, bỏ VAE encoder và gộp LoRA khi suy luận.

Các bước cắt giảm này vẫn để lại một decoder diffusion đáng kể. Tile nhỏ giúp giảm bộ nhớ nhưng có thể tăng thời gian và thay đổi output. Do đó cần đo đồng thời Q̂, ảnh trực quan, bitstream, thời gian và VRAM. Không dùng mức VRAM suy luận để dự đoán trực tiếp khả năng fine-tune: trainer còn có backward, optimizer, CLIP, DINOv3 discriminator và các mạng tính loss.

#### Phạm vi diễn giải

Q̂ v0 là metric đại diện của dự án, không phải phiếu người xem. Việc thêm candidate vào allocator và đánh giá trên chính 30 ảnh này là kết quả validation nội bộ; cần held-out độc lập trước khi tuyên bố khả năng tổng quát. Pixel parity trong hai tiến trình trên cùng môi trường L40 chưa chứng minh parity trên L4 hoặc một môi trường thư viện khác.

Thời gian API trên GPU dùng chung không tương đương thời gian toàn gói decoder trong môi trường thi. Mức peak lấy từ trace NVML theo PID, chu kỳ 0,25 giây; có thể bỏ lỡ đỉnh ngắn. Không dùng counter peak của PyTorch làm peak toàn run vì VAEHook tác giả reset counter trong quá trình giải mã.

Việc chấm A/B của người xem được để trống. Nhận xét trực quan của trợ lý, nếu có, được ghi riêng và không tính thành phiếu người thật.

#### Chuẩn bị fine-tune

Đã kiểm tra trainer, cấu hình và 800 ảnh DIV2K trên server. Chi tiết các điều kiện trước launch, lỗi lệch batch giữa shell/YAML, giới hạn dừng pilot và recipe một GPU nằm trong [TRAINING_READINESS.md](../results/2026-10-20_aeic_full30/TRAINING_READINESS.md). Huấn luyện chỉ được khởi động sau khi các ngưỡng chất lượng, trực quan và chi phí được đáp ứng.

#### Bằng chứng và chạy lại

Server: `guess:~/clic/aeic_20261016/full30_20261020`. Bitstream và ảnh đầy đủ được giữ tại đây. Bản local gồm log, metadata, metrics, allocation, parity, trace VRAM và crop: [thư mục kết quả](../results/2026-10-20_aeic_full30/README.md).

Hash code/baseline, manifest 30 ảnh và hash ảnh nguồn: [provenance.json](../results/2026-10-20_aeic_full30/full30_20261020/provenance.json). Bản code baseline trên server là snapshot không có `.git`; những file Python liên quan được hash nội dung. Phiên bản môi trường và hashes weights có trong [bằng chứng pilot](../results/2026-10-20_aeic_pilot/README.md).

Artifact server `full30_evidence.tgz` được tải về và kiểm SHA256: `b1951ca73279a0e8ef113d73fc12bb0c3781ffd81a1fbebec954c682d0bc0368`. Contact sheet và ghi chú AI được tạo thêm tại local, sau khi tải artifact, nên không nằm trong hash tar này.

### 5.12 Các probe khác đã chạy và dữ liệu chưa đủ

| Phương pháp / phép thử | Tập và điều kiện | bpp thực | PSNR | LPIPS-Alex ↓ | DISTS ↓ | Q̂ v0 | Phạm vi kết luận |
|---|---|---:|---:|---:|---:|---:|---|
| Cool-Chic 5.0.1, 2.000 iterations | Một ảnh `2684452d` | 0,1114 | 29,11 | 0,109 | 0,091 | −0,41 | Không xếp chung bảng 30 ảnh |
| Cool-Chic tag CLIC2025 | Một ảnh `2684452d`, perceptive config | 0,557 | 32,67 | — | — | +1,33 | Ngoài cả ba ngân sách mục tiêu |
| S1 r4, step 200.000 | Kodak, snapshot `results/s1` | 0,0710 | 21,70 | 0,1521 | 0,1058 | — | Không có đủ features để tính Q̂ v0 |
| S1 r7, cùng snapshot | Kodak | 0,1215 | 23,29 | 0,0980 | 0,0621 | — | Khác tập với bảng chính |
| DC-AE ceiling | Kodak, latent không bị nén bởi S1 | Không có | 23,66 | 0,0905 | 0,0583 | — | Kiểm backbone; **không phải codec có bitrate thật** |
| DC-AE + residual giả định base 0,03 bpp | Probe B6 natural/screen | Xem nguồn | Theo nhóm | Theo nhóm | Theo nhóm | — | Chi phí base giả định; không trộn với true-bitstream codec |

`—` là không có số tương ứng trong bằng chứng này, không phải 0. Nguồn: [Cool-Chic 5.0.1](../results/2026-10-09_coolchic_probe/summary.md), [Cool-Chic CLIC2025](../results/2026-10-09_coolchic2025_probe/summary.md), [S1/Kodak](../results/s1/s1_eval/summary.md), [B6](../results/clic_b/b6_results/summary.md). Nhiều checkpoint/ablation distillation đã được chạy; bảng chính dùng student được đưa vào bake-off, không trộn các snapshot huấn luyện trung gian thành codec mới.

PICO, ProGIC, EGIC và PORT trong báo cáo khảo sát là các hướng đã đọc/tìm implementation; chưa có kết quả inference hợp lệ trong bộ artifact này, nên không gán metric cho chúng.

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
7.  **PULSE-L bổ sung được cho mix:** sau tích hợp, Q̂ đạt 0,7891/1,4978 ở 0,15/0,3 trên 30 ảnh; decoder nhẹ là lý do tiếp tục đầu tư, nhưng chưa có bằng chứng người chấm cho mix mới.
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

### 6.3 Đọc đồng thời các metric để quyết định

Ở ngân sách 0,075, DiT-IC có Q̂ tốt nhất trong các codec đơn được phân bổ gần hết ngân sách; VTM có PSNR/MS-SSIM cao hơn nhưng LPIPS/DISTS kém hơn. Đây là ví dụ rõ cho việc các metric đánh giá những thuộc tính khác nhau.

Ở 0,15, MS-ILLM đạt Q̂ 0,5447; mix có PULSE đạt 0,7891 nhờ chia byte giữa nhiều họ model. DiT-IC chỉ dùng khoảng 0,0958 bpp nên điểm ở cột này không mô tả khả năng của một checkpoint DiT-IC 0,15 bpp chưa được thử.

Ở 0,3, PULSE-L đạt Q̂ 1,4234, MS-ILLM 1,3831, CRDR 1,2716; mix đạt 1,4978. Khoảng cách này thuộc metric đại diện trên validation, chưa phải kết luận thắng theo người chấm.

Thêm AEIC tại 0,075 làm Q̂ tăng từ 0,1841 lên 0,1903 nhưng PSNR giảm 22,38 → 22,10 dB, MS-SSIM giảm 0,8758 → 0,8735, LPIPS-Alex tăng 0,0847 → 0,0883; DISTS giảm nhẹ 0,0711 → 0,0699. **Q̂ tăng không có nghĩa mọi metric đều tốt lên.** Quyết định chưa fine-tune AEIC vẫn giữ nguyên theo [mục 5.11](#aeic-full30).

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
| [AEIC](https://github.com/LuizScarlet/AEIC) | Ultra-low bitrate với shallow encoder | Đã thử ME/SE ft2 trên 30 ảnh; 60/60 parity, mức tăng mix +0,00617 ở 0,075; chưa fine-tune (mục 5.11) |
| [PORT — CLIC 2025](https://www.jdl.link/doc/2011/20251221_232_32_18.PORT%20Perception-Oriented%20Image%20Compression.pdf) | Decoder nhanh, loss semantic/patch và phân bổ theo ROI | Tìm được paper của PKUSZ-AliMerlin; chưa tìm checkpoint công khai để thử |
| [Cool-Chic — tag CLIC 2025](https://github.com/Orange-OpenSource/Cool-Chic/releases) | Tối ưu từng ảnh, decoder C nhẹ | Đã có phép thử trong repo ngày 09/10; chưa chạy lại trong vòng mới |

Cool-Chic CLIC2025 trên ảnh `2684452d` mất khoảng 1 giờ 39 phút encode, đạt Q̂ v0 1.33 tại 0.557 bpp; MS-ILLM cùng ảnh đạt 1.46 tại 0.267 bpp. Đây là bằng chứng trên một ảnh và cấu hình cụ thể, đủ để chưa ưu tiên quét hàng loạt, không đủ để loại mọi cấu hình Cool-Chic. Nếu thử tiếp, giới hạn ở screen/texture và lambda phù hợp 0.075–0.3. [Kết quả Cool-Chic cũ](D:/AwesomeCV/ratflow-codec/results/2026-10-09_coolchic2025_probe/summary.md).

Tìm được paper không đồng nghĩa có thể reproduce. Các mục “chưa tìm checkpoint” phản ánh phạm vi tìm kiếm hiện tại, không khẳng định checkpoint không tồn tại. Việc đánh giá trên bộ dữ liệu CLIC cũng không tự chứng minh một model là bài dự thi CLIC.

### 9.2 Hướng A — lựa chọn codec có tính chi phí triển khai

**Giả thuyết:** lựa chọn theo từng ảnh có thể đạt chất lượng tốt hơn tại cùng ngân sách byte và thời gian nếu tính cả chi phí nạp từng họ decoder. Bằng chứng khởi đầu là PULSE-L được chọn trên 10/30 và 22/30 ảnh ở hai budget cao sau tích hợp.

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
| Đã hoàn tất đến 20/10 | Tích hợp PULSE-L và sàng lọc AEIC-ME/SE full30 | Bảng metric và ảnh trong mục 5.2/5.11; chưa fine-tune AEIC vì không đạt ngưỡng |
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

Dự án đã có bằng chứng ủng hộ hệ thống nhiều codec: DiT-IC có tín hiệu tốt ở rate thấp; MS-ILLM và PULSE-L bổ sung ở rate cao hơn; VTM-SCC giữ vai trò trên một số nội dung screen. Sau tích hợp, mix có PULSE-L đạt Q̂ 0,1841/0,7891/1,4978 tại ba ngân sách. AEIC tăng thêm +0,00617 ở rate thấp nhưng chưa đạt ngưỡng để fine-tune; kết quả chưa được xác nhận bằng người. Ưu tiên tiếp theo là chấm mù, holdout và đo gói decoder toàn mix, đồng thời sàng lọc có giới hạn các giả thuyết A/B trên server.

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

Các bảng tổng hợp cập nhật lấy từ artifact đã lưu đến ngày 20/10/2026; ngày biên tập tài liệu là 21/10/2026. Các thí nghiệm A/B/C trong kế hoạch chưa được thực hiện; số liệu của chúng không nằm trong các bảng kết quả hiện tại.

#### Nguồn bảng và hình tổng hợp

- [CSV 30 ảnh, đủ metrics và nguồn mỗi hàng](../results/2026-10-20_metrics_comparison/full30_budget_metrics.csv).
- [CSV AEIC checkpoint cố định](../results/2026-10-20_metrics_comparison/aeic_fixed_metrics.csv).
- [CSV pilot 6 ảnh](../results/2026-10-20_metrics_comparison/pilot6_budget_metrics.csv).
- [Bitrate tối thiểu của các điểm đã thử](../results/2026-10-20_metrics_comparison/rate_floors.json): MBT khoảng 0,1314 bpp, CRDR 0,1184 bpp và PULSE-L 0,1921 bpp trên toàn bộ 30 ảnh; dùng để kiểm tra các ô `Không vừa`.
- [Script tổng hợp không dùng GPU](../results/2026-10-20_metrics_comparison/build_comparison.py), [hash nguồn](../results/2026-10-20_metrics_comparison/sources.sha256.json).

Hash nguồn văn bản chuẩn hóa CRLF thành LF (giữ BOM nếu có) để tái kiểm chứng trên Windows/Linux. Script ghép allocation với metric từng ảnh, tính lại Q̂ từ weights v0 và pooled PSNR, đối chiếu summary đã lưu, kiểm đủ số ảnh và không vượt ngân sách. Với AEIC không lưu MSE, MSE được suy ngược từ PSNR từng ảnh trước khi gộp. Không suy ra metric bị thiếu bằng cách nội suy giữa các model, và không chạy lại inference trong bước này.

Bảy đồ thị metric được nhúng trực tiếp tại mục 4.1, 5.2 và 5.8. [Script vẽ](figures/metrics_2026-10-20/make_figures.py), [manifest nguồn hình](figures/metrics_2026-10-20/manifest.json), [ghi chú kiểm tra](figures/metrics_2026-10-20/figure_notes.md). Kết quả AEIC, hình phân bổ và ảnh crop được nhúng tại mục 5.11.

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

<a id="model-mechanisms"></a>

## Phụ lục B. Cơ chế và kiến trúc các model — đọc paper, đối chiếu implementation

Phạm vi gồm các codec đã chạy trong repo và SANA/DC-AE/SD-Turbo làm nền tảng. Các ứng viên mới chỉ liệt kê ở mục 9.1 chưa được xem như implementation đã kiểm chứng. Sơ đồ dưới đây do báo cáo diễn giải ở mức luồng dữ liệu; không thay thế sơ đồ lớp đầy đủ của tác giả. Nhãn **Theo paper**, **Trong repo** và **Hàm ý** giúp phân biệt nguồn kiến thức với bằng chứng nội bộ.

### B.1 Cách đọc một neural codec

Đặt $x$ là ảnh, $y=g_a(x)$ là latent chính, $\hat y=Q(y)$ là latent đã lượng tử hóa. Với hyperprior, $z=h_a(y)$ cung cấp thông tin phụ để dự đoán phân phối của $\hat y$. Rate ước lượng thường có dạng:

$$
R_{\mathrm{est}}=-\log_2 p(\hat z)-\log_2 p(\hat y\mid\hat z),
\qquad \hat x=g_s(\hat y).
$$

Entropy model quyết định cách biến symbol thành bit; synthesis transform quyết định cách biến latent đã giải mã thành ảnh. Đó là hai vai trò khác nhau. Mean-scale hyperprior mô hình hóa cả trung bình và độ phân tán; context model có thể bổ sung phụ thuộc giữa các symbol. [Minnen, Ballé và Toderici, NeurIPS 2018, mục 2–3](https://arxiv.org/pdf/1809.02736).

```mermaid
flowchart LR
    X["Ảnh gốc x"] --> GA["Analysis transform"] --> Y["Latent y"] --> Q["Lượng tử hóa"]
    Y --> HA["Hyper encoder"] --> Z["Hyperlatent lượng tử"]
    Z --> BZ["Entropy encode: stream phụ"] --> DZ["Giải hyperlatent"] --> HS["Hyper decoder"]
    HS --> P["Tham số xác suất"]
    Q --> AE["Entropy encode"] --> BY["Stream chính"] --> AD["Entropy decode"]
    P --> AE
    P --> AD
    AD --> GS["Synthesis transform"] --> O["Ảnh tái tạo"]
```

**Đối chiếu triển khai:** trong report, bpp thực luôn phải tính từ file truyền: payload + kích thước/quality + mode + framing cần thiết. Discriminator, teacher hoặc nhãn sinh từ ảnh gốc chỉ dùng khi train không tự động trở thành dữ liệu decoder được phép sử dụng. Với output số thực, cần tách ba kiểm tra: đọc được entropy stream, khớp latent/symbol và khớp pixel PNG.

<a id="model-msillm"></a>

### B.2 MS-ILLM — học thống kê cục bộ, giữ đường nén mean-scale hyperprior

**Theo paper.** Muckley và cộng sự, *Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models*, ICML 2023. VQ-VAE biến ảnh thật thành bản đồ nhãn codebook. Discriminator U-Net dự đoán nhãn cục bộ hoặc fake; codec học tái tạo ảnh có cấu trúc nhãn phù hợp. Đây là cách đưa tín hiệu thống kê cục bộ vào training. Đường nén vẫn là encoder → latent/hyperlatent → entropy coding → decoder. VQ-VAE tạo nhãn và discriminator không cần chạy khi giải mã ảnh. [Paper, mục 4.1–4.5](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf).

```mermaid
flowchart LR
    X["Ảnh x"] --> C["Codec mean-scale hyperprior"] --> R["Ảnh tái tạo"]
    X -. "chỉ train" .-> V["VQ-VAE: nhãn cục bộ"]
    X -.-> D["Discriminator U-Net"]
    R -.-> D
    V -. "nhãn giám sát" .-> D
    D -. "gradient khi train" .-> C
```

**Trong repo.** `_msillm_load` nạp checkpoint public q1–q5. Encoder/decoder ảnh chạy GPU, phần entropy sử dụng cách phân bố tensor do NeuralCompression quy định. Kiểm mới lưu cả hai payload cùng kích thước ảnh/padding: header 28 byte, không còn chỉ cộng giả định 4 byte. Bật deterministic đạt parity 5/5 ảnh giữa tiến trình; đây là kiểm triển khai nội bộ, không phải kết quả trong paper. [Code bake-off](../experiments/bakeoff/bakeoff.py), [kiểm stream](CLIC2027_msillm_stream_audit_2026-10-15.md).

**Hàm ý:** nếu distill MS-ILLM sang decoder khác, tín hiệu teacher có thể lấy ở image/feature space; không được giả định latent hoặc entropy format tương thích. Phụ lục A trình bày kỹ hơn loss của ILLM.

<a id="model-ditic"></a>

### B.3 DiT-IC — ba alignment để đưa DiT về nhiệm vụ nén ảnh

**Theo paper.** DiT-IC làm việc ở latent giảm kích thước không gian 32 lần. Ba cơ chế chính: variance-guided flow điều chỉnh mức tái tạo theo uncertainty; self-distillation dùng biểu diễn của encoder đóng băng làm đích cho latent tái tạo; latent-conditioned guidance thay điều kiện văn bản bằng embedding dự đoán từ latent. Training dùng alignment và rate–distortion, sau đó thêm đối kháng; LoRA thích nghi DiT và decoder. Khi inference không cần prompt văn bản. [Shi và cộng sự, CVPR 2026, mục 3.1–3.4](https://arxiv.org/html/2603.13162v1#S3).

**Hai nền tảng.** DC-AE nén không gian sâu bằng residual autoencoding: shortcut space-to-channel giúp mạng học phần dư thay vì tự học toàn bộ phép giảm resolution; training nhiều giai đoạn xử lý khó khăn ở ảnh lớn. SANA kết hợp autoencoder nén sâu với linear-attention DiT và FFN có locality. DC-AE một mình chưa tạo file nén entropy, còn SANA gốc là hệ sinh ảnh. [DC-AE, mục 3](https://arxiv.org/html/2410.10733v1); [SANA, mục 3](https://arxiv.org/html/2410.10629v1).

**Trong repo — luồng decoder thực tế:**

```mermaid
flowchart LR
    B["Entropy strings + shape"] --> E["Latent codec decode"]
    E --> M["mean và log-variance"]
    E --> R["residual"]
    E --> P["prompt latent"] --> PR["Prompter"]
    M --> D["SANA DiT, một lần forward"]
    PR --> D
    D --> S["Scheduler step với scale"]
    M --> S
    S --> A["Cộng residual; hiệu chỉnh scale"]
    R --> A
    A --> V["DC-AE decoder"] --> O["RGB"]
```

Sơ đồ bám hàm `decompress` trong `_DITIC_SCRIPT`: entropy path giữ precision riêng; phần DiT/prompter/DC-AE dùng dtype cấu hình, cùng crop latent, bucket và RMSNorm tối ưu. Đây là bản triển khai đã điều chỉnh cho benchmark, không phải chép nguyên mọi chi tiết paper. [Code repo](../experiments/bakeoff/bakeoff.py), [implementation tác giả](https://github.com/Eric-qi/DiT-IC).

**Phân biệt hai loại distillation:** self-distillation trong paper phục vụ tái tạo một bước của DiT-IC. Thí nghiệm `ditics` của chúng ta thay các tầng DC-AE độ phân giải cao bằng student hẹp hơn và học output teacher. Nó không phải tái lập nguyên recipe self-distillation của paper, cũng không thay toàn bộ DiT bằng mạng nhỏ.

<a id="model-pulse"></a>

### B.4 PULSE / PULSE-L — ít tính toán ở decoder, kiểm soát riêng đường entropy

**Theo paper.** Encoder dùng patch embedding stride 16 và depthwise-convolution blocks. Hyperlatent ở độ phân giải 1/64. Latent chính chia checkerboard không gian–kênh, giải qua hai lượt thay vì quét tuần tự từng pixel. Decoder xử lý đặc trưng thấp resolution, rồi AdaLN điều chế MLP renderer trên lưới 1/4; PixelShuffle dựng patch RGB. Linear CDF predictor số nguyên chọn bảng xác suất; Meta Prior truyền chỉ số bank để thích nghi phân phối hyperlatent. Training perceptual đi từ LPIPS sang GAN và loss ROI chữ/mặt. [Jia và cộng sự, PULSE, arXiv 2026, mục 2–5](https://arxiv.org/html/2609.18602v1).

```mermaid
flowchart LR
    B["Stream + QP"] --> Z["Meta Prior: giải hyperlatent"]
    B --> E["Giải latent: hai lượt"]
    Z --> C["Integer Linear CDF"] --> E
    Z --> H["Hyper features / mean"] --> E
    E --> L["Low-resolution body"] --> A["AdaLN modulation"]
    E --> U["Content lên lưới 1/4"]
    A --> P["MLP renderer + PixelShuffle"]
    U --> P --> O["RGB"]
```

**Theo release.** Base và L cùng họ kiến trúc nhưng khác dung lượng; L perceptual khoảng 20 kMAC/pixel, base khoảng 5.2. Một model hỗ trợ q0–q7. Bundle gồm cấu hình, trọng số float và entropy-control integer; bitstream không chứa toàn bộ mạng. Tác giả chỉ bảo đảm tính tất định cho entropy-control trên nền tảng hỗ trợ, không hứa mọi pixel float giống nhau. [README chính thức, Models / Compress and decompress](https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md).

**Trong repo.** Bundle L đã chạy 150 điểm q0–q4 và kiểm decoder độc lập. Header `PL` bốn byte của repo bọc payload tác giả; file thật được tính byte. Benchmark 12.235 giây bao gồm khởi động, giải nén gói, nạp model và ghi PNG trên L40. Không dùng số CPU/base trong paper để thay cho phép đo này.

**Hàm ý:** đây là nền tảng phù hợp để nghiên cứu decoder nhanh. “Thêm ROI chữ/mặt” tự nó chưa phải đóng góp mới vì paper đã làm; hướng distillation theo độ tin cậy cần ablation riêng và kiểm phạm vi prior art.

<a id="model-codlite"></a>

### B.5 CoD-Lite — condition nén giàu thông tin thay cho backbone sinh ảnh lớn

**Theo paper.** Encoder–decoder convolution tạo condition qua vector quantization; chỉ số codebook được mã hóa độ dài cố định. Condition dẫn dắt diffusion convolution một bước trong pixel space. Backbone thay attention bằng depthwise convolution và channel attention, đầu MLP dựng pixel. Pretraining hướng compression cung cấp điều kiện giàu thông tin; fine-tune cố định timestep/noise, dùng reconstruction/perceptual loss, DMD từ teacher CoD và GAN. Teacher/discriminator chỉ phục vụ training. [Jia và cộng sự, ICML 2026](https://proceedings.mlr.press/v306/jia26i.html); [paper, mục 5](https://arxiv.org/html/2604.12525v1#S5).

```mermaid
flowchart LR
    X["Ảnh"] --> E["Encoder"] --> V["VQ indices"] --> B["Fixed-length bits"]
    B --> C["Condition decoder"] --> D["Convolution diffusion: một bước"] --> O["RGB"]
```

**Trong repo.** `points_codlite` gọi `compress`, dựng condition bằng `decompress`, rồi `inference` với tensor khởi đầu bằng không. Config/checkpoint gắn với từng operating point; không xem một checkpoint bất kỳ là model variable-rate liên tục. Ảnh được pad theo bội số 64 rồi crop về kích thước gốc. [Code bake-off](../experiments/bakeoff/bakeoff.py), [release CoD-Lite](https://github.com/microsoft/GenCodec/tree/main/CoD_Lite).

**Hàm ý:** paper cho thấy một hướng giảm tải bằng condition tốt và distillation, nhưng lợi ích FID trên tập của tác giả không chứng minh lợi ích Q̂/preference trên validation của ta. Việc giữ CoD-Lite trong bảng là để đối chiếu phương án này, không vì nhãn “real-time” tự động đáp ứng bài thi.

<a id="model-dcvic"></a>

### B.6 DC-VIC — dự đoán token và sửa feature của VQGAN

**Theo paper.** Compression encoder dựa trên residual/attention nhận ảnh, đặc trưng VQGAN và hai điều kiện. Hyperprior cùng channel autoregressive model mã hóa latent. Ở decoder, token estimator gồm CNN/Swin dự đoán index codebook; nhánh SFT tạo đặc trưng điều chế các tầng của VQGAN decoder đóng băng. $\beta_{rate}$ điều khiển tổng rate, $\beta_{vq}$ điều khiển độ chính xác dự đoán token, từ đó cân bằng thông tin cho token và sửa feature. Training qua lấy mẫu điều kiện, chọn cặp điều kiện rồi fine-tune. [Iwai, Miyazaki và Omachi, IEEE Access 2024, mục III, hình 2–6; DOI](https://doi.org/10.1109/ACCESS.2024.3522238), [bản toàn văn tác giả](https://www.researchgate.net/publication/387411891_Dual-Conditioned_Training_to_Exploit_Pre-trained_Codebook-based_Generative_Model_in_Image_Compression).

```mermaid
flowchart LR
    B["Bits + điều kiện"] --> E["Entropy decode"] --> Y["Latent"]
    Y --> T["Token estimator"] --> C["Tra codebook"] --> V["VQGAN decoder cố định"]
    Y --> S["SFT extractor"] --> V
    V --> O["RGB"]
```

**Trong repo.** Đã chạy checkpoint **PatchGAN**, không phải OASIS. API quality 0–4 chọn cấu hình đã được release; không hiểu q là số bit/pixel cố định. Các tensor entropy cũ được chuyển tên cho stack CompressAI hiện tại rồi strict-load. Pilot 18 điểm hợp lệ chỉ hỗ trợ nhận xét về cấu hình này. [Release chính thức](https://github.com/iwa-shi/DC_VIC), [ghi nhận thử nghiệm](CLIC2027_new_models_2026-10-15.md).

**Hàm ý:** prior giúp tái tạo texture, nhưng dự đoán sai token vẫn có thể tạo chi tiết sai; SFT có vai trò sửa theo thông tin đã truyền. Decoder cần cả bộ dự đoán và VQGAN, nên phải đo tổng load/runtime thay vì chỉ đếm số bước sinh ảnh.

<a id="model-afp"></a>

### B.7 AFP-GIC — chuyển prior hợp nhất từ AdaCode sang codec

**Theo paper.** AdaCode đóng băng hợp nhất nhiều codebook bằng trọng số theo nội dung. Prior Feature Adapter đưa đặc trưng đó vào encoder. Decoder không nhận prior gốc: Prior Estimator fully-convolutional dự đoán prior tương ứng từ latent đã giải mã và biến điều khiển; SFT điều chế decoder AdaCode đóng băng. Loss alignment giúp prior dự đoán gần prior từ ảnh gốc. Vì vậy thông tin vẫn phải đi qua bottleneck entropy; không có đường truyền prior “miễn phí”. [Pei, Liu và Ling, *Adaptive Fused Prior Transfer for Controllable Generative Image Compression*, bản arXiv v2 năm 2026, mục III, bảng I–II](https://arxiv.org/html/2605.16817v2#S3).

```mermaid
flowchart LR
    X["Ảnh gốc"] --> A["AdaCode prior extractor cố định"] --> P["Prior adapter"] --> E["Compression encoder"]
    X --> E --> B["Entropy bitstream"] --> Y["Latent đã giải mã"]
    Y --> PE["Prior Estimator"] --> D["AdaCode decoder + SFT"]
    Y --> S["SFT extractor"] --> D --> O["RGB"]
```

**Trong repo.** Chạy `afp_gic_release.pth.tar` qua runtime public; model có nhiều q nhưng pilot chỉ đo q0/1/4. Smoke load thiếu entropy parameters bị loại; bảng kết quả dùng strict-load. Prior ở đây là prior dựng ảnh, cần phân biệt với entropy prior tính xác suất symbol. [Release](https://github.com/yifeipet/AFP_GIC), [probe hiện có](../experiments/new_codecs/probe.py).

**Hàm ý:** ý tưởng đáng học là supervision bất đối xứng: encoder nhìn ảnh gốc, decoder học dự đoán từ bit đã nhận. Pilot của chúng ta chưa cho thấy lợi ích đủ để mở train dài; kết quả đó không phủ định kiến trúc trên mọi tập dữ liệu.

<a id="model-crdr"></a>

### B.8 CRDR — một model, hai nút điều khiển

**Theo paper.** CRDR dùng encoder kiểu ELIC, entropy model theo kênh và generator có beta-conditioning. Interpolation Channel Attention điều chỉnh theo q; $\beta$ được Fourier-encode rồi đưa qua MLP vào residual blocks để đổi realism. HRRGAN dùng reconstruction ở rate cao hơn làm đối chiếu cho generator, giảm áp lực so trực tiếp mọi reconstruction với ảnh gốc. Discriminator chỉ dùng khi train. [Iwai, Miyazaki và Omachi, WACV 2024, mục 3, hình 2–4](https://arxiv.org/html/2405.16817v1#S3).

**Luồng:** `ảnh + q → encoder → entropy stream → latent → generator(latent, q, beta) → RGB`. q làm thay đổi representation và rate; beta đổi cách tái tạo từ representation. Vì vậy tăng beta không tự động tăng lượng thông tin đúng về ảnh gốc.

**Trong repo.** Một ảnh/q được nén một lần rồi thử nhiều beta. Phần hyperprior/context chạy CPU theo setup tác giả. Cần truyền hoặc cố định rõ beta trong gói decoder cuối; nó không được suy ra nhờ ảnh gốc. [Code `points_crdr`](../experiments/bakeoff/bakeoff.py), [implementation tác giả](https://github.com/iwa-shi/CRDR).

**Hàm ý:** cơ chế nhiều rate/realism giúp giảm số checkpoint cần đóng gói. Tuy nhiên, quyết định sử dụng vẫn phải dựa trên chất lượng và thời gian đo được, không chỉ số checkpoint ít hơn.

<a id="model-coolchic"></a>

### B.9 Cool-Chic — encode bằng tối ưu riêng cho từng ảnh

**Theo paper gốc.** Cool-Chic tối ưu latent phân cấp nhiều resolution cùng mạng synthesis nhỏ và probability model cho từng ảnh. Latent được đưa về cùng resolution để dựng RGB; entropy model dùng ngữ cảnh nhân quả. Bitstream phải mang cả latent lẫn tham số mạng đã lượng tử hóa. Decoder nhẹ vì chi phí tối ưu chuyển sang encoder. [Ladune và cộng sự, ICCV 2023, mục 2.1–2.4](https://arxiv.org/pdf/2212.05458).

```mermaid
flowchart LR
    X["Một ảnh"] --> OPT["Tối ưu latent và mạng cho ảnh đó"] --> B["Bits latent + tham số mạng"]
    B --> E["Entropy decode"] --> L["Latent nhiều resolution"] --> U["Upsampling + synthesis"] --> O["RGB"]
    B --> W["Tham số mạng"] --> U
    W --> E
```

**Trong repo.** Checkout được pin ở tag `clic2025`, có decoder C/AVX2; không phải cấu hình MLP nhỏ nhất của paper 2023 hoặc bản 5.x hiện hành. Run dùng lựa chọn perceptual/Wasserstein theo config và tối ưu per-image. Do kiến trúc, entropy và recipe đã thay đổi qua các phiên bản, không gán số parameter/MAC của paper gốc cho binary CLIC 2025. [Tag đã chọn](https://github.com/Orange-OpenSource/Cool-Chic/tree/clic2025), [wrapper](../experiments/bakeoff/bakeoff.py).

**Hàm ý:** thích hợp để thử có chọn lọc trên ảnh mà encoder có thể dành nhiều thời gian. Khi đánh giá phải tính byte tham số mạng; “decoder ít tham số” không có nghĩa mọi ảnh chỉ tốn byte latent.

<a id="model-mbt"></a>

### B.10 MBT2018-mean — phân biệt bản mean-scale và bản autoregressive

**Theo paper và thư viện.** Paper Minnen–Ballé–Toderici kết hợp hierarchical prior và autoregressive context để dự đoán Gaussian cho latent. Trong CompressAI, `MeanScaleHyperprior` chỉ dự đoán mean/scale từ hyperprior; `JointAutoregressiveHierarchicalPriors` bổ sung context. Hai lớp không phải cùng một decoder. [Paper NeurIPS 2018](https://arxiv.org/pdf/1809.02736); [tài liệu model CompressAI](https://interdigitalinc.github.io/CompressAI/models.html#meanscalehyperprior).

**Trong repo.** Candidate `mbt` gọi `mbt2018_mean`, qualities 1–4, metric MSE. Đường đi là analysis/synthesis CNN với hyperprior, không phải baseline raster-autoregressive đầy đủ. Vì thế kết luận tốc độ hoặc chất lượng của phép chạy này chỉ áp dụng cho bản mean-scale. [Hàm `points_mbt`](../experiments/bakeoff/bakeoff.py).

**Hàm ý:** đây là mốc tham chiếu hữu ích để tách lợi ích của generative decoder khỏi khả năng mã hóa latent thông thường. Không diễn giải kết quả kém theo Q̂ thành thất bại của cả dòng entropy model MBT.

<a id="model-vtm"></a>

### B.11 VTM / VTM-SCC — codec khối, không có neural teacher

**Theo tài liệu chuẩn.** VVC kết hợp chia khối, dự đoán, biến đổi/lượng tử residual, entropy coding và lọc vòng lặp. Với ảnh tĩnh dùng cấu hình intra. Screen-content tools khai thác vùng phẳng, màu lặp và cấu trúc lặp: IBC lấy tham chiếu trong cùng ảnh; palette dùng bảng màu; transform skip/BDPCM phục vụ residual phù hợp. [Bross và cộng sự, *Overview of the VVC Standard and its Applications*, TCSVT 2021](https://doi.org/10.1109/TCSVT.2021.3101953).

**Trong repo.** Nhánh `vtm420` dùng YUV 4:2:0 10-bit; `vtmscc` dùng 4:4:4, thử bật IBC/PLT/BDPCM/transform-skip và ghi flags thực vào point. Wrapper có cơ chế fallback nếu binary từ chối tổ hợp flags; vì vậy không mặc định mọi point đều dùng đủ công cụ SCC. [Chuyển màu và lệnh VTM](../experiments/clic_b/b6_vtm_residual.py), [wrapper bake-off](../experiments/bakeoff/bakeoff.py).

**Hàm ý:** cạnh chữ và vùng màu phẳng là lý do hợp lý để giữ mode này trong mix; cần kiểm từng lựa chọn bằng dữ liệu, không coi mọi ảnh screen đều nên dùng VTM.

<a id="model-internal"></a>

### B.12 S1 và SD-Turbo refiner — đâu là thành phần paper, đâu là thiết kế nội bộ?

**S1 trong repo.** DC-AE tạo latent 32 kênh ở resolution 1/32. `LatentCodec` dùng ResBlock cho analysis/synthesis, gain theo rate, hyperprior và subtractive dither. Hyper-synthesis cùng các quyết định CDF dùng đường integer; reconstruction transform vẫn float. Loss hiện có là rate cộng MSE latent, không trực tiếp tối ưu preference ảnh. Khi ghép residual VTM, phải cộng byte cả base và residual. Đây là thiết kế nội bộ đọc từ [latent_codec.py](../src/clic27/codec/latent_codec.py) và [trainer S1](../experiments/s1/train_s1.py), không gán thành một paper S1 công khai.

**SD-Turbo theo paper nền tảng.** Adversarial Diffusion Distillation kết hợp teacher diffusion và adversarial loss để huấn luyện student sinh ảnh với ít bước. Mục tiêu là tăng tốc generation; bản thân phương pháp này không cung cấp bitstream codec. [Sauer và cộng sự, *Adversarial Diffusion Distillation*, bản arXiv 2023, mục 3](https://arxiv.org/html/2311.17042v1).

**Refiner trong repo.** Pipeline lấy reconstruction MS-ILLM làm đầu vào SD-Turbo img2img. Đây là hậu xử lý do repo ghép, không phải codec SD-Turbo của tác giả paper. Nó tăng chi phí decoder và có thể đổi chi tiết dù không truyền thêm thông tin ảnh. Vì vậy cần chấm chất lượng, tính trọng số/runtime và xác định seed/control để tái lập.

### B.13 Từ kiến trúc đến các thí nghiệm cần làm

Các kết luận sau là **suy luận thiết kế cho repo**, chưa phải kết quả thực nghiệm mới:

| Câu hỏi | Bài học từ kiến trúc | Kiểm chứng cụ thể |
|---|---|---|
| Vì sao một bước vẫn có thể chậm? | Backbone, entropy CPU, load model và ghi ảnh đều có chi phí | Benchmark toàn gói; tách load / entropy / synthesis / I/O |
| Có nên distill latent giữa các model? | Representation và bitstream không tự tương thích | Bắt đầu từ image-space teacher; nếu dùng latent, phải học và kiểm projection |
| Làm sao giữ chữ/mặt? | Prior sinh có thể bù texture nhưng tạo chi tiết sai | Chấm mù theo nhóm ảnh, kiểm text/face; so uniform và confidence-weighted distillation |
| Làm sao giảm số model trong gói? | Variable-rate giảm nhu cầu checkpoint riêng nhưng không bảo đảm đủ dải rate | Kiểm byte thật ở rate thấp và chất lượng tại cùng budget |
| Khi nào gọi decoder tái lập được? | CDF đúng và PNG đúng là hai phép kiểm khác nhau | So symbol/latent, rồi pixel trên tiến trình và thiết bị đích |

**Ghi chú đọc nguồn:** các phần mô tả phương pháp dựa trên thân bài, hình kiến trúc và mục training, không chỉ abstract. DiT-IC và CoD-Lite được đối chiếu với proceedings; PULSE ghi rõ là preprint 2026; AFP-GIC tham chiếu chính xác bản v2 đã đọc. Với DC-VIC, đọc bản toàn văn tác giả và đối chiếu metadata với [danh mục của nhóm Tohoku](https://www.iic.ecei.tohoku.ac.jp/~machi/paper/paper-e.html). Công cụ `paper-search` không trả kết quả trong lần truy vấn này; tra cứu được hoàn tất qua nguồn paper/proceedings và repo chính thức. Không bổ sung tuyên bố SOTA, xếp hạng hoặc tốc độ mới nếu chưa có phép đo tương ứng của dự án.


Lưu ý này phản ánh tính **thực tế kỹ thuật** trong xử lý tín hiệu và nén dữ liệu: các codec nén ảnh/video không phải là một "vòi nước" có thể vặn mở chính xác đến từng bit dung lượng theo ý muốn, mà vận hành dựa trên các quy luật rời rạc và thống kê.

---

## Phụ lục C. Tại sao Codec không thể xuất ra chính xác số byte mong muốn?

* **Tham số điều phối rời rạc (Discrete Parameters):** Hầu hết các codec điều khiển dung lượng thông qua các tham số rời rạc, như bước lượng hóa $QP$ (Quantization Parameter trong HEVC/VVC) hoặc nấc chất lượng $q \in [1, 100]$ (trong JPEG). Bạn không thể chọn một $QP = 22.473$ nếu codec chỉ nhận số nguyên.
* **Bản chất biến đổi của Mã hóa Entropy (Entropy Coding):** Các bộ mã hóa entropy (như CABAC hay Arithmetic Coding) nén dữ liệu dựa trên xác suất xuất hiện của các ký hiệu. Dung lượng bitstream thực tế phụ thuộc hoàn toàn vào cấu trúc chi tiết của từng bức ảnh, khiến việc tính toán chính xác số byte trước khi nén thực sự là bất khả thi.
* **Chi phí cố định (Overhead / Side Information):** Mỗi bitstream đều chứa phần đầu (file header), bảng lượng hóa, metadata, hoặc thông tin phụ cho bộ giải mã. Những thành phần này chiếm dung lượng rời rạc và không giảm tuyến tính khi bạn hạ thấp chất lượng ảnh.

---

### 1. Phương pháp phân bổ Bitrate thực tế

Để đưa tổng dung lượng toàn tập về sát ngân sách $B_{\text{budget}}$, quy trình kỹ thuật được thực hiện theo 3 bước:

#### Bước 1: Đo đạc các điểm Rate–Distortion (R-D) mốc

Cho từng bức ảnh $i$, tiến hành nén thử nghiệm ở một tập hợp các nấc chất lượng cố định (ví dụ $q \in \{10, 20, 30, \dots, 100\}$).

Ghi nhận các cặp giá trị thực tế:


$$(R_{i, j}, D_{i, j}) = (\text{Số byte thực tế}, \text{Điểm quality } \hat{Q})$$

#### Bước 2: Xây dựng đường cong R-D và Nội suy (Interpolation)

* **Nội suy (Interpolation):** Vẽ đường cong Rate–Distortion nối các điểm mốc $(R, D)$. Nếu ngân sách tối ưu cho ảnh $i$ rơi vào khoảng giữa hai nấc $q = 30$ và $q = 40$, ta dùng nội suy (tuyến tính hoặc spline) để ước lượng nấc chất lượng trung gian hoặc tỷ lệ hòa trộn thích hợp.
* **Thử nghiệm bổ sung (Fine-tuning / Binary Search):** Với các codec cho phép tinh chỉnh tham số mịn hơn (hoặc chỉnh hệ số Lagrange $\lambda$ trong nén neural), thuật toán sẽ thực hiện **tìm kiếm nhị phân (Binary Search)** xung quanh vùng dự đoán để tìm ra tham số tiệm cận nhất với số byte mục tiêu.

#### Bước 3: Giải bài toán Tối ưu hóa (Knapsack / DP)

Sử dụng các điểm R-D (cả điểm đo thực tế và điểm nội suy) đưa vào thuật toán Quy hoạch động (Dynamic Programming) để chọn ra cấu hình nấc nén $j_i$ cho từng ảnh sao cho tổng $\sum \hat{Q}_i$ lớn nhất.

---

### 2. Tại sao bắt buộc phải "Kiểm tra lại bằng kích thước bitstream thực tế"?

Nội suy chỉ là phép **ước lượng toán học**, không phải kết quả nén thật.

1. **Tránh nguy cơ Disqualification:** Trong các cuộc thi như CLIC, ngân sách $B_{\text{budget}}$ là ranh giới cứng. Nếu tổng dung lượng thực tế vượt ngân sách dù chỉ **1 byte** do sai số nội suy, bài thi sẽ bị tính là phạm quy.
2. **Tránh lãng phí Ngân sách:** Nếu tổng dung lượng thực tế nhỏ hơn nhiều so với ngân sách cho phép, bạn đang bỏ phí dung lượng mà đáng lẽ có thể dùng để nâng cao chất lượng $\hat{Q}$ cho các ảnh khó.
3. **Tính toán đúng Side Information:** Việc xuất bitstream hoàn chỉnh và đóng gói thành file thực tế là cách duy nhất để đo chính xác 100% dung lượng bao gồm cả header, metadata và side information.

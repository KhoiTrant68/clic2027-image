# Hướng nghiên cứu trên MLIC++ cho CLIC 2027 và một công trình độc lập

**Ngày khảo sát:** 04/11/2026. **Ngôn ngữ:** tiếng Việt; giữ thuật ngữ kỹ thuật khi cần. **Trạng thái bản gốc:** đề xuất trước huấn luyện; kết quả thực nghiệm sau đó được liên kết trong cập nhật dưới đây.

> **Cập nhật sau khảo sát:** Đã thực hiện [pilot adapter](MLICpp_adapter_pilot_2026-11-04.md). Baseline perceptual cải thiện LPIPS/DISTS, nhưng head Risk hiện tại chưa vượt đối chứng CVaR và chưa đạt tiêu chí đi tiếp. Phần dưới giữ nguyên lập luận đề xuất trước thí nghiệm; không coi giả thuyết trong đó là kết quả đã được xác nhận.

## Tóm tắt và khuyến nghị

**Nên dùng MLIC++ làm nền phát triển, nhưng chưa nên chốt đóng góp là “thêm refiner”, “bảo toàn chữ” hoặc “thêm uncertainty map”.** Những cơ chế này đã có nhiều tiền lệ, gồm cả công trình rất mới thử trực tiếp trên MLIC++. Kết quả khảo sát dẫn đến một câu hỏi hẹp hơn: có thể nâng chất lượng cảm nhận của MLIC++ mà hạn chế các lỗi cấu trúc nghiêm trọng ở những vùng khó nhất, với cùng bitrate và chi phí giải mã thực tế hay không?

Mình ưu tiên **structure-risk-aware perceptual adaptation**: học một bộ điều chỉnh nhỏ bên trong synthesis transform, hướng vào đánh đổi giữa chất lượng trung bình và đuôi phân phối lỗi theo vùng. Đây là tên mô tả ứng viên do AI đề xuất, chưa phải tên phương pháp hoặc tuyên bố đã xác nhận tính mới. Bước đầu cần chứng minh lợi ích so với cùng backbone được fine-tune perceptual thông thường, text-aware loss và bộ bù latent có ràng buộc. Nếu thay đổi chỉ tương đương tăng trọng số loss, chưa đủ cơ sở gọi là đóng góp mô hình.

Nên bắt đầu bằng pilot có giới hạn thay vì huấn luyện một hệ diffusion lớn. Đầu ra cần có là đường đánh đổi **rate–perception–structural tail error–latency**, bằng chứng cơ chế qua ablation và kiểm tra ngoài 30 ảnh đã dùng chọn ý tưởng. Thứ hạng CLIC và khả năng xuất bản được đánh giá riêng, không được bảo đảm từ đề xuất này.

## 1. Điểm xuất phát của dự án

[Thử nghiệm ngày 02/11](CLIC2027_MLICpp_full30_2026-11-02.md) đã đo 30 ảnh × 4 checkpoint MSE legacy. Tại ngân sách 0,15 bpp, MLIC++ đạt PSNR 29,55 dB và Q̂ −0,9476; mix lịch sử đạt 24,59 dB và Q̂ 0,7891. Tại 0,30 bpp, hai phương án lần lượt đạt 31,96/29,34 dB và Q̂ 0,2566/1,4978. Như vậy, baseline hiện tại có ưu thế pixel fidelity nhưng chưa cạnh tranh theo proxy preference của repo.

Đây là lý do để nghiên cứu perceptual adaptation, **chưa chứng minh ảnh MLIC++ có lỗi chữ cụ thể**, cũng chưa chứng minh tối ưu Q̂ sẽ cải thiện human preference. Bản đã thử là kiến trúc legacy khoảng 116,7M tham số. README tác giả ghi một bản sửa LRP/synthesis năm 2024 còn khoảng 83,5M; đồng thời đã có MLICv2. Cần giữ tách biệt các biến thể này khi lập bảng baseline. [MLIC chính thức](https://github.com/JiangWeibeta/MLIC), [MLICv2](https://arxiv.org/abs/2504.19119).

Theo lựa chọn của bạn, **MLIC++ là baseline nghiên cứu chính**. Mix hiện tại vẫn hữu ích như mốc hiệu quả thực tế cho bài thi. Hai vai trò này không mâu thuẫn: một baseline thuận tiện để nghiên cứu không nhất thiết đang là phương án thi tốt nhất.

## 2. Phương pháp khảo sát và độ tin cậy

Khảo sát ưu tiên nguồn công khai năm 2024–2026 đến ngày 04/11/2026; bổ sung các nghiên cứu cũ trực tiếp liên quan để tránh nhận lại ý tưởng đã có. Các nhánh tìm kiếm gồm perceptual/generative decoding, content preservation, latent adaptation, entropy/context, cross-platform coding và luật CLIC. Chỉ dùng nguồn gốc: arXiv, proceedings, publisher, trang cơ quan và repo tác giả. Phạm vi đã đọc được ghi riêng cho từng nguồn trong [catalog 22 nguồn cốt lõi](research_mlicpp_2026-11-04/sources.md).

Đây là khảo sát định hướng, không phải systematic review hoặc xác nhận đã tìm hết mọi tài liệu. Không so trực tiếp BD-rate, tốc độ hoặc thứ hạng lấy từ các paper có dataset, hardware và metric khác nhau. Các số của tác giả là kết quả họ báo cáo, không phải kết quả tái lập trên server của dự án. Nhật ký truy vấn, quyết định loại ý tưởng và giới hạn truy cập nằm trong [search_and_synthesis.md](research_mlicpp_2026-11-04/search_and_synthesis.md).

## 3. Những công trình làm thay đổi lựa chọn hướng đi

| Công trình | Nội dung liên quan đã xác minh | Hệ quả cho đề xuất |
| --- | --- | --- |
| **MLICv2**, v4 tháng 02/2026 | Hậu duệ trực tiếp, bổ sung transform/context, selective compression và latent refinement | Thắng MLIC++ legacy chưa đủ để kết luận vượt phương pháp hiện đại. [Nguồn](https://arxiv.org/html/2504.19119v4) |
| **PerQ**, preprint 19/10/2026 | Bộ bù flow matching theo rate trên codec đóng băng; có thí nghiệm MLIC++ trong Table 4/Appendix B | “MLIC++ + bộ bù latent để tăng perception” đã có đối chứng rất sát. [Nguồn](https://arxiv.org/html/2610.27374v1) |
| **PICO**, CVPR 2026 | Thiết kế đồng thời chất lượng và runtime; text mask với L1 mạnh, giảm GAN trong vùng chữ | “Text-aware perceptual loss” đơn thuần không đủ mới. [Nguồn, §4.2](https://arxiv.org/html/2605.05148v1) |
| **ConHa**, bản đầy đủ 2025 | Học mức hallucination theo nội dung từ preference, dùng để điều chỉnh loss | Điều khiển mức sinh chi tiết theo nội dung đã có. [Nguồn](https://arxiv.org/html/2410.19493v2) |
| **CADC**, CVPR 2026 | Uncertainty-guided quantization, auxiliary decoder và conditioning từ ảnh phụ | Không coi uncertainty map hoặc content-adaptive rate là khoảng trống chung. [Nguồn](https://openaccess.thecvf.com/content/CVPR2026/html/Sheng_CADC_Content_Adaptive_Diffusion-Based_Generative_Image_Compression_CVPR_2026_paper.html) |
| **OneDC**, NeurIPS 2025 | One-step generation, hướng dẫn ngữ nghĩa từ hyperprior và distillation | One-step/hyperprior conditioning không phải đóng góp tự thân. [Nguồn](https://proceedings.neurips.cc/paper_files/paper/2025/hash/352a67c085e607acdcfc0076899750f4-Abstract-Conference.html) |
| **VLIC**, CVPR 2026 | Post-training codec bằng đánh giá cặp của VLM | Dùng VLM/reward để học preference đã có prior art. [Nguồn](https://openaccess.thecvf.com/content/CVPR2026/html/Sargent_VLIC_Vision-Language_Models_As_Perceptual_Judges_for_Human-Aligned_Image_Compression_CVPR_2026_paper.html) |
| **Selective Detail Decoding / AGDL**, 2020–2021 | Chọn decoder theo vùng; hoặc gửi refinement cho các pixel quan trọng | “Bảo vệ chữ/mặt bằng nhánh riêng” và “gửi sparse structural evidence” không mới ở mức khái quát. [Selective decoding](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w7/Akutsu_Ultra_Low_Bitrate_Learned_Image_Compression_by_Selective_Detail_Decoding_CVPRW_2020_paper.pdf), [AGDL](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Attention-Guided_Image_Compression_by_Deep_Reconstruction_of_Compressive_Sensed_Saliency_CVPR_2021_paper.html) |
| **GCR-C / ACV-Gate**, preprint 08–09/2026 | Đánh giá candidate sau khi hoàn tất cùng packet budget; học bộ xấp xỉ evaluator | Counterfactual selection và amortized scorer cũng cần ghi nhận tiền lệ. [GCR-C](https://arxiv.org/html/2608.16192v1), [ACV-Gate](https://arxiv.org/abs/2609.30756) |
| **HPCM / dictionary entropy**, ICCV/CVPR 2025 | Context phân cấp và dictionary prior cho entropy | Tránh chọn thay attention/memory như đóng góp chính nếu không có cơ chế hoặc bằng chứng mạnh hơn. [HPCM](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learned_Image_Compression_with_Hierarchical_Progressive_Context_Modeling_ICCV_2025_paper.html), [dictionary entropy](https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_Learned_Image_Compression_with_Dictionary-based_Entropy_Model_CVPR_2025_paper.pdf) |

Kết luận từ các nguồn trên là **mức độ trùng lặp của các ý tưởng khái quát cao**. Không suy ra rằng mọi cải tiến trên MLIC++ đều hết cơ hội. Cần biến ý tưởng thành một câu hỏi có đối chứng đủ gần và một cơ chế cụ thể có thể bị thực nghiệm bác bỏ.

## 4. Ba hướng đáng cân nhắc

Thứ tự dưới là đánh giá ưu tiên cho dự án này, không phải xác suất thành công hoặc điểm novelty đã đo.

| Ưu tiên | Hướng | Câu hỏi có thể kiểm chứng | Rủi ro chính |
| --- | --- | --- | --- |
| **1** | **Perceptual adaptation có xét lỗi cấu trúc ở đuôi phân phối** | Tại cùng rate và chi phí, learned synthesis adaptation có cải thiện frontier cảm nhận–lỗi nặng nhất hơn loss weighting/gating thông thường không? | Có thể chỉ là CVaR training trên kiến trúc cũ; nếu vậy đóng góp mô hình yếu. |
| **2** | **Structural innovation có tính đủ byte** | Khi không thể suy ra cấu trúc từ base latent, gửi ít evidence có hiệu quả hơn dành cùng số bit cho base hoặc residual thường không? | AGDL/GCR-C đã gần; overhead, vị trí và tương tác giữa vùng có thể xóa hết lợi ích. |
| **3** | **Entropy chịu sai khác thiết bị với latency thấp** | Có thể giảm nhu cầu tính toán chính xác ở mọi bước context mà vẫn bảo đảm đúng symbol trong giả thiết sai khác xác định không? | Khó về proof và triển khai; chưa trực tiếp giải quyết perception. PMATIC và integer entropy đã có tiền lệ. |

Với hướng 3, phải đọc và đối chiếu sâu [Adler & Tang, ICML 2026](https://proceedings.mlr.press/v306/adler26b.html), [EF-LIC, ICML 2026](https://proceedings.mlr.press/v306/cao26e.html) và [PULSE, preprint 09/2026](https://arxiv.org/abs/2609.18602). Chỉ dùng INT8 hoặc ép chạy CPU không đủ làm tuyên bố mới. Tương tự, high-rate teacher cho low-rate encoder đã có [self-teacher distillation, ECCV 2026 theo EURECOM](https://www.eurecom.eu/en/publication/8851); đây là đối chứng hoặc công cụ hỗ trợ, chưa phải đóng góp chính đề xuất.

**Khuyến nghị chọn hướng 1 để làm pilot đầu tiên.** Hướng 2 chỉ mở sau khi xác định rõ phần lỗi không khắc phục được bằng decoder-only adaptation. Giữ hướng 3 như yêu cầu triển khai và một nhánh riêng nếu profiling cho thấy đó mới là nút thắt đáng nghiên cứu.

## 5. Hướng ưu tiên: học tái tạo cảm nhận với ràng buộc lỗi cấu trúc theo vùng

### 5.1. Giả thuyết và phạm vi claim

**Giả thuyết đề xuất:** tối ưu chất lượng trung bình có thể để lại một số vùng hỏng nặng. Một synthesis adapter được học để phản ứng với dấu hiệu mất thông tin trong latent có thể tạo đường đánh đổi tốt hơn giữa perception trung bình và lỗi cấu trúc ở đuôi phân phối, so với dùng một trọng số perceptual chung hoặc một mask cố định.

Đây là giả thuyết của người khảo sát, **chưa phải kết quả của các paper được dẫn**. “Lỗi cấu trúc” phải được định nghĩa bằng phép đo cụ thể, chẳng hạn lỗi ký tự trên vùng có transcript, sai khác cạnh/hình học và sai khác đặc trưng theo vùng. Không dùng một detector hoặc một OCR score để đại diện cho mọi thay đổi ngữ nghĩa. Các quan sát về nội dung bị thay đổi trong [Bergmann và cộng sự, 2026](https://link.springer.com/article/10.1007/s11042-026-21317-3) là động cơ đánh giá riêng lỗi nội dung, không chứng minh baseline local đã có những lỗi đó.

### 5.2. Thay đổi tối thiểu trong mô hình

Giai đoạn đầu giữ analysis transform, hyperprior và entropy/context của MLIC++ cố định. Sau khi giải mã symbol, một adapter nhỏ nhận latent đã giải mã, feature synthesis và nhãn rate để dự đoán **hiệu chỉnh feature có giới hạn biên độ**. Adapter nằm trong synthesis transform, không sửa xác suất entropy hay nhận ảnh gốc lúc decode. Khởi tạo hiệu chỉnh bằng 0 để có điểm bắt đầu đúng baseline.

```text
Ảnh gốc → MLIC++ encoder → stream thực
                               ↓
                       entropy decoder cố định
                               ↓
                 latent + feature synthesis + rate
                               ↓
                synthesis adapter → ảnh tái tạo
```

Trong training, loss cấu trúc có thể nhìn ảnh gốc và annotation; decoder triển khai chỉ được dùng thông tin tái tạo từ stream. Nếu một nhánh sau này cần mask hoặc token tính từ ảnh gốc, chúng phải được truyền và tính vào rate. Giữ nguyên entropy không tự bảo đảm compatibility nếu ta sửa latent/context trước khi hoàn tất giải mã; đặt adapter sau symbol decoding giúp tránh sự nhập nhằng này.

Không cần Stable Diffusion ở pilot đầu. Giới hạn kích thước adapter là một biến thực nghiệm, không mặc định càng nhỏ càng tốt. Đối chứng phải có cùng số tham số hoặc cùng latency để tránh kết luận rằng thêm năng lực mô hình tự nó là cơ chế đề xuất.

**Một cấu hình đủ cụ thể để thử:** tại một tầng synthesis có feature `f`, nhánh bottleneck đề xuất hiệu chỉnh `Δf = a·tanh(Aφ(f, ŷ, q))`. Một head nhỏ nhận cả feature lẫn **hiệu chỉnh dự kiến** để ước lượng nguy cơ tăng lỗi: `s = Hψ(f, Δf, ŷ, q)`. Feature mới là `f′ = f + g(s, t)·Δf`, với `t` là mức chấp nhận thay đổi và `g` là gate khả vi. Khởi tạo đầu ra A bằng 0. Thử một vị trí trước; chưa cần ba tầng hoặc hai decoder đầy đủ. Đây là candidate implementation, không phải kiến trúc đã được kiểm nghiệm.

Điểm cần kiểm ở head là **nguy cơ do một thay đổi tái tạo cụ thể gây ra**, khác với chỉ dự đoán “đây là vùng chữ”. Trong training, nhãn surrogate được tạo từ chênh lệch lỗi vùng giữa ảnh có/không có hiệu chỉnh, dùng cùng ảnh nguồn; lúc decode head chỉ nhận feature đã giải mã và `Δf`. So head này với text mask, uncertainty từ scale, gate không có `Δf` và gate cùng tham số nhưng không học nguy cơ. Nhãn phụ thuộc OCR/edge proxy, nên head chưa được gọi là bộ dự báo rủi ro có calibration. Generic residual gating cũng chưa được nhận là mới; đóng góp chỉ được cân nhắc nếu phép tách này tạo lợi ích vượt các control.

### 5.3. Mục tiêu học dự kiến

Đặt `d_s(x, x̂, r)` là lỗi cấu trúc trên vùng `r`. Một phiên bản thử nghiệm sử dụng:

```text
L = R + λd Dpixel + λp Dperceptual
    + γ [τ + E(max(d_s − τ, 0)) / (1 − α)]
```

Biểu thức trong ngoặc là dạng tối ưu của **CVaR**, công cụ chuẩn để tập trung vào phần đuôi lỗi lớn; không nhận công thức CVaR là đóng góp. Phải quy định rõ đơn vị lấy mẫu: ví dụ chọn ảnh đều trước rồi chọn vùng theo quy trình cố định, thay vì để ảnh nhiều chữ tự chiếm gần hết batch. Khi encoder/entropy đóng băng và không gửi thêm payload, `R` không thay đổi theo adapter; lúc đó loss rate là hằng và không được quảng bá như đã tối ưu joint rate. Chỉ fine-tune encoder ở giai đoạn sau mới làm bài toán rate thay đổi thực sự.

`τ` phải được tối ưu cùng objective trên tập training, có thể tách theo rate đã định trước; không đặt tùy ý rồi gọi biểu thức thu được là CVaR. `α`, định nghĩa vùng và cách chuẩn hóa lỗi được chốt bằng validation. Với mini-batch nhỏ, ước lượng đuôi có thể nhiễu; cần báo phân phối lỗi trên tập đánh giá đủ lớn thay vì dùng một batch loss để kết luận.

Sẽ thử đồng thời hai cách định nghĩa lỗi: **lỗi tuyệt đối so với ảnh gốc** và **phần lỗi tăng thêm so với decoder MSE**. Chỉ dùng loại thứ hai có thể chấp nhận việc giữ nguyên một chữ vốn đã sai ở base; đó là lỗi cần tránh. Các vùng không có chữ vẫn phải được đánh giá để mô hình không chuyển lỗi sang texture hoặc hình học.

Để tránh tùy chọn metric sau khi thấy kết quả, **pilot đầu chốt một surrogate khả vi**: sai khác gradient Sobel trung bình trên các vùng 128 × 128, chuẩn hóa bằng một hệ số cố định ước lượng từ training split rồi chặn về [0,1]. Chọn ảnh đều, sau đó vùng đều; `τ` là scalar được học riêng cho mỗi rate. Đây chỉ là proxy cạnh, không phải lỗi ngữ nghĩa. Nhãn cho head là chênh lệch proxy khi có/không có hiệu chỉnh; loss chính vẫn phạt lỗi tuyệt đối. Đánh giá độc lập bổ sung CER trên tập chữ có transcript và các phép đo cạnh khác surrogate training. Không cộng CER không khả vi trực tiếp vào training loss. Một surrogate khác chỉ được thử như robustness check đã khai báo, không thay metric chính để chọn kết quả đẹp hơn.

Thử `α` theo một lưới nhỏ trên validation riêng, rồi cố định trước test. Không chọn `α` bằng chính 30 ảnh CLIC đã dùng hình thành ý tưởng. Empirical CVaR không phải bảo chứng population risk, per-image safety hoặc giữ đúng mọi ký tự. Muốn thêm calibration/theory phải có giả thiết và đối chiếu riêng với [CoRAS](https://arxiv.org/abs/2607.26887) và [Conf-Gen](https://proceedings.mlr.press/v306/loaiza-ganem26a.html); không đưa cam kết xác suất vào claim hiện tại.

### 5.4. Phần nào có thể thành đóng góp?

Một kết quả có giá trị cần trả lời được: **dấu hiệu nào từ compressed representation giúp chọn cách tái tạo, vì sao cơ chế đó cải thiện frontier, và nó khác gì với chỉ đổi loss?** Có thể giữ entropy-derived uncertainty như một input để thử, nhưng phải so với CADC và input đơn giản hơn. Không mặc định đó là tín hiệu rủi ro có calibration.

Nếu cùng kiến trúc chỉ thêm CVaR đạt hết lợi ích, phải mô tả trung thực là thay đổi objective, không giữ một adapter phức tạp để làm đẹp câu chuyện. Nếu weighted mean hoặc text-aware loss đạt cùng frontier, giả thuyết về cơ chế mới chưa được hỗ trợ. Nếu chỉ có lợi trên MLIC++ legacy mà không chuyển được sang backbone hoặc checkpoint khác, phạm vi đóng góp hẹp và cần nói rõ.

## 6. Pilot để quyết định có đáng đầu tư tiếp không

Đây là kế hoạch đề xuất, chưa phải job đã chạy. Các mốc là thời gian làm việc và ngân sách thử dự kiến, không phải dự báo tốc độ huấn luyện đã benchmark. Giữ giới hạn VRAM đã dùng của dự án; xác nhận GPU thực tế còn sẵn và profile trước khi ước lượng tổng GPU-giờ.

| Giai đoạn | Việc cần làm | Điều kiện đi tiếp |
| --- | --- | --- |
| **P0a — audit/profile**, khoảng 1–2 ngày làm việc | Chốt biến thể MLIC++, tải/kiểm checkpoint mới phù hợp; kiểm training data/license; dựng phép đo vùng và baseline perceptual ở rate đã tái lập, ưu tiên khoảng 0,15 bpp. | Tái lập decode thật; đo được loại lỗi mục tiêu và có dữ liệu đánh giá độc lập. |
| **P0b — độ phủ rate thấp**, chưa ước lượng thời lượng | Tìm checkpoint hoặc fine-tune cấu hình cho 0,075 bpp, kiểm số byte thật; không giả sử q1 hiện có đã đủ. Có thể làm sau pilot một rate. | Đủ rate cho mục tiêu thi; nếu chưa đủ thì ghi rõ, không kéo dài mốc audit 1–2 ngày thành cam kết training. |
| **P1 — pilot nhỏ**, cap đề xuất 24–48 GPU-giờ sau profiling | Fine-tune một rate trên crop với cùng lịch học: perceptual baseline, text-aware baseline, CVaR cùng architecture, rồi adapter có/không tín hiệu rủi ro. | Có đường đánh đổi tốt hơn control trên validation riêng; không chỉ tăng Q̂. |
| **P2 — xác nhận** | Mở rộng nhiều rate, nhiều seed; kiểm transfer sang bản MLIC++ khác hoặc MLICv2 nếu tài nguyên cho phép; profiling full-resolution trên L4. | Hiệu ứng tồn tại ngoài pilot, tính đủ chi phí và không chỉ do thêm tham số. |
| **P3 — evidence cho paper/CLIC** | Test độc lập đã khóa, ablation đầy đủ, đánh giá cặp người xem theo kế hoạch phù hợp; đóng gói decoder trước thời điểm freeze chính thức. | Chỉ đưa claim phù hợp với bằng chứng; lựa chọn bài thi và paper được quyết định riêng. |

Trong P1, không đồng thời đổi encoder, entropy model, discriminator, dữ liệu và adapter. Cần một baseline perceptual đủ ổn rồi thay từng yếu tố; nếu không sẽ không biết lợi ích đến từ đâu. Chi phí pretraining của phương pháp đối thủ và chi phí fine-tune của ta phải báo cáo riêng, không gọi một so sánh như vậy là cùng training budget nếu chưa thực hiện được.

**Điều kiện dừng:** nếu sau pilot không cải thiện frontier so với control đơn giản trong sai số đo, dừng nhánh adapter và kiểm lại giả thuyết. Không cộng thêm diffusion, selector và side stream chỉ để duy trì tên framework. Nếu ưu thế chỉ xuất hiện trên proxy được dùng huấn luyện, cần kiểm tra khả năng khai thác điểm yếu metric trước khi mở rộng.

Ngay trong P0, phải xác nhận các lỗi vùng đủ thường gặp trên phân phối mục tiêu và baseline perceptual để đáng tối ưu. Nếu không quan sát được vấn đề này ngoài những ví dụ được chọn riêng, dừng ưu tiên tail-risk. Khi so frontier, giữ rate như nhau và quét trọng số/tolerance của **mọi** control với ngân sách tuning tương đương; chỉ so một cặp trọng số thuận lợi cho mô hình mới là không đủ.

**Ngưỡng quyết định P1 đề xuất, cần khóa trước khi chạy:** cùng base stream nên byte bằng nhau; so vùng trade-off có LPIPS và DISTS không tăng quá 2% so với perceptual control, tìm mức giảm ít nhất 10% của CVaR lỗi vùng trên holdout. Báo CI bootstrap theo ảnh cho chênh lệch, mean error và subset OCR; không chấp nhận tăng lỗi nội dung rõ rệt chỉ để đạt ngưỡng proxy. Đo adapter có tăng tổng decode quá 15% trên cùng máy hay không. Các mức 2%/10%/15% là **ngưỡng quản lý thử nghiệm do dự án đề xuất**, không phải dự báo hiệu quả hoặc luật CLIC. Một seed chỉ đủ sàng lọc; P2 phải kiểm độ ổn định và metric độc lập trước khi có claim nghiên cứu.

## 7. Ma trận đối chứng tối thiểu

| Câu hỏi | Đối chứng cần có |
| --- | --- |
| Có phải chỉ đổi MSE sang perceptual? | MLIC++ MSE; cùng MLIC++ fine-tune perceptual chuẩn, cùng dữ liệu/lịch học |
| Có phải chỉ bảo vệ chữ bằng loss? | PICO-style text-aware training được ghi rõ là reimplementation có giới hạn; loss gốc/weighted mean/CVaR cùng architecture |
| Có phải lợi ích từ semantic/spectral loss? | Nếu dùng CLIP/DINO hoặc frequency guidance, thêm control loss tương ứng với [SSCL, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38515); không quy lợi ích của loss cho adapter |
| Adapter có cần thiết? | Không adapter; adapter cùng số tham số nhưng không risk input; adapter dùng input đơn giản; sửa tất cả synthesis tương đương compute |
| Có trùng bounded compensation? | PerQ official/reproduction khả dụng trên rate chồng lấp; nếu chưa chạy được, ghi thiếu closest baseline, không gọi là vượt SOTA |
| Có phải một baseline cũ quá yếu? | Bản MLIC++ mới và MLICv2; thêm một backbone ngoài họ nếu khả thi |
| Lợi ích có tổng quát? | Ảnh tự nhiên, screen/game, chữ nhỏ và cấu trúc lặp; tách theo nguồn và nội dung |
| Có đáng dùng để thi? | Mix hiện tại, decoder mới trên L4, cùng ngân sách byte toàn tập |

Với nhánh structural innovation dự phòng, phải giữ **cùng tổng budget B**: base dùng `B − b`, repair dùng `b`, so với toàn bộ `B` dành cho base. Đối chứng mạnh và rẻ là cùng selector nhưng payload chỉ gồm sparse DCT/wavelet hoặc patch residual được serialize thật. Không chèn nguyên patch ground truth mà bỏ qua số byte rồi coi đó là codec đạt được. Đây là thiết kế thực nghiệm đề xuất; generic residual coding không được nhận là đóng góp mới.

Tách hai tầng so sánh: **quy đóng góp cho cơ chế** dùng cùng MLIC++ checkpoint/data/schedule; **cạnh tranh với phương pháp ngoài** dùng official hoặc reproduction đủ tương thích, công bố khác biệt training/pretraining. PerQ đã retrain MLIC++ đa rate nên không tự là matched-training control cho checkpoint legacy của ta. PICO-style loss cũng không đồng nghĩa đã tái lập toàn bộ PICO.

## 8. Đánh giá cần đủ sức đỡ claim

**Rate và chất lượng:** đo file nén thật, đủ header và mọi side information; dùng nhiều điểm rate có miền chồng lấp. Báo PSNR, LPIPS, DISTS, MS-SSIM và Q̂; không suy ra mức thắng người xem từ Q̂. Bảng so với lịch sử phải ghi khác môi trường. Lưu ý FAQ CLIC dùng trung bình MS-SSIM có trọng số pixel, trong khi báo cáo nội bộ hiện lấy trung bình theo ảnh; nên báo cả hai cách hoặc chuyển bảng official-aligned sang đúng quy ước, không trộn âm thầm. [FAQ CLIC](https://clic2027.compression.cc/faq/).

**Lỗi cấu trúc:** báo phân phối và phần đuôi, không chỉ một số trung bình. Với OCR, cần transcript hoặc subset có ground truth đáng tin cậy; báo coverage, insertion/deletion/substitution và ví dụ lỗi, không chỉ những crop OCR đọc thành công. Với cạnh/hình học, kiểm cả vùng không có chữ. Nếu cùng OCR hoặc feature extractor xuất hiện trong loss và test, bổ sung recognizer/metric độc lập. Không biến việc khớp embedding thành bằng chứng giữ nguyên ý nghĩa.

**Thống kê:** bootstrap theo ảnh, giữ các vùng cùng ảnh trong cùng cụm; báo nhiều seed ở thí nghiệm chính khi khả thi. Chọn hyperparameter trên validation riêng. Test 30 ảnh CLIC là một benchmark nhỏ và đã được xem nhiều lần; cần một test độc lập lớn hơn, loại trùng và tránh crop của cùng ảnh rơi vào nhiều split. BD-rate chỉ tính với metric/đường cong phù hợp, không nội suy qua miền rate không được đo.

**Triển khai:** đo thời gian đầu-cuối gồm load/I/O và VRAM tiến trình trên L4; kiểm entropy symbol/CDF consistency khác tiến trình và khác thiết bị. Không lấy 42–43 giây API decode trên A30 hiện tại làm tốc độ L4. Inference không dùng ảnh gốc; mọi thông tin phụ đều phải có nguồn decoder nhận được. Nếu làm human study mới, cần kế hoạch phù hợp với đơn vị chịu trách nhiệm; bản khảo sát này không tuyển người hoặc xác nhận phê duyệt.

## 9. CLIC và paper: hai lộ trình có liên hệ nhưng độc lập

CLIC yêu cầu decoder ở validation, decoder đóng băng cho test và whitepaper. **Whitepaper không bắt buộc technical novelty**; paper PCS qua phản biện riêng. Không top 1 vẫn có đường viết whitepaper, nhưng chưa đồng nghĩa có paper nghiên cứu độc lập. [Quy định chính thức](https://clic2027.compression.cc/tasks/).

Ba rate là 0,075/0,15/0,30 bpp, xếp hạng theo human preference. Giới hạn gồm decoder 4 GB, L4 24 GB VRAM, 2 CPU, 12 GB RAM. Trần 5 giờ và loại 25% chậm nhất là hai điều kiện riêng; chưa có ngưỡng giây cố định bảo đảm eligibility. Cần kiểm lại trước khi nộp. [Image task](https://clic2027.compression.cc/tasks/).

Cho paper độc lập, câu chuyện nên là **một đánh đổi được phát hiện và một cơ chế có bằng chứng**, thay vì “MLIC++ cộng nhiều module”. Tiêu đề làm việc có thể là *Perceptual Adaptation of Learned Image Codecs with Regional Structural Tail-Risk Objectives*. Tên này chỉ mô tả câu hỏi; chưa dùng những từ như “first”, “guaranteed” hoặc “state of the art”. Nếu learned mechanism không đóng góp vượt control, cần thu hẹp claim hoặc dừng, thay vì coi việc có tên mô hình là đủ cho paper.

## 10. Giới hạn và quyết định đề xuất

Tính mới hiện vẫn là **có điều kiện, chưa xác nhận toàn diện**. Tìm kiếm không đủ chứng minh vắng mặt một công trình. Một số nguồn chỉ đọc abstract hoặc sections; Semantic Scholar API và vài PDF không truy cập được. Chưa kiểm hết proceedings cuối năm, bằng sáng chế, retraction/COI hoặc giấy phép mọi checkpoint. Các nguồn mới phát hiện như ProChroma phải được đọc trước nếu nhánh sau chạm trực tiếp sparse/counterfactual coding.

Chưa có bằng chứng tail risk là vấn đề nổi trội nhất của đúng checkpoint local, cũng chưa có số GPU-giờ đáng tin cậy cho training. P0 phải đo để kiểm tiền đề này. Human preference thực tế và top 1 CLIC không thể suy ra từ objective hoặc từ việc dùng MLIC++.

**Quyết định đề xuất:** giữ MLIC++ làm nền, chuẩn hóa baseline hiện đại, rồi chạy một pilot hẹp cho **perceptual synthesis adaptation với mục tiêu lỗi cấu trúc theo vùng**. Chỉ nâng lên đóng góp mô hình khi chứng minh được lợi ích vượt text-aware loss, objective-only controls và closest perceptual baseline tại cùng rate/chi phí. Chưa đầu tư nhánh sparse evidence hoặc diffusion lớn trước khi pilot chỉ ra chúng thực sự cần thiết.

## Tài liệu và minh bạch nghiên cứu

- [Danh mục nguồn, metadata và phạm vi đã đọc](research_mlicpp_2026-11-04/sources.md).
- [Chiến lược tìm kiếm, tổng hợp và phản biện](research_mlicpp_2026-11-04/search_and_synthesis.md).
- [Phạm vi được tác giả xác nhận](../plans/MLICpp_research_scope_2026-11-04.md).
- [Bằng chứng baseline MLIC++ của dự án](../results/2026-11-02_mlicpp/full30/comparison.csv).

AI được dùng để tìm, phân công đọc nguồn, đối chiếu, phản biện và soạn đề xuất. Các nhánh đọc dùng cùng hệ thống nên không được coi là các reviewer độc lập thống kê. Người dùng đã xác nhận phạm vi và bỏ qua lựa chọn review form; chưa phê duyệt thiết kế nghiên cứu cuối cùng hoặc kết quả novelty. Mọi cơ chế mới trong báo cáo là giả thuyết cần kiểm, không phải kết quả thực nghiệm được tạo ra trong lần khảo sát này.

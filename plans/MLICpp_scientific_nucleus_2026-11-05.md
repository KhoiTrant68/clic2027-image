# Hạt nhân khoa học đề xuất: mã hóa để sửa prior tin sai

## Quyết định và trạng thái

**Cập nhật định hướng paper:** [khảo sát top venues ngày 05/11](../reports/MLICpp_top_venue_research_2026-11-05.md) ưu tiên diagnostic về predictive accessibility của bằng chứng từ bitstream trước khi chọn cơ chế mới. Prior-conflict bên dưới vẫn chưa được xác nhận; không coi hướng mới là tên khác của một kết quả đã thành công. Novelty tiếp tục REVISE, chỉ đề xuất feasibility pilot có điều kiện dừng. [Vòng fixed mới](../reports/MLICpp_fixed_improvement_2026-11-05.md) giảm LPIPS bằng tăng train/search, chưa tạo đóng góp mới.

**Bổ sung theo yêu cầu chứng minh:** [ghi chú toán học](../reports/MLICpp_prior_conflict_math_2026-11-05.md) có phép phân rã gain, phản ví dụ uncertainty misspecification, Gaussian bit converse và chặn lỗi lượng tử hóa. Đây là các mệnh đề nền/công cụ chuẩn, không chứng nhận novelty. Phản ví dụ còn cho thấy phải so với ordinary residual và uncertainty đã hiệu chỉnh; không được nhận mọi lợi ích của một decoder tốt hơn là gain do side bits.

**Hoàn thiện phần hình thức:** ghi chú đã bổ sung residual equivalence, chứng minh conditional basis thắng fixed rank-one trong một lớp nguồn cụ thể, điều kiện cùng tổng bit, selector regret và thuật toán finite-codebook. Toy codec đã serialize/giải mã 16 packet thật với chi phí bằng nhau. Không có theorem thắng mọi residual codec; neural model prior-conflict trên MLIC++ vẫn chưa được xây hoặc kiểm hiện tượng. Đây là hoàn thiện lập luận và prototype toán học, không phải hoàn tất một đóng góp CVPR.

Người dùng yêu cầu một đóng góp khoa học nổi bật, không chỉ fine-tuning hoặc thêm module. Đề xuất tập trung vào **prior-conflict innovation coding**: học cách dành bit cho phần nội dung mà decoder dự đoán sai một cách quá tự tin, thay vì chỉ dành bit theo độ bất định hoặc độ lớn residual.

Đây là **giả thuyết nghiên cứu**, chưa phải phát hiện trên MLIC++ hoặc novelty đã xác nhận. Không khởi chạy full training từ tài liệu này. Novelty verdict: **REVISE** — đã đối chiếu các cơ chế gần, nhưng phần “sửa prior tin sai bằng conditional information” còn có thể trùng semantic residual/mismatch adaptation. Cần kiểm nguồn gần hơn và chứng minh hiện tượng trước khi chốt kiến trúc. Thí nghiệm full-synthesis đã xong là nền kỹ thuật, không phải bằng chứng cho hypothesis mới.

Tài liệu này thu hẹp hạt nhân của [đề xuất innovation trước](MLICpp_perceptual_contender_2026-11-05.md): `U(ŷ)c` là một đối chứng biểu diễn có thể dùng, không đủ làm contribution trung tâm. Không đổi tên cơ chế cũ thành một đóng góp đã có.

## Một câu hỏi duy nhất

**Với cùng tổng bit và cùng giới hạn decoder, liệu mã hóa thông tin để thay đổi một prior tin sai có hiệu quả hơn mã hóa residual hoặc giảm posterior variance?**

Hai tình huống cần phân biệt:

- **Uncertain but relatively harmless:** nhiều cấu hình texture có thể hợp lý và không làm thay đổi nội dung quan trọng. Posterior variation lớn không tự làm mỗi bit giải quyết variation đó có giá trị cao.
- **Confidently wrong:** các dự đoán của mô hình tập trung quanh một nội dung sai so với nguồn. Disagreement/variance thấp không có nghĩa nguồn được khôi phục đúng.

Hai khái niệm này là cách đặt vấn đề của dự án; chưa có nhãn hoặc dữ liệu chứng minh trên codec hiện tại. Không gọi mọi lỗi lớn, mọi confidence thấp, hay mọi ảnh ngoài miền là prior conflict.

## Cơ chế cụ thể để nghiên cứu

Base stream `b` tạo điều kiện cho một phân bố tái tạo học được `q₀(x | b)`. Encoder có nguồn và có thể kiểm reconstruction/prediction của decoder. Nó tạo một innovation code `c` lượng tử hóa, mã hóa theo `p(c | b)` và truyền cùng base.

Decoder mới học `qθ(x | b, c)`, trong đó `c` có thể **thay đổi conditional mode/điều kiện tái tạo**, không chỉ điều chỉnh một trọng số realism hoặc cường độ adapter. Nếu chọn mô hình bounded support, phải kiểm support đó có cho phép sửa các lỗi cần sửa hay không; nhân likelihood vào một phân bố có xác suất đúng bằng 0 không thể tạo support mới.

Đóng góp model có thể nằm ở representation của `c`, cách cập nhật prior và objective học mức cải thiện trên mỗi bit. Điểm khác biệt cần chứng minh không phải có side stream, likelihood conditioning, Bayes rule, residual hoặc mixture-of-experts — tất cả đều là công cụ đã biết.

Giá trị thông tin cần đo trên candidate thực:

`gain(c; x,b) = d(x, G(b)) − d(x, G(b,c))`

và giá phải là `bytes(b,c) − bytes(b)`, tính đủ header, scale, index/location và padding. Đây là thước đo operational cho pilot, không phải theorem mới hay bảo đảm human preference. Khi chọn candidate theo một metric, phải đánh giá bằng metric khác và về sau là người xem.

## Đối chiếu nguồn đã kiểm

| Nguồn primary và phạm vi | Phần đã có | Delta còn phải kiểm |
| --- | --- | --- |
| [PSC, v3](https://arxiv.org/html/2407.09896v3), đã đọc §1–3 và algorithm chọn rows | Chọn phép đo từ posterior uncertainty/PCA; decoder dựng cùng transform | Không nhận adaptive uncertainty coding là mới. Prior-conflict phải tạo lợi ích ngoài variance ranking, không chỉ đổi PCA thành mạng. |
| [PerQ](https://arxiv.org/html/2610.27374v1), methods/appendix đã đọc | Frozen RD codec + conditional flow compensation trong quantization cell | Truyền thông tin nguồn để đổi conditional reconstruction là can thiệp khác, nhưng cần so gain/rate với compensation không thêm bit. Không tự suy rằng PerQ chắc chắn không sửa được lỗi cụ thể. |
| [SDGIC v2](https://arxiv.org/html/2512.06344v2), §3.2.3–3.4 đã đọc | Source-specific reconstruction-aware semantic residual tokens | Nguồn gần nhất chống novelty chung. Cần chứng minh prior-conflict representation/update khác và hiệu quả hơn semantic residual under same byte budget. |
| [LoRA-comp, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Test-Time_Fine-Tuning_of_Image_Compression_Models_for_Multi-Task_Adaptability_CVPR_2025_paper.pdf), nhánh audit đã đọc methods | Instance-specific decoder adaptation và truyền low-dimensional updates | Không nhận sửa decoder bằng ít hệ số là contribution; giữ như control gần. |
| [GCR-C](https://arxiv.org/html/2608.16192v1), protocol/method đã đọc | Full-budget counterfactual candidate evaluation | Gain-based selection, xét future budget hoặc scorer không tự là mới. Cơ chế cần nằm ở representation/prior update, không chỉ policy chọn vùng. |
| [GR-GIC](https://arxiv.org/html/2610.29691v1), §2–3 đã đọc | Entropy context parameterizes latent refinement; perturbation regularization | Không nhận dùng entropy context hoặc predicted Gaussian scale làm sáng kiến. Latent entropy scale không mặc định bằng posterior uncertainty của nội dung ảnh. |

Tìm bổ sung với các nhóm từ: generative compression + prior mismatch/confidence/negative guidance; posterior adaptive sensing; entropy context reuse; successive rate-perception refinement. Các hướng cuối không được chọn làm hạt nhân trong tài liệu này. Không có exact-title hit không chứng minh novelty. Đây là khảo sát định hướng, không phải systematic review.

## Phép thử đầu tiên phải bác bỏ được giả thuyết

**Fatal assumption:** có đủ trường hợp prior dự đoán tập trung nhưng sai, và chúng nhận được lợi ích correction cao hơn mức đã giải thích bởi variance/residual thông thường.

MLIC++ full-synthesis hiện tại là **deterministic**. Chưa có conditional posterior được kiểm định. Không được tạo noise/dropout tùy ý rồi gọi độ phân tán là posterior uncertainty; không dùng σ entropy của MLIC++ như confidence về hình ảnh. Trước hết cần xác minh một conditional generative model hoặc calibrated predictive model đủ năng lực trên nguồn giữ riêng. Mô hình thay thế đó phải được ghi rõ, không suy ra nó đại diện cho decoder hiện tại.

Khi có measurement phù hợp, phép thử bounded là:

1. Cùng nguồn/base stream, xây correction candidates theo một protocol cố định. Đo actual bytes và gain sau giải mã thật. Chưa học một framework nhiều module.
2. So khả năng dự đoán gain của variance, residual magnitude và score conflict. Kiểm conflict còn giá trị **sau khi đã kiểm soát hai yếu tố kia**, không chỉ report một tương quan đơn.
3. Phân nhóm bằng thông tin nguồn/calibration trước khi xem gain. Học/tune trên train/development; xác nhận trên ảnh nguồn mới. Không dùng 32 validation đã đọc để báo phát hiện xác nhận.
4. So cả residual thông thường, base-more-bits, latent TTO và conditional/global-basis control dưới cùng tổng byte. Không nội suy chất lượng từ checkpoint khác rồi coi là rate matched.

**Dừng:** không đo được prior confidence có ý nghĩa; conflict không dự báo correction value ngoài controls; correction không thắng ordinary residual/base-more-bits; hoặc gain chỉ xuất hiện ở metric được tối ưu. Khi một điều kiện này xảy ra, bỏ hạt nhân này, không chuyển kết quả null thành bằng chứng ủng hộ.

## Những gì hiện chưa được phép kết luận

- Chưa có kết quả “confidently wrong” trên model local; các screen OCR/CNN cũ không chứng minh điều đó.
- Chưa có model prior-update, code innovation hoặc human evaluation tương ứng.
- Chưa thể nói có lower bound/theorem mới; decision theory và information gain là công cụ chuẩn.
- Chưa có cơ sở nói tốt hơn PSC/PerQ/SDGIC, đạt top CLIC hoặc đủ cho CVPR.

Phạm vi hiện tại là **candidate scope + literature grounding**, không phải kế hoạch full run. Hạt nhân đáng đầu tư hơn một vector hiệu chỉnh chung vì nó chỉ rõ hiện tượng, cơ chế và phép phản chứng; nó chỉ trở thành đóng góp nổi bật khi các bằng chứng trên đứng vững.

## Cập nhật sau phép thử conditional correction

Đã thực hiện [pilot train/infer/eval trên server](../reports/MLICpp_innovation_pilot_2026-11-05.md): ba nhánh 1.000 bước, 32 nguồn test mới, 160/160 fresh decode matches. Conditional kém fixed về LPIPS; DISTS chưa phân biệt rõ. Bỏ claim về lợi ích conditioning ở recipe đã thử. Gain so base có thêm byte và không chứng minh prior-conflict. Pilot dùng decoder deterministic, chưa đo confidence có ý nghĩa, nên không thể coi kết quả này là phép kiểm trực tiếp cho giả thuyết prior-conflict; giả thuyết đó vẫn chưa được xác nhận.

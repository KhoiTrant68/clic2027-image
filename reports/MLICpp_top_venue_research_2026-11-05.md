# Hướng nghiên cứu từ MLIC++ tới một công trình đủ sức cạnh tranh ở hội nghị thị giác máy tính

**Ngày khảo sát:** 05/11/2026. **Mục tiêu của tác giả:** đóng góp khoa học nổi bật, ưu tiên chất lượng hơn kỳ nộp gần nhất; đồng thời tiếp tục cải thiện codec cho CLIC. Tài liệu viết bằng tiếng Việt, giữ thuật ngữ chuyên môn khi cần.

## Tóm tắt và quyết định

**Không nên xây luận điểm paper quanh fixed correction, học lâu hơn hoặc tìm hệ số lâu hơn.** Vòng mới giảm LPIPS 8,63% ở cùng byte, nhưng PSNR giảm và DISTS chưa phân biệt rõ; đó là baseline kỹ thuật hữu ích, chưa xác nhận cơ chế mới. Conditional basis trước đó không vượt fixed. [Kết quả vòng mới](MLICpp_fixed_improvement_2026-11-05.md), [pilot conditional/fixed](MLICpp_innovation_pilot_2026-11-05.md).

**Ưu tiên nghiên cứu:** kiểm tra khả năng *thể hiện lại bằng chứng nguồn đã có trong bitstream* khi chuyển từ decoder distortion sang decoder perceptual. Nếu tìm được một sai lệch đủ lớn, tái lập được và không do readout yếu, mới thiết kế cơ chế bảo toàn bằng chứng đó. Hướng này không đòi mặc định decoder có posterior uncertainty được calibration — điều MLIC++ hiện chưa có. Tuy nhiên, nó **chưa là phát hiện hoặc phương pháp mới được xác nhận**.

Hướng dự phòng là tối ưu encoder theo **operational rate**: chỉ có giá trị khoa học nếu tìm được sai lệch có hệ thống giữa rate surrogate và byte thật, rồi vượt corrected-cost/SGA/encode-and-reject dưới cùng wall time. Guard chống treo rANS vừa sửa không tự trở thành contribution.

**Novelty: REVISE. Execution: FEASIBILITY_PILOT_ONLY.** Chưa có hạt nhân nào đủ bằng chứng để nhận là công trình sẵn sàng nộp CVPR/ICCV/ECCV/WACV. Khảo sát này chọn phép kiểm có giá trị trước khi đầu tư full training; không gán xác suất accept hoặc hứa bảo đảm accept.

## 1. Câu hỏi, nguồn và cách khảo sát

RQ: **Cơ chế nào, ngoài tăng thời gian học/search, có thể cải thiện trade-off rate–distortion–perception bằng một đóng góp kiểm chứng được, vượt đối chứng mạnh ở byte thật, chuyển được khỏi MLIC++ và có đường thử khả thi trên một L40?**

Ba câu hỏi phụ: phần nào đã được giải quyết bởi prior art; hiện tượng nào có thể đo được bằng công cụ hiện có; kết quả nào sẽ buộc bỏ hướng thay vì thêm module?

Khảo sát ưu tiên proceedings CVF/PMLR/NeurIPS/OpenReview, arXiv bản có version, trang tác giả và repo chính thức đến ngày khảo sát. Công trình 2025–2026 được ưu tiên, bổ sung nền tảng cũ khi trực tiếp chặn novelty. Chỉ dùng nguồn sơ cấp cho kết luận kỹ thuật; không lấy AI paper notes hoặc một trang tổng hợp làm bằng chứng phương pháp.

Inclusion: learned/generative image compression, source fidelity, quantization/refinement, encoder adaptation, operational coding và đánh giá liên quan trực tiếp. Exclusion: chỉ cùng từ khóa nhưng khác bài toán; không xác minh được nguồn; kết quả tương lai sau ngày chốt. Venue chưa xác minh được được ghi riêng, không nâng preprint thành accepted paper.

Queries chính:

```text
CVPR 2026 learned image compression entropy adaptive
ICCV 2025 image compression quantization adaptation
generative image compression hallucination fidelity
image compression quantization cell semantic disambiguation
actual bitrate latent optimization quantized CDF rate bound
learned compression trust region entropy model train-test mismatch
image compression minimal pair source twins same bitstream counterfactual syndrome
image compression prediction sets set-valued semantic conformal
generative compression conformal prediction
```

Sau discovery, tra exact title và đối chiếu paper/repo/venue. Hai nhánh retrieval chia perception/fidelity và coding/adaptation; phiên chính kiểm lại các nguồn sát quyết định, hồ sơ cũ và kết quả mới. Đây là **khảo sát định hướng có giới hạn**, không systematic review hoặc PRISMA; không có tổng số hit đại diện cho toàn bộ literature. Dedup theo title/authors/version; Semantic Scholar IDs chưa resolve. Review-form note đã được tác giả skip ở lượt trước của dự án, không mở lại.

## 2. Những gì dữ liệu của dự án thực sự cho thấy

| Bằng chứng | Điều được hỗ trợ | Điều chưa được hỗ trợ |
| --- | --- | --- |
| Conditional so fixed, cùng 46 byte; 32 nguồn mới | Conditional kém về LPIPS; DISTS chưa rõ | Lợi ích của conditioning; prior-conflict |
| Fixed 5.000/160 so 1.000/40; 32 nguồn mới khác | LPIPS −8,63%, CI delta [−0,012623; −0,004123]; 160/160 fresh matches | Chất lượng tổng thể tốt hơn: PSNR −0,0602 dB, DISTS CI chứa 0 |
| Prefix train 1.000 bước | 41 mốc loss khớp chính xác; duration ablation giữ được quá trình học ban đầu | Training/search budget là một cơ chế mới |
| Glyph source CNN 255/256; MSE 251/256; Mean 250/256 | Bộ đo có năng lực trên miền tổng hợp; chỉ một lỗi tăng ròng | Hiện tượng semantic erasure lớn, phổ biến |
| S1: 0/56 exact collisions trên cặp đủ điều kiện | Không thấy collision trong screen đã chạy | Phát hiện semantic collision tự nhiên hoặc phủ định mọi collision |
| S3: 1-NN nguồn 54/128, symbols 45/128 | Readout chưa đủ năng lực chuyển font | Latent đã mất thông tin; decoder bỏ phí thông tin |
| TTO treo native rANS ở symbol cực lớn | Có lỗi vận hành tái hiện được, đã thêm guard/timeout | Mismatch rate có hệ thống hoặc một thuật toán coding mới |

Nguồn: [recognition gate](MLICpp_recognition_gate_2026-11-05.md), [CVPR screen](MLICpp_CVPR_screen_2026-11-05.md), hai báo cáo pilot phía trên. Các tập khác nhau không được gộp thành một benchmark. Tỷ lệ saturation 45,61% ở search dài là chẩn đoán hậu nghiệm, chưa chứng minh nguyên nhân hoặc giải pháp.

## 3. Prior art gần nhất và các claim cần bỏ

Phạm vi đọc được ghi rõ; khi chỉ đọc abstract/README, không đưa ra đánh giá điểm yếu phương pháp. Tóm tắt dưới đây phục vụ đối chiếu cơ chế, không xếp hạng SOTA bằng các số khác protocol.

| Nguồn xác minh | Trạng thái / phạm vi đọc | Cơ chế đã có; hệ quả đối với dự án |
| --- | --- | --- |
| Peng, T., Kwan, H. M., Zhang, F., Liu, S., & Bull, D. (2026). *PerQ: Inverse Generative Modeling for Neural Image Compression via Quantization Error Compensation*. [Paper](https://arxiv.org/html/2610.27374v1) | v1 19/10/2026; NeurIPS 2026 accepted theo [trang tác giả](https://jasminepp.github.io/), chưa đối chiếu proceedings. Đọc §3 và Appendix A/E | Flow trong quantization cell trên backbone đóng băng, gồm MLIC++. Generic bounded latent compensation không mới. Bound quanh backbone output không đồng nghĩa đúng mọi thuộc tính nguồn. |
| Kuang, H., Yang, W., Guo, Z., & Liu, J. (2025). *Cross-Granularity Online Optimization with Masked Compensated Information for Learned Image Compression*. [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Kuang_Cross-Granularity_Online_Optimization_with_Masked_Compensated_Information_for_Learned_Image_ICCV_2025_paper.pdf) | CVF proceedings; abstract/introduction/overview từ bản index | CGOO có online optimization, adaptive QP, additional fitted bitstream và selective compensation. “Truyền ít bit để sửa chi tiết” riêng lẻ đã có đối thủ rất gần. |
| Wang, K., He, L., Fu, H., Bi, H., & Li, F. (2026). *SDGIC: A Semantic Disambiguation-Guided Generative Image Compression Method for Ultra-Low Bitrates*. [v2](https://arxiv.org/html/2512.06344v2) | Preprint, v2 29/06/2026; §3.2–3.4 | Caption, compressed image và reconstruction-aware semantic residual tokens giải quyết phần ngữ nghĩa còn mơ hồ. Generic semantic side information hoặc disambiguation chưa là delta. |
| Elata, N., Michaeli, T., & Elad, M. *PSC: Posterior Sampling-Based Compression*. [v3](https://arxiv.org/html/2407.09896v3) | TMLR 2025 theo [tác giả](https://noamelata.github.io/); cơ chế, §6, Appendix A/B | Chọn phép đo từ posterior samples/SVD, truyền phép đo lượng tử. Adaptive evidence acquisition đã có; uncertainty phải có định nghĩa và khả năng đo thật. |
| Xu, T., et al. (2024). *Idempotence and Perceptual Image Compression*. [ICLR](https://openreview.net/pdf?id=Cy5v64DqEF) | Accepted paper; §3, README | Conditional generative decoding gắn với cùng code/idempotence. Giữ code không tự chứng nhận semantics đúng. |
| Vaisman, A., Ohayon, G., Manor, H., Elad, M., & Michaeli, T. (2026). *Turbo-DDCM: Fast and Flexible Zero-Shot Diffusion-Based Image Compression*. [Paper/project](https://amitvaisman.github.io/turbo_ddcm/) | ICLR 2026; abstract/project/README | Có ROI priority và distortion-controlled target PSNR theo ảnh. Phân bổ bit theo vùng quan trọng hoặc target PSNR không đủ mới. |
| Kim, S., Tregidgo, H. F. J., Jin, C., Figini, M., & Alexander, D. C. (2026). *HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration*. [CVPR](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_HalluGen_Synthesizing_Realistic_and_Controllable_Hallucinations_for_Evaluating_Image_Restoration_CVPR_2026_paper.pdf) | Paper và supplemental methods/limitations | Đã nghiên cứu hallucination vẫn measurement-consistent và công cụ đánh giá. Không suy “thuộc cùng cell ⇒ đúng nội dung”; chuyển phép đo sang codec tự nhiên vẫn phải xác minh. |
| Zhang, T., Jia, Z., Li, H., & Liu, D. (2026). *Rethinking Generative Image Compression at Extremely Low Bitrates*. [v1](https://arxiv.org/html/2609.39315v1) | Preprint 30/09/2026; đọc §1–3, protocol overview | RAE-CoD đã phân biệt recognizability/quality/source consistency, phân tích gradient conflict và align compressed condition với nguồn. Không nhận lại câu chuyện “ảnh đẹp nhưng sai nguồn” hoặc semantic alignment nói chung. |
| Jiang, W., et al. (2026). *SSCL: Adversarially Guided Image Compression via Semantic and Spectral Consistency Learning*. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38515) | Proceedings, abstract; không đánh giá method weaknesses | Semantic/frequency discriminators đã bổ sung global semantics cho perceptual codec. Một semantic loss/discriminator mới chưa đủ đóng góp. |
| Xu, L., Zhang, Y., Xu, K., He, G., & Li, Y. (2026). *Towards Unified Human Perception and Machine Understanding: Token Flow Guided Compression Framework*. [CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Towards_Unified_Human_Perception_and_Machine_Understanding_Token_Flow_Guided_CVPR_2026_paper.html) | Proceedings, abstract; không đánh giá method weaknesses | TFGC cho LVLM đọc compressed tokens trực tiếp và semantic alignment. “Latent vẫn có thông tin cho máy” hoặc đồng tối ưu human/machine không tự là novelty. |
| Chen, K., et al. (2025). *Test-Time Adaptation for Image Compression with Distribution Regularization*. [Paper](https://arxiv.org/html/2410.12191v1) | ICLR 2025; §3 | TTA-IC đã xử lý mismatch của refined latents bằng distribution regularization. Generic trust-region/distribution-preserving TTO có rủi ro trùng. |
| Yang, J., & Zhang, Y. (2026). *Conformalized Rate-Adaptive Sensing*. [v1](https://arxiv.org/html/2607.26887v1) | Preprint 29/07/2026; abstract, §1, cấu trúc method và experiments | CoRAS calibration stopping-time để chọn acquisition/compression rate với probabilistic quality target. Không nhận generic conformal rate control là mới. |
| Zecchin, M., Ganesan, U. K., Durisi, G., Popovski, P., & Simeone, O. *Prediction-Powered Communication with Distortion Guarantees*. [Paper](https://arxiv.org/abs/2509.24373) | Primary abstract; [tổ chức tác giả](https://www.eurecom.org/en/publication/8647) ghi journal 2026; không đánh giá full methods | Online conformal compression và distortion guarantees đã được nghiên cứu, thí nghiệm semantic text. Khác modality không tự tạo novelty. |
| Madden, J., Dorje, L., & Li, X. (2025). *Bitstream Collisions in Neural Image Compression via Adversarial Perturbations*. [v1](https://arxiv.org/html/2503.19817v1) | Preprint; abstract và bố cục methods | Exact collisions dưới adversarial perturbation đã có. Không đổi tên source twins để nhận phát hiện collision lần đầu. |
| Cao, H., Huang, Y., Guo, W., & Qin, Z. (2026). *Efficient Generative Image Compression with Quantized Latents Refinement*. [v1](https://arxiv.org/html/2610.29691v1) | Preprint 21/10/2026; abstract/method outline, không audit full experiments | GR-GIC là đối thủ mới cần đọc sâu nếu quay lại latent refinement; quantization collapse/refinement không phải động cơ mới tự thân. |

Các nguồn bổ sung phải giữ trong comparator map: [PICO, CVPR 2026](https://arxiv.org/html/2605.05148v1) có text-aware fidelity và adaptive quantization; [LoRA_comp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Test-Time_Fine-Tuning_of_Image_Compression_Models_for_Multi-Task_Adaptability_CVPR_2025_paper.pdf) truyền instance-specific decoder updates; [Code Editing, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/4f3820576130a8f796ddbf204c841487-Paper-Conference.pdf) đã bàn latent/entropy mismatch và quantization-step optimization.

**Tổng hợp của người khảo sát:** các claim chung về correction, uncertainty, semantic fidelity, generative decoding và tăng search đều đã đông prior art. Phần còn có thể nghiên cứu phải nằm ở một hiện tượng cụ thể, một can thiệp tách biệt và lợi ích vượt comparator gần nhất. Không tìm thấy exact-title match không đủ xác nhận khoảng trống.

## 4. Hướng ưu tiên: kiểm chứng khả năng thể hiện bằng chứng từ bitstream

### 4.1. Giả thuyết có thể bị bác bỏ

Cho $B$ là **raw base stream cố định**, $Y$ là thuộc tính nguồn có nhãn độc lập, $h(B)$ là readout từ latent đã entropy-decode và $r(G(B))$ là readout trên ảnh tái tạo. Câu hỏi là liệu có một tập hợp có tần suất đáng kể mà readout từ $B$ dự đoán đúng $Y$, nhưng decoder perceptual thể hiện thuộc tính sai, dù decoder distortion với **đúng $B$ đó** thể hiện tốt hơn hay không.

Đây là giả thuyết **predictive accessibility trong lớp readout đã thử**, chưa phải chứng minh information-theoretic rằng decoder làm mất thông tin. Probe yếu không chứng minh thông tin vắng mặt; probe mạnh hơn image reader cũng có thể chỉ phản ánh chênh lệch năng lực hoặc shortcut. Một kiểm nghiệm phải giữ đầy đủ các khả năng giải thích đó.

Nếu dùng correction packet, $B$ đầy đủ phải gồm base + coefficients cho cả probe và decoder. Không coi thông tin từ 46 byte nguồn bổ sung là bằng chứng vốn có trong raw base. Diagnostic đầu nên bỏ packet để tránh confound này.

### 4.2. Cơ chế ứng viên sau khi hiện tượng sống sót

**AI đề xuất, chưa xác nhận novelty:** dùng các thuộc tính đủ tin cậy suy từ bitstream làm ràng buộc cho perceptual decoder, thay vì chỉ ép gần một vector semantic trung bình. Có thể khảo sát prediction set $C(B)$: khi tập chỉ có một nhãn, decoder cố giữ thuộc tính ấy; khi rỗng hoặc nhiều nhãn, một chính sách cụ thể là trả decoder distortion nền. Phải báo đầy đủ tỷ lệ fallback và chất lượng của nó; không nói một ảnh “không quyết định” nếu chưa định nghĩa thao tác hiển thị.

Delta cần xác minh là **calibrated evidence-conditioned reconstruction**, gồm hiệu quả của ràng buộc và chi phí thực, không phải chỉ gắn conformal wrapper vào semantic loss. Nếu một loss semantic, confidence gate hoặc blend đơn giản tạo cùng frontier, bỏ phần cơ chế phức tạp.

So với RAE-CoD/SSCL: phải tách supervision từ nguồn với tín hiệu dùng được ở decoder từ bitstream. So với PerQ/IPIC: phân biệt codec/cell consistency với thuộc tính đo được. So với TFGC: không chỉ đánh giá máy đọc token, mà kiểm ảnh tái tạo thể hiện đúng nội dung. So với CoRAS: đây là can thiệp reconstruction, không phải stopping rule tăng rate. Các khác biệt này mới là **nghĩa vụ kiểm chứng**, chưa phải kết luận “prior work không làm”.

Đối chứng rất mạnh và rẻ cần có: **truyền nhãn thuộc tính trực tiếp**. Bốn lớp chỉ cần hai bit payload nếu vị trí/nhiệm vụ đã biết; metadata và điều kiện decoder phải tính đủ. Một zero-extra-bit method không có giá trị lớn chỉ vì tiết kiệm hai bit nhưng thêm model/compute đáng kể.

### 4.3. Có thể chứng minh gì, và không chứng minh gì?

Conformal prediction có nền tảng chuẩn về prediction-set coverage dưới các giả thiết phù hợp, đặc biệt exchangeability và calibration tách biệt. [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511). **Không coi công cụ đó hoặc hệ quả dưới đây là theorem mới.**

Giả sử $\Pr\{Y\notin C(B)\}\leq\alpha$. Đặt $A=\{|C(B)|=1\}$ và xác suất constraint không được giữ trên $A$ là $\varepsilon$. Khi $A$ xảy ra và cả coverage lẫn constraint đều đúng, image readout bắt buộc bằng $Y$. Bởi vậy:

$$
\Pr\{A\cap[r(G(B))\ne Y]\}\leq\alpha+\varepsilon.
$$

Nếu $\Pr(A)>0$, chỉ suy ra chặn conditional $\min\{1,(\alpha+\varepsilon)/\Pr(A)\}$, **không phải $\alpha$**. $\varepsilon$ là xác suất thực, không được thay bằng tỷ lệ mẫu mà bỏ qua bất định. Nếu singleton coverage thấp, bound có thể vô dụng. Chọn tùy ý một nhãn từ tập nhiều phần tử không có bảo đảm đúng nhãn nguồn.

Bảo đảm này nói về **nhãn và image readout được định nghĩa**, không phải mọi nội dung hoặc khả năng đọc của người xem. Readout bị decoder “đánh lừa” vẫn có thể vượt constraint trong khi người xem đọc sai. Domain shift, đổi encoder/rate/TTO sau calibration, nhiều thuộc tính và lỗi validator đều cần xử lý riêng. Không dùng entropy scale làm posterior confidence, không nhận semantic guarantee toàn ảnh từ một label-set inequality.

## 5. Phép thử tiếp theo đáng đầu tư

Đây là **diagnostic pilot**, chưa phải lệnh full train cho model mới. Trần đề xuất: **hai GPU-giờ trên một L40**, không API trả phí; dừng sớm nếu measurement gate thất bại. Owner là phiên nghiên cứu tiếp theo của dự án.

1. Khóa một nhiệm vụ thuộc tính hữu hạn có ground truth từ generator/annotation và các nhóm nuisance độc lập. Mức dự kiến 512 family test mới, đủ train/development riêng; chốt inventory và denominator trước khi mở output. Không tái dùng các font/layout/test cũ để tìm harmful cases. Nội dung sinh chỉ là kiểm năng lực ban đầu, chưa suy sang ảnh tự nhiên.
2. Kiểm năng lực readout ảnh trên nguồn giữ riêng, tối thiểu 95% theo ngưỡng vận hành của dự án. Kiểm bằng validator khác kiến trúc hoặc audit nội dung trực tiếp. Còn phải xác minh readout trên ảnh tái tạo development, không chỉ ảnh nguồn; nếu chưa có xác nhận nội dung độc lập, chỉ báo endpoint của công cụ, chưa nhận khả năng đọc của người xem. Split theo nguồn/family/nuisance; không đưa filename, nhãn hoặc cache metadata vào latent probe. Năng lực readout latent và image phải được kiểm soát, không so một probe lớn với một reader yếu rồi quy lỗi cho decoder.
3. Cố định raw MLIC++ q2 stream; so MSE và full-synthesis perceptual đã có. Primary là chênh lệch lỗi thuộc tính ghép cặp; probe từ entropy-decoded representation là chẩn đoán hỗ trợ. **Mức gap tối thiểu đề xuất 5 điểm phần trăm**, CI theo family có chiều tăng lỗi, cùng sự xác nhận của validator thứ hai. Đây là tiêu chí lựa chọn hướng của dự án, không phải tiêu chuẩn conference hoặc power calculation hoàn chỉnh.
4. Nếu không đạt gate, gap nhỏ/không tái lập hoặc biến mất sau nuisance/readout controls: **bỏ claim về hiện tượng này trong phạm vi đã thử**. Không tăng độ khó cho đến khi có lỗi rồi báo tần suất trên tập đã chọn. Khi tỷ lệ lỗi gần 1/256, cần tính lại sample size nếu muốn nghiên cứu hiệu ứng nhỏ; không mặc định 512 family đủ lực cho mọi hiệu ứng.
5. Chỉ nếu hiện tượng sống sót mới mở can thiệp decoder trên development: semantic loss đơn giản, blend/text-aware control, ràng buộc evidence có hoặc không calibration. Test mới, hiệu quả ngoài loss và chi phí thực là điều kiện tiếp tục. Benchmark trên backbone thứ hai chỉ được triển khai sau khi xác nhận đường code/checkpoint và timing.

Calibration và test exchangeable trong cùng miền phải được tách khỏi kiểm font/domain mới. OOD coverage là kết quả thực nghiệm, không được hưởng bảo đảm ID mặc định. Chưa thu human judgments; nếu muốn nhận human preference/readability, cần thiết kế đánh giá mù và quy trình phù hợp trước thu thập, không suy từ proxy.

## 6. Hướng dự phòng: operational-rate optimization

Giả thuyết cần kiểm trước là likelihood surrogate sai lệch có hệ thống so integer CDF/escape/bypass/framing, làm encoder dùng budget kém ngay cả khi đã sửa lỗi optimizer. Lỗi symbol cực lớn đơn lẻ chưa chứng minh điều đó.

[EMTC, Adler & Tang, ICML 2026](https://proceedings.mlr.press/v306/adler26b.html) nghiên cứu bounded prediction mismatch giữa encoder/decoder; cần phân biệt với rate proxy trong cùng encoder. [Finite-precision rANS analysis](https://arxiv.org/html/2408.07322v1) và [rANS implementation](https://raw.githubusercontent.com/rygorous/ryg_rans/master/rans_byte.h) đã có công cụ tính code length. Đếm byte chính xác, upper bound chuẩn hoặc thêm timeout không tự mới.

Nếu khảo sát hướng này, contrast quyết định phải là optimizer mới so corrected likelihood/tail cost, SGA hoặc line search và encode-and-reject, **cùng wall time, decoder, source và hard total byte**. Acceptance là quality tốt hơn hoặc đạt budget hiệu quả hơn trên một tập không chọn theo failure, không chỉ giảm số crash. Hiện chưa có bằng chứng cho contrast đó. Mở pilot riêng tối đa hai GPU-giờ chỉ sau khi khóa falsifier này; không dùng thất bại của hướng ưu tiên để tự động cấp một câu chuyện thành công cho hướng dự phòng.

## 7. Chọn venue theo đóng góp thực tế

**Top 1 CLIC không phải điều kiện cần để có paper mạnh.** ICCV/ECCV hướng dẫn reviewer cân nhắc novelty và impact, không bác chỉ vì chưa vượt SOTA. Điều đó cũng không có nghĩa một gain nhỏ trên baseline cũ sẽ đủ. [ICCV 2025](https://iccv.thecvf.com/Conferences/2025/ReviewerGuidelines), [ECCV 2026](https://eccv.ecva.net/Conferences/2026/ReviewerGuide).

| Thành quả thực sự đạt được | Cách định vị hợp lý |
| --- | --- |
| Cơ chế mới, giải thích được, vượt controls và chuyển backbone/rate | Paper methods cho CVPR/ICCV/ECCV; eligibility tùy kỳ gọi bài |
| Một benchmark/protocol có measurement validity, insight mới, release hữu dụng | Xem contribution phù hợp về evaluation/datasets; không cần ép thành kiến trúc mới |
| Codec/systems có cải thiện thực về deployment và đánh giá đầy đủ | Hướng applied/systems nếu đúng trọng tâm và đủ khác biệt |
| Chỉ tăng train/search, giảm LPIPS trên 32 crop | Tiếp tục làm baseline, chưa định vị như main conference paper |

[ECCV 2026 contribution types](https://eccv.ecva.net/Conferences/2026/ReviewerContributionTypes) công nhận nhiều dạng đóng góp. [WACV 2027 reviewer guidelines](https://wacv.thecvf.com/Conferences/2027/ReviewerGuidelines) phân biệt application, algorithms và evaluations/datasets; không diễn giải WACV như một nơi tự động nhận công trình thiếu novelty. [CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) yêu cầu nghiên cứu nguyên gốc, chất lượng cao. Các policy dẫn đúng phiên bản đã đọc, không mặc định giữ nguyên ở kỳ sau.

Mức bằng chứng **đề xuất cho dự án**, không phải checklist bắt buộc của conference: nhiều rate, ít nhất ba seed cho experiment chính, MLIC++ và một backbone hiện đại ngoài họ; actual bytes/metadata, full-resolution và runtime decoder; strong simple baselines; endpoint ngoài loss; test khóa trước; human evaluation nếu nhận preference. Không gộp published numbers khác protocol thành bảng SOTA của dự án. [DCAE official repo](https://github.com/CVL-UESTC/DCAE) có đường train/eval/bitstream/checkpoint là ứng viên kiểm access; chưa chạy trên server nên chưa xác nhận tái lập được. PerQ/PICO public page không đồng nghĩa có weights runnable.

Không ép kịp deadline gần bằng cách viết kết quả chưa có. Hạt nhân paper chỉ được chốt khi hiện tượng, comparator delta và can thiệp cùng đứng vững; ưu tiên kỳ sau đã được tác giả chấp nhận.

## 8. Phản biện, giới hạn và trạng thái cuối

**Checkpoint 1:** PASS cho phạm vi khảo sát, không phải cho một cơ chế. **Checkpoint 2:** REVISE; đã sửa framing từ “decoder làm mất information” thành predictive accessibility, tách raw stream khỏi side bits, bỏ source-twin collision làm hạt nhân, giới hạn calibration claim và thêm cheap explicit-label control. Diagnostic pilot được ủng hộ có điều kiện measurement/denominator/kill threshold; full training chưa được thông qua.

**Checkpoint 3:** PASS cho lập luận và framing của báo cáo, không phải chứng nhận novelty hoặc khả năng accept. Phản biện cuối nhấn mạnh readout cần kiểm trên reconstruction, 512 family/hai GPU-giờ chưa là power/timing certificate, và calibration không giải quyết việc decoder đánh lừa reader. Editorial/integrity check của phiên chính giữ nguyên kết quả âm, đối chiếu link, phân biệt proposed/observed, và không phát hiện vấn đề cần dừng khảo sát. Chưa có peer review hoặc tái lập độc lập bởi người khác.

Hồ sơ hiện có **không chứng minh semantic erasure lớn**, posterior confidence, novelty hoặc khả năng accept. Khảo sát chưa bao phủ mọi patent/journal/proceedings cuối 2026; một số PDF/project page bị 403 hoặc không truy cập được. Phạm vi đọc không đồng đều, không audit toàn bộ proof hoặc reproduction của từng tác giả. Việc chia vai AI không tạo hai nguồn bằng chứng thống kê độc lập.

AI hỗ trợ retrieval, kiểm nguồn, phản biện, lập phương án và code/phân tích pilot; tác giả phải kiểm, tái lập và chịu trách nhiệm về claim. Không gửi bài, liên hệ người khác, thu human data hoặc khởi chạy full training từ tài liệu này. Đầu ra cụ thể của lượt này là baseline đã cải thiện có bằng chứng, bản đồ đối thủ gần và **một diagnostic có thể loại hướng nghiên cứu** — chưa phải cam kết một model sẽ được accept.

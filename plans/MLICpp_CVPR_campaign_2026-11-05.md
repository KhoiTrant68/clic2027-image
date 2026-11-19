# Chiến dịch nghiên cứu MLIC++ hướng tới CVPR

Ngày cập nhật: 05/11/2026; các job sàng lọc bắt đầu tối 04/11. Người dùng chọn **ưu tiên công trình đủ mạnh, chấp nhận kỳ CVPR sau nếu cần**. Không ép kết quả sơ bộ thành một bài để kịp deadline. Đây là kế hoạch thực nghiệm và các giả thuyết do AI đề xuất, chưa phải chứng nhận novelty hoặc khả năng được nhận bài.

**Cập nhật thực nghiệm:** [S1, S2 và S3 đã hoàn tất](../reports/MLICpp_CVPR_screen_2026-11-05.md). Không thấy exact collision ở 56 cặp nguồn đủ điều kiện; perceptual adapter làm accuracy OCR q2 giảm 6,90 điểm phần trăm. Hard moment projection có lợi nhỏ nhưng 5/6 control chưa khớp LPIPS trên validation. Readout tách font yếu ngay trên ảnh nguồn, nên chưa phân giải được thông tin mất ở encoder hay decoder. Chưa có hướng nào đủ bằng chứng để nâng lên full run; ưu tiên sửa phép đo trên dữ liệu giữ riêng mới. Các nhánh bên dưới là giả thuyết có điều kiện, không phải kế hoạch huấn luyện lớn đã chốt.

**Bước kiểm phép đo đã hoàn tất:** [CNN trên họ font mới](../reports/MLICpp_recognition_gate_2026-11-05.md) đạt 255/256 trên nguồn, 251/256 MSE và 250/256 Mean. Tesseract chỉ đạt 202/256 trên nguồn nhưng báo giảm lớn hơn giữa hai decoder. Chưa đủ bằng chứng để chọn tổn hại nhận dạng làm cơ chế trung tâm; cần dữ liệu khó hơn và bộ đo độc lập đủ năng lực, không mở rộng Risk head hoặc suy luận mất thông tin từ latent 1-NN yếu.

**Đổi trọng tâm theo yêu cầu cạnh tranh CLIC:** [kế hoạch perceptual contender](MLICpp_perceptual_contender_2026-11-05.md) mở học toàn synthesis và đối chứng conditional GAN trước. Không tiếp tục lấy chữ làm luận điểm chính. Hướng mới cần kiểm là innovation có điều kiện theo base latent, với ordinary residual và base-more-bits làm đối chứng bắt buộc; tính mới chưa xác nhận.

## Điều đã biết và quyết định thay đổi

[Pilot Risk head](../reports/MLICpp_adapter_pilot_2026-11-04.md) chưa thắng đối chứng CVaR; dừng mở rộng kiến trúc đó. Baseline perceptual nhỏ có ích để nghiên cứu, nhưng giảm LPIPS sau fine-tune không đủ làm đóng góp. Các đợt sau phải xét mã entropy thực, nội dung có ground truth và tập ảnh ngoài những ảnh đã dùng chọn ý tưởng.

CVPR 2027 hiện đăng ký bài ngày 10/11/2026 AoE, nộp ngày 16/11 AoE, tương ứng 18:59 ngày 11/11 và 17/11 giờ Việt Nam. Người dùng không chọn chạy theo mốc này bằng mọi giá. Không có thao tác đăng ký/nộp bài trong chiến dịch. [Nguồn chính thức](https://cvpr.thecvf.com/Conferences/2027/Dates).

## Câu hỏi ưu tiên: thông tin mất ở đâu?

Một lỗi chữ sau nén có thể có ít nhất hai nguyên nhân khác nhau: thông tin phân biệt đã bị encoder/lượng tử hóa bỏ đi; hoặc thông tin vẫn tồn tại trong mã nhưng synthesis decoder không tái tạo nó đúng. Hai nguyên nhân cần can thiệp khác nhau. **Distinct symbols không đủ chứng minh còn giữ đầy đủ ý nghĩa; OCR thất bại cũng không đủ chứng minh thông tin đã mất.** Chỉ exact code collision giữa hai nguồn khác nhãn mới cho kết luận bất khả phân biệt trong family cụ thể, và kết luận đó là nguyên lý chuẩn, không phải theorem mới của dự án.

## Ma trận sàng lọc đầu tiên

| ID | Giả thuyết / yếu tố thay đổi | Đối chứng cố định | Kết quả cần đo | Điều kiện bác bỏ |
| --- | --- | --- | --- | --- |
| S1 | Một thay đổi một ký tự có thể bị đồng nhất thành cùng mã | q1/q2 chính thức; cùng font/background/layout; thêm dịch vị trí 1 pixel nhưng cùng nhãn | Full stream equality, actual y/z symbol differences, OCR nguồn và tái tạo, byte thực, fresh parity | Nếu exact collisions gần như vắng mặt trong family legible đã thử, không dùng “same-code semantic aliasing” làm câu chuyện chính cho lỗi decoder |
| S2 | Ràng buộc cứng giữ moment có lợi hơn chỉ giảm perceptual enhancement | Cùng base/refiner, random signed moment cùng rank, scalar blending chọn trên validation với LPIPS tương đương | Pixel invariant sau uint8, LPIPS/DISTS, RGB finite-difference error, cùng stream/bpp; CI theo ảnh | Nếu không hơn scalar control ở matched perception hoặc chỉ khóa sai số của base, không claim cơ chế bảo toàn cấu trúc mới |
| S3 | Nếu mã vẫn khác, một readout có thể tìm lại ký tự mà synthesis bỏ sót | Pixel readout và actual-symbol readout; tách font/family; chọn hyperparameter trên font train | Accuracy held-out font, decoder OCR, chance controls | Probe thất bại không chứng minh mất thông tin; chỉ đi tiếp nếu gap được kiểm độc lập và không do leakage |

S1 và S2 là **diagnostics/known-method controls**, không được đổi tên thành hai mô hình mới. S3 phụ thuộc S1 và chưa là một phương pháp compression. Kết quả dùng để chọn can thiệp thực sự, thay vì tiếp tục thêm loss theo phỏng đoán.

## Thiết kế cụ thể và tài nguyên

- **S1:** 64 family, mỗi family có hai minimal semantic variants và một nuisance variant; tổng 192 ảnh 256². Hai font, bốn cỡ chữ, hai mức tương phản, hai background, hai cặp glyph. Dùng fixed character cells để chỉ một glyph thay đổi. Tesseract 5.5.3 và font trong môi trường riêng. Báo riêng subset nguồn OCR đọc đúng và toàn bộ dữ liệu; OCR-correct là bộ lọc vận hành, không phải xác nhận legibility từ con người.
- Encode q1/q2: tổng 384 stream, lấy đúng symbols qua `quantize(..., 'symbols', means)` của đường `compress`, không thay bằng `round(y)` giả. So MSE và adapter Mean ở q2. Giải mã tiến trình mới để kiểm pixel. Raw data/streams nằm ngoài Git.
- **S2:** 32 ảnh validation cũ để chọn duy nhất các hệ số blend toàn cục; **64 ảnh test mới**, không trùng 224 ảnh pilot đầu. Crop mới seed 2037; base/refiner cùng q2 và cùng actual stream. Hard projection giữ tổng RGB theo block 8/16/32, có box constraints và balanced integer rounding để giữ equality trên uint8. Control random signed moment có cùng rank.
- Scalar blend được quét và refine trên validation, chọn control có LPIPS không tệ hơn projection quá 0,2% nếu có; nếu không có phải đánh dấu unmatched. Giữ hệ số trước test; không chọn riêng theo ảnh test. Mọi nhánh đều phải báo actual LPIPS trên holdout, không mặc định validation match suy ra test match.
- GPU: dùng hai L40 cho hai screen độc lập, guard VRAM có sẵn, không can thiệp job khác. Đợt sàng lọc đầu giới hạn kế hoạch dưới 2 GPU-giờ tổng, gồm encode/verify/evaluate; không API trả phí. Ước tính lưu trữ dưới vài GB, không train diffusion foundation model. Đây là ngân sách thử, không phải mức chi đã đo.
- Monitoring: JSON/JSONL/log cùng kiểu repo, trạng thái từng stage và hash mã nguồn. Nếu stage lỗi giữ nguyên bằng chứng và không ghi đè run.

## Hướng can thiệp tiếp theo, có điều kiện

**Ứng viên A — counterfactual quantizer response.** Nếu S1/S3 cho bằng chứng đúng loại lỗi ở encoder, thử objective dùng geometry của quyết định lượng tử cho cặp một-glyph/thay đổi junction, nhằm giữ khác biệt có nghĩa với chi phí rate tối thiểu thực nghiệm. Không nhận “semantic contrastive learning” là mới. Cần so năm recipe cùng data/compute: RD/perceptual; OCR/semantic reconstruction loss; continuous latent contrastive; quantized-space contrastive; boundary-aware candidate. Surrogate phải khớp mean-centering/context của actual symbols. Không ép font/ánh sáng khác nhau có full code giống nhau.

**Ứng viên B — decoder response to preserved evidence.** Nếu symbols/readout vẫn phân biệt tốt nhưng output glyph hỏng, xem xét học đáp ứng tái tạo đối với minimal semantic changes, thay vì buộc thêm symbol separation. Phải thắng text-mask weighted loss và cùng pair-data nhưng independent reconstruction loss. Lợi ích cần chuyển sang font/layout và scene text chưa thấy, không chỉ template tổng hợp.

**Ứng viên C — source-specific evidence, chỉ sau kiểm tra thiếu thông tin.** Nếu decoder-only không thể khôi phục, thử representation/evidence thật và tính đủ byte. So candidate-index, extra base rate, conventional residual và shuffled evidence ở cùng tổng budget. Đây là nhánh có prior art sát; chỉ theo nếu tìm được cơ chế khác biệt và empirical headroom, không coi “thêm vài bit sửa chữ” là đóng góp.

Đây là các ngã rẽ nghiên cứu, không phải ba kiến trúc được mặc định huấn luyện hết. Chọn theo dữ liệu để tránh một cuộc sweep module không có giả thuyết.

## Các tiền lệ phải đối chiếu

- Nullspace/data consistency đã có trong [Deep Null Space Learning](https://arxiv.org/abs/1806.06137), [DDNM](https://arxiv.org/abs/2212.00490) và [Idempotent Learned Image Compression with Right-Inverse, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html). S2 không được nhận generic hard projection là mới.
- [Idempotence and Perceptual Image Compression, ICLR 2024](https://arxiv.org/html/2401.08920v2) và [PerQ](https://arxiv.org/html/2610.27374v1) đã ràng buộc perceptual decoding bằng thông tin codec/quantization. Bảo toàn invariant của base không đồng nghĩa giữ đúng nguồn.
- [Bitstream Collisions in Neural Image Compression via Adversarial Perturbations, 2025](https://arxiv.org/html/2503.19817v1) đã có exact collisions qua attack. S1 không nhận phát hiện collisions lần đầu; hiện không dùng adversarial optimization để tạo số đẹp.
- [SDGIC v2, 29/06/2026](https://arxiv.org/html/2512.06344v2), [AGDL, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Attention-Guided_Image_Compression_by_Deep_Reconstruction_of_Compressive_Sensed_Saliency_CVPR_2021_paper.html) và [PICD, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_PICD_Versatile_Perceptual_Image_Compression_with_Diffusion_Rendering_CVPR_2025_paper.html) đã có source-dependent semantic/structural evidence.
- [Rate-Distortion-Cognition Controllable Versatile Neural Image Compression, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07338.pdf), [Masked Sparse Visual Representation, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Jiang_Neural_Image_Compression_Using_Masked_Sparse_Visual_Representation_WACV_2024_paper.pdf), [MeCoQ, AAAI 2022](https://arxiv.org/abs/2109.05205) và [ImageFolder, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/82519a9d318f39096ea507615c61c340-Paper-Conference.pdf) chặn claim chung về semantic/quantized contrastive learning. Encoder boundary-aware phải hơn các control này trong phạm vi thích hợp.
- [GR-GIC, preprint 21/10/2026](https://arxiv.org/html/2610.29691v1) đã nêu quantization-cell collapse và refinement sau lượng tử. Không nhận động cơ “quantization aliasing” là phát hiện mới.

Các nguồn được đọc với phạm vi khác nhau qua primary abstracts/method sections; đây không phải kiểm định đầy đủ mọi proof/experiment. Không thấy trùng chính xác qua tìm kiếm không chứng minh chưa có công trình tương tự. Fano, data processing, contrastive loss, projection và balanced rounding không tự là đóng góp của dự án.

## Tiêu chuẩn để nâng từ pilot lên ứng viên paper

1. Có một cơ chế còn đứng vững sau đối chứng mạnh nhất, chứ không chỉ vượt MSE legacy.
2. Giữ lợi ích ở **matched total actual bytes**, không chỉ cùng λ; đo perception lẫn nội dung/recognition bằng công cụ ngoài loss.
3. Có nhiều rate, ít nhất ba seed ở thí nghiệm chính, full-resolution, font/layout/scene text chưa thấy và nhóm ảnh không có chữ.
4. Baseline hiện đại trong họ MLICv2 và một backbone ngoài họ; closest methods theo claim. Tách cost pretraining/fine-tune và đo decoder thực trên L4.
5. Khoảng tin cậy theo ảnh/family; khoá test/hyperparameter; không tự chuyển metric sau khi xem kết quả. Human evaluation trước khi claim human preference.
6. Nếu continuous/quantized contrastive hoặc text-aware loss đơn giản đạt cùng frontier, loại phần cơ chế không đóng góp. Không giữ một framework nhiều module chỉ vì cần tên paper.

AI hỗ trợ tìm tài liệu, thiết kế, coding và phân tích. Các nhánh audit không phải reviewer độc lập thống kê. Chưa có claim SOTA, novelty hay xác suất được nhận CVPR.

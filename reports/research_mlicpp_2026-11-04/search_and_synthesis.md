# Nhật ký tìm kiếm và tổng hợp

Ngày: 04/11/2026. User xác nhận phạm vi bằng “cứ làm đi”, sau đó trả lời “skip và tiếp tục” cho review-form note. Không mở lại checkpoint đó. Phạm vi được chốt trong [RQ Brief](../../plans/MLICpp_research_scope_2026-11-04.md).

## Chiến lược tìm kiếm, Phase 2

Khảo sát có mục tiêu, không tuyên bố PRISMA/systematic hoặc độ bao phủ toàn bộ. Ưu tiên 2024–2026 đến ngày 04/11, bổ sung 2020–2023 để kiểm prior art của selective decoding và refinement. Nguồn: web search, arXiv, CVF, PMLR, proceedings NeurIPS, publisher, repo/trang cơ quan tác giả và CLIC chính thức. Aggregator/Reddit chỉ là đầu mối, không làm bằng chứng kết luận.

Nhận tài liệu về learned image compression và các cơ chế perception/content preservation, entropy efficiency, cross-platform decoding. Nguồn abstract-only được giữ với read scope giới hạn. Loại model-weight compression, video không liên hệ trực tiếp, trang bán hàng trùng tên và nội dung không xác minh được. Không loại công trình chỉ vì phủ định tính mới dự kiến. Không bịa số lượng hits hay screening flow từ kết quả web giới hạn.

Các truy vấn đã thực hiện, gồm tìm exact-title sau discovery:

```text
learned image compression 2026 perceptual compression MLIC++ structure preservation
learned image compression 2025 2026 content adaptive perceptual diffusion hallucination
MLIC++ 2025 2026 successor entropy model image compression
site:arxiv.org image compression 2026 perceptual structure fidelity uncertainty residual
site:openaccess.thecvf.com 2026 learned image compression perceptual
site:arxiv.org "image compression" "Oct" "2026" perceptual
site:arxiv.org "image compression" "Sep" "2026" learned
"image compression" "2026" "uncertainty" "calibration"
"generative image compression" "residual" "selective"
"perceptual image compression" "correction" "2026"
"image compression" "structural" "side information" diffusion 2025 2026
"image compression" "counterfactual"
"image compression" "CVaR"
"perceptual compression" "risk" "tail"
"learned image compression" "worst-case" perceptual
"image compression" "semantic risk"
"Attention-Guided Image Compression by Deep Reconstruction" CVPR 2021
"ProChroma" compression
"MLICv2"
"Efficient Mismatch-Tolerant Coding"
"learned image compression" cross-platform 2025 2026
"image compression" "perceptual" "adapter"
"image compression" "uncertainty" "posterior"
CLIC 2027 image compression tasks GPU rules
```

Tách hai nhánh retrieval/verification: perception và entropy/deployment. Đây là phân công cùng công cụ/mô hình, không phải hai quy trình sai số độc lập. Source catalog hợp nhất được lưu trong [sources.md](sources.md).

## Tổng hợp theo chủ đề, Phase 3

1. **Nâng perception trên baseline RD đã là lĩnh vực đông nghiên cứu.** PerQ, OneDC và CADC làm suy yếu claim “thêm generative decoder” nói chung. PerQ đặc biệt gần do đã thử MLIC++.
2. **Bảo toàn nội dung cũng không phải khoảng trống chung.** Selective Detail Decoding, AGDL, ConHa và PICO bao phủ routing, structural refinement, content-dependent preference và text-aware training. Ý tưởng “gặp chữ thì giảm GAN” không đủ mới.
3. **Context hoặc latent adaptation phải so hậu duệ trực tiếp.** MLICv2, HPCM và dictionary entropy là đối chứng; chỉ thắng legacy MSE có thể phản ánh baseline cũ hoặc objective khác.
4. **Counterfactual allocation không phải lối thoát novelty tự động.** GCR-C và ACV-Gate đã có fixed-budget evaluation và amortization. Sparse correction chỉ có ý nghĩa nếu chứng minh cơ chế, lợi ích và khác biệt với refinement cũ.
5. **Khoảng khảo sát cần tiếp tục kiểm:** cách học synthesis của MLIC++ để đạt điểm cảm nhận tốt dưới ràng buộc đuôi phân phối lỗi cấu trúc, thay vì chỉ giảm metric trung bình. Đây là ứng viên do AI đề xuất; tìm kiếm không chứng minh chưa ai làm. CVaR là công cụ chuẩn, không tự là đóng góp.
6. **Hai đích đánh giá không đồng nhất.** CLIC human preference và eligibility runtime là đích thi; câu hỏi nghiên cứu cần đối chứng/cơ chế/khả năng khái quát. Whitepaper CLIC và conference paper khác nhau theo S22.

Đây là diễn giải tổng hợp của người khảo sát từ nguồn có link trong catalog. Không gộp các con số BD-rate khác dataset/metric/hardware thành bảng SOTA thống nhất. Không lấy số báo cáo của tác giả làm số tái lập của dự án.

## Devil's Advocate checkpoint 2

**REVISE, đã xử lý trước khi soạn đề xuất:** ý tưởng sơ bộ sparse structural tokens + counterfactual selector trùng nhiều phần với AGDL/GCR-C/ACV-Gate. Không giữ như đóng góp chính đã xác nhận. Chuyển sang câu hỏi kiểm soát tail risk với decoder adaptation; đưa mask, quantization-cell compensation và generic sparse residual vào đối chứng. Không tuyên bố khoảng trống chắc chắn.

**PASS để lập kế hoạch pilot, không phải chứng nhận novelty:** chưa có kết quả thực nghiệm cho mô hình mới; kết luận được giới hạn ở ưu tiên khảo sát và tiêu chí bác bỏ. Nếu baseline đơn giản đạt cùng frontier, dừng claim model contribution thay vì thêm module.

## Giới hạn độ phủ

Nguồn tập trung vào CVPR/ICCV/ICML và arXiv năm 2025–2026; có bổ sung NeurIPS, ECCV, journal và CLIC workshop cũ để chống thiên lệch thời gian/venue. Chưa rà hết các journal, bằng sáng chế, mọi proceedings cuối 2026 hoặc nguồn ngoài tiếng Anh. Không dùng số citation để xếp giá trị ý tưởng. Semantic Scholar resolution không khả dụng, một số PDF bị 403. Tài liệu unavailable là đầu mối chưa xác minh, không phải chứng cứ hỗ trợ kết luận.

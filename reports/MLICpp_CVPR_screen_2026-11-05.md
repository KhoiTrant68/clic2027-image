# MLIC++ hướng tới CVPR: kết quả sàng lọc ngày 05/11/2026

**Ưu tiên đã chốt: xây dựng công trình đủ mạnh, chấp nhận kỳ CVPR sau nếu cần.** Ba phép thử dưới đây đã chạy xong. Chưa có cơ chế nào đủ bằng chứng để gọi là đóng góp mới cho CVPR. Kết quả giúp loại giả thuyết yếu và xác định phép đo cần sửa trước khi đầu tư huấn luyện lớn.

**Thực nghiệm tiếp nối:** [reader CNN đã kiểm trên nguồn và font mới](MLICpp_recognition_gate_2026-11-05.md) đạt 255/256 nguồn, 251/256 MSE, 250/256 Mean; Tesseract có năng lực nguồn thấp hơn và báo mức giảm lớn hơn. Vì vậy, mức giảm OCR trong screen dưới đây vẫn là kết quả đo hợp lệ theo công cụ đó, nhưng chưa đủ làm bằng chứng về tổn hại nội dung hay khả năng đọc của người xem.

Đọc cùng [kế hoạch nghiên cứu](../plans/MLICpp_CVPR_campaign_2026-11-05.md), [pilot adapter trước đó](MLICpp_adapter_pilot_2026-11-04.md) và [khảo sát tài liệu](MLICpp_research_proposal_2026-11-04.md). Đây là thí nghiệm crop và dữ liệu tổng hợp; không bổ sung các con số này vào bảng 30 ảnh CLIC như thể cùng điều kiện đánh giá.

## 1. Cấu hình và bằng chứng

| Mục | Cấu hình đã chạy |
| --- | --- |
| Codec | MLIC++ **legacy**, upstream `4aa9d2a512eb9db382172058e6c278fb0d376e17`; không phải MLICv2 |
| Checkpoint | MSE q1 và q2 cho ký tự; q2 cho ảnh DIV2K mới |
| Nhánh perceptual | Adapter Mean 34.113 tham số, checkpoint cuối của pilot ngày 04/11; không huấn luyện lại trên dữ liệu sàng lọc |
| Runtime | PyTorch 2.0.1+cu118, CompressAI 1.2.0b3; FP32, deterministic, TF32 tắt; cuDNN v7 và NVRTC như pilot |
| Tài nguyên | S1 trên L40 GPU 1, S2 trên L40 GPU 0; guard VRAM; S3 chỉ dùng CPU |
| Bitrate | Byte của stream thực, gồm container đang dùng; không báo entropy ước lượng thay cho kích thước tệp |
| Kiểm giải mã | **1.472/1.472 đầu ra** khớp hash pixel khi giải mã ở tiến trình mới, không đọc ảnh nguồn |
| Phân tích | Bootstrap ghép cặp 10.000 lần, seed 2037; theo ảnh ở S2 và theo family ở so sánh OCR q2 |

Bằng chứng nhỏ được lưu tại [thư mục kết quả](../results/2026-11-05_mlicpp_cvpr_screen/analysis.json), gồm số liệu từng ảnh, cấu hình, log, hash mã nguồn và bản sao mã chạy. Stream, symbols và ảnh lớn giữ ở server, dưới `/home/jovyan/clic2027-artifacts/mlicpp/cvpr20261104/`. Khoảng tin cậy là **thăm dò, chưa hiệu chỉnh cho nhiều phép so sánh**, không thay thế nghiên cứu xác nhận trên test mới.

## 2. S1 — Hai ký tự khác nhau có bị mã hóa thành cùng mã không?

Tạo 64 family: hai font, bốn cỡ chữ, hai mức tương phản, hai loại nền và hai cặp ký tự `6/8`, `E/F`. Mỗi family có hai ảnh chỉ khác một ký tự giữa chuỗi và một biến thể giữ nguyên nội dung nhưng dịch toàn bộ chữ một pixel. Tổng cộng 192 ảnh 256 × 256, mã hóa ở hai quality thành **384 stream**.

Tesseract 5.5.3 đọc đúng 116/128 ảnh nguồn thuộc hai biến thể ngữ nghĩa; 56/64 family có cả hai nguồn được đọc đúng. Đây là bộ lọc vận hành bằng OCR, chưa phải xác nhận khả năng đọc bằng người xem. Symbols được lấy từ chính đường `compress`, với means của entropy model; không thay bằng `round(y)`.

| Nhánh | Stream trùng trên 56 cặp đủ điều kiện | Symbols trùng trên 56 cặp | OCR đúng toàn chuỗi trên 116 nguồn đủ điều kiện | Bpp thực, toàn bộ 192 ảnh |
| --- | ---: | ---: | ---: | ---: |
| q1 MSE | 0 | 0 | 76/116 = 65,52% | 0,020449 |
| q2 MSE | 0 | 0 | 90/116 = 77,59% | 0,025205 |
| q2 + adapter Mean | 0 | 0 | 82/116 = 70,69% | 0,025205 |

Ở q2, 80 ảnh được cả hai decoder đọc đúng, 24 ảnh đều sai; **10 ảnh chuyển từ đúng sang sai và 2 ảnh chuyển từ sai sang đúng** sau adapter. Chênh lệch accuracy là −6,90 điểm phần trăm; CI 95% bootstrap theo family: [−13,91; −0,85] điểm phần trăm. Nhóm tương phản thấp chiếm 7/8 lỗi tăng ròng; phân nhóm này chỉ mang tính mô tả sau khi xem kết quả.

**Diễn giải:** trong family đã thử, không có bằng chứng về việc hai ký tự bị đồng nhất thành cùng stream. Vì vậy, không dùng “exact code collision gây lỗi chữ” làm câu chuyện chính. Tuy nhiên, mã khác nhau không chứng minh thông tin ngữ nghĩa đã được giữ đầy đủ. Lỗi OCR có thể liên quan đến encoder, decoder, dịch chuyển miền hoặc chính bộ đo. Nhánh perceptual làm giảm accuracy OCR trong phép thử này, chưa chứng minh giảm khả năng đọc của người xem.

## 3. S2 — Giữ moment màu có hơn phép trộn ảnh đơn giản?

Dùng **64 ảnh nguồn DIV2K mới**, không nằm trong 224 ảnh của pilot adapter. Mỗi ảnh lấy một crop 256 × 256 với seed cố định 2037. Đối chứng blend được chọn chỉ trên 32 ảnh validation cũ. Tất cả nhánh dùng cùng stream q2, đạt **0,18718719 bpp thực**; chưa kiểm overlap với dữ liệu pretraining của MLIC++.

Phép chiếu giữ đúng tổng RGB của ảnh MSE theo block 8/16/32, kể cả sau làm tròn uint8. Đối chứng random giữ signed moment theo dấu Rademacher với cùng số ràng buộc. Giá trị tham chiếu lấy từ **ảnh MSE giải mã**, không lấy từ nguồn. Sai số biên là chênh lệch finite differences trên RGB, độc lập với Sobel grayscale dùng khi huấn luyện adapter; đây vẫn chỉ là proxy cấu trúc.

| Nhánh | PSNR gộp ↑ | LPIPS-Alex ↓ | DISTS ↓ | Sai số biên ↓ |
| --- | ---: | ---: | ---: | ---: |
| MSE gốc | 30,8891 | 0,164954 | 0,192486 | 0,016801 |
| Adapter Mean | 29,7672 | 0,107925 | 0,154313 | 0,020292 |
| Giữ tổng RGB, block 8 | 29,8485 | 0,107450 | 0,150920 | 0,020261 |
| Giữ tổng RGB, block 16 | 29,8083 | 0,107098 | 0,151199 | 0,020268 |
| Giữ tổng RGB, block 32 | 29,7939 | 0,107312 | 0,151981 | 0,020273 |
| Signed moment ngẫu nhiên, block 8 | 29,7716 | 0,107717 | 0,154098 | 0,020393 |
| Signed moment ngẫu nhiên, block 16 | 29,7662 | 0,107489 | 0,154033 | 0,020362 |
| Signed moment ngẫu nhiên, block 32 | 29,7650 | 0,107755 | 0,154262 | 0,020333 |

**Đối chứng chưa đạt mục tiêu khớp LPIPS ở 5/6 nhánh trên validation.** Mọi hệ số blend được chọn đều bằng 1, tức giữ nguyên adapter Mean. Chỉ moment-32 đạt dung sai LPIPS 0,2% đã đặt trước. Các tên `matched-blend` trong tệp là tên dự kiến của control, không phải chứng nhận đã khớp; cờ `matched_validation` ghi đúng trạng thái.

So với Mean, moment-8 giảm DISTS 0,003392, khoảng 2,20%; CI chênh lệch [−0,005386; −0,001780]. Sai số biên chỉ giảm khoảng **0,153%**; CI chênh lệch [−0,00005358; −0,00001315]. LPIPS giảm trung bình 0,000475 nhưng CI [−0,001223; +0,000317] còn chứa 0. Moment-16 giảm LPIPS 0,000826, CI [−0,001267; −0,000422], song đây là một trong nhiều nhánh được thăm dò, chưa phải phương án chọn trước cho xác nhận.

Giữ tổng màu tốt hơn ràng buộc ngẫu nhiên cùng rank về DISTS và sai số biên trong mẫu này. Tuy nhiên, mức cải thiện biên quá nhỏ, và mọi nhánh perceptual vẫn có sai số biên cao hơn MSE gốc. **Chưa có bằng chứng về bảo toàn nội dung hay bước đột phá.** Giữ invariant của ảnh MSE cũng có thể giữ lại lỗi của chính ảnh đó.

## 4. S3 — Đọc ký tự trực tiếp từ symbols, với font giữ riêng

Chạy thêm chẩn đoán CPU: 1-nearest-neighbor cố định, train trên một font, test trên font còn lại; bỏ biến thể nuisance. Mỗi fold có 64 train và 64 test. Feature được trừ trung bình train rồi chuẩn hóa L2; không chọn hyperparameter bằng test. Nhãn là **ký tự giữa trong bốn lớp 6/8/E/F**, khác với yêu cầu đọc đúng toàn chuỗi ở S1.

| Representation / công cụ, q2 | Đúng ký tự giữa trên toàn bộ 128 ảnh |
| --- | ---: |
| 1-NN trên crop ký tự của ảnh nguồn | 54/128 = 42,19% |
| 1-NN trên crop ảnh MSE | 51/128 = 39,84% |
| 1-NN trên crop ảnh perceptual | 50/128 = 39,06% |
| 1-NN trên actual y/z symbols | 45/128 = 35,16% |
| Ký tự giữa từ OCR ảnh MSE, chỉ lấy khi chuỗi dài đúng 5 | 96/128 = 75,00% |
| Ký tự giữa từ OCR ảnh perceptual, cùng quy tắc | 90/128 = 70,31% |

**Probe chưa đủ năng lực đo:** ngay cả trên ảnh nguồn, 1-NN cũng chuyển font kém. Không dùng kết quả thấp của latent probe để kết luận encoder đã làm mất thông tin, cũng không suy ra decoder đã bỏ phí thông tin hữu ích. Đây là kết quả chưa phân giải được nguyên nhân. Pixel probe có bbox của generator; symbol probe có số chiều lớn hơn; hai representation chưa có readout mạnh tương đương. [Protocol và kết quả](../results/2026-11-05_mlicpp_cvpr_screen/readout-font-heldout/protocol.json).

## 5. Quyết định nghiên cứu

| Hướng | Quyết định hiện tại | Lý do |
| --- | --- | --- |
| Risk head của pilot trước | Dừng mở rộng | Chưa vượt CVaR và không đạt tiêu chí đã đặt |
| Exact-code semantic aliasing | Không dùng làm luận điểm chính | 0/56 collision ở cả q1 và q2 trong screen |
| Hard moment projection | Giữ làm đối chứng kỹ thuật | Lợi ích nhỏ; cơ chế projection đã có tiền lệ; chưa có nội dung/nhận dạng tốt hơn |
| “Latent giữ được chữ nhưng decoder bỏ phí” | Chưa xác nhận | S3 thất bại ngay ở kiểm tra năng lực readout trên nguồn |
| Bảo toàn nội dung khi nâng perceptual quality | Câu hỏi còn đáng nghiên cứu | Có tổn hại OCR đo được ở cùng stream, nhưng chưa xác định cơ chế hay tính mới |

Không chọn một tên model mới chỉ từ các screen này. Bước tiếp theo là sửa phép đo: nhận dạng bốn ký tự bằng readout có năng lực trên nguồn và font giữ riêng, kiểm song song ảnh nguồn/ảnh tái tạo/mã thật. Cần bộ font/layout mới vì các font hiện tại đã được dùng phân tích. Chỉ khi phép đo này phân giải được nguyên nhân mới chọn can thiệp encoder hay synthesis decoder.

Đợt xác minh tiếp theo nên giới hạn ở pilot: tối đa 2 GPU-giờ, không full-training. Điều kiện tối thiểu cho readout là đạt 95% trên nguồn giữ riêng; nếu chưa đạt, dừng suy luận về thông tin còn/mất trong latent. Nếu can thiệp decoder, đối chứng bắt buộc có text-mask/L1 hoặc OCR loss đơn giản, blending và projection; đánh giá ở cùng actual byte, có nhóm ảnh không chữ. Generic text-aware loss không tự là novelty.

## 6. Tính mới và giới hạn

Projection/nullspace đã có trong [Idempotent Learned Image Compression with Right-Inverse, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html); perceptual decoding gắn với codec/quantization có các tiền lệ như [PerQ](https://arxiv.org/html/2610.27374v1). Exact bitstream collision qua perturbation cũng đã được nghiên cứu trong [Bitstream Collisions in Neural Image Compression](https://arxiv.org/html/2503.19817v1). Không nhận các thành phần chung này là đóng góp mới của dự án. Danh sách đối chiếu đầy đủ hơn nằm trong kế hoạch nghiên cứu.

Các giới hạn còn lại: chỉ hai font và hai cặp glyph; nền tổng hợp quá đơn giản, nên bitrate không đại diện ảnh tự nhiên; OCR không đại diện human preference; 64 crop chưa đại diện full-resolution; một backbone legacy, một checkpoint adapter/seed, S2 chỉ một rate; chưa có L4 latency, bộ đo nhận dạng độc lập thứ hai hay human evaluation. Chưa chứng minh lợi ích trên MLICv2 hoặc so sánh với các phương pháp perceptual hiện đại dưới cùng điều kiện.

AI hỗ trợ tìm nguồn, thiết kế, viết code và phân tích. Kết luận hiện tại là kết luận sàng lọc; chưa có claim SOTA, tính mới được xác nhận hoặc khả năng được nhận CVPR.

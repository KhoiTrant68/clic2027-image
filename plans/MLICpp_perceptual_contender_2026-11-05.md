# Hướng MLIC++ cho CLIC và đóng góp model: quyết định ngày 05/11/2026

## Quyết định chính

**Thu hẹp contribution sau yêu cầu mới:** [hạt nhân prior-conflict](MLICpp_scientific_nucleus_2026-11-05.md) đặt câu hỏi về thông tin để sửa prior tin sai. Generic `U(ŷ)c` bên dưới chỉ là biểu diễn/control ứng viên; chưa đủ làm đóng góp khoa học chính. Hạt nhân mới chưa xác nhận hiện tượng, novelty hoặc measurement; không kế thừa việc pilot fine-tuning chạy được thành authorization cho full training mới.

**Cập nhật:** [pilot full synthesis đã hoàn tất](../reports/MLICpp_full_synthesis_pilot_2026-11-05.md), 128/128 fresh pixel matches. Nhánh không GAN giảm LPIPS 20,07%, DISTS 10,81% và tăng PSNR 0,3817 dB so với adapter nhỏ trên development32, cùng byte. Chọn nhánh này làm nền tiếp tục; GAN chưa xác nhận lợi ích. Conditional innovation vẫn là hypothesis chưa cài đặt.

**Chuyển từ adapter nhỏ sang synthesis decoder được huấn luyện perceptual đầy đủ.** Bắt đầu bằng đối chứng có/không conditional GAN, giữ nguyên entropy và stream đã kiểm. Đây là nâng cấp nền chất lượng, không gọi là novelty. Không tiếp tục lấy lỗi glyph hoặc Risk head làm câu chuyện chính.

**Ứng viên đóng góp mới để kiểm:** biểu diễn innovation có điều kiện theo latent đã giải mã, cho phép encoder dùng một ngân sách bit nhỏ để điều khiển một synthesis decoder cảm nhận chạy nhanh. Câu hỏi là liệu điều kiện từ base stream có giúp mã hóa phần hiệu chỉnh rẻ hơn residual thông thường và tăng bitrate base hay không. Chưa xác minh được tính mới hoặc lợi ích; không hứa top CLIC/CVPR từ tên kiến trúc.

Người dùng ưu tiên công trình đủ mạnh, chấp nhận kỳ CVPR sau. Không có đăng ký hoặc nộp bài tự động. Khảo sát này tiếp nối deep-research đã được cho phép; không phải systematic review hay khẳng định tìm hết mọi tài liệu.

## Vì sao đổi hướng

Pilot Mean giảm LPIPS/DISTS nhưng chỉ học 34.113 tham số trên một phần feature; phần synthesis chính bị đóng băng. Risk chưa vượt CVaR. Các screen chữ không tạo bằng chứng chắc cho semantic preservation: bộ đo đủ năng lực chỉ thấy chênh một ảnh trên test font mới. Do đó, trước khi thêm head hoặc side information, cần một decoder perceptual đủ mạnh để làm đối chứng.

Thi CLIC và viết paper có hai tiêu chí khác nhau. Để cạnh tranh cần chất lượng người xem ở actual byte hợp lệ và môi trường chấm. Để có đóng góp khoa học còn phải chứng minh cơ chế mới hơn đối chứng gần nhất. Generic GAN, flow, TTO hay knapsack không tự là contribution.

## Đối chiếu phương pháp mạnh và mã nguồn

Các nguồn dưới được kiểm ngày 05/11/2026; ghi rõ phạm vi đọc. Số liệu paper không phải kết quả local và không chuyển thẳng thành dự báo hạng thi.

| Nguồn primary | Bằng chứng và khả năng dùng | Ý nghĩa cho lựa chọn |
| --- | --- | --- |
| [CLIC2025 awards](https://archive.compression.cc/2025/awards/), [Vcoder whitepaper](https://openreview.net/pdf?id=zLYJJTc2CY) | Awards xác nhận Vcoder thắng GPU track. Nhánh audit đọc whitepaper qua bản indexed: DCVC-RT intra với pixel/perceptual/style/adversarial training. Truy cập PDF trực tiếp trong phiên chính gặp browser challenge. Chưa tái lập hoặc xác minh pretrained winner weights. | Thiết kế training perceptual đầy đủ có bằng chứng thi thật; không lấy PSNR làm mục tiêu thắng chính. |
| [PICO paper](https://arxiv.org/html/2605.05148v1), [repo tác giả](https://github.com/apple-aiml-research/ml-pico) | Methods đã đọc ở khảo sát trước, repo kiểm lại hiện có viewer/assets/reconstruction data; chưa xác minh codec/checkpoints được phát hành. | Đối chiếu human quality/runtime; không lập kế hoạch tải checkpoint chưa có. |
| [PerQ](https://arxiv.org/html/2610.27374v1) | Đọc §3, kết quả và Appendix; frozen RD codec + rate-conditioned flow, có MLIC++. Project URL hiện không truy cập được; code/weights chưa xác minh. | Closest cho flow compensator; tự triển khai không được gọi là tái lập PerQ. |
| [OneDC official code](https://github.com/onedc-codec/onedc) | Audit README/config: code, entropy coder và link pretrained weights; CC-BY-NC-SA 4.0 cùng điều kiện thành phần. Chưa chạy local, chưa đo gói/L4. | Đối chứng generative có artifact rõ; không mặc định native rate khớp CLIC hoặc đã qua giới hạn 4 GB. |
| [VLIC CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sargent_VLIC_Vision-Language_Models_As_Perceptual_Judges_for_Human-Aligned_Image_Compression_CVPR_2026_paper.html) | Audit methods/human study; chưa xác minh code/checkpoints. | Preference training đã có tiền lệ; cần đánh giá độc lập với reward dùng học. |
| [MLICv2 v4](https://arxiv.org/html/2504.19119v4) | Đã đọc §3.1–3.5; có latent refinement. | TTO trên latent và chỉ thắng legacy không đủ mới. |
| [SDGIC v2](https://arxiv.org/html/2512.06344v2) | Đọc §3.2.3–3.4; reconstruction-aware source-specific residual tokens. | “Gửi ít semantic side bits” đã có; ứng viên phải hơn bằng cơ chế/byte/cost cụ thể. |
| [LoRA-comp CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Test-Time_Fine-Tuning_of_Image_Compression_Models_for_Multi-Task_Adaptability_CVPR_2025_paper.pdf), [ISER 2026](https://www.mdpi.com/2079-9292/15/17/3807) | Audit methods/source; instance-specific adaptation và truyền singular values đã có. | Không đổi tên low-rank updates rồi nhận là mới. |
| [DDCM ICML2025](https://proceedings.mlr.press/v267/ohayon25a.html), [H-DDCM preprint 13/10/2026](https://arxiv.org/html/2610.18125v1) | DDCM có code chính thức; H-DDCM đọc phương pháp dictionary/coding. | Chọn seed/codeword theo nguồn rồi truyền index không phải novelty. |
| [Rate-Adaptive One-Step Diffusion Compression, preprint 25/09/2026](https://arxiv.org/html/2609.31795v1) | Đọc §3 và abstract; one-step AEIC + latent TTO + conditioning refresh + actual-byte allocation. | Chặn claim chung “one-step generator + TTO + byte thật” là đóng góp mới. Không dùng số điểm AIGC challenge làm dự báo CLIC. |

## Kiến trúc ứng viên, ở mức giả thuyết

Đường base: `x → MLIC++ entropy stream b → latent ŷ → synthesis perceptual G → x̂`.

Đường innovation có điều kiện: decoder dựng basis `U(ŷ)` từ latent đã giải mã. Encoder nhìn nguồn và chọn/lượng tử hóa hệ số `ĉ`; decoder tái tạo `f′ = f + U(ŷ)ĉ`, rồi chạy synthesis một lần. Toàn bộ header, scale, index, coefficient và padding của packet phải tính vào `B_total = B_base + B_innovation`. Không gửi ảnh nguồn, không học lại model tại decoder, không gọi dịch vụ ngoài.

Đây mới là **hypothesis về hiệu quả biểu diễn**, chưa là model đã cài đặt. Điểm cần chứng minh là basis có điều kiện tiết kiệm bit hơn fixed basis, pixel residual, low-rank adaptation và thêm bit cho base, chứ không phải việc có thêm packet. Nếu không vượt các đối chứng đó thì bỏ claim model này.

Một tham số hóa nhỏ có thể thử sau khi qua screen: từ feature `f`, mạng tạo K spatial maps `φ_k(f)`; learned channel vectors `w_k` tạo direction tách biến `U_k(f) = w_k φ_k(f)`. Code nguồn là K hệ số hoặc K × grid hệ số đã lượng tử hóa, không phải toàn bộ feature tensor. Đây chỉ là thiết kế ứng viên, chưa phải chứng minh mới hoặc hiệu quả. Header phải chỉ rõ mode/grid/precision; scale cố định chung thì không tốn byte, còn scale theo ảnh phải serialize. Cần đo cả gói mô hình/dictionary trong decoder.

**Đối chứng quyết định cho mechanism:** ngoài random/PCA, phải có **learned global basis `U₀c`** không nhận feature từng ảnh. Giữ cùng rank, quantizer, packet syntax, coefficient-search budget, dữ liệu và loss. Nếu `U(ŷ)c` không thắng `U₀c` ở cùng tổng byte và chi phí, không có bằng chứng rằng conditioning theo base stream là đóng góp; không giữ module đó chỉ để đặt tên mới.

Một conditional residual flow 1/2/4 bước có thể là thành phần generative ở bước sau. Phải so với mạng deterministic cùng capacity; một bước Euler không tự có chất lượng của mô hình đã distill một bước. Không thêm flow chỉ để làm phương pháp trông phức tạp.

## Thứ tự thực nghiệm đã chốt

| Bước | Can thiệp | Đối chứng / điều kiện tiếp tục |
| --- | --- | --- |
| 1 — triển khai ngay | Fine-tune **toàn bộ g_s** từ MSE q2: reconstruction loss và cùng loss + conditional hinge GAN | Cùng init, nguồn, batches, bước, G LR; chỉ adversarial training khác. Development validation trước, chưa mở test mới. Đây là baseline kỹ thuật. |
| 2 | Encoder latent TTO qua decoder cuối | Serialize candidates thật; actual-byte budget, so TTO qua decoder MSE; không tăng computation decoder. Chỉ làm khi decoder nền ổn định. |
| 3 | Pilot source-coded feature innovation | Trước hết fixed PCA/random/channel basis; continuous diagnostic rồi packet thật. So ordinary residual và base-more-bits; không nội suy quality để giả vờ matched rate. |
| 4 | Học basis có điều kiện và/hoặc compact flow | Chỉ mở nếu bước 3 có tín hiệu dưới budget; đối chứng deterministic cùng capacity và fixed basis. |
| 5 | Đánh giá ứng viên mạnh | Data/test mới, nhiều rate/seed, full-resolution, modern MLIC family/OneDC, container L4 và đánh giá cặp người xem. |

### Cấu hình pilot bước 1

- Giữ 160 ảnh train, hai crop 256²/ảnh (320 mẫu) và 32 validation crop cũ. Validation đã dùng trong nghiên cứu trước, nên **chỉ là development**, không báo như test độc lập. Không đọc 32 test cũ hoặc 30 ảnh CLIC.
- Từ từng stream q2 đã có, giải mã và cache **input đầy đủ của g_s**; kiểm uint8 base khớp cache cũ. Không dùng encoder feature chưa qua quantization.
- Fine-tune g_s, entropy/encoder đóng băng. Hai nhánh 1.000 bước, batch 4, seed 1984, Adam G LR 1e−5; loss `10 MSE + LPIPS + 0,1 edge_rational` như pilot cũ. LR giảm từ adapter 1e−4 có chủ đích do mở toàn synthesis.
- Nhánh GAN thêm conditional PatchGAN trên cặp (ảnh MSE, ảnh tái tạo), hinge loss, D LR 1e−4; adversarial weight 0,01 sau warmup 100 bước rồi tăng tuyến tính 100 bước. Không claim đây là công thức Vcoder được tái lập nguyên bản.
- Checkpoint bước cuối cố định. Theo dõi training JSONL, status stage, VRAM; không chọn checkpoint bằng validation metrics sau khi chạy. Đánh giá PSNR/LPIPS/DISTS/MS-SSIM/sai số biên, bootstrap ghép cặp theo ảnh; metric không thay human preference. Gói synthesis cuối được fresh-decode từ stream thật để kiểm pixel.
- Smoke hai bước/nhánh phải qua cả GAN gradient và fresh parity trước run chính. Output mới, không ghi đè. GPU 0 L40 đang có background khoảng 1,3 GB; dùng guard có sẵn, không động vào job GPU khác.
- Ước tính 15–45 phút sau smoke, **ceiling 2 giờ trên một GPU** cho toàn pipeline; không API trả phí; dự kiến dưới 2 GB artifact. Cấu hình tại `experiments/mlicpp/full_synthesis_q2.json`, runner `experiments/mlicpp/run_full_synthesis.sh`. Mốc giờ kết thúc phải hiệu chỉnh theo tốc độ smoke thực tế, theo giờ Việt Nam.

### Điều kiện bác bỏ

Không xem một metric tốt hơn là thắng CLIC. Nếu GAN bất ổn, hoặc không tạo frontier có lợi hơn reconstruction-only trên bộ đo độc lập với objective, dừng mở rộng và giữ log. Nếu gain chỉ từ hy sinh fidelity giống blending, giữ làm baseline chứ không nhận cơ chế mới. Nếu packet innovation mất lợi khi tính đủ header/scale hoặc bị ordinary residual/base-more-bits vượt, bỏ phần đóng góp đó. Chỉ thêm flow sau khi có lý do thực nghiệm, không bù một nền huấn luyện chưa đạt.

Hai nhánh audit đã kiểm định hướng này: khuyến nghị xây full-synthesis perceptual baseline; phản biện yêu cầu learned global basis control nêu trên và cảnh báo validation reuse/metric exploitation. Đây là phân vai audit của AI, không phải hai bằng chứng độc lập cho novelty. Một nút thắt CLIC riêng vẫn còn: q1 legacy hiện chưa phủ ngân sách 0,075 bpp cho toàn 30 ảnh. Pilot q2 này không giải quyết rate coverage đó; phải xử lý rõ khi chọn gói thi, không gắn nhãn q2 crop là đã đạt cả ba rate.

## Trạng thái khoa học

Smoke đã qua cả hai nhánh và 8/8 fresh pixel matches. Run chính được khởi chạy ngày 05/11 khoảng 14:15 giờ Việt Nam, tên `div2k-q2-full-synthesis-conditional-gan`, trên L40 GPU 0; ETA ban đầu 14:30–14:45, hard ceiling hai giờ. Output dưới `/home/jovyan/clic2027-artifacts/mlicpp/full-synthesis-20261105/`. Trạng thái và kết quả cuối phải đọc từ artifact, không suy ra thành công chỉ vì process đã được khởi chạy.

Pilot bước 1 đã hoàn tất trong 313,60 giây, nhanh hơn ETA ban đầu; hai checkpoint cuối và số liệu được lưu, nhánh reconstruction được chọn cho nghiên cứu tiếp theo. Kết quả là development, không phải test/human evaluation. Tính mới của conditional innovation còn **chưa xác nhận**. Chưa có cơ sở nói đạt top CLIC, human preference tăng hay đủ cho CVPR. AI hỗ trợ tìm nguồn, audit, đề xuất và code; các nhánh audit không phải đánh giá độc lập thống kê.

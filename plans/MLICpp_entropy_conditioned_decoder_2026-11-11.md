# Decoder MLIC++ dùng thống kê của ô lượng tử

## Câu hỏi nghiên cứu

Ở cùng bitstream, thống kê về phần chưa biết bên trong ô lượng tử có giúp decoder một lượt tái tạo ảnh tốt hơn một decoder có cùng số tham số hay không? Phép thử ưu tiên 0,075 bpp, nơi Mixed đang là đối chứng nội bộ mạnh nhất theo LPIPS-Alex và DISTS. Single và Perceptual là đối chứng ở hai ngân sách còn lại. [Kết quả CLIC30](../reports/MLICpp_clic30_decoder_comparison_2026-11-11.md)

## Đối chiếu nghiên cứu liên quan

Khảo sát dùng các cụm *MLICv2 pretrained*, *learned image compression quantization error compensation*, *latent shift entropy gradient*, *Gaussian quantization cell conditional moments* và *fast perceptual image compression*. Ưu tiên bài gốc, trang tác giả và mã chính thức mô tả trực tiếp các cơ chế liên quan. Phạm vi này phục vụ thiết kế prototype và chưa đủ để khẳng định tính mới.

| Nguồn | Cơ chế liên quan | Hệ quả đối với phép thử |
| --- | --- | --- |
| [MLICv2, Jiang và cộng sự, 2025](https://arxiv.org/html/2504.19119v1) | Thay transform và mô hình entropy. MLICv2+ tối ưu latent theo ảnh ở phía encoder. | Ứng viên backbone thứ hai. Hàm pretrained trong [CompressAI](https://interdigitalinc.github.io/CompressAI/_modules/compressai/zoo/image.html#mlicv2) hiện báo chưa có trọng số. Prototype dùng MLIC++ q2 đã tái lập. |
| [Channel-wise Autoregressive Entropy Models, Minnen và Singh, 2020](https://arxiv.org/abs/2007.08739) | Mô hình ngữ cảnh theo lát kênh và latent residual prediction (LRP). | Đo thống kê trước LRP và áp dụng hiệu chỉnh sau khi giải mã entropy để giữ nguyên ngữ cảnh nối tiếp. |
| [Exploiting Latent Properties to Optimize Neural Codecs, 2025](https://arxiv.org/abs/2501.01231) | Latent Shift dùng gradient entropy sẵn có ở decoder để hiệu chỉnh latent. | Cần đối chứng với hiệu chỉnh từ score Gaussian hoặc trung bình hậu nghiệm để xác định giá trị của phần học được. |
| [PerQ, Peng và cộng sự, 2026](https://arxiv.org/html/2610.27374v1) | Bù sai số latent với flow matching, dùng thông tin lượng tử và backbone cố định. | Công trình gần nhất. Giá trị cần kiểm chứng ở đây là moment Gaussian tường minh và chi phí của một lượt giải mã. Bù sai số ở decoder, giữ bitstream và giới hạn độ hiệu chỉnh đều đã có tiền lệ. |
| [PICO, Tatwawadi và cộng sự, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.pdf) | Đo chất lượng cảm nhận, thời gian giải mã và các lỗi thực tế như chữ hoặc đường nối tile. | Đo tốc độ toàn ảnh và kiểm lỗi chữ. LPIPS phục vụ sàng lọc, còn chấm cặp của người xem phục vụ chọn bản thi. |

## Kiến trúc và bảo đảm có thể kiểm

Gọi μ và σ là trung bình và scale do entropy model dự đoán, q là symbol đã giải mã, Δ là bước lượng tử. Viết latent trước lượng tử dưới dạng y = μ + Δ(q + t), với t thuộc [−1/2, 1/2]. Theo mô hình Gaussian liên tục, mật độ có điều kiện của t tỉ lệ với exp[−(q + t)² / (2(σ/Δ)²)] trên ô. Trung bình có điều kiện tối ưu bình phương sai số của t theo mô hình này. Phương sai mô tả độ phân tán còn lại trong ô. Chất lượng ảnh vẫn cần kiểm bằng thực nghiệm.

Mã xấp xỉ moment bằng Gauss–Legendre ba điểm, rồi lấy trung bình theo kênh của moment bậc nhất, trị tuyệt đối của nó và phương sai. Ba bản đồ đi vào nhánh hiệu chỉnh có bottleneck 16 kênh và depthwise convolution. Nhánh có 10.784 tham số ở 320 kênh latent, chạy một lần và khởi tạo với hiệu chỉnh bằng 0. Hàm Δ tanh(·)/2 chặn biên độ hiệu chỉnh so với latent sau LRP. Giới hạn này không suy ra latent cuối vẫn thuộc ô gốc. Prototype dùng scale liên tục đã chặn dưới và moment xấp xỉ, nhất là ở đuôi phân bố. [Mã module](../experiments/mlicpp/entropy_synthesis.py)

Wrapper đọc giá trị đã giải mã trước LRP và trả nguyên giá trị đó về entropy model. Khi đủ các lát, synthesis nhận bản đồ thống kê. Mỗi phép thử đối chiếu hash stream giữa các nhánh và hash pixel sau khi giải mã trong tiến trình riêng.

## Thí nghiệm và quyết định

| Nhánh | Cấu hình | Mục đích |
| --- | --- | --- |
| Plain | Synthesis gốc được fine-tune | Đo lợi ích của huấn luyện cùng lịch |
| Entropy-blind | Có nhánh mới, ba bản đồ đầu vào bằng 0 | Đối chứng cùng số tham số và giới hạn residual |
| Entropy | Nhận ba bản đồ moment theo vị trí | Đo giá trị của tín hiệu entropy |

Ba nhánh dùng cùng checkpoint khởi tạo, 160 crop DIV2K, cùng seed, 1.000 bước, loss 10 MSE + LPIPS-Alex và factor 1/2/4. Smoke dùng 2 bước mỗi nhánh. Trên 32 crop development, đo LPIPS-Alex, DISTS, PSNR và latency có tính phần quan sát entropy. So theo cặp ảnh ở cùng stream và báo khoảng tin cậy bootstrap theo ảnh.

Nếu Entropy vượt Plain nhưng ngang Entropy-blind, nhánh bổ sung hoặc giới hạn residual có thể giải thích lợi ích. Nếu Entropy vượt cả hai, so thêm với bản chỉ dùng scale, moment hoán vị theo vị trí và hiệu chỉnh giải tích. Kiểm độ chính xác của moment bằng tích phân số và latent trước lượng tử trên ảnh giữ riêng. Chạy thêm seed và backbone trước khi khái quát.

Để quyết định bản thi, dùng đúng gói CLIC30 đã đóng băng ở 0,075 / 0,15 / 0,30 bpp. So với Mixed, Single, Perceptual và đối chứng mạnh có đủ ảnh. Kiểm thời gian toàn tập gần cấu hình L4, chất lượng chữ và chấm cặp mù trên ảnh giữ riêng. CLIC yêu cầu đủ ba ngân sách, dùng người chấm để chọn người thắng và loại 25% bài giải mã chậm nhất ở mỗi track khỏi xét giải. [Thể lệ CLIC](https://clic2027.compression.cc/tasks/)

## Tái lập và giới hạn

Script server nằm tại /home/jovyan/clic2027-artifacts/mlicpp/entropy-synthesis-20261111/run.sh. Nó dùng GPU 1, NVRTC của venv và cấu hình MLIC++ q2 cũ. Smoke chạy trước. Khi đạt 12 lượt giải mã khớp, script khởi chạy ba nhánh 1.000 bước. Mỗi lượt lưu trạng thái, log, checkpoint và số đo từng ảnh. Evaluator CLIC30 nhận ba nhánh plain, entropy-blind và entropy.

Phép thử crop chỉ dùng để sàng lọc. Hiệu quả cảm nhận toàn ảnh, chi phí trên L4, sức cạnh tranh với codec ngoài MLIC++ và khả năng chuyển sang MLICv2 còn cần đo. Thiết kế và khảo sát có sử dụng công cụ AI.

Smoke trên L40 đã hoàn tất. Cả ba nhánh train hai bước và 12 lượt giải mã trong tiến trình riêng khớp hash pixel. So sánh hash xác nhận các nhánh dùng cùng bitstream tại từng factor. Phép thử số xác nhận moment đối xứng theo dấu symbol, phương sai tiến về 1/12 khi scale lớn và hiệu chỉnh khởi tạo bằng 0. Trên lưới kiểm tra nhỏ có cả đuôi phân bố, sai số tuyệt đối lớn nhất của trung bình xấp xỉ ba điểm là 0,1111 đơn vị symbol so với tích phân lưới mịn.

## Kết quả train và đối chiếu ngày 13/11

Server xác nhận cả ba nhánh đã hoàn thành 1.000 bước mỗi nhánh, lưu checkpoint và ghi nhận 384 lượt giải mã khớp. Ở factor 4, tương đương 0,08727 bpp trên 32 crop development, kiểm từng tên ảnh và hash bitstream cho thấy cả ba đối chứng dùng cùng stream với Entropy. Khoảng 95% dưới đây là percentile bootstrap ghép cặp theo ảnh với 10.000 lượt lấy mẫu. Số âm có lợi cho Entropy.

| Đối chứng | Δ LPIPS-Alex [95%] | Δ DISTS [95%] |
| --- | ---: | ---: |
| Checkpoint khởi tạo | −0,090666 [−0,114058; −0,070056] | −0,047218 [−0,057623; −0,036828] |
| Plain fine-tune | +0,000659 [−0,000317; +0,001719] | +0,000024 [−0,000479; +0,000540] |
| Entropy-blind | −0,000231 [−0,000438; −0,000054] | −0,000107 [−0,000329; +0,000128] |

Tín hiệu moment giảm nhẹ LPIPS so với nhánh cùng số tham số. Entropy chưa vượt Plain ở mức bitrate thấp đã thử. Cả ba nhánh fine-tune có LPIPS và DISTS cao hơn checkpoint khởi tạo ở factor 1. Kết quả hiện tại là thăm dò trên crop, một seed, và chưa có đối chứng Mixed trên ảnh CLIC30 toàn độ phân giải.

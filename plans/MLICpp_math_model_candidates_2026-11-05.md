# Ba ứng viên thiên về toán và kiến trúc codec

Đề xuất theo yêu cầu ngày 05/11/2026. Đây là các ý tưởng do AI đề xuất, chưa cài đặt, chưa xác nhận novelty và chưa thay thế diagnostic đã được đồng ý ở hướng evidence decoder. Các đồng nhất thức nền dưới đây không phải theorem mới của dự án. Specific novelty verdict: **NOT_ASSESSED** sau screening; cần đọc sâu đối thủ trực tiếp trước khi chốt hạt nhân.

## 1. Quantizer theo hình học decoder, đồng bộ bằng context

**Ưu tiên đề xuất:** cao nhất cho mục tiêu vừa đổi model vừa có bài toán toán học. Can thiệp vào quantizer và entropy model, không chỉ sửa ảnh sau giải mã.

Đặt F(y) = Φ(g_s(y)), trong đó Φ có thể ghép pixel và feature cố định với trọng số đã khóa. Với Jacobian J_F, local metric là G(y) = J_F(y)ᵀJ_F(y). Khi F đủ trơn trong vùng xét:

$$
\|F(y+e)-F(y)\|^2=e^\top G(y)e+O(\|e\|^3).
$$

Đây là biến đổi đầu ra **do quantization**, không mặc định bằng distortion so nguồn x. Source reconstruction residual, vùng không trơn và quantization lớn đều cần phân tích riêng. G có thể singular; cần regularization hoặc lý thuyết trên support, không mặc định G luôn SPD.

Chia latent thành block. Dùng context cᵢ chỉ gồm hyperlatent đã decode và symbols ở các bước trước để dự đoán transform Aᵢ khả nghịch:

$$
k_i=\operatorname{round}\!\left(A_i(c_i)(y_i-\mu_i)/\Delta_i\right),\qquad
\hat y_i=\mu_i+A_i(c_i)^{-1}(\Delta_i k_i).
$$

Hai phía dựng cùng Aᵢ từ cᵢ, rồi entropy-code kᵢ bằng model mới. Không dựng Aᵢ từ current unquantized yᵢ tại encoder rồi cho decoder dùng miễn phí. Nếu gửi một phần transform thì phải tính metadata. Hyperlatent cũng tốn byte; thay hyperprior để dự đoán metric phải tính cả chi phí đó.

Một kiến trúc nhỏ để kiểm khả thi là transform block có conditioning được kiểm soát, tham số hóa bằng triangular hoặc diagonal + low-rank và regularize condition number. Teacher metric từ JVP/VJP chỉ dùng offline khi train; decoder chạy predictor nhẹ. Không xây full Jacobian cho toàn ảnh.

**Bài toán lý thuyết đáng làm:** distortion/rate gap khi geometry lý tưởng phải được xấp xỉ từ thông tin decoder thực sự biết; sai số dự đoán metric, phi tuyến và finite precision tác động thế nào? Cần một kết quả có nội dung vượt Taylor expansion hoặc AM–GM chuẩn.

Động cơ nền: dưới mô hình lỗi zero-mean isotropic trong tọa độ biến đổi, surrogate distortion tỷ lệ với tr(G A⁻¹A⁻ᵀ). Đổi shape có thể có ích; **rotation đơn thuần giữ isotropic covariance không tự giảm trace**. Cùng cell volume cũng không có nghĩa cùng entropy-coded rate ở bitrate thấp.

**Prior art phải vượt:** [OLVQ, NeurIPS 2024; arXiv v2 04/11/2026](https://arxiv.org/abs/2411.16119), [adaptive LVQ, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.pdf), [Lattice Transform Coding, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5170d42c38539462236b23e03808068c-Abstract-Conference.html), và [task-induced pullback metric](https://cyberiada.github.io/TaskInducedViTs/). Nguồn cuối mới xác minh trang tác giả, chưa xác minh venue. Lattice, pullback metric và distill Jacobian không tự là novelty.

**Falsifier rẻ:** với backbone đóng băng, so oracle geometry, predictor từ decoded context, learned global transform, diagonal transform và quantizer gốc. Đầu tiên xem oracle có giảm loss hard-quantized không; sau đó mới kiểm predictor giữ được lợi ích và mã hóa thật. Nếu oracle không có gain, hoặc entropy/header/compute xóa gain, dừng. Nếu conditional không vượt global, bỏ claim conditioning như đã làm với pilot basis cũ. Quantizer/entropy thay đổi tạo codec và bitstream mới; không đọc stream cũ bằng quantizer mới như thể tương thích.

## 2. Entropy model có đồ thị phụ thuộc và ngân sách song song

**Ưu tiên đề xuất:** thứ hai; sát phần entropy của MLIC++, có đường đo byte/latency rõ hơn việc nhận preference.

Mô hình học latent representation, graph dependencies và partition thành T nhóm decode song song. Lịch nhóm phải cố định hoặc suy được từ hyperlatent đã decode. Nếu lịch được chọn bằng thông tin nguồn chưa truyền, phải tính phí truyền lịch.

Với lịch partition cố định khi biết Z, model factorize độc lập trong mỗi nhóm Gₜ. Ideal expected log-loss tách thành:

$$
R=H(Y\mid Z)
+\sum_t\operatorname{TC}(Y_{\mathcal G_t}\mid Y_{\mathcal G_{<t}},Z)
+\mathcal E_{model}.
$$

TC là conditional total correlation bị bỏ qua khi decode song song; E_model là tổng marginal conditional KL do predictor sai. Đây là identity cho discrete log-loss lý tưởng, chưa gồm hyperstream, integer CDF, coder flush/header hoặc hardware time. Entropy H(Y|Z) không đổi chỉ vì đổi thứ tự chain rule khi model hoàn hảo.

**Model ứng viên:** graph router từ hyperprior, nhóm có số bước T giới hạn, entropy predictor theo nhóm, và transform được học để giảm phụ thuộc gây tốn bit giữa các latent phải decode cùng lúc. Objective kết hợp actual-rate proxy, distortion và decoder latency đã đo. Đổi representation mới có thể đổi H; schedule đơn thuần tác động approximation/parallelism.

**Bài toán lý thuyết:** với lớp nguồn Gaussian có cấu trúc/sparse precision được nêu rõ, liên hệ lỗi partition, cross-group dependency và excess rate; xây thuật toán partition có chặn hợp lệ. Không nhận conditional MI luôn submodular hoặc greedy có tỷ lệ xấp xỉ nếu chưa chứng minh giả thiết.

**Prior art:** [HPCM, ICCV 2025](https://arxiv.org/abs/2507.19125) đã có hierarchical schedule và progressive contexts; [GLIC, CVPR 2026](https://arxiv.org/html/2603.25316v1) đã có graph-based adaptive compression. Cần vượt cả flexible coding order và grouping trong ELIC/MLIC; không nhận generic graph/router/đổi thứ tự là mới.

**Falsifier:** đầu tiên cố định encoder và T, so cùng model capacity, training budget và actual byte/latency với nhóm cố định. Nếu graph không cải thiện frontier hoặc routing/synchronization chậm hơn phần tiết kiệm, dừng. Việc giảm T mà bitrate tăng không tự là thất bại, nhưng phải tạo điểm trade-off hữu ích theo mục tiêu đã khóa.

## 3. Codec progressive với posterior refinement giữa các mức rate

**Ưu tiên đề xuất:** thứ ba; nhiều cơ hội toán nhưng rủi ro và khối lượng thay model cao nhất.

Một stream có base và enhancement layers. Decoder conditional dùng từng prefix, thay vì model/checkpoint độc lập cho mỗi rate. Một hướng cấu trúc là nested quantizers kết hợp conditional generative model có coupling giữa các tầng.

Với sigma-field Fₜ sinh bởi prefix và conditional mean lý tưởng mₜ = E[X|Fₜ], nguồn có moment bậc hai hữu hạn:

$$
\mathbb E\|X-m_t\|^2-\mathbb E\|X-m_{t+1}\|^2
=\mathbb E\|m_{t+1}-m_t\|^2\geq0.
$$

Đây là martingale/Pythagoras chuẩn. Nó không bảo đảm mỗi ảnh, một neural approximation hoặc LPIPS/human preference đơn điệu. Không phải mọi nguồn đều successively refinable mà không mất optimality.

**Bài toán mới cần tìm:** finite-packet refinement penalty và approximation gap của một neural family cụ thể, gắn được với nested quantizer + posterior coupling chạy thật. Không lấy theorem trong giới hạn asymptotic/unlimited randomness rồi áp thẳng cho codec hữu hạn.

**Prior art:** [LTC](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5170d42c38539462236b23e03808068c-Abstract-Conference.html) đã gồm nested lattices; [Successive Refinement Under Strong-Sense Perfect Perception, preprint 09/2026](https://arxiv.org/abs/2609.14404) đã nghiên cứu RDP region với unlimited common randomness. Không nhận progressive coding hoặc perception refinement nói chung là novelty.

**Falsifier:** so prefix codec với các operating point độc lập ở cùng tổng byte, gồm framing. Nếu nesting penalty lớn, thấp rate không phủ được mục tiêu hoặc neural posterior chỉ cải thiện metric được tối ưu, chưa có cơ sở mở rộng.

## Khuyến nghị

Nếu ưu tiên toán dẫn kiến trúc và có bước thử giới hạn trên model hiện có, bắt đầu kiểm ứng viên 1. Ưu tiên 2 nếu muốn tập trung bitrate–latency và entropy thay vì generative perception. Chưa ghép ba hướng thành một framework. Trước khi train lớn, đọc sâu đối thủ gần nhất và kiểm oracle/feasibility tương ứng; các identity nền không thay thế novelty hoặc kết quả codec thật. Khảo sát mới là screening có giới hạn tới 05/11/2026, không chứng minh chưa ai làm cơ chế tương tự.

# Ghi chú thuyết trình CLIC 2027

Khoảng 50 phút trình bày và 10 phút thảo luận.

## 1. CLIC 2027

Thời lượng dự kiến: 50 phút trình bày và 10 phút trao đổi. Mở đầu bằng bài toán: chất lượng cảm nhận dưới ngân sách byte, đồng thời decoder phải chạy được trong môi trường thi. Các kết quả là snapshot ngày 15/10/2026, không phải xếp hạng cuộc thi.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 2. Nội dung trình bày

Dành khoảng 8 phút cho bối cảnh và cách đánh giá, 22 phút cho các mô hình, 13 phút cho kết quả, 7 phút cho kế hoạch. Phần hỏi đáp khoảng 10 phút. Có thể lướt qua các mô hình tham khảo nếu cần dành nhiều thời gian cho thảo luận.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 3. Bài toán CLIC 2027 Image GPU

Luật được kiểm lại từ trang chính thức ngày 15/10/2026. Cần phân biệt giới hạn 5 giờ của một submission với điều kiện loại 25% bài chậm nhất khỏi giải thưởng. Mốc 20 giây của dự án chỉ là mục tiêu nội bộ. Mỗi decoder dùng ở test phải khớp hash decoder đã đăng ký trong validation.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://clic2027.compression.cc/tasks/](https://clic2027.compression.cc/tasks/)

## 4. Chất lượng ảnh có hai khía cạnh

Giải thích bằng trường hợp chữ: ảnh có texture sắc nét vẫn có thể sinh sai ký tự. Với vùng cỏ hoặc lông, sai số pixel thấp có thể tạo cảm giác bị làm mượt. Mục tiêu là cân bằng fidelity với ảnh gốc và độ tự nhiên, sau đó xác nhận bằng người. Không suy chất lượng cảm nhận từ PSNR đơn lẻ.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf)

## 5. Pipeline đánh giá của dự án

Mỗi bước tạo artifact kiểm được. Tập 30 ảnh là validation chính thức, trùng test CLIC 2025. Những thí nghiệm trên cùng tập này hỗ trợ lựa chọn kỹ thuật nhưng cần holdout độc lập để đánh giá tổng quát hóa. Không dùng validation để train codec.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://clic2027.compression.cc/tasks/](https://clic2027.compression.cc/tasks/)

## 6. Q̂ học thứ tự preference từ nhiều metric

Trình bày Bradley–Terry: mô hình dự đoán xác suất một reconstruction được thích hơn reconstruction kia. Mỗi metric đi qua biến đổi nhất quán trước khi cộng trọng số. Bias vị trí A/B chỉ dùng cho xác suất, không cộng vào utility từng ảnh của allocator. Q̂ là thước đo sàng lọc nội bộ.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 7. Q̂ có tín hiệu tốt trên phép kiểm cross-year

Cùng miền test 2024t, dấu PSNR đạt 0.705 và Q̂ v0 đạt 0.812. Q̂ v1 đạt 0.833 nhưng ranking ở rate thấp chưa phù hợp nên bảng codec vẫn dùng v0. Không trộn với kết quả 5-fold trên 8.000 cặp. Cần kiểm split theo ảnh, leakage và calibration.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 8. So sánh công bằng dưới ngân sách byte

Bpp toàn tập tính từ tổng byte và tổng pixel, không lấy trung bình bpp từng ảnh nếu kích thước khác nhau. Allocator chọn từ các điểm đã encode thật. Mỗi q không tương đương một bpp cố định giữa model. Header, mode và các tham số cần để decode đều phải tính vào stream.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 9. Bản đồ các họ mô hình

Dùng bảng này để định vị trước khi đi vào chi tiết. Các nhóm chồng lấn: PULSE vẫn có hyperprior, DiT-IC vẫn cần entropy coding. Điểm phân biệt là đường dựng ảnh và nơi đặt chi phí. Đây là cách tổ chức của báo cáo, không phải phân loại chính thức của tác giả.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 10. Nền tảng mean-scale hyperprior

Encoder tạo latent y, hyper encoder tạo z. Phải entropy-decode z trước để dự đoán mean và scale cho y. Mạng synthesis biến latent giải mã thành RGB. Trong repo, mbt2018_mean là MeanScaleHyperprior, không phải biến thể raster-autoregressive đầy đủ trong paper MBT.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/pdf/1809.02736](https://arxiv.org/pdf/1809.02736) · [https://interdigitalinc.github.io/CompressAI/models.html#meanscalehyperprior](https://interdigitalinc.github.io/CompressAI/models.html#meanscalehyperprior)

## 11. MS-ILLM dùng nhãn cục bộ khi training

Paper ICML 2023 dùng VQ-VAE để tạo nhãn đặc trưng cục bộ. U-Net discriminator dự đoán codebook label của ảnh thật hoặc fake. Loss buộc reconstruction giữ cấu trúc nhãn của ảnh gốc. Khi decode chỉ cần codec, không cần VQ-VAE tạo nhãn hoặc discriminator.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf)

## 12. MS-ILLM giữ decoder triển khai độc lập

Giai đoạn 1 train autoencoder không discriminator trong 1 triệu bước. Giai đoạn 2 fine-tune decoder với mục tiêu rate-distortion-perception. Không cần trình bày toàn bộ hyperparameter trong slide, có trong phụ lục A report. Nhấn mạnh tín hiệu training tốt có thể giúp decoder mà không mang teacher vào gói cuối.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf)

## 13. DiT-IC ghép latent codec với SANA và DC-AE

Giải thích theo code đang chạy: latent codec giải mean, log-variance, residual và prompt latent. Prompter cung cấp conditioning cho SANA. Sau một forward DiT và scheduler step, cộng residual và hiệu chỉnh scale rồi DC-AE dựng RGB. Một bước diffusion vẫn có thể dùng backbone lớn.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2603.13162v1](https://arxiv.org/html/2603.13162v1) · [https://github.com/Eric-qi/DiT-IC](https://github.com/Eric-qi/DiT-IC)

## 14. Ba alignment của DiT-IC

Variance-guided flow đưa uncertainty vào mức tái tạo. Self-distillation dùng biểu diễn encoder đóng băng làm đích. Latent guidance thay prompt văn bản bằng embedding từ latent. Phân biệt recipe trong paper với thí nghiệm student của dự án, vốn thu hẹp một phần decoder DC-AE.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2603.13162v1](https://arxiv.org/html/2603.13162v1)

## 15. DC-AE và SANA tạo nền tảng cho DiT-IC

DC-AE giải quyết khó khăn khi nén không gian sâu qua residual autoencoding và training nhiều giai đoạn. SANA giảm chi phí attention trong latent nhỏ. Cả hai là nền tảng representation và generation. Để thành codec cần thêm lượng tử hóa, entropy model và bitstream độc lập.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2410.10733v1](https://arxiv.org/html/2410.10733v1) · [https://arxiv.org/html/2410.10629v1](https://arxiv.org/html/2410.10629v1)

## 16. Kiến trúc PULSE

Hình nguyên bản từ repo Microsoft GenCodec. Đi từ trái sang phải: encoder low-resolution, entropy path tách CDF số nguyên, cuối cùng là synthesis nhẹ dùng AdaLN renderer. Dừng ở vùng entropy để giải thích hai lượt. Giới thiệu sự khác nhau giữa base và L trên slide tiếp theo.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2609.18602v1](https://arxiv.org/html/2609.18602v1) · [https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md](https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md) · [https://raw.githubusercontent.com/microsoft/GenCodec/main/PULSE/assets/architecture.png](https://raw.githubusercontent.com/microsoft/GenCodec/main/PULSE/assets/architecture.png)

## 17. PULSE tách entropy tất định khỏi synthesis float

Integer Linear CDF chọn bảng xác suất. Meta Prior truyền bank index cho hyperlatent. Mạng synthesis float có thể khác pixel giữa các nền tảng ngay cả khi entropy symbol khớp. PULSE-L là bản perceptual lớn hơn, không được lấy 5.2 kMAC/pixel của base để mô tả L.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2609.18602v1](https://arxiv.org/html/2609.18602v1) · [https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md](https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md)

## 18. CoD-Lite dùng condition giàu thông tin

Paper dùng VQ với mã chỉ số độ dài cố định. Condition decoder phục hồi đặc trưng cho diffusion convolution một bước trong pixel space. DMD teacher CoD và GAN chỉ phục vụ training. Trong repo inference khởi đầu bằng tensor không, pad theo bội số 64 và crop lại.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2604.12525v1](https://arxiv.org/html/2604.12525v1)

## 19. DC-VIC và AFP-GIC chuyển prior vào decoder

DC-VIC dự đoán token VQGAN và dùng SFT để sửa feature. AFP-GIC dùng prior hợp nhất từ AdaCode, encoder nhìn prior gốc còn decoder ước lượng từ latent. Cả hai không được dùng prior từ ảnh gốc tại decoder. Trong repo chỉ có pilot, không kết luận tổng quát về toàn bộ kiến trúc.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://doi.org/10.1109/ACCESS.2024.3522238](https://doi.org/10.1109/ACCESS.2024.3522238) · [https://arxiv.org/html/2605.16817v2](https://arxiv.org/html/2605.16817v2)

## 20. CRDR có hai nút điều khiển

q thay representation và rate. Beta điều chỉnh realism trong generator qua Fourier embedding và các residual block. Cùng một latent có thể decode với nhiều beta, nhưng beta không bổ sung thông tin từ ảnh gốc. HRRGAN dùng reconstruction ở rate cao hơn làm đối chiếu khi train.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2405.16817v1](https://arxiv.org/html/2405.16817v1)

## 21. Cool-Chic chuyển chi phí sang encoder

Cool-Chic tối ưu latent và mạng riêng cho từng ảnh. Bitstream mang cả latent và tham số mạng lượng tử. Decoder có thể rất nhỏ nhưng encoder cần nhiều thời gian tối ưu. Phiên bản repo pin tag clic2025 khác paper gốc 2023, không gán MAC của paper gốc cho binary mới.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/pdf/2212.05458](https://arxiv.org/pdf/2212.05458) · [https://github.com/Orange-OpenSource/Cool-Chic/tree/clic2025](https://github.com/Orange-OpenSource/Cool-Chic/tree/clic2025)

## 22. VTM-SCC phục vụ ảnh chữ và màn hình

VVC dùng chia khối, dự đoán intra và residual. Các công cụ screen content khai thác cấu trúc lặp, vùng phẳng và bảng màu. Wrapper trong repo có fallback flags nếu binary từ chối. Do đó log flags thực phải đi cùng điểm kết quả.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://doi.org/10.1109/TCSVT.2021.3101953](https://doi.org/10.1109/TCSVT.2021.3101953)

## 23. S1 và SD-Turbo là các nhánh nội bộ

S1 là thiết kế đọc từ code, không có paper S1 độc lập. Nó nén latent DC-AE với hyperprior, gain và dither. Refiner SD-Turbo là cách repo ghép img2img vào reconstruction MS-ILLM. ADD gốc tăng tốc generation, không định nghĩa codec. Mọi trọng số refiner phải tính vào gói cuối.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/html/2311.17042v1](https://arxiv.org/html/2311.17042v1) · [D:/AwesomeCV/ratflow-codec/src/clic27/codec/latent_codec.py](D:/AwesomeCV/ratflow-codec/src/clic27/codec/latent_codec.py)

## 24. DiT-IC có lợi thế trong bake-off rate thấp

Biểu đồ lấy snapshot bake-off ngày 13/10 tại 0.075 bpp. Q̂ âm chỉ thể hiện vị trí trên thang proxy, không phải chất lượng âm. DiT-IC có điểm cao nhất trong các codec đơn được hiển thị. Mix có thể chọn quality khác nhau cho từng ảnh.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 25. PULSE-L nâng Q̂ của mix ở rate cao hơn

Dùng số mới nhất của vòng tích hợp q0–q4, không lấy 1.4912 của vòng pilot trước đó. Mức tăng tại 0.3 là khoảng 0.0763 điểm Q̂. Không chuyển mức tăng này thành phần trăm chất lượng. Các điểm là bake-off trước thay decoder MS-ILLM deterministic, output mới cần chấm lại.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md)

## 26. PULSE-L chiếm 22 trong 30 lựa chọn

Đây là số ảnh, không phải tỷ trọng byte. Mix còn giữ MS-ILLM, VTM-SCC và DiT-IC nên chi phí load nhiều họ model vẫn tồn tại. Tổng stream trong bake-off là 3.231.292 byte, tương đương 0.299730 bpp. Header MS-ILLM sau audit tăng thêm 120 byte và cần chấm lại output.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md)

## 27. Chấm mù sơ bộ ủng hộ DiT-IC trên ảnh tự nhiên

90 crop 512x512 do một người chấm. DiT-IC thắng 62/90 tổng thể, 55/75 ảnh tự nhiên và 4/12 ảnh chữ/màn hình. Các nhóm hiển thị không bao gồm toàn bộ loại nội dung. Có 10 cặp lặp và độ nhất quán 7/10. Crop cùng ảnh phụ thuộc nhau nên không coi 90 cặp là 90 mẫu độc lập.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 28. Student DiT-IC đánh đổi chất lượng để tăng tốc

Thí nghiệm giảm kênh ở các tầng DC-AE resolution cao, train L1 và LPIPS rồi GAN. Q̂ giảm ở q2, q3 và q4. Số tốc độ lịch sử teacher 17.5 giây so student 11.4 giây trên 30 ảnh L40 chưa bao gồm đầy đủ cùng protocol cold-start. Quyết định cuối cần benchmark L4.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 29. Ghi PNG là nút thắt trong gói PULSE-L

Mỗi cấu hình đo một lượt trên L40. V2 tắt nén lại PNG đầu ra nhưng giữ pixel nguyên vẹn. Bitrate codec tính trên stream đầu vào, không phải kích thước PNG dựng lại. Tổng wall gồm unzip, khởi động, nạp model, đọc stream và ghi ảnh. Không ngoại suy thành tốc độ toàn mix hoặc L4.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md)

## 30. Phạm vi đã kiểm của gói PULSE-L

Gói ZIP chưa mang Python, Torch, NumPy, Pillow và CUDA. Extension build theo server Linux x86_64 Python 3.12. Đã dùng thư mục sạch và tiến trình mới, chặn TCP trong Python cùng offline flags, chưa cô lập mạng cấp hệ điều hành. Đây là bằng chứng deployment từng nhánh.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md)

## 31. Tính tất định cần kiểm ở nhiều tầng

CDF lệch có thể làm symbol entropy sai và ảnh hỏng. Tuy nhiên CDF/symbol khớp chưa đảm bảo pixel float khớp. MS-ILLM mới kiểm 5 ảnh trong allocation, deterministic đạt parity giữa tiến trình. Output có khác baseline nhẹ nên chưa chuyển điểm Q̂ cũ cho output mới.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md) · [https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md](https://github.com/microsoft/GenCodec/blob/main/PULSE/README.md)

## 32. Những kết luận hiện tại còn có giới hạn

Đây là slide chuyển từ kết quả sang kế hoạch. Có bằng chứng hợp lý để chọn hướng, nhưng chưa đủ gọi hệ thống đã sẵn sàng dự thi. Việc bổ sung checkpoint vào allocator dễ hơn đóng gói một decoder chạy độc lập. Chấm mù mới có một người và mix mới chưa qua chấm mù.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 33. Hướng A: chọn codec có tính chi phí load

Đây là giả thuyết nghiên cứu, chưa chạy. Nếu một codec chỉ được chọn cho một ảnh nhưng có chi phí load lớn, lựa chọn tốt theo Q̂ có thể làm wall time xấu đi. Mô hình thời gian thêm chi phí dùng chung theo họ decoder. Vẫn phải benchmark gói thật vì I/O và scheduling có thể phi tuyến.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 34. Ablation cho bộ chọn codec

Giữ checkpoint, ảnh và tập điểm RD cố định. Mỗi nhánh thêm một yếu tố để xác định nguyên nhân cải thiện. A3 chỉ triển khai khi đã có calibration và dữ liệu độc lập cho uncertainty. Kết quả chính phải là Pareto chất lượng và thời gian thực, không chỉ objective của solver.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 35. Hướng B: đưa PULSE-L xuống rate thấp

PULSE-L hiện không có lựa chọn vừa budget riêng ở 0.075 hoặc 0.15 trong lưới đã thử. Đề xuất fine-tune RD và rate conditioning trước khi distillation. Dùng teacher image-space vì latent DiT-IC và PULSE không tương thích. Vùng chữ/mặt ưu tiên độ trung thực, texture có thể nhận tín hiệu teacher.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 36. Ablation B trong hạn mức 30 GB/GPU

Đây là ngân sách đề xuất, không phải các run đã khởi chạy. Cùng số update, crop, dữ liệu và seed ở pilot. Báo GPU-giờ riêng vì mỗi nhánh có chi phí khác. Cache teacher offline để tránh nạp hai model nặng đồng thời. Nếu finalist có lợi, xác nhận bằng nhiều seed.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 37. Thứ tự thực hiện và điều kiện chuyển bước

Không dùng lịch cứng khi L4 còn là phụ thuộc bên ngoài. Bước một tạo baseline deployment đáng tin, bước hai xác nhận chất lượng, bước ba mới so A/B. Giữ mọi run có version, checksum và artifact tái lập. Nhánh refinement theo vùng xếp sau A/B để tránh phân tán tài nguyên.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 38. Nội dung cần trao đổi

Dành khoảng 10 phút thảo luận. Tóm tắt bằng chứng: DiT-IC có tín hiệu rate thấp, PULSE-L cải thiện mix ở rate cao hơn, deployment toàn mix còn thiếu. Các quyết định cần ý kiến là trọng tâm đóng góp khoa học, thiết kế đánh giá độc lập và nguồn lực L4.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md)

## 39. Tài liệu phương pháp

Slide tham khảo, không cần trình bày tuần tự. Liên kết đầy đủ nằm trong speaker notes và trong report.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://arxiv.org/pdf/1809.02736](https://arxiv.org/pdf/1809.02736) · [https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf](https://proceedings.mlr.press/v202/muckley23a/muckley23a.pdf) · [https://arxiv.org/html/2603.13162v1](https://arxiv.org/html/2603.13162v1) · [https://arxiv.org/html/2410.10733v1](https://arxiv.org/html/2410.10733v1) · [https://arxiv.org/html/2410.10629v1](https://arxiv.org/html/2410.10629v1) · [https://arxiv.org/html/2609.18602v1](https://arxiv.org/html/2609.18602v1) · [https://arxiv.org/html/2604.12525v1](https://arxiv.org/html/2604.12525v1)

## 40. Tài liệu và artifact bổ sung

Các URL cụ thể nằm trong speaker notes. Số liệu nội bộ có thể đối chiếu từ report tích hợp PULSE-L và audit stream MS-ILLM. Report là snapshot 15/10/2026. Hai slide cuối phục vụ tra cứu sau buổi thuyết trình.

Nguồn: [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_report_for_adviser.md) · [https://doi.org/10.1109/ACCESS.2024.3522238](https://doi.org/10.1109/ACCESS.2024.3522238) · [https://arxiv.org/html/2605.16817v2](https://arxiv.org/html/2605.16817v2) · [https://arxiv.org/html/2405.16817v1](https://arxiv.org/html/2405.16817v1) · [https://arxiv.org/pdf/2212.05458](https://arxiv.org/pdf/2212.05458) · [https://doi.org/10.1109/TCSVT.2021.3101953](https://doi.org/10.1109/TCSVT.2021.3101953) · [https://arxiv.org/html/2311.17042v1](https://arxiv.org/html/2311.17042v1) · [https://clic2027.compression.cc/tasks/](https://clic2027.compression.cc/tasks/) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_pulse_integration_2026-10-15.md) · [D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md](D:/AwesomeCV/ratflow-codec/reports/CLIC2027_msillm_stream_audit_2026-10-15.md)
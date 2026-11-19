# Các kiến trúc để phát triển tiếp từ MLIC++

Khảo sát ngày 13/11/2026, tập trung vào nén ảnh cảm nhận, khả năng ghép với checkpoint Mixed và chi phí giải mã. Kết quả gồm các bài CVPR/ICCV 2025–2026 và preprint 2026. Chưa chạy các kiến trúc trong danh sách này trên CLIC30 của dự án.

## Kiến trúc và khả năng thử nghiệm

| Nguồn | Cơ chế | Cách liên hệ với MLIC++ | Mã và trọng số đã kiểm |
| --- | --- | --- | --- |
| MixCompress, arXiv 07/2026 | Mixture-of-Experts, Mixture-of-Depths và biến đổi phụ thuộc điều kiện để giảm xung đột giữa các bitrate | Thử synthesis có thân chung và nhánh chuyên biệt theo rate. Cần đối chứng với ba decoder độc lập | Chưa xác minh được repo chính thức |
| RDVQ, CVPR 2026 | VQ khả vi, codebook dùng chung đa tỉ lệ, entropy model Masked Transformer | Làm đối chứng codec hoàn chỉnh trước. Sau đó thử codebook học được cho residual của MLIC++ | Repo có code train và đường đánh giá bitstream thật. Chưa tải và kiểm checkpoint |
| DLF, ICCV 2025 | Hai nhánh latent ngữ nghĩa và chi tiết, tương tác để giảm thông tin trùng | Thử lớp nền MLIC++ và nhánh chi tiết học được có entropy model điều kiện theo lớp nền | Đã xác minh repo chính thức. Chưa chạy |
| DiT-IC, CVPR 2026 | Diffusion Transformer một bước trong latent giảm độ phân giải 32 lần, conditioning bằng latent | Đối chứng generative độc lập và ứng viên teacher. Việc thay latent đầu vào bằng MLIC++ cần adapter và train | Repo có code train, compress và liên kết checkpoint |
| SAMIC, arXiv 05/2026 | Mamba quét theo nhóm ngữ nghĩa và giảm dư thừa latent bằng phép xấp xỉ hạng thấp lấy cảm hứng từ SVD | Có thể thử block Mamba trong synthesis. Thay encoder sẽ cần train lại codec và bitstream | Repo hiện chỉ thấy README |
| CADC, CVPR 2026; MRIDC, CVPR 2025 | Generative compression thích nghi theo nội dung hoặc vùng | Nguồn đối chứng cho giả thuyết giữ cấu trúc ở chữ và sinh chi tiết ở texture | Chưa kiểm đủ khả năng tái lập trong lượt khảo sát này |

Nguồn chính thức cho từng hàng

- [MixCompress](https://arxiv.org/html/2607.14334v1)
- [RDVQ](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Differentiable_Vector_Quantization_for_Rate-Distortion_Optimization_of_Generative_Image_Compression_CVPR_2026_paper.html), [repo tác giả](https://github.com/CVL-UESTC/RDVQ)
- [DLF](https://openaccess.thecvf.com/content/ICCV2025/html/Xue_DLF_Extreme_Image_Compression_with_Dual-generative_Latent_Fusion_ICCV_2025_paper.html), [repo tác giả](https://github.com/dlfcodec/Dual-generative-Latent-Fusion)
- [DiT-IC](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_DiT-IC_Aligned_Diffusion_Transformer_for_Efficient_Image_Compression_CVPR_2026_paper.html), [repo tác giả](https://github.com/Eric-qi/DiT-IC)
- [SAMIC](https://arxiv.org/abs/2605.04560), [repo tác giả](https://github.com/Jasmine-aiq/SAMIC)
- [CADC](https://arxiv.org/abs/2602.21591), [MRIDC](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.pdf)

## Hai prototype đáng thử

### Synthesis có nhánh chuyên biệt theo bitrate

Động cơ xuất phát từ số liệu nội bộ. Mixed tốt nhất ở 0,075 bpp trong các decoder đã so, nhưng Single hoặc Perceptual tốt hơn tại 0,15 và 0,30 bpp. Xung đột giữa các mức lượng tử là một giả thuyết giải thích hiện tượng này, cần đo bằng đối chứng.

Giữ encoder và entropy model, dùng một phần synthesis chung cùng các adapter chuyên biệt. Router chỉ đọc latent đã giải mã và factor đã truyền trong stream. Nếu chọn nhánh dựa trên thông tin decoder chưa có thì cần truyền lựa chọn và tính số byte đó. Đo cả bộ nhớ của toàn bộ trọng số lẫn chi phí điều phối trên GPU.

So với Mixed, Single/Perceptual theo từng rate, adapter cố định theo factor và router học được. Mục tiêu đầu tiên là giữ lợi thế bitrate thấp đồng thời phục hồi chất lượng ở bitrate cao. MixCompress đã nghiên cứu MoE cho variable-rate, nên tính mới của việc thêm router có mức tin cậy thấp.

### Lớp nền MLIC++ và token hiệu chỉnh học được

Encoder tính residual so với lớp nền đã giải mã, lượng tử residual bằng codebook học được và entropy-code token dựa trên context lớp nền. Decoder kết hợp latent nền và token để phục hồi ảnh. Cả token, vị trí, thông tin lựa chọn và overhead đều phải tính vào ngân sách.

Phép thử DCT trước đây cho thấy việc rút byte khỏi lớp nền có thể gây hại nhiều hơn lợi ích của token. Vì vậy cần so với MLIC++ dùng toàn bộ ngân sách, lớp nền giảm ngân sách và nhánh hiệu chỉnh dùng cùng tổng byte. Đo trên tập ảnh giữ riêng trước khi chấm CLIC30.

Câu hỏi nghiên cứu có thể kiểm là khi nào một byte nên cấp cho lớp nền hay lớp hiệu chỉnh để tăng chất lượng cảm nhận. DLF, RDVQ và các phương pháp scalable compression là các nguồn gần cần đối chiếu. Độ tin cậy về tính mới hiện thấp, vì chưa hoàn thành khảo sát toàn bộ rate allocation và residual coding.

## Thứ tự thực hiện đề xuất

1. Tái lập RDVQ hoặc DiT-IC để có đối chứng ngoài họ MLIC++ ở bitrate thực tương đương.
2. Thử adapter synthesis chuyên biệt theo rate vì tái sử dụng được latent, decoder và bộ chấm hiện có.
3. Thử lớp token hiệu chỉnh nếu mục tiêu là phát triển thuật toán nén có phần truyền thông tin mới.
4. Chỉ mở rộng dữ liệu và lịch train khi kết quả vượt đối chứng cùng ngân sách và có tín hiệu chấm mù.

Paper Search CLI trả errors rỗng cho truy vấn arXiv, nhưng các kết quả đầu thiên về bài cũ và ít liên quan. Danh sách trên được xác minh thêm qua trang proceedings, arXiv và repo tác giả. Một số trang HTML của CVF bị lỗi truy cập, nên đối chiếu qua bản PDF hoặc repo chính thức. Đây là khảo sát định hướng, chưa đủ để tuyên bố một khoảng trống nghiên cứu hoàn toàn mới.

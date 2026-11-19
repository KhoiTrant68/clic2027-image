# Phạm vi nghiên cứu đóng góp mới trên MLIC++

Ngày lập: 04/11/2026. Trạng thái: phạm vi đã được tác giả xác nhận bằng “cứ làm đi”; review-form note được bỏ qua qua câu trả lời “skip và tiếp tục”. Chế độ `deep-research/full`; người dùng yêu cầu hệ thống đề xuất ý tưởng. Nội dung được soạn với hỗ trợ AI.

## Mục tiêu và bằng chứng xuất phát

Dùng MLIC++ làm baseline để phát triển một đóng góp ở cấp mô hình hoặc phương pháp học, phục vụ CLIC 2027 và có thể đánh giá như một công trình nghiên cứu độc lập với thứ hạng thi. Chưa thể bảo đảm tính mới, khả năng xuất bản hoặc thứ hạng CLIC.

[Thử nghiệm hiện có](../reports/CLIC2027_MLICpp_full30_2026-11-02.md) chỉ bao gồm bốn checkpoint MSE legacy trên 30 ảnh: MLIC++ có PSNR cao nhưng Q̂ thấp hơn mix hiện tại. Đã kiểm 120 stream giữa hai tiến trình trên cùng A30; chưa kiểm khác GPU. Đây là động cơ nghiên cứu, không phải bằng chứng một hướng cải tiến cụ thể sẽ thành công. MLIC++ vẫn được chọn làm baseline theo yêu cầu mới của người dùng; mix cũ là đối chứng bổ sung.

## Câu hỏi chính được đề xuất

Có thể cải tiến MLIC++ để nâng chất lượng cảm nhận ở bitrate thấp, đồng thời kiểm soát sai lệch cấu trúc và nội dung, với chi phí giải mã và khả năng tái lập đủ thực tế cho CLIC hay không?

Phạm vi tìm tài liệu dự kiến: công bố và preprint công khai đến **04/11/2026**, ưu tiên 2024–2026 và các công trình nền tảng trực tiếp liên quan. Ngày công bố, phiên bản, venue và phạm vi đọc sẽ được ghi riêng. Không dùng “mới nhất” để hàm ý đã tìm hết tài liệu.

## Các câu hỏi ứng viên do AI đề xuất để khảo sát

Các mục dưới là **hướng tìm kiếm**, chưa phải kết luận về khoảng trống hay đề xuất đã xác nhận tính mới.

1. **Chất lượng cảm nhận có ràng buộc nội dung:** điều kiện hóa hoặc điều khiển tái tạo từ latent MLIC++ thế nào để cải thiện cảm nhận ở bitrate thấp mà hạn chế sửa sai chữ, cạnh và chi tiết có cấu trúc?
2. **Phân bổ thông tin theo nội dung:** có thể điều chỉnh lượng tử hóa hoặc phân bổ latent theo vùng để đạt đánh đổi rate–distortion–perception tốt hơn baseline cùng ngân sách, sau khi tính đủ side information?
3. **Entropy và giải mã thực tế:** thay đổi mô hình ngữ cảnh hoặc cách biểu diễn xác suất có giúp cải thiện đánh đổi bitrate–độ trễ và độ ổn định giữa thiết bị, với tổn thất chất lượng được đo rõ ràng?

Ba câu hỏi đều kế thừa phạm vi: learned image compression; nguồn toàn cầu; tài liệu công khai đến 04/11/2026; mô hình dựa trên MLIC++ và các đối chứng liên quan; ảnh tự nhiên và nội dung có cấu trúc trong bộ đánh giá được xác định trước. Không có mở rộng phạm vi đã được tác giả phê duyệt.

## Đánh giá FINER sơ bộ

Điểm 1–5 là nhận định lập kế hoạch, không phải kết quả đánh giá tài liệu. Điểm Novel = 3 chỉ là giá trị tạm để soạn phạm vi; **chưa được xác minh và không được dùng để kết luận tính mới**.

| Câu hỏi | Feasible | Interesting | Novel, tạm | Ethical | Relevant | Trung bình, tạm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Cảm nhận và bảo toàn nội dung | 3 | 5 | 3 | 4 | 5 | 4,0 |
| 2. Phân bổ thông tin theo nội dung | 3 | 4 | 3 | 4 | 5 | 3,8 |
| 3. Entropy và giải mã thực tế | 3 | 4 | 3 | 5 | 4 | 3,8 |

Khả thi chỉ chấm 3 vì chưa biết ngân sách GPU-giờ hiện tại, chưa tái lập training và chưa kiểm giấy phép dữ liệu/trọng số cho mọi hướng. Hướng 1 được ưu tiên khảo sát vì liên hệ trực tiếp với chênh lệch pixel fidelity và Q̂ đã quan sát; không có nghĩa hướng này mới hơn hai hướng còn lại. Hướng có tái tạo sinh ảnh phải đánh giá nguy cơ thay đổi nội dung. Không tự khởi chạy training, thuê GPU hoặc thu thập dữ liệu người tham gia trong giai đoạn nghiên cứu tài liệu.

## Thiết kế đánh giá dự kiến

- Phân tích định lượng và thực nghiệm có đối chứng; xác định đóng góp cơ chế, không coi việc cộng thêm loss hoặc thay module là đủ mới.
- Kiểm lại MLIC++ chính thức, bao gồm quan hệ giữa checkpoint legacy hiện dùng và phiên bản mới hơn. Đối chứng trực tiếp phải cùng mục tiêu học và ngân sách huấn luyện hợp lý; thêm các phương pháp gần nhất sau khi xác minh tài liệu.
- Tách tập training, validation và test. Bộ 30 ảnh cũ dùng chẩn đoán, không đồng thời dùng để chọn ý tưởng rồi chứng minh khả năng khái quát. Chọn tập ảnh công khai và quyền sử dụng sau khảo sát.
- Đo byte stream thật, đủ header/side information; nhiều điểm rate phủ các mục tiêu hiện dùng 0,075 / 0,15 / 0,30 bpp. Đây là mục tiêu nội bộ đang dùng, chưa phải xác nhận lại quy định CLIC 2027.
- Báo cáo PSNR, MS-SSIM, LPIPS, DISTS, Q̂ và lỗi nội dung theo thiết kế; không dùng Q̂ làm bằng chứng duy nhất. Đo latency đầu-cuối, bộ nhớ và kiểm giải mã khác tiến trình/thiết bị.
- Ablation từng thành phần và đối chứng với thay đổi đơn giản hơn, cùng chi phí; nhiều seed khi khả thi, khoảng tin cậy bootstrap theo ảnh. BD-rate chỉ dùng khi đường đo và miền chồng lấp phù hợp.
- Human evaluation là yêu cầu thiết kế cần cân nhắc cho khẳng định preference; chưa có kế hoạch tuyển người hoặc thu thập dữ liệu trong bước này. Các yêu cầu hành chính, nếu triển khai, cần xác định riêng theo cơ quan chịu trách nhiệm.

## Kiểm tra phản biện trước khi tìm tài liệu

1. **Tính mới chưa được chứng minh:** tìm cả các nghiên cứu phủ định khoảng trống dự kiến, gồm biến thể trực tiếp của MLIC++, tối ưu cảm nhận, bảo toàn nội dung, variable-rate, latent adaptation và entropy nhanh/ổn định. Không đặt tên phương pháp mới trước khi đối chiếu prior art.
2. **Baseline có thể chưa đại diện:** cải thiện so với checkpoint MSE legacy chưa đủ để chứng minh vượt baseline mạnh được tối ưu cảm nhận hoặc thế hệ mới.
3. **Đích thi và paper khác nhau:** top 1 phụ thuộc đối thủ và luật thi; paper cần câu hỏi, cơ chế và bằng chứng khái quát. Hai đích được theo dõi riêng.
4. **Nguy cơ chọn theo proxy:** tối ưu đúng Q̂ rồi báo tăng Q̂ không xác lập tăng preference thực tế. Cần metric ngoài mục tiêu học và đánh giá độc lập phù hợp.
5. **Tài nguyên chưa được xác nhận hiện thời:** A30/L40 là phần cứng đã dùng, không mặc định hiện vẫn sẵn có. Kế hoạch thử nghiệm sau khảo sát sẽ chia pilot nhỏ và giai đoạn mở rộng có tiêu chí dừng.

Không có mâu thuẫn khiến phạm vi không thể nghiên cứu; các điểm trên là điều kiện phải giải quyết trước khi chọn đóng góp cuối. Chưa có kết luận về prior art.

## Đầu ra sau khi xác nhận phạm vi

Bảng các nghiên cứu liên quan có link nguồn gốc, ngày/venue và mức độ đã đọc; phân biệt kết quả tác giả báo cáo với suy luận của người khảo sát. Đối chiếu 3–5 hướng đóng góp theo công trình gần nhất, tính mới có điều kiện, rủi ro, nguồn lực và ablation. Đề xuất một hướng ưu tiên cùng pilot có tiêu chí tiếp tục/dừng. Kiểm tra lại quy định CLIC hiện hành từ nguồn chính thức trước khi đưa vào ràng buộc triển khai.

Tài liệu này lưu phạm vi Phase 1, không phải tuyên bố novelty. Sau xác nhận đã thực hiện khảo sát; xem [báo cáo đề xuất](../reports/MLICpp_research_proposal_2026-11-04.md) và [nhật ký tìm kiếm](../reports/research_mlicpp_2026-11-04/search_and_synthesis.md). Các hướng sơ bộ trong tài liệu này được xem lại theo prior art trong báo cáo, không tự trở thành đóng góp cuối cùng.

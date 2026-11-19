# So sánh bốn decoder MLIC++ trên CLIC30

Bốn decoder dùng cùng ảnh gốc và cùng bitstream cho từng ảnh tại mỗi ngân sách. Phép thử giữ cố định encoder, phân bổ byte và quá trình giải mã entropy; chỉ thay mạng tái tạo. CLIC30 là tập validation đã dùng trong dự án, gồm 30 ảnh và 86.245.376 pixel.

| Ngân sách (bpp) | bpp thực | Decoder | PSNR TB ↑ | LPIPS-Alex ↓ | DISTS ↓ |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 0.075 | 0.074396 | Gốc | **28.2089** | 0.28712 | 0.19525 |
| 0.075 | 0.074396 | Perceptual | 27.7636 | 0.21398 | 0.17638 |
| 0.075 | 0.074396 | Single | 27.6856 | 0.21079 | 0.17452 |
| 0.075 | 0.074396 | Mixed | 27.8037 | **0.18293** | **0.15771** |
| 0.15 | 0.149973 | Gốc | **31.3964** | 0.19855 | 0.14108 |
| 0.15 | 0.149973 | Perceptual | 30.7279 | 0.11873 | **0.10545** |
| 0.15 | 0.149973 | Single | 30.5950 | **0.11639** | 0.10592 |
| 0.15 | 0.149973 | Mixed | 30.4035 | 0.12111 | 0.11639 |
| 0.30 | 0.298753 | Gốc | **33.0945** | 0.18046 | 0.13686 |
| 0.30 | 0.298753 | Perceptual | 32.4162 | 0.10018 | 0.09615 |
| 0.30 | 0.298753 | Single | 32.2442 | **0.09838** | **0.09580** |
| 0.30 | 0.298753 | Mixed | 31.7222 | 0.10778 | 0.11104 |

PSNR là trung bình PSNR theo ảnh, không phải PSNR gộp theo pixel. bpp thực là bitrate của gói stream dùng chung, gồm chi phí đóng gói; checkpoint không được tính vào gói thí nghiệm.

**Kết quả.** Ở 0.075 bpp, Mixed giảm LPIPS-Alex khoảng 14,5% và DISTS khoảng 10,6% so với Perceptual. Ở 0.15 bpp, Single có LPIPS-Alex thấp nhất còn Perceptual có DISTS thấp nhất. Ở 0.30 bpp, Single đứng đầu hai chỉ số cảm nhận, với chênh lệch nhỏ so với Perceptual. Decoder gốc giữ PSNR cao nhất cả ba mức. Các chỉ số cho thấy lợi thế của từng decoder phụ thuộc ngân sách và tiêu chí; chưa có phiếu người chấm để kết luận về sở thích thị giác.

**Kiểm tra và giới hạn.** Có 81 tổ hợp ảnh và hệ số lượng tử khác nhau trong ba ngân sách. Lượt chạy xác nhận cả 81 ảnh giải mã gốc trùng từng pixel với bản đã lưu, rồi đo lại đủ 81 cấu hình cho mỗi checkpoint Single và Mixed. Vòng chấm mới tách theo ngân sách và cặp decoder. Kết quả lịch sử của VTM-SCC, PULSE, AEIC và các đối chứng khác được giữ riêng vì chưa xác minh đủ bộ ảnh tái tạo tương ứng. Chưa đo Q̂ cho phép thử này; chưa thể suy rộng sang tập test của challenge hoặc xếp hạng theo người chấm.

Điểm từng ảnh và SHA-256 của bitstream nằm trong `per_image.json`; bảng tổng hợp của hai checkpoint mới nằm trong `comparison.json`, tại thư mục server `clic2027-artifacts/mlicpp/multirate-20261109/clic30-frozen-budget-20261111`.

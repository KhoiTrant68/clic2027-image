# Kế hoạch và kiểm tra hình — 20/10/2026

Phạm vi: trực quan hóa số đo đã lưu trong báo cáo so sánh, bằng tiếng Việt; không chạy inference, không dùng GPU, không triển khai website công khai.

Áp dụng `visual-paper` và `math-figures` trong phạm vi hình khoa học: nguồn có hash, công thức có nguồn, phân biệt số đo với minh họa, kiểm tra độc lập và xem ảnh render. Đây không phải một visual paper toán học có TeX/Lean, proof graph hay chứng nhận audit toán học. Yêu cầu của người dùng cho phép tạo hình và chèn vào báo cáo so sánh; các cổng xuất bản website không áp dụng.

## Nguồn và cách tái tạo

- `results/2026-10-20_metrics_comparison/full30_budget_metrics.csv`
- `results/2026-10-20_metrics_comparison/aeic_fixed_metrics.csv`
- `results/2026-10-20_metrics_comparison/pilot6_budget_metrics.csv`
- `results/2026-10-20_metrics_comparison/rate_floors.json`
- `results/2026-10-15_new_codecs/qhat_v0.json`
- `results/2026-10-15_new_codecs/pilot_scores/summary.json`
- `results/2026-10-06_sessiona/bakeoff/metrics.csv`

Bối cảnh diễn giải: `reports/CLIC2027_metrics_comparison_2026-10-20.md`. Manifest pin các nguồn dữ liệu trực tiếp, không pin bản report chứa chính các hình.

`manifest.json` ghi SHA-256 và số dòng nguồn cho từng hình. Hash văn bản chuẩn hóa CRLF thành LF trước khi tính (giữ BOM nếu có), để khớp chính sách xuống dòng của Git trên Windows/Linux. Script chỉ đọc các nguồn trên; mọi điểm đo lấy trực tiếp từ CSV. Chạy `python reports/figures/metrics_2026-10-20/make_figures.py` trong môi trường có numpy và matplotlib.

## Các hình

1. **Q̂ theo ngân sách, 30 ảnh.** Câu hỏi: codec nào có Q̂ cao ở từng budget? Màu tuần tự dùng chung thang cho tất cả ô; chữ ghi Q̂ và bpp thực. Ô thiếu không được gán 0. S1 thiếu metric được ghi riêng. Không có thang “đạt” hay phần trăm chất lượng.
2. **Q̂ và PSNR theo bpp thực, 30 ảnh.** Câu hỏi: xếp hạng thay đổi ra sao khi xét cảm nhận và sai số pixel? Một tập codec tiêu biểu, đường nối chỉ giúp theo dõi các điểm đã đo, không khẳng định đường RD liên tục. AEIC là hai điểm riêng; không nội suy. Hai mix trùng nhau ở rate cao không vẽ đè, dùng mix mới nhất.
3–5. **Sáu metric tại từng ngân sách.** Câu hỏi: Q̂ cao có đồng nghĩa mọi metric đều tốt? Sáu dot plot dùng giá trị gốc và ghi chiều tốt, không chuẩn hóa về một thang tổng hợp hay dùng diện tích radar. Cùng thứ tự phương pháp trong sáu panel, sắp theo Q̂ giảm dần. Tên hàng kèm bpp thực. Trục x có thể cắt ngắn: đây là vị trí điểm, không phải chiều dài cột tính từ 0.
6. **Pilot 6 ảnh.** Câu hỏi: AFP/DC-VIC/PULSE đóng góp thế nào trong đúng pilot? Tách khỏi 30 ảnh; dùng cùng thang màu Q̂ với hình 1 để màu không đổi ý nghĩa nhưng không suy ra xếp hạng giữa hai tập. PULSE base không có allocation hợp lệ, ghi rõ dưới hình.
7. **Diễn giải ΔQ̂.** Câu hỏi: chênh lệch điểm liên hệ xác suất mô hình như thế nào? Đường sigmoid của cùng một cặp ảnh, bias = 0 để loại thiên lệch vị trí. Là minh họa công thức, không phải số phiếu thực, không áp dụng sigmoid lên chênh lệch trung bình toàn tập. Đường có bias fit được ghi riêng để kiểm tra hệ số.

## Quy ước chung

- Nguồn số đo là snapshot cũ, không có confidence interval hay thử lặp đủ để vẽ error bar.
- Không trộn pilot với full30, không ngoại suy checkpoint chưa chạy.
- CSV dùng bpp từ stream thật; các giới hạn về audit decoder giữ nguyên trong báo cáo.
- Hình được xuất PNG để đọc nhanh, SVG để chỉnh sửa/phóng to, PDF nhiều trang để trình chiếu.
- Kiểm tra trước khi giao: nguồn, số dòng, giá trị hữu hạn, số ảnh, budget; render không lỗi glyph; xem cả bảy ảnh; review độc lập script và caption.

## Nhật ký

- 20/10: Chọn hình tĩnh theo yêu cầu đồ thị của người dùng; không dựng website hay proof graph.
- Sau review độc lập: sửa `PULSE` thành `PULSE-L` ở các mix full30; giải thích `PULSE(-L)` của pilot là tập ứng viên chứa cả hai biến thể.
- Sau review độc lập: đổi tiêu đề hình 02 thành “Chất lượng theo bitrate thực: Q̂ và PSNR”; bổ sung pin nguồn và assert cho S1 28/30 cùng các ô không vừa ngân sách.
- Đã xem đủ 7 PNG; reviewer độc lập cũng kiểm tra đủ 7 hình, script, số đo và caption. Kiểm lại sau sửa: không còn lỗi cần sửa, nhãn không bị cắt, hash nguồn khớp.
- Kiểm tự động: 7 bộ nguồn, 30 liên kết nội bộ và PDF 7 trang có text trích xuất được đều đạt. Không có warning glyph khi render.
- Trạng thái: đã hoàn tất kiểm tra và tích hợp report; chờ phản hồi của người dùng. Review này không phải blind audit khác họ model hoặc chứng nhận toán học theo quy trình xuất bản visual-paper đầy đủ.

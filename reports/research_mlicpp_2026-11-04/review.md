# Rà soát đề xuất và điều chỉnh

Ngày: 04/11/2026. Đây là rà soát báo cáo định hướng bởi cùng hệ thống AI, không phải peer review của venue, không chứng nhận tính mới hoặc khả năng xuất bản.

## Các checkpoint

1. **Phase 1 / checkpoint 1:** phạm vi nghiên cứu có thể triển khai; novelty chưa xác minh, chưa biết training budget hiện hành. RQ Brief đã nêu hai đích thi/paper riêng, baseline legacy và đánh giá proxy. User xác nhận và trả lời skip review-form note trước khi tìm tài liệu.
2. **Phase 3 / checkpoint 2 — REVISE, đã xử lý:** ý tưởng sơ bộ sparse structural innovation + counterfactual selection trùng nhiều thành phần của AGDL, GCR-C và ACV-Gate. Chưa chốt thành đóng góp. Đưa về nhánh dự phòng và bổ sung conventional residual/allocation-to-base controls; chuyển khuyến nghị chính thành pilot kiểm tra lỗi cấu trúc ở đuôi phân phối.
3. **Phase 5 / checkpoint 3 — PASS cho lập kế hoạch pilot:** hai nhánh đọc rà soát report và catalog; không phát hiện blocker integrity. Chưa đủ bằng chứng chốt đóng góp mô hình mới. Phản biện chính là adapter còn chung chung và CVaR có thể chỉ là thay objective; đã ghi nhận và sửa.

## Nhận xét và cách xử lý, một vòng chỉnh sửa

| Nhận xét | Thay đổi trong báo cáo cuối |
| --- | --- |
| Adapter chưa cụ thể | Thêm bounded feature residual, head nhận cả proposed change, rate và decoded feature; gate triển khai sau entropy decoder. Ghi rõ generic gating không được nhận là mới. |
| Structural loss chưa được chốt; OCR không khả vi | Pilot chọn gradient surrogate trên vùng cố định, chuẩn hóa từ training; CER có transcript dành cho đánh giá độc lập, không cộng trực tiếp vào training loss. |
| CVaR cần tối ưu threshold | Ghi τ learned theo rate, đơn vị lấy mẫu ảnh rồi vùng, alpha chốt trên validation; không gọi empirical CVaR là bảo đảm population/per-image. |
| Thiếu kiểm chứng tiền đề tail failure | P0 phải đo prevalence/severity trên holdout; nếu không có vấn đề phù hợp thì dừng ưu tiên nhánh này. |
| Audit và low-rate training có thời lượng khác nhau | Tách P0a audit/profile 1–2 ngày với P0b phủ rate thấp chưa ước lượng. Pilot bắt đầu ở rate đã tái lập. |
| Điều kiện đi tiếp chung chung | Thêm ngưỡng pilot đề xuất 2%/10%/15%, báo CI theo ảnh; ghi rõ đây là ngưỡng quản lý thử nghiệm, chưa phải số cải thiện thực tế hoặc luật thi. |
| Closest baseline khác training budget | Tách mechanism attribution cùng checkpoint/recipe với external competitiveness; PerQ multirate không tự là matched-training control, PICO-style loss không phải full PICO reproduction. |
| Semantic/spectral loss có prior art | Bổ sung SSCL control khi dùng các guidance này. |

## Integrity và ranh giới đánh giá

Đã kiểm nguồn tồn tại và claims trọng tâm từ primary sources, gồm MLIC README/MLICv2, PerQ methods/Table4/AppB, ConHa, CADC, PICO, AGDL, GCR-C, nguồn proceedings và CLIC Tasks/FAQ; catalog ghi phạm vi đọc để tránh gọi abstract-only là fulltext audit. Citation coverage cho các thông tin cốt lõi vượt nửa tập 22 nguồn; không xác nhận proof hay kết quả chưa tái lập. Nguồn chỉ abstract không bị gán điểm yếu suy đoán.

Không phát hiện fabrication hoặc plagiarism trong phạm vi đã rà. Có AI disclosure và attribution cho các cơ chế đã có; kết quả local được tách khỏi author-reported benchmarks và AI-generated hypotheses. Chưa làm retraction/COI database audit; trạng thái đó chưa được xác minh. Không thực hiện human study trong lượt này, không suy ra approval hoặc institutional readiness.

Rủi ro còn lại được giữ công khai: tính mới chưa bảo đảm, adapter có thể không hơn loss-only, surrogate có thể bị khai thác, không có hardware budget training đã profile, chưa có human preference evidence. Đây là giới hạn của một kế hoạch nghiên cứu, không được đổi thành claim đã giải quyết.

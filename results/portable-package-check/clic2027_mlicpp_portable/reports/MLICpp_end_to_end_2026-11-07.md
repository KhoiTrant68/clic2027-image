# Trạng thái thí nghiệm MLIC++ toàn bộ backbone

Cập nhật khoảng **07:22 ngày 07/11/2026 (UTC+7)**. Đây là trạng thái khởi chạy, chưa phải báo cáo kết quả rate–distortion.

## Đã hoàn thành

- Triển khai hai nhánh fine-tune: MLIC++ thông thường và MLIC++ học thêm bước lượng tử theo kênh.
- Audit dữ liệu train/development; tải đủ 32 ảnh test mới, kiểm CRC và SHA-256, kiểm không trùng nguồn trong pilot224.
- CPU smoke qua: giữ nguyên forward/bitstream/pixel tại bước lượng tử bằng 1; gradient hữu hạn; optimizer cập nhật được backbone và bước lượng tử.
- Cài pipeline đo byte thực, giải mã trong tiến trình mới, PSNR/LPIPS-Alex/DISTS và thời gian encode/decode.
- Chuẩn bị phép so sánh nội suy trong khoảng bitrate chung và bước tạo bộ chấm mù có điều kiện bitrate.
- Compile Python trên cả máy local và server qua; kiểm tra hàm so sánh với hai đường cong giống nhau cho delta 0, và hai đường cong không giao nhau không bị ngoại suy.

## Đang chờ

Session server: `mlic-e2e-campaign-20261107`.

Thư mục kết quả:

```text
/home/jovyan/clic2027-artifacts/mlicpp/end-to-end-20261107/div2k160-fullrd-vs-learnedsteps-3rates-2seeds
```

Trạng thái quan sát được: **waiting_for_gpu**, giai đoạn tiếp theo `train-baseline-q1-seed9041`. A30 còn **766 MiB**, trong khi yêu cầu **18.432 MiB** trống. Chưa có lượt train GPU nào bắt đầu trong campaign này.

Campaign tự kiểm tra tài nguyên mỗi 30 giây, chờ tối đa 6 giờ cho giai đoạn này (khoảng 13:22 cùng ngày). Ngân sách tổng tối đa 24 giờ kể từ khởi chạy (khoảng 07:20 ngày 08/11). Đây là giới hạn dừng, không phải dự đoán hoàn thành. Không dừng hoặc thay đổi job khác.

## Kết quả chưa có

Chưa có checkpoint fine-tune hoàn chỉnh, delta chất lượng, kiểm fresh decode sau fine-tune hoặc phiếu chấm của người dùng. Không thể kết luận cải thiện MLIC++, đạt điều kiện challenge, hay có đóng góp đủ mới cho paper từ smoke test này.

Sau khi GPU đủ bộ nhớ, pipeline sẽ lần lượt chạy 12 lượt train và 15 cấu hình đánh giá. Bộ chấm mù chỉ được tạo khi sai lệch bitrate trung bình không quá 5%; vẫn cần người thật chấm. Nếu lỗi hoặc hết ngân sách, pipeline ghi trạng thái dừng và giữ kết quả đã có.

Xem [protocol cố định](../plans/MLICpp_end_to_end_2026-11-07.md) và [bằng chứng CPU smoke](../results/2026-11-07_mlicpp_end_to_end/smoke.json).

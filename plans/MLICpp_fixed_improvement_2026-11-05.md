# Pilot cải thiện fixed correction: thời gian học và tìm hệ số

## Quyết định trước khi chạy

Tiếp tục yêu cầu cải thiện/so sánh bằng một pilot kỹ thuật giới hạn. **FEASIBILITY_PILOT_ONLY**; không nhận tăng thời gian học hoặc encoder search là đóng góp mới. Conditional ở recipe cũ đã không vượt fixed; không mở lại claim đó bằng cách tune theo test.

Giả thuyết: fixed basis chưa hội tụ ở 1.000 bước và/hoặc encoder chưa tìm được code tốt trong 40 bước. Đơn vị đánh giá là nguồn ảnh, cùng crop/base stream, cùng packet 46 byte. Fatal assumption: thêm train/search phải cải thiện metric sau hard quantization và giải mã thật; giảm training loss không đủ.

| Nhánh | Bước huấn luyện | Bước encoder search | Yếu tố thay đổi so mốc |
| --- | ---: | ---: | --- |
| `reference_s40` | 1.000, checkpoint cũ | 40 | Mốc fixed tốt nhất vòng trước |
| `reference_s160` | 1.000, cùng checkpoint | 160 | Chỉ search budget |
| `long_s40` | 5.000, học lại từ cùng init | 40 | Chỉ thời gian huấn luyện |
| `long_s160` | 5.000, cùng checkpoint | 160 | Tổ hợp hai yếu tố |
| `base` | Synthesis nền khóa | 0 | Tham chiếu không packet, ít byte hơn |

Giữ kiến trúc, 38.240 tham số, seed 9041, init, batch/order, Adam LR 1e−4, betas (0,9; 0,99), loss `10 MSE + LPIPS + 0,1 edge_rational`, quantizer và format. Học lại 5.000 bước vì checkpoint 1.000 cũ không lưu optimizer state. Kiểm template/train manifest và loss tại các bước chung trong prefix 1.000 để phát hiện thay đổi ngoài ý muốn; nếu khác, dừng diễn giải duration-only.

Search giữ proposal/zero initialization, LR 0,03 và best hard-quantized candidate của thuật toán cũ; snapshot ở 40 và 160 trên cùng trajectory. Bước 160 không thể có selection loss cao hơn snapshot 40, nhưng LPIPS/DISTS sau uint8 không được bảo đảm. Đây là encoder compute bổ sung; decoder và packet giữ nguyên. Ghi search time riêng, chưa coi là benchmark tổng encode/decode.

## Dữ liệu và phép kiểm

- Train 320 crop từ 160 nguồn cũ; development 32 validation cũ. Không dùng test vòng trước để chọn LR/loss/số bước.
- Test **32 nguồn mới**, permutation seed 1984, chỉ số `[322:354]`, một crop 256²/nguồn theo quy tắc crop cũ. Loại toàn bộ 322 nguồn đã được dùng đến hết pilot trước và smoke. Kiểm SHA/CRC, crop thật, template và base stream.
- Smoke riêng dùng đúng hai nguồn smoke cũ, train 4 bước và search 2/4 bước. So snapshot ngắn với `innovation_pilot.choose` để kiểm refactor không đổi thuật toán. Không chọn cấu hình bằng smoke metrics.
- Primary contrast khóa trước: `long_s160 − reference_s40` trên LPIPS và DISTS, cùng từng packet byte. Hai contrast một yếu tố và PSNR/MS-SSIM/edge là phân tích hỗ trợ. Báo cả bốn nhánh, không chỉ winner.
- Bootstrap ghép cặp 10.000 lần, seed 9041, CI 95% pointwise thăm dò. Một seed/một rate, không điều chỉnh multiple comparisons; không nhận significance tổng quát hoặc human preference.
- Fresh process đọc stream/model assets, không dùng target hay feature theo ảnh; yêu cầu 160/160 uint8 hashes. Không trộn crop vào benchmark 30 ảnh.

## Reality gate và tài nguyên

Đường train/cache/stream/metric đã chạy được ở pilot trước: pass cho access và measurement kỹ thuật. Lợi ích của hai can thiệp còn unknown; pilot này là phép kiểm hữu hạn, không phải full run cho claim khoa học. Closest-competitor novelty không áp dụng cho claim kỹ thuật tăng train/search; không tạo một tên phương pháp mới.

Preflight đã kiểm checkpoint, nguồn train, 260 GB ổ trống và L40 GPU 0 còn headroom; GPU khác có job đang chạy. Dùng runtime/cuDNN v7/NVRTC, FP32, guard VRAM và JSON status/step/infer logs như run tham chiếu. Không dùng API trả phí. Dự kiến 15–30 phút sau smoke và tải dữ liệu; **ceiling 2 giờ, một GPU**, thêm khoảng 2 GB artifact. ETA phải cập nhật theo smoke thực tế, theo giờ Việt Nam.

Acceptance: đầy đủ 160 output thật, cùng packet size, prefix/reference giữ đúng; primary LPIPS giảm với CI không chứa 0 và DISTS không có suy giảm rõ. Nếu chỉ selection loss giảm, hoặc quality không hơn mốc, giữ kết quả âm và không mở rộng cấu hình. Không tăng bước tiếp theo hoặc tune loss sau khi đã mở test này. Không claim thắng residual/base-more-bits tối ưu, mới khoa học, đủ CVPR hay top CLIC.

Owner: phiên làm việc hiện tại. Run mới, không ghi đè run cũ. Code/config: `experiments/mlicpp/fixed_improvement.py`, `fixed_improvement_q2.json`; analysis khóa trước trong `analyze_fixed_improvement.py`.

## Preflight và launch

Smoke `smoke-fixed-training-search-2x2` hoàn tất, gồm train 4 bước, snapshot search 2/4, so snapshot ngắn với `choose` gốc, source/crop audit và fresh decode. Test mới tải đủ 32/32 nguồn kèm CRC/SHA. Config diff có chủ đích là số bước train/search, test split và phạm vi chỉ fixed; các đường checkpoint/data đã kiểm tồn tại. Monitoring dùng JSONL/status file có sẵn, không cần dịch vụ tracker ngoài.

Run chính khởi chạy khoảng **18:31 ngày 05/11/2026, giờ Việt Nam**, tên `div2k-q2-fixed-train1000-5000-search40-160-new32`. ETA ban đầu **18:45–19:00**, hard ceiling hai giờ. Smoke và run chính là hai output độc lập. Chỉ kết luận thành công sau khi kiểm status, số liệu và fresh hashes.

## Kết quả

Hoàn tất 766,34 giây, train đủ 5.000 bước, 160/160 fresh matches. Primary LPIPS giảm 8,63%, CI delta [−0,012623; −0,004123]; DISTS +0,000651, CI [−0,001641; +0,003208]; PSNR −0,0602 dB. Có cải thiện LPIPS, chưa chứng minh DISTS tương đương/không suy giảm. Không mở rộng cấu hình theo test đã đọc. [Báo cáo đầy đủ](../reports/MLICpp_fixed_improvement_2026-11-05.md).

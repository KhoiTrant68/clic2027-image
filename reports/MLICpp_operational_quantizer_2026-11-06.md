# MLIC++: tối ưu bước lượng tử hóa bằng RD thực

**Kết luận:** Tối ưu trực tiếp qua bitstream và ảnh giải mã chưa tạo ra cải thiện đủ rõ để mở rộng hướng chỉnh gain. Pilot hoàn tất trong 711,4 giây trên A30 GPU 2; **96/96 bitstream** giải mã ở tiến trình mới khớp SHA pixel. Không có thí nghiệm train dài được khởi chạy từ kết quả này.

## Thiết kế và thay đổi

Thay mục tiêu dự đoán độ nhạy trong pilot geometry bằng mục tiêu đo trực tiếp:

\[
J=\mathrm{bpp}_{\mathrm{file}}+0.0035\,\mathrm{MSE}_{\mathrm{RGB},255}.
\]

Mỗi candidate chạy toàn bộ encode/decode với hard quantization, entropy coding, context và LRP thật. MLIC++ MSE legacy q2 được đóng băng. Tối ưu 10 tham số bước lượng tử hóa theo nhóm bằng coordinate search; **không train neural backbone**. Không dùng teacher geometry, không ép thể tích cell bằng nhau. Điều này kiểm tra một khả năng sửa mục tiêu, chưa cô lập hoặc chứng minh nguyên nhân thất bại của geometry.

Đối chứng gồm identity (bước 1), scalar (một bước chung tối ưu trên 40 candidate) và slices (10 bước riêng, 2 vòng, 40 candidate). Tổng cộng 81 lần chấm trên tập train, tính cả identity dùng chung. Ngân sách số candidate bằng nhau không có nghĩa là thuật toán hoặc không gian tìm kiếm tương đương.

Tập train gồm 8 crop từ 8 nguồn; development validation gồm 8 nguồn khác đã có trong pilot trước. **Không đọc 16 ảnh test của geometry để chọn tham số.** Đây vẫn là development nhỏ, không phải holdout độc lập. Mọi nhánh có cùng header OPQ1 10 byte; các bước học được là model asset dùng chung, không có side information riêng theo ảnh. Smoke identity khớp payload và pixel của codec gốc.

## Kết quả chính trên validation

So sánh dưới đây ở factor 1, cùng lambda, **không cùng bitrate**. PSNR là trung bình PSNR từng ảnh; không so trực tiếp với PSNR gộp trong báo cáo geometry.

| Nhánh | Bpp thực | MSE RGB ↓ | PSNR trung bình ↑ | J ↓ |
|---|---:|---:|---:|---:|
| identity | 0,166443 | 63,090861 | 31,3113 | 0,387260885 |
| scalar | 0,159119 | 65,352448 | 31,1486 | 0,387852221 |
| slices | 0,162109 | 64,329220 | 31,2260 | 0,387261644 |

Slices giảm J **0,1523%** so với scalar. Chênh lệch tuyệt đối là −0,000590576, bootstrap paired 95% theo ảnh: **[−0,001551658; +0,000179063]**, 2.000 lần, seed 9041. Không đạt gate thăm dò giảm ít nhất 0,5% đã chốt. Khoảng chứa 0 không chứng minh tương đương.

So với identity, slices có J gần như bằng nhau và hơi cao hơn, khoảng **0,000196%**. Vì vậy chỉ so với scalar được tối ưu trên 8 ảnh train sẽ tạo ấn tượng tiến bộ không đúng mức. Trên train, J của identity/scalar/slices lần lượt là 0,317674517 / 0,317426803 / 0,317031164.

Scalar học được bước 1,0551766. Slices chỉ tăng bước của các nhóm 1, 2 và 9 lên 1,1274969; các nhóm còn lại giữ 1. Đây là thay đổi nhỏ, không phải một kiến trúc mới.

Tất cả bốn factor `{0.8, 1, 1.25, 1.6}` và số liệu từng ảnh được giữ trong artifact, không chọn riêng điểm thuận lợi để báo cáo. Chưa đánh giá LPIPS, DISTS, Q̂ hoặc human preference ở pilot này.

## Quyết định nghiên cứu

Hai phép thử hiện cho bằng chứng hạn chế nhưng nhất quán: distillation độ nhạy của riêng synthesis chưa thắng uniform, còn tìm gain bằng RD thực chỉ đem lại thay đổi rất nhỏ trong thiết lập này. **Không tiếp tục thêm MLP hoặc train dài chỉ để tìm gain tốt hơn.** Kết quả chưa bác bỏ lượng tử hóa theo block, tương tác giữa kênh, hay huấn luyện lại representation; các phương pháp đó chưa được thử.

Một hướng model đáng xem xét tiếp là transform lượng tử hóa theo block được dự đoán từ context đã giải mã, học đồng thời với entropy model và LRP. Khác biệt phải được kiểm chứng ở tương tác giữa các kênh và chi phí mã hóa thực, có đối chứng diagonal/gain mạnh. Không coi việc thêm ma trận hoặc công thức Jacobian tự nó là novelty. Cần kiểm tra prior art trước một đợt triển khai lớn.

Adaptive LVQ đã có tại [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.html). Post-training đa rate và reconstruction offsets đã được nghiên cứu trong [DCC 2024](https://arxiv.org/abs/2402.18930); [Soft then Hard, ICML 2021](https://proceedings.mlr.press/v139/guo21c.html) cũng đã xử lý mismatch của lượng tử hóa khi train/infer. Do đó, pilot hiện tại là đối chứng kỹ thuật để loại hướng yếu, chưa là đóng góp paper.

## Bằng chứng và giới hạn

Đã kiểm lại SHA của 96 stream sau khi tải về máy và tính lại bootstrap primary bằng NumPy. Fresh decode trên cùng A30 không chứng minh tính xác định giữa các dòng GPU. Chỉ có 8 ảnh validation, một checkpoint, một lambda và search ngắn; chưa có cơ sở khẳng định mọi cách tối ưu gain đều vô ích hoặc suy ra khả năng cạnh tranh CLIC.

- [Protocol trước khi chạy](../plans/MLICpp_operational_quantizer_2026-11-06.md)
- [Mã triển khai](../experiments/mlicpp/operational_quantizer.py)
- [Phân tích đầy đủ](../results/2026-11-06_mlicpp_operational/analysis.json)
- [Metrics từng ảnh](../results/2026-11-06_mlicpp_operational/metrics.json)
- [Tham số quantizer học được](../results/2026-11-06_mlicpp_operational/model_asset.json)
- [Fresh decode](../results/2026-11-06_mlicpp_operational/fresh_decode.json)

Server run: `/home/jovyan/clic2027-artifacts/mlicpp/operational-20261106/div2k-q2-hard-rd-scalar-vs-slices`.

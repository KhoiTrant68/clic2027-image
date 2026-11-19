# Kiểm tra tối ưu lượng tử hóa bằng rate–distortion thực

## Giả thuyết và quyết định trước khi chạy

Pilot geometry trước chỉ train predictor của log độ nhạy synthesis, không tối ưu RD. Thử xem có thể cải thiện bằng cách tối ưu trực tiếp `J = bpp_file + 0.0035 × MSE_RGB_255` qua toàn bộ encode/decode, giữ context và LRP chạy thật. Đây là kiểm tra một nguyên nhân khả dĩ, chưa khẳng định nguyên nhân thất bại.

| Nhánh | Tham số được tối ưu | Ngân sách lựa chọn |
|---|---|---|
| identity | bước 1 | đối chứng gốc |
| scalar | một bước chung | 40 giá trị log-spaced trong [0.7,1.5] và identity |
| slices | 10 bước cho 10 nhóm latent | hai vòng coordinate search, mỗi tọa độ thử ±0.12 trong log step, 40 candidate và identity |

Mọi nhánh có chung header OPQ1 10 byte, cùng backbone q2, cùng entropy tables và cùng LRP. Thang học được là model asset cố định dùng chung, không chọn riêng cho từng ảnh. Nhánh slices khởi tạo identity, không dùng metric teacher, không ép tổng log step bằng 0; giới hạn bước [0.5,2]. Cùng số candidate không có nghĩa là không gian tìm kiếm hay thuật toán có độ khó bằng nhau.

Chọn 8 nguồn train đầu tiên không trùng nhau từ cache geometry; 8 nguồn validation đầu tiên. Không mở cache test của geometry. Validation này đã được dùng trong pilot trước nên là **development**, không phải holdout độc lập. Không train neural backbone mới; coordinate descent là tối ưu 10 tham số của model quantizer với hard quantization và rate thật.

Đánh giá validation tại factor {0.8,1,1.25,1.6}; tất cả điểm được báo cáo. Primary là J tại factor 1; báo cả MSE/bpp và đường RD. Chỉ coi là tín hiệu để thiết kế thí nghiệm lớn hơn nếu slices giảm J validation ít nhất 0.5% so với scalar; đây là gate thăm dò, không là kiểm định thống kê hoặc bảo đảm paper. Không tìm tiếp hyperparameter trên validation sau khi thấy kết quả. Chưa dùng LPIPS/Qhat để ra quyết định ở gate này.

Smoke identity phải khớp bytes/pixel với codec gốc; decoder ở tiến trình mới phải đọc đúng asset và stream, không source/cache. Bootstrap paired J theo 8 ảnh 2.000 lần là mô tả độ bất định development, không chứng nhận tổng quát hóa.

## Tài nguyên và thực thi

A30 GPU 2 dùng chung, batch 1, FP32, TF32 off, 2 CPU threads; guard mặc định nửa VRAM toàn GPU, chỉ dừng tiến trình của pilot khi vượt. Ước tính 10–30 phút do tải dùng chung; timeout 90 phút. Khoảng 650 lượt encode/decode cho search, thêm 96 lượt đánh giá và 96 fresh decode. JSONL, progress/status và model_asset.json thay tracking ngoài; không cần dataset mới. Giữ nguyên toàn bộ evidence của geometry.

```bash
CUDA_VISIBLE_DEVICES=2 bash experiments/mlicpp/run_operational_quantizer.sh \
  --config experiments/mlicpp/operational_quantizer.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/operational-20261106/div2k-q2-hard-rd-scalar-vs-slices
```

## Prior art và hướng nghiên cứu

Adaptive quantization nói chung không mới: [Adaptive LVQ, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.html). Reconstruction offsets và post-training đa rate cũng đã có: [Kamisli et al., DCC 2024](https://arxiv.org/abs/2402.18930). [Soft then Hard, ICML 2021](https://proceedings.mlr.press/v139/guo21c.html) đã nghiên cứu mismatch của lượng tử hóa khi train/infer. Rà soát này mới đọc abstract/metadata của các nguồn trên; không phải full novelty audit.

Nếu có tín hiệu RD thực, ứng viên nghiên cứu tiếp là **học hình học của quantizer có xét toàn bộ quá trình giải mã nhân quả và chi phí entropy**, với transform block và predictor từ decoded context. Cần đối chứng scalar/slice gain, LVQ và offset trước khi gọi là đóng góp. Nếu pilot âm, chưa đủ cơ sở thêm MLP hoặc transformer để cứu câu chuyện.

## Kết quả

Đã hoàn tất: 96/96 fresh decode, 711,4 giây. Slices giảm J validation 0,1523% so với scalar, CI chứa 0; không đạt gate 0,5%. J gần như bằng identity. Không mở rộng hướng chỉnh gain từ kết quả này. [Báo cáo chi tiết](../reports/MLICpp_operational_quantizer_2026-11-06.md).

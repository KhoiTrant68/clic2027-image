# Entropy coding của DiT-IC có tất định giữa các thiết bị không? (12/10/2026)

Script: `experiments/determinism/ditic_check.py`.
- **Cách thử:** nén đầu vào codec của 6 ảnh validation trên thiết bị A và B, rồi giải mỗi bitstream trên cả hai.
- **Đếm:** số chỉ số bảng scale lệch (`build_indexes`), số ký hiệu rANS lệch, và độ lệch đầu ra.
- **Thêm:** khoảng cách của các scale tới ranh giới bảng 64 mức của CompressAI.

**Vì sao quan trọng:** DiT-IC giải y theo 4 nhóm checkerboard nối tiếp. Mạng số thực tính scale và mean của từng nhóm. Scale chọn
bảng CDF cho rANS, và nhóm trước là đầu vào của nhóm sau. Chỉ cần decoder (L4 của CLIC) tính scale lệch encoder đủ để vượt một
ranh giới là rANS giải sai từ đó trở đi.

## Kết quả (6 ảnh, q3)

| A ↔ B | precision | bitstream A/B | chỉ số lệch | giải mã codec |
|---|---|---|---|---|
| CPU ↔ GPU L40 | fp32 (cuDNN TF32 bật) | khác nhau (vài byte) | **15–60% ở mọi ảnh** → ảnh hỏng | 0.14 s (GPU) |
| CPU ↔ GPU L40 | fp64 | giống hệt | 0 (scale lệch ~1e-14) | 1.5 s (GPU), 1.1 s (CPU, 16 luồng) |
| GPU L40 #0 ↔ GPU L40 #1 | fp32 | giống hệt | **0 (scale lệch đúng 0)** | 0.14 s |

- **Scale sát ranh giới:** mỗi ảnh có ~300 scale cách ranh giới < 1e-4 (tương đối) và ~30 scale < 1e-5.
  Sai số fp32 giữa hai thiết bị khác loại đủ lớn để vượt các ranh giới này.
- **fp64 an toàn nhưng chậm:** gần như toàn bộ 1.5 s nằm ở chuỗi tham số entropy (h_s, adapter, context ba DepthConvBlock, LRP;
  ~210 GFLOP cho ảnh 2K). Phần còn lại (g_s, scale, aux, prompt) chỉ 6 ms. GPU tính fp64 chậm ~64 lần fp32; L4 còn chậm hơn L40.

## Hệ quả

Mọi codec học có hyperprior số thực đều dính vấn đề này, **kể cả MS-ILLM**. Nén trên máy mình rồi giải trên server CLIC chỉ an
toàn khi hai bên tính ra cùng một kết quả từng bit.

**Hướng A (chính): nén trên đúng phần cứng và phần mềm của decoder CLIC.**
- **Môi trường:** L4, image docker `clic-gpu` từ devkit (torch 2.6, cùng phiên bản cuDNN),
  `cudnn.deterministic = True`, `benchmark = False`.
- **Đã kiểm chứng:** hai GPU cùng loại trong cùng một máy cho kết quả khớp tuyệt đối.
- **Còn phải kiểm chứng:**
  1. Hai máy L4 thuê khác nhau (khác driver/máy chủ).
  2. Server validation của CLIC: nộp decoder DiT-IC và kiểm tra PSNR server trả về khớp số đo ở nhà.

**Hướng B (dự phòng nếu A không qua):** chuỗi tham số entropy bằng số nguyên.
- **Cách làm:** int8 weights và activation, `torch._int_mm` cộng dồn int32, requantize bằng nhân/dịch số nguyên,
  hàm kích hoạt bằng bảng tra.
- **Ưu điểm:** tất định trên mọi máy và nhanh.
- **Chi phí:** phải fine-tune có lượng tử hoá phần codec của DiT-IC (khoảng một tuần), rate tăng nhẹ.
  `src/clic27/entropy/intnet.py` hiện chở số nguyên trong fp64 (đúng nhưng chậm như fp64 với mạng cỡ này).

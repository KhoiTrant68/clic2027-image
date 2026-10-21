# AEIC ME/SE pilot — 20/10/2026

## Kết quả hoàn tất

**Hoàn tất lúc 00:40:57 ngày 20/10/2026 (Asia/Bangkok): 6/6 điểm có bitstream thật, 6/6 ảnh khớp pixel khi decode ở tiến trình mới, scoring hoàn tất.**

| Model | Bitrate gộp 3 ảnh | Q̂ v0 trung bình ↑ | Decode độc lập trung bình/ảnh | Peak VRAM tiến trình |
|---|---:|---:|---:|---:|
| AEIC-SE ft2 | 0.02317 bpp | -1.5044 | 26.17 s | 6.14 GiB |
| AEIC-ME ft2 | 0.02674 bpp | -1.2374 | 27.56 s | 7.22 GiB |

Peak đo qua `nvidia-smi` theo PID (SE 6288 MiB, ME 7390 MiB), bao gồm CUDA context và bộ nhớ ngoài allocator. Scoring đạt 4028 MiB. Tất cả dưới trần riêng tiến trình 10 GiB. Tốc độ decode ở bảng không gồm model load, process startup và ghi PNG; GPU dùng chung nên không xem đây là benchmark tốc độ chuẩn.

Ở ngân sách 0.075 bpp cho ba ảnh, bổ sung AEIC vào baseline mix cũ làm Q̂ tăng từ **-0.07341 lên -0.00149** (+0.07192), bitrate thực 0.07065 bpp. Allocator chọn AEIC-ME cho hai ảnh natural/texture và VTM-SCC cho screen. Không cải thiện mix ở 0.15 hoặc 0.3 bpp trong pilot này. Đây là dấu hiệu hữu ích cho phân bổ ngân sách thấp, **chưa chứng minh vượt baseline/PULSE hiện tại trên toàn bộ 30 ảnh**.

Chỉ thử một checkpoint mỗi biến thể; các operating point AEIC thấp hơn nhiều mức 0.075 bpp. Không dùng Q̂ trung bình đơn lẻ để kết luận model thua ở cùng bitrate. Hướng tiếp theo là kiểm tra 30 ảnh cùng checkpoint và so với mix tích hợp PULSE mới nhất trước khi quyết định fine-tune.

Bằng chứng: [metrics](pilot_20261020_shared8g_v2/scores/metrics.csv), [summary](pilot_20261020_shared8g_v2/scores/summary.csv), [allocation](pilot_20261020_shared8g_v2/scores/allocation.csv), [SE parity](pilot_20261020_shared8g_v2/aeic_se_ft2/parity.json), [ME parity](pilot_20261020_shared8g_v2/aeic_me_ft2/parity.json).

- Server: `guess`, thư mục `/mnt/home/guess03/clic/aeic_20261016`.
- Hàng đợi cũ PID `579472` đã dừng theo yêu cầu chuyển sang chạy ít VRAM.
- Pilot đã hoàn tất: `pilot_20261020_shared8g_v2`, PID điều phối lúc chạy `909527`, GPU 0.
- Lần kiểm tra trước launch: GPU 0 dùng 33062 MiB, GPU 1 dùng 33046 MiB, thuộc tiến trình tài khoản khác. Không dừng hay sửa các tiến trình đó.
- Trạng thái sống: `pilot_20261020_shared8g_v2/status.json` trên server. Các file trạng thái ở đây là snapshot, không tự cập nhật.

## Phạm vi và cấu hình

Nguồn: [AEIC](https://github.com/LuizScarlet/AEIC), commit `7222c2c85e8db7ea0b829a139313ab7f4fdea88c`.
Hai checkpoint chính thức `AEIC_ME_ft2.pkl` và `AEIC_SE_ft2.pkl`, SD-Turbo và VAE `halfDecoder.ckpt` của AdcSR.
`ft2` là tên checkpoint tác giả; thư mục `q2` chỉ là cách ghép vào scorer hiện có, không tương đương q2 của PULSE/DiT-IC.

Ba ảnh full-resolution: `2684452d` (natural), `937476dd` (texture), `ebfd571f` (screen).
Mỗi model encode bitstream thật, decode ngay để tạo reference, sau đó một tiến trình mới decode lại không nhận đường dẫn ảnh gốc.
So SHA256 của pixel; nếu khác, dừng trước khi chấm điểm. Giữ kiểm tra symbol/CDF trên thiết bị khác cho bước sau.

Khác mặc định tác giả:

- Tắt color fix dựa trên ảnh gốc; không truyền thống kê màu ngoài bitstream.
- Tắt `torch.compile`, TF32 và cuDNN benchmark; bật cuDNN deterministic, seed 0, FP32, batch 1.
- Giữ xformers; giảm latent tile xuống 64/overlap 16 cho cả hai model, VAE tile 64. Khác mặc định tác giả (ME 96/32, SE 192/64, VAE 160), cần lưu ý khi so chất lượng/tốc độ.
- Thêm 4 byte envelope định danh variant và checkpoint; cộng đủ vào bitrate.
- Opt-in `AEIC_SHARED_SMALL=1`: allocator tối đa 8 GiB, giám sát PID của chính job và dừng khi đạt 10 GiB hoặc toàn GPU còn dưới 1,5 GiB trống. Bộ nhớ của tài khoản khác không tính vào trần riêng của AEIC. Đây là software guard, không phải phân vùng GPU cứng.
- `--shared-small` chọn GPU còn ít nhất 12 GiB trống, chạy tuần tự SE rồi ME trên một GPU. Mỗi pha có timeout 1 giờ. Cấu hình mặc định của các experiment cũ vẫn giữ guard nửa thiết bị.

Sau parity, scorer sẵn có tính PSNR/LPIPS/DISTS/Q̂ và so với baseline `work_merge_1013_v0` trên đúng ba ảnh. Baseline này chưa gồm PULSE tích hợp mới nhất; không dùng bảng pilot để tuyên bố vượt hệ thống hiện tại hay kết quả toàn bộ 30 ảnh.

## Kiểm tra đã hoàn tất

- Python 3.10, Torch 2.1.2+cu121, diffusers 0.25.1, xformers 0.0.23.post1; phiên bản đầy đủ trong `requirements.lock.txt`.
- Build entropy coder thành công. Patch `PendingTask p{}` khởi tạo biến rõ ràng để qua GCC 13; xem `runtime.patch`.
- Nạp đúng trọng số codec ME/SE. SE bỏ đúng 8 key projection chỉ dùng distillation theo loader inference tác giả; các key còn lại được load strict.
- CPU preflight với tensor synthetic 128×128: SE stream 48 byte, ME 52 byte; decode hai lần cho latent giống hệt và hữu hạn. Đây là kiểm tra entropy/runtime, **không phải** phép đo chất lượng ảnh hay xác nhận toàn bộ diffusion decoder.
- Python compilation và self-check allocator guard đều qua.
- Lần GPU đầu thiếu `pytorch-lightning` khi nạp VAE; đã bổ sung 2.1.3 cùng torchmetrics 1.6.2, kiểm tra nạp trên CPU rồi chạy lại vào output mới `_v2`.
- `cpu_preflight.json` chứa hash checkpoint và kết quả. GPU model load, ảnh full-resolution, pixel parity, peak VRAM và scoring đã hoàn tất; metadata/timing/log/trace nằm trong thư mục evidence. `weights.sha256` ghi cả trọng số nền.

## Theo dõi và chạy lại

```bash
ssh guess 'cat ~/clic/aeic_20261016/pilot_20261020_shared8g_v2/status.json'
ssh guess 'tail -40 ~/clic/aeic_20261016/queue_shared8g_v2.log'
```

Lệnh hàng đợi đã khởi động (không chạy lại vào cùng output):

```bash
cd ~/clic/aeic_20261016
py310/bin/python -u probe_aeic.py --stage queue --shared-small \
  --root /mnt/home/guess03/clic/aeic_20261016 \
  --out /mnt/home/guess03/clic/aeic_20261016/pilot_20261020_shared8g_v2 \
  --data /mnt/home/guess03/clic/work_full_gpu/bakeoff/valid \
  --only 2684452d 937476dd ebfd571f
```

Khi `state=complete`, thu các JSON/timing/parity, `scores/*.csv`, log và VRAM trace về đây. Khi `failed`, đọc log pha tương ứng; không coi output dở dang là kết quả hợp lệ. Khi `expired_without_free_gpu`, cần một lần launch mới sau khi có tài nguyên.

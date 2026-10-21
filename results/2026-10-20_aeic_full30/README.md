# AEIC full30 — hoàn tất 20/10/2026

**60/60 bitstream thật; 60/60 pixel parity trong tiến trình mới. Scoring kết thúc 06:49:43 Asia/Bangkok. Không fine-tune ở vòng này vì không đạt ngưỡng cải thiện đã định trước.**

| Budget bpp | Baseline có PULSE | Thêm AEIC | Delta Qhat | AEIC được chọn |
|---|---:|---:|---:|---:|
| 0.075 | 0.184142 | 0.190313 | +0.006171 | 3 ME |
| 0.15 | 0.789137 | 0.789137 | 0 | 0 |
| 0.3 | 1.497811 | 1.497811 | 0 | 0 |

Ba lựa chọn AEIC tiết kiệm tổng 45.390 byte nhưng giảm điểm riêng từng ảnh; allocator chuyển phần bit đó sang ba ảnh khác. Ngưỡng đã định là +0.02 mean Qhat và ít nhất 3 ảnh dùng AEIC. Chỉ điều kiện số ảnh đạt; điều kiện cải thiện không đạt.

Peak NVML riêng tiến trình: SE 7006 MiB, ME 7496 MiB, scorer 4072 MiB. Guard allocator 8 GiB / dừng tiến trình 10 GiB. API decode độc lập trung bình SE 24.26 s/ảnh, ME 21.90 s/ảnh trên hai L40 dùng chung; không phải benchmark so tốc độ kiến trúc hoặc gói nộp bài.

## Tài liệu và artifact

- [Báo cáo đầy đủ với biểu đồ và crop](../../reports/CLIC2027_AEIC_full30_2026-10-20.md).
- [Protocol trước scoring](PLAN.md) và [kiểm tra khả năng fine-tune](TRAINING_READINESS.md).
- [Comparison](full30_20261020/comparison.json), [resource summary](full30_20261020/resource_summary.json), [thay đổi từng ảnh](full30_20261020/changed_images.csv).
- [Metrics](full30_20261020/scores/metrics.csv), [allocation](full30_20261020/scores/allocation.csv), [provenance](full30_20261020/provenance.json).
- [SE parity](full30_20261020/aeic_se_ft2/parity.json), [ME parity](full30_20261020/aeic_me_ft2/parity.json).
- [Bộ chấm A/B](full30_20261020/blind_review/index.html): 18 cặp crop, 54 ảnh lossless, phiếu người chưa có. Mở file HTML bằng trình duyệt trên máy, giữ thư mục `img` cạnh file. Tên model/khóa đáp án không nằm trong HTML.
- [Quan sát AI trước khi mở khóa](full30_20261020/visual_audit/AI_NOTES_BLIND.md): tách riêng khỏi phiếu người. Muốn chấm mù, hãy chấm trước khi đọc khóa và ghi chú.

Đã kiểm đủ 54 ảnh 512×512 và tính ẩn danh của danh sách cặp trong HTML; đã xem 18 crop qua contact sheet. Chưa xác nhận UI tương tác vì trình duyệt tích hợp chặn URL file cục bộ. Không có human preference score được suy diễn hoặc điền sẵn.

## Tái lập

Server giữ stream và PNG đầy đủ tại `~/clic/aeic_20261016/full30_20261020`. Thư mục local chỉ chứa bằng chứng gọn và crop. Hai tiến trình codec và scorer đã kết thúc; không có job fine-tune được khởi động.

Tái tạo biểu đồ và resource summary trên server:

```bash
cd ~/clic/aeic_20261016
py310/bin/python summarize_aeic.py full30_20261020
```

Lệnh codec lúc chạy (mỗi biến thể một GPU; **dùng output mới nếu chạy lại**):

```bash
AEIC_SHARED_SMALL=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
py310/bin/python probe_aeic.py --stage pair --variant SE \
  --root /mnt/home/guess03/clic/aeic_20261016 \
  --out /mnt/home/guess03/clic/aeic_20261016/full30_20261020/aeic_se_ft2 \
  --data /mnt/home/guess03/clic/work_full_gpu/bakeoff/valid
# ME: CUDA_VISIBLE_DEVICES=1, --variant ME, thư mục aeic_me_ft2.
```

`full30_evidence.tgz` SHA256 đã kiểm sau tải: `b1951ca73279a0e8ef113d73fc12bb0c3781ffd81a1fbebec954c682d0bc0368`. 158 file đã giải nén an toàn. Contact sheet, examples.png, ghi chú AI và README này là sản phẩm local tạo sau khi tải, không nằm trong tar server. Hash weights, lockfile môi trường và patch C++ xem [pilot](../2026-10-20_aeic_pilot/README.md).

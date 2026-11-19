# Kiểm tra decoder perceptual trên ảnh đầy đủ — GPU 1

Đã khởi chạy khoảng 15:53 ngày 09/11/2026 (UTC+7), session `mlic-gpu1-perceptual-full32-20261109`. Đây là infer/eval, không phải training mới.

## Thiết kế cố định

- 32 ảnh development DIV2K đầy đủ, không resize; nguồn development đã dùng trước, không gọi là test độc lập.
- q2 legacy gốc so với synthesis reconstruction đã train ngày 05/11; cùng encoder, entropy, padding replicate và hệ số lượng tử.
- Bốn factor cố định 0,5 / 1 / 2 / 4, hai decoder: 8 cấu hình, 256 ảnh đánh giá và 256 lượt fresh decode.
- Kiểm bitstream SHA-256 phải giống giữa hai decoder tại mỗi factor; không chọn factor theo metric test.
- Báo cáo byte thực, pooled/mean PSNR, LPIPS-Alex, DISTS và thời gian encode/decode. Metric perceptual dùng tile theo evaluator hiện tại.
- Đây là đường cong khảo sát, chưa điều khiển chính xác 0,075 / 0,15 / 0,3 bpp và chưa có human preference.

## Kiểm tra trước chạy

So với CPU perceptual probe, thay đổi có chủ đích là crop8 → full32, CPU → GPU L40 số 1, factor 1/2 → 0,5/1/2/4. Giữ nguyên weights và không train. Các so sánh giữa hai decoder trong mỗi factor dùng cùng cấu hình.

Đã xác nhận weights q2, synthesis checkpoint, manifest và môi trường; runner kiểm hash 32 ảnh và lưu source/config/protocol. GPU 1 còn 45.026 MiB lúc kiểm tra; yêu cầu ít nhất 32.768 MiB trước khi bắt đầu, allocator PyTorch giới hạn 70%. Không chiếm VRAM giả và không can thiệp job khác.

Theo dõi qua `status.json`, `pipeline.log` và `metrics.json` tăng dần mỗi ảnh. Ước tính ban đầu 20–60 phút, tương ứng khoảng 16:13–16:53; không coi là ETA đo được. Ngân sách bốn giờ được kiểm trước cấu hình mới, không kill tiến trình đang chạy. Output mới, giữ nguyên kết quả cũ.

```text
/home/jovyan/clic2027-artifacts/mlicpp/perceptual-full-20261109/div2k-dev32-q2-original-vs-perceptual-4factors-gpu1
```

## Cơ sở lựa chọn

Campaign 12 lượt train trước đã hoàn thành, nhưng learned steps không cải thiện đồng đều ở các mức rate/seeds. Pilot perceptual CPU có lợi về LPIPS/DISTS ở cùng byte, nên cần kiểm tra trên ảnh đầy đủ trước khi mở rộng train. Không diễn giải kết quả crop thành thứ hạng CLIC.

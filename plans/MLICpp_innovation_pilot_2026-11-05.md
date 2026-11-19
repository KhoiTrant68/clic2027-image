# Pilot train → stream → decode → eval cho conditional innovation

## Phạm vi đã khóa

Người dùng yêu cầu thực hiện trên server. Pilot này hiện thực **hiệu chỉnh có điều kiện**, với đối chứng basis cố định và residual pixel, trên decoder full-synthesis đã huấn luyện. Đây không phải một generative posterior đã được kiểm confidence, nên chưa gọi kết quả là xác nhận “prior-conflict”.

Đơn vị đánh giá: 32 nguồn DIV2K mới, một crop 256²/nguồn, giữ riêng khỏi train/development và hai ảnh smoke. Source permutation seed 1984, indices `[290:322]`; crop seed `9041/filename`. Smoke chỉ dùng hai nguồn ở indices `[288:290]`, không dùng metric smoke để chọn hyperparameter. Hai nhóm cũng không trùng 224 ảnh pilot đầu hoặc 64 ảnh screen tiếp theo. Overlap với pretraining gốc của MLIC++ chưa audit.

## Model và đối chứng

| Nhánh | Model học / tìm kiếm | Byte và mục đích |
| --- | --- | --- |
| `base` | Full-synthesis reconstruction checkpoint đã khóa | Stream q2 không thêm packet; là đường gốc, không phải control cùng byte tốt nhất |
| `conditional` | Học maps từ feature entropy-decoded và channel basis; source encoder đề xuất hệ số | 8 hướng × grid 2² = 32 hệ số int8; nhánh đề xuất |
| `fixed` | Cùng model/khởi tạo/optimizer nhưng maps nhận template feature trung bình train, không nhận nội dung ảnh | Cùng 32 hệ số và cùng packet; primary contrast chỉ thay conditioning |
| `pixel` | Residual DCT/RGB cố định theo tile, source encoder học hệ số | Cùng 32 hệ số/packet; control residual thông thường, không matched decoder capacity |
| `latent_tto` | Tối ưu main latent qua **decoder cuối**, encoder-only; serialize rANS candidates | Mỗi ảnh phải ≤ `base_bytes + 46`; đối chứng dùng thêm bit cho base mà không thêm correction decoder |

Basis có điều kiện và cố định cùng số tham số, encoder nguồn, rank/grid, init, data order và training steps. Fixed template là trung bình feature từ train, được lưu như model asset chung; không có ảnh test trong template. Synthesis đầy đủ/entropy đóng băng; chỉ innovation/source encoder học. Pixel control dùng basis khác nên không nhận là ablation capacity tương đương.

Tham chiếu: `full_synthesis_q2.json`, checkpoint `reconstruction/last.pt` bước 1.000. Thay đổi có chủ đích: đóng băng synthesis đã fine-tune, thêm code32 và basis; Adam LR 1e−4 cho phần mới, seed 9041. Loss giữ `10 MSE + LPIPS-Alex + 0,1 edge_rational`. Mỗi nhánh 1.000 bước, batch 4, cùng 320 crop từ 160 ảnh train. Checkpoint cuối cố định; không chọn bằng test hoặc validation.

## Packet và đường giải mã

Format `INOV` có header **14 byte** và payload **32 byte**. Header chứa version, mode, rank, grid, base length và coefficient count. Coefficient step `1/64` và amplitude `0,25` là tham số model đã khóa, không phải scale theo ảnh bị bỏ quên. Không có bbox/index vị trí tự chọn: grid 2 × 2 cố định. Tổng overhead **46 byte/ảnh**, tương đương 0,00561523 bpp cho crop 256².

Encoder chọn coefficients bằng source encoder, so thêm zero candidate và refine 40 bước Adam LR 0,03. Tất cả candidate được đánh giá với hard quantization trong forward; không báo gain từ coefficient chưa lượng tử hóa. Selection dùng loss đã khóa ở trên, không dùng DISTS. Decoder chỉ đọc base stream + coefficients và áp tail synthesis; không nhìn nguồn, không gọi mạng ngoài, không chạy optimizer.

`latent_tto` dùng cùng 40 bước, LR 0,02, rate surrogate penalty 0,5 khi vượt target. Cứ 5 bước tạo rANS stream thật, chỉ giữ candidate trong byte budget và có source loss tốt hơn. Original base stream là fallback. Thời gian/compute mỗi bước không tương đương coefficient TTO; đây là same-iteration control, chưa phải equal-compute.

## Phân tích khóa trước

- Primary contrast: conditional − fixed trên LPIPS/DISTS ở cùng từng packet size. Báo PSNR gộp, MS-SSIM và RGB finite-difference error; không chỉ metric dùng học.
- So conditional với pixel và latent TTO; TTO có thể dùng ít byte hơn target, phải báo actual byte và số candidate feasible. Không nội suy quality từ rate ước lượng.
- Bootstrap ghép cặp 10.000 lần theo nguồn ảnh, seed 9041; exploratory pointwise CI, một seed và một rate. Không diễn giải thành human preference.
- So baseline không packet chỉ cho thấy gain với byte bổ sung, không đủ chứng minh đóng góp. Claim conditioning chỉ tồn tại nếu hơn fixed control; claim coding efficiency còn cần hơn pixel/TTO và thử nhiều rate, test mới, backbone hiện đại.
- Fresh process giải mã toàn bộ stream, dùng decoder/module thật và kiểm hash uint8. Không dùng feature cache hay nguồn để vượt qua bước này.
- Nếu conditional không hơn fixed/pixel/TTO, giữ nguyên kết quả và dừng claim tương ứng. Không chỉnh hyperparameter theo test vừa mở.

## Preflight và chạy

Smoke đầy đủ đã qua: ba nhánh hai bước, coefficient search, latent TTO, summary và **10/10 fresh pixel hashes** trên hai ảnh riêng. Actual packet framing đã kiểm trên CPU trước smoke GPU. Paths cache, full-synthesis checkpoint, MLIC q2 weights và test manifest đều kiểm tồn tại.

Chạy một L40 GPU 0, FP32 deterministic, TF32 tắt, cuDNN v7/NVRTC; hai CPU threads và guard VRAM hiện có. Không đụng job trên GPU khác. Monitoring: `status.json`, `prepare.progress.json`, từng mode `progress.json`/`train.jsonl`, `infer.progress.json`, `latent-search.progress.json`, VRAM logs và checkpoint SHA. Giữ run lỗi nếu có, không ghi đè.

Ước tính 10–30 phút; hard ceiling **2 giờ trên một GPU** cho chuỗi stage, không API trả phí. ETA lúc launch khoảng 17:00–17:20 giờ Việt Nam ngày 05/11/2026. Dữ liệu/feature cache và checkpoint dự kiến vài GB, đủ chỗ trên ổ server còn hơn 270 GB. Thời gian thực sẽ được ghi vào status, không dùng ETA làm kết quả.

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mlicpp/run_innovation.sh \
  --stage run --config experiments/mlicpp/innovation_q2.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/innovation-20261105/div2k-q2-conditional-fixed-pixel-new32
```

Chưa phải gói submission CLIC, chưa đánh giá full-resolution, đủ ba rate hoặc L4. Không gộp metric crop vào bảng 30 ảnh. Mệnh đề toán học về fixed-rank family không tự bảo đảm neural pilot thắng; đây là phép thử thực tế cần thiết để bác bỏ hoặc giữ hướng đó.

## Kết quả thực hiện ngày 05/11/2026

Đã train ba nhánh đủ 1.000 bước, infer/eval trên 32 nguồn mới và kiểm 160/160 fresh pixel matches. Run TTO đầu treo native entropy coder sau 29 ảnh; recovery giữ nguyên đầu ra đã hoàn tất và xử lý ba ảnh còn lại bằng worker có guard/timeout, không đổi model hoặc hyperparameter. Cả run lỗi và lineage được giữ lại.

**Conditional không vượt fixed:** LPIPS chênh +0,002294, CI 95% [+0,001384; +0,003419]; DISTS chênh +0,000170, CI chứa 0. Dừng claim lợi ích conditioning ở recipe này; không tune theo test đã mở. Fixed giảm LPIPS 14,62% so base khi thêm 46 byte/ảnh, chưa là gain cùng rate hoặc đóng góp mới. TTO chỉ thay đầu ra 1/32 ảnh nên chưa là comparator tối ưu mạnh. [Báo cáo và bằng chứng](../reports/MLICpp_innovation_pilot_2026-11-05.md).

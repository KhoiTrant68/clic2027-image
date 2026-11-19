# Pilot adapter MLIC++ — 04/11/2026

Pilot kiểm tra học decoder sau entropy; không phải tái lập training toàn paper, không phải phiên bản MLICv2 hoặc submission hoàn chỉnh.

Đã hoàn tất hai cohort; xem [báo cáo kết quả](../../reports/MLICpp_adapter_pilot_2026-11-04.md). Head Risk chưa đạt tiêu chí đi tiếp; kết quả này không xác lập đóng góp mới.

## Cấu hình cố định

- Nền: checkpoint **MLIC++ legacy q2**, λ MSE 0,0035, cùng commit/runtime đã kiểm trong `probe.py`. Không giả sử bitrate đúng 0,15 bpp; đo byte của từng stream crop.
- 224 ảnh từ DIV2K train HR: **160 train / 32 validation / 32 test**, chia theo ảnh bằng seed 1984. Train hai crop 256 × 256/ảnh, các split còn lại một crop/ảnh; tổng 384 stream. Không dùng 30 ảnh CLIC để huấn luyện adapter. Chưa audit việc pretraining gốc của MLIC++ có chứa các ảnh này.
- Mirror công khai được pin revision; từng tệp phải khớp CRC trong central directory của ZIP chính thức. Lưu thêm SHA-256. DIV2K dành cho academic research; [điều khoản nguồn](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Cache **FP32** feature từ input của `legacy synthesis_transform[5]` sau decode stream thực, tương ứng độ phân giải 1/4. Tail gốc đóng băng; adapter bottleneck và head nhỏ là phần được học. Encoder, context entropy và base weights không đổi.
- Ba nhánh `mean`, `cvar`, `risk` có cùng kiến trúc, khởi tạo, thứ tự batch, Adam, 1.000 bước, batch 4. `mean`: pixel + LPIPS-Alex + edge mean. `cvar`: thêm CVaR. `risk`: thêm giám sát head bằng tác động của hiệu chỉnh dự kiến lên lỗi cạnh. Đây là fine-tune perceptual nhẹ, **không có GAN**.
- Edge surrogate: Sobel grayscale, 2 × 2 vùng của crop; chuẩn hóa bằng percentile 95 của lỗi baseline **train**. Run `pilot_q2.json` dùng hard clamp [0,1]; validation phát hiện tail bão hòa nên cohort `pilot_q2_rational.json` thay bằng `u/(1+u)`, giữ thứ tự và gradient ở đuôi. Quyết định sửa được đưa ra trước khi xem test outcomes; chạy lại cả ba nhánh, không gộp metric edge giữa hai transform. Threshold CVaR được tối ưu, khởi tạo từ percentile α trên train theo đúng transform. Head dự đoán nhãn tăng lỗi proxy vượt `harm_margin`, không phải xác suất lỗi ngữ nghĩa đã calibration.
- Checkpoint dùng **bước cuối cố định**; validation chỉ log, không chọn theo test. Các số LR/loss/alpha nằm trong [pilot_q2.json](pilot_q2.json). Mọi nhánh chạy cùng GPU để so sánh training. Nhánh risk có thêm forward để tạo nhãn, nên training compute không bằng hai nhánh kia.

## Kiểm chứng

`pilot_model.py` có self-check cho zero-initialized identity, gradient, Sobel-zero và CVaR đuôi phân số. Smoke test chạy đầy đủ pipeline trên 6 ảnh, hai bước/nhánh trước pilot chính. Phiên bản smoke đầu phát hiện adaptive pooling CUDA backward không deterministic trong Torch 2.0; đã thay bằng pooling cố định không chồng lấn, không tắt deterministic.

Test dùng RGB uint8, module metric chung của repo để đo PSNR, MS-SSIM, LPIPS-Alex và DISTS. Có thêm edge mean/CVaR, ablation `risk_hide_change`; không tính Q̂ thiếu feature hoặc gọi subset metric này là Q̂. Stream và byte giống nhau cho tất cả nhánh trên cùng một crop. Giải mã lại stream test bằng **tiến trình mới**, adapter chèn thật vào synthesis, rồi kiểm hash pixel với đường cache-evaluation.

Thời gian `suffix_s` chỉ đo tail synthesis + adapter trên crop, gồm đồng bộ GPU; **không phải tổng decode hoặc số đo L4**. So sánh latency cuối cần warmup/repeated measurement và full-resolution. Tail metric trên 32 ảnh chưa đủ để khẳng định một cải tiến nghiên cứu; chưa có OCR với transcript, human evaluation, nhiều seed, GAN hoặc đối chứng PerQ/PICO đầy đủ.

## Chạy

```bash
python experiments/mlicpp/pilot_model.py
python experiments/mlicpp/fetch_pilot_data.py --out /home/jovyan/clic2027-artifacts/data/div2k/pilot224
CUDA_VISIBLE_DEVICES=1 bash experiments/mlicpp/pilot_server.sh \
  --config experiments/mlicpp/pilot_q2_rational.json --stage run
```

Mỗi lần thử phải có output/cache mới. Runner từ chối ghi đè thư mục run/cache đã có. Nếu một stage lỗi, giữ log và trạng thái lỗi; chỉ chạy stage tiếp tục sau khi kiểm rõ checkpoint/cache, không xóa bằng chứng lỗi. Stage `run` tạo tiến trình mới cho từng nhánh và phép kiểm decoder, log CSV/JSON/JSONL, checkpoint atomic tại mỗi mốc validation. Trọng số, ảnh và cache lớn ở ngoài Git.

Cohort rational có `reuse_cache=true` để dùng đúng stream/FP32 feature của cohort đầu; kiểm data/checkpoint hash trước khi dùng. Nếu chạy từ đầu ở máy khác, tạo cache bằng cấu hình gốc trước, hoặc trỏ một cấu hình rational mới với `reuse_cache=false` tới output/cache mới. Không ghi đè run cũ. Mã nguồn cohort clamp đã được lưu trong `source_archive` của bằng chứng run đó.

Theo dõi `status.json`, `prepare.progress.json`, `<mode>/progress.json`, `<mode>/validation.jsonl`, `summary.csv`, `fresh-<mode>.json`. Guard VRAM giữ allocator dưới ngưỡng bảo thủ có sẵn của repo; khi GPU bị tác vụ khác chiếm vượt ngưỡng, chỉ dừng tiến trình của thử nghiệm.

## Pilot conditional innovation

Pilot dùng decoder full-synthesis đã khóa, học ba nhánh `conditional`, `fixed`, `pixel`, rồi so với encoder latent TTO. Đã chạy đủ 1.000 bước/nhánh, test 32 nguồn mới và 160/160 fresh matches; conditional không vượt fixed. [Kết quả](../../reports/MLICpp_innovation_pilot_2026-11-05.md), [protocol và test split](../../plans/MLICpp_innovation_pilot_2026-11-05.md). Đây là conditional correction, chưa phải kiểm định prior confidence.

```bash
python experiments/mlicpp/innovation_stream.py
CUDA_VISIBLE_DEVICES=0 bash experiments/mlicpp/run_innovation.sh \
  --stage run --config experiments/mlicpp/innovation_q2.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/innovation-20261105/div2k-q2-conditional-fixed-pixel-new32
```

Run chính cần test manifest từ `fetch_pilot_data.py --holdout-only --holdout-start 290 --holdout-count 32`; hai nguồn trước đó dành cho smoke. Output phải là thư mục mới. Stream `INOV` gồm base stream và 32 hệ số int8 với tổng overhead 46 byte. Decoder nhận scale cố định từ model asset, không từ ảnh nguồn.

Theo dõi `status.json`, từng mode `progress.json`, `infer.progress.json`, `worker-*.json`, `worker-results/*.json`, `fresh.json` và `summary.csv`. TTO hiện chạy worker riêng có guard/timeout để tránh treo native rANS do symbol cực lớn; giữ candidate hợp lệ cuối hoặc original base, không bỏ ảnh. Artifact lớn giữ trên server; sau khi tải số liệu nhỏ và mã chạy vào `source_archive`, phân tích bằng `analyze_innovation.py --root <evidence-folder>`. Các checkpoint, template và full-synthesis weights đều phải đúng hash; không thay trọng số rồi đọc stream như thể cùng decoder.

## Pilot cải thiện fixed

So sánh 2 × 2: train 1.000/5.000 bước và encoder search 40/160 bước, giữ packet 46 byte, trên 32 nguồn mới `[322:354]`. Đây là tối ưu kỹ thuật baseline, chưa là đóng góp mới. [Protocol khóa trước](../../plans/MLICpp_fixed_improvement_2026-11-05.md).

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mlicpp/run_fixed_improvement.sh \
  --stage run --config experiments/mlicpp/fixed_improvement_q2.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/innovation-20261105/new-fixed-run
```

Output phải chưa tồn tại. Reference 1.000 bước được copy vào output; train 5.000 bước học lại từ cùng khởi tạo vì checkpoint cũ không lưu optimizer. Snapshot search ở 40/160 trên cùng trajectory, fresh decode kiểm cả bốn packet family. Theo dõi `fixed/progress.json`, `infer.progress.json`, `status.json`; phân tích lại bằng `analyze_fixed_improvement.py --root <output>`. Không dùng test mới này để tiếp tục chọn hyperparameter.

Đã hoàn tất: 160/160 fresh matches; tổ hợp 5.000/160 giảm LPIPS 8,63% so mốc cùng byte, PSNR giảm 0,0602 dB, DISTS chưa phân biệt rõ. [Báo cáo và CI](../../reports/MLICpp_fixed_improvement_2026-11-05.md).

## Pilot lượng tử hóa theo hình học decoder

[Thiết kế](../../plans/MLICpp_geometry_pilot_2026-11-05.md): đóng băng MLIC++ q2 MSE, ước lượng độ nhạy RGB bằng VJP và train MLP dự đoán bước lượng tử hóa từ hyperprior đã giải mã. So sánh uniform/global/predicted/oracle tại bốn hệ số. Oracle truyền 1.280 byte thang float32; mọi nhánh có header bổ sung 12 byte. Đây là distillation teacher, chưa train end-to-end theo RD.

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mlicpp/run_geometry.sh \
  --config experiments/mlicpp/geometry_q2.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/geometry-20261105/new-geometry-run
python experiments/mlicpp/analyze_geometry.py <output>
```

Output phải chưa tồn tại. Test lấy nguồn `[354:370]` theo shuffle seed 1984; dùng `fetch_pilot_data.py --holdout-only --holdout-start 354 --holdout-count 16` để chuẩn bị. Chạy `geometry_smoke.json` trước: bước bằng 1 phải tạo đúng payload codec gốc. Theo dõi `prepare.progress.json`, `train.progress.json`, `metrics.json`, `fresh_decode.json` và `status.json`; timeout toàn pilot 90 phút. Decoder mới cần backbone đúng hash và `predictor.pt` tương ứng; format GEO1 không tự mang trọng số. Không coi codec đã đạt kiểm tra tính xác định giữa các loại GPU chỉ từ fresh decode trên cùng L40.

Đã hoàn tất: 256/256 fresh decode, 160/160 crop audit. [Kết quả và giới hạn](../../reports/MLICpp_geometry_pilot_2026-11-05.md). Giữ nguyên test đã dùng; không tái sử dụng để chọn hyperparameter rồi báo là đánh giá mới.

## Pilot tác động tiếp diễn của symbol

[Protocol](../../plans/MLICpp_lookahead_pilot_2026-11-06.md): chạy CPU với 2 luồng khi GPU bận; 4 ảnh development, 15 can thiệp integer độc lập và baseline cho mỗi ảnh. Tính lại context/LRP/suffix, dùng nguyên định dạng và decoder gốc. Đây là chẩn đoán trước training, chưa có critic được học.

```bash
bash experiments/mlicpp/run_lookahead.sh \
  --config experiments/mlicpp/lookahead_pilot.json \
  --out /home/jovyan/clic2027-artifacts/mlicpp/lookahead-20261106/new-symbol-suffix-run
```

Output phải chưa tồn tại. Theo dõi `status.json`, `progress.jsonl`, `metrics.json`, `fresh_decode.json`, `analysis.json`. Timeout 1 giờ. Nhánh CPU bỏ lời gọi CUDA synchronize chỉ phục vụ timer trong subprocess; đường GPU mặc định của loader không đổi. Đối chứng bỏ future-rate vẫn sử dụng distortion ảnh cuối, vì vậy không được gọi là thuật toán local search cùng runtime.

Đã hoàn tất trên CPU trong 175,27 giây: 64/64 fresh decode, không đạt gate. [Kết quả và audit tập candidate có escape](../../reports/MLICpp_lookahead_pilot_2026-11-06.md). Dùng `audit_lookahead.py <evidence-folder>` để kiểm tra checksum và so sánh trên cùng tập candidate đủ điều kiện phân rã CDF.

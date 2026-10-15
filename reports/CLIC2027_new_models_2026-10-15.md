# Tìm hướng mới và thử model cho CLIC 2027 — 15/10/2026

## Quyết định sau vòng thử

**Đưa PULSE-L vào danh sách ứng viên của bake-off tiếp theo.** Đây là hướng mới đáng đầu tư nhất trong các model vừa chạy: decoder nhẹ, có bitstream thật, bổ sung được cho baseline. Chưa thay DiT-IC/MS-ILLM toàn bộ và chưa gọi kết quả này là thắng CLIC.

Đã hoàn tất 204 lượt encode/decode: AFP-GIC 18, DC-VIC 18, PULSE 24, PULSE-L pilot 24 và PULSE-L full 120. Có thêm ba phép decode PULSE-L từ tiến trình mới. Đây là inference/evaluation checkpoint công khai; chưa fine-tune hoặc train mới.

Người dùng đã nâng giới hạn từ nửa VRAM lên **30 GB trên mỗi GPU**. Toàn bộ lượt này vẫn dùng guard nửa thiết bị chặt hơn vì đủ bộ nhớ. Không cần cấp thêm VRAM chỉ vì được phép.

## 1. Kết quả đủ 30 ảnh

Tập ảnh là 30 ảnh trong `~/clic/work_full_gpu/bakeoff/valid`, cùng tập baseline `~/clic/work_merge_1013_v0`. Điểm Q̂ v0 qua chính `clic27.eval.metrics`, `experiments/qhat/qhat.py` và `bakeoff.qhat_score`. Chọn một điểm/ảnh bằng `allocate_one`, tính byte bitstream thật, cùng tổng pixel và giới hạn byte. Điểm cao hơn là tốt hơn. Không so điểm pilot 6 ảnh với bảng 30 ảnh này.

| Ứng viên | 0.075 bpp | 0.15 bpp | 0.3 bpp |
|---|---:|---:|---:|
| DiT-IC hiện có | 0.1644 | 0.2788 | 0.2788 |
| MS-ILLM | -0.4513 | 0.5447 | 1.3831 |
| PULSE-L perceptual, q0/1/2/4 | Không vừa budget | Không vừa budget | **1.4030** |
| Mix baseline | 0.1836 | 0.7582 | 1.4215 |
| Mix baseline + PULSE-L | **0.1841** | **0.7891** | **1.4912** |
| Mức tăng của mix | +0.0006 | **+0.0309** | **+0.0697** |

Allocation mix mới dùng PULSE-L cho 1/30, 10/30 và 20/30 ảnh tương ứng ba budget. Ở 0.3, còn 7 MS-ILLM, 2 VTM-SCC, 1 DiT-IC. Lợi ích ở 0.075 quá nhỏ để ưu tiên thêm decoder riêng chỉ vì điểm này.

Tất cả ảnh dùng q0 của PULSE-L đã tốn **0.1921 bpp** tổng thể. Vì vậy checkpoint hiện có không thể đứng độc lập ở 0.075/0.15 trên tập này; vẫn hữu ích cho một số ảnh trong mix. Không nội suy điểm dưới rate thấp nhất rồi coi đó là kết quả thực.

Thời gian PULSE-L trên L40, Torch 2.6.0+cu124, hai CPU thread:

- Trung vị decoder trên 120 điểm: **57.2 ms/ảnh**; khoảng 30.8–121.4 ms.
- Tổng các lần decode được chọn cho PULSE-L riêng ở 0.3: **1.762 s/30 ảnh**. Model load: **0.730 s**.
- Đây là thời gian API giải bitstream, chưa gồm đọc/ghi file, unpack, khởi động Python hoặc đóng gói bài thi. Các ảnh đã qua encode trong cùng tiến trình nên không coi đây là cold-start.
- Decode trong tiến trình mới, không gọi encoder, không đưa ảnh gốc vào decoder: 3/3 ảnh khớp pixel tuyệt đối; lần đầu 402 ms, hai lần sau 131/110 ms. Chưa chứng minh parity L40↔L4, CPU↔GPU hoặc tổng thời gian trên máy CLIC.

Q̂ là metric đại diện đã dùng trong repo, không phải điểm người chấm CLIC. Allocation được tối ưu trên chính các điểm này; chưa có holdout mới, CI hay chấm mù. Các số tăng là kết quả sàng lọc, không phải bằng chứng tổng quát hóa. Bảng hiện chưa cộng metadata chọn codec chung hoặc kiểm kích thước/tải tất cả decoder của mix; cần đánh giá gói cuối cùng.

## 2. Pilot sáu ảnh của các model còn lại

Chọn trước ba ảnh tự nhiên (`02809272`, `0369d229`, `07b9f93f`), texture (`937476dd`), hai ảnh màn hình (`bb7344a2`, `ebfd571f`). Ảnh nguyên kích thước, không giảm resolution để vượt giới hạn VRAM. Pilot nhỏ nhằm loại hướng ít triển vọng, không đại diện cả phân phối.

| Ứng viên | Mức q đã chạy | Q̂ @ .075 | Q̂ @ .15 | Q̂ @ .3 | Trung vị decode/ảnh |
|---|---|---:|---:|---:|---:|
| AFP-GIC | 0,1,4 | -0.7304 | 0.3158 | 0.8712 | 7.00 s |
| DC-VIC PatchGAN | 0,1,4 | Không vừa | 0.4599 | 1.0237 | 9.49 s |
| PULSE perceptual | 0,2,4,7 | Không vừa | Không vừa | Không vừa | 51.3 ms |
| PULSE-L perceptual | 0,2,4,7 | Không vừa | Không vừa | 1.6057 | 63.0 ms |
| MS-ILLM baseline | Điểm đã có | -0.3030 | 0.6423 | 1.7323 | Không đo lại |
| Mix baseline | Điểm đã có | 0.3806 | 1.0409 | 1.8629 | Không đo lại |
| Mix thêm cả bốn model | Các điểm trên | 0.3806 | 1.0560 | 1.9194 | Chưa đo mix |

AFP/DC-VIC dùng Torch 2.11.0+cu128, CompressAI 1.2.8, khác stack gốc của tác giả. Đã đổi tên tensor entropy cũ sang ParameterList mới và bắt buộc `strict=True`, không bỏ qua trọng số. Một smoke AFP trước khi sửa đã load không đầy đủ entropy parameters: **loại toàn bộ smoke đó khỏi kết quả**. Kết quả bảng chỉ lấy thư mục `*_strict`. Đây là chạy thử tương thích, không tuyên bố tái lập mọi con số trong paper.

Quyết định: tạm không dành train dài cho AFP/DC-VIC. Pilot chưa vượt baseline, thời gian giải mã cao; cần lý do mới hoặc cấu hình tác giả tốt hơn trước khi mở rộng. PULSE base cũng không ưu tiên với checkpoint này vì vùng rate không phù hợp. Chọn PULSE-L chạy đủ 30 ảnh và đã có bảng ở trên.

## 3. VRAM và khả năng chạy lại

Server có 2 L40, mỗi GPU báo 46,068 MiB. Dịch vụ sẵn có dùng khoảng 1,517 MiB/GPU. Guard giới hạn PyTorch allocator tối đa 18 GiB, giảm tiếp theo headroom thực tế; lấy tổng `nvidia-smi` mỗi 0.25 s, dừng chính job trước nửa VRAM. Đây là guard phần mềm, không phải phân vùng GPU cứng; một tiến trình khác vẫn có thể tăng bộ nhớ giữa hai lần lấy mẫu.

| Run | Đỉnh tổng VRAM quan sát, gồm dịch vụ khác |
|---|---:|
| AFP-GIC pilot | 11,264 MiB (11.0 GiB) |
| DC-VIC pilot | 7,718 MiB |
| PULSE pilot | 3,130 MiB |
| PULSE-L pilot | 4,140 MiB |
| PULSE-L 30 ảnh | 4,304 MiB |

Giới hạn mới 30 GB được hiểu bảo thủ là 30,000,000,000 byte cho mỗi thiết bị. Khi cần train lớn hơn guard hiện tại, phải cấu hình lại cap với headroom và smoke trước, không bỏ guard. Batch nhỏ + gradient accumulation, crop train + activation checkpointing là các cách tăng hiệu quả; không âm thầm tile/resize evaluation vì sẽ đổi kết quả codec.

Artifacts nhẹ được lưu ở `results/2026-10-15_new_codecs`: `manifest.json`, checksum trọng số, revision repo, JSON từng điểm, log VRAM, metric, allocation, summary và kiểm decode độc lập. Bitstream, PNG và trọng số lớn ở `~/clic/research_20261015` trên server; không ghi đè repo remote đang dirty. Script trong `experiments/new_codecs`; guard self-check, py_compile và actual bitstream round-trip đã qua.

## 4. Các hướng mới và model CLIC cũ đã tìm

| Nguồn chính thức | Ý đáng quan tâm | Trạng thái trong vòng này |
|---|---|---|
| [PULSE, 2026](https://github.com/microsoft/GenCodec/tree/main/PULSE) | Decoder nhẹ, variable-rate, integer entropy path; có training và checkpoint | Đã chạy base/L; L chạy đủ 30 ảnh |
| [AFP-GIC](https://github.com/yifeipet/AFP_GIC) | Kết hợp prior sinh ảnh, một model nhiều rate | Đã chạy 18 điểm; chưa thắng baseline |
| [DC-VIC](https://github.com/iwa-shi/DC_VIC) | Khai thác generative codebook qua dual conditioning | Đã chạy PatchGAN 18 điểm |
| [PICO, 2026](https://github.com/apple-aiml-research/ml-pico) | Tối ưu chất lượng người xem cùng runtime, quan tâm cross-platform | Đã kiểm repo: assets/viewer và reconstructions, chưa tìm thấy runtime/checkpoint để chạy trên 30 ảnh của ta |
| [ProGIC, 2026](https://arxiv.org/abs/2603.02897) | RVQ progressive, backbone nhẹ | Có paper; chưa xác minh được checkpoint chạy được, chưa đo |
| [AEIC, CVPR 2026](https://github.com/LuizScarlet/AEIC) | Ultra-low-rate, shallow encoder | Có repo; chưa chạy. Xếp sau PULSE-L vì cần kiểm khả năng đáp ứng tốc độ decoder và rate mục tiêu |
| [EGIC, ECCV 2024](https://github.com/Nikolai10/EGIC) | Điều khiển trade-off distortion/perception | Chưa chạy; chi phí tích hợp TensorFlow riêng, ưu tiên sau |
| [PORT, CLIC 2025, PKUSZ-AliMerlin](https://www.jdl.link/doc/2011/20251221_232_32_18.PORT%20Perception-Oriented%20Image%20Compression.pdf) | Decoder nhanh, semantic/patch loss, ROI và variable-rate | Đã tìm paper; chưa tìm checkpoint công khai có thể chạy. Không coi đọc paper là đã thử model |
| [Cool-Chic CLIC 2025](https://github.com/Orange-OpenSource/Cool-Chic/releases) | Overfit từng ảnh, decoder C nhẹ | Repo này đã thử ngày 09/10; xem kết quả cũ bên dưới |

Cool-Chic tag `clic2025` đã có kết quả tại `results/2026-10-09_coolchic2025_probe/summary.md`. Mẫu 2684452d tốn ~1 h 39 phút encode, 0.557 bpp, Q̂ v0 1.33; MS-ILLM cùng ảnh đạt 1.46 ở 0.267 bpp. Run cũ OOM khi giới hạn 20 GB và chạy được ở 28 GB; đó là lịch sử trước ràng buộc mới, không phải run ngày hôm nay. Chưa có lý do chạy lại hàng loạt. Nay được phép 30 GB, có thể thử riêng screen/texture ở lambda đúng dải, nhưng phải tính cả dịch vụ và overhead; cap allocator 28 GB không tự bảo đảm tổng GPU ≤30 GB.

Các bài CLIC cũ có paper nhưng chưa tìm được weights không được gắn nhãn đã reproduce. Tìm kiếm có hạn chế truy cập OpenReview và không chứng minh rằng checkpoint không tồn tại ở nơi khác. Việc dùng bộ dữ liệu CLIC trong paper cũng không đồng nghĩa tác giả là đội dự thi CLIC.

## 5. Ba giả thuyết nghiên cứu nên ưu tiên

### A. Chọn codec theo chất lượng, thời gian và chi phí tải model

**Làm trước**, vì bảng mới cho thấy tính bổ sung thực tế. Thay mục tiêu chỉ tối đa Q̂ dưới budget byte bằng mục tiêu có thêm thời gian decode, chi phí load chung cho mỗi họ model và mức không chắc chắn của điểm chất lượng. Decoder nhẹ có thể nhận nhiều ảnh dễ, DiT-IC nhận ảnh thực sự cần nó.

Triển khai tối thiểu: bổ sung candidate PULSE-L vào bake-off hiện có; đo cold-start cho từng họ; tính byte mode ID/header; so rate-only, rate+time, rate+time+load. Sweep giới hạn thời gian, giữ đường Pareto chất lượng–thời gian. Chấm mù các cặp có chênh lệch lớn trước. Không xây framework mới.

Tính mới dự kiến nằm ở bài toán lựa chọn dưới chi phí tải dùng chung và độ bất định đánh giá, nếu chứng minh tốt hơn routing/knapsack đã có. Routing và adaptive compute tự thân không mới. Đây hiện là giả thuyết đóng góp, chưa xác nhận novelty.

### B. Hạ rate PULSE-L với distillation có bảo vệ chữ và mặt

Checkpoint hiện tại có sàn 0.1921 bpp trên cả tập. Mục tiêu có thể đo: tạo điểm actual-rate 0.075/0.15 mà giữ lợi thế decoder nhẹ. Dùng PULSE-L làm nền trainable; fine-tune RD với rate embedding/gain và entropy model phù hợp. Không chỉ ép q âm ngoài dải rồi tuyên bố codec hợp lệ.

Teacher perceptual cung cấp mục tiêu cho texture; vùng chữ/mặt ưu tiên trung thực ảnh gốc, hạn chế ép student học chi tiết teacher bịa. DiT-IC latent không tương thích mặc định với PULSE: dùng image-space teacher hoặc thiết kế/training projection rõ ràng, không ghép hai decoder bằng cách đổi tên tensor.

Ma trận đầu tiên: B0 RD fine-tune thuần; B1 thêm teacher đồng đều; B2 teacher theo độ tin cậy/vùng. Cùng seed, crop, số ảnh và số update; 200–500 bước smoke đo VRAM, rồi pilot tối đa 6 GPU-giờ/nhánh. Chỉ mở rộng khi đủ rate, decoder đọc stream độc lập, chất lượng không giảm ở text/face, và chấm mù có tín hiệu.

ROI/hallucination control đã có tiền lệ: [PORT](https://www.jdl.link/doc/2011/20251221_232_32_18.PORT%20Perception-Oriented%20Image%20Compression.pdf), [Conditional Hallucinations](https://arxiv.org/abs/2410.19493), [Region-Adaptive Generative Compression](https://arxiv.org/abs/2604.01122). Vì vậy không gọi thêm ROI mask là mới. Cần chứng minh lợi ích của **ước lượng độ tin cậy teacher khi distill xuống decoder nhanh trong dải rate thấp** bằng ablation và dữ liệu độc lập.

### C. Residual refinement theo vùng, chỉ trả thêm bit khi có ích

Mục tiêu: sửa chữ, biên, khuôn mặt bị sai với lượng residual hữu hạn; để vùng texture dùng prior sinh. Thử encoder-side refinement trước để ít tăng chi phí decoder. Mọi mask, vị trí patch, gain, seed hoặc latent sửa phải truyền/tính byte; decoder không có ảnh gốc.

So baseline không refinement, refinement đồng đều và refinement theo độ tin cậy, cùng tổng byte và cùng ngân sách encode. Giữ lại nếu tăng chất lượng người chấm ở 0.15/0.3, không chỉ tăng một metric. Adaptive residual/ROI đã có prior art; tính mới cần ở tiêu chí chọn vùng và chứng minh mức tăng trên đường rate–quality–time.

## 6. Thứ tự công việc sau vòng này

1. Tích hợp PULSE-L vào bake-off chính, dùng đúng checkpoint/hash đã đo. Giữ baseline bất biến. Bổ sung q3 và các mức thiếu chỉ khi cần làm dày đường RD.
2. Đóng gói decoder và đo **tổng** cold-start, RAM, VRAM, bytes gồm metadata. Đo L4 thật khi có máy; không suy tốc độ L4 từ L40.
3. Chấm mù mix mới với mix cũ, tập trung 10/30 và 20/30 ảnh được PULSE-L chọn; thêm dev/holdout ngoài 30 ảnh validation. Không train trên bộ validation này.
4. Nếu xác nhận có ích, chạy ma trận A; tiếp đến smoke B0/B1/B2 dưới 30 GB/GPU. Chỉ nâng ngân sách sau kết quả pilot.
5. Dành tối đa một slot thử nghiệm cho AEIC hoặc checkpoint CLIC cũ khi có runtime/weights xác minh được. Cool-Chic chỉ thử mục tiêu screen/texture, không quét hàng loạt tốn GPU trước.

Mốc hoàn thành vòng hiện tại: 204 điểm đã chạy/chấm, 3 kiểm decoder độc lập đã qua; mọi job của vòng này đã kết thúc. Chưa train hướng A/B/C và chưa thay codec mặc định của repo.

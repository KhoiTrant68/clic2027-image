# Pilot adapter MLIC++: kết quả thực nghiệm ngày 04/11/2026

**Đã hoàn tất huấn luyện và đánh giá.** Adapter perceptual thông thường cải thiện rõ LPIPS và DISTS so với checkpoint MSE. Tuy nhiên, **head dự đoán nguy cơ chưa vượt được đối chứng CVaR đơn giản** và chưa đạt tiêu chí tiếp tục của đề xuất. Kết quả này ủng hộ việc xây một baseline perceptual tốt trên MLIC++, chưa ủng hộ claim đóng góp mới cho head hiện tại.

Đây là thử nghiệm trên **crop 256 × 256, 32 ảnh test giữ riêng cho adapter**, không phải kết quả full-resolution trên 30 ảnh CLIC hoặc phép đo đúng 0,15 bpp.

## 1. Thiết kế đã chạy

| Thành phần | Cấu hình thực tế |
| --- | --- |
| Nền | MLIC++ MSE legacy q2, λ = 0,0035; upstream commit `4aa9d2a512eb9db382172058e6c278fb0d376e17` |
| GPU | NVIDIA L40, physical GPU 1 trên server mới; không sử dụng A30 đang có tác vụ khác |
| Runtime | PyTorch 2.0.1+cu118, CompressAI 1.2.0b3; fp32, tắt TF32, deterministic, cuDNN v7 + NVRTC như inference đã kiểm |
| Dữ liệu | 224 ảnh DIV2K train HR: 160 train, 32 validation, 32 test; chia theo ảnh nguồn, seed 1984 |
| Crop | Hai crop cố định/ảnh train, một crop/ảnh validation và test; tổng 384 stream thực |
| Adapter | 34.113 tham số, bottleneck 32 kênh; chèn trước `legacy synthesis_transform[5]`, ở độ phân giải 1/4 |
| Phần đóng băng | Encoder, hyperprior, context entropy và toàn bộ synthesis gốc; chỉ học adapter và head |
| Huấn luyện | Mỗi nhánh 1.000 bước, batch 4, Adam, LR adapter 1e−4, LR τ 1e−3; cùng khởi tạo và thứ tự batch |
| Checkpoint đánh giá | Bước cuối cố định; validation chỉ ghi log, không dùng test để chọn checkpoint |
| Rate | Cùng stream/byte giữa mọi nhánh; **bpp test thực = 0,19934082**, không gắn nhãn 0,15 bpp |

DIV2K được tải qua mirror pin revision; từng tệp khớp CRC trong ZIP chính thức và được lưu SHA-256. Không có ảnh trùng tên hoặc hash trong tập chia. 30 ảnh CLIC không dùng để huấn luyện adapter. **Chưa audit overlap giữa các ảnh này và pretraining gốc của MLIC++**, nên “giữ riêng” chỉ áp dụng cho lần huấn luyện adapter này.

Feature cache lấy từ **giải mã stream thực**, giữ FP32; không dùng entropy ước lượng để báo bitrate. Việc cache prefix đóng băng giúp pilot nhanh nhưng không thay đổi định nghĩa decoder: sau huấn luyện đã chèn adapter vào synthesis thật và giải mã lại stream bằng tiến trình mới.

## 2. Các nhánh đối chứng

- **Base MSE:** checkpoint tác giả, không adapter.
- **Mean:** adapter với pixel MSE + LPIPS-Alex + lỗi cạnh trung bình.
- **CVaR:** cùng kiến trúc Mean, thêm mục tiêu CVaR lỗi cạnh theo vùng.
- **Risk:** cùng kiến trúc CVaR, thêm giám sát head bằng lỗi tăng thêm của hiệu chỉnh dự kiến.
- **Risk, ẩn hiệu chỉnh:** ablation lúc test, đưa 0 thay cho `Δf` ở input head, giữ nguyên nhánh hiệu chỉnh.

Cả ba nhánh học có cùng số tham số, số bước, dữ liệu và optimizer. Risk cần thêm forward tạo nhãn nên chi phí training cao hơn; không gọi đây là cùng training compute tuyệt đối. Pilot **không dùng GAN**, Q̂ hoặc OCR làm loss; đây là baseline perceptual nhẹ để kiểm cơ chế, chưa phải tối ưu perception mạnh nhất.

## 3. Sửa phép đo bão hòa trước khi kết luận

Run đầu chuẩn hóa lỗi Sobel bằng percentile 95 của train rồi chặn về [0,1]. Validation cho CVaR bằng 1 ở các nhánh học; phần đuôi bị mất thứ tự và gradient khi chạm trần. Run này vẫn được giữ nguyên cùng mã nguồn trong [bằng chứng cohort clamp](../results/2026-11-04_mlicpp_adapter/clipped/summary.md).

Trước khi xem kết quả test của run đầu, đã quyết định thay hard clamp bằng biến đổi đơn điệu:

```text
u = lỗi Sobel trung bình trong vùng / scale từ train
d_s = u / (1 + u)
```

Sau đó chạy lại **đủ cả ba nhánh**, giữ nguyên stream, feature FP32, seed, data order, số bước và các trọng số loss. Threshold CVaR được khởi tạo theo biến đổi mới và tiếp tục được tối ưu. Không gộp giá trị edge của hai cohort vào cùng bảng. Các kết quả dưới đây đều thuộc cohort **rational**.

## 4. Kết quả trên 32 ảnh test

Tất cả hàng dùng cùng 0,19934082 bpp. PSNR gộp từ MSE, các metric còn lại lấy trung bình theo ảnh. Edge CVaR dùng α = 0,9 trên 128 vùng, bốn vùng/ảnh; giá trị thấp hơn là tốt hơn theo surrogate này.

| Phương án | PSNR gộp ↑ | LPIPS-Alex ↓ | DISTS ↓ | Edge mean ↓ | Edge CVaR ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base MSE | 30,6045 | 0,173504 | 0,202334 | 0,305183 | 0,499218 |
| Mean | 29,4255 | **0,098065** | 0,143204 | 0,352365 | 0,528026 |
| CVaR | 29,4889 | 0,098374 | **0,142136** | 0,350888 | 0,525010 |
| Risk | 29,5303 | 0,101331 | 0,143970 | 0,348629 | 0,526539 |
| Risk, ẩn hiệu chỉnh khi test | 28,6016 | 0,125376 | 0,202725 | 0,400313 | 0,541153 |

**Perceptual adaptation có hiệu quả trong phạm vi pilot:** Mean giảm LPIPS khoảng **43,48%**, DISTS **29,22%** so với Base MSE, đổi lại PSNR giảm khoảng **1,18 dB** và lỗi cạnh tăng. LPIPS-Alex có trong training loss; DISTS không có trong loss, nên kết quả DISTS bổ sung một đối chiếu ngoài mục tiêu học trực tiếp. Đây là giảm giá trị metric, không phải phần trăm cải thiện chất lượng do người xem chấm.

**CVaR có tác động nhỏ so với Mean:** Edge CVaR giảm khoảng **0,57%**, LPIPS tăng khoảng 0,31%, DISTS giảm khoảng 0,75%. Hiệu ứng ở surrogate đo được nhưng còn xa ngưỡng giảm tail 10% đã đề xuất cho pilot.

**Risk chưa cho lợi ích ở đuôi so với đối chứng đơn giản:** so với CVaR, LPIPS tăng khoảng **3,01%**, DISTS tăng khoảng **1,29%**, còn Edge CVaR tăng khoảng **0,29%**. Risk có edge mean thấp hơn, nhưng cải thiện trung bình này không chuyển thành ưu thế ở đuôi lỗi. So với Mean, mức giảm tail chỉ khoảng **0,28%**, trong khi LPIPS tăng khoảng **3,33%**; không đạt tiêu chí LPIPS không tăng quá 2% và tail giảm ít nhất 10%.

## 5. Khoảng tin cậy theo ảnh

Bootstrap cặp 2.000 lần, seed 1984; lấy mẫu lại **ảnh**, giữ bốn vùng của mỗi ảnh trong cùng cụm. Bảng báo chênh lệch `nhánh trước − nhánh đối chứng`, chưa hiệu chỉnh nhiều phép so sánh; đây là phân tích thăm dò một seed.

| So sánh | Metric | Chênh lệch | CI 95% theo ảnh |
| --- | --- | ---: | --- |
| Mean − Base | LPIPS | −0,075439 | [−0,094548; −0,057249] |
| Mean − Base | DISTS | −0,059130 | [−0,070248; −0,046877] |
| CVaR − Mean | Edge CVaR | −0,003016 | [−0,003576; −0,002281] |
| Risk − Mean | Edge CVaR | −0,001487 | [−0,004695; +0,000569] |
| Risk − CVaR | Edge CVaR | +0,001529 | [−0,001519; +0,002881] |
| Risk − CVaR | LPIPS | +0,002958 | [+0,001395; +0,004798] |

Theo các khoảng này, chưa xác lập được ưu thế tail của Risk so với Mean/CVaR. Không diễn giải CI chứa 0 thành chứng minh hai phương pháp tương đương; cũng không dùng CI theo ảnh để thay thế biến thiên giữa nhiều seed.

## 6. Kiểm chứng decoder và tài nguyên

- Đã hoàn tất **384 stream** dùng chung cho các nhánh, ba nhánh × 1.000 bước ở mỗi cohort.
- Với mỗi cohort, **96/96 ảnh test** giải mã bằng tiến trình mới khớp hash pixel với đường cache-evaluation: 32 ảnh × 3 decoder đã học. Hai cohort đều qua kiểm tra này.
- Nhánh Mean/CVaR dùng bộ nhớ PyTorch cấp phát đỉnh khoảng **725 MiB** khi train; Risk khoảng **846 MiB**. Đây không phải toàn bộ VRAM tiến trình.
- GPU 1 đã được giải phóng sau khi chạy xong; không dừng tác vụ khác trên A30.
- Log `suffix_s` chỉ đo tail synthesis trên crop; không dùng để kết luận adapter nhanh hơn baseline hoặc đạt giới hạn submission. Chưa có warmup/repeated full-decode latency benchmark trên L4.

Việc ẩn `Δf` ở input head làm kết quả xấu đi, nhưng đây là can thiệp lúc test có thể gây lệch phân phối input. **Không dùng ablation này làm bằng chứng head là đóng góp tốt**, nhất là khi Risk không thắng đối chứng CVaR. Cần đối chứng head được huấn luyện ngay từ đầu không có `Δf` nếu theo tiếp cơ chế này.

## 7. Kết luận và bước tiếp theo

**Không mở rộng head Risk hiện tại thành mô hình chính hoặc claim paper.** Pilot chưa đạt điều kiện đi tiếp đã đặt ra. Kết quả âm áp dụng cho đúng kiến trúc, surrogate, trọng số và số bước này; không chứng minh mọi phương pháp risk-aware đều thất bại.

Kết quả tích cực là đã có một **baseline perceptual trên MLIC++ với adapter nhỏ và byte không đổi**. Mean mạnh về LPIPS, CVaR nhỉnh hơn về DISTS/tail surrogate; chưa đủ dữ liệu để chọn phương án thi theo human preference. Nếu tiếp tục, bước có giá trị nhất là kiểm các checkpoint này ở full resolution, rate thấp phù hợp và lỗi nội dung có ground truth, rồi mới quyết định đổi cơ chế mô hình. Cần thêm modern MLIC++/MLICv2 và closest baseline như PerQ cho claim cạnh tranh khoa học.

Không đưa các hàng này vào bảng so sánh 30 ảnh như cùng một benchmark. Chúng dùng tập test, crop, rate và mục tiêu học khác. Đã lưu đầy đủ cả kết quả không thuận lợi và lần sửa metric.

## 8. Bằng chứng và cách chạy lại

- [Hướng dẫn pilot](../experiments/mlicpp/PILOT.md), [cấu hình cohort rational](../experiments/mlicpp/pilot_q2_rational.json), [mô hình adapter](../experiments/mlicpp/pilot_model.py), [runner](../experiments/mlicpp/pilot.py).
- [Bảng metric](../results/2026-11-04_mlicpp_adapter/rational/summary.csv), [metric từng ảnh](../results/2026-11-04_mlicpp_adapter/rational/test_metrics.json), [bootstrap](../results/2026-11-04_mlicpp_adapter/rational/paired_bootstrap.csv).
- [Trạng thái hoàn tất](../results/2026-11-04_mlicpp_adapter/rational/status.json), [manifest dữ liệu](../results/2026-11-04_mlicpp_adapter/rational/data_manifest.json), [hash bằng chứng](../results/2026-11-04_mlicpp_adapter/rational/evidence.sha256.json), [hash checkpoint adapter](../results/2026-11-04_mlicpp_adapter/rational/adapter_checkpoints.sha256.json).
- Checkpoint adapter đã tải về local: `results/2026-11-04_mlicpp_adapter/checkpoints/rational-{mean,cvar,risk}.pt`; các tệp `.pt` được Git bỏ qua. Mô hình MLIC++ gốc vẫn cần checkpoint riêng để giải mã.
- Artifact lớn trên server: `/home/jovyan/clic2027-artifacts/mlicpp/pilot20261104/`. Cohort cuối: `div2k224-q2-risk-adapter-seed1984-rational`; cohort đầu giữ nguyên trong thư mục không có hậu tố `-rational`.

**Giới hạn:** một seed, 32 ảnh test và crop 256; chưa audit pretraining overlap, chưa có GAN, human evaluation, OCR có transcript, Q̂ đủ sáu feature, L4 hoặc đánh giá full30 cho adapter. Sobel/CVaR là proxy cấu trúc, không bảo đảm giữ đúng chữ hoặc ý nghĩa. Các code self-check, smoke test toàn pipeline và hash decoder đã qua; chưa có bằng chứng để tuyên bố SOTA, novelty hoặc lợi thế thi CLIC.

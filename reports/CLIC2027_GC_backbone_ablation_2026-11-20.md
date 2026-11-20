# Ablation backbone cho GC: ELIC và MLIC++ với cùng công thức decoder GAN (20/11/2026)

**Kết luận:** nên dùng backbone kiểu MLIC++ cho GC. Có ba lý do:

- **Lợi thế MLIC++ lớn hơn khi dùng decoder GAN.** So với ELIC, MLIC++ tiết kiệm 10,9% bit ở cùng Q̂ khi dùng decoder MSE. Với decoder GAN theo công thức GC, mức tiết kiệm tăng lên 16,9% (BD-rate Q̂ v0 trên CLIC30).
- **Decoder GAN cho lợi thế như nhau trên hai backbone.** Ở dải thấp, Q̂ tăng khoảng +0,6 đến +0,8. Ở dải cao hơn, mức tăng nhỏ dần.
- **Ablation chưa thắng được mix hiện tại, và điều này là dự kiến.** Ở đây chỉ fine-tune g_s trong 20k bước. Q̂ tốt nhất ở 0,15 bpp là −0,32 (MLIC++ + GAN), trong khi mix đạt 0,79. Khi thêm vào pool, không có ảnh nào được chuyển sang hai codec này.

Mục tiêu của ablation là chọn backbone, không phải tạo ra một ứng viên nộp bài. Khoảng cách tới mix phải được lấp bằng phần train đầy đủ: train cả encoder và entropy model end-to-end ~1M bước, thêm rate control và Q̂ v2.

## 1. Giao thức

Mã nguồn nằm ở `experiments/gc_backbone/`: `train_gs.py`, `run_arm.sh`, `drive.sh`, `analyze.py`.

- **Phần cố định:** codec MSE pretrained của từng backbone giữ nguyên analysis, hyperprior, entropy model và stream.
  - ELIC: TestModel tái lập, compressai 1.1.5, q2–q5.
  - MLIC++: checkpoint chính thức, compressai 1.2.0b3, q1–q4.

  Đánh giá giải mã lại đúng các stream gốc của run `full30`, nên bpp của hai decoder giống nhau tuyệt đối. Mọi khác biệt chất lượng đều do decoder.
- **Phần được train:** chỉ g_s, khởi tạo từ g_s MSE, theo công thức gan của GC:
  - loss: k_mse·MSE + LPIPS(alex) + 0,15·GAN non-saturating;
  - discriminator `MultiScaleD` của GC, điều kiện trên ŷ;
  - optimizer: Adam 1e-4, D betas (0,5; 0,999), clip 1, EMA 0,999; lr giảm ×0,1 ở 10% cuối;
  - chạy 20k bước, batch 8, crop 256.
- **Dữ liệu train:** DIV2K train (800 ảnh).
  - Pool 32.768 crop cố định, seed 12345, giống nhau cho cả hai arm, có downscale ngẫu nhiên 0,5–1 và lật ảnh.
  - ŷ được tính trước một lần. Hook lấy ŷ đã được kiểm tra khớp tuyệt đối với decoder thật (max_abs 0).
- **Độ chính xác số:** bật TF32 cho cả hai arm. Ảnh hưởng đã đo: PSNR lệch 0,001 dB, sai khác pixel trung bình 0,006/255.
- **Chấm điểm:**
  - Q̂ v0 (`results/2026-10-15_new_codecs/qhat_v0.json`) tính bằng module metric chung;
  - BD-rate dùng fit bậc ba, chỉ trên vùng hai đường cong chồng nhau;
  - phân bổ theo ngân sách dùng knapsack của bake-off.
- **Máy chạy:** lab09, GPU 6 dùng chung, giới hạn 20 GiB VRAM có guard. Hai arm chạy song song. Có một lần guard dừng do nvidia-smi treo; job resume từ `last.pt` (bước 8000).

## 2. Kết quả theo điểm

| Run | q | bpp | Q̂ | LPIPS | DISTS | PSNR |
|---|---:|---:|---:|---:|---:|---:|
| ELIC MSE | 2 | 0,0837 | −2,629 | 0,289 | 0,195 | 27,08 |
| ELIC GAN | 2 | 0,0837 | −1,860 | 0,196 | 0,156 | 25,80 |
| ELIC MSE | 3 | 0,1296 | −1,741 | 0,238 | 0,163 | 28,69 |
| ELIC GAN | 3 | 0,1296 | −1,128 | 0,156 | 0,132 | 27,34 |
| ELIC MSE | 4 | 0,1970 | −0,971 | 0,194 | 0,136 | 30,45 |
| ELIC GAN | 4 | 0,1970 | −0,464 | 0,127 | 0,110 | 29,08 |
| ELIC MSE | 5 | 0,4497 | 0,642 | 0,116 | 0,082 | 34,38 |
| ELIC GAN | 5 | 0,4497 | 0,696 | 0,077 | 0,072 | 32,82 |
| MLIC++ MSE | 1 | 0,1173 | −1,716 | 0,236 | 0,164 | 29,07 |
| MLIC++ GAN | 1 | 0,1173 | −0,943 | 0,149 | 0,124 | 27,17 |
| MLIC++ MSE | 2 | 0,1794 | −0,937 | 0,190 | 0,137 | 30,70 |
| MLIC++ GAN | 2 | 0,1794 | −0,277 | 0,115 | 0,103 | 29,20 |
| MLIC++ MSE | 3 | 0,2610 | −0,220 | 0,153 | 0,112 | 32,33 |
| MLIC++ GAN | 3 | 0,2610 | 0,144 | 0,100 | 0,091 | 30,61 |
| MLIC++ MSE | 4 | 0,3935 | 0,554 | 0,118 | 0,085 | 34,13 |
| MLIC++ GAN | 4 | 0,3935 | 0,740 | 0,072 | 0,070 | 32,17 |

**Lợi thế của decoder GAN trên cùng stream (ΔQ̂):**
- ELIC: +0,77 / +0,61 / +0,51 / +0,06 (q2→q5);
- MLIC++: +0,77 / +0,66 / +0,36 / +0,19 (q1→q4).

Đổi lại, PSNR giảm 1,3–2,0 dB. Lợi thế thu hẹp dần khi bpp tăng, đúng xu hướng thường thấy với decoder GAN.

## 3. MLIC++ so với ELIC (Bjøntegaard)

| Decoder | Metric | BD-rate MLIC++ | BD-quality (MLIC++ − ELIC) |
|---|---|---:|---:|
| MSE | Q̂ v0 | −10,85% | +0,216 |
| MSE | LPIPS | −12,16% | −0,013 |
| MSE | PSNR | −12,65% | +0,62 dB |
| GAN (GC) | Q̂ v0 | **−16,88%** | **+0,268** |
| GAN (GC) | LPIPS | −20,43% | −0,014 |
| GAN (GC) | DISTS | −16,89% | −0,009 |
| GAN (GC) | PSNR | −7,17% | +0,32 dB |

Với decoder GAN, lợi thế của MLIC++ trên Q̂ và LPIPS tăng lên. Lợi thế trên PSNR giảm, đúng như mong đợi khi decoder được tối ưu theo cảm quan. Như vậy entropy model tốt hơn của MLIC++ vẫn giữ nguyên tác dụng khi chuyển sang decoder GAN.

## 4. So với mix theo ngân sách (phân bổ từng ảnh, 30 ảnh)

| Ngân sách | Mix hiện tại | ELIC MSE | ELIC GAN | MLIC++ MSE | MLIC++ GAN |
|---|---:|---:|---:|---:|---:|
| 0,075 | 0,184 | −2,499 | không đủ thấp | không đủ thấp | không đủ thấp |
| 0,15 | 0,789 | −1,111 | −0,571 | −0,948 | −0,322 |
| 0,3 | 1,498 | 0,132 | 0,359 | 0,257 | 0,555 |

- **Thêm vào pool:** gộp từng codec, hoặc cả bốn, vào pool tham chiếu thì gần như không ảnh nào được chọn. Ngoại lệ duy nhất: ở 0,075, một ảnh chuyển sang MLIC++ MSE, Q̂ tăng +0,001.
- **"Không đủ thấp":** điểm thấp nhất đã train vượt 0,075 bpp, nên knapsack không ra lời giải. Điểm thấp nhất của ELIC GAN là q2 = 0,084 bpp (q1 không được train), của MLIC++ là q1 = 0,117 bpp.

## 5. Hệ quả cho GC

1. **Backbone:** chuyển phần entropy model của GC sang kiểu MLIC++ (context đa chiều: kênh, cục bộ, toàn cục), hoặc dùng MLIC++ làm khởi tạo. Đánh đổi đi kèm: decode chậm hơn ELIC, và cần giữ được tính tất định số nguyên của entropy model GC. Phần này cần một thiết kế riêng.
2. **Cần có điểm rate rất thấp (≤ 0,075 bpp).** Cả hai backbone pretrained đều không có điểm này. GC phải tự train, nhờ stage `rate` với 8 target có mức thấp nhất 0,035.
3. **Fine-tune riêng g_s không đủ.** Cần train end-to-end và thêm alignment Q̂ v2, như kế hoạch 19/11.

## 6. File

- **Kết quả nhẹ:** `results/2026-11-20_gc_backbone/`, gồm `analysis/summary.md|json`, `compare/summary.md|json`, `eval/*/metrics.csv`, json decode và meta train.
- **Checkpoint g_s và stream:** chỉ có trên lab09 tại `/mnt/data/khoitm8/clic2027/gcb/`, gồm `train/*/gs.pt` và `eval/*`.

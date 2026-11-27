# GC v2 sau bước D: kết quả và việc cần làm (27/11/2026)

**Nguồn:**
- Bước D chạy trên A30 (GPU 2, server nexai-sv20) và xong lúc 08:43 UTC ngày 26/11, không lỗi.
- Số liệu: `results/2026-11-26_gc_stepd/` (`analysis/summary.md`, `bd/`).
- Code, config và snapshot model: `experiments/gc_stepd/`. Cách chạy: `experiments/gc_stepd/HUONG_DAN.md`.

**Tóm tắt:**
- Chưa có thay đổi nào qua được cổng (BD-rate ≤ −2% trên cả Q̂ v0 lẫn LPIPS), trừ g_s v2. Nhưng g_s v2 được thử trên nền D4, mà D4 là bước hỏng, nên chưa biết g_s v2 tự nó có lợi hay không.
- Cấu hình cộng dồn tốt nhất là **D3**: −2,12% Q̂ v0 và −1,94% LPIPS so với D0. LPIPS vẫn chưa tới −2%.
- Việc tiếp theo là **D6 = D3 + g_s v2, bỏ Sq**, để đo riêng g_s v2.

## 1. Kết quả bước D

BD-rate trên cả 8 level. Âm nghĩa là ít bit hơn ở cùng chất lượng.

| Run | Thay đổi | so với run trước (Q̂ / LPIPS) | Cổng | so với D0 (Q̂ / LPIPS) | Giải mã A30 |
|---|---|---:|---|---:|---:|
| D0 | v1 (mốc) | | | | 0,30 s/ảnh |
| D1 | + E1/E2 | −2,07% / −1,58% | trượt | −2,07% / −1,58% | 0,33 |
| D2 | + loss CCA 0,3 | −0,06% / +4,53% | trượt | −2,16% / +2,93% | 0,33 |
| D3 | + 10 slice × 32 | +0,00% / −4,84% | trượt | **−2,12% / −1,94%** | 0,43 |
| D4 | + tách đường Sq | +12,68% / +18,23% | trượt | +10,11% / +15,35% | 0,40 |
| D5 | + g_s v2 | −9,86% / −6,34% | **qua** | −0,54% / +8,20% | 0,42 |

**Cách đọc:**
- **E1/E2 (D1)** có lợi nhưng nhỏ, và LPIPS chưa tới cổng.
- **CCA 0,3 (D2)** làm LPIPS tệ đi 4,5%.
- **10 slice (D3)** gỡ lại phần LPIPS mà CCA làm mất (−4,8%), nhưng không lợi gì về Q̂, và tăng thời gian giải mã từ 0,33 lên 0,43 s/ảnh. Có thể 10 slice chỉ đang bù cho CCA. Chưa tách riêng được hai yếu tố này.
- **Sq (D4)** hỏng rõ ràng, cả ở level thấp (0–3: +11,4% / +16,2%). Thang rate cũng bị kéo giãn: q7 lên 0,51 bpp, trong khi D3 là 0,35 bpp. Vì vậy D4 so với các run khác trên một dải bpp lệch, và BD-rate của D4 kém tin cậy hơn các run còn lại.
- **g_s v2 (D5)** chỉ được đo trên nền D4 hỏng. −9,9% so với D4 có thể chỉ là gỡ lại một phần thiệt hại do Sq gây ra.

## 2. Quyết định đề xuất

| Thành phần | Đề xuất | Lý do |
|---|---|---|
| E1/E2 | giữ tạm | lợi nhỏ ở cả hai metric, chi phí giải mã thấp (+0,03 s) |
| CCA 0,3 | nghi ngờ, cần ablation (D7) | LPIPS +4,5% |
| 10 slice × 32 | giữ tạm, cần ablation (D7) | tốt nhất khi cộng dồn, nhưng tốn +30% thời gian giải mã |
| Sq tách đường | **bỏ khỏi đường chính**, để sau cờ tắt | +12,7% / +18,2%, thang rate bị kéo giãn |
| g_s v2 | chưa kết luận, chờ D6 | chỉ mới được đo trên nền D4 |

## 3. Run tiếp theo (GPU 2 / A30 đang rảnh)

Cả hai run giữ nguyên quy trình của bước D:
- rd 100k bước, cùng seed, cùng dữ liệu, cùng thang λ;
- 8 level trên CLIC30;
- chấm Q̂ v0 và LPIPS;
- áp cùng cổng ≤ −2%.

**Cách tạo cấu hình:** lấy `configs_a30/` làm gốc rồi sửa trong `make_cfgs.py`; không sửa tay file JSON.

| Ưu tiên | Run | Cấu hình | So với | Trả lời câu hỏi |
|---|---|---|---|---|
| 1 | **D6** | D3 + g_s v2, không Sq | D3 | g_s v2 có lợi thật không, khi không có Sq? |
| 2 | **D7** | D1 + 10 slice × 32, không CCA | D1, D3 | CCA có hại không, và 10 slice có lợi khi không có CCA không? |

**Chọn cấu hình tiếp theo:**
- Nếu D6 qua cổng so với D3, D6 thành mốc mới.
- Nếu D7 tốt hơn D3, bỏ CCA khỏi mốc.

A30 24 GB chạy được 2 run cùng lúc (5–10 GB/run), nên D6 và D7 có thể chạy song song, mất khoảng một lượt bước D.

## 4. Việc khác

1. **Tìm lỗi Sq trước khi bỏ hẳn.** Thang rate bị kéo giãn ở D4 (q7: 0,51 so với 0,35 bpp) gợi ý rằng σ số nguyên trên q đang tính sai rate, chứ không chỉ là ý tưởng kém. Cần kiểm tra:
   - bit thật so với bit ước lượng lúc train ở từng level;
   - phân bố σ của D3 so với D4.
2. **Đưa code_snapshot vào `src/`.**
   - `experiments/gc_stepd/code_snapshot/` chứa 10 file khác với `main`, chủ yếu là `gc/model.py`, `gc/codec.py`, `gc/layers.py` và `entropy/intnet.py`. 38 file còn lại của snapshot giống hệt `main`.
   - Chưa đưa vào `src/` vội. Chờ D6/D7 xong rồi merge một lần, với Sq và các thành phần bị bỏ nằm sau cờ tắt mặc định.
   - Kiểm tra khớp bit CPU/GPU sau khi merge.
3. **Ngân sách thời gian giải mã.**
   - Trên A30, D3 mất 0,43 s/ảnh, tức khoảng 13 s cho 30 ảnh.
   - Giới hạn là ≤ 20 s/30 ảnh trên L4, và L4 chậm hơn A30, nên phải đo D6 trên L4 trước khi chốt 10 slice.
4. **Tài liệu thiếu.** `HUONG_DAN.md` dẫn tới `plans/GC_v2_architecture_2026-11-23.md` (mục 5 và 8), nhưng file này chưa có trong repo. Cần lấy về từ máy đã viết nó.
5. **Dữ liệu chưa đưa lên.** Checkpoint (~1 GB/run), ảnh giải mã và bitstream (~5,5 GB) vẫn nằm ở `~/stepd/work` trên server. Repo chỉ giữ số liệu và log. Giữ lại `~/stepd/work` cho tới khi D6/D7 xong, để còn so sánh hoặc chấm lại ảnh của D0–D5 nếu cần.

# Hình cho paper

Hình theo style của figures4papers (skill `scientific-figure-making`). Style chung (bảng màu, font, export PNG và PDF 300 dpi) nằm trong `pubstyle.py`. Màu có ý nghĩa cố định trên mọi hình:
- **xanh đậm:** OT coupling / phương pháp của mình
- **xanh nhạt:** minibatch OT
- **đỏ:** coupling tự nhiên / baseline
- **xám:** coupling độc lập / tham chiếu

Vẽ lại toàn bộ bằng một lệnh; hình nào chưa có dữ liệu sẽ tự bỏ qua:

```bash
uv run --no-project --with numpy,scipy,matplotlib,torch python make_all.py
```

| Hình (`out/`) | Nội dung | Dữ liệu | Trạng thái |
|---|---|---|---|
| `fig_toy_dp` | Ứng viên cho Hình 3: mặt phẳng D–P, gồm (a) toy tách được và (b) toy không tách được. So decode 1 bước với ODE cho từng coupling | `experiments/toy_gaussian/results/`, `experiments/toy_gaussian/results_C/` | ✅ |
| `fig_dither_rate` | Mức phạt rate của dither theo số bit trên mỗi phần tử latent, kèm dải của SD3.5 VAE và của DC-AE ở 0.01–0.05 bpp. Hình này minh họa lý do chọn SANA | tính lại từ công thức đóng (cache ở `data/`) | ✅ |
| `fig_learnability` | (a) Lỗi của map học được ở d = 64 dồn vào các chiều bị nén. (b) Thí nghiệm A: đánh đổi của nhiễu khởi đầu | `results/d64_*`, `results_A/` | (a) ✅, (b) chờ A |
| `fig_toyB` | Kích thước pool của minibatch OT: chất lượng coupling và decode 1 bước | `experiments/toy_gaussian/results_B/` | chờ B |
| `fig_gonogo1` | Go/no-go 1: denoiser dùng mô hình nhiễu chính xác so với giả định Gaussian | `experiments/quant_noise/runs/gonogo1/` | chờ Kaggle |
| `fig_quantstat` | Thống kê sai số lượng tử theo bpp (kurtosis, corr, tương quan kênh, phụ thuộc nội dung) | `experiments/quant_noise/runs/kodak/` | chờ Kaggle |

Khi có kết quả từ Kaggle, copy các thư mục `results_A`, `results_B`, `runs/gonogo1` và `runs/kodak` về đúng đường dẫn trên rồi chạy `make_all.py`.

## Sơ đồ drawio (`drawio/`)

Mở bằng draw.io (diagrams.net). Các file được sinh bằng script, nên sửa script rồi sinh lại, đừng sửa tay:

| file | nội dung | sinh bằng |
|---|---|---|
| `1_kien_truc.drawio` | kiến trúc tổng thể | `python drawio/make_drawio.py drawio` |
| `2_pipeline_ma_hoa_giai_ma.drawio` | pipeline mã hóa và giải mã | như trên |
| `3_huan_luyen.drawio` | các giai đoạn huấn luyện | như trên |
| `4_lo_trinh.drawio` | lộ trình | như trên |
| `5_codec_c1_chi_tiet.drawio` | codec C1 (DC-AE + S1) theo từng lớp | `python drawio/make_codec_c1.py drawio` |

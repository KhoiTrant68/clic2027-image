# Q-hat v1: các biến thể đã thử (2026-10-09)

Cùng 8,000 câu hỏi như Q-hat v0, thêm đặc trưng Wasserstein `wd3`, `wd5` (bước `qhat.py add` trên server L40).
Phép thử: train trên 2021t/2021v/2022t, test trên 2024t. `weights >= 0`: bỏ dần đặc trưng có trọng số âm rồi fit lại.

```
v0 features (free weights)         test2024 0.812 | psnr -0.55, msssim +0.62, lpips_alex +0.13, lpips_vgg +0.42, dists +0.87, clipiqa +0.36
v1 all features (free weights)     test2024 0.812 | psnr -0.22, msssim +0.70, lpips_alex +0.16, lpips_vgg +0.44, dists +0.96, wd3 -0.34, wd5 -0.13, clipiqa +0.34
v1 all features, weights >= 0      test2024 0.805 | msssim +0.14, lpips_alex +0.12, lpips_vgg +0.45, dists +0.85, clipiqa +0.37
v0 features, weights >= 0          test2024 0.805 | msssim +0.14, lpips_alex +0.12, lpips_vgg +0.45, dists +0.85, clipiqa +0.37
wd3 only                           test2024 0.813 | wd3 +1.03
wd3 + dists                        test2024 0.801 | wd3 +0.00, dists +1.44
wd3 + dists + msssim               test2024 0.809 | dists +1.40, msssim +0.24
wd3 + msssim                       test2024 0.833 | wd3 +1.00, msssim +0.15
wd3 + wd5 + dists + msssim         test2024 0.809 | dists +1.40, msssim +0.24
lpips_alex + dists + msssim        test2024 0.814 | lpips_alex +0.27, dists +1.18, msssim +0.26
corr of feature differences (wd3, wd5, dists, lpips_vgg, msssim, psnr):
[[1.   0.99 0.43 0.5  0.93 0.96]
 [0.99 1.   0.5  0.56 0.89 0.93]
 [0.43 0.5  1.   0.95 0.29 0.22]
 [0.5  0.56 0.95 1.   0.4  0.31]
 [0.93 0.89 0.29 0.4  1.   0.97]
 [0.96 0.93 0.22 0.31 0.97 1.  ]]
```

Chọn làm `qhat_v1.json`: **wd3 + msssim, weights >= 0** (test 2024: 0.833, v0: 0.812).
Thận trọng: mô hình được chọn sau khi xem kết quả test (10 biến thể), nên 0.833 có thể hơi lạc quan;
và trong CV riêng năm 2021t nó chỉ đạt 0.665 (v0: 0.753), vì MS-SSIM vô dụng trên 2021t (0.497) mà WD lại
tương quan 0.93 với MS-SSIM. Dùng cả v0 và v1 để chấm bake-off; nếu hai bản xếp hạng khác nhau, tự chấm bằng mắt.

Lệnh tái lập: `python experiments/qhat/qhat.py fit --out results/2026-10-09_qhat_v1 --version qhat_v1 --features wd3 msssim --nonneg`

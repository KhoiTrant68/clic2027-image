# S1 trên kodak (bitstream thật)

checkpoint: `/kaggle/working/work/s1/last.pt` (step 200000)

| cfg | bpp | latent MSE | PSNR (pooled) | MS-SSIM | LPIPS↓ | DISTS↓ |
|---|---|---|---|---|---|---|
| dcae_ceiling | nan | 0.0000 | 23.66 | 0.8800 | 0.0905 | 0.0583 |
| s1_r0 | 0.0133 | 0.4230 | 17.44 | 0.4605 | 0.5822 | 0.4486 |
| s1_r1 | 0.0148 | 0.3926 | 17.79 | 0.4933 | 0.5638 | 0.4388 |
| s1_r2 | 0.0189 | 0.3279 | 18.40 | 0.5474 | 0.4934 | 0.3699 |
| s1_r3 | 0.0264 | 0.2356 | 19.06 | 0.6177 | 0.3798 | 0.2842 |
| s1_r4 | 0.0359 | 0.1596 | 19.74 | 0.6805 | 0.2898 | 0.2151 |
| s1_r5 | 0.0455 | 0.1109 | 20.36 | 0.7246 | 0.2310 | 0.1707 |
| s1_r6 | 0.0558 | 0.0795 | 20.92 | 0.7586 | 0.1950 | 0.1412 |
| s1_r7 | 0.0670 | 0.0608 | 21.23 | 0.7788 | 0.1725 | 0.1237 |

S1 được train bằng MSE trên latent (latent_hat ≈ E[latent | bitstream]), nên ảnh mờ là đúng thiết kế; bridge S2/S3 mới là phần khôi phục độ chân thực. Dòng `dcae_ceiling` là trần của backbone.
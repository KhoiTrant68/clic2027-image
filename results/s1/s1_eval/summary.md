# S1 trên kodak (bitstream thật)

checkpoint: `s1_out/last.pt` (step 200000)

| cfg | bpp | latent MSE | PSNR (pooled) | MS-SSIM | LPIPS↓ | DISTS↓ |
|---|---|---|---|---|---|---|
| dcae_ceiling | nan | 0.0000 | 23.66 | 0.8800 | 0.0905 | 0.0583 |
| s1_r0 | 0.0155 | 0.3673 | 17.83 | 0.5046 | 0.5229 | 0.3829 |
| s1_r1 | 0.0204 | 0.2859 | 18.57 | 0.5782 | 0.4485 | 0.3292 |
| s1_r2 | 0.0342 | 0.1687 | 19.62 | 0.6707 | 0.3091 | 0.2326 |
| s1_r3 | 0.0531 | 0.0870 | 20.71 | 0.7488 | 0.2058 | 0.1538 |
| s1_r4 | 0.0710 | 0.0452 | 21.70 | 0.8011 | 0.1521 | 0.1058 |
| s1_r5 | 0.0885 | 0.0232 | 22.45 | 0.8363 | 0.1216 | 0.0806 |
| s1_r6 | 0.1057 | 0.0113 | 22.99 | 0.8573 | 0.1051 | 0.0678 |
| s1_r7 | 0.1215 | 0.0060 | 23.29 | 0.8676 | 0.0980 | 0.0621 |

S1 được train bằng MSE trên latent (latent_hat ≈ E[latent | bitstream]), nên ảnh mờ là đúng thiết kế; bridge S2/S3 mới là phần khôi phục độ chân thực. Dòng `dcae_ceiling` là trần của backbone.
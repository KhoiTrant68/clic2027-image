# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan03/snap_8000.pt (step 8000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.382 | 0.8729 | 0.1098 | 0.0875 | 0.2826 | -0.245 |

Q-hat v0 student - teacher: mean -0.401, worst -1.253, images worse by > 0.1: 29/30

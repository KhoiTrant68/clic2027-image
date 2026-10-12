# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan03/snap_4000.pt (step 4000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.676 | 0.8777 | 0.1297 | 0.1009 | 0.3298 | -0.642 |

Q-hat v0 student - teacher: mean -0.799, worst -1.410, images worse by > 0.1: 30/30

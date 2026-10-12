# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_22000.pt (step 22000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.702 | 0.8810 | 0.0979 | 0.0786 | 0.2460 | -0.058 |

Q-hat v0 student - teacher: mean -0.215, worst -0.931, images worse by > 0.1: 26/30
